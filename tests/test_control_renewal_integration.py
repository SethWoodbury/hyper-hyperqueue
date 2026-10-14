"""Real local HQ journal/runner continuity; mocked Slurm accounting only."""

import contextlib
import os
import signal
import subprocess
import time

import pytest

from hyper_hyperqueue.backends.hyperqueue import HQBackend, HQError, locate_binary
from hyper_hyperqueue.backends.slurm import SlurmJob
from hyper_hyperqueue.controller import runner_argv
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config, Resources
from hyper_hyperqueue.service import renewal
from hyper_hyperqueue.service.leadership import LeadershipGuard, atomic_record
from hyper_hyperqueue.state import StateStore
from hyper_hyperqueue.worker import detect_assigned, worker_argv

pytestmark = pytest.mark.integration


def eventually(fn):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            result = fn()
            if result:
                return result
        except (HQError, OSError, subprocess.TimeoutExpired):
            pass
        time.sleep(0.1)
    raise AssertionError("Local HQ operation did not settle")


def test_planned_rollover_restores_waiting_task_without_repeating_success(tmp_path, monkeypatch):
    try:
        binary = locate_binary()
        __import__("hyperqueue")
    except (RuntimeError, ImportError):
        pytest.skip("Pinned optional HQ runtime unavailable")
    monkeypatch.setenv("SLURM_JOB_ID", "17")
    store = StateStore(tmp_path / "state")
    source = tmp_path / "commands"
    source.write_text("echo first >> count\necho second >> count\n")
    campaign = ingest_file(store, source, mode="pooled", cwd=tmp_path,
                           resources=Resources(memory_mib=128, runtime_seconds=1))
    tasks = [store.task(campaign["campaign_id"], tid) for tid in campaign["task_ids"]]
    config = Config(state_dir=str(store.root), control={"auto_renew": True, "hq_binary": binary},
                    profiles={"cpu": {"partition": "cpu", "cpus": 1, "memory_mib": 1024,
                                      "allocation_seconds": 120}})
    service = store.root / "service"
    old = LeadershipGuard(service).acquire()
    old.heartbeat("ACTIVE", deployment_token="parent_token")
    source_binding = {"epoch": old.epoch, "owner_id": old.owner_id, "job_id": "17", "token": "parent_token"}
    successor = {"status": "accepted", "allocation": {
        "token": "successor_token", "job_id": "19", "status": "accepted", "role": "renewal",
        "renewal_from": source_binding, "created_at": time.time(),
    }}
    atomic_record(service / "deployment.json", {
        "deployment_id": "deployment", "desired_state": "RUNNING", "hosting": "slurm",
        "created_at": time.time(), "allocations": [
            {"token": "parent_token", "job_id": "17", "created_at": time.time()}, successor["allocation"],
        ],
    })
    processes, handles, new = [], [], None

    def spawn(args, env=None, fd=None):
        log = (tmp_path / f"process-{len(processes)}.log").open("w")
        handles.append(log)
        process = subprocess.Popen(args, stdout=log, stderr=log, env=env,
                                   start_new_session=True, pass_fds=() if fd is None else (fd,))
        processes.append(process)
        return process

    def stop(process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)

    def server(guard):
        process = spawn([binary, "--server-dir", str(service / "hq"), "server", "start", "--host", "127.0.0.1",
                         "--journal", str(service / "hq.journal"), "--journal-flush-period=1s"], fd=guard.fd)
        backend = HQBackend(service / "hq", binary)
        eventually(lambda: backend.json("server", "info"))
        return process, backend

    def worker(epoch):
        argv = worker_argv(config, "cpu", detect_assigned(config.profiles["cpu"], environment={}))
        return spawn(argv, {**os.environ, "HHQ_WORKER_EPOCH": str(epoch), "SLURM_JOB_ID": "18"})

    try:
        first_server, backend = server(old)
        for task in tasks:
            store.update_task_state(task.campaign_id, task.task_id, {"state": "submitted", "server_epoch": old.epoch})
        backend.submit([tasks[0]], "first_intent", lambda t: runner_argv(store, t), epoch=old.epoch)
        first_worker = worker(old.epoch)
        eventually(lambda: store.task_state(tasks[0].campaign_id, tasks[0].task_id)["state"] == "succeeded")
        stop(first_worker)
        waiting = backend.submit([tasks[1]], "waiting_intent", lambda t: runner_argv(store, t), epoch=old.epoch)
        assert not store.attempts(tasks[1].campaign_id, tasks[1].task_id)
        backend.json("journal", "flush")
        stop(first_server)
        assert renewal.record_handoff(config, old, successor,
                                      {"settled": True, "workers": [], "journal_flushed_at": time.time()})
        old.close(children_quiescent=True)
        previous = old.record

        class Slurm:
            user = "owner"

            def rows(self):
                return [SlurmJob(job_id="17", name="hhq-parent_token", user=self.user, state="COMPLETED",
                                 partition="cpu", comment="hhq:parent_token"),
                        SlurmJob(job_id="19", name="hhq-successor_token", user=self.user, state="RUNNING",
                                 partition="cpu", comment="hhq:successor_token")]

            def find_submission(self, token, **kwargs):
                return [r for r in self.rows() if r.comment == "hhq:" + token]

            def query_jobs(self):
                return [r for r in self.rows() if not r.terminal]

            def accounting(self, ids, **kwargs):
                return [r for r in self.rows() if r.job_id in ids and r.terminal]

        proof = renewal.takeover_proof(config, previous, "successor_token", "19", client=Slurm())
        monkeypatch.setenv("SLURM_JOB_ID", "19")
        new = LeadershipGuard(service).acquire(takeover_proof=proof)
        new.heartbeat("ACTIVE")
        assert new.epoch == old.epoch + 1
        _, restored = server(new)
        assert restored.find_intent("waiting_intent")["id"] == waiting["job_id"]
        worker(new.epoch)
        eventually(lambda: store.task_state(tasks[1].campaign_id, tasks[1].task_id)["state"] == "succeeded")
        assert (tmp_path / "count").read_text().splitlines() == ["first", "second"]
        assert len(store.attempts(tasks[0].campaign_id, tasks[0].task_id)) == 1
        attempt = store.attempts(tasks[1].campaign_id, tasks[1].task_id)[0]
        environment = attempt["context"]["environment"]
        assert environment["HHQ_WORKER_EPOCH"] == str(new.epoch)
        assert environment["HHQ_SERVER_EPOCH"] == str(old.epoch)
    finally:
        for process in reversed(processes):
            with contextlib.suppress(ProcessLookupError):
                stop(process)
        for handle in handles:
            handle.close()
        old.close()
        if new:
            new.close(children_quiescent=True)
