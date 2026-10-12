"""Real pinned HQ subprocess tests. No Slurm mutations or workload containers."""

import contextlib
import os
import signal
import subprocess
import sys
import time

import pytest

from hyper_hyperqueue.backends.hyperqueue import HQBackend, HQError, HQSubmissionUnknown, locate_binary
from hyper_hyperqueue.models import Config, Profile, Resources, TaskSpec
from hyper_hyperqueue.state import StateStore
from hyper_hyperqueue.worker import detect_assigned, worker_argv

pytestmark = pytest.mark.integration


def eventually(fn, seconds=20):
    end = time.monotonic() + seconds
    last = None
    while time.monotonic() < end:
        try:
            value = fn()
            if value:
                return value
        except (HQError, OSError, subprocess.TimeoutExpired) as exc:
            last = exc
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for real HQ state: {last}")


@pytest.fixture
def cluster(tmp_path):
    try:
        binary = locate_binary()
        __import__("hyperqueue")
    except (RuntimeError, ImportError):
        pytest.skip("Pinned optional HQ runtime not installed")
    root = tmp_path / "state"
    directory = root / "service"
    directory.mkdir(parents=True)
    processes = []
    handles = []

    def start_server():
        log = (directory / f"server-{len(processes)}.log").open("w")
        handles.append(log)
        process = subprocess.Popen(
            [
                binary,
                "--server-dir",
                str(directory / "hq"),
                "server",
                "start",
                "--host",
                "127.0.0.1",
                "--journal",
                str(directory / "journal"),
                "--journal-flush-period",
                "1s",
            ],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        processes.append(process)
        backend = HQBackend(directory / "hq", binary)
        eventually(lambda: backend.json("server", "info"))
        return process, backend

    config = Config(
        state_dir=str(root),
        profiles={"cpu": Profile(partition="test", cpus=2, memory_mib=1024, allocation_seconds=120)},
    )
    config.policy.cleanup_seconds = 1

    def start_worker():
        args = worker_argv(config, "cpu", detect_assigned(config.profiles["cpu"]))
        log = (directory / f"worker-{len(processes)}.log").open("w")
        handles.append(log)
        process = subprocess.Popen(args, stdout=log, stderr=log, start_new_session=True)
        processes.append(process)
        return process

    try:
        yield root, start_server, start_worker
    finally:
        for p in reversed(processes):
            if p.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        for h in handles:
            h.close()


def make_task(root, i, command):
    return TaskSpec(
        task_id=f"t{i}",
        campaign_id="c1",
        command=command,
        cwd=str(root),
        import_id="i1",
        command_digest=f"hash{i}",
        backend="pooled",
        resources=Resources(memory_mib=128, runtime_seconds=1),
    )


def test_real_two_workers_pack_tasks_record_failure_and_append(cluster):
    root, start_server, start_worker = cluster
    _, backend = start_server()
    start_worker()
    start_worker()
    eventually(lambda: len(backend.workers()) == 2)
    store = StateStore(root)
    tasks = [make_task(root, i, "sleep .7; printf '%s' ok") for i in range(8)]
    tasks.append(make_task(root, 8, "exit 7"))
    store.create_campaign({"campaign_id": "c1", "task_ids": [t.task_id for t in tasks]}, tasks)

    def runner(t):
        return [
            sys.executable,
            "-m",
            "hyper_hyperqueue.runner",
            "--state-dir",
            str(root),
            "--campaign",
            "c1",
            "--task",
            t.task_id,
        ]

    submitted = backend.submit(tasks, "integration_initial", runner, epoch=1)

    def done():
        rows = backend.tasks(submitted["job_id"])[str(submitted["job_id"])]
        return rows if all(r["state"] in {"finished", "failed"} for r in rows) else None

    rows = eventually(done)
    assert sum(r["state"] == "finished" for r in rows) == 8
    assert sum(r["state"] == "failed" for r in rows) == 1
    assert {store.task_state("c1", t.task_id)["status"] for t in tasks[:8]} == {"succeeded"}
    assert store.task_state("c1", "t8")["status"] == "failed"
    workers_used = set()
    for row in rows:
        workers_used.add(row["worker"])
    # Worker IDs are part of detailed task state, not a count inferred from processes.
    assert len(workers_used) == 2, rows
    # More simultaneous commands than workers proves CPU packing, not just
    # eventual distribution across two serial executors.
    from datetime import datetime

    events = []
    for task in tasks[:8]:
        attempt = store.attempts("c1", task.task_id)[0]
        events.extend(
            [
                (datetime.fromisoformat(attempt["started_at"]), 1),
                (datetime.fromisoformat(attempt["result"]["finished_at"]), -1),
            ]
        )
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    assert peak >= 3, events
    with pytest.raises(HQSubmissionUnknown):
        backend.submit(tasks, "integration_initial", runner)
    extra = make_task(root, 9, "printf appended")
    store.add_task(extra)
    added = backend.submit([extra], "integration_append", runner, epoch=1)
    eventually(lambda: store.task_state("c1", extra.task_id)["status"] == "succeeded")
    assert added["job_id"] != submitted["job_id"]


def test_real_journal_restart_preserves_waiting_job(cluster):
    root, start_server, start_worker = cluster
    server, backend = start_server()
    t = make_task(root, 1, "true")
    job = backend.submit([t], "restart_test", lambda _: ["/bin/true"])
    backend.json("journal", "flush")
    server.terminate()
    server.wait(timeout=5)
    _, restored = start_server()
    assert restored.find_intent("restart_test")["id"] == job["job_id"]
    assert restored.workers() == []
    start_worker()
    eventually(lambda: restored.tasks(job["job_id"])[str(job["job_id"])][0]["state"] == "finished")


def test_real_worker_loss_fails_only_affected_task_without_hidden_retry(cluster):
    root, start_server, start_worker = cluster
    _, backend = start_server()
    worker = start_worker()
    eventually(lambda: len(backend.workers()) == 1)
    t = make_task(root, 1, "sleep 20")
    job = backend.submit([t], "worker_loss_test", lambda _: ["/bin/sleep", "20"])
    eventually(lambda: backend.tasks(job["job_id"])[str(job["job_id"])][0]["state"] == "running")
    os.killpg(worker.pid, signal.SIGKILL)
    worker.wait(timeout=5)
    eventually(lambda: backend.tasks(job["job_id"])[str(job["job_id"])][0]["state"] == "failed")
    start_worker()
    replay = backend.submit([t], "worker_loss_explicit_retry", lambda _: ["/bin/true"])
    eventually(lambda: backend.tasks(replay["job_id"])[str(replay["job_id"])][0]["state"] == "finished")


def test_real_supervisor_restarts_controller_without_restarting_hq(tmp_path):
    from hyper_hyperqueue.config import save_config
    from hyper_hyperqueue.service.leadership import read_record
    from hyper_hyperqueue.service.supervisor import stop_service

    root = tmp_path / "state"
    config = Config(state_dir=str(root))
    config.control.hosting = "local"
    config.control.advertise_host = "127.0.0.1"
    config_path = tmp_path / "config.json"
    save_config(config, config_path)
    code = (
        "from pathlib import Path; import sys; from hyper_hyperqueue.config import load_config; "
        "from hyper_hyperqueue.service.supervisor import run_supervisor; "
        "p=Path(sys.argv[1]); run_supervisor(load_config(p),p, "
        'controller_argv=[sys.executable,"-c","import time; time.sleep(90)"],max_seconds=30)'
    )
    with (tmp_path / "supervisor.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", code, str(config_path)], stdout=log, stderr=log, start_new_session=True
        )
        try:

            def active():
                row = read_record(root / "service/leader.json", {})
                return row if row.get("state") == "ACTIVE" and len(row.get("children", [])) == 2 else None

            before = eventually(active)
            old_hq, old_controller = before["children"]
            os.kill(old_controller["pid"], signal.SIGKILL)
            after = eventually(
                lambda: r if (r := active()) and r["children"][1]["pid"] != old_controller["pid"] else None
            )
            assert after["epoch"] == before["epoch"] and after["children"][0] == old_hq
            assert HQBackend(root / "service/hq").json("server", "info")["pid"] == old_hq["pid"]
            stop_service(config, drain=True)
            assert process.wait(timeout=10) == 0
            assert read_record(root / "service/leader.json")["state"] == "STOPPED"
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
