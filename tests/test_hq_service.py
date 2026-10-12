import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hyper_hyperqueue.backends.hyperqueue import task_resources
from hyper_hyperqueue.models import Config, Profile, Resources, TaskSpec
from hyper_hyperqueue.service.leadership import (
    LeadershipError,
    LeadershipGuard,
    assert_epoch,
    verify_readiness,
)
from hyper_hyperqueue.service.recovery import fence_slurm_generation
from hyper_hyperqueue.worker import detect_assigned, worker_argv


def task(tmp_path, **kwargs):
    return TaskSpec(
        task_id="t1",
        campaign_id="c1",
        argv=["/bin/true"],
        cwd=str(tmp_path),
        import_id="i1",
        command_digest="digest",
        backend="pooled",
        **kwargs,
    )


def test_cpu_gpu_requirements_are_separate_and_profile_alternatives(tmp_path):
    cpu = task_resources(task(tmp_path, allowed_profiles=["a", "b"]))
    assert len(cpu) == 2 and all(r["resources"]["hhq/cpu"] == 1 for r in cpu)
    gpu = task_resources(
        task(tmp_path, resources=Resources(gpus=1, min_vram_mib=4000, gpu_models=["A", "B"]))
    )
    assert len(gpu) == 2 and all(r["resources"]["hhq/gpu"] == 1 for r in gpu)
    assert all("hhq/cpu" not in r["resources"] for r in gpu)
    with pytest.raises(ValueError, match="one node"):
        task_resources(task(tmp_path, resources=Resources(nodes=2)))


def test_worker_uses_assignment_not_all_host_hardware():
    p = Profile(partition="test", cpus=16, memory_mib=16000)
    a = detect_assigned(
        p, environment={"SLURM_CPUS_PER_TASK": "2", "SLURM_MEM_PER_NODE": "2048"}, affinity=range(64)
    )
    assert a["cpus"] == 2 and a["memory_mib"] == 2048 and a["gpu"] is None
    cfg = Config(profiles={"cpu": p})
    argv = worker_argv(cfg, "cpu", a, remaining_seconds=300)
    assert "--detect-resources=none" in argv
    assert "hhq/memory=sum(1792)" in argv and "hhq/cpu=sum(2)" in argv


def test_gpu_detection_queries_only_assigned_device():
    p = Profile(partition="test", gres="gpu:large:1", gpu_models=["A100"])
    calls = []

    def query(identifier):
        calls.append(identifier)
        return {"uuid": "GPU-assigned", "model": "A100", "memory_mib": 40000}

    a = detect_assigned(
        p, environment={"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "3"}, affinity=[0], gpu_query=query
    )
    assert calls == [0] and a["gpu"]["visible_identifier"] == "0"
    assert a["gpu"]["slurm_job_gpus"] == "3"
    with pytest.raises(ValueError, match="explicitly assigned"):
        detect_assigned(p, environment={}, affinity=[0], gpu_query=query)


def test_lock_is_not_heartbeat_lease_and_epoch_fences(tmp_path, monkeypatch):
    guard = LeadershipGuard(tmp_path).acquire()
    guard.heartbeat("ACTIVE")
    monkeypatch.setenv("HHQ_LEADER_FD", str(guard.fd))
    assert assert_epoch(tmp_path, guard.epoch)["owner_id"] == guard.owner_id
    with pytest.raises(LeadershipError):
        LeadershipGuard(tmp_path).acquire(takeover_proof={"quiescent": True})
    with pytest.raises(LeadershipError):
        assert_epoch(tmp_path, guard.epoch + 1)
    epoch = guard.epoch
    guard.close(children_quiescent=True)
    second = LeadershipGuard(tmp_path).acquire()
    assert second.epoch == epoch + 1
    second.close(children_quiescent=True)


def test_unclean_owner_requires_positive_proof(tmp_path):
    old = LeadershipGuard(tmp_path).acquire()
    record = dict(old.record)
    old.close()
    with pytest.raises(LeadershipError, match="fencing"):
        LeadershipGuard(tmp_path).acquire()
    guard = LeadershipGuard(tmp_path).acquire(
        takeover_proof={
            "previous_owner": record["owner_id"],
            "previous_epoch": record["epoch"],
            "quiescent": True,
        }
    )
    guard.close(children_quiescent=True)


def test_lock_inode_replacement_fails_closed(tmp_path):
    guard = LeadershipGuard(tmp_path).acquire()
    (tmp_path / "leadership.lock").rename(tmp_path / "original.lock")
    (tmp_path / "leadership.lock").touch()
    with pytest.raises(LeadershipError, match="inode"):
        guard.assert_owned()
    guard.close()


def test_standby_missing_or_partial_readiness_fails_closed(tmp_path):
    with pytest.raises(LeadershipError):
        verify_readiness(None, tmp_path)
    p = tmp_path / "partial.json"
    p.write_text(json.dumps({"schema": "hhq-failover-readiness/1", "cross_node_exclusion": True}))
    with pytest.raises(LeadershipError):
        verify_readiness(str(p), tmp_path)


def test_fencing_requires_owned_terminal_accounting_and_no_live_job():
    previous = {"slurm_job_id": "17", "owner_id": "old", "epoch": 2}
    allocations = [{"job_id": "17", "token": "token"}]
    cancels = []

    def missing(_):
        return {"job_id": "17", "owned": True, "live": False, "accounting_state": None}

    no = fence_slurm_generation(
        previous,
        allocations,
        query=missing,
        cancel=cancels.append,
        readiness={"verified": True},
        deadline_seconds=0,
    )
    assert not no.ready and not cancels
    observed = {"job_id": "17", "owned": True, "live": True, "accounting_state": None}

    def cancel(a):
        cancels.append(a)
        observed.update(live=False, accounting_state="CANCELLED")

    yes = fence_slurm_generation(
        previous,
        allocations,
        query=lambda _: observed.copy(),
        cancel=cancel,
        readiness={"verified": True},
        deadline_seconds=0,
    )
    assert yes.ready and yes.proof["previous_epoch"] == 2 and len(cancels) == 1


def test_hq_import_is_lazy_for_direct_mode():
    code = "import sys; import hyper_hyperqueue.backends.hyperqueue; assert 'hyperqueue' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_worker_lifetime_uses_allocated_start_not_pending_age():
    from types import SimpleNamespace

    from hyper_hyperqueue.worker import allocation_remaining

    row = SimpleNamespace(job_id="17", state="RUNNING", start_time=950, end_time=None, time_limit_seconds=100)
    assert allocation_remaining(environment={"SLURM_JOB_ID": "17"}, now=1000, query=lambda: [row]) == 50
    assert (
        allocation_remaining(environment={"SLURM_JOB_ID": "17", "SLURM_JOB_END_TIME": "1020"}, now=1000) == 20
    )
    with pytest.raises(ValueError, match="remaining lifetime"):
        allocation_remaining(environment={"SLURM_JOB_ID": "17"}, query=list)
    with pytest.raises(ValueError, match="expired"):
        allocation_remaining(environment={"SLURM_JOB_ID": "17", "SLURM_JOB_END_TIME": "900"}, now=1000)


def test_standby_replenishment_persists_intent_and_never_exceeds_two(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hyper_hyperqueue.service import supervisor
    from hyper_hyperqueue.service.leadership import atomic_record, read_record

    root = tmp_path / "state"
    service = root / "service"
    guard = LeadershipGuard(service).acquire()
    guard.heartbeat("ACTIVE", slurm_job_id="1")
    config = Config(state_dir=str(root))
    config.control.standby_replicas = 1
    config_path = supervisor._config_snapshot(config, service)
    deployment = {
        "hosting": "slurm",
        "created_at": time.time(),
        "config": str(config_path),
        "allocations": [
            {"token": "oldtoken1", "job_id": "1", "status": "accepted"},
            {"token": "oldtoken2", "job_id": "2", "status": "accepted"},
        ],
    }
    atomic_record(service / "deployment.json", deployment)
    monkeypatch.setattr(supervisor, "verify_readiness", lambda *a: {})
    monkeypatch.setattr(supervisor, "__name__", "__main__")

    class Client:
        def __init__(self):
            self.jobs = {
                "oldtoken1": [SimpleNamespace(job_id="1", terminal=False)],
                "oldtoken2": [
                    SimpleNamespace(
                        job_id="2", terminal=True, to_dict=lambda: {"job_id": "2", "state": "COMPLETED"}
                    )
                ],
            }
            self.submissions = 0

        def find_submission(self, token, since):
            return self.jobs.get(token, [])

        def submit(self, script, token, job_name, extra_args):
            assert read_record(service / "deployment.json")["allocations"][-1]["status"] == "submitting"
            assert "--exclude" in extra_args
            assert "-m hyper_hyperqueue.service.supervisor run" in Path(script).read_text()
            self.submissions += 1
            self.jobs[token] = [SimpleNamespace(job_id="3", terminal=False)]
            value = {"status": "accepted", "job_id": "3", "token": token, "job_name": job_name + "-" + token}
            return SimpleNamespace(**value, to_dict=lambda: value)

    client = Client()
    try:
        assert supervisor.maintain_standby(config, guard, client=client)["status"] == "replacement_accepted"
        assert supervisor.maintain_standby(config, guard, client=client)["status"] == "capacity_present"
        assert client.submissions == 1
        current = read_record(service / "deployment.json")
        current["allocations"][-1]["status"] = "unknown"
        atomic_record(service / "deployment.json", current)
        client.jobs.pop(current["allocations"][-1]["token"])
        assert (
            supervisor.maintain_standby(config, guard, client=client)["status"] == "blocked_unknown_control"
        )
        assert client.submissions == 1
    finally:
        guard.close(children_quiescent=True)


def test_repeated_terminal_control_failures_stop_replenishment(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from hyper_hyperqueue.service import supervisor
    from hyper_hyperqueue.service.leadership import atomic_record

    service = tmp_path / "service"
    guard = LeadershipGuard(service).acquire()
    guard.heartbeat("ACTIVE", slurm_job_id="1")
    config = Config(state_dir=str(tmp_path))
    config.control.standby_replicas = 1
    records = [{"token": "active01", "job_id": "1", "status": "accepted"}]
    records.extend(
        {
            "token": f"failed0{i}",
            "job_id": str(i),
            "status": "accepted",
            "terminal_evidence": [{"job_id": str(i), "state": "FAILED"}],
        }
        for i in range(2, 5)
    )
    atomic_record(
        service / "deployment.json", {"hosting": "slurm", "created_at": time.time(), "allocations": records}
    )
    monkeypatch.setattr(supervisor, "verify_readiness", lambda *a: {})

    class Client:
        def find_submission(self, *a, **k):
            return [SimpleNamespace(job_id="1", terminal=False)]

        def submit(self, *a, **k):
            raise AssertionError("Must not create another failed control")

    try:
        assert (
            supervisor.maintain_standby(config, guard, client=Client())["status"]
            == "blocked_control_startup_failures"
        )
    finally:
        guard.close(children_quiescent=True)


def test_stop_pending_service_prevents_later_control_start(tmp_path, monkeypatch):
    from hyper_hyperqueue.service import supervisor
    from hyper_hyperqueue.service.leadership import atomic_record, read_record

    config = Config(state_dir=str(tmp_path))
    service = tmp_path / "service"
    atomic_record(
        service / "deployment.json",
        {
            "deployment_id": "pending",
            "hosting": "slurm",
            "desired_state": "RUNNING",
            "allocations": [{"job_id": "17", "status": "accepted"}],
        },
    )
    result = supervisor.stop_service(config)
    assert result["status"] == "requested" and result["epoch"] is None
    assert read_record(service / "deployment.json")["desired_state"] == "STOPPED"

    def forbidden(*a, **k):
        raise AssertionError("Stopped pending control must never start HQ")

    monkeypatch.setattr(supervisor, "locate_binary", forbidden)
    supervisor.run_supervisor(config, tmp_path / "unused-config.json", role="standby")
    assert not (service / "leader.json").exists()


@pytest.mark.parametrize("previous_state", ["STOPPED", "ACTIVE"])
def test_explicit_single_control_restart_requires_terminal_generation_without_readiness(
    tmp_path, monkeypatch, previous_state
):
    from hyper_hyperqueue.backends.slurm import SlurmJob
    from hyper_hyperqueue.service import supervisor
    from hyper_hyperqueue.service.leadership import atomic_record, read_record
    from hyper_hyperqueue.state import StateStore

    config = Config(state_dir=str(tmp_path))
    service = tmp_path / "service"
    store = StateStore(tmp_path)
    previous = {"owner_id": "old", "epoch": 5, "slurm_job_id": "17", "state": previous_state}
    atomic_record(
        service / "deployment.json",
        {
            "hosting": "slurm",
            "created_at": time.time(),
            "allocations": [{"job_id": "17", "token": "control_token"}],
        },
    )
    atomic_record(
        service / "workers/18.json", {"epoch": 5, "slurm_job_id": "18", "allocation_token": "worker_token"}
    )
    store.put_intent(
        {
            "backend": "worker",
            "epoch": 5,
            "job_id": "18",
            "token": "worker_token",
            "state": "terminal",
            "created_at": time.time(),
        }
    )

    def record(jid, token, state="COMPLETED"):
        return SlurmJob(
            job_id=jid, name="hhq-" + token, user="test", state=state, partition="test", comment=""
        )

    class Client:
        user = "test"

        def __init__(self):
            self.live = []
            self.records = [record("17", "control_token"), record("18", "worker_token")]

        def query_jobs(self):
            return self.live

        def accounting(self, *a, **k):
            return self.records

        def find_submission(self, token, **k):
            return [j for j in self.records if j.name.endswith(token)]

        def cancel(self, *a, **k):
            raise AssertionError("Explicit read-only restart cannot cancel anything")

    client = Client()

    def forbidden(*a, **k):
        raise AssertionError("No automatic-failover readiness needed for already terminal resources")

    monkeypatch.setattr(supervisor, "verify_readiness", forbidden)
    proof = supervisor._terminal_generation_proof(config, previous, client=client)
    assert proof["quiescent"] and proof["previous_epoch"] == 5 and len(proof["allocations"]) == 2
    assert read_record(service / "manual-restart-5.json")["schema"] == "hhq-manual-restart-proof/1"
    client.live = [record("18", "worker_token", "RUNNING")]
    with pytest.raises(LeadershipError, match="live or terminal ownership"):
        supervisor._terminal_generation_proof(config, previous, client=client)
    client.live = []
    client.records = [record("17", "control_token"), record("18", "wrong_token")]
    with pytest.raises(LeadershipError, match="unresolved ownership"):
        supervisor._terminal_generation_proof(config, previous, client=client)
