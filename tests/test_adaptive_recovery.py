import json
import sys

import pytest

from hyper_hyperqueue.adaptive_recovery import effective_config, effective_task
from hyper_hyperqueue.allocation_budget import allocation_budget_status, budget_admission
from hyper_hyperqueue.controller import retry_ready_failures
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config, Profile, RetryPolicy
from hyper_hyperqueue.runner import authorize_retry, run_task
from hyper_hyperqueue.service.completion import work_status
from hyper_hyperqueue.state import StateStore, atomic_json, fingerprint


def imported(tmp_path, *, command=None, mode="direct", gpu=0, memory=1000, reasons=None):
    source = tmp_path / "commands"
    source.write_text(json.dumps({"command": command or f'{sys.executable} -c "raise MemoryError()"'}))
    store = StateStore(tmp_path / "state")
    campaign = ingest_file(store, source, mode=mode, cwd=tmp_path, format="jsonl",
        resources={"memory_mib": memory, "gpus": gpu}, profile_names=["work"],
        retry={"max_attempts": 8, "reasons": reasons or ["host_oom", "preempted", "allocation_timeout", "worker_lost"],
               "oom_memory_multipliers": [1.25, 1.5, 2.0]},
        output={"repeatable": True, "mode": "in_place"})
    return store, campaign, campaign["task_ids"][0]


def fake_attempt(store, campaign, tid, *, result=True, gpu=1, state="TIMEOUT", raw_cancel=False):
    task = store.task(campaign["campaign_id"], tid)
    directory = store.campaign_dir(task.campaign_id) / "attempts" / tid / "000001_test"
    attempt = {"attempt_id": "000001_test", "task_id": tid, "campaign_id": task.campaign_id,
               "task_fingerprint": fingerprint(task.model_dump(mode="json")), "attempt_dir": str(directory),
               "resources": task.resources.model_dump(),
               "context": {"environment": {"SLURM_JOB_ID": "123", "HHQ_WORKER_EPOCH": "7"}}}
    atomic_json(directory / "attempt.json", attempt)
    if result:
        atomic_json(directory / "result.json", {"status": "canceled", "reason": "user_cancellation",
                    "execution": {"forwarded_signals": [15]}, "exit_code": 143,
                    "finished_at": "2026-10-14T01:00:00+00:00"})
    store.put_intent({"token": "owner", "backend": "worker", "profile": "work", "job_id": "123",
                     "epoch": 7, "state": "terminal", "extra_args": ["--gres", f"gpu:{gpu}", "--time", "01:00:00"],
                     "terminal_evidence": [{"job_id": "123", "comment": "hhq:owner", "state": state, "end_time": 1791939601.0}]})
    if raw_cancel:
        store.update_task_state(task.campaign_id, tid, {"state": "canceled"})
    return attempt


def test_memory_ladder_original_request_and_immutable_identity(tmp_path):
    store, campaign, tid = imported(tmp_path)
    cid = campaign["campaign_id"]
    original = (store.campaign_dir(cid) / "tasks" / f"{tid}.json").read_bytes()
    for expected in [1000, 1250, 1500, 2000]:
        assert run_task(store.root, cid, tid) == 1
        latest = store.attempts(cid, tid)[-1]
        assert latest["resources"]["memory_mib"] == expected
        assert latest["result"]["reason"] == "host_oom"
        assert run_task(store.root, cid, tid) == 75  # No silent same-RAM replay.
        if expected < 2000:
            authorize_retry(store, cid, tid)
    with pytest.raises(ValueError, match="escalation exhausted"):
        authorize_retry(store, cid, tid)
    assert (store.campaign_dir(cid) / "tasks" / f"{tid}.json").read_bytes() == original
    assert len(list((store.campaign_dir(cid) / "resource_overrides" / tid).glob("*.json"))) == 3


@pytest.mark.parametrize("message,exit_code,reason", [
    ("", 137, "application_exit"), ("torch.OutOfMemoryError: CUDA out of memory", 1, "gpu_oom"),
    ("I may have a MemoryError", 1, "application_exit"),
    ("MemoryError\nRuntimeError: unrelated final failure", 1, "application_exit")])
def test_no_host_escalation_without_positive_host_diagnostic(tmp_path, message, exit_code, reason):
    command = f'{sys.executable} -c {__import__("shlex").quote(f"import sys; print({message!r},file=sys.stderr); sys.exit({exit_code})")}'
    store, campaign, tid = imported(tmp_path, command=command)
    assert run_task(store.root, campaign["campaign_id"], tid) == exit_code
    assert store.attempts(campaign["campaign_id"], tid)[-1]["result"]["reason"] == reason
    with pytest.raises(ValueError, match="Failure reason"):
        authorize_retry(store, campaign["campaign_id"], tid)


@pytest.mark.parametrize("slurm_state,expected,ram", [("TIMEOUT", "allocation_timeout", 1000),
    ("PREEMPTED", "preempted", 1000), ("OUT_OF_MEMORY", "host_oom", 1250), ("CANCELLED", None, 1000),
    ("COMPLETED", None, 1000)])
def test_positive_scheduler_proof_can_resolve_finalized_sigterm(tmp_path, slurm_state, expected, ram):
    store, campaign, tid = imported(tmp_path, mode="pooled", gpu=1)
    fake_attempt(store, campaign, tid, state=slurm_state)
    decisions = retry_ready_failures(store, Config(), [campaign])
    if expected:
        assert decisions[0]["decision"] == "retry_authorized"
        assert decisions[0]["reason"] == expected
    else:
        assert decisions == []
    assert effective_task(store, store.task(campaign["campaign_id"], tid)).resources.memory_mib == ram


@pytest.mark.parametrize("cancel", ["task", "campaign"])
def test_user_cancel_always_wins_over_terminal_timeout(tmp_path, cancel):
    store, campaign, tid = imported(tmp_path, mode="pooled", gpu=1)
    fake_attempt(store, campaign, tid, raw_cancel=cancel == "task")
    if cancel == "campaign":
        atomic_json(store.root / "requests/cancel-test.done.json", {"campaign_id": campaign["campaign_id"],
                    "action": "cancel_hq_campaign"})
    assert retry_ready_failures(store, Config(), [campaign]) == []


def test_pooled_cpu_oom_victim_retries_without_blame_or_ram_increase(tmp_path):
    store, campaign, tid = imported(tmp_path, mode="pooled", gpu=0)
    fake_attempt(store, campaign, tid, gpu=0, state="OUT_OF_MEMORY")
    decisions = retry_ready_failures(store, Config(), [campaign])
    assert decisions[0]["reason"] == "worker_lost"
    assert effective_task(store, store.task(campaign["campaign_id"], tid)).resources.memory_mib == 1000


def test_completion_waits_for_final_signal_accounting(tmp_path):
    store, campaign, tid = imported(tmp_path, mode="pooled", gpu=1)
    fake_attempt(store, campaign, tid)
    intent = store.intents()[0]
    intent.update(state="running", terminal_evidence=[])
    store.put_intent(intent)
    assert not work_status(store)["settled"]
    assert retry_ready_failures(store, Config(), [campaign]) == []


def test_natural_hq_lifetime_requires_exact_receipt_and_terminal_proof(tmp_path):
    store, campaign, tid = imported(tmp_path, mode="pooled", gpu=1)
    fake_attempt(store, campaign, tid, state="COMPLETED")
    path = store.root / "service/worker_exits/123.json"
    receipt = {"termination_reason": "time_limit", "allocation_token": "wrong", "epoch": 7,
               "slurm_job_id": "123", "received_signals": [], "returncode": 0,
               "configured_end_time": 123, "finished": 124}
    atomic_json(path, receipt)
    assert retry_ready_failures(store, Config(), [campaign]) == []
    receipt["allocation_token"] = "owner"
    atomic_json(path, receipt)
    assert retry_ready_failures(store, Config(), [campaign])[0]["reason"] == "worker_lost"


def test_override_binding_and_derived_profile_preserve_original(tmp_path):
    store, campaign, tid = imported(tmp_path)
    cid = campaign["campaign_id"]
    assert run_task(store.root, cid, tid) == 1
    authorize_retry(store, cid, tid)
    task = store.task(cid, tid)
    config = Config(profiles={"work": Profile(partition="cpu", memory_mib=1000)})
    derived = effective_config(store, config, [task])
    assert derived.profiles["work"].memory_mib == 1250
    assert config.profiles["work"].memory_mib == 1000
    binding = store.task_state(cid, tid)["resource_override"]
    binding["after_attempt_id"] = "other"
    store.update_task_state(cid, tid, {"resource_override": binding})
    with pytest.raises(ValueError, match="attempt binding"):
        effective_task(store, task)


def test_legacy_policy_dump_does_not_change_task_fingerprint():
    assert RetryPolicy().model_dump() == {"max_attempts": 1, "reasons": ["worker_lost", "preempted"],
        "memory_increment_mib": 2048, "max_memory_mib": None}
    for values in [[1], [1.5, 1.25], [1.25, 1.25], [float("nan")]]:
        with pytest.raises(ValueError):
            RetryPolicy(oom_memory_multipliers=values)


def test_allocation_budget_reserves_unknown_pending_and_running_then_uses_terminal_elapsed(tmp_path):
    store = StateStore(tmp_path / "state")
    config = Config(policy={"allocation_gpu_hours_budget": 2, "allocation_budget_allowance": .5})
    profile = Profile(partition="gpu", gres="gpu:1", allocation_seconds=3600)
    intent = {"token": "test", "backend": "worker", "job_id": "12", "state": "unknown",
              "extra_args": ["--gres", "gpu:1", "--time", "02:00:00"]}
    store.put_intent(intent)
    assert budget_admission(store, config, profile)["allowed"]
    assert not budget_admission(store, config, profile, elements=2)["allowed"]
    intent.update(state="terminal", terminal_evidence=[{"job_id": "12", "elapsed_seconds": 1800, "gpus": 1}])
    store.put_intent(intent)
    report = allocation_budget_status(store, config)
    assert report["recorded_allocation_gpu_hours"] == .5
    assert report["reserved_gpu_hours"] == 0
    config.policy.allocation_budget_allowance = 0
    assert budget_admission(store, config, profile)["allowed"]
    assert not budget_admission(store, config, profile, elements=2)["allowed"]
    config.policy.allocation_gpu_hours_budget = None
    assert budget_admission(store, config, profile, elements=100)["allowed"]


def test_proven_resultless_oom_exhaustion_is_terminal_failure(tmp_path):
    store, campaign, tid = imported(tmp_path, mode="pooled", gpu=1)
    attempt = fake_attempt(store, campaign, tid, state="OUT_OF_MEMORY", result=False)
    # Simulate the previously authorized largest resource envelope on the last attempt.
    attempt["resources"]["memory_mib"] = 2000
    atomic_json(__import__("pathlib").Path(attempt["attempt_dir"]) / "attempt.json", attempt)
    decisions = retry_ready_failures(store, Config(), [campaign])
    assert decisions[0]["decision"] == "retry_blocked"
    assert store.task_state(campaign["campaign_id"], tid)["reason"] == "retry_exhausted"
    assert work_status(store, decisions)["settled"]


def test_direct_cpu_partial_bundle_reuses_completed_application_row(tmp_path):
    # Application receipt models RFD4's per-row completion, without importing a model.
    code = '''from pathlib import Path
import os
row=Path("row1.complete")
if not row.exists():
    Path("row1.executions").write_text("once")
    row.write_text("validated")
if os.environ["HHQ_ATTEMPT_ID"].startswith("000001_"):
    raise MemoryError()
Path("row2.complete").write_text("validated")
'''
    store, campaign, tid = imported(tmp_path, command=f'{sys.executable} -c {__import__("shlex").quote(code)}')
    cid = campaign["campaign_id"]
    assert run_task(store.root, cid, tid) == 1
    authorize_retry(store, cid, tid)
    assert run_task(store.root, cid, tid) == 0
    assert run_task(store.root, cid, tid) == 0
    assert (tmp_path / "row1.executions").read_text() == "once"
    assert (tmp_path / "row2.complete").exists()
    assert len(store.attempts(cid, tid)) == 2


def test_budget_unknown_request_blocks_only_if_enabled(tmp_path):
    store = StateStore(tmp_path / "state")
    store.put_intent({"token": "test", "backend": "worker", "state": "unknown",
                      "extra_args": ["--gres", "gpu:1"]})
    config = Config(policy={"allocation_gpu_hours_budget": 100})
    profile = Profile(partition="gpu", gres="gpu:1")
    assert not budget_admission(store, config, profile)["allowed"]
    config.policy.allocation_gpu_hours_budget = None
    assert budget_admission(store, config, profile)["allowed"]


@pytest.mark.parametrize("reason,exit_code", [("gpu_oom", 1), ("timeout", 124), ("output_validation", 65), ("application_exit", 1)])
def test_later_allocation_failure_never_reclassifies_specific_finished_error(tmp_path, reason, exit_code):
    store, campaign, tid = imported(tmp_path, mode="pooled", gpu=1)
    attempt = fake_attempt(store, campaign, tid, state="OUT_OF_MEMORY")
    from pathlib import Path
    result_path = Path(attempt["attempt_dir"]) / "result.json"
    result = json.loads(result_path.read_text())
    result.update(status="failed", reason=reason, exit_code=exit_code)
    atomic_json(result_path, result)
    decision = retry_ready_failures(store, Config(), [campaign])[0]
    assert decision["decision"] == "retry_blocked"
    assert effective_task(store, store.task(campaign["campaign_id"], tid)).resources.memory_mib == 1000


def test_old_generic_kill_is_not_caused_by_later_worker_preemption(tmp_path):
    store, campaign, tid = imported(tmp_path, mode="pooled", gpu=1)
    attempt = fake_attempt(store, campaign, tid, state="PREEMPTED")
    from pathlib import Path
    path = Path(attempt["attempt_dir"]) / "result.json"
    result = json.loads(path.read_text())
    result["finished_at"] = "2026-10-13T01:00:00+00:00"
    atomic_json(path, result)
    assert retry_ready_failures(store, Config(), [campaign]) == []


def test_direct_retry_requests_only_its_own_group_extra_memory(tmp_path):
    from hyper_hyperqueue.backends.direct import submit_ready
    from hyper_hyperqueue.models import Resources

    source = tmp_path / "commands.jsonl"
    source.write_text(json.dumps({"argv": [sys.executable, "-c", "raise MemoryError()"]})
                      + "\n" + json.dumps({"argv": [sys.executable, "-c", "print('baseline')"]}) + "\n")
    store = StateStore(tmp_path / "state")
    campaign = ingest_file(store, source, mode="direct", cwd=tmp_path, format="jsonl",
        resources=Resources(memory_mib=1000, runtime_seconds=1), profile_names=["cpu"],
        retry={"max_attempts": 4, "reasons": ["host_oom"], "oom_memory_multipliers": [1.25, 1.5, 2]},
        output={"repeatable": True, "mode": "in_place"})
    cid, tid = campaign["campaign_id"], campaign["task_ids"][0]
    assert run_task(store.root, cid, tid) == 1
    authorize_retry(store, cid, tid)
    config = Config(profiles={"cpu": Profile(partition="cpu", memory_mib=1000)})
    class Slurm:
        def query_jobs(self):
            return []
        def submit(self, script, **kwargs):
            return {"status": "accepted", "job_id": str(len(store.intents()) + 100)}
    with store.lock():
        result = submit_ready(store, config, cid, Slurm())
    memory = [i["extra_args"][i["extra_args"].index("--mem") + 1] for i in result["submitted_arrays"]]
    assert sorted(memory) == ["1000M", "1250M"]
    assert config.profiles["cpu"].memory_mib == 1000


def test_later_same_worker_command_prevents_attributing_its_oom_to_previous_task(tmp_path):
    from pathlib import Path
    store, campaign, tid = imported(tmp_path, mode="pooled", gpu=1)
    first = fake_attempt(store, campaign, tid, state="OUT_OF_MEMORY")
    # A later task on the same allocation can be the OOM culprit, even within120s.
    later = {**first, "attempt_id": "000002_later", "started_at": "2026-10-14T01:00:00.500000+00:00"}
    path = Path(first["attempt_dir"]).parent / "000002_later" / "attempt.json"
    atomic_json(path, later)
    from hyper_hyperqueue.adaptive_recovery import allocation_caused_failure, scheduler_proof
    first["result"] = json.loads((Path(first["attempt_dir"]) / "result.json").read_text())
    proof = scheduler_proof(store, store.task(campaign["campaign_id"], tid), first)
    assert proof["later_attempt_on_allocation"]
    assert not allocation_caused_failure(proof, first)
