import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path

from hyper_hyperqueue import controller
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config
from hyper_hyperqueue.runner import internal_fencing_evidence, run_task
from hyper_hyperqueue.state import StateStore, atomic_json


def campaign(tmp_path, commands="exit 3\ntrue", **kwargs):
    source = tmp_path / "commands"
    source.write_text(commands)
    store = StateStore(tmp_path / "state")
    value = ingest_file(
        store,
        source,
        mode="pooled",
        cwd=tmp_path,
        retry={"max_attempts": 2, "reasons": ["application_exit", "worker_lost"]},
        output={"repeatable": True, "mode": "in_place"},
        **kwargs,
    )
    return store, value


class Slurm:
    def __init__(self, records=(), matches=()):
        self.records, self.matches = list(records), list(matches)

    def accounting(self, ids, since):
        assert datetime.now(UTC) - since < __import__("datetime").timedelta(days=7)
        return self.records

    def find_submission(self, token, since):
        return self.matches

    def query_jobs(self):
        return []


def test_unknown_array_adoption_deduplicates_parent(tmp_path):
    store = StateStore(tmp_path / "state")
    store.put_intent(
        {
            "token": "abcdefgh",
            "backend": "direct",
            "campaign_id": "campaign",
            "task_ids": [],
            "state": "unknown",
        }
    )
    rows = [
        {"job_id": "100_0", "array_job_id": "100", "name": "hhq-abcdefgh"},
        {"job_id": "100_1", "array_job_id": "100", "name": "hhq-abcdefgh"},
    ]
    controller.reconcile_slurm(store, Slurm(matches=rows), [])
    assert store.intents()[0]["job_id"] == "100"
    assert store.intents()[0]["job_name"] == "hhq-abcdefgh"
    assert store.intents()[0]["state"] == "submitted"


def test_hq_retry_retires_only_failed_occurrence_and_honors_cap(tmp_path):
    store, value = campaign(tmp_path)
    cid = value["campaign_id"]
    failed, untouched = value["task_ids"]
    assert run_task(store.root, cid, failed) == 3
    store.put_intent(
        {
            "token": "abcdefgh",
            "backend": "hq",
            "campaign_id": cid,
            "task_ids": value["task_ids"],
            "state": "submitted",
            "hq_job_id": 7,
        }
    )
    decisions = controller.retry_ready_failures(store, Config(state_dir=str(store.root)), [value])
    assert decisions[0]["decision"] == "retry_authorized"
    assert store.task_state(cid, failed)["state"] == "retry_wait"
    assert store.task_state(cid, untouched)["state"] == "ready"
    assert store.intents()[0]["retired_task_ids"] == [failed]
    assert run_task(store.root, cid, failed) == 3
    decisions = controller.retry_ready_failures(store, Config(), [value])
    assert decisions[0]["decision"] == "retry_blocked"
    assert "limit" in decisions[0]["reason"]
    store.update_task_state(cid, failed, {"state": "canceled"})
    assert controller.retry_ready_failures(store, Config(), [value]) == []


def test_actual_hq_task_mapping_canceled_never_becomes_worker_loss(tmp_path):
    store, value = campaign(tmp_path, commands="true")
    tid = value["task_ids"][0]
    cid = value["campaign_id"]
    store.update_task_state(cid, tid, {"state": "submitted", "submission_token": "abcdefgh"})
    store.put_intent(
        {
            "token": "abcdefgh",
            "backend": "hq",
            "campaign_id": cid,
            "task_ids": [tid],
            "state": "submitted",
            "hq_job_id": 8,
            "epoch": 1,
        }
    )

    class HQ:
        def tasks(self, job_id):
            return {str(job_id): [{"id": 0, "state": "canceled"}]}

    controller.reconcile_hq(store, HQ())
    assert store.task_state(cid, tid)["state"] == "canceled"
    assert store.task_state(cid, tid)["reason"] == "user_cancelled"


def test_unknown_attempt_needs_exact_terminal_owned_allocation(tmp_path, monkeypatch):
    store, value = campaign(tmp_path, commands="true")
    cid = value["campaign_id"]
    tid = value["task_ids"][0]
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("HHQ_SERVER_EPOCH", "2")
    assert run_task(store.root, cid, tid) == 0
    attempt = store.attempts(cid, tid)[0]
    (Path(attempt["attempt_dir"]) / "result.json").unlink()
    (store.campaign_dir(cid) / "completions" / f"{tid}.json").unlink()
    assert controller.retry_ready_failures(store, Config(), [value]) == []
    intent = {
        "token": "abcdefgh",
        "backend": "worker",
        "profile": "cpu",
        "epoch": 2,
        "job_id": "100",
        "job_name": "hhq-abcdefgh",
        "state": "submitted",
    }
    store.put_intent(intent)
    wrong = {"job_id": "100", "name": "unrelated", "comment": "wrong", "state": "FAILED"}
    controller.reconcile_slurm(store, Slurm(records=[wrong]), [])
    assert store.intents()[0]["state"] == "submitted"
    assert controller.retry_ready_failures(store, Config(), [value]) == []
    owned = {**wrong, "name": "hhq-abcdefgh", "comment": ""}
    controller.reconcile_slurm(store, Slurm(records=[owned]), [])
    assert store.intents()[0]["state"] == "terminal"
    result = controller.retry_ready_failures(store, Config(), [value])
    assert result[0]["decision"] == "retry_authorized"
    assert store.task_state(cid, tid)["recovery_proof"]["attempt_id"] == attempt["attempt_id"]
    assert run_task(store.root, cid, tid) == 0


def test_startup_failure_circuit_cooldown_and_explicit_reset(tmp_path):
    store = StateStore(tmp_path / "state")
    config = Config(
        profiles={"cpu": {"partition": "cpu"}}, policy={"max_rpc_failures": 2, "submit_cooldown_seconds": 30}
    )
    now = time.time()
    owned = [
        {
            "token": f"token{i}",
            "profile": "cpu",
            "state": "terminal",
            "job_id": str(i),
            "created_at": datetime.fromtimestamp(now - 60 + i, UTC).isoformat(),
            "terminal_evidence": [{"state": "FAILED", "elapsed_seconds": 1}],
        }
        for i in range(2)
    ]
    paused, reasons = controller.allocation_gates(store, config, owned, clock=now)
    assert paused == {"cpu"} and "STARTUP_CIRCUIT" in reasons[0]
    reset = {"cpu": datetime.fromtimestamp(now - 1, UTC).isoformat()}
    assert controller.allocation_gates(store, config, owned, clock=now, reset_after=reset) == (set(), [])
    owned = [
        {**owned[-1], "created_at": datetime.fromtimestamp(now - 1, UTC).isoformat(), "state": "submitted"}
    ]
    assert controller.allocation_gates(store, config, owned, clock=now)[0] == {"cpu"}


def test_completed_cancel_request_not_reprocessed(tmp_path, monkeypatch):
    store, value = campaign(tmp_path, commands="true")
    value["draining"] = True
    store.save_campaign(value)
    path = store.root / "requests" / "cancel-campaign.done.json"
    atomic_json(path, {"action": "cancel_hq_campaign", "campaign_id": value["campaign_id"]})
    monkeypatch.setattr(controller, "assert_epoch", lambda *_: None)
    result = controller.tick(Config(state_dir=str(store.root)), epoch=1, slurm=Slurm(), hq=object())
    assert path.exists()
    assert not path.with_suffix(".done.json").exists()
    assert result["plan"]["actions"] == []


def test_running_allocation_tracks_actual_remaining_lifetime(tmp_path):
    store = StateStore(tmp_path / "state")
    store.put_intent(
        {"token": "abcdefgh", "backend": "worker", "profile": "cpu", "state": "submitted", "job_id": "99"}
    )
    controller.reconcile_slurm(
        store,
        Slurm(),
        [{"job_id": "99", "state": "RUNNING", "start_time": time.time() - 90, "time_limit_seconds": 100}],
    )
    assert 9 < store.intents()[0]["remaining_seconds"] <= 10


def test_pending_allocation_forecast_does_not_inflate_lifetime(tmp_path):
    store = StateStore(tmp_path / "state")
    store.put_intent(
        {"token": "abcdefgh", "backend": "worker", "profile": "cpu", "state": "submitted", "job_id": "99"}
    )
    controller.reconcile_slurm(
        store,
        Slurm(),
        [{"job_id": "99", "state": "PENDING", "start_time": time.time() + 6000, "time_limit_seconds": 900}],
    )
    assert store.intents()[0]["remaining_seconds"] == 900


def test_uncertain_hq_admission_stops_new_submissions(tmp_path):
    store, value = campaign(tmp_path)
    store.put_intent(
        {
            "token": "abcdefgh",
            "backend": "hq",
            "campaign_id": value["campaign_id"],
            "task_ids": [value["task_ids"][0]],
            "state": "unknown",
        }
    )
    # There is still a second unsubmitted task; uncertainty prevents growth.
    assert controller.admit_hq(store, Config(), object(), value, epoch=None) is False
    assert len(store.intents()) == 1


def test_old_failure_cannot_authorize_replay_of_new_unknown_submission(tmp_path):
    store, value = campaign(tmp_path, commands="exit 3")
    tid = value["task_ids"][0]
    cid = value["campaign_id"]
    assert run_task(store.root, cid, tid) == 3
    attempt = store.attempts(cid, tid)[0]
    store.update_task_state(
        cid,
        tid,
        {"state": "unknown", "submission_token": "newtoken", "after_attempt_id": attempt["attempt_id"]},
    )
    store.put_intent(
        {"token": "newtoken", "backend": "direct", "campaign_id": cid, "task_ids": [tid], "state": "unknown"}
    )
    decisions = controller.retry_ready_failures(store, Config(), [value])
    assert decisions[0]["decision"] == "retry_blocked"
    assert "uncertain submission" in decisions[0]["reason"]
    assert store.intents()[0]["state"] == "unknown"


def test_direct_unknown_and_rejection_circuits(tmp_path):
    store = StateStore(tmp_path / "state")
    config = Config(policy={"max_rpc_failures": 2})
    base = {"backend": "direct", "campaign_id": "campaign", "task_ids": [], "state": "rejected"}
    for i in range(2):
        store.put_intent({**base, "token": f"token{i}"})
    assert "CIRCUIT" in controller.direct_admission_gate(store, config, "campaign")
    store.put_intent({**base, "token": "unknown", "state": "unknown"})
    assert "UNKNOWN" in controller.direct_admission_gate(store, config, "campaign")


def fenced_cancellation(tmp_path, monkeypatch, *, dispatch_epoch=4, worker_epoch=None):
    store, value = campaign(tmp_path, commands="true")
    cid = value["campaign_id"]
    tid = value["task_ids"][0]
    monkeypatch.setenv("SLURM_JOB_ID", "700")
    monkeypatch.setenv("HHQ_SERVER_EPOCH", str(dispatch_epoch))
    physical_epoch = dispatch_epoch if worker_epoch is None else worker_epoch
    if worker_epoch is not None:
        monkeypatch.setenv("HHQ_WORKER_EPOCH", str(worker_epoch))
    store.update_task_state(cid, tid, {"server_epoch": dispatch_epoch})
    assert run_task(store.root, cid, tid) == 0
    previous = store.attempts(cid, tid)[0]
    (store.campaign_dir(cid) / "completions" / f"{tid}.json").unlink()
    result = previous["result"]
    result.update(
        status="canceled",
        reason="user_cancellation",
        exit_code=143,
        finished_at=datetime.now(UTC).isoformat(),
    )
    atomic_json(Path(previous["attempt_dir"]) / "result.json", result)
    store.put_intent(
        {
            "token": "worker_token",
            "backend": "worker",
            "profile": "cpu",
            "epoch": physical_epoch,
            "job_id": "700",
            "state": "terminal",
        }
    )
    intent_path = store.root / "service" / f"recovery-intent-{physical_epoch}.json"
    atomic_json(
        intent_path,
        {
            "schema": "hhq-service-fencing-intent/1",
            "reason": "generation_recovery",
            "previous_epoch": physical_epoch,
            "created_at": time.time() - 2,
            "allocations": [{"job_id": "700", "token": "worker_token"}],
        },
    )
    proof_path = store.root / "service" / f"recovery-{physical_epoch}.json"
    atomic_json(
        proof_path,
        {
            "ready": True,
            "proof": {
                "schema": "hhq-fencing-proof/1",
                "quiescent": True,
                "previous_epoch": physical_epoch,
                "intent": {
                    "path": str(intent_path),
                    "sha256": hashlib.sha256(intent_path.read_bytes()).hexdigest(),
                },
                "allocations": [
                    {
                        "job_id": "700",
                        "token": "worker_token",
                        "owned": True,
                        "live": False,
                        "accounting_state": "CANCELLED",
                    }
                ],
            },
        },
    )
    return store, value, intent_path, proof_path


def test_bound_internal_generation_cancel_retries_without_rewriting_history(tmp_path, monkeypatch):
    store, value, _intent_path, proof_path = fenced_cancellation(tmp_path, monkeypatch)
    tid = value["task_ids"][0]
    cid = value["campaign_id"]
    previous = store.attempts(cid, tid)[0]
    result_path = Path(previous["attempt_dir"]) / "result.json"
    immutable_result = result_path.read_bytes()
    assert internal_fencing_evidence(store, store.task(cid, tid), previous, proof_path)
    decisions = controller.retry_ready_failures(store, Config(), [value])
    assert decisions[0]["decision"] == "retry_authorized"
    assert result_path.read_bytes() == immutable_result
    assert run_task(store.root, cid, tid) == 0
    assert len(store.attempts(cid, tid)) == 2


def test_service_fence_cannot_override_user_cancel_or_changed_intent(tmp_path, monkeypatch):
    store, value, intent_path, _proof_path = fenced_cancellation(tmp_path, monkeypatch)
    cid = value["campaign_id"]
    original = intent_path.read_bytes()
    intent_path.write_bytes(original + b" ")
    assert controller.retry_ready_failures(store, Config(), [value]) == []
    intent_path.write_bytes(original)
    atomic_json(
        store.root / "requests" / f"cancel-{cid}.done.json",
        {"action": "cancel_hq_campaign", "campaign_id": cid},
    )
    assert controller.retry_ready_failures(store, Config(), [value]) == []


def test_cancellation_that_predates_fencing_remains_canceled(tmp_path, monkeypatch):
    store, value, _, _proof_path = fenced_cancellation(tmp_path, monkeypatch)
    tid = value["task_ids"][0]
    cid = value["campaign_id"]
    previous = store.attempts(cid, tid)[0]
    result = previous["result"]
    result["finished_at"] = datetime.fromtimestamp(time.time() - 60, UTC).isoformat()
    atomic_json(Path(previous["attempt_dir"]) / "result.json", result)
    assert controller.retry_ready_failures(store, Config(), [value]) == []
    assert store.task_state(cid, tid)["state"] == "canceled"


def test_restored_waiting_task_recovers_from_second_physical_generation(tmp_path, monkeypatch):
    # Journal restored a never-started epoch-5 dispatch onto an epoch-6 worker.
    # A second takeover must fence worker 6, while stale-dispatch checks still
    # use the immutable HQ submission's epoch independently.
    store, value, _intent, proof = fenced_cancellation(
        tmp_path, monkeypatch, dispatch_epoch=5, worker_epoch=6
    )
    tid = value["task_ids"][0]
    cid = value["campaign_id"]
    previous = store.attempts(cid, tid)[0]
    env = previous["context"]["environment"]
    assert env["HHQ_SERVER_EPOCH"] == "5" and env["HHQ_WORKER_EPOCH"] == "6"
    assert internal_fencing_evidence(store, store.task(cid, tid), previous, proof)
    assert controller.retry_ready_failures(store, Config(), [value])[0]["decision"] == "retry_authorized"
    assert store.task_state(cid, tid)["recovery_proof"]["server_epoch"] == "6"
    store.update_task_state(cid, tid, {"server_epoch": 7})
    monkeypatch.setenv("HHQ_WORKER_EPOCH", "7")
    assert run_task(store.root, cid, tid) == 75  # Old dispatch is now stale.
    assert len(store.attempts(cid, tid)) == 1
    monkeypatch.setenv("HHQ_SERVER_EPOCH", "7")
    assert run_task(store.root, cid, tid) == 0
    assert len(store.attempts(cid, tid)) == 2


def test_resultless_restored_task_matches_physical_allocation_epoch(tmp_path, monkeypatch):
    store, value = campaign(tmp_path, commands="true")
    tid = value["task_ids"][0]
    cid = value["campaign_id"]
    monkeypatch.setenv("SLURM_JOB_ID", "800")
    monkeypatch.setenv("HHQ_SERVER_EPOCH", "5")
    monkeypatch.setenv("HHQ_WORKER_EPOCH", "6")
    assert run_task(store.root, cid, tid) == 0
    previous = store.attempts(cid, tid)[0]
    (Path(previous["attempt_dir"]) / "result.json").unlink()
    (store.campaign_dir(cid) / "completions" / f"{tid}.json").unlink()
    store.put_intent(
        {
            "token": "physical_worker",
            "backend": "worker",
            "profile": "cpu",
            "epoch": 6,
            "job_id": "800",
            "state": "terminal",
            "terminal_evidence": [
                {"job_id": "800", "name": "hhq-physical_worker", "comment": "", "state": "FAILED"}
            ],
        }
    )
    decisions = controller.retry_ready_failures(store, Config(), [value])
    assert decisions[0]["decision"] == "retry_authorized"
    assert store.task_state(cid, tid)["recovery_proof"]["server_epoch"] == "6"
