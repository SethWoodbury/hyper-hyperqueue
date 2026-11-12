"""Tiny synthetic CPU commands and scheduler fixtures; never submit Slurm jobs."""
import json
from pathlib import Path

import pytest

from hyper_hyperqueue import cli, controller
from hyper_hyperqueue.adaptive_recovery import terminal_failure_valid
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config, Resources, RetryPolicy
from hyper_hyperqueue.runner import authorize_retry, run_task
from hyper_hyperqueue.state import StateStore, atomic_json


def make(tmp_path, command="true", output=None, **policy):
    commands = tmp_path / "commands"
    commands.write_text(command + "\n")
    store = StateStore(tmp_path / "state")
    campaign = ingest_file(
        store, commands, mode="pooled", cwd=tmp_path,
        retry={"max_attempts": 3, "max_infrastructure_retries": 3, "same_error_limit": 2,
               "reasons": ["application_exit", "worker_lost", "preempted"], **policy},
        output=output or {"repeatable": True, "mode": "in_place"},
    )
    return store, campaign, campaign["task_ids"][0]


def test_legacy_policy_serialization_unchanged():
    raw = {"max_attempts": 3, "reasons": ["worker_lost", "preempted"],
           "memory_increment_mib": 2048, "max_memory_mib": None}
    assert RetryPolicy.model_validate(raw).model_dump() == raw


def test_two_matching_errors_stop_and_duplicate_cannot_retry(tmp_path):
    store, campaign, tid = make(tmp_path, "echo 'ValueError: broken input' >&2; exit 3")
    cid = campaign["campaign_id"]
    assert run_task(store.root, cid, tid) == 3
    assert run_task(store.root, cid, tid) == 75  # Controller authorization is required.
    authorize_retry(store, cid, tid)
    assert run_task(store.root, cid, tid) == 3
    with pytest.raises(ValueError, match="Same application error repeated 2"):
        authorize_retry(StateStore(store.root), cid, tid)
    assert len(store.attempts(cid, tid)) == 2


def test_transient_error_then_success_is_not_run_again(tmp_path):
    store, campaign, tid = make(tmp_path, "if [ ! -f once ]; then touch once; exit 1; fi")
    cid = campaign["campaign_id"]
    assert run_task(store.root, cid, tid) == 1
    authorize_retry(store, cid, tid)
    assert run_task(store.root, cid, tid) == 0
    assert run_task(store.root, cid, tid) == 0
    assert len(store.attempts(cid, tid)) == 2


def test_application_limit_remains_three_for_different_errors(tmp_path):
    store, campaign, tid = make(tmp_path, 'echo "$HHQ_ATTEMPT_ID" >&2; exit 1', same_error_limit=0)
    cid = campaign["campaign_id"]
    for _ in range(3):
        assert run_task(store.root, cid, tid) != 0
        if len(store.attempts(cid, tid)) < 3:
            authorize_retry(store, cid, tid)
    with pytest.raises(ValueError, match="Application attempt limit"):
        authorize_retry(store, cid, tid)


def test_proven_preemption_has_separate_budget_and_is_idempotent(tmp_path, monkeypatch):
    store, campaign, tid = make(tmp_path, max_attempts=1, max_infrastructure_retries=1)
    cid = campaign["campaign_id"]
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("HHQ_SERVER_EPOCH", "2")
    store.update_task_state(cid, tid, {"server_epoch": 2})
    assert run_task(store.root, cid, tid) == 0
    attempt = store.attempts(cid, tid)[0]
    (Path(attempt["attempt_dir"]) / "result.json").unlink()
    (store.campaign_dir(cid) / "completions" / f"{tid}.json").unlink()
    store.put_intent({"token": "worker-token", "backend": "worker", "job_id": "100",
                      "epoch": 2, "state": "terminal", "terminal_evidence": [
                          {"job_id": "100", "name": "hhq-worker-token", "comment": "",
                           "state": "PREEMPTED"}]})
    result = controller.retry_ready_failures(store, Config(), [campaign])
    assert result[0]["decision"] == "retry_authorized"
    assert store.task_state(cid, tid)["infrastructure_attempt_ids"] == [attempt["attempt_id"]]
    assert controller.retry_ready_failures(store, Config(), [campaign]) == []
    assert run_task(store.root, cid, tid) == 0
    assert len(store.attempts(cid, tid)) == 2  # One infrastructure loss + one application attempt.


def unstarted(tmp_path, **policy):
    store, campaign, tid = make(tmp_path, **policy)
    cid = campaign["campaign_id"]
    store.put_intent({"token": "hq-token", "backend": "hq", "campaign_id": cid,
                      "task_ids": [tid], "hq_job_id": 7, "hq_task_ids": {tid: 0},
                      "state": "submitted", "epoch": 1})
    store.put_intent({"token": "worker-token", "backend": "worker", "job_id": "100",
                      "state": "terminal", "epoch": 2, "terminal_evidence": [
                          {"job_id": "100", "name": "hhq-worker-token", "state": "PREEMPTED"}]})
    store.update_task_state(cid, tid, {
        "state": "interrupted", "reason": "worker_lost", "submission_token": "hq-token",
        "hq_job_id": 7, "hq_task_id": 0, "server_epoch": 1,
        "hq_terminal_evidence": {"id": 0, "state": "failed", "error": "worker was lost"},
    })
    return store, campaign, tid


def test_loss_before_attempt_recovers_once_and_retires_exact_dispatch(tmp_path):
    store, campaign, tid = unstarted(tmp_path)
    cid = campaign["campaign_id"]
    assert controller.retry_ready_failures(store, Config(), [campaign])[0]["decision"] == "retry_authorized"
    state = store.task_state(cid, tid)
    assert state["prestart_retry_dispatches"] == ["hq-token:7:0"]
    assert controller.retry_ready_failures(StateStore(store.root), Config(), [campaign]) == []
    assert next(i for i in store.intents() if i["token"] == "hq-token")["retired_task_ids"] == [tid]
    store.update_task_state(cid, tid, {"server_epoch": None})
    assert run_task(store.root, cid, tid) == 0


@pytest.mark.parametrize("change", ["live", "unknown", "wrong_token", "wrong_mapping", "canceled"])
def test_unstarted_loss_does_not_bypass_uncertainty_or_cancellation(tmp_path, change):
    store, campaign, tid = unstarted(tmp_path)
    if change in {"live", "unknown"}:
        store.put_intent({"token": "other-worker", "backend": "worker", "job_id": "101",
                          "state": "running" if change == "live" else "unknown", "epoch": 3})
    elif change == "wrong_token":
        store.update_task_state(campaign["campaign_id"], tid, {"submission_token": "unrelated"})
    elif change == "wrong_mapping":
        store.update_task_state(campaign["campaign_id"], tid, {"hq_task_id": 1})
    else:
        store.update_task_state(campaign["campaign_id"], tid, {"state": "canceled"})
    decisions = controller.retry_ready_failures(store, Config(), [campaign])
    assert not any(d["decision"] == "retry_authorized" for d in decisions)
    assert not store.attempts(campaign["campaign_id"], tid)


@pytest.mark.parametrize("output", [{"mode": "isolated", "repeatable": False},
                                   {"mode": "manual_on_ambiguity", "repeatable": True}])
def test_unstarted_loss_does_not_replay_ambiguous_outputs(tmp_path, output):
    store, campaign, tid = unstarted(tmp_path, output=output)
    from hyper_hyperqueue.runner import authorize_unstarted_retry
    with pytest.raises(ValueError, match="Policy does not allow replay"):
        authorize_unstarted_retry(store, campaign["campaign_id"], tid)


def test_success_record_is_adopted_without_retry_authorization(tmp_path):
    store, campaign, tid = make(tmp_path)
    cid = campaign["campaign_id"]
    assert run_task(store.root, cid, tid) == 0
    (store.campaign_dir(cid) / "completions" / f"{tid}.json").unlink()
    assert run_task(store.root, cid, tid) == 0
    assert len(store.attempts(cid, tid)) == 1


def test_exhausted_unstarted_loss_is_a_proven_terminal_failure(tmp_path):
    store, campaign, tid = unstarted(tmp_path, max_infrastructure_retries=0)
    cid = campaign["campaign_id"]
    result = controller.retry_ready_failures(store, Config(), [campaign])
    assert "Infrastructure retry limit" in result[0]["reason"]
    task, state = store.task(cid, tid), store.task_state(cid, tid)
    assert state["state"] == "failed"
    assert terminal_failure_valid(store, task, state)
    assert run_task(store.root, cid, tid) == 75
    assert not store.attempts(cid, tid)
    store.put_intent({"token": "new-worker", "backend": "worker", "job_id": "101", "state": "running"})
    assert terminal_failure_valid(store, task, state)  # Later unrelated work cannot invalidate the proof.


def test_prestart_recovery_survives_crash_before_old_hq_retirement(tmp_path):
    store, campaign, tid = unstarted(tmp_path)
    from hyper_hyperqueue.runner import authorize_unstarted_retry
    authorize_unstarted_retry(store, campaign["campaign_id"], tid)

    class HQ:
        def tasks(self, job_id):
            return [{"id": 0, "state": "failed", "error": "worker was lost"}]

    controller.reconcile_hq(StateStore(store.root), HQ())
    intent = next(i for i in store.intents() if i["token"] == "hq-token")
    assert intent["retired_task_ids"] == [tid]


def test_uncertain_or_user_cancelled_worker_is_not_a_prestart_proof(tmp_path):
    store, campaign, _tid = unstarted(tmp_path)
    intent = next(i for i in store.intents() if i["backend"] == "worker")
    intent["terminal_evidence"][0]["state"] = "CANCELLED"
    store.put_intent(intent)
    result = controller.retry_ready_failures(store, Config(), [campaign])
    assert result[0]["decision"] == "retry_blocked"


def test_resultless_infrastructure_exhaustion_can_settle(tmp_path, monkeypatch):
    store, campaign, tid = make(tmp_path, max_infrastructure_retries=0)
    cid = campaign["campaign_id"]
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("HHQ_SERVER_EPOCH", "2")
    assert run_task(store.root, cid, tid) == 0
    previous = store.attempts(cid, tid)[0]
    (Path(previous["attempt_dir"]) / "result.json").unlink()
    (store.campaign_dir(cid) / "completions" / f"{tid}.json").unlink()
    store.put_intent({"token": "worker-token", "backend": "worker", "job_id": "100", "epoch": 2,
                      "state": "terminal", "terminal_evidence": [
                          {"job_id": "100", "name": "hhq-worker-token", "state": "PREEMPTED"}]})
    result = controller.retry_ready_failures(store, Config(), [campaign])
    assert "Infrastructure retry limit" in result[0]["reason"]
    assert terminal_failure_valid(store, store.task(cid, tid), store.task_state(cid, tid))


def test_terminal_identity_fallback_requires_previous_owned_incarnation(tmp_path):
    intent = {"token": "worker-token", "ownership_observation": {
        "job_id": "100", "user": "alice", "submit_time": 1, "start_time": 2}}
    record = {**intent["ownership_observation"], "name": "allocation", "comment": "", "state": "PREEMPTED"}
    assert controller.owned_record(record, intent)
    assert not controller.owned_record({**record, "submit_time": 3}, intent)
    assert not controller.owned_record({**record, "comment": "foreign"}, intent)
    assert not controller.owned_record(record, {"token": "worker-token"})


def test_live_owned_observation_binds_terminal_accounting(tmp_path):
    store = StateStore(tmp_path / "state")
    store.put_intent({"token": "worker-token", "backend": "worker", "job_id": "100",
                      "state": "submitted"})

    class Slurm:
        user = "alice"

        def accounting(self, ids, since):
            return [{"job_id": "100", "user": "alice", "submit_time": 1, "start_time": 2,
                     "state": "PREEMPTED", "name": "allocation", "comment": ""}]

    row = {"job_id": "100", "user": "alice", "submit_time": 1, "start_time": 2,
           "state": "RUNNING", "name": "hhq-worker-token", "comment": ""}
    controller.reconcile_slurm(store, Slurm(), [row])
    controller.reconcile_slurm(store, Slurm(), [])
    assert store.intents()[0]["state"] == "terminal"


def test_positive_submission_rejection_does_not_block_unstarted_recovery(tmp_path):
    store, campaign, _tid = unstarted(tmp_path)
    store.put_intent({"token": "rejected-worker", "backend": "worker", "state": "rejected",
                      "response": {"status": "rejected"}})
    assert controller.retry_ready_failures(store, Config(), [campaign])[0]["decision"] == "retry_authorized"


def test_new_cli_defaults_and_replay_gate_are_explicit(tmp_path, monkeypatch):
    saved = {}
    monkeypatch.setattr(cli, "options", lambda *a: (Config(), None, Resources()))
    monkeypatch.setattr(cli.api, "submit", lambda *a, **kw: saved.update(kw) or {})
    monkeypatch.setattr(cli, "output", lambda value: None)
    cli.submit(tmp_path / "commands", no_service=True)
    assert saved["retry"].max_attempts == 3
    assert saved["retry"].max_infrastructure_retries == 3
    assert saved["retry"].same_error_limit == 2
    assert "application_exit" in saved["retry"].reasons
    assert saved["output"].repeatable is False


def test_infrastructure_limit_is_not_reset_by_reopening_store(tmp_path):
    store, campaign, tid = make(tmp_path)
    state = {"infrastructure_attempt_ids": ["lost1", "lost2", "lost3"]}
    atomic_json(store.campaign_dir(campaign["campaign_id"]) / "task_states" / f"{tid}.json", state)
    from hyper_hyperqueue.retry_policy import retry_budget_error
    task = StateStore(store.root).task(campaign["campaign_id"], tid)
    assert "Infrastructure retry limit" in retry_budget_error(
        task, [], json.loads((store.campaign_dir(campaign["campaign_id"]) / "task_states" / f"{tid}.json").read_text()),
        infrastructure_attempt="lost4",
    )
