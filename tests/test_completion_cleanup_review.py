"""Independent adversarial checks for automatic end-of-work cleanup."""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from hyper_hyperqueue.backends.slurm import CommandResult
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config
from hyper_hyperqueue.runner import run_task
from hyper_hyperqueue.service.completion import cleanup_workers, hq_settled, work_status
from hyper_hyperqueue.service.supervisor import finish_completed_service
from hyper_hyperqueue.state import StateStore, atomic_json


def add_campaign(tmp_path, store, name, command="true", **kwargs):
    source = tmp_path / f"{name}.commands"
    source.write_text(command + "\n")
    return ingest_file(store, source, mode="pooled", cwd=tmp_path, import_key=name, **kwargs)


@pytest.mark.parametrize("paused", ["pause_dispatch", "draining"])
def test_other_campaign_paused_unadmitted_work_blocks_cleanup(tmp_path, paused):
    store = StateStore(tmp_path / "state")
    first = add_campaign(tmp_path, store, "complete")
    assert run_task(store.root, first["campaign_id"], first["task_ids"][0]) == 0
    second = add_campaign(tmp_path, store, "later")
    second[paused] = True
    store.save_campaign(second)
    result = work_status(store)
    assert not result["settled"]
    assert result["counts"] == {"succeeded": 1, "ready": 1}
    assert result["blockers"][0]["campaign_id"] == second["campaign_id"]


def test_success_flag_without_validated_completion_is_not_enough(tmp_path):
    store = StateStore(tmp_path / "state")
    campaign = add_campaign(tmp_path, store, "unverified")
    store.update_task_state(campaign["campaign_id"], campaign["task_ids"][0], {"state": "succeeded"})
    result = work_status(store)
    assert not result["settled"]
    assert result["blockers"][0]["reason"] == "missing_validated_completion"


def test_final_failure_does_not_skip_a_future_retry_decision(tmp_path):
    store = StateStore(tmp_path / "state")
    campaign = add_campaign(tmp_path, store, "failed", command="exit 3",
                            retry={"max_attempts": 2, "reasons": ["application_exit"]},
                            output={"mode": "in_place", "repeatable": True})
    cid, tid = campaign["campaign_id"], campaign["task_ids"][0]
    assert run_task(store.root, cid, tid) == 3
    assert work_status(store)["blockers"][0]["reason"] == "retry_decision_pending"
    decided = [{"task_id": tid, "decision": "retry_blocked", "reason": "operator review"}]
    assert work_status(store, decided)["settled"]


def test_uncertain_submission_blocks_cleanup_after_every_task_succeeded(tmp_path):
    store = StateStore(tmp_path / "state")
    campaign = add_campaign(tmp_path, store, "complete")
    assert run_task(store.root, campaign["campaign_id"], campaign["task_ids"][0]) == 0
    store.put_intent({"token": "lost_ack_token", "backend": "worker", "state": "unknown"})
    result = work_status(store)
    assert not result["settled"]
    assert result["blockers"] == [{"token": "lost_ack_token", "reason": "submission_unknown"}]


@pytest.mark.parametrize("hq_state", ["running", "waiting"])
def test_hq_must_finish_even_after_runner_completion_is_durable(tmp_path, hq_state):
    store = StateStore(tmp_path / "state")
    campaign = add_campaign(tmp_path, store, "complete")
    assert run_task(store.root, campaign["campaign_id"], campaign["task_ids"][0]) == 0
    store.put_intent({"token": "finished_runner", "backend": "hq", "state": "terminal",
                      "hq_job_id": 7, "task_ids": campaign["task_ids"]})
    hq = SimpleNamespace(jobs=lambda: [{"id": 7, "name": "hhq:finished_runner"}],
                         tasks=lambda _: [{"id": 0, "state": hq_state}])
    assert work_status(store)["settled"]
    assert not hq_settled(store, hq)


def test_unregistered_hq_job_prevents_shutdown(tmp_path):
    store = StateStore(tmp_path / "state")
    hq = SimpleNamespace(jobs=lambda: [{"id": 99, "name": "some_other_owner"}])
    assert not hq_settled(store, hq)


def test_cleanup_pending_to_running_race_never_claims_terminal(tmp_path):
    store = StateStore(tmp_path / "state")
    store.put_intent({"token": "pending_token", "backend": "worker", "state": "submitted",
                      "job_id": "123", "epoch": 7})
    calls = []

    def already_started(job_ids, expected_token):
        calls.append((job_ids, expected_token))
        return CommandResult([], 0, "Already RUNNING; no cancellation")

    slurm = SimpleNamespace(user="alice", cancel_pending=already_started)
    rows = [{"job_id": "123", "user": "alice", "state": "PENDING", "comment": "hhq:pending_token"}]
    result = cleanup_workers(store, slurm, rows, 7)
    assert calls == [(["123"], "pending_token")]
    assert not result["settled"]
    assert store.intents()[0]["state"] == "submitted"


def test_cleanup_foreign_pending_worker_is_never_canceled(tmp_path):
    store = StateStore(tmp_path / "state")
    store.put_intent({"token": "our_token", "backend": "worker", "state": "submitted",
                      "job_id": "123", "epoch": 7})
    rows = [{"job_id": "123", "user": "alice", "state": "PENDING", "comment": "hhq:other_token"}]
    result = cleanup_workers(store, SimpleNamespace(user="alice"), rows, 7)
    assert not result["settled"]
    assert result["blockers"][0]["reason"] == "worker_ownership_unproven"


def test_terminal_worker_flag_without_accounting_proof_keeps_controller_alive(tmp_path):
    store = StateStore(tmp_path / "state")
    store.put_intent({"token": "terminal_token", "backend": "worker", "state": "terminal",
                      "job_id": "123", "epoch": 7})
    assert not cleanup_workers(store, SimpleNamespace(user="alice"), [], 7)["settled"]


def test_fresh_empty_service_does_not_stop_before_first_import(tmp_path):
    result = work_status(StateStore(tmp_path / "state"))
    assert not result["settled"]
    assert result["blockers"] == [{"reason": "no_registered_work"}]


def test_partial_recovery_export_is_not_a_completed_retirement(tmp_path):
    store = StateStore(tmp_path / "state")
    campaign = add_campaign(tmp_path, store, "not_yet_exported")
    store.update_task_state(campaign["campaign_id"], campaign["task_ids"][0], {
        "state": "blocked", "reason": "exported_to_new_generation",
        "recovery_export": str(tmp_path / "not_published/manifest.json"),
    })
    assert not work_status(store)["settled"]


def finished_service(tmp_path):
    store = StateStore(tmp_path / "state")
    campaign = add_campaign(tmp_path, store, "complete")
    assert run_task(store.root, campaign["campaign_id"], campaign["task_ids"][0]) == 0
    config = Config(state_dir=str(store.root), control={"stop_when_complete": True})
    guard = SimpleNamespace(epoch=7, owner_id="owner", record={}, assert_owned=lambda: None)
    backend = SimpleNamespace(jobs=list, workers=list)
    atomic_json(store.root / "service/deployment.json",
                {"hosting": "local", "deployment_id": "deployment", "desired_state": "RUNNING"})
    atomic_json(store.root / "controller_status.json",
                {"state": "healthy", "retry_decisions": [],
                 "completion": {"ready": True, "epoch": 7, "observed_at": time.time()}})
    return store, config, guard, backend


def test_fresh_work_after_ready_report_prevents_supervisor_exit(tmp_path):
    store, config, guard, backend = finished_service(tmp_path)
    add_campaign(tmp_path, store, "arrived_after_poll")
    stopped = []
    assert not finish_completed_service(config, guard, backend, lambda: stopped.append(True))
    assert not stopped
    assert json.loads((store.root / "service/deployment.json").read_text())["desired_state"] == "RUNNING"


def test_stale_ready_report_cannot_authorize_shutdown(tmp_path):
    store, config, guard, backend = finished_service(tmp_path)
    path = store.root / "controller_status.json"
    report = json.loads(path.read_text())
    report["completion"]["observed_at"] = time.time() - 86400
    atomic_json(path, report)
    stopped = []
    assert not finish_completed_service(config, guard, backend, lambda: stopped.append(True))
    assert not stopped


def test_lost_cancellation_ack_does_not_free_owned_worker(tmp_path):
    store = StateStore(tmp_path / "state")
    store.put_intent({"token": "pending_token", "backend": "worker", "state": "submitted",
                      "job_id": "123", "epoch": 7})

    def lost_ack(*args, **kwargs):
        raise TimeoutError("accepted cancellation, acknowledgement lost")

    rows = [{"job_id": "123", "user": "alice", "state": "PENDING", "comment": "hhq:pending_token"}]
    result = cleanup_workers(store, SimpleNamespace(user="alice", cancel_pending=lost_ack), rows, 7)
    assert not result["settled"]
    assert store.intents()[0]["state"] == "submitted"
    receipt = json.loads((store.root / "service/completion_cancellations/pending_token.json").read_text())
    assert "acknowledgement lost" in receipt["error"]


def test_import_waiting_on_final_shutdown_barrier_cannot_strand_ready_work(tmp_path):
    store, config, guard, backend = finished_service(tmp_path)
    waiting, done = threading.Event(), threading.Event()
    result = {}
    source = tmp_path / "late.commands"
    source.write_text("true\n")

    def late_import():
        waiting.set()
        try:
            result["campaign"] = ingest_file(StateStore(store.root), source, mode="pooled",
                                             cwd=tmp_path, import_key="late_arrival")
        except (ValueError, RuntimeError) as exc:
            result["error"] = str(exc)
        finally:
            done.set()

    thread = threading.Thread(target=late_import, daemon=True)

    def shutdown():
        thread.start()
        assert waiting.wait(2)
        # The actual child shutdown is still inside the shared mutation barrier.
        assert not done.wait(0.05)

    assert finish_completed_service(config, guard, backend, shutdown)
    thread.join(2)
    assert done.is_set(), "Import must unblock after shutdown releases the state lock"
    assert "campaign" not in result
    assert "no new work was imported" in result["error"].lower()
    assert work_status(store)["counts"] == {"succeeded": 1}
