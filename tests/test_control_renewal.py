"""Planned rollover ownership, completion and acknowledgement tests; no Slurm calls."""

import time
from datetime import UTC, datetime

import pytest

from hyper_hyperqueue.backends.slurm import CommandResult, SlurmJob, SubmissionResult
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config
from hyper_hyperqueue.runner import run_task
from hyper_hyperqueue.service import renewal
from hyper_hyperqueue.service.leadership import LeadershipError, LeadershipGuard, atomic_record, read_record
from hyper_hyperqueue.state import StateStore, atomic_json


def job(jid, token, state="RUNNING"):
    return SlurmJob(job_id=jid, name="hhq-" + token, user="owner", state=state,
                    partition="cpu", comment="hhq:" + token)


class Slurm:
    user = "owner"

    def __init__(self):
        self.live = [job("17", "parent_token")]
        self.history = []
        self.submissions, self.cancellations = [], []
        self.submit_status = "accepted"

    def find_submission(self, token, **kwargs):
        return [r for r in self.live + self.history if r.comment == "hhq:" + token]

    def query_jobs(self):
        return self.live

    def accounting(self, ids, **kwargs):
        return [r for r in self.history if r.job_id in ids]

    def submit(self, script, token, job_name, extra_args):
        self.submissions.append((script, token, extra_args))
        self.live.append(job("19", token, "PENDING"))
        return SubmissionResult(self.submit_status, "19" if self.submit_status == "accepted" else None,
                                token, job_name, ["sbatch", *extra_args], "", "", 0, "test")

    def cancel_pending(self, ids, expected_token):
        self.cancellations.append((ids, expected_token))
        return CommandResult(["scancel", "--state=PENDING", *ids], 0)


@pytest.fixture
def planned(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "17")
    store = StateStore(tmp_path / "state")
    source = tmp_path / "commands"
    source.write_text("true\n")
    campaign = ingest_file(store, source, mode="pooled", cwd=tmp_path)
    config = Config(state_dir=str(store.root), control={"auto_renew": True},
                    profiles={"cpu": {"partition": "cpu", "memory_mib": 2048}})
    service = store.root / "service"
    guard = LeadershipGuard(service).acquire()
    guard.heartbeat("ACTIVE", deployment_token="parent_token")
    atomic_record(service / "deployment.json", {
        "deployment_id": "deployment", "hosting": "slurm", "desired_state": "RUNNING",
        "config": str(tmp_path / "config.json"), "created_at": time.time(),
        "allocations": [{"job_id": "17", "token": "parent_token", "created_at": time.time(), "role": "active"}],
    })
    client = Slurm()
    try:
        yield config, store, campaign, guard, client
    finally:
        guard.close()


def test_disabled_default_and_setting_limits():
    assert not Config().control.auto_renew
    assert Config().control.renew_lead_seconds == 600
    for control in ({"auto_renew": True, "hosting": "local"},
                    {"auto_renew": True, "standby_replicas": 1},
                    {"auto_renew": True, "walltime_seconds": 600}):
        with pytest.raises(ValueError):
            renewal.validate_settings(Config(control=control))


@pytest.mark.parametrize("status", ["accepted", "unknown"])
def test_exactly_one_preallocated_successor_even_when_acknowledgement_is_unknown(planned, status):
    config, store, _, guard, client = planned
    client.submit_status = status
    first = renewal.prepare_successor(config, guard, client=client)
    again = renewal.prepare_successor(config, guard, client=client)
    assert first == again and first["status"] == status
    assert len(client.submissions) == 1
    script, token, args = client.submissions[0]
    assert "--dependency" not in args
    assert "--role renewal" in script.read_text()
    assert token in script.read_text()
    assert len(read_record(store.root / "service/deployment.json")["allocations"]) == 2


@pytest.mark.parametrize("blocked", ["stop", "completed", "paused", "unknown", "paused_pool"])
def test_no_successor_when_stopped_settled_or_without_runnable_work(planned, blocked):
    config, store, campaign, guard, client = planned
    if blocked == "stop":
        path = store.root / "service/deployment.json"
        d = read_record(path)
        d["desired_state"] = "STOPPED"
        atomic_record(path, d)
    elif blocked == "completed":
        assert run_task(store.root, campaign["campaign_id"], campaign["task_ids"][0]) == 0
    elif blocked == "paused":
        path = store.campaign_dir(campaign["campaign_id"]) / "campaign.json"
        c = read_record(path)
        c["pause_dispatch"] = True
        atomic_json(path, c)
    elif blocked == "unknown":
        store.put_intent({"backend": "hq", "token": "uncertain", "state": "unknown"})
    else:
        atomic_json(store.root / "controls.json", {"paused_pools": ["cpu"]})
    result = renewal.prepare_successor(config, guard, client=client)
    assert result["status"] in {"stop_requested", "no_unfinished_work", "blocked_no_runnable_work", "blocked_unknown_submission"}
    assert not client.submissions


def add_worker(store, guard, client, state):
    store.put_intent({"backend": "worker", "epoch": guard.epoch, "token": "worker_token",
                      "job_id": "18", "state": "submitted", "created_at": time.time()})
    row = job("18", "worker_token", state)
    (client.history if row.terminal else client.live).append(row)


@pytest.mark.parametrize("state", ["RUNNING", "PENDING", "COMPLETED"])
def test_drain_only_cancels_pending_workers_and_requires_terminal_accounting(planned, state):
    config, store, _, guard, client = planned
    add_worker(store, guard, client, state)
    result = renewal.drain_workers(config, guard, client=client)
    assert result["settled"] is (state == "COMPLETED")
    assert client.cancellations == ([(["18"], "worker_token")] if state == "PENDING" else [])


def ready_handoff(planned):
    config, _store, _, guard, client = planned
    successor = renewal.prepare_successor(config, guard, client=client)
    workers = {"settled": True, "workers": [], "journal_flushed_at": time.time()}
    assert renewal.record_handoff(config, guard, successor, workers)
    guard.heartbeat("STOPPED")
    previous = dict(guard.record)
    client.live = [job("19", successor["allocation"]["token"])]
    client.history = [job("17", "parent_token", "COMPLETED")]
    return successor, previous


def test_takeover_requires_graceful_receipt_and_every_exact_old_allocation_terminal(planned):
    config, store, _, guard, client = planned
    add_worker(store, guard, client, "COMPLETED")
    successor, previous = ready_handoff(planned)
    client.history.append(job("18", "worker_token", "COMPLETED"))
    proof = renewal.takeover_proof(config, previous, successor["allocation"]["token"], "19", client=client)
    assert proof["quiescent"] and proof["previous_epoch"] == guard.epoch
    assert len(proof["allocations"]) == 2 and not client.cancellations
    client.live.append(job("18", "worker_token"))
    with pytest.raises(LeadershipError, match="live or terminal"):
        renewal.takeover_proof(config, previous, successor["allocation"]["token"], "19", client=client)


@pytest.mark.parametrize("failure", ["receipt", "parent_failed", "unknown_parent", "other_successor", "wrong_owner"])
def test_unplanned_or_uncertain_predecessor_never_becomes_automatic_failover(planned, failure):
    config, store, _, guard, client = planned
    successor, previous = ready_handoff(planned)
    if failure == "receipt":
        (store.root / "service" / f"renewal-{guard.epoch}.json").unlink()
    elif failure == "parent_failed":
        client.history = [job("17", "parent_token", "TIMEOUT")]
    elif failure == "unknown_parent":
        client.history = []
    elif failure == "wrong_owner":
        client.history = [job("17", "someone_else", "COMPLETED")]
    token = "wrong_token" if failure == "other_successor" else successor["allocation"]["token"]
    with pytest.raises(LeadershipError):
        renewal.takeover_proof(config, previous, token, "19", client=client)
    assert not client.cancellations


@pytest.mark.parametrize("reason", ["stop", "completed", "raw_signal"])
def test_stop_or_completion_cancels_exact_pending_successor_and_blocks_handoff(planned, reason):
    config, store, campaign, guard, client = planned
    successor = renewal.prepare_successor(config, guard, client=client)
    if reason == "stop":
        path = store.root / "service/deployment.json"
        d = read_record(path)
        d["desired_state"] = "STOPPED"
        atomic_record(path, d)
    elif reason == "completed":
        assert run_task(store.root, campaign["campaign_id"], campaign["task_ids"][0]) == 0

    class Backend:
        def jobs(self):
            return []

    assert renewal.cancel_if_unneeded(config, guard, successor, Backend(), client=client, abort=reason == "raw_signal")
    assert client.cancellations == [(["19"], successor["allocation"]["token"])]
    assert not renewal.record_handoff(config, guard, successor, {"settled": True, "journal_flushed_at": time.time()})
    with pytest.raises(renewal.RenewalNotNeeded):
        renewal.takeover_proof(config, dict(guard.record), successor["allocation"]["token"], "19", client=client)


def test_missing_flush_or_unsettled_workers_cannot_publish_handoff(planned):
    config, store, _, guard, client = planned
    successor = renewal.prepare_successor(config, guard, client=client)
    assert not renewal.record_handoff(config, guard, successor, {"settled": True})
    assert not (store.root / "service" / f"renewal-{guard.epoch}.json").exists()


def test_successor_readiness_neither_acquires_leadership_nor_allows_live_takeover(planned):
    config, store, _, guard, client = planned
    successor = renewal.prepare_successor(config, guard, client=client)
    token = successor["allocation"]["token"]
    assert not renewal.successor_ready(config, guard, successor, client=client)["ready"]
    with pytest.raises(LeadershipError, match="owned and running"):
        renewal.announce_waiting(config, guard.record, token, "19", client=client)
    client.live = [job("17", "parent_token"), job("19", token)]
    ready = renewal.announce_waiting(config, guard.record, token, "19", client=client)
    assert ready["phase"] == "waiting_without_leadership"
    assert renewal.successor_ready(config, guard, successor, client=client)["ready"]
    with pytest.raises(LeadershipError, match="holds leadership"):
        LeadershipGuard(store.root / "service").acquire()
    with pytest.raises(LeadershipError, match="graceful handoff"):
        renewal.takeover_proof(config, guard.record, token, "19", client=client)
    assert not client.cancellations


def test_stale_or_changed_successor_readiness_does_not_start_drain(planned):
    config, store, _, guard, client = planned
    successor = renewal.prepare_successor(config, guard, client=client)
    token = successor["allocation"]["token"]
    client.live = [job("17", "parent_token"), job("19", token)]
    receipt = renewal.announce_waiting(config, guard.record, token, "19", client=client)
    receipt["observed_at"] -= 100
    atomic_record(store.root / "service" / f"renewal-ready-{guard.epoch}.json", receipt)
    assert not renewal.successor_ready(config, guard, successor, client=client)["ready"]


def test_fresh_quiescent_lifetime_deferral_can_renew_early(planned):
    config, store, _, guard, _ = planned
    report = {"state": "healthy", "epoch": guard.epoch, "at": datetime.now(UTC).isoformat(),
              "plan": {"control_remaining_blocked_tasks": 2, "uncovered_tasks": 2, "actions": []}}
    path = store.root / "controller_status.json"
    atomic_record(path, report)
    assert renewal.early_renewal_due(config, guard)
    store.put_intent({"backend": "worker", "token": "still_running", "state": "running"})
    assert not renewal.early_renewal_due(config, guard)
    store.put_intent({"backend": "worker", "token": "still_running", "state": "terminal"})
    report["plan"]["uncovered_tasks"] = 3
    atomic_record(path, report)
    assert not renewal.early_renewal_due(config, guard)
    report["plan"]["uncovered_tasks"] = 2
    report["epoch"] -= 1
    atomic_record(path, report)
    assert not renewal.early_renewal_due(config, guard)
