"""Planned Slurm control continuation after a positively quiescent predecessor.

This is not standby/failover: it never cancels a running predecessor or worker,
and an unplanned control loss never produces the required handoff receipt.
"""

from __future__ import annotations

import fcntl
import time
from datetime import datetime
from pathlib import Path

from ..backends.slurm import SlurmClient, matches_submission
from ..state import StateStore
from .completion import hq_settled, work_status
from .leadership import LeadershipError, atomic_record, read_record


class RenewalNotNeeded(LeadershipError):
    """The explicit stop/completion state wins over planned continuation."""


def validate_settings(config):
    if config.control.auto_renew and (
        config.control.hosting != "slurm" or config.control.standby_replicas
    ):
        raise ValueError("Planned auto_renew requires Slurm hosting with standby_replicas=0")
    if config.control.auto_renew and config.control.renew_lead_seconds >= config.control.walltime_seconds:
        raise ValueError("renew_lead_seconds must be shorter than the control walltime")


def _has_renewable_work(store, config):
    from ..adaptive_recovery import effective_config, effective_task
    from ..allocation_budget import budget_admission
    from ..controller import allocation_gates
    from ..planner import compatible

    controls = read_record(store.root / "controls.json", {})
    owned = [i for i in store.intents() if i.get("backend") == "worker"]
    _, reasons = allocation_gates(store, config, owned, reset_after=controls.get("startup_reset_after"))
    # A short submission cooldown is not a permanent blocker to another day.
    paused = set(controls.get("paused_pools", [])) | {
        name for name in config.profiles if any(r.startswith(f"{name}: STARTUP_CIRCUIT_OPEN") for r in reasons)
    }
    for campaign in store.campaigns():
        if campaign.get("mode") != "pooled" or campaign.get("draining"):
            continue
        cid = campaign["campaign_id"]
        for tid in store.task_ids(cid):
            state = store.task_state(cid, tid).get("state")
            if state == "running":
                return True
            if state not in {"ready", "submitted", "retry_wait"} or (
                campaign.get("pause_dispatch") and state != "submitted"
            ):
                continue
            task = effective_task(store, store.task(cid, tid))
            available_config = effective_config(store, config, [task])
            if any(name not in paused and profile.max_workers > 0
                   and (not task.allowed_profiles or name in task.allowed_profiles)
                   and compatible(task.resources, profile, available_config)[0]
                   and budget_admission(store, config, profile)["allowed"]
                   for name, profile in available_config.profiles.items()):
                return True
    return False


def early_renewal_due(config, guard):
    """Avoid an idle control tail when every waiting task needs a fresh lifetime."""
    store = StateStore(config.state_dir)
    report = read_record(store.root / "controller_status.json", {})
    try:
        age = time.time() - datetime.fromisoformat(report["at"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return False
    plan = report.get("plan", {})
    blocked = plan.get("control_remaining_blocked_tasks", 0)
    if (report.get("state") != "healthy" or report.get("epoch") != guard.epoch
            or not 0 <= age <= 2 * config.policy.reconcile_interval_seconds + 30
            or not blocked or plan.get("uncovered_tasks") != blocked or plan.get("actions")):
        return False
    return not any(i.get("backend") == "worker" and i.get("state") not in {"terminal", "rejected"}
                   for i in store.intents())


def prepare_successor(config, guard, *, client=None):
    """One waiting successor intent per epoch; unknown acknowledgement reserves it."""
    from .supervisor import _submit_control

    if not config.control.auto_renew:
        return {"status": "disabled"}
    validate_settings(config)
    client = client or SlurmClient()
    store = StateStore(config.state_dir)
    directory = store.root / "service"
    with store.lock(), (directory / "bootstrap.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        guard.assert_owned()
        deployment = read_record(directory / "deployment.json", {})
        if deployment.get("desired_state") != "RUNNING":
            return {"status": "stop_requested"}
        report = read_record(store.root / "controller_status.json", {})
        work = work_status(store, report.get("retry_decisions", []))
        if not work["counts"] or work["settled"]:
            return {"status": "no_unfinished_work"}
        if any(b.get("reason") == "submission_unknown" for b in work["blockers"]):
            return {"status": "blocked_unknown_submission"}
        if not _has_renewable_work(store, config):
            return {"status": "blocked_no_runnable_work"}
        jid = str(guard.record.get("slurm_job_id", ""))
        parents = [a for a in deployment.get("allocations", []) if str(a.get("job_id")) == jid]
        if len(parents) != 1 or not jid:
            raise LeadershipError("Renewal predecessor is not an exact declared control allocation")
        parent = parents[0]
        observed = client.find_submission(parent["token"], since=parent["created_at"])
        if not observed or any(
            str(row.job_id) != jid or row.state != "RUNNING"
            or not matches_submission(row, parent["token"], client.user) for row in observed
        ):
            raise LeadershipError("Renewal predecessor ownership/running state is unproven")
        endpoint = read_record(directory / "endpoint.json", {})
        source = {"epoch": guard.epoch, "owner_id": guard.owner_id,
                  "job_id": jid, "token": parent["token"],
                  "control_end_time": endpoint.get("control_end_time")}
        existing = [a for a in deployment["allocations"] if a.get("renewal_from") == source]
        if len(existing) > 1:
            raise LeadershipError("Multiple renewal successor intents require reconciliation")
        if existing:
            return {"status": existing[0]["status"], "allocation": existing[0]}
        _submit_control(config, directory, deployment, "renewal", client, renewal_from=source)
        return {"status": deployment["allocations"][-1]["status"],
                "allocation": deployment["allocations"][-1]}


def announce_waiting(config, previous, token, job_id, *, client=None):
    """A preallocated successor announces readiness but acquires no leadership."""
    validate_settings(config)
    if not config.control.auto_renew or not previous or not token or not job_id:
        raise LeadershipError("Waiting renewal lacks an exact predecessor/successor identity")
    client = client or SlurmClient()
    directory = Path(config.state_dir) / "service"
    with (directory / "bootstrap.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        deployment = read_record(directory / "deployment.json", {})
        if deployment.get("desired_state") != "RUNNING":
            raise RenewalNotNeeded("Stopped renewal exits without acquiring leadership")
        candidates = [a for a in deployment.get("allocations", []) if a["token"] == token]
        if len(candidates) != 1 or candidates[0].get("role") != "renewal":
            raise LeadershipError("Waiting renewal is not a declared control allocation")
        allocation = candidates[0]
        source = allocation.get("renewal_from", {})
        if (source.get("epoch") != previous["epoch"] or source.get("owner_id") != previous["owner_id"]
                or str(source.get("job_id")) != str(previous.get("slurm_job_id"))):
            raise LeadershipError("Waiting renewal predecessor changed")
        own = client.find_submission(token, since=allocation["created_at"])
        if not own or any(str(r.job_id) != str(job_id) or r.state != "RUNNING"
                          or not matches_submission(r, token, client.user) for r in own):
            raise LeadershipError("Waiting successor is not positively owned and running")
        allocation.update(job_id=str(job_id), status="accepted")
        atomic_record(directory / "deployment.json", deployment)
        receipt = {"schema": "hhq-renewal-ready/1", "predecessor": source, "successor_token": token,
                   "job_id": str(job_id), "deployment_id": deployment["deployment_id"],
                   "observed_at": time.time(), "phase": "waiting_without_leadership"}
        atomic_record(directory / f"renewal-ready-{previous['epoch']}.json", receipt)
        return receipt


def successor_ready(config, guard, successor, *, client=None):
    """The old leader drains only after its exact successor is already running."""
    guard.assert_owned()
    allocation = successor["allocation"]
    receipt = read_record(Path(config.state_dir) / "service" / f"renewal-ready-{guard.epoch}.json", {})
    if (receipt.get("schema") != "hhq-renewal-ready/1"
            or receipt.get("predecessor") != allocation.get("renewal_from")
            or receipt.get("successor_token") != allocation["token"]
            or time.time() - receipt.get("observed_at", 0) > 90):
        return {"ready": False, "reason": "waiting_for_successor_readiness"}
    client = client or SlurmClient()
    rows = client.find_submission(allocation["token"], since=allocation["created_at"])
    ready = bool(rows) and all(str(r.job_id) == receipt["job_id"] and r.state == "RUNNING"
                              and matches_submission(r, allocation["token"], client.user) for r in rows)
    return {"ready": ready, "job_id": receipt["job_id"],
            "reason": "exact_successor_running" if ready else "successor_liveness_unproven"}


def cancel_if_unneeded(config, guard, successor, backend, *, client=None, abort=False):
    """Completion/user stop wins even after a successor was queued or began waiting."""
    client = client or SlurmClient()
    store = StateStore(config.state_dir)
    directory = store.root / "service"
    with store.lock(), (directory / "bootstrap.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        guard.assert_owned()
        deployment = read_record(directory / "deployment.json", {})
        if abort:
            deployment.update(desired_state="STOPPED", renewal_aborted_at=time.time())
            atomic_record(directory / "deployment.json", deployment)
        report = read_record(store.root / "controller_status.json", {})
        explicit_stop = deployment.get("desired_state") == "STOPPED"
        settled = (not explicit_stop and work_status(store, report.get("retry_decisions", []))["settled"]
                   and hq_settled(store, backend))
        if not explicit_stop and not settled:
            return False
        if settled:
            deployment["desired_state"] = "STOPPED"
            deployment["renewal_completion_at"] = time.time()
            atomic_record(directory / "deployment.json", deployment)
        allocation = successor["allocation"]
        rows = client.find_submission(allocation["token"], since=allocation["created_at"])
        if not rows or not all(matches_submission(row, allocation["token"], client.user) for row in rows):
            raise LeadershipError("Stopped renewal successor ownership is unproven")
        pending = [row.job_id for row in rows if row.state == "PENDING"]
        if pending:
            atomic_record(directory / f"renewal-cancel-{guard.epoch}.json",
                          {"token": allocation["token"], "jobs": pending, "reason": "stop_or_completion",
                           "requested_at": time.time()})
            client.cancel_pending(pending, expected_token=allocation["token"])
        return True


def drain_workers(config, guard, *, client=None):
    """Cancel only pending exact-owned workers and observe all old workers terminal."""
    from .supervisor import _generation_allocations

    guard.assert_owned()
    client = client or SlurmClient()
    resolved = _generation_allocations(config, guard.record, client)
    if resolved is None:
        return {"settled": False, "reason": "unresolved_worker_ownership"}
    deployment, allocations = resolved
    workers = [a for a in allocations if str(a["job_id"]) != str(guard.record["slurm_job_id"])]
    live = client.query_jobs() if workers else []
    records = client.accounting([a["job_id"] for a in workers], since=deployment["created_at"]) if workers else []
    terminal, blockers = [], []
    for allocation in workers:
        jid, token = str(allocation["job_id"]), allocation["token"]
        current = [row for row in live if str(row.job_id) == jid]
        historical = [row for row in records if str(row.job_id) == jid]
        rows = current or historical
        if not rows or not all(matches_submission(row, token, client.user) for row in rows):
            blockers.append({"job_id": jid, "reason": "ownership_or_accounting_unknown"})
        elif current:
            if all(row.state == "PENDING" for row in current):
                intent = {"epoch": guard.epoch, "job_id": jid, "token": token,
                          "reason": "planned_control_renewal", "requested_at": time.time()}
                path = Path(config.state_dir) / "service" / "renewal_cancellations" / f"{token}.json"
                atomic_record(path, intent)
                client.cancel_pending([jid], expected_token=token)
            blockers.append({"job_id": jid, "reason": "awaiting_worker_termination"})
        elif not all(row.terminal for row in historical):
            blockers.append({"job_id": jid, "reason": "worker_not_terminal"})
        else:
            terminal.append({"job_id": jid, "token": token, "accounting": [r.to_dict() for r in historical]})
    return {"settled": not blockers, "workers": terminal, "blockers": blockers, "observed_at": time.time()}


def record_handoff(config, guard, successor, workers):
    """Called only after both HQ/controller children have positively exited."""
    directory = Path(config.state_dir) / "service"
    with (directory / "bootstrap.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        guard.assert_owned()
        deployment = read_record(directory / "deployment.json", {})
        if deployment.get("desired_state") != "RUNNING":
            return False
        allocation = successor["allocation"]
        if (not workers.get("settled") or not workers.get("journal_flushed_at")
                or allocation.get("status") not in {"accepted", "unknown", "submitting"}):
            return False
        receipt = {"schema": "hhq-planned-renewal/1", "phase": "children_stopped",
                   "predecessor": allocation["renewal_from"], "successor_token": allocation["token"],
                   "deployment_id": deployment["deployment_id"], "workers": workers,
                   "children_stopped_at": time.time()}
        atomic_record(directory / f"renewal-{guard.epoch}.json", receipt)
        return True


def takeover_proof(config, previous, token, job_id, *, client=None):
    """Read-only predecessor proof; no heartbeat takeover or live-job fencing."""
    from .supervisor import _terminal_generation_proof

    validate_settings(config)
    if not config.control.auto_renew or not previous or not token or not job_id:
        raise LeadershipError("Planned renewal needs an enabled, exact predecessor/successor identity")
    store = StateStore(config.state_dir)
    directory = store.root / "service"
    client = client or SlurmClient()
    with store.lock(), (directory / "bootstrap.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        deployment = read_record(directory / "deployment.json", {})
        if deployment.get("desired_state") != "RUNNING":
            raise RenewalNotNeeded("Renewal canceled by explicit service stop/completion")
        candidates = [a for a in deployment.get("allocations", []) if a["token"] == token]
        if len(candidates) != 1 or candidates[0].get("role") != "renewal":
            raise LeadershipError("Renewal allocation is not declared in this deployment")
        allocation = candidates[0]
        source = allocation.get("renewal_from", {})
        receipt = read_record(directory / f"renewal-{previous['epoch']}.json", {})
        if (source.get("epoch") != previous["epoch"] or source.get("owner_id") != previous["owner_id"]
                or str(source.get("job_id")) != str(previous.get("slurm_job_id"))
                or receipt.get("schema") != "hhq-planned-renewal/1"
                or receipt.get("phase") != "children_stopped" or receipt.get("predecessor") != source
                or receipt.get("successor_token") != token
                or receipt.get("deployment_id") != deployment.get("deployment_id")
                or not receipt.get("workers", {}).get("settled") or previous.get("state") != "STOPPED"):
            raise LeadershipError("Missing exact graceful handoff; unplanned failure cannot renew automatically")
        own_rows = client.find_submission(token, since=allocation["created_at"])
        if not own_rows or any(
            str(row.job_id) != str(job_id) or row.state != "RUNNING"
            or not matches_submission(row, token, client.user) for row in own_rows
        ):
            raise LeadershipError("Renewal successor allocation ownership is unproven")
        proof = _terminal_generation_proof(config, previous, client=client)
        parent = [a for a in proof["allocations"] if str(a["job_id"]) == str(source["job_id"])]
        if len(parent) != 1 or any(r["state"] != "COMPLETED" for r in parent[0]["accounting"]):
            raise LeadershipError("Planned predecessor did not complete normally; inspect before explicit restart")
        allocation.update(job_id=str(job_id), status="accepted")  # Adopt an exact lost acknowledgement.
        report = read_record(store.root / "controller_status.json", {})
        if work_status(store, report.get("retry_decisions", []))["settled"]:
            deployment["desired_state"] = "STOPPED"
            atomic_record(directory / "deployment.json", deployment)
            raise RenewalNotNeeded("All registered work settled during the planned drain")
        atomic_record(directory / "deployment.json", deployment)
        proof["basis"] = "planned graceful handoff plus exact terminal accounting; no live fencing"
        atomic_record(directory / f"renewal-proof-{previous['epoch']}.json", proof)
        return proof
