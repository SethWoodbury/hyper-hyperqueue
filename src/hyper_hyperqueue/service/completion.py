"""Conservative end-of-work cleanup; never a second admission or retry owner.

Callers hold StateStore.lock().  Empty runnable capacity is not completion:
paused work, uncertain submissions and undecided retries all keep the service up.
"""

from __future__ import annotations

import dataclasses
import time
from collections import Counter

from ..state import atomic_json, read_json


def _cancellation_retry_pending(store, task, attempts):
    if not attempts or len(attempts) >= task.retry.max_attempts:
        return False
    from ..runner import attempt_epoch, internal_fencing_evidence

    current = read_json(store.campaign_dir(task.campaign_id) / "task_states" / f"{task.task_id}.json")
    if current.get("state", current.get("status")) == "canceled":
        return False  # Explicit user cancellation is terminal.
    previous = attempts[-1]
    from ..adaptive_recovery import allocation_caused_failure, scheduler_proof, user_canceled

    if not user_canceled(store, task):
        env = previous.get("context", {}).get("environment", {})
        jid = env.get("SLURM_ARRAY_JOB_ID") or env.get("SLURM_JOB_ID")
        recoverable = set(task.retry.reasons) & {"worker_lost", "preempted", "allocation_timeout", "host_oom"}
        if jid is not None and recoverable:
            owners = [i for i in store.intents() if i.get("backend") in {"worker", "direct"}
                      and str(i.get("job_id")) == str(jid)]
            if len(owners) != 1 or owners[0].get("state") != "terminal":
                return True  # Wait for scheduler classification, not a raw SIGTERM label.
            proof = scheduler_proof(store, task, previous)
            if (proof and proof["reason"] in recoverable
                    and allocation_caused_failure(proof, previous)):
                return True
    epoch = attempt_epoch(previous)
    if epoch is None:
        return False
    proof = store.root / "service" / f"recovery-{epoch}.json"
    if proof.exists():
        return bool(internal_fencing_evidence(store, task, previous, proof))
    intent = store.root / "service" / f"recovery-intent-{epoch}.json"
    if not intent.exists():
        return False
    record = read_json(intent)
    job = previous.get("context", {}).get("environment", {}).get("SLURM_JOB_ID")
    return (record.get("reason") == "generation_recovery" and job is not None
            and str(record.get("previous_epoch")) == str(epoch)
            and any(str(a.get("job_id")) == str(job) for a in record.get("allocations", [])))


def work_status(store, retry_decisions=()):
    """Read all registered campaigns, including paused and direct campaigns."""
    counts, blockers, exports = Counter(), [], {}
    decided = {r["task_id"] for r in retry_decisions if r["decision"] == "retry_blocked"}
    campaigns = store.campaigns()
    for campaign in campaigns:
        cid = campaign["campaign_id"]
        for tid in store.task_ids(cid):
            current = store.task_state(cid, tid)
            state = current.get("state", "unknown")
            counts[state] += 1
            reason = None
            if state == "blocked" and current.get("reason") == "exported_to_new_generation":
                export = current.get("recovery_export")
                if export not in exports:
                    from ..recovery_export import load_export

                    try:
                        exports[export] = bool(export) and bool(load_export(export, store=store))
                    except (OSError, ValueError, KeyError, TypeError):
                        exports[export] = False
                if not exports[export]:
                    reason = "unproven_export"
            elif state not in {"succeeded", "failed", "canceled"}:
                reason = state
            elif state == "succeeded" and "completion" not in current:
                reason = "missing_validated_completion"
            elif state == "failed":
                task = store.task(cid, tid)
                attempts = store.attempts(cid, tid)
                if not attempts or "result" not in attempts[-1]:
                    from ..adaptive_recovery import terminal_failure_valid
                    if not terminal_failure_valid(store, task, current):
                        reason = "missing_final_failure"
                elif (current.get("reason") in task.retry.reasons
                      and len(attempts) < task.retry.max_attempts and tid not in decided):
                    reason = "retry_decision_pending"
            elif state == "canceled":
                task = store.task(cid, tid)
                if tid not in decided and _cancellation_retry_pending(store, task, store.attempts(cid, tid)):
                    reason = "retry_decision_pending"
            if reason:
                blockers.append({"campaign_id": cid, "task_id": tid, "reason": reason})
    for intent in store.intents():
        if intent.get("state") in {"intent", "unknown"}:
            blockers.append({"token": intent["token"], "reason": "submission_unknown"})
    if not counts:
        blockers.append({"reason": "no_registered_work"})
    return {"settled": not blockers, "counts": dict(counts), "blockers": blockers,
            "campaigns": len(campaigns)}


def _row(value):
    return value if isinstance(value, dict) else dataclasses.asdict(value)


def _owned(row, intent, user):
    return row.get("user") == user and row.get("comment") == f"hhq:{intent['token']}"


def hq_settled(store, hq, *, allow_running=False):
    """Confirm the actual server backlog, not only early runner result writes."""
    if hq is None:
        return not any(c["mode"] == "pooled" for c in store.campaigns())
    jobs = hq.jobs()
    if not isinstance(jobs, list):
        raise TypeError("Unknown HQ job-list schema during completion check")
    known = {str(i["hq_job_id"]): i for i in store.intents()
             if i.get("backend") == "hq" and i.get("hq_job_id") is not None}
    if set(known) != {str(job["id"]) for job in jobs}:
        return False
    for job in jobs:
        jid = str(job["id"])
        intent = known.get(jid)
        if intent is None or job.get("name") != f"hhq:{intent['token']}":
            return False  # Do not stop an unregistered job someone added to HQ.
        rows = hq.tasks(int(jid))
        if isinstance(rows, dict):
            rows = rows.get(jid, rows.get("tasks"))
        if not isinstance(rows, list) or len(rows) != len(intent["task_ids"]):
            return False
        for row in rows:
            state = row.get("state", row.get("status"))
            if isinstance(state, dict):
                state = state.get("state", state.get("type"))
            accepted = {"finished", "failed", "canceled", "cancelled"}
            if allow_running:
                accepted.add("running")
            if str(state).lower() not in accepted:
                return False
    return True


def cleanup_workers(store, slurm, jobs, epoch, *, reason="all_registered_work_settled"):
    """Cancel pending exact-owned workers; require terminal proof before exit.

    SlurmClient.cancel_pending rechecks ownership and filters PENDING inside the
    controller RPC.  A successful scancel is merely a request, never final proof.
    """
    blockers, requested = [], []
    live = {_row(row)["job_id"]: _row(row) for row in jobs}
    for intent in store.intents():
        if intent.get("backend") != "worker" or intent.get("state") == "rejected":
            continue
        jid = str(intent.get("job_id", ""))
        row = live.get(jid)
        if intent.get("state") == "terminal" and row is None and intent.get("terminal_evidence"):
            continue
        if intent.get("epoch") != epoch:
            blockers.append({"token": intent["token"], "reason": "other_epoch_worker_unresolved"})
            continue
        if row is not None and not _owned(row, intent, slurm.user):
            blockers.append({"token": intent["token"], "reason": "worker_ownership_unproven"})
            continue
        if row is not None and row.get("state") == "PENDING":
            request = {"epoch": epoch, "job_id": jid, "token": intent["token"],
                       "reason": reason, "requested_at": time.time()}
            folder = "completion_cancellations" if reason == "all_registered_work_settled" else "pending_drain_cancellations"
            path = store.root / "service" / folder / f"{intent['token']}.json"
            atomic_json(path, request)  # Durable intent before the scheduler RPC.
            try:
                result = slurm.cancel_pending([jid], expected_token=intent["token"])
                request["response"] = dataclasses.asdict(result)
                if result.argv:
                    requested.append(jid)
            except Exception as exc:  # noqa: BLE001 -- acknowledgement may have been lost
                request["error"] = str(exc)
            atomic_json(path, request)
        blockers.append({"token": intent["token"], "job_id": jid,
                         "reason": "awaiting_worker_termination"})
    return {"settled": not blockers, "pending_cancel_requests": requested, "blockers": blockers}


def reconcile_pending_drain(store, config, slurm, jobs, hq, epoch, retry_decisions):
    """Release surplus pending workers once all work is assigned; do not stop HQ.

    Running attempts remain active. Unknown, paused/unassigned or undecided retry
    work blocks this optional optimization, and new retry work can replenish later.
    Callers hold the shared state lock, so imports cannot race this snapshot.
    """
    if not config.policy.cancel_pending_when_assigned or epoch is None:
        return {"enabled": False}
    work = work_status(store, retry_decisions)
    blockers = [row for row in work["blockers"] if row["reason"] != "running"]
    result = {"enabled": True, "epoch": epoch, "observed_at": time.time(),
              "counts": work["counts"], "blockers": blockers, "all_work_assigned": False}
    if blockers:
        return result
    result["hq_has_no_waiting_tasks"] = hq_settled(store, hq, allow_running=True)
    if not result["hq_has_no_waiting_tasks"]:
        return result
    result["all_work_assigned"] = True
    result["workers"] = cleanup_workers(store, slurm, jobs, epoch, reason="all_registered_work_assigned")
    return result


def reconcile_completion(store, config, slurm, jobs, hq, epoch, retry_decisions):
    if not config.control.stop_when_complete or epoch is None:
        return {"enabled": False}
    result = {"enabled": True, "epoch": epoch, "observed_at": time.time(),
              "work": work_status(store, retry_decisions), "ready": False}
    if not result["work"]["settled"]:
        return result
    result["hq_settled"] = hq_settled(store, hq)
    if not result["hq_settled"]:
        return result
    result["workers"] = cleanup_workers(store, slurm, jobs, epoch)
    result["ready"] = result["workers"]["settled"]
    return result
