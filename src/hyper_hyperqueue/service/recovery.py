"""Recovery decisions require positive quiescence, never an absent heartbeat."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .leadership import same_process

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
}


@dataclass
class RecoveryDecision:
    ready: bool
    reasons: list[str] = field(default_factory=list)
    proof: dict | None = None


def fence_slurm_generation(
    previous: dict,
    allocations: list[dict],
    *,
    query: Callable,
    cancel: Callable,
    readiness: dict,
    deadline_seconds=30,
) -> RecoveryDecision:
    """Query/cancel callbacks must positively bind job ID AND stored intent.

    query returns {job_id, owned, live, accounting_state}. A missing/unknown
    accounting record is not terminal proof. Only the exact declared allocations
    may be canceled. Caller must subsequently acquire the lifetime lock.
    """
    if not readiness:
        return RecoveryDecision(False, ["cross_node_readiness_missing"])
    if not previous.get("slurm_job_id"):
        return RecoveryDecision(False, ["previous_leader_not_slurm_hosted"])
    if not any(str(a["job_id"]) == str(previous["slurm_job_id"]) for a in allocations):
        return RecoveryDecision(False, ["previous_control_allocation_not_bound"])
    snapshots = []
    try:
        for allocation in allocations:
            observed = query(allocation)
            if observed.get("owned") is not True or str(observed.get("job_id")) != str(allocation["job_id"]):
                return RecoveryDecision(False, ["allocation_ownership_unproven"])
            if observed.get("live"):
                cancel(allocation)
        deadline = time.monotonic() + deadline_seconds
        while True:
            snapshots = [query(a) for a in allocations]
            if any(s.get("owned") is not True for s in snapshots):
                return RecoveryDecision(False, ["ownership_changed_during_fence"])
            if all(s.get("live") is False and s.get("accounting_state") in TERMINAL for s in snapshots):
                proof = {
                    "schema": "hhq-fencing-proof/1",
                    "previous_owner": previous["owner_id"],
                    "previous_epoch": previous["epoch"],
                    "quiescent": True,
                    "allocations": snapshots,
                    "verified_at": time.time(),
                    "readiness": readiness,
                }
                return RecoveryDecision(True, ["exact_owned_generation_terminal"], proof)
            if time.monotonic() >= deadline:
                return RecoveryDecision(False, ["terminal_accounting_or_quiescence_unavailable"])
            time.sleep(min(1, max(0, deadline - time.monotonic())))
    except (OSError, RuntimeError, TimeoutError) as exc:
        return RecoveryDecision(False, ["fencing_backend_unavailable", str(exc)])


def local_quiescence(previous: dict, *, host: str, children: list[dict]) -> RecoveryDecision:
    if previous.get("host") != host or previous.get("slurm_job_id"):
        return RecoveryDecision(False, ["local_proof_not_applicable_to_remote_or_slurm_owner"])
    try:
        if same_process(previous["process"]) or any(same_process(p) for p in children):
            return RecoveryDecision(False, ["old_local_process_still_alive"])
    except (OSError, KeyError):
        return RecoveryDecision(False, ["local_process_identity_unavailable"])
    return RecoveryDecision(
        True,
        ["local_recorded_processes_quiescent"],
        {
            "previous_owner": previous["owner_id"],
            "previous_epoch": previous["epoch"],
            "quiescent": True,
            "verified_at": time.time(),
        },
    )


def replay_decision(task, latest_attempt: dict | None, proof: dict | None) -> str:
    result = (latest_attempt or {}).get("result") or {}
    if result.get("status") == "succeeded":
        return "reconcile_completion"
    if result.get("reason") == "canceled" or result.get("status") == "canceled":
        return "keep_canceled"
    if not proof or proof.get("quiescent") is not True:
        return "blocked_unproven_quiescence"
    if not task.output.repeatable or task.output.mode == "manual_on_ambiguity":
        return "manual_on_ambiguity"
    if latest_attempt and latest_attempt.get("attempt_number", 1) >= task.retry.max_attempts:
        return "retry_exhausted"
    return "eligible_for_bounded_retry"
