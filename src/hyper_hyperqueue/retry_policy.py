"""Small shared retry bounds; legacy task policies keep their original meaning."""

from __future__ import annotations

import hashlib
import json
import re

INFRASTRUCTURE_REASONS = {"worker_lost", "preempted", "allocation_timeout"}


def failure_signature(result, stderr, tail_bytes=65536):
    """Hash reason/exit and terminal diagnostics, reading at most 64 KiB by default."""
    diagnostic = result.get("error", "")
    if stderr.exists():
        with stderr.open("rb") as handle:
            handle.seek(max(0, stderr.stat().st_size - tail_bytes))
            lines = handle.read().decode(errors="replace").splitlines()
        errors = [line.strip() for line in lines if re.search(r"(?:Error|Exception):", line)]
        diagnostic = errors[-1] if errors else next(
            (line.strip() for line in reversed(lines) if line.strip()), diagnostic
        )
    return hashlib.sha256(json.dumps(
        [result.get("reason"), result.get("exit_code"), diagnostic]
    ).encode()).hexdigest()


def retry_budget_error(task, attempts, state, *, infrastructure_attempt=None,
                       prestart_dispatch=None, automatic=True):
    """Return a stop reason without I/O; counters survive controller restarts."""
    policy = task.retry
    infra = set(state.get("infrastructure_attempt_ids", []))
    dispatches = set(state.get("prestart_retry_dispatches", []))
    if infrastructure_attempt is not None:
        infra.add(infrastructure_attempt)
    if prestart_dispatch is not None:
        dispatches.add(prestart_dispatch)
    if policy.max_infrastructure_retries is None:
        # Lost dispatches also consume the old single, total-attempt allowance.
        if len(attempts) + len(dispatches) >= policy.max_attempts:
            return "Retry attempt limit exhausted; import a new generation to change policy"
    else:
        if len(infra) + len(dispatches) > policy.max_infrastructure_retries:
            return "Infrastructure retry limit exhausted"
        application_attempts = sum(a["attempt_id"] not in infra for a in attempts)
        if application_attempts >= policy.max_attempts:
            return "Application attempt limit exhausted"
    if automatic and policy.same_error_limit and infrastructure_attempt is None:
        signatures = []
        for attempt in reversed(attempts):
            if attempt["attempt_id"] in infra:
                continue
            signature = attempt.get("result", {}).get("failure_signature")
            if not signature or (signatures and signature != signatures[0]):
                break
            signatures.append(signature)
        if len(signatures) >= policy.same_error_limit:
            return f"Same application error repeated {len(signatures)} times"
    return None


def unstarted_loss_proof(store, task, state, worker_tokens=None):
    """Prove an exact lost HQ dispatch and quiescence across all worker epochs."""
    from .controller import TERMINAL, owned_record, terminal_state

    item = state.get("hq_terminal_evidence", {})
    if (str(item.get("state")).lower() != "failed"
            or "worker" not in item.get("error", "").lower()
            or "lost" not in item.get("error", "").lower()
            or item.get("id") != state.get("hq_task_id")):
        raise ValueError("Missing exact HQ worker-loss evidence before attempt start")
    intents = store.intents()
    owners = [i for i in intents if i.get("token") == state.get("submission_token")
              and i.get("backend") == "hq" and i.get("campaign_id") == task.campaign_id
              and i.get("hq_job_id") == state.get("hq_job_id")
              and i.get("hq_task_ids", {}).get(task.task_id) == state.get("hq_task_id")]
    if len(owners) != 1:
        raise ValueError("Lost dispatch ownership is not reconciled")
    workers = [i for i in intents if i.get("backend") == "worker"
               and not (i.get("state") == "rejected" and i.get("response", {}).get("status") == "rejected")]
    if worker_tokens is not None:
        workers = [i for i in workers if i["token"] in worker_tokens]
        if {i["token"] for i in workers} != set(worker_tokens):
            raise ValueError("Missing worker ownership evidence from the recovery snapshot")
    if not workers or any(i.get("state") != "terminal" or not i.get("terminal_evidence")
                          or not all(owned_record(r, i) and terminal_state(r["state"]) in TERMINAL
                                     and terminal_state(r["state"]) != "CANCELLED"
                                     for r in i["terminal_evidence"]) for i in workers):
        raise ValueError("Waiting for positive termination of all possibly involved workers")
    dispatch = f"{owners[0]['token']}:{state['hq_job_id']}:{state['hq_task_id']}"
    proof = {"dispatch": dispatch, "workers": [
        {"token": i["token"], "job_id": i["job_id"], "terminal_evidence": i["terminal_evidence"]}
        for i in sorted(workers, key=lambda i: i["token"])
    ]}
    return (dispatch, hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest(),
            sorted(i["token"] for i in workers))
