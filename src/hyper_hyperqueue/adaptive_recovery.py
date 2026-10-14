"""Bounded retry resources and exact scheduler termination evidence.

Original TaskSpec files never change. A retry appends one resource decision,
then the coordinator points to that hash-bound decision for future dispatch.
"""
from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from pathlib import Path

from .models import Resources
from .state import fingerprint, immutable_json, read_json


def classify_memory_failure(stderr: Path) -> str | None:
    """Explicit terminal exception diagnostics only; exit 137 is not OOM evidence."""
    with stderr.open("rb") as handle:
        handle.seek(max(0, stderr.stat().st_size - 65536))
        text = handle.read().decode("utf-8", errors="replace")
    if re.search(r"(?m)^(?:torch\.(?:cuda\.)?OutOfMemoryError:|.*CUDA out of memory)", text):
        return "gpu_oom"  # More host RAM cannot fix a GPU VRAM shortage.
    last_line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    if re.fullmatch(r"(?:MemoryError(?:\s*:.*)?|numpy\.[\w.]*_ArrayMemoryError:.*)", last_line):
        return "host_oom"
    return None


def user_canceled(store, task) -> bool:
    raw = read_json(store.campaign_dir(task.campaign_id) / "task_states" / f"{task.task_id}.json")
    if raw.get("state", raw.get("status")) == "canceled":
        return True
    for path in (store.root / "requests").glob("cancel-*.json"):
        record = read_json(path)
        if record.get("campaign_id") == task.campaign_id and record.get("action") == "cancel_hq_campaign":
            return True
    return False


def scheduler_proof(store, task, attempt):
    """Read already reconciled exact allocation evidence; no scheduler RPC here."""
    from .controller import TERMINAL, owned_record, terminal_state
    from .runner import attempt_epoch

    if user_canceled(store, task):
        return None
    env = attempt.get("context", {}).get("environment", {})
    jid = env.get("SLURM_ARRAY_JOB_ID") or env.get("SLURM_JOB_ID")
    if jid is None:
        return None
    owners = [i for i in store.intents() if i.get("backend") in ("worker", "direct")
              and str(i.get("job_id")) == str(jid) and i.get("state") == "terminal"]
    if len(owners) != 1:
        return None
    allocation = owners[0]
    epoch = attempt_epoch(attempt)
    if allocation.get("backend") == "worker" and str(allocation.get("epoch")) != epoch:
        return None
    records = allocation.get("terminal_evidence", [])
    if env.get("SLURM_ARRAY_TASK_ID") is not None:
        element = f"{jid}_{env['SLURM_ARRAY_TASK_ID']}"
        records = [r for r in records if str(r.get("job_id", "")).split(".")[0] == element]
    if not records or not all(owned_record(r, allocation) and terminal_state(r["state"]) in TERMINAL
                              for r in records):
        return None
    states = {terminal_state(r["state"]) for r in records}
    if "CANCELLED" in states:
        return None  # Scheduler cancellation alone does not establish site preemption.
    reason = ("host_oom" if "OUT_OF_MEMORY" in states else "allocation_timeout" if "TIMEOUT" in states
              else "preempted" if "PREEMPTED" in states else "worker_lost")
    if reason == "host_oom" and allocation.get("backend") == "worker" and task.resources.gpus == 0:
        # Pooled CPU tasks can share RAM: job-level OOM does not identify a culprit.
        reason = "host_oom" if attempt.get("result", {}).get("reason") == "host_oom" else "worker_lost"
    natural_limit = False
    exit_path = store.root / "service/worker_exits" / f"{jid}.json"
    if allocation.get("backend") == "worker" and states == {"COMPLETED"} and exit_path.exists():
        record = read_json(exit_path)
        try:
            times_valid = (math.isfinite(float(record["configured_end_time"]))
                           and math.isfinite(float(record["finished"]))
                           and 0 < float(record["configured_end_time"]) <= float(record["finished"]))
        except (KeyError, TypeError, ValueError):
            times_valid = False
        natural_limit = (times_valid and record.get("termination_reason") == "time_limit"
                         and record.get("allocation_token") == allocation["token"]
                         and str(record.get("epoch")) == epoch
                         and str(record.get("slurm_job_id")) == str(jid)
                         and record.get("received_signals") == []
                         and record.get("returncode") is not None and record["returncode"] >= 0)
    return {"schema_version": 1, "kind": "scheduler_termination", "quiescent": True,
            "previous_epoch": epoch, "attempt_id": attempt["attempt_id"],
            "campaign_id": task.campaign_id, "task_id": task.task_id,
            "allocation_token": allocation["token"], "terminal_evidence": records,
            "reason": reason, "natural_worker_limit": natural_limit,
            "later_attempt_on_allocation": _later_allocation_attempt(store, attempt),
            "allocation_interruption": bool(states & {"OUT_OF_MEMORY", "TIMEOUT", "PREEMPTED", "NODE_FAIL", "BOOT_FAIL"}),
            "basis": "exact owned allocation absent from live query and positively terminal in accounting"}


def _later_allocation_attempt(store, previous):
    """A later command on the same allocation defeats retrospective attribution."""
    result = previous.get("result")
    if result is None:
        return False
    try:
        finished = datetime.fromisoformat(result["finished_at"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return False  # The attribution gate separately rejects missing timing.
    def identity(item):
        env = item.get("context", {}).get("environment", {})
        return (env.get("SLURM_ARRAY_JOB_ID") or env.get("SLURM_JOB_ID"), env.get("SLURM_ARRAY_TASK_ID"))
    target = identity(previous)
    for campaign in store.campaigns():
        for tid in store.task_ids(campaign["campaign_id"]):
            for attempt in store.attempts(campaign["campaign_id"], tid):
                if attempt["attempt_id"] == previous["attempt_id"] or identity(attempt) != target:
                    continue
                try:
                    if datetime.fromisoformat(attempt["started_at"]).timestamp() > finished:
                        return True
                except (KeyError, TypeError, ValueError):
                    continue
    return False


def allocation_caused_failure(proof, attempt):
    """Do not blame a later allocation failure for an earlier finished command."""
    result = attempt.get("result")
    if result is None:
        return True
    if proof.get("later_attempt_on_allocation") or not (proof.get("allocation_interruption") or proof.get("natural_worker_limit")):
        return False
    reason, status = result.get("reason"), result.get("status")
    ambiguous = (reason in {"user_cancellation", "interrupted", "application_signal"}
                 or (reason == "application_exit" and result.get("exit_code") in {137, 143}))
    if not ambiguous or status == "succeeded":
        return False  # Explicit GPU OOM, validation errors and task timeouts stay unchanged.
    try:
        finished = datetime.fromisoformat(result["finished_at"]).timestamp()
        ends = [float(row["end_time"]) for row in proof["terminal_evidence"] if row.get("end_time")]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(ends) and -30 <= max(ends) - finished <= 120


def verified_scheduler_proof(store, task, attempt, path):
    if path is None:
        return None
    proof = read_json(Path(path))
    if proof.get("kind") != "scheduler_termination":
        return None
    expected = scheduler_proof(store, task, attempt)
    if expected is None or any(proof.get(key) != value for key, value in expected.items()):
        return None
    if not allocation_caused_failure(proof, attempt):
        return None
    return proof


def effective_task(store, task):
    """Return execution resources without changing the immutable scientific task."""
    raw = read_json(store.campaign_dir(task.campaign_id) / "task_states" / f"{task.task_id}.json")
    binding = raw.get("resource_override")
    if not binding:
        return task
    path = Path(binding["path"])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding["sha256"]:
        raise ValueError("Retry resource override hash mismatch")
    record = read_json(path)
    if (record.get("task_fingerprint") != fingerprint(task.model_dump(mode="json"))
            or record.get("campaign_id") != task.campaign_id or record.get("task_id") != task.task_id):
        raise ValueError("Retry resource override does not bind immutable task")
    if record.get("after_attempt_id") != binding.get("after_attempt_id"):
        raise ValueError("Retry resource override attempt binding mismatch")
    if record["after_attempt_id"] not in {a["attempt_id"] for a in store.attempts(task.campaign_id, task.task_id)}:
        raise ValueError("Retry resource override refers to an unknown attempt")
    resources = Resources.model_validate(record["resources"])
    original = task.resources.model_dump()
    if any(resources.model_dump()[key] != value for key, value in original.items() if key != "memory_mib"):
        raise ValueError("Retry may change host memory only")
    allowed = {task.resources.memory_mib, *(math.ceil(task.resources.memory_mib * value)
                                          for value in task.retry.oom_memory_multipliers)}
    if resources.memory_mib not in allowed:
        raise ValueError("Retry memory is outside immutable multipliers")
    return task.model_copy(update={"resources": resources})


def prepare_resource_override(store, task, previous, reason):
    """Append an idempotent attempt decision; the caller atomically attaches it."""
    effective = effective_task(store, task)
    if reason != "host_oom" or not task.retry.oom_memory_multipliers:
        return None
    previous_ram = previous.get("resources", {}).get("memory_mib", effective.resources.memory_mib)
    choices = [math.ceil(task.resources.memory_mib * multiplier)
               for multiplier in task.retry.oom_memory_multipliers]
    memory = next((value for value in choices if value > previous_ram), None)
    if memory is None:
        raise ValueError("Host OOM memory escalation exhausted; no further automatic retry")
    resources = task.resources.model_copy(update={"memory_mib": memory})
    path = store.campaign_dir(task.campaign_id) / "resource_overrides" / task.task_id / f"{previous['attempt_id']}.json"
    record = {"schema_version": 1, "campaign_id": task.campaign_id, "task_id": task.task_id,
              "task_fingerprint": fingerprint(task.model_dump(mode="json")),
              "after_attempt_id": previous["attempt_id"], "reason": reason,
              "original_memory_mib": task.resources.memory_mib, "resources": resources.model_dump()}
    immutable_json(path, record)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "after_attempt_id": previous["attempt_id"]}


def save_scheduler_proof(store, task, attempt, proof):
    path = store.root / "recovery" / f"{attempt['attempt_id']}-{task.task_id}.json"
    immutable_json(path, proof)
    return path


def effective_config(store, config, tasks):
    """Derive future worker RAM envelopes; persisted base configuration stays intact.

    Actual per-allocation snapshots freeze the requested envelope. Old workers
    retain their original advertised memory and cannot accept an oversized retry.
    """
    from .planner import compatible

    result = config.model_copy(deep=True)
    for value in tasks:
        from .models import TaskSpec
        task = TaskSpec.model_validate(value) if isinstance(value, dict) else value
        original = store.task(task.campaign_id, task.task_id)
        effective = effective_task(store, original)
        if effective.resources.memory_mib <= original.resources.memory_mib:
            continue
        overhead = config.policy.worker_overhead_mib if task.backend == "pooled" else 0
        for name in task.allowed_profiles:
            if name not in result.profiles:
                continue
            profile = result.profiles[name]
            enlarged = profile.model_copy(update={"memory_mib": max(profile.memory_mib,
                                                       effective.resources.memory_mib + overhead)})
            if compatible(effective.resources, enlarged, config, pooled=task.backend == "pooled")[0]:
                result.profiles[name] = enlarged
    return result


def terminal_failure_valid(store, task, state):
    """A bounded, exhausted retry may settle a resultless but proven-dead attempt."""
    binding = state.get("terminal_retry_failure", {})
    previous = state.get("latest_attempt", {})
    if not binding or binding.get("attempt_id") != previous.get("attempt_id"):
        return False
    path = Path(binding["evidence_path"])
    try:
        if hashlib.sha256(path.read_bytes()).hexdigest() != binding["evidence_sha256"]:
            return False
        return verified_scheduler_proof(store, task, previous, path) is not None
    except (OSError, ValueError, KeyError):
        return False
