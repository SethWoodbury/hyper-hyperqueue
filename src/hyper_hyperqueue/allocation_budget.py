"""Optional cumulative allocation GPU-hour ceiling, with conservative reservations.

Uses cached owned intents/accounting only. This is neither site billing nor a
walltime predictor; pending/unknown/running allocations reserve their full limit.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from .planner import gpu_count


def _argument(intent, flag):
    args = intent.get("extra_args", [])
    for index, value in enumerate(args):
        if value == flag and index + 1 < len(args):
            return args[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return None


def _seconds(text):
    if text is None:
        raise ValueError("Allocation has no frozen walltime request")
    days, rest = text.split("-", 1) if "-" in text else ("0", text)
    fields = [float(value) for value in rest.split(":")]
    if len(fields) == 3:
        hours, minutes, seconds = fields
    elif len(fields) == 2:
        hours, minutes, seconds = 0, *fields
    elif len(fields) == 1:
        hours, minutes, seconds = 0, fields[0], 0
    else:
        raise ValueError("Unrecognized frozen allocation walltime")
    return float(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def _finite_nonnegative(value):
    try:
        return math.isfinite(float(value)) and float(value) >= 0
    except (ValueError, TypeError):
        return False


def allocation_budget_status(store, config):
    baseline = config.policy.allocation_gpu_hours_budget
    ceiling = None if baseline is None else baseline * (1 + config.policy.allocation_budget_allowance)
    spent, reserved, uncertain, seen = 0.0, 0.0, [], set()
    for intent in store.intents():
        if intent.get("backend") not in {"worker", "direct"} or intent.get("state") == "rejected":
            continue
        key = ("job", str(intent["job_id"])) if intent.get("job_id") else ("token", intent["token"])
        if key in seen:
            continue
        seen.add(key)
        gres = _argument(intent, "--gres")
        count = gpu_count(SimpleNamespace(gres=gres))
        if count == 0:
            profile = config.profiles.get(intent.get("profile"))
            if gres is None and profile is not None and gpu_count(profile):
                uncertain.append(intent["token"])
            continue
        try:
            request = _seconds(_argument(intent, "--time")) * count * int(intent.get("elements", 1)) / 3600
            if not math.isfinite(request) or request < 0:
                raise ValueError("Invalid frozen allocation time request")
        except (ValueError, TypeError):
            uncertain.append(intent["token"])
            continue
        records = intent.get("terminal_evidence", []) if intent.get("state") == "terminal" else []
        rows = {str(row.get("job_id")): row for row in records if "." not in str(row.get("job_id", ""))}
        children = {key: row for key, row in rows.items() if "_" in key}
        if children:
            rows = children
        if (rows and len(rows) >= int(intent.get("elements", 1))
                and all(_finite_nonnegative(row.get("elapsed_seconds"))
                        and (row.get("gpus") is None or _finite_nonnegative(row["gpus"]))
                        for row in rows.values())):
            spent += sum(float(row["elapsed_seconds"]) * (count if row.get("gpus") is None else float(row["gpus"]))
                         / 3600 for row in rows.values())
        else:
            reserved += request
    return {"enabled": baseline is not None, "baseline_gpu_hours": baseline,
            "allowance_fraction": config.policy.allocation_budget_allowance,
            "ceiling_gpu_hours": ceiling, "recorded_allocation_gpu_hours": spent,
            "reserved_gpu_hours": reserved, "unknown_requests": uncertain,
            "remaining_gpu_hours": None if ceiling is None else max(0, ceiling - spent - reserved),
            "basis": "owned allocations; terminal recorded elapsed; all unresolved allocations reserve full requested time"}


def budget_admission(store, config, profile, *, elements=1):
    """Caller holds the existing submission lock; reserve by writing its intent next."""
    report = allocation_budget_status(store, config)
    requested = gpu_count(profile) * profile.allocation_seconds * elements / 3600
    allowed = (not report["enabled"] or (not report["unknown_requests"]
               and requested <= report["remaining_gpu_hours"] + 1e-9))
    if not math.isfinite(requested):
        allowed = False
    return {**report, "allowed": allowed, "proposed_gpu_hours": requested,
            "reason": None if allowed else "ALLOCATION_GPU_HOUR_BUDGET_EXHAUSTED_OR_UNKNOWN"}
