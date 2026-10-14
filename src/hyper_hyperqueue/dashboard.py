"""Cached, read-only views of durable task evidence; never contact a scheduler.

This intentionally does not instantiate StateStore: its constructor and task-lock
probes may write metadata. Missing-final attempts remain unproven activity.
"""

from __future__ import annotations

import copy
import datetime as dt
import heapq
import itertools
import json
import math
import os
import stat
import statistics
import threading
import time
from collections import Counter
from pathlib import Path

from .estimates import quantile
from .models import TaskSpec
from .state import fingerprint, identifier

_TERMINAL = {"succeeded", "failed", "canceled"}
_NOTES = [
    "Runtime is measured successful task time, not allocated GPU time or GPU utilization; no ETA is inferred.",
    "Host RAM is sampled process-tree peak RSS, not a cgroup peak. Shared pages may be counted twice and short peaks missed.",
    "Requested RAM is the attempt's effective host-RAM request: the task's declared value, or a recorded retry override.",
    "GPU VRAM use and GPU utilization are not measured. Device model identifies placement only.",
    "Unfinished attempts have no result yet. They are neither running nor failed in this view; no liveness or scheduler probe is made and age is not proof of activity.",
    "Counts are a cached, nontransactional view of durable records; collection/retry decisions remain in the existing controller.",
]


UNFINISHED_REASON = "unfinished_attempt_liveness_not_probed"


class DashboardUnavailable(RuntimeError):
    """No snapshot has been read yet and the latest scan failed."""


def open_regular(path):
    """Open for binary reading, refusing FIFOs/devices without blocking on them."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("not_a_regular_file")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def read_json_object(path, maximum=4 * 1024 * 1024):
    with open_regular(path) as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("record_too_large")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise TypeError("record_is_not_object")
    return value


def gpu_label(devices, gpus_requested):
    """Name every recorded device model; anything incomplete stays unknown."""
    if not devices:
        return "unknown" if gpus_requested else "cpu"
    models = [device.get("model") for device in devices]
    if any(not isinstance(model, str) or not model.strip() for model in models):
        return "unknown"
    return ", ".join(
        f"{count}× {model}" if count > 1 else model for model, count in sorted(Counter(models).items())
    )


def _number(value):
    return (
        value
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        else None
    )


def _validated(record):
    value = record.get("validation")
    return isinstance(value, dict) and value.get("valid") is True


def _timestamp(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        try:
            parsed = dt.datetime.fromisoformat(value)
            if parsed.tzinfo is not None:
                return parsed.timestamp()
        except ValueError:
            pass
    return None


def _timing(values):
    return {
        "samples": len(values),
        "mean_seconds": statistics.mean(values) if values else None,
        "median_seconds": statistics.median(values) if values else None,
        "p90_seconds": quantile(values, 0.9) if values else None,
        "measurement_scope": "successful validated attempts with recorded wall_seconds",
    }


def _memory(values):
    peaks = [peak for peak, _ in values]
    requested = [request for _, request in values if request is not None]
    ratios = [peak / request for peak, request in values if request is not None and request > 0]
    return {
        "samples": len(peaks),
        "mean_peak_mib": statistics.mean(peaks) if peaks else None,
        "max_peak_mib": max(peaks) if peaks else None,
        "mean_requested_mib": statistics.mean(requested) if requested else None,
        "requested_samples": len(requested),
        "mean_peak_to_requested_ratio": statistics.mean(ratios) if ratios else None,
        "measurement_scope": "all measured attempts, including failures; sampled host RAM only",
        "request_scope": "effective attempt request: declared task RAM or recorded retry override",
        "vram_usage_measured": False,
    }


def _progress(counts, total, observed, unfinished=0):
    completed = sum(counts.get(state, 0) for state in _TERMINAL)
    return {
        # Presentation bucket only: these tasks keep backend state "unknown".
        "unfinished": unfinished,
        "unknown_other": counts.get("unknown", 0) - unfinished,
        "total": total,
        "observed_tasks": observed,
        "unscanned_tasks": total - observed,
        "counts": dict(counts),
        "completed": completed,
        "succeeded": counts.get("succeeded", 0),
        "failed": counts.get("failed", 0),
        "canceled": counts.get("canceled", 0),
        "active": counts.get("running", 0),
        "remaining": total - completed,
        "completion_fraction": completed / total if total else None,
        "progress_note": "Terminal task records, including failed/canceled; not scientific pass rate or retry finality",
    }


def resolve_task_state(
    coordinator, latest, latest_result, completion, task_fingerprint, *, attempts_truncated=False
):
    """Resolve durable precedence without probing locks, processes or a scheduler."""
    raw_state = coordinator.get("state", coordinator.get("status", "unknown"))
    current = raw_state if isinstance(raw_state, str) else "unknown"
    reason = coordinator.get("reason")
    extra = {}
    if latest is not None:
        extra["latest_attempt"] = {
            **latest,
            **({"result": latest_result} if latest_result is not None else {}),
        }
    if completion is not None:
        if completion.get("task_fingerprint") != task_fingerprint or not _validated(completion):
            raise ValueError("Invalid completion identity/validation")
        current, reason = "succeeded", "validated_completion"
        extra["completion"] = completion
    elif current in {"canceled", "blocked"}:
        pass
    elif attempts_truncated:
        current, reason = "unknown", "attempt_scan_truncated"
    elif latest is not None:
        if coordinator.get("after_attempt_id") == latest.get("attempt_id"):
            pass
        elif latest_result is not None:
            current = latest_result.get("status", "unknown")
            reason = latest_result.get("reason")
            if current == "succeeded" and not _validated(latest_result):
                current, reason = "unknown", "success_without_validation"
        else:
            current, reason = "unknown", UNFINISHED_REASON
    elif current == "succeeded":
        current, reason = "unknown", "success_without_validated_record"
    if not isinstance(current, str):
        current, reason = "unknown", "invalid_state_record"
    return {
        **coordinator,
        **extra,
        "state": current,
        "status": current,
        "reason": reason,
        "liveness": "not_probed",
    }


def read_task_state(root, campaign_id, task_id) -> dict:
    """Read one task for a detail page without creating a store or touching locks."""
    directory = Path(root).expanduser() / "campaigns" / identifier(campaign_id)
    task_id = identifier(task_id)

    def read(path):
        try:
            return read_json_object(path)
        except TypeError as error:
            raise ValueError("Detail record is not a JSON object") from error

    spec = TaskSpec.model_validate(read(directory / "tasks" / f"{task_id}.json"))
    if spec.campaign_id != campaign_id or spec.task_id != task_id:
        raise ValueError("Task identity mismatch")
    task_fp = fingerprint(spec.model_dump(mode="json"))
    coordinator = read(directory / "task_states" / f"{task_id}.json")
    paths = sorted(itertools.islice((directory / "attempts" / task_id).glob("*/attempt.json"), 101))
    latest = read(paths[-1]) if paths else None
    if latest is not None and (
        latest.get("task_fingerprint") != task_fp
        or latest.get("attempt_id") != paths[-1].parent.name
        or latest.get("task_id") != task_id
        or latest.get("campaign_id") != campaign_id
    ):
        raise ValueError("Attempt identity mismatch")
    result_path = paths[-1].parent / "result.json" if paths else None
    result = read(result_path) if result_path is not None and result_path.exists() else None
    if result is not None and (
        result.get("task_fingerprint") != task_fp or result.get("attempt_id") != latest.get("attempt_id")
    ):
        raise ValueError("Result identity mismatch")
    completion_path = directory / "completions" / f"{task_id}.json"
    completion = read(completion_path) if completion_path.exists() else None
    return resolve_task_state(
        coordinator, latest, result, completion, task_fp, attempts_truncated=len(paths) > 100
    )


class DashboardSnapshot:
    """One instance per server, shared across browser polls; scans at most every TTL.

    The optional limits bound read work and response size. Truncated scans are
    explicitly labeled and unscanned task states remain unknown.
    """

    def __init__(
        self,
        config,
        *,
        ttl_seconds=10,
        max_tasks=50000,
        max_attempts=100000,
        max_campaigns=500,
        max_intents=10000,
        max_attempts_per_task=100,
        max_active=100,
        max_groups=100,
        max_recent=50,
        error_backoff_seconds=5,
        clock=None,
    ):
        self.config = config
        self.root = Path(config.state_dir).expanduser()
        self.ttl_seconds = float(ttl_seconds)
        if not math.isfinite(self.ttl_seconds) or self.ttl_seconds <= 0:
            raise ValueError("Dashboard cache TTL must be positive")
        self.limits = {
            "tasks": max_tasks,
            "attempts": max_attempts,
            "campaigns": max_campaigns,
            "intents": max_intents,
            "attempts_per_task": max_attempts_per_task,
            "active_tasks": max_active,
            "groups": max_groups,
            "recent_runs": max_recent,
        }
        if any(type(value) is not int or value < 1 for value in self.limits.values()):
            raise ValueError("Dashboard limits must be positive integers")
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._cached = None
        self._cached_at = None
        self._effective_ttl = max(10.0, self.ttl_seconds)
        self._scan_seconds = None
        self.error_backoff_seconds = float(error_backoff_seconds)
        self._error = None
        self._retry_at = None

    def get(self, force=False) -> dict:
        """Return cached data; a failed scan keeps the last data and backs off briefly."""
        with self._lock:
            now = self._clock()
            due = self._cached is None or force or not 0 <= now - self._cached_at < self._effective_ttl
            if due and not force and self._retry_at is not None and 0 <= self._retry_at - now:
                due = False  # Recent failure: do not rescan on every browser poll.
            refreshed = False
            if due:
                try:
                    scanned = self._scan(now)
                except Exception as error:  # noqa: BLE001 -- surfaced to the viewer, never fatal
                    elapsed = max(0, self._clock() - now)
                    self._error = {
                        "kind": type(error).__name__,
                        "message": str(error)[:300],
                        "at": dt.datetime.fromtimestamp(now, dt.UTC).isoformat(),
                    }
                    self._retry_at = self._clock() + max(self.error_backoff_seconds, 2 * elapsed)
                else:
                    self._cached, self._cached_at = scanned, self._clock()
                    self._scan_seconds = max(0, self._cached_at - now)
                    self._effective_ttl = max(10.0, self.ttl_seconds, 5 * self._scan_seconds)
                    self._error = self._retry_at = None
                    refreshed = True
            if self._cached is None:
                raise DashboardUnavailable(
                    "Workspace records could not be read: " + (self._error or {}).get("kind", "unknown")
                )
            result = copy.deepcopy(self._cached)
            result["cache"] = {
                "ttl_seconds": self._effective_ttl,
                "configured_ttl_seconds": self.ttl_seconds,
                "scan_seconds": self._scan_seconds,
                "age_seconds": max(0, self._clock() - self._cached_at),
                "refreshed": refreshed,
                "error": copy.deepcopy(self._error),
                "retry_after_seconds": max(0, self._retry_at - self._clock()) if self._retry_at else None,
            }
            return result

    def _scan(self, now):
        diagnostics = {
            "partial_records": 0,
            "errors": [],
            "truncation": {},
            "scan_limits": dict(self.limits),
            "notes": list(_NOTES),
        }

        def issue(path, kind):
            diagnostics["partial_records"] += 1
            if len(diagnostics["errors"]) < 20:
                path = Path(path)
                shown = path.relative_to(self.root) if path.is_relative_to(self.root) else path
                diagnostics["errors"].append({"path": str(shown), "kind": kind})

        def read(path, *, required=False, maximum=4 * 1024 * 1024):
            try:
                return read_json_object(path, maximum)
            except FileNotFoundError:
                if required:
                    issue(path, "missing_record")
            except (OSError, ValueError, TypeError, UnicodeError) as error:
                issue(path, type(error).__name__)
            return None

        def paths(pattern, limit, label):
            try:
                values = list(itertools.islice(self.root.glob(pattern), limit + 1))
            except OSError:
                issue(self.root, f"unavailable_{label}")
                return []
            if len(values) > limit:
                diagnostics["truncation"][label] = True
            return sorted(values[:limit])

        def mapping(value, path, field):
            if value is None:
                return {}
            if not isinstance(value, dict):
                issue(path, f"invalid_{field}")
                return {}
            return value

        controller = read(self.root / "controller_status.json")
        allocation_plan = mapping((controller or {}).get("plan"), "controller_status.json", "plan")
        pending_reserves = mapping(
            allocation_plan.get("pending_reserves"), "controller_status.json", "pending_reserves"
        )
        leader = read(self.root / "service/leader.json")
        stale_after = max(120, 3 * self.config.policy.reconcile_interval_seconds)

        def health(record, field):
            stamp = _timestamp((record or {}).get(field))
            age = max(0, now - stamp) if stamp is not None else None
            details = {
                key: record[key]
                for key in ("epoch", "consecutive_errors", "slurm_job_id")
                if record and key in record
            }
            if record and record.get("error"):
                details["error"] = str(record["error"])[:300]
            return {
                "status": (record or {}).get("state", "not_recorded"),
                "age_seconds": age,
                "recorded_at": (record or {}).get(field),
                "stale": age > stale_after if age is not None else None,
                "stale_after_seconds": stale_after,
                "details": details,
            }

        allocation_counts, by_profile, hq_counts = Counter(), {}, Counter()
        for path in paths("intents/*.json", self.limits["intents"], "intents"):
            record = read(path, required=True)
            if record is None:
                continue
            state = record.get("state", "unknown")
            if not isinstance(state, str):
                issue(path, "invalid_intent_state")
                state = "unknown"
            if record.get("backend") in {"worker", "direct"}:
                allocation_counts[state] += 1
                profile = str(record.get("profile", "unknown"))
                by_profile.setdefault(profile, Counter())[state] += 1
            elif record.get("backend") == "hq":
                hq_counts[state] += 1

        campaigns, global_counts, group_data = [], Counter(), {}
        times, memory, recent, active = [], [], [], []
        totals = {"tasks": 0, "observed_tasks": 0, "attempts": 0, "unfinished": 0}
        retries = {
            "tasks_retried": 0,
            "additional_attempts": 0,
            "failed_attempts": 0,
            "canceled_attempts": 0,
            "unfinished_attempts": 0,
        }
        outcomes = Counter()
        recent_index = active_count = 0
        campaign_paths = paths("campaigns/*/campaign.json", self.limits["campaigns"], "campaigns")
        for campaign_path in campaign_paths:
            campaign = read(campaign_path, required=True, maximum=32 * 1024 * 1024)
            if campaign is None:
                continue
            directory = campaign_path.parent
            cid, task_ids = campaign.get("campaign_id"), campaign.get("task_ids")
            if cid != directory.name or not isinstance(task_ids, list):
                issue(campaign_path, "invalid_campaign_identity")
                continue
            counts, observed, unfinished = Counter(), 0, 0
            totals["tasks"] += len(task_ids)
            for tid in task_ids:
                if totals["observed_tasks"] >= self.limits["tasks"]:
                    diagnostics["truncation"]["tasks"] = True
                    break
                observed += 1
                totals["observed_tasks"] += 1
                try:
                    identifier(tid)
                except (TypeError, ValueError):
                    issue(campaign_path, "invalid_task_identifier")
                    counts["unknown"] += 1
                    continue
                task_path = directory / "tasks" / f"{tid}.json"
                task = read(task_path, required=True)
                coordinator = read(directory / "task_states" / f"{tid}.json", required=True)
                if task is None or coordinator is None:
                    counts["unknown"] += 1
                    continue
                try:
                    spec = TaskSpec.model_validate(task)
                    if spec.task_id != tid or spec.campaign_id != cid:
                        raise ValueError("task_identity_mismatch")
                    task_fp = fingerprint(spec.model_dump(mode="json"))
                except ValueError:
                    issue(task_path, "invalid_task_identity")
                    counts["unknown"] += 1
                    continue
                attempt_paths = list(
                    itertools.islice(
                        (directory / "attempts" / tid).glob("*/attempt.json"),
                        self.limits["attempts_per_task"] + 1,
                    )
                )
                attempt_paths.sort()
                attempts_truncated = len(attempt_paths) > self.limits["attempts_per_task"]
                if attempts_truncated:
                    diagnostics["truncation"]["attempts_per_task"] = True
                    attempt_paths = attempt_paths[-self.limits["attempts_per_task"] :]
                retries["tasks_retried"] += int(len(attempt_paths) > 1)
                retries["additional_attempts"] += max(0, len(attempt_paths) - 1)
                latest, latest_result = None, None
                for attempt_path in attempt_paths:
                    if totals["attempts"] >= self.limits["attempts"]:
                        diagnostics["truncation"]["attempts"] = True
                        attempts_truncated = True
                        break
                    totals["attempts"] += 1
                    attempt = read(attempt_path, required=True)
                    if attempt is None:
                        attempts_truncated = True
                        continue
                    if (
                        attempt.get("task_fingerprint") != task_fp
                        or attempt.get("attempt_id") != attempt_path.parent.name
                        or attempt.get("task_id") != tid
                        or attempt.get("campaign_id") != cid
                    ):
                        issue(attempt_path, "invalid_attempt_identity")
                        attempts_truncated = True
                        continue
                    result_path = attempt_path.parent / "result.json"
                    result = read(result_path)
                    if result is not None and (
                        result.get("task_fingerprint") != task_fp
                        or result.get("attempt_id") != attempt.get("attempt_id")
                    ):
                        issue(result_path, "invalid_result_identity")
                        result = None
                    latest, latest_result = attempt, result
                    outcome = (result or {}).get("status", "unfinished")
                    if not isinstance(outcome, str):
                        issue(result_path, "invalid_result_status")
                        outcome = "unknown"
                    outcomes[outcome] += 1
                    retries["failed_attempts"] += int(outcome == "failed")
                    retries["canceled_attempts"] += int(outcome == "canceled")
                    retries["unfinished_attempts"] += int(result is None)
                    if result is None:
                        continue
                    execution = mapping(result.get("execution", result), result_path, "execution")
                    telemetry = mapping(
                        execution.get("telemetry", result.get("telemetry")), result_path, "telemetry"
                    )
                    context = mapping(attempt.get("context"), attempt_path, "context")
                    gpu_record = mapping(
                        result.get("assigned_gpu", context.get("assigned_gpu")), result_path, "assigned_gpu"
                    )
                    devices = gpu_record.get("devices", [])
                    if not isinstance(devices, list) or any(
                        not isinstance(device, dict) for device in devices
                    ):
                        issue(result_path, "invalid_gpu_devices")
                        devices = []
                    gpu = gpu_label(devices, spec.resources.gpus)
                    group_key = (
                        spec.workload,
                        gpu,
                        spec.software_fingerprint or "unknown",
                        spec.resources.cpus,
                    )
                    group = group_data.setdefault(
                        group_key, {"times": [], "memory": [], "outcomes": Counter()}
                    )
                    group["outcomes"][outcome] += 1
                    peak_bytes = _number(telemetry.get("sampled_peak_rss_bytes"))
                    peak = peak_bytes / 1048576 if peak_bytes is not None else None
                    declared = spec.resources.memory_mib
                    attempt_resources = attempt.get("resources")
                    effective = _number(
                        attempt_resources.get("memory_mib") if isinstance(attempt_resources, dict) else None
                    )
                    requested = effective if effective is not None else declared
                    if peak is not None:
                        memory.append((peak, requested))
                        group["memory"].append((peak, requested))
                    duration = _number(telemetry.get("wall_seconds"))
                    if outcome == "succeeded" and _validated(result):
                        if duration is not None:
                            times.append(duration)
                            group["times"].append(duration)
                        row = {
                            "campaign_id": cid,
                            "task_id": tid,
                            "attempt_id": attempt["attempt_id"],
                            "finished_at": result.get("finished_at"),
                            "duration_seconds": duration,
                            "peak_rss_mib": peak,
                            "requested_mib": requested,
                            "declared_mib": declared,
                            "ram_override": requested != declared,
                            "workload": spec.workload,
                            "gpu_model": gpu,
                        }
                        finished = _timestamp(row["finished_at"])
                        if finished is not None:
                            heapq.heappush(recent, (finished, recent_index, row))
                            recent_index += 1
                            if len(recent) > self.limits["recent_runs"]:
                                heapq.heappop(recent)
                completion_path = directory / "completions" / f"{tid}.json"
                completion = read(completion_path)
                raw_state = coordinator.get("state", coordinator.get("status", "unknown"))
                try:
                    resolved = resolve_task_state(
                        coordinator,
                        latest,
                        latest_result,
                        completion,
                        task_fp,
                        attempts_truncated=attempts_truncated,
                    )
                    current, reason = resolved["state"], resolved["reason"]
                except (ValueError, AttributeError):
                    issue(completion_path, "invalid_completion_identity")
                    current, reason = "unknown", "invalid_completion"
                counts[current] += 1
                unfinished_task = current == "unknown" and reason == UNFINISHED_REASON
                unfinished += unfinished_task
                if current == "running" or (
                    latest is not None
                    and latest_result is None
                    and current not in _TERMINAL | {"blocked", "retry_wait"}
                ):
                    active_count += 1
                    if len(active) < self.limits["active_tasks"]:
                        started = _timestamp((latest or {}).get("started_at"))
                        active.append(
                            {
                                "campaign_id": cid,
                                "task_id": tid,
                                "state": current,
                                "display_state": "unfinished" if unfinished_task else current,
                                "last_coordinator_state": raw_state,
                                "reason": reason,
                                "workload": spec.workload,
                                "started_at": (latest or {}).get("started_at"),
                                "elapsed_seconds": max(0, now - started) if started is not None else None,
                                "attempt_number": (latest or {}).get("attempt_number"),
                                "liveness": "not_probed",
                            }
                        )
            counts["unknown"] += len(task_ids) - observed
            global_counts.update(counts)
            totals["unfinished"] += unfinished
            campaigns.append(
                {
                    "campaign_id": cid,
                    "name": str(campaign.get("name", cid)),
                    "mode": campaign.get("mode", "unknown"),
                    "draining": bool(campaign.get("draining")),
                    **_progress(counts, len(task_ids), observed, unfinished),
                }
            )
        groups = []
        for key, group in sorted(group_data.items()):
            groups.append(
                {
                    "workload": key[0],
                    "gpu_model": key[1],
                    "software_fingerprint": key[2],
                    "cpus": key[3],
                    "attempt_outcomes": dict(group["outcomes"]),
                    "timing": _timing(group["times"]),
                    "memory": _memory(group["memory"]),
                }
            )
        for key, count in (
            ("active_tasks", active_count),
            ("groups", len(groups)),
            ("recent_runs", recent_index),
        ):
            if count > self.limits[key]:
                diagnostics["truncation"][key] = {"available": count, "shown": self.limits[key]}
        diagnostics["partial"] = bool(diagnostics["partial_records"] or diagnostics["truncation"])
        return {
            "schema": "hhq-dashboard/1",
            "generated_at": dt.datetime.fromtimestamp(now, dt.UTC).isoformat(),
            "state_exists": self.root.is_dir(),
            "campaigns": campaigns,
            "totals": {
                "tasks": totals["tasks"],
                **_progress(
                    global_counts, totals["tasks"], totals["observed_tasks"], totals["unfinished"]
                ),
            },
            "health": {"controller": health(controller, "at"), "service": health(leader, "heartbeat_at")},
            "pending_reserves": {
                "recorded_at": (controller or {}).get("at"),
                "budgets": pending_reserves,
                "reasons": allocation_plan.get("reasons", []),
                "note": "Last controller plan; planned submissions are not confirmed pending jobs. See controller age.",
            },
            "timing": _timing(times),
            "memory": _memory(memory),
            "groups": groups[: self.limits["groups"]],
            "recent_runs": [row for _, _, row in sorted(recent)],
            "active_tasks": active,
            "active_attempt_candidates": active_count,
            "attempt_outcomes": dict(outcomes),
            "retries": retries,
            "allocations": {
                "counts": dict(allocation_counts),
                "by_profile": [
                    {"profile": name, "counts": dict(counts)} for name, counts in sorted(by_profile.items())
                ],
                "hq_admission_counts": dict(hq_counts),
                "note": "Cached worker/direct submission intents, not a live Slurm queue or count of GPU devices/array slots",
            },
            "diagnostics": diagnostics,
        }
