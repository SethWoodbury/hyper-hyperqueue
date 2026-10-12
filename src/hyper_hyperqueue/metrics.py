"""Read-only task measurements; suggestions never silently change requests."""

from __future__ import annotations

from collections import Counter

from .estimates import summarize_measurements
from .state import StateStore


def measurements(config, *, campaign_id=None, minimum_samples=5):
    store = StateStore(config.state_dir)
    rows = []
    outcomes = Counter()
    for campaign in store.campaigns():
        if campaign_id and campaign["campaign_id"] != campaign_id:
            continue
        cid = campaign["campaign_id"]
        for tid in campaign["task_ids"]:
            task = store.task(cid, tid)
            for attempt in store.attempts(cid, tid):
                result = attempt.get("result", {})
                execution = result.get("execution", result)
                telemetry = execution.get("telemetry", result.get("telemetry", {}))
                state = result.get("state", result.get("status", "incomplete"))
                outcomes[state] += 1
                context = attempt.get("context", {})
                gpu = context.get("assigned_gpu", result.get("assigned_gpu", {}))
                devices = gpu.get("devices", []) if isinstance(gpu, dict) else []
                peak = telemetry.get("sampled_peak_rss_bytes")
                rows.append(
                    {
                        "campaign_id": cid,
                        "task_id": tid,
                        "attempt_id": attempt["attempt_id"],
                        "state": state,
                        "workload": task.workload,
                        "software_fingerprint": task.software_fingerprint or "unknown",
                        "gpu_model": devices[0]["model"]
                        if len(devices) == 1
                        else ("unknown" if task.resources.gpus else "cpu"),
                        "cpus": task.resources.cpus,
                        "concurrency": 1 if task.resources.gpus else "unknown",
                        "cache_condition": "unknown",
                        "wall_seconds": telemetry.get("wall_seconds"),
                        "peak_memory_mib": peak / 1048576 if peak is not None else None,
                        "cpu_seconds": telemetry.get("sampled_cpu_seconds"),
                        "validated": result.get("validation", {}).get("valid", state == "succeeded"),
                    }
                )
    good = [r for r in rows if r["state"] == "succeeded"]
    return {
        "schema_version": 1,
        "attempt_outcomes": dict(outcomes),
        "successful_task_seconds": sum(r["wall_seconds"] or 0 for r in good),
        "failed_or_incomplete_task_seconds": sum(
            r["wall_seconds"] or 0 for r in rows if r["state"] != "succeeded"
        ),
        "groups": summarize_measurements(rows, minimum_samples=minimum_samples),
        "limitations": [
            "CPU/RAM are sampled process-tree measurements, not cgroup peaks. GPU memory use is not sampled.",
            "Unknown software/cache/concurrency remain unknown. Keep separate campaigns for incomparable work.",
            "Task time is not allocated GPU/CPU time; these totals cannot establish allocation utilization.",
            "Suggestions are advisory. Change explicit configuration to apply them; failed runtimes are excluded.",
        ],
    }
