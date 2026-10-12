"""Explainable grouped measurements; failures are never fitted as short successes."""

import math
import statistics
from collections import defaultdict


def quantile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(p * len(values)) - 1))]


def summarize_measurements(records, *, minimum_samples=5, memory_headroom_mib=2048, time_margin_seconds=60):
    groups = defaultdict(list)
    for row in records:
        if row.get("state", row.get("status")) != "succeeded" or row.get("validated", True) is False:
            continue
        key = tuple(
            row.get(k, "unknown")
            for k in (
                "workload",
                "software_fingerprint",
                "gpu_model",
                "cpus",
                "concurrency",
                "cache_condition",
            )
        )
        groups[key].append(row)
    result = []
    for key, rows in sorted(groups.items(), key=lambda x: str(x[0])):
        times = [r["wall_seconds"] for r in rows if r.get("wall_seconds") is not None]
        memory = [r["peak_memory_mib"] for r in rows if r.get("peak_memory_mib") is not None]
        enough = len(times) >= minimum_samples
        result.append(
            {
                "group": dict(
                    zip(
                        (
                            "workload",
                            "software_fingerprint",
                            "gpu_model",
                            "cpus",
                            "concurrency",
                            "cache_condition",
                        ),
                        key,
                    )
                ),
                "samples": len(rows),
                "confidence": "measured" if enough else "insufficient_samples",
                "runtime_median_seconds": statistics.median(times) if times else None,
                "suggested_runtime_seconds": quantile(times, 0.9) + time_margin_seconds if enough else None,
                "suggested_memory_mib": max(memory) + memory_headroom_mib
                if len(memory) >= minimum_samples
                else None,
                "memory_note": "Observed sampled process-tree peak plus fixed headroom; not a hard bound or GPU VRAM estimate.",
                "automatic_policy_changed": False,
            }
        )
    return result
