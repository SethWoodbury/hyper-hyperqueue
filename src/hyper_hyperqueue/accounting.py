"""Explicit, read-only allocation reports; never infer fair-share from GPU utilization.

Run after a campaign (or while it runs) with exact Slurm job IDs. This is separate
from the controller/dashboard refresh, so a browser never triggers scheduler RPCs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from .backends.slurm import JOB_RE, _memory, run_command

FIELDS = (
    "DBIndex",
    "JobID",
    "User",
    "Partition",
    "QOS",
    "State",
    "Submit",
    "Start",
    "End",
    "ElapsedRaw",
    "TimelimitRaw",
    "AllocTRES",
    "NodeList",
)


def _numeric(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (ValueError, TypeError):
        return None


def _epoch(value):
    try:
        return dt.datetime.fromisoformat(value).replace(tzinfo=dt.UTC).timestamp()
    except (ValueError, TypeError):
        return None


def parse_allocations(text: str, *, usage_factors=None) -> list[dict]:
    """Parse allocation-only sacct rows, retaining missing measurements as None.

    A requeued job may have several accounting records: DBIndex + start + submit
    identify them. Exact repeats are discarded. Steps are never counted twice.
    """
    if any(_numeric(value) is None for value in (usage_factors or {}).values()):
        raise ValueError("QoS usage factors must be finite and nonnegative")
    rows, seen = [], set()
    for raw in csv.DictReader(io.StringIO(text), delimiter="|"):
        job_id = raw.get("JobID", "")
        if not JOB_RE.fullmatch(job_id) or "[" in job_id:
            continue
        identity = (raw.get("DBIndex"), job_id, raw.get("Submit"), raw.get("Start"))
        if identity in seen:
            continue
        seen.add(identity)
        tres = dict(item.split("=", 1) for item in raw.get("AllocTRES", "").split(",") if "=" in item)
        elapsed = _numeric(raw.get("ElapsedRaw"))
        started = _epoch(raw.get("Start"))
        submitted = _epoch(raw.get("Submit"))
        hours = elapsed / 3600 if elapsed is not None else None
        gpu = _numeric(tres.get("gres/gpu"))
        if gpu is None and tres:
            typed = [_numeric(v) for k, v in tres.items() if k.startswith("gres/gpu:")]
            gpu = sum(typed) if all(v is not None for v in typed) else None
        memory = _memory(tres.get("mem"))
        billing = _numeric(tres.get("billing"))
        cpu = _numeric(tres.get("cpu"))
        # Never-started pending/canceled jobs have no allocated elapsed usage.
        never_allocated = not tres and started is None and elapsed == 0
        if never_allocated:
            gpu = cpu = memory = billing = 0
        factor = (usage_factors or {}).get(raw.get("QOS"))

        def used(amount, hours=hours):
            return amount * hours if amount is not None and hours is not None else None

        rows.append(
            {
                "job_id": job_id,
                "accounting_id": raw.get("DBIndex"),
                "user": raw.get("User"),
                "partition": raw.get("Partition"),
                "qos": raw.get("QOS"),
                "state": raw.get("State"),
                "submit_utc": raw.get("Submit"),
                "start_utc": raw.get("Start"),
                "end_utc": raw.get("End"),
                "nodes": raw.get("NodeList"),
                "elapsed_seconds": elapsed,
                "requested_walltime_minutes": _numeric(raw.get("TimelimitRaw")),
                "submit_to_start_seconds": max(0, started - submitted)
                if started is not None and submitted is not None
                else None,
                "allocated_tres": tres,
                "allocated_gpus": gpu,
                "allocated_cpus": cpu,
                "allocated_memory_mib": memory,
                "gpu_hours": used(gpu),
                "cpu_hours": used(cpu),
                "memory_gib_hours": used(memory / 1024 if memory is not None else None),
                "billing_unit_hours": used(billing),
                "qos_usage_factor": factor,
                "weighted_billing_unit_hours": used(billing * factor)
                if billing is not None and factor is not None
                else None,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    def totals(group):
        result = {"allocation_records": len(group)}
        for metric in (
            "gpu_hours",
            "cpu_hours",
            "memory_gib_hours",
            "billing_unit_hours",
            "weighted_billing_unit_hours",
        ):
            known = [row[metric] for row in group if row[metric] is not None]
            result[metric] = sum(known) if known else None
            result[metric + "_missing_records"] = len(group) - len(known)
        return result

    groups = defaultdict(list)
    for row in rows:
        groups[row["partition"]].append(row)
    return {
        "schema": "hhq-allocation-accounting/1",
        "collected_at": dt.datetime.now(dt.UTC).isoformat(),
        "totals": totals(rows),
        "by_partition": {key: totals(value) for key, value in groups.items()},
        "allocations": rows,
        "limitations": [
            "Allocated time includes startup and idle time; it is not GPU utilization or inference time.",
            "Running records are partial. Missing accounting is unknown, never successful or zero cost.",
            "Billing units are Slurm allocation weights, not money or a predicted change in priority.",
            "Fair-share depends on all user/account activity, decay and other users; backend causality is not isolated.",
            "GPU TRES types can be coarse; use attempt/worker device telemetry for actual GPU model and UUID.",
            "Submit-to-start includes holds/dependencies and is not necessarily eligible queue wait.",
        ],
    }


def allocation_report(job_ids, *, command=run_command, usage_factors=None) -> dict:
    ids = sorted({str(value) for value in job_ids})
    if not ids or any(not JOB_RE.fullmatch(value) or "[" in value for value in ids):
        raise ValueError("Supply explicit numeric Slurm job/array IDs, not expressions or shell text.")
    all_rows = []
    # Bounded argv and one RPC per 200 allocations, never one query per task.
    for first in range(0, len(ids), 200):
        result = command(
            [
                "sacct",
                "--allocations",
                "--duplicates",
                "--parsable2",
                "--jobs",
                ",".join(ids[first : first + 200]),
                "--format",
                ",".join(FIELDS),
            ]
        )
        if result.returncode != 0:
            raise RuntimeError("Slurm accounting query failed; usage is unknown: " + result.stderr[:500])
        all_rows.extend(parse_allocations(result.stdout, usage_factors=usage_factors))
    # A caller can supply both an array ID and its individual task IDs.
    unique = {(r["accounting_id"], r["job_id"], r["submit_utc"], r["start_utc"]): r for r in all_rows}
    report = summarize(list(unique.values()))
    observed = {row["job_id"] for row in unique.values()}
    report["requested_job_ids"] = ids
    report["missing_job_ids"] = [
        jid for jid in ids if jid not in observed and not any(row.startswith(jid + "_") for row in observed)
    ]
    return report


def fairshare_snapshot(user: str, *, command=run_command) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", user):
        raise ValueError("Invalid user name")
    result = command(
        [
            "sshare",
            "--noheader",
            "--parsable2",
            "--users",
            user,
            "--format",
            "Account,User,Partition,RawShares,NormShares,RawUsage,EffectvUsage,FairShare,LevelFS",
        ]
    )
    return {
        "collected_at": dt.datetime.now(dt.UTC).isoformat(),
        "user": user,
        "available": result.returncode == 0,
        "fields": "Account|User|Partition|RawShares|NormShares|RawUsage|EffectvUsage|FairShare|LevelFS",
        "rows": result.stdout.splitlines() if result.returncode == 0 else [],
        "error": result.stderr[:500] if result.returncode != 0 else None,
        "meaning": "Context snapshot across user/account work, not attributable to one experiment/backend.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fairshare-user")
    parser.add_argument("--qos-factor", action="append", default=[], metavar="NAME=FACTOR")
    args = parser.parse_args()
    factors = {}
    for entry in args.qos_factor:
        key, value = entry.split("=", 1)
        factors[key] = float(value)
        if not math.isfinite(factors[key]) or factors[key] < 0:
            parser.error("QoS usage factors must be finite and nonnegative")
    result = allocation_report(args.jobs, usage_factors=factors)
    if args.fairshare_user:
        result["fairshare_snapshot"] = fairshare_snapshot(args.fairshare_user)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(args.output)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
