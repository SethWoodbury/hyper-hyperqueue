"""Read-only cluster discovery with private raw evidence and honest unknowns."""

from __future__ import annotations

import collections
import datetime as dt
import getpass
import hashlib
import json
import os
import re
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .backends.slurm import SlurmClient, SlurmError, duration, run_command


def _write_private(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(text)


def _json(text: str) -> dict:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) and not value.get("errors") else {}
    except ValueError:
        return {}


def parse_config(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z][A-Za-z0-9_\[\]]*)\s*=\s*(.*?)\s*$", line)
        if match:
            out[match[1]] = match[2]
    return out


def parse_partition(text: str) -> dict:
    fields = dict(re.findall(r"(\w+)=(.*?)(?=\s+\w+=|$)", text))
    return {
        "name": fields.get("PartitionName"),
        "state": fields.get("State"),
        "qos": None if fields.get("QoS") in {None, "N/A"} else fields["QoS"],
        "max_time_seconds": duration(fields.get("MaxTime")),
        "preempt_mode": fields.get("PreemptMode"),
        "allow_accounts": fields.get("AllowAccounts"),
        "allow_groups": fields.get("AllowGroups"),
        "allow_qos": fields.get("AllowQos"),
        "total_nodes": int(fields["TotalNodes"]) if fields.get("TotalNodes", "").isdigit() else None,
        "tres": fields.get("TRES"),
        "priority_tier": fields.get("PriorityTier"),
        "max_memory_per_node": fields.get("MaxMemPerNode"),
        "source": "scontrol show partition -o",
    }


def _limit(value: str | None, source: str) -> dict:
    if value in {None, ""}:
        return {"value": None, "status": "not_set_at_this_scope", "source": source}
    if str(value).upper() in {"UNLIMITED", "INFINITE"}:
        return {"value": None, "status": "unlimited_at_this_scope", "source": source}
    return {"value": int(value) if str(value).isdigit() else value, "status": "observed", "source": source}


def summarize(raw: dict[str, dict], user: str, jobs: list[dict], queue_error: str | None = None) -> dict:
    def output(key):
        record = raw.get(key, {})
        return record.get("stdout", "") if record.get("returncode") == 0 else ""

    config = parse_config(output("config_text"))
    nodes = _json(output("nodes_json")).get("nodes", [])
    partitions = [
        parse_partition(line) for line in output("partitions_text").splitlines() if "PartitionName=" in line
    ]
    inventory = collections.Counter()
    for node in nodes:
        key = (
            tuple(sorted(node.get("partitions", []))),
            node.get("gres", ""),
            tuple(sorted(node.get("features", []))),
            node.get("cpus"),
            node.get("real_memory"),
        )
        inventory[key] += 1
    assoc = []
    for line in output("associations_text").splitlines():
        f = line.split("|")
        if len(f) < 12:
            continue
        (
            cluster,
            account,
            owner,
            part,
            qos,
            default,
            maxjobs,
            maxsubmit,
            grpjobs,
            grpsubmit,
            maxwall,
            grptres,
        ) = f[:12]
        assoc.append(
            {
                "cluster": cluster,
                "account": account,
                "user": owner,
                "partition": part,
                "qos": qos.split(",") if qos else [],
                "default_qos": default or None,
                "limits": {
                    k: _limit(v, "user association")
                    for k, v in {
                        "max_jobs": maxjobs,
                        "max_submit_jobs": maxsubmit,
                        "group_jobs": grpjobs,
                        "group_submit_jobs": grpsubmit,
                        "max_wall": maxwall,
                        "group_tres": grptres,
                    }.items()
                },
            }
        )
    qos_rows = []
    for line in output("qos_text").splitlines():
        f = line.split("|")
        if len(f) < 13:
            continue
        (
            name,
            priority,
            flags,
            maxjobs,
            maxsubmit,
            maxwall,
            grpjobs,
            grpsubmit,
            grptres,
            maxtresuser,
            maxtresjob,
            preempt,
            preemptmode,
        ) = f[:13]
        qos_rows.append(
            {
                "name": name,
                "priority": priority,
                "flags": flags,
                "preempt": preempt,
                "preempt_mode": preemptmode,
                "limits": {
                    k: _limit(v, f"QOS {name}")
                    for k, v in {
                        "max_jobs_per_user": maxjobs,
                        "max_submit_per_user": maxsubmit,
                        "max_wall": maxwall,
                        "group_jobs": grpjobs,
                        "group_submit": grpsubmit,
                        "group_tres": grptres,
                        "max_tres_per_user": maxtresuser,
                        "max_tres_per_job": maxtresjob,
                    }.items()
                },
            }
        )
    scheduler = dict(
        x.split("=", 1) if "=" in x else (x, True)
        for x in config.get("SchedulerParameters", "").split(",")
        if x
    )
    maxarray = config.get("MaxArraySize")
    max_tasks = scheduler.get("max_array_tasks")
    limits = {
        "max_array_size": _limit(maxarray, "scontrol show config MaxArraySize"),
        "max_array_tasks": _limit(max_tasks, "SchedulerParameters max_array_tasks"),
        "max_job_count": _limit(config.get("MaxJobCount"), "cluster-wide MaxJobCount, NOT user budget"),
        "effective_user_submit_limit": {
            "value": None,
            "status": "unknown",
            "source": "hierarchy and job-submit plugin not fully observable",
        },
        "scheduler_backfill_jobs_per_user": _limit(
            scheduler.get("bf_max_job_user"), "backfill inspection limit, NOT submission cap"
        ),
    }
    if max_tasks is None and maxarray and maxarray.isdigit():
        limits["max_array_tasks"] = {
            "value": int(maxarray),
            "status": "documented_default",
            "source": "no max_array_tasks override; bounded by MaxArraySize",
        }
    by_partition = {}
    for row in jobs:
        counter = by_partition.setdefault(row["partition"], collections.Counter())
        counter[row["state"]] += row.get("array_task_count", 1)
    storage = []
    for key, record in raw.items():
        if key.startswith("storage_"):
            for fs in _json(record.get("stdout", "")).get("filesystems", []):
                storage.append(
                    {
                        "requested_path": record["argv"][-1],
                        **fs,
                        "cross_node_visibility_verified": False,
                        "lock_and_fencing_verified": False,
                    }
                )
    return {
        "schema": "hhq-cluster-discovery/1",
        "observed_at": dt.datetime.now(dt.UTC).isoformat(),
        "cluster": config.get("ClusterName", "unknown"),
        "slurm_version": config.get("SLURM_VERSION", output("version").strip()),
        "host": socket.gethostname(),
        "partitions": partitions,
        "associations": assoc,
        "qos": qos_rows,
        "nodes_summary": [
            {
                "partitions": list(k[0]),
                "gres": k[1],
                "features": list(k[2]),
                "node_count": n,
                "cpus": k[3],
                "memory_mib": k[4],
            }
            for k, n in sorted(inventory.items())
        ],
        "limits": limits,
        "scheduler": {
            k: config.get(k)
            for k in [
                "JobRequeue",
                "PreemptType",
                "PreemptMode",
                "AccountingStorageEnforce",
                "JobSubmitPlugins",
                "MinJobAge",
                "SlurmctldTimeout",
                "SelectType",
                "TaskPlugin",
            ]
        },
        "current_user": {
            "user": user,
            "submitted": sum(j.get("array_task_count", 1) for j in jobs) if queue_error is None else None,
            "jobs": jobs,
            "counts_by_state": {
                state: sum(j.get("array_task_count", 1) for j in jobs if j["state"] == state)
                for state in {j["state"] for j in jobs}
            },
            "counts_by_partition": {k: dict(v) for k, v in by_partition.items()},
            "error": queue_error,
            "array_elements_expanded": all(not j.get("array_task_expression") for j in jobs),
        },
        "storage": storage,
        "unknowns": [
            "Effective limits inherited from unqueried parent associations or job-submit plugins",
            "GPU VRAM, architecture/runtime compatibility and device isolation require allocated probes",
            "Cross-node storage visibility, locking and stale-owner fencing require allocated probes",
            "A partition/account association is not a guarantee of immediate admission or runtime compatibility",
        ],
        "command_failures": [
            {"name": k, "returncode": r["returncode"], "stderr": r.get("stderr", "")[-2000:]}
            for k, r in raw.items()
            if r.get("returncode") != 0
        ],
    }


def discover(
    output_dir: str | Path,
    raw_dir: str | Path | None = None,
    user: str | None = None,
    runner=None,
    storage_paths: list[str] | None = None,
) -> dict:
    """Run bounded read-only probes. Never submits jobs or assumes blank limits are infinite."""
    user = user or getpass.getuser()
    output_dir = Path(output_dir).expanduser().resolve()
    raw_dir = Path(raw_dir).expanduser().resolve() if raw_dir else output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    runner = runner or run_command
    commands = {
        "version": ["scontrol", "--version"],
        "nodes_json": ["scontrol", "--json", "show", "nodes"],
        "partitions_text": ["scontrol", "show", "partition", "-o"],
        "config_text": ["scontrol", "show", "config"],
        "associations_text": [
            "sacctmgr",
            "-n",
            "-P",
            "show",
            "assoc",
            "where",
            f"user={user}",
            "format=Cluster,Account,User,Partition,QOS,DefaultQOS,MaxJobs,MaxSubmitJobs,GrpJobs,GrpSubmitJobs,MaxWall,GrpTRES",
        ],
        "qos_text": [
            "sacctmgr",
            "-n",
            "-P",
            "show",
            "qos",
            "format=Name,Priority,Flags,MaxJobsPU,MaxSubmitPU,MaxWall,GrpJobs,GrpSubmit,GrpTRES,MaxTRESPU,MaxTRESPJ,Preempt,PreemptMode",
        ],
    }
    for i, path in enumerate(storage_paths or [str(Path.home())]):
        commands[f"storage_{i}"] = ["findmnt", "--json", "--target", str(Path(path).expanduser())]

    def capture(item):
        key, argv = item
        started = dt.datetime.now(dt.UTC).isoformat()
        result = runner(argv, 30)
        record = {
            "argv": argv,
            "started_at": started,
            "ended_at": dt.datetime.now(dt.UTC).isoformat(),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
        }
        for stream in ["stdout", "stderr"]:
            path = raw_dir / f"{key}.{stream}"
            _write_private(path, record[stream])
            record[f"{stream}_sha256"] = hashlib.sha256(record[stream].encode()).hexdigest()
            record[f"{stream}_path"] = str(path)
        return key, record

    with ThreadPoolExecutor(max_workers=4) as pool:
        raw = dict(pool.map(capture, commands.items()))
    # Traverse only this user's account ancestry, not every cluster user.
    pending_accounts = {
        line.split("|")[1]
        for line in raw["associations_text"]["stdout"].splitlines()
        if len(line.split("|")) >= 2
    }
    visited_accounts = set()
    for _depth in range(8):
        next_accounts = set()
        for account in sorted(pending_accounts - visited_accounts):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", account):
                continue
            visited_accounts.add(account)
            key, record = capture(
                (
                    f"account_{account}",
                    [
                        "sacctmgr",
                        "-n",
                        "-P",
                        "show",
                        "assoc",
                        "where",
                        f"account={account}",
                        "user=",
                        "format=Cluster,Account,User,Partition,ParentName,QOS,DefaultQOS,MaxJobs,MaxSubmitJobs,GrpJobs,GrpSubmitJobs,MaxWall,GrpTRES",
                    ],
                )
            )
            raw[key] = record
            if record["returncode"] == 0:
                for line in record["stdout"].splitlines():
                    fields = line.split("|")
                    if len(fields) >= 5 and fields[4]:
                        next_accounts.add(fields[4])
        if not next_accounts - visited_accounts:
            break
        pending_accounts = next_accounts
    client = SlurmClient(user=user, runner=runner)
    try:
        jobs = [j.to_dict() for j in client.query_jobs()]
        queue_error = None
    except (SlurmError, OSError, ValueError, KeyError, TypeError) as error:
        jobs, queue_error = [], f"{type(error).__name__}: {error}"
    snapshot = summarize(raw, user, jobs, queue_error)
    snapshot["account_hierarchy"] = []
    hierarchy_fields = [
        "cluster",
        "account",
        "user",
        "partition",
        "parent",
        "qos",
        "default_qos",
        "max_jobs",
        "max_submit_jobs",
        "group_jobs",
        "group_submit_jobs",
        "max_wall",
        "group_tres",
    ]
    for key, record in raw.items():
        if key.startswith("account_") and record["returncode"] == 0:
            for line in record["stdout"].splitlines():
                values = line.split("|")
                if len(values) >= len(hierarchy_fields):
                    snapshot["account_hierarchy"].append(dict(zip(hierarchy_fields, values, strict=False)))
    snapshot["capabilities"] = {
        "json": client.json_capabilities,
        "nodes_json": bool(_json(raw["nodes_json"]["stdout"]).get("nodes")),
    }
    snapshot["provenance"] = {
        "raw_directory": str(raw_dir),
        "commands": {
            key: {k: v for k, v in r.items() if k not in {"stdout", "stderr"}} for key, r in raw.items()
        },
    }
    _write_private(raw_dir / "manifest.json", json.dumps(snapshot["provenance"], indent=2) + "\n")
    _write_private(output_dir / "snapshot.json", json.dumps(snapshot, indent=2) + "\n")
    _write_private(output_dir / "cluster-report.md", render_report(snapshot))
    return snapshot


def load_snapshot(path: str | Path) -> dict:
    data = json.loads(Path(path).expanduser().read_text())
    if data.get("schema") != "hhq-cluster-discovery/1":
        raise ValueError("Unsupported discovery snapshot schema")
    return data


def validate_config_against_snapshot(config, snapshot: dict) -> dict:
    """Validate observable placement facts, without claiming admission or GPU compatibility."""
    errors, warnings = [], []
    partitions = {r["name"]: r for r in snapshot.get("partitions", [])}
    associations = snapshot.get("associations", [])
    inventory = snapshot.get("nodes_summary", [])
    for name, profile in config.profiles.items():
        part = partitions.get(profile.partition)
        if part is None:
            errors.append(f"{name}: partition absent from discovery: {profile.partition}")
            continue
        if part.get("state") not in {"UP", None}:
            errors.append(f"{name}: partition state is {part['state']}")
        maximum = part.get("max_time_seconds")
        if maximum is not None and profile.allocation_seconds > maximum:
            errors.append(f"{name}: allocation exceeds observed partition walltime")
        matching = [
            a
            for a in associations
            if a.get("account") == profile.account and a.get("partition") in {"", profile.partition}
        ]
        if profile.account and associations and not matching:
            errors.append(f"{name}: no observed user association for account/partition")
        if profile.qos and matching and not any(profile.qos in a.get("qos", []) for a in matching):
            errors.append(f"{name}: requested job QOS absent from observed association")
        eligible_nodes = [row for row in inventory if profile.partition in row["partitions"]]
        if profile.constraint:
            if re.fullmatch(r"[A-Za-z0-9_.-]+", profile.constraint):
                eligible_nodes = [row for row in eligible_nodes if profile.constraint in row["features"]]
                if inventory and not eligible_nodes:
                    errors.append(f"{name}: constraint absent from observed partition node features")
            else:
                warnings.append(f"{name}: compound constraint requires explicit placement validation")
        known_sizes = [
            row for row in eligible_nodes if row.get("cpus") is not None and row.get("memory_mib") is not None
        ]
        if known_sizes and not any(
            row["cpus"] >= profile.cpus and row["memory_mib"] >= profile.memory_mib for row in known_sizes
        ):
            errors.append(f"{name}: CPU/memory allocation exceeds every observed node in the partition")
        if profile.gres and not inventory:
            errors.append(f"{name}: GPU placement is unresolved because node inventory is unavailable")
        if profile.gres:
            match = re.fullmatch(r"gpu(?::([A-Za-z0-9_-]+))?:(\d+)", profile.gres)
            if match:
                kind, count = match[1], int(match[2])
                possible = False
                for row in eligible_nodes:
                    if row.get("cpus") is not None and row["cpus"] < profile.cpus:
                        continue
                    if row.get("memory_mib") is not None and row["memory_mib"] < profile.memory_mib:
                        continue
                    for item in re.finditer(r"(?:^|,)gpu(?::([^,:()]+))?:(\d+)", row["gres"]):
                        if (kind is None or kind == item[1]) and count <= int(item[2]):
                            possible = True
                if inventory and not possible:
                    errors.append(f"{name}: requested GPU GRES/count absent from partition inventory")
            else:
                warnings.append(f"{name}: complex GRES expression needs explicit allocated validation")
        if profile.min_vram_mib or profile.gpu_models:
            warnings.append(f"{name}: GPU VRAM/software compatibility requires an allocated worker probe")
    for key in ["max_array_size", "max_array_tasks"]:
        bound = snapshot.get("limits", {}).get(key, {}).get("value")
        if isinstance(bound, int) and getattr(config.policy, key) > bound:
            errors.append(f"{key} exceeds discovered bound {bound}")
    warnings.extend(snapshot.get("unknowns", []))
    return {
        "valid_observable_constraints": not errors,
        "errors": errors,
        "warnings": warnings,
        "discovery_observed_at": snapshot.get("observed_at"),
    }


def render_report(snapshot: dict) -> str:
    """Readable private report; generated alongside raw machine-readable evidence."""
    lines = [
        "# Cluster discovery",
        "",
        f"Observed: {snapshot.get('observed_at', 'unknown')}",
        f"Cluster: {snapshot.get('cluster', 'unknown')}; Slurm: {snapshot.get('slurm_version', 'unknown')}",
        "",
        "This report records configuration and current-user accounting. It does not predict free nodes.",
        "",
        "## Partitions",
        "",
        "| Partition | Walltime (hours) | Preemption | Partition QOS | Nodes |",
        "|---|---:|---|---|---:|",
    ]
    for part in snapshot.get("partitions", []):
        seconds = part.get("max_time_seconds")
        hours = f"{seconds / 3600:g}" if seconds is not None else "unknown/unlimited"
        lines.append(
            f"| {part['name']} | {hours} | {part.get('preempt_mode')} | {part.get('qos') or 'none'} | {part.get('total_nodes')} |"
        )
    lines += ["", "## Scheduler limits", "", "| Setting | Value | Evidence scope |", "|---|---|---|"]
    for name, item in snapshot.get("limits", {}).items():
        value = item.get("value") if item.get("value") is not None else item.get("status", "unknown")
        lines.append(f"| {name} | {value} | {item.get('source', 'unknown')} |")
    lines += [
        "",
        "A blank association/QOS limit means unset at that scope. It does not prove unlimited effective admission.",
        "",
        "## QOS and account scopes",
        "",
        "| QOS | Max running jobs/user | Max submitted/user | TRES/user |",
        "|---|---|---|---|",
    ]
    for row in snapshot.get("qos", []):

        def value(key, row=row):
            limit = row["limits"].get(key, {})
            return limit.get("value") if limit.get("value") is not None else limit.get("status", "unknown")

        lines.append(
            f"| {row['name']} | {value('max_jobs_per_user')} | {value('max_submit_per_user')} | {value('max_tres_per_user')} |"
        )
    lines += [
        "",
        "User associations: "
        + ", ".join(f"{a['account']}/{a['partition'] or '*'}" for a in snapshot.get("associations", [])),
        "Queried account ancestry: "
        + ", ".join(
            f"{a['account']} → {a['parent'] or '(root)'}" for a in snapshot.get("account_hierarchy", [])
        ),
        "",
        "## Current user snapshot",
        "",
    ]
    user = snapshot.get("current_user", {})
    lines += [
        f"User: {user.get('user')}; submitted elements: {user.get('submitted')}; states: {user.get('counts_by_state', {})}.",
        "Large compressed arrays carry exact element counts; job-record count is not the submitted-element count.",
    ]
    if user.get("error"):
        lines += [f"Queue observation failed: {user['error']}. Capacity is unknown."]
    lines += ["", "## Storage and unverified assumptions", ""]
    for storage in snapshot.get("storage", []):
        lines += [
            f"- {storage.get('requested_path')}: {storage.get('fstype', 'unknown')} ({storage.get('options', 'options unavailable')})."
        ]
    lines += [""] + [f"- {item}" for item in snapshot.get("unknowns", [])]
    lines += [
        "",
        "Configured worker and queue targets are operator policy. They are not inferred from free-node snapshots or fabricated from unknown limits.",
        "",
        "## Provenance",
        "",
        f"Raw private evidence: {snapshot.get('provenance', {}).get('raw_directory', 'not supplied')}",
        "The adjacent snapshot.json binds command arguments, timestamps, response hashes and failures.",
        "Keep raw inventory, user/job details and internal endpoints outside version control.",
        "",
    ]
    return "\n".join(lines)
