"""Serverless arrays. Commands live in immutable task files, never shell templates."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import sys
import uuid
from pathlib import Path

from ..adaptive_recovery import effective_config, effective_task
from ..allocation_budget import budget_admission
from ..logs import log_directory
from ..models import Config, Profile, Resources
from ..planner import array_plan, compatible, shared_budget_usage
from ..state import StateStore, atomic_json, utc_now


def asdict(value):
    return value if isinstance(value, dict) else dataclasses.asdict(value)


def walltime(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def resource_args(profile: Profile, resources: Resources | None = None) -> list[str]:
    args = [
        "--partition",
        profile.partition,
        "--nodes",
        str(resources.nodes if resources else 1),
        "--ntasks",
        str(resources.ranks if resources else 1),
        "--cpus-per-task",
        str(profile.cpus),
        "--mem",
        f"{profile.memory_mib}M",
        "--time",
        walltime(profile.allocation_seconds),
        "--no-requeue",
    ]
    for flag, value in [
        ("--account", profile.account),
        ("--qos", profile.qos),
        ("--gres", profile.gres),
        ("--constraint", profile.constraint),
        ("--exclude", ",".join(profile.exclude) or None),
    ]:
        if value:
            args.extend([flag, value])
    allowed = {"--nice", "--mail-type", "--mail-user", "--licenses", "--reservation"}
    i = 0
    while i < len(profile.extra_sbatch):
        value = profile.extra_sbatch[i]
        flag = value.split("=", 1)[0]
        if flag not in allowed:
            raise ValueError(
                f"Conflicting or unsupported extra_sbatch argument {flag}; use structured fields"
            )
        args.append(value)
        if "=" not in value:
            i += 1
            if i >= len(profile.extra_sbatch) or profile.extra_sbatch[i].startswith("-"):
                raise ValueError(f"Missing value for {flag}")
            args.append(profile.extra_sbatch[i])
        i += 1
    return args


def submit_ready(
    store: StateStore, config: Config, campaign_id: str, slurm, *, max_parallel=None, commands_per_job=1
) -> dict:
    """Caller holds state mutation lock; an intent consumes capacity before RPC."""
    campaign = store.campaign(campaign_id)
    # CLI --partition profiles are immutable campaign-specific definitions. A
    # later plain `resume` can recover them without resurrecting removed named
    # operator profiles or changing any currently configured profile.
    config = config.model_copy(deep=True)
    previous_campaigns = {campaign_id: campaign}
    for intent in store.intents():
        if intent.get("state") in {"intent", "unknown", "submitted", "running"} and str(
            intent.get("profile", "")
        ).startswith("explicit_"):
            cid = intent.get("campaign_id")
            if cid and cid not in previous_campaigns:
                previous_campaigns[cid] = store.campaign(cid)
    for previous in previous_campaigns.values():
        saved = previous.get("resolved_config", {})
        for name, value in saved.get("profiles", {}).items():
            if name.startswith("explicit_") and name not in config.profiles:
                config.profiles[name] = Profile.model_validate(value)
                budget_name = value.get("budget")
                if (
                    budget_name
                    and budget_name not in config.budgets
                    and budget_name in saved.get("budgets", {})
                ):
                    from ..models import Budget

                    config.budgets[budget_name] = Budget.model_validate(saved["budgets"][budget_name])
    if config.discovery_snapshot:
        from ..discovery import load_snapshot, validate_config_against_snapshot

        validation = validate_config_against_snapshot(config, load_snapshot(config.discovery_snapshot))
        if validation["errors"]:
            raise ValueError("; ".join(validation["errors"]))
    if campaign.get("draining") or campaign.get("pause_dispatch"):
        return {"submitted_arrays": [], "reason": "CAMPAIGN_ADMISSION_PAUSED"}
    intents = [x for x in store.intents() if x.get("backend") == "direct"]
    active = [x for x in intents if x.get("state") in ("intent", "unknown", "submitted", "running")]
    own = [x for x in active if x["campaign_id"] == campaign_id]
    assigned = {tid for x in own for tid in x["task_ids"]}
    ready = [
        tid
        for tid in store.task_ids(campaign_id)
        if tid not in assigned
        and store.task_state(campaign_id, tid).get("state", "ready") in ("ready", "retry_wait")
    ]
    # Preserve an actual aggregate cap by reserving the sum of outstanding array throttles.
    already_parallel = sum(x.get("max_parallel") or x["elements"] for x in own)
    available_parallel = None if max_parallel is None else max(0, max_parallel - already_parallel)
    if available_parallel == 0:
        return {"submitted_arrays": [], "unsent_tasks": len(ready), "reason": "CAMPAIGN_CONCURRENCY_RESERVED"}
    occupied = sum(x["elements"] for x in active)
    # Include outside jobs in the configured shared submitted-task ceiling.
    jobs = [asdict(j) for j in slurm.query_jobs()]
    owned_ids = {str(x.get("job_id")) for x in active}
    external = sum(
        int(j.get("array_task_count") or 1)
        for j in jobs
        if str(j.get("job_id")).split("_")[0] not in owned_ids
    )
    available = max(0, config.policy.max_direct_submitted - occupied - external)
    base_profiles = dict(config.profiles)
    config = effective_config(store, config, [store.task(campaign_id, tid) for tid in ready])
    # Group homogeneous task resources; the default imports are homogeneous but JSONL may differ.
    groups = {}
    for tid in ready:
        task = effective_task(store, store.task(campaign_id, tid))
        if hasattr(task, "model_dump"):
            task = task.model_dump()
        key = json.dumps([task["resources"], task["allowed_profiles"]], sort_keys=True)
        groups.setdefault(key, []).append(tid)
    submissions = []
    budget_usage = shared_budget_usage(config, jobs, store.intents())
    unsent = len(ready)
    validation_errors = []
    for key, ids in groups.items():
        if available <= 0 or available_parallel == 0:
            break
        resources_data, names = json.loads(key)
        resources = Resources.model_validate(resources_data)
        duration = (
            resources.timeout_seconds or resources.runtime_seconds or config.policy.unknown_runtime_seconds
        )
        eligible = [
            name
            for name in names
            if name in config.profiles
            and compatible(resources, config.profiles[name], config, pooled=False)[0]
            and duration * min(commands_per_job, len(ids)) <= config.profiles[name].allocation_seconds
        ]
        if not eligible:
            validation_errors.append({"task_ids": ids, "reason": "NO_PROFILE_FITS_CURRENT_TASK_RESOURCES"})
            continue
        name = next(
            (
                name
                for name in eligible
                if not config.profiles[name].budget
                or budget_usage[config.profiles[name].budget]
                < config.budgets[config.profiles[name].budget].max_submitted
            ),
            None,
        )
        if name is None:
            continue
        profile = config.profiles[name].model_copy(update={
            "memory_mib": max(base_profiles[name].memory_mib, resources.memory_mib)})
        group_available = available
        if profile.budget:
            group_available = min(
                group_available,
                max(0, config.budgets[profile.budget].max_submitted - budget_usage[profile.budget]),
            )
        arrays, _ = array_plan(
            ids,
            config,
            max_parallel=available_parallel,
            commands_per_job=commands_per_job,
            available_submissions=group_available,
        )
        for array in arrays:
            allocation_budget = budget_admission(store, config, profile, elements=array["elements"])
            if not allocation_budget["allowed"]:
                return {"submitted_arrays": submissions, "unsent_tasks": unsent,
                        "reason": allocation_budget["reason"], "allocation_budget": allocation_budget}
            token = uuid.uuid4().hex
            directory = store.root / "submissions" / token
            directory.mkdir(parents=True, mode=0o700)
            logs = log_directory(config, "allocations", token, fallback=directory)
            mapping_dir = directory / "mapping"
            mapping_dir.mkdir()
            for index, task_ids in enumerate(array["mapping"]):
                atomic_json(mapping_dir / f"{index}.json", {"campaign_id": campaign_id, "task_ids": task_ids})
            argv = [
                sys.executable,
                "-m",
                "hyper_hyperqueue.backends.direct",
                "run-array",
                "--state-dir",
                str(store.root),
                "--mapping-dir",
                str(mapping_dir),
            ]
            script = directory / "job.sh"
            script.write_text("#!/bin/bash\nexec " + shlex.join(argv) + "\n")
            spec = f"0-{array['elements'] - 1}" + (
                f"%{array['max_parallel']}" if array["max_parallel"] else ""
            )
            args = resource_args(profile, resources) + [
                "--array",
                spec,
                "--output",
                str(logs / "%A_%a.slurm.out"),
                "--error",
                str(logs / "%A_%a.slurm.err"),
            ]
            task_ids = [tid for group in array["mapping"] for tid in group]
            intent = {
                "token": token,
                "backend": "direct",
                "campaign_id": campaign_id,
                "task_ids": task_ids,
                "state": "intent",
                "script": str(script),
                "extra_args": args,
                "profile": name,
                "created_at": utc_now(),
                "placement": {key: getattr(profile, key) for key in ("partition", "account", "qos")},
                "elements": array["elements"],
                "max_parallel": array["max_parallel"],
                "job_name": f"hhq-{token[:16]}",
            }
            store.put_intent(intent)
            response = asdict(slurm.submit(script, token=token, job_name=intent["job_name"], extra_args=args))
            intent.update(
                response=response,
                job_id=response.get("job_id"),
                job_name=response.get("job_name", intent["job_name"]),
                state="submitted" if response["status"] == "accepted" else response["status"],
            )
            store.put_intent(intent)
            store.event("direct_submission", campaign_id=campaign_id, token=token, response=response)
            if intent["state"] in ("submitted", "unknown"):
                for tid in task_ids:
                    store.update_task_state(
                        campaign_id,
                        tid,
                        {
                            "state": "submitted" if intent["state"] == "submitted" else "unknown",
                            "submission_token": token,
                            "slurm_job_id": intent.get("job_id"),
                        },
                    )
                unsent -= len(task_ids)
            submissions.append(intent)
            available -= array["elements"]
            if profile.budget:
                budget_usage[profile.budget] += array["elements"]
            if available_parallel is not None:
                available_parallel -= array["max_parallel"]
            if intent["state"] != "submitted":
                return {
                    "submitted_arrays": submissions,
                    "unsent_tasks": unsent,
                    "reason": "SUBMISSION_UNCERTAIN_OR_REJECTED_NO_BLIND_RETRY",
                }
    return {
        "submitted_arrays": submissions,
        "unsent_tasks": unsent,
        "validation_errors": validation_errors,
        "budget_usage": dict(budget_usage),
        "concurrency_semantics": "Strict campaign cap: sum of all outstanding array throttles; later resume feeds unsent remainder.",
    }


def run_array(state_dir, mapping_dir) -> int:
    from ..runner import run_task

    slot = int(os.environ["SLURM_ARRAY_TASK_ID"])
    mapping = json.loads((Path(mapping_dir) / f"{slot}.json").read_text())
    exit_code = 0
    for task_id in mapping["task_ids"]:
        code = run_task(state_dir, mapping["campaign_id"], task_id)
        if code:
            exit_code = code
        if code in (130, 137, 143):
            break
    return exit_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run-array"])
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--mapping-dir", required=True)
    args = parser.parse_args()
    raise SystemExit(run_array(args.state_dir, args.mapping_dir))
