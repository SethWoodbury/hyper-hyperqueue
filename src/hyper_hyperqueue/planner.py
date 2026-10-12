"""Pure policy: no Slurm RPCs, filesystem writes, or scheduler forecasts."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, is_dataclass

from .models import Config, Plan, Profile, Resources


def gpu_count(profile: Profile) -> int:
    if not profile.gres:
        return 0
    return sum(int(g.split(":")[-1]) for g in profile.gres.split(",") if g.startswith("gpu:"))


def compatible(resources: Resources, profile: Profile, config: Config, *, pooled=True) -> tuple[bool, str]:
    if pooled and (resources.nodes != 1 or resources.ranks != 1):
        return False, "POOLED_MPI_UNSUPPORTED: use direct mode with an explicit launcher"
    if pooled and gpu_count(profile) not in (0, 1):
        return False, "POOLED_MULTI_GPU_UNSUPPORTED: use direct mode"
    if pooled and bool(resources.gpus) != bool(gpu_count(profile)):
        return False, "CPU_GPU_POOL_SEPARATION"
    if resources.gpus > gpu_count(profile):
        return False, "INSUFFICIENT_GPUS"
    if resources.gpu_type and f"gpu:{resources.gpu_type}:" not in (profile.gres or ""):
        return False, "GPU_TYPE_MISMATCH"
    if (
        resources.gpu_models
        and profile.gpu_models
        and not set(resources.gpu_models) & set(profile.gpu_models)
    ):
        return False, "GPU_MODEL_MISMATCH"
    if resources.min_vram_mib and profile.min_vram_mib and resources.min_vram_mib > profile.min_vram_mib:
        return False, "INSUFFICIENT_DECLARED_VRAM"
    overhead = config.policy.worker_overhead_mib if pooled else 0
    if resources.cpus > profile.cpus or resources.memory_mib + overhead > profile.memory_mib:
        return False, "INSUFFICIENT_CPU_OR_HOST_RAM"
    duration = resources.timeout_seconds or resources.runtime_seconds or config.policy.unknown_runtime_seconds
    if (
        pooled
        and config.control.hosting == "slurm"
        and duration + config.policy.cleanup_seconds > config.control.walltime_seconds
    ):
        return False, "CONTROL_LIFETIME_TOO_SHORT: increase control walltime or use direct mode"
    if duration + (config.policy.cleanup_seconds if pooled else 0) > profile.allocation_seconds:
        return False, "TASK_WILL_NOT_FIT_LIFETIME"
    return True, "ELIGIBLE_ACTUAL_GPU_CHECK_AT_STARTUP" if resources.gpus else "ELIGIBLE"


def choose_mode(requested: str, resources: Resources, count: int, config: Config) -> tuple[str, list[str]]:
    if requested != "auto":
        return requested, ["EXPLICIT_MODE"]
    if resources.gpus:
        return "pooled", ["GPU_LONG_LIVED_WORKERS"]
    if resources.nodes > 1 or resources.ranks > 1:
        return "direct", ["DECLARED_MPI_DIRECT"]
    if resources.runtime_seconds is None:
        return "direct", ["RUNTIME_UNKNOWN_CPU_DIRECT_ASSUMPTION", "OVERRIDE_WITH_MODE_POOLED"]
    if resources.runtime_seconds >= config.policy.short_cpu_seconds:
        return "direct", ["LONG_INDEPENDENT_CPU"]
    return "pooled", ["SHORT_CPU_TASK_PACKING"]


def array_plan(
    task_ids: list[str],
    config: Config,
    *,
    max_parallel: int | None = None,
    commands_per_job: int = 1,
    available_submissions: int | None = None,
) -> tuple[list[dict], list[str]]:
    if commands_per_job < 1 or (max_parallel is not None and max_parallel < 1):
        raise ValueError("commands_per_job and max_parallel must be positive")
    limit = min(config.policy.max_array_size, config.policy.max_array_tasks)
    budget = (
        config.policy.max_direct_submitted if available_submissions is None else max(0, available_submissions)
    )
    groups = [task_ids[i : i + commands_per_job] for i in range(0, len(task_ids), commands_per_job)]
    groups = groups[:budget]
    chunks = [groups[i : i + limit] for i in range(0, len(groups), limit)]
    if max_parallel is not None:
        chunks = chunks[:max_parallel]  # each outstanding array reserves at least one slot
    quota = max_parallel
    arrays = []
    for i, chunk in enumerate(chunks):
        throttle = None if quota is None else min(len(chunk), max(1, quota // (len(chunks) - i)))
        if quota is not None:
            quota -= throttle
        arrays.append({"mapping": chunk, "elements": len(chunk), "max_parallel": throttle})
    consumed = sum(len(ids) for array in arrays for ids in array["mapping"])
    return arrays, task_ids[consumed:]


def make_plan(
    task_count: int,
    resources: Resources,
    config: Config,
    names: list[str],
    *,
    mode="auto",
    max_parallel=None,
    commands_per_job=1,
) -> Plan:
    selected, reasons = choose_mode(mode, resources, task_count, config)
    valid, rejected = [], []
    for name in names:
        ok, why = compatible(resources, config.profiles[name], config, pooled=selected == "pooled")
        if ok and selected == "direct" and commands_per_job > 1:
            duration = (
                resources.timeout_seconds
                or resources.runtime_seconds
                or config.policy.unknown_runtime_seconds
            )
            if duration * min(commands_per_job, task_count) > config.profiles[name].allocation_seconds:
                ok, why = False, "DIRECT_BUNDLE_WILL_NOT_FIT_LIFETIME"
        (valid if ok else rejected).append(name if ok else f"{name}: {why}")
    if not valid:
        raise ValueError("No feasible allocation profile: " + "; ".join(rejected))
    if selected == "pooled" and commands_per_job != 1:
        raise ValueError("commands_per_job applies only to direct mode; HQ dispatches individual tasks")
    arrays, unsent = (
        ([], [])
        if selected == "pooled"
        else array_plan(
            [str(i) for i in range(task_count)],
            config,
            max_parallel=max_parallel,
            commands_per_job=commands_per_job,
        )
    )
    preview = [
        {
            "elements": a["elements"],
            "max_parallel": a["max_parallel"],
            "first_source_index": a["mapping"][0][0],
            "last_source_index": a["mapping"][-1][-1],
            "commands_per_job": commands_per_job,
        }
        for a in arrays
    ]
    return Plan(
        mode=selected,
        task_count=task_count,
        profiles=valid,
        reasons=reasons,
        arrays=preview,
        unsent_tasks=len(unsent),
        service_required=selected == "pooled",
        requests=[
            {
                "profile": n,
                **config.profiles[n].model_dump(),
                "nodes": resources.nodes,
                "ranks": resources.ranks,
            }
            for n in valid
        ],
        warnings=rejected
        + (
            ["DIRECT_MPI_REQUIRES_EXPLICIT_LAUNCHER_IN_COMMAND"]
            if selected == "direct" and (resources.nodes > 1 or resources.ranks > 1)
            else []
        )
        + (["UNKNOWN_RUNTIME_USING_DECLARED_POLICY_ASSUMPTION"] if resources.runtime_seconds is None else []),
    )


def job_dict(job):
    if isinstance(job, dict):
        return job
    return asdict(job) if is_dataclass(job) else job.model_dump()


ACTIVE_INTENT_STATES = {"intent", "unknown", "submitted", "running"}


def shared_budget_usage(config: Config, jobs: list, reservations: list[dict]) -> Counter:
    """Count each live external job or durable allocation reservation exactly once.

    Entire outstanding arrays reserve their element count until reconciliation;
    this may underfill briefly, but accepted/ambiguous work never disappears.
    """
    active = [
        row
        for row in reservations
        if row.get("state") in ACTIVE_INTENT_STATES and row.get("backend") in (None, "direct", "worker")
    ]
    observed = [job_dict(row) for row in jobs]

    def belongs(job, reservation):
        parent = str(job.get("array_job_id") or job["job_id"]).split("_")[0]
        return (reservation.get("job_id") is not None and parent == str(reservation["job_id"])) or (
            bool(reservation.get("token"))
            and (
                job.get("comment") == "hhq:" + reservation["token"]
                or str(job.get("name", "")).endswith("-" + reservation["token"])
            )
        )

    usage = Counter()
    for name, budget in config.budgets.items():

        def applies(partition, account, qos, budget=budget):
            return (
                (not budget.partitions or not partition or partition in budget.partitions)
                and (not budget.account or not account or account == budget.account)
                and (not budget.qos or not qos or qos == budget.qos)
            )

        for row in active:
            placement = row.get("placement")
            if placement is None:
                matches = [job for job in observed if belongs(job, row)]
                if matches:
                    placement = {key: matches[0].get(key) for key in ("partition", "account", "qos")}
            if placement is None:
                profile = config.profiles.get(row.get("profile"))
                if profile is None:
                    raise ValueError(
                        f"Outstanding reservation references missing profile: {row.get('profile')}"
                    )
                placement = {key: getattr(profile, key) for key in ("partition", "account", "qos")}
            if applies(placement.get("partition"), placement.get("account"), placement.get("qos")):
                usage[name] += int(row.get("elements", 1))
        if not budget.include_external_user_jobs:
            continue
        for job in observed:
            if str(job.get("state") or "UNKNOWN").split()[0].rstrip("+") in {
                "COMPLETED",
                "CANCELLED",
                "FAILED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "BOOT_FAIL",
                "PREEMPTED",
                "DEADLINE",
                "REVOKED",
            }:
                continue
            if any(belongs(job, row) for row in active):
                continue
            if applies(job.get("partition"), job.get("account"), job.get("qos")):
                usage[name] += int(job.get("array_task_count") or 1)
    return usage


def capacity_plan(
    config: Config,
    tasks: list[dict],
    jobs: list,
    owned: list[dict],
    paused: set[str] | None = None,
    *,
    reservations: list[dict] | None = None,
) -> dict:
    """Greedy shared-backlog assignment; an eligible task is consumed only once.

    Each proposed worker covers one allocation lifetime of estimated demand.
    Existing pending/running workers consume capacity first. No free-node gate.
    """
    paused = paused or set()
    usage = shared_budget_usage(config, jobs, owned if reservations is None else reservations)
    remaining = sorted(tasks, key=lambda t: (-t.get("priority", 0), t.get("created_at", ""), t["task_id"]))
    profile_counts = Counter(
        x["profile"] for x in owned if x.get("state") in ("intent", "unknown", "submitted", "running")
    )

    def consume(profile_name, remaining_seconds=None):
        profile = config.profiles[profile_name]
        seconds = max(
            0,
            (profile.allocation_seconds if remaining_seconds is None else remaining_seconds)
            - config.policy.cleanup_seconds,
        )
        if seconds <= 0:
            return []
        slots = 1 if gpu_count(profile) else profile.cpus
        budget_seconds = seconds * slots
        used = 0.0
        picked = []
        for task in list(remaining):
            if task.get("allowed_profiles") and profile_name not in task["allowed_profiles"]:
                continue
            resources = Resources.model_validate(task["resources"])
            if compatible(resources, profile, config)[0]:
                duration = resources.runtime_seconds or config.policy.unknown_runtime_seconds
                if duration > seconds:
                    continue
                effective_cpus = max(
                    resources.cpus,
                    math.ceil(
                        resources.memory_mib
                        / max(1, profile.memory_mib - config.policy.worker_overhead_mib)
                        * profile.cpus
                    ),
                )
                cost = duration * (1 if gpu_count(profile) else effective_cpus)
                if picked and used + cost > budget_seconds:
                    continue
                picked.append(task["task_id"])
                remaining.remove(task)
                used += cost
                if used >= budget_seconds:
                    break
        return picked

    # Existing allocations retain their full conservative lifetime here; workers
    # expose actual remaining-time eligibility to HQ separately.
    for allocation in owned:
        if allocation.get("state") in ("intent", "unknown", "submitted", "running"):
            consume(allocation["profile"], allocation.get("remaining_seconds"))
    actions, reasons = [], []
    while remaining and len(actions) < config.policy.max_submissions_per_cycle:
        names = sorted(
            config.profiles,
            key=lambda n: (
                profile_counts[n] >= config.profiles[n].min_workers,
                profile_counts[n] / config.profiles[n].weight,
                n,
            ),
        )
        made = False
        for name in names:
            profile = config.profiles[name]
            if name in paused or profile_counts[name] >= profile.max_workers:
                continue
            if not profile.budget:
                reasons.append(f"{name}: NO_EXPLICIT_BUDGET")
                continue
            budget = config.budgets[profile.budget]
            cap = min(budget.max_submitted, budget.running_target + budget.pending_reserve)
            if usage[profile.budget] >= cap:
                continue
            picked = consume(name)
            if not picked:
                continue
            actions.append({"profile": name, "reason": "READY_SHARED_BACKLOG", "covered_task_ids": picked})
            usage[profile.budget] += 1
            profile_counts[name] += 1
            made = True
            break  # Recompute weighted shares after each allocation.
        if not made:
            break
    return {
        "actions": actions,
        "budget_usage_after_plan": dict(usage),
        "uncovered_tasks": len(remaining),
        "reasons": sorted(set(reasons)),
    }
