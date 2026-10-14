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


def fits_remaining_control(task, config, remaining_seconds):
    """Both the dispatch estimate and any hard timeout must fit this generation."""
    if remaining_seconds is None:
        return True
    resources = Resources.model_validate(task["resources"])
    duration = max(resources.runtime_seconds or config.policy.unknown_runtime_seconds,
                   resources.timeout_seconds or 0)
    return duration + config.policy.cleanup_seconds <= remaining_seconds


def _budget_applies(budget, placement):
    return all(
        not expected or not placement.get(key) or placement[key] in expected
        for key, expected in (
            ("partition", budget.partitions),
            ("account", [budget.account] if budget.account else []),
            ("qos", [budget.qos] if budget.qos else []),
        )
    )


def _profile_budgets(config, profile):
    placement = {key: getattr(profile, key) for key in ("partition", "account", "qos")}
    # The explicitly assigned budget remains binding even if its scope is misconfigured.
    return {
        name
        for name, budget in config.budgets.items()
        if name == profile.budget or _budget_applies(budget, placement)
    }


def _budget_priority(config, name):
    try:
        return config.policy.budget_priority.index(name)
    except ValueError:
        return len(config.policy.budget_priority)


def _positive_worker_job(job, allocation):
    token = allocation.get("token")
    return bool(token) and (
        job.get("comment") == "hhq:" + token
        or (not job.get("comment") and str(job.get("name", "")).endswith("-" + token))
    ) and (allocation.get("job_id") is None or str(job["job_id"]) == str(allocation["job_id"]))


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
        if config.policy.speculative_partition_workers and reservation.get("backend") == "worker":
            return _positive_worker_job(job, reservation)
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
            return _budget_applies(budget, {"partition": partition, "account": account, "qos": qos})

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


def _profile_waiting_priority(config, profile_name, ordered_tasks):
    """Highest eligible waiting priority; callers supply descending-priority tasks.

    Task priorities must influence allocation choice as well as HQ dispatch:
    disjoint resource profiles cannot compete on the same existing HQ worker.
    Zero-priority backlogs retain the previous weighted profile ordering.
    """
    if not ordered_tasks or (
        ordered_tasks[0].get("priority", 0) == ordered_tasks[-1].get("priority", 0) == 0
    ):
        return 0  # No extra resource parsing for the default all-zero policy.
    profile = config.profiles[profile_name]
    for task in ordered_tasks:
        if task.get("allowed_profiles") and profile_name not in task["allowed_profiles"]:
            continue
        resources = Resources.model_validate(task["resources"])
        runtime = resources.runtime_seconds or config.policy.unknown_runtime_seconds
        if (
            compatible(resources, profile, config)[0]
            and runtime + config.policy.cleanup_seconds <= profile.allocation_seconds
            and (
                config.control.hosting != "slurm"
                or runtime + config.policy.cleanup_seconds <= config.control.walltime_seconds
            )
        ):
            return task.get("priority", 0)
    return 0


def _pending_plan(config, tasks, jobs, owned, paused, usage, profile_counts):
    """Reserve distinct waiting-task anchors, independent of running-worker lifetimes.

    These anchors justify allocations; they do not bind HQ dispatch. Unacknowledged
    allocations occupy pending slots until reconciliation proves their outcome.
    External jobs and direct arrays constrain hard budgets but cannot supply this
    service's worker reserve. Optional cross-partition speculation shares waiting
    anchors between partitions, never task ownership or execution attempts.
    """
    targets = {
        name: b.min_pending
        for name, b in config.budgets.items()
        if b.min_pending or b.fill_running_target or b.max_pending is not None
    }
    if not targets:
        return [], {}, []
    observed = [job_dict(job) for job in jobs]
    pending, uncertain, running, planned = Counter(), Counter(), Counter(), Counter()
    existing_reservations = []
    demand = list(tasks)
    partition_demand = {profile.partition: list(tasks) for profile in config.profiles.values()}

    def waiting(profile_name):
        if config.policy.speculative_partition_workers:
            return partition_demand[config.profiles[profile_name].partition]
        return demand

    def reserve(profile_name, task, *, unknown=False):
        pools = list(partition_demand.values()) if unknown and config.policy.speculative_partition_workers else [waiting(profile_name)]
        for pool in pools:
            if task in pool:
                pool.remove(task)

    def anchor(profile_name):
        profile = config.profiles[profile_name]
        for task in waiting(profile_name):
            if task.get("allowed_profiles") and profile_name not in task["allowed_profiles"]:
                continue
            resources = Resources.model_validate(task["resources"])
            runtime = resources.runtime_seconds or config.policy.unknown_runtime_seconds
            if (
                compatible(resources, profile, config)[0]
                and runtime + config.policy.cleanup_seconds <= profile.allocation_seconds
                and (
                    config.control.hosting != "slurm"
                    or runtime + config.policy.cleanup_seconds <= config.control.walltime_seconds
                )
            ):
                return task
        return None

    for allocation in owned:
        if allocation.get("state") not in ACTIVE_INTENT_STATES:
            continue
        profile_name = allocation["profile"]
        profile = config.profiles[profile_name]
        matches = [
            job
            for job in observed
            if (allocation.get("job_id") is not None and str(job["job_id"]) == str(allocation["job_id"]))
            or (allocation.get("token") and job.get("comment") == "hhq:" + allocation["token"])
        ]
        states = {str(job.get("state") or "UNKNOWN").split()[0].rstrip("+") for job in matches}
        if config.policy.speculative_partition_workers and not all(_positive_worker_job(job, allocation) for job in matches):
            states = {"UNKNOWN"}  # Job-number reuse cannot relax cross-partition reservations.
        if "RUNNING" in states:
            counts = running
        elif states == {"PENDING"}:
            counts = pending
        else:
            counts = uncertain
        placement = allocation.get("placement") or (
            {key: matches[0].get(key) for key in ("partition", "account", "qos")}
            if matches
            else {key: getattr(profile, key) for key in ("partition", "account", "qos")}
        )
        for name in targets:
            if _budget_applies(config.budgets[name], placement):
                counts[name] += 1
        if counts is not running:
            existing_reservations.append((profile_name, counts is uncertain))

    # Resolve uncertain ownership conservatively before known partition-specific
    # pending workers can exhaust its anchor pool. Unknown reservations bind all
    # partitions; they never become an excuse for speculative duplicate capacity.
    ordered_reservations = (
        sorted(existing_reservations, key=lambda item: not item[1])
        if config.policy.speculative_partition_workers else existing_reservations
    )
    for profile_name, unknown in ordered_reservations:
        task = anchor(profile_name)
        if task is not None:
            reserve(profile_name, task, unknown=unknown)

    # Optional eager growth is an operator's concurrency target, not a forecast
    # of free nodes. Each pending worker still needs distinct waiting work.
    for name in targets:
        budget = config.budgets[name]
        if budget.fill_running_target:
            targets[name] = max(budget.min_pending, budget.running_target - running[name])

    def slots(name):
        return pending[name] + uncertain[name] + planned[name]

    def profiles(name):
        return sorted(
            (n for n, p in config.profiles.items() if p.budget == name),
            key=lambda n: (
                profile_counts[n] >= config.profiles[n].min_workers,
                -_profile_waiting_priority(config, n, waiting(n)),
                profile_counts[n] / config.profiles[n].weight,
                n,
            ),
        )

    def blockers(name):
        blocked = set()
        names = profiles(name)
        if not names:
            blocked.add("NO_ELIGIBLE_PROFILES")
        for profile_name in names:
            profile = config.profiles[profile_name]
            if profile_name in paused:
                blocked.add("PROFILE_PAUSED_OR_GATED")
                continue
            if profile_counts[profile_name] >= profile.max_workers:
                blocked.add("PROFILE_MAX_WORKERS")
                continue
            full = [
                n for n in _profile_budgets(config, profile) if usage[n] >= config.budgets[n].max_submitted
            ]
            if full:
                blocked.update(f"MAX_SUBMITTED:{n}" for n in full)
                continue
            pending_full = [n for n in _profile_budgets(config, profile)
                            if config.budgets[n].max_pending is not None
                            and slots(n) >= config.budgets[n].max_pending]
            if pending_full:
                blocked.update(f"MAX_PENDING:{n}" for n in pending_full)
                continue
            if anchor(profile_name) is None:
                blocked.add("SHARED_WAITING_DEMAND_EXHAUSTED" if not waiting(profile_name) else "NO_COMPATIBLE_WAITING_TASK")
                continue
            return profile_name, blocked
        return None, blocked

    actions = []
    while len(actions) < config.policy.max_submissions_per_cycle:
        made = False
        # Optional two-phase admission grows each route before replenishing reserves.
        # A capped/incompatible growth route yields immediately to the next route.
        phases = ("growth", "reserve") if config.policy.pending_reserves_after_growth else ("combined",)
        selected = None
        for phase in phases:
            for name in sorted(targets, key=lambda n: (_budget_priority(config, n), slots(n) / max(1, targets[n]), n)):
                target = targets[name]
                if phase == "growth":
                    target = (max(0, config.budgets[name].running_target - running[name])
                              if config.budgets[name].fill_running_target else 0)
                if slots(name) >= target:
                    continue
                profile_name, _ = blockers(name)
                if profile_name is not None:
                    selected = name, profile_name, phase
                    break
            if selected is not None:
                break
        if selected is not None:
            name, profile_name, phase = selected
            profile = config.profiles[profile_name]
            task = anchor(profile_name)
            reserve(profile_name, task)
            actions.append(
                {
                    "profile": profile_name,
                    "reason": (
                        "FILL_RUNNING_TARGET"
                        if phase == "growth" or (phase == "combined"
                        and config.budgets[name].fill_running_target
                        and slots(name) >= config.budgets[name].min_pending)
                        else "MIN_PENDING_RESERVE"
                    ),
                    "covered_task_ids": [task["task_id"]],
                }
            )
            profile_counts[profile_name] += 1
            for budget_name in _profile_budgets(config, profile):
                usage[budget_name] += 1
                planned[budget_name] += 1
            made = True
        if not made:
            break

    status, reasons = {}, []
    for name, target in targets.items():
        deficit = max(0, target - slots(name))
        blocked = set()
        if deficit:
            available, blocked = blockers(name)
            if available is not None and len(actions) >= config.policy.max_submissions_per_cycle:
                blocked.add("MAX_SUBMISSIONS_PER_CYCLE")
            label = "WORKER_TARGET_DEFICIT" if config.budgets[name].fill_running_target else "MIN_PENDING_DEFICIT"
            reasons.extend(f"{name}: {label} {deficit} ({reason})" for reason in sorted(blocked))
        status[name] = {
            "target": target,
            "observed_pending": pending[name],
            "uncertain_reservations": uncertain[name],
            "observed_running": running[name],
            "planned": planned[name],
            "reserved_after_plan": slots(name),
            "deficit_after_plan": deficit,
            "blocked_by": sorted(blocked),
            "max_pending": config.budgets[name].max_pending,
        }
        if config.budgets[name].fill_running_target:
            status[name].update(
                fill_running_target=True,
                minimum_pending_target=config.budgets[name].min_pending,
                running_target=config.budgets[name].running_target,
            )
    return actions, status, reasons


def capacity_plan(
    config: Config,
    tasks: list[dict],
    jobs: list,
    owned: list[dict],
    paused: set[str] | None = None,
    *,
    reservations: list[dict] | None = None,
    remaining_control_seconds: float | None = None,
) -> dict:
    """Greedy allocation planning over one shared execution backlog.

    Each proposed worker covers one allocation lifetime of estimated demand.
    Existing pending/running workers consume capacity first. No free-node gate.
    Opt-in speculative reserves may cover the same waiting task in different
    partitions; HQ still dispatches each task once, and surplus workers idle out.
    """
    paused = paused or set()
    usage = shared_budget_usage(config, jobs, owned if reservations is None else reservations)
    eligible = [t for t in tasks if fits_remaining_control(t, config, remaining_control_seconds)]
    blocked_control = len(tasks) - len(eligible)
    remaining = sorted(eligible, key=lambda t: (-t.get("priority", 0), t.get("created_at", ""), t["task_id"]))
    profile_counts = Counter(
        x["profile"] for x in owned if x.get("state") in ("intent", "unknown", "submitted", "running")
    )
    actions, pending_status, reasons = _pending_plan(
        config, remaining, jobs, owned, paused, usage, profile_counts
    )
    if blocked_control:
        reasons.append(f"CONTROL_REMAINING_TOO_SHORT: {blocked_control} waiting task(s) deferred")
    reserved_ids = {tid for action in actions for tid in action["covered_task_ids"]}
    remaining = [task for task in remaining if task["task_id"] not in reserved_ids]

    def consume(profile_name, remaining_seconds=None, reserved=()):
        profile = config.profiles[profile_name]
        lifetime = profile.allocation_seconds if remaining_seconds is None else remaining_seconds
        if remaining_control_seconds is not None:
            lifetime = min(lifetime, remaining_control_seconds)
        seconds = max(
            0,
            lifetime - config.policy.cleanup_seconds,
        )
        if seconds <= 0:
            return []
        slots = 1 if gpu_count(profile) else profile.cpus
        budget_seconds = seconds * slots

        def task_cost(task):
            resources = Resources.model_validate(task["resources"])
            duration = resources.runtime_seconds or config.policy.unknown_runtime_seconds
            effective_cpus = max(
                resources.cpus,
                math.ceil(
                    resources.memory_mib
                    / max(1, profile.memory_mib - config.policy.worker_overhead_mib)
                    * profile.cpus
                ),
            )
            return duration * (1 if gpu_count(profile) else effective_cpus)

        used = sum(task_cost(task) for task in reserved)
        picked = [task["task_id"] for task in reserved]
        for task in list(remaining):
            if task.get("allowed_profiles") and profile_name not in task["allowed_profiles"]:
                continue
            resources = Resources.model_validate(task["resources"])
            if compatible(resources, profile, config)[0]:
                duration = resources.runtime_seconds or config.policy.unknown_runtime_seconds
                if duration > seconds:
                    continue
                cost = task_cost(task)
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
    task_by_id = {task["task_id"]: task for task in tasks}
    for action in actions:
        reserved = [task_by_id[tid] for tid in action["covered_task_ids"]]
        action["covered_task_ids"] = consume(action["profile"], reserved=reserved)
    while remaining and len(actions) < config.policy.max_submissions_per_cycle:
        names = sorted(
            config.profiles,
            key=lambda n: (
                _budget_priority(config, config.profiles[n].budget),
                profile_counts[n] >= config.profiles[n].min_workers,
                -_profile_waiting_priority(config, n, remaining),
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
            scopes = _profile_budgets(config, profile) if pending_status else {profile.budget}
            if any(usage[n] >= config.budgets[n].max_submitted for n in scopes):
                continue
            if any(config.budgets[n].max_pending is not None
                   and pending_status.get(n, {}).get("reserved_after_plan", 0) >= config.budgets[n].max_pending
                   for n in scopes):
                continue
            picked = consume(name)
            if not picked:
                continue
            actions.append({"profile": name, "reason": "READY_SHARED_BACKLOG", "covered_task_ids": picked})
            for scope in scopes:
                usage[scope] += 1
                if scope in pending_status:
                    pending_status[scope]["reserved_after_plan"] += 1
            profile_counts[name] += 1
            made = True
            break  # Recompute weighted shares after each allocation.
        if not made:
            break
    result = {
        "actions": actions,
        "budget_usage_after_plan": dict(usage),
        "uncovered_tasks": len(remaining) + blocked_control,
        "reasons": sorted(set(reasons)),
    }
    if remaining_control_seconds is not None:
        result.update(control_remaining_seconds=max(0, remaining_control_seconds),
                      control_remaining_blocked_tasks=blocked_control)
    if config.policy.speculative_partition_workers:
        covered = [tid for action in actions for tid in action["covered_task_ids"]]
        result.update(speculative_partition_workers=True,
                      speculative_anchor_reuses=len(covered) - len(set(covered)))
    if pending_status:
        for name, status in pending_status.items():
            status["planned"] = sum(
                name in _profile_budgets(config, config.profiles[action["profile"]]) for action in actions
            )
            status["reserved_after_plan"] = (
                status["observed_pending"] + status["uncertain_reservations"] + status["planned"]
            )
            status["deficit_after_plan"] = max(0, status["target"] - status["reserved_after_plan"])
            if not status["deficit_after_plan"]:
                status["blocked_by"] = []
                result["reasons"] = [
                    r for r in result["reasons"]
                    if not r.startswith((f"{name}: MIN_PENDING", f"{name}: WORKER_TARGET"))
                ]
        result["pending_reserves"] = pending_status
    return result
