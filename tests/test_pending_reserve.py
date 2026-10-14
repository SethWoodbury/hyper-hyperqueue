from collections import Counter

import pytest
from pydantic import ValidationError

from hyper_hyperqueue.models import Budget, Config, Policy, Profile, Resources
from hyper_hyperqueue.planner import capacity_plan


def config(*, targets=None, cycle=100, cap=100):
    targets = targets or {"gpu": 20, "gpu-bf": 20, "gpu-train": 1}
    return Config(
        budgets={
            name: Budget(
                partitions=[name],
                min_pending=target,
                max_submitted=cap,
                running_target=6,
                pending_reserve=2,
            )
            for name, target in targets.items()
        },
        profiles={
            name: Profile(
                partition=name,
                budget=name,
                gres="gpu:1",
                memory_mib=2048,
                allocation_seconds=12000,
                max_workers=cap,
            )
            for name in targets
        },
        policy=Policy(max_submissions_per_cycle=cycle),
    )


def tasks(count=100, *, profiles=None, runtime=60, memory=1024):
    return [
        {
            "task_id": str(i),
            "resources": Resources(gpus=1, memory_mib=memory, runtime_seconds=runtime).model_dump(),
            "allowed_profiles": profiles or [],
        }
        for i in range(count)
    ]


def allocation(i, *, profile="gpu", state="PENDING"):
    intent = {
        "token": f"worker{i}",
        "backend": "worker",
        "job_id": str(i),
        "profile": profile,
        "state": "running" if state == "RUNNING" else "submitted",
        "remaining_seconds": 12000,
    }
    job = {"job_id": str(i), "partition": profile, "state": state, "comment": f"hhq:worker{i}"}
    return intent, job


def test_targets_are_distinct_partition_pending_floors():
    c = config()
    plan = capacity_plan(c, tasks(), [], [])
    counts = Counter(action["profile"] for action in plan["actions"])
    assert counts == {"gpu": 20, "gpu-bf": 20, "gpu-train": 1}
    ids = [tid for action in plan["actions"] for tid in action["covered_task_ids"]]
    assert len(ids) == len(set(ids)) == 100
    assert all(row["deficit_after_plan"] == 0 for row in plan["pending_reserves"].values())


def test_running_workers_do_not_suppress_pending_refill():
    c = config(targets={"gpu": 20})
    owned, jobs = zip(*(allocation(i, state="RUNNING") for i in range(8)))
    plan = capacity_plan(c, tasks(), list(jobs), list(owned))
    assert len(plan["actions"]) == 20  # Old total target is only 6+2.
    status = plan["pending_reserves"]["gpu"]
    assert status["observed_running"] == 8
    assert status["observed_pending"] == 0
    assert status["planned"] == 20


def test_observed_pending_is_maintained_as_jobs_start():
    c = config(targets={"gpu": 3})
    owned, jobs = zip(*(allocation(i) for i in range(3)))
    assert not capacity_plan(c, tasks(), list(jobs), list(owned))["actions"]
    owned, jobs = list(owned), list(jobs)
    owned[0], jobs[0] = allocation(0, state="RUNNING")
    plan = capacity_plan(c, tasks(), jobs, owned)
    assert len(plan["actions"]) == 1
    assert plan["pending_reserves"]["gpu"]["observed_pending"] == 2


@pytest.mark.parametrize("state", ["intent", "unknown", "submitted", "running"])
def test_unobserved_intents_reserve_slots_without_claiming_confirmed_pending(state):
    c = config(targets={"gpu": 2})
    owned = [{"token": "uncertain", "backend": "worker", "profile": "gpu", "state": state}]
    plan = capacity_plan(c, tasks(), [], owned)
    assert len(plan["actions"]) == 1
    status = plan["pending_reserves"]["gpu"]
    assert status["observed_pending"] == status["observed_running"] == 0
    assert status["uncertain_reservations"] == 1


def test_unknown_ack_reconciled_by_token_does_not_duplicate():
    c = config(targets={"gpu": 1})
    owned, job = allocation(1)
    owned.update(job_id=None, state="unknown")
    plan = capacity_plan(c, tasks(), [job], [owned])
    assert not plan["actions"]
    assert plan["pending_reserves"]["gpu"]["observed_pending"] == 1


def test_small_shared_backlog_limits_reserve_across_all_routes():
    plan = capacity_plan(config(), tasks(2), [], [])
    assert len(plan["actions"]) == 2
    ids = [tid for action in plan["actions"] for tid in action["covered_task_ids"]]
    assert len(ids) == len(set(ids)) == 2
    assert any("SHARED_WAITING_DEMAND_EXHAUSTED" in reason for reason in plan["reasons"])


def test_existing_pending_workers_claim_shared_demand_first():
    c = config()
    owned, job = allocation(1)
    plan = capacity_plan(c, tasks(1), [job], [owned])
    assert not plan["actions"]


def test_no_remaining_work_does_not_create_idle_workers():
    plan = capacity_plan(config(), [], [], [])
    assert not plan["actions"]
    assert plan["pending_reserves"]["gpu"]["deficit_after_plan"] == 20


def test_disabled_partition_and_incompatible_resources_do_not_create_workers():
    c = config()
    plan = capacity_plan(c, tasks(profiles=["gpu"]), [], [])
    assert {action["profile"] for action in plan["actions"]} == {"gpu"}
    assert "NO_COMPATIBLE_WAITING_TASK" in plan["pending_reserves"]["gpu-train"]["blocked_by"]
    assert not capacity_plan(c, tasks(memory=8192), [], [])["actions"]


def test_pauses_and_zero_profile_cap_block_reserve_with_reasons():
    c = config()
    c.profiles["gpu-bf"].max_workers = 0
    plan = capacity_plan(c, tasks(), [], [], paused={"gpu", "gpu-train"})
    assert not plan["actions"]
    assert "PROFILE_MAX_WORKERS" in plan["pending_reserves"]["gpu-bf"]["blocked_by"]
    assert "PROFILE_PAUSED_OR_GATED" in plan["pending_reserves"]["gpu"]["blocked_by"]


def test_bounded_cycles_share_progress_including_optional_train():
    plan = capacity_plan(config(cycle=4), tasks(), [], [])
    assert len(plan["actions"]) == 4
    assert {action["profile"] for action in plan["actions"]} == {"gpu", "gpu-bf", "gpu-train"}
    assert any("MAX_SUBMISSIONS_PER_CYCLE" in reason for reason in plan["reasons"])


def test_shared_profile_budget_does_not_multiply_floor():
    c = config(targets={"gpu": 20})
    c.profiles["gpu-large"] = c.profiles["gpu"].model_copy()
    c.profiles["gpu-beam"] = c.profiles["gpu"].model_copy()
    plan = capacity_plan(c, tasks(), [], [])
    assert len(plan["actions"]) == 20
    assert {action["profile"] for action in plan["actions"]} == {"gpu", "gpu-large", "gpu-beam"}


def test_max_submitted_remains_hard_when_running_jobs_use_headroom():
    c = config(targets={"gpu": 20}, cap=8)
    owned, jobs = zip(*(allocation(i, state="RUNNING") for i in range(6)))
    plan = capacity_plan(c, tasks(), list(jobs), list(owned))
    assert len(plan["actions"]) == 2
    status = plan["pending_reserves"]["gpu"]
    assert status["deficit_after_plan"] == 18
    assert status["blocked_by"] == ["PROFILE_MAX_WORKERS"]


def test_external_jobs_and_direct_arrays_constrain_caps_but_not_floor():
    c = config(targets={"gpu": 4}, cap=5)
    external = {"job_id": "external", "partition": "gpu", "state": "PENDING"}
    direct = {
        "backend": "direct",
        "token": "direct",
        "state": "unknown",
        "profile": "gpu",
        "elements": 3,
    }
    plan = capacity_plan(c, tasks(), [external], [], reservations=[direct])
    assert len(plan["actions"]) == 1
    status = plan["pending_reserves"]["gpu"]
    assert status["observed_pending"] == status["uncertain_reservations"] == 0
    assert status["deficit_after_plan"] == 3
    assert status["blocked_by"] == ["MAX_SUBMITTED:gpu"]


def test_overlapping_hard_budget_counts_all_proposed_routes():
    c = config()
    c.budgets["all-gpu"] = Budget(partitions=list(c.budgets), max_submitted=2)
    plan = capacity_plan(c, tasks(), [], [])
    assert len(plan["actions"]) == 2
    assert plan["budget_usage_after_plan"]["all-gpu"] == 2
    assert any("MAX_SUBMITTED:all-gpu" in reason for reason in plan["reasons"])


def test_overlapping_cap_counts_existing_external_and_direct_reservations():
    c = config()
    c.budgets["all-gpu"] = Budget(partitions=list(c.budgets), max_submitted=3)
    external = {"job_id": "external", "partition": "gpu-train", "state": "RUNNING"}
    direct = {"backend": "direct", "token": "direct", "state": "unknown", "profile": "gpu-bf"}
    plan = capacity_plan(c, tasks(), [external], [], reservations=[direct])
    assert len(plan["actions"]) == 1
    assert plan["budget_usage_after_plan"]["all-gpu"] == 3


def test_task_and_control_lifetime_checks_still_apply():
    c = config()
    assert not capacity_plan(c, tasks(runtime=12000), [], [])["actions"]
    c.control.walltime_seconds = 120
    assert not capacity_plan(c, tasks(runtime=90), [], [])["actions"]


def test_short_timeout_does_not_hide_hq_minimum_runtime_for_reserve():
    c = config(targets={"gpu": 2})
    requested = tasks(1, runtime=15000)
    requested[0]["resources"]["timeout_seconds"] = 1
    assert not capacity_plan(c, requested, [], [])["actions"]


def test_reserved_anchor_runtime_is_charged_to_its_worker():
    c = config(targets={"gpu": 2})
    c.profiles["gpu"].allocation_seconds = 3600
    plan = capacity_plan(c, tasks(3, runtime=2000), [], [])
    assert len(plan["actions"]) == 3
    assert [len(action["covered_task_ids"]) for action in plan["actions"]] == [1, 1, 1]
    assert plan["pending_reserves"]["gpu"]["planned"] == 3


def test_opt_out_preserves_original_demand_sizing_and_output():
    c = config(targets={"gpu": 0})
    plan = capacity_plan(c, tasks(10), [], [])
    assert len(plan["actions"]) == 1
    assert plan["actions"][0]["reason"] == "READY_SHARED_BACKLOG"
    assert "pending_reserves" not in plan


def test_pending_minimum_defaults_to_zero_and_cannot_be_negative():
    assert Budget().min_pending == 0
    with pytest.raises(ValidationError):
        Budget(min_pending=-1)


def priority_config(*, cycle=4):
    c = config(targets={"gpu": 8, "gpu-bf": 3, "gpu-train": 1}, cycle=cycle)
    c.policy.budget_priority = ["gpu", "gpu-train", "gpu-bf"]
    return c


def append_pending(plan, owned, jobs):
    for action in plan["actions"]:
        intent, job = allocation(len(owned), profile=action["profile"])
        owned.append(intent)
        jobs.append(job)


def test_priority_fills_eight_gpu_then_one_train_then_three_backfill_over_bounded_cycles():
    c = priority_config()
    owned, jobs = [], []
    expected = [["gpu"] * 4, ["gpu"] * 4, ["gpu-train", "gpu-bf", "gpu-bf", "gpu-bf"]]
    for profiles in expected:
        plan = capacity_plan(c, tasks(), jobs, owned)
        assert [action["profile"] for action in plan["actions"]] == profiles
        append_pending(plan, owned, jobs)
    final = capacity_plan(c, tasks(), jobs, owned)
    assert not final["actions"]
    assert {name: row["observed_pending"] for name, row in final["pending_reserves"].items()} == {
        "gpu": 8,
        "gpu-train": 1,
        "gpu-bf": 3,
    }


def test_priority_refills_gpu_first_after_several_queues_start():
    c = priority_config(cycle=100)
    owned, jobs = [], []
    append_pending(capacity_plan(c, tasks(), jobs, owned), owned, jobs)
    for index in (0, 1, 8, 9):
        profile = owned[index]["profile"]
        owned[index], jobs[index] = allocation(index, profile=profile, state="RUNNING")
    c.policy.max_submissions_per_cycle = 4
    plan = capacity_plan(c, tasks(), jobs, owned)
    assert [action["profile"] for action in plan["actions"]] == ["gpu", "gpu", "gpu-train", "gpu-bf"]


@pytest.mark.parametrize("blocked", ["paused", "budget_cap", "profile_cap", "ineligible"])
def test_priority_skips_blocked_regular_gpu_without_starving_lower_queues(blocked):
    c = priority_config()
    paused = set()
    requested = tasks()
    if blocked == "paused":
        paused.add("gpu")
    elif blocked == "budget_cap":
        c.budgets["gpu"].max_submitted = 0
    elif blocked == "profile_cap":
        c.profiles["gpu"].max_workers = 0
    else:
        requested = tasks(profiles=["gpu-train", "gpu-bf"])
    plan = capacity_plan(c, requested, [], [], paused)
    assert [action["profile"] for action in plan["actions"]] == ["gpu-train", "gpu-bf", "gpu-bf", "gpu-bf"]


def test_priority_skips_disabled_train_and_preserves_backfill_floor():
    plan = capacity_plan(priority_config(cycle=100), tasks(profiles=["gpu", "gpu-bf"]), [], [])
    assert [action["profile"] for action in plan["actions"]] == ["gpu"] * 8 + ["gpu-bf"] * 3


def test_priority_still_bounds_shared_demand_and_overlapping_hard_caps():
    c = priority_config()
    plan = capacity_plan(c, tasks(2), [], [])
    assert [action["profile"] for action in plan["actions"]] == ["gpu", "gpu"]
    assert {tid for action in plan["actions"] for tid in action["covered_task_ids"]} == {"0", "1"}
    c.budgets["all-gpu"] = Budget(partitions=list(c.budgets), max_submitted=2)
    plan = capacity_plan(c, tasks(), [], [])
    assert [action["profile"] for action in plan["actions"]] == ["gpu", "gpu"]
    assert plan["budget_usage_after_plan"]["all-gpu"] == 2


def test_priority_applies_to_normal_demand_growth_without_pending_floors():
    c = priority_config()
    for budget in c.budgets.values():
        budget.min_pending = 0
    for profile in c.profiles.values():
        profile.allocation_seconds = 3600
    requested = tasks(runtime=3500)
    plan = capacity_plan(c, requested, [], [])
    assert [action["profile"] for action in plan["actions"]] == ["gpu"] * 4
    plan = capacity_plan(c, requested, [], [], paused={"gpu"})
    assert [action["profile"] for action in plan["actions"]] == ["gpu-train"] * 4


def test_priority_is_configurable_for_arbitrary_budget_names():
    c = config(targets={"queue-a": 2, "queue-z": 2})
    c.policy.budget_priority = ["queue-z", "queue-a"]
    plan = capacity_plan(c, tasks(), [], [])
    assert [action["profile"] for action in plan["actions"]] == ["queue-z"] * 2 + ["queue-a"] * 2
    assert Policy().budget_priority == []
