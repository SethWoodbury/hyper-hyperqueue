"""Task preference must also work when workloads have disjoint worker profiles."""

import pytest

from hyper_hyperqueue.models import Budget, Config, Policy, Profile, Resources
from hyper_hyperqueue.planner import capacity_plan


def fixture(pending):
    config = Config(
        budgets={"gpu": Budget(partitions=["gpu"], min_pending=pending,
                               running_target=8, max_submitted=8)},
        profiles={name: Profile(partition="gpu", budget="gpu", gres="gpu:1",
                                memory_mib=4096, allocation_seconds=3600)
                  for name in ["beam", "gradient"]},
        policy=Policy(max_submissions_per_cycle=1),
    )
    tasks = [{"task_id": name, "priority": priority, "allowed_profiles": [name],
              "resources": Resources(gpus=1, memory_mib=2048, runtime_seconds=600).model_dump()}
             for name, priority in [("gradient", 3), ("beam", 2)]]
    return config, tasks


@pytest.mark.parametrize("pending", [0, 1])
def test_higher_priority_disjoint_profile_precedes_alphabetical_order(pending):
    config, tasks = fixture(pending)
    actions = capacity_plan(config, tasks, [], [])["actions"]
    assert [action["profile"] for action in actions] == ["gradient"]
    assert actions[0]["reason"] == ("MIN_PENDING_RESERVE" if pending else "READY_SHARED_BACKLOG")


@pytest.mark.parametrize("pending", [0, 1])
def test_zero_priorities_preserve_legacy_profile_tiebreaking(pending):
    config, tasks = fixture(pending)
    for task in tasks:
        task["priority"] = 0
    assert capacity_plan(config, tasks, [], [])["actions"][0]["profile"] == "beam"


@pytest.mark.parametrize("pending", [0, 1])
def test_negative_priorities_are_ordered_normally(pending):
    config, tasks = fixture(pending)
    tasks[0]["priority"], tasks[1]["priority"] = -1, -5
    assert capacity_plan(config, tasks, [], [])["actions"][0]["profile"] == "gradient"


@pytest.mark.parametrize("pending", [0, 1])
@pytest.mark.parametrize("block", ["paused", "capped", "memory", "runtime"])
def test_ineligible_higher_priority_does_not_block_lower_priority(pending, block):
    config, tasks = fixture(pending)
    paused = set()
    if block == "paused":
        paused.add("gradient")
    elif block == "capped":
        config.profiles["gradient"].max_workers = 0
    elif block == "memory":
        tasks[0]["resources"]["memory_mib"] = 8192
    else:
        tasks[0]["resources"].update(runtime_seconds=7200, timeout_seconds=1)
    assert capacity_plan(config, tasks, [], [], paused=paused)["actions"][0]["profile"] == "beam"


@pytest.mark.parametrize("pending", [0, 1])
def test_partition_priority_remains_above_workload_priority(pending):
    config, tasks = fixture(pending)
    config.budgets["gpu-bf"] = Budget(partitions=["gpu-bf"], min_pending=pending,
                                     running_target=8, max_submitted=8)
    config.profiles["gradient"].budget = "gpu-bf"
    config.profiles["gradient"].partition = "gpu-bf"
    config.policy.budget_priority = ["gpu", "gpu-bf"]
    assert capacity_plan(config, tasks, [], [])["actions"][0]["profile"] == "beam"


def test_existing_worker_can_cover_high_priority_and_next_profile_can_fill_capacity():
    config, tasks = fixture(0)
    owned = [{"backend": "worker", "profile": "gradient", "state": "running",
              "job_id": "existing", "remaining_seconds": 1800, "token": "existing-token"}]
    jobs = [{"job_id": "existing", "partition": "gpu", "state": "RUNNING",
             "comment": "hhq:existing-token"}]
    plan = capacity_plan(config, tasks, jobs, owned)
    assert plan["actions"][0]["profile"] == "beam"


@pytest.mark.parametrize("pending", [0, 1])
def test_profile_preference_recomputes_after_high_priority_work_is_covered(pending):
    config, tasks = fixture(pending)
    config.policy.max_submissions_per_cycle = 2
    actions = capacity_plan(config, tasks, [], [])["actions"]
    assert [action["profile"] for action in actions] == ["gradient", "beam"]
    covered = [task_id for action in actions for task_id in action["covered_task_ids"]]
    assert covered == ["gradient", "beam"]
