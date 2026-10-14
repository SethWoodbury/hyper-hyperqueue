"""Optional eager worker growth uses declared targets, never node forecasts."""

import json
from collections import Counter
from datetime import UTC, datetime

import pytest

from hyper_hyperqueue import controller
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Budget, Config, Policy, Profile, Resources
from hyper_hyperqueue.planner import capacity_plan, shared_budget_usage
from hyper_hyperqueue.state import StateStore, atomic_json


def config(*, eager=True, floor=8, target=200, cap=208, burst=200):
    return Config(
        budgets={"gpu": Budget(partitions=["gpu"], running_target=target,
                               fill_running_target=eager, min_pending=floor,
                               pending_reserve=floor, max_submitted=cap)},
        profiles={"gpu": Profile(partition="gpu", budget="gpu", gres="gpu:1", cpus=2,
                                 memory_mib=18688, allocation_seconds=28800, max_workers=cap)},
        policy=Policy(max_submissions_per_cycle=burst, submit_cooldown_seconds=0),
    )


def tasks(count=500):
    return [{"task_id": str(i), "resources": Resources(gpus=1, cpus=2, memory_mib=18432,
             runtime_seconds=6000 if i < 24 else 9000).model_dump()} for i in range(count)]


def allocation(i, state="RUNNING"):
    owned = {"backend": "worker", "token": f"worker{i}", "job_id": str(i), "profile": "gpu",
             "state": "running" if state == "RUNNING" else "submitted", "remaining_seconds": 28800}
    job = {"job_id": str(i), "partition": "gpu", "state": state, "comment": f"hhq:worker{i}"}
    return owned, job


def assert_unique_coverage(plan):
    covered = [tid for action in plan["actions"] for tid in action["covered_task_ids"]]
    assert len(covered) == len(set(covered))


def test_eager_200_worker_burst_is_not_lifetime_packed():
    c = config()
    plan = capacity_plan(c, tasks(), [], [])
    assert len(plan["actions"]) == 200
    assert Counter(a["reason"] for a in plan["actions"]) == {
        "MIN_PENDING_RESERVE": 8, "FILL_RUNNING_TARGET": 192}
    assert_unique_coverage(plan)
    assert plan["pending_reserves"]["gpu"]["target"] == 200
    assert plan["pending_reserves"]["gpu"]["minimum_pending_target"] == 8


def test_default_retains_demand_packing_and_184_bundles_cannot_use_200_workers():
    assert Budget().fill_running_target is False
    assert len(capacity_plan(config(eager=False), tasks(184), [], [])["actions"]) == 61
    eager = capacity_plan(config(), tasks(184), [], [])
    assert len(eager["actions"]) == 184
    assert eager["pending_reserves"]["gpu"]["deficit_after_plan"] == 16
    assert "SHARED_WAITING_DEMAND_EXHAUSTED" in eager["pending_reserves"]["gpu"]["blocked_by"]
    assert_unique_coverage(eager)
    assert not capacity_plan(config(), [], [], [])["actions"]


def test_partial_running_and_pending_are_counted_separately():
    owned, jobs = zip(*(allocation(i, "RUNNING" if i < 40 else "PENDING") for i in range(45)))
    plan = capacity_plan(config(), tasks(), list(jobs), list(owned))
    assert len(plan["actions"]) == 155
    status = plan["pending_reserves"]["gpu"]
    assert (status["observed_running"], status["observed_pending"], status["target"]) == (40, 5, 160)
    assert status["deficit_after_plan"] == 0
    assert_unique_coverage(plan)


@pytest.mark.parametrize("floor,expected", [(8, 8), (0, 0)])
def test_running_target_met_keeps_only_pending_floor(floor, expected):
    owned, jobs = zip(*(allocation(i) for i in range(200)))
    plan = capacity_plan(config(floor=floor), tasks(), list(jobs), list(owned))
    assert len(plan["actions"]) == expected
    assert plan["pending_reserves"]["gpu"]["target"] == floor


def test_unknown_submissions_reserve_capacity_without_claiming_running():
    owned = [{"backend": "worker", "token": f"unknown{i}", "profile": "gpu", "state": "unknown"}
             for i in range(20)]
    plan = capacity_plan(config(), tasks(), [], owned)
    assert len(plan["actions"]) == 180
    assert plan["pending_reserves"]["gpu"]["uncertain_reservations"] == 20
    assert plan["pending_reserves"]["gpu"]["observed_running"] == 0
    assert_unique_coverage(plan)


@pytest.mark.parametrize("block,expected", [("external", 198), ("profile", 17), ("paused", 0), ("burst", 4)])
def test_eager_growth_respects_existing_limits(block, expected):
    c, jobs, paused = config(), [], set()
    if block == "external":
        jobs = [{"job_id": f"external{i}", "partition": "gpu", "state": "RUNNING"} for i in range(10)]
    elif block == "profile":
        c.profiles["gpu"].max_workers = 17
    elif block == "paused":
        paused.add("gpu")
    else:
        c.policy.max_submissions_per_cycle = 4
    plan = capacity_plan(c, tasks(), jobs, [], paused)
    assert len(plan["actions"]) == expected
    assert_unique_coverage(plan)


def test_three_route_priority_and_shared_demand():
    c = config(burst=500)
    for name, target, floor, cap in [("gpu-train", 2, 1, 3), ("gpu-bf", 200, 3, 203)]:
        c.budgets[name] = Budget(partitions=[name], running_target=target, fill_running_target=True,
                                 min_pending=floor, max_submitted=cap)
        c.profiles[name] = c.profiles["gpu"].model_copy(update={"partition": name, "budget": name,
                                                               "max_workers": cap})
    c.policy.budget_priority = ["gpu", "gpu-train", "gpu-bf"]
    plan = capacity_plan(c, tasks(600), [], [])
    assert [a["profile"] for a in plan["actions"]] == ["gpu"] * 200 + ["gpu-train"] * 2 + ["gpu-bf"] * 200
    assert_unique_coverage(plan)
    c.budgets["all-gpu"] = Budget(partitions=["gpu", "gpu-train", "gpu-bf"], max_submitted=210)
    plan = capacity_plan(c, tasks(600), [], [])
    assert [a["profile"] for a in plan["actions"]] == ["gpu"] * 200 + ["gpu-train"] * 2 + ["gpu-bf"] * 8
    assert_unique_coverage(plan)
    c.policy.max_submissions_per_cycle = 200
    for count in (96, 184):
        small = capacity_plan(c, tasks(count), [], [])
        assert [action["profile"] for action in small["actions"]] == ["gpu"] * count


def test_submission_cooldown_still_gates_eager_growth(tmp_path):
    c = config()
    c.policy.submit_cooldown_seconds = 30
    owned, job = allocation(0, "PENDING")
    now = datetime.now(UTC).timestamp()
    owned["created_at"] = datetime.fromtimestamp(now, UTC).isoformat()
    paused, reasons = controller.allocation_gates(StateStore(tmp_path), c, [owned], clock=now)
    assert paused == {"gpu"} and reasons == ["gpu: SUBMISSION_COOLDOWN"]
    assert not capacity_plan(c, tasks(), [job], [owned], paused)["actions"]


@pytest.mark.parametrize("unknown_at,expire_after", [(None, None), (12, None), (None, 3)])
def test_controller_submits_burst_without_per_worker_sleep_or_queue_poll(
    tmp_path, monkeypatch, unknown_at, expire_after,
):
    from hyper_hyperqueue.service.leadership import LeadershipGuard

    c = config()
    c.state_dir = str(tmp_path / "state")
    store = StateStore(c.state_dir)
    source = tmp_path / "commands.txt"
    source.write_text("true\n" * 200)
    ingest_file(store, source, mode="pooled", cwd=tmp_path,
                resources=Resources(gpus=1, cpus=2, memory_mib=18432, runtime_seconds=6000).model_dump())
    monkeypatch.setattr(controller, "admit_hq", lambda *args: True)
    monkeypatch.setattr(controller, "reconcile_hq", lambda *args: None)
    monkeypatch.setattr(controller, "retry_ready_failures", lambda *args: [])
    monkeypatch.setattr(controller.time, "sleep", lambda *args: pytest.fail("No sleep inside a submission burst"))
    clock = [controller.time.time()]
    monkeypatch.setattr(controller.time, "time", lambda: clock[0])

    class Slurm:
        def __init__(self):
            self.queries = 0
            self.submissions = []

        def query_jobs(self):
            self.queries += 1
            return []

        def submit(self, script, *, token, **kwargs):
            assert any(i["token"] == token and i["state"] == "intent" for i in store.intents())
            self.submissions.append(token)
            if len(self.submissions) == expire_after:
                clock[0] += 1000
            status = "unknown" if len(self.submissions) == unknown_at else "accepted"
            return {"status": status, "job_id": str(len(self.submissions)) if status == "accepted" else None}

    slurm = Slurm()
    guard = LeadershipGuard(store.root / "service").acquire()
    guard.heartbeat("ACTIVE")
    monkeypatch.setenv("HHQ_LEADER_FD", str(guard.fd))
    atomic_json(store.root / "service/endpoint.json", {
        "epoch": guard.epoch, "status": "ACTIVE", "control_end_time": clock[0] + 7000,
    })
    try:
        result = controller.tick(c, epoch=guard.epoch, slurm=slurm, hq=object())
    finally:
        guard.close(children_quiescent=True)
    assert slurm.queries == 1
    assert len(slurm.submissions) == (unknown_at or expire_after or 200)
    assert len(set(slurm.submissions)) == len(slurm.submissions)
    assert len(store.intents()) == len(slurm.submissions)
    if expire_after:
        assert "CONTROL_REMAINING_TOO_SHORT: stopped worker submission burst" in result["plan"]["reasons"]


def test_control_remaining_defers_long_bundles_but_admits_shorter_waiting_work():
    c = config()
    work = tasks(184)
    plan = capacity_plan(c, work, [], [], remaining_control_seconds=3600)
    assert not plan["actions"]
    assert plan["uncovered_tasks"] == plan["control_remaining_blocked_tasks"] == 184
    assert any(r.startswith("CONTROL_REMAINING_TOO_SHORT") for r in plan["reasons"])
    short = {"task_id": "short", "resources": Resources(gpus=1, runtime_seconds=300).model_dump()}
    plan = capacity_plan(c, work + [short], [], [], remaining_control_seconds=3600)
    assert [a["covered_task_ids"] for a in plan["actions"]] == [["short"]]
    assert plan["uncovered_tasks"] == 184
    assert c.control.walltime_seconds == Config().control.walltime_seconds
    assert c.profiles["gpu"].allocation_seconds == 28800  # No frozen request mutation.


def test_control_remaining_also_caps_lifetime_packing():
    work = [{"task_id": str(i), "resources": Resources(gpus=1, runtime_seconds=1000).model_dump()}
            for i in range(10)]
    c = config(eager=False, floor=0)
    assert len(capacity_plan(c, work, [], [])["actions"]) == 1
    limited = capacity_plan(c, work, [], [], remaining_control_seconds=2060)
    assert len(limited["actions"]) == 5
    assert all(len(a["covered_task_ids"]) == 2 for a in limited["actions"])
    assert_unique_coverage(limited)


@pytest.mark.parametrize("runtime,timeout,remaining,expected", [
    (300, None, 360, 1), (300, None, 359, 0), (100, 400, 400, 0),
    (None, None, 959, 0), (None, None, 960, 1), (300, None, -1, 0),
])
def test_remaining_time_checks_runtime_timeout_unknown_and_cleanup(runtime, timeout, remaining, expected):
    c = config()
    c.policy.unknown_runtime_seconds = 900
    work = [{"task_id": "t", "resources": Resources(
        gpus=1, runtime_seconds=runtime, timeout_seconds=timeout).model_dump()}]
    assert len(capacity_plan(c, work, [], [], remaining_control_seconds=remaining)["actions"]) == expected


@pytest.mark.parametrize("endpoint", [None, {"epoch": 0, "status": "ACTIVE", "control_end_time": 10000},
    {"epoch": 1, "status": "DRAINING", "control_end_time": 10000},
    {"epoch": 1, "status": "STOPPED", "control_end_time": 10000},
    {"epoch": 1, "status": "ACTIVE", "control_end_time": None},
    {"epoch": 1, "status": "ACTIVE", "control_end_time": "10000"},
    {"epoch": 1, "status": "ACTIVE", "control_end_time": float("nan")},
])
def test_slurm_controller_deadline_requires_current_finite_endpoint(tmp_path, endpoint):
    store = StateStore(tmp_path)
    if endpoint is not None:
        path = store.root / "service/endpoint.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(endpoint))  # Include malformed external NaN evidence.
    with pytest.raises(RuntimeError, match="CONTROL_ENDPOINT_NOT_ACTIVE|CONTROL_REMAINING_UNKNOWN"):
        controller.control_deadline(store, config(), 1)


def test_local_or_direct_controller_deadline_compatibility(tmp_path):
    store, c = StateStore(tmp_path), config()
    assert controller.control_deadline(store, c, None) is None
    c.control.hosting = "local"
    assert controller.control_deadline(store, c, 1) is None
    atomic_json(store.root / "service/endpoint.json", {
        "epoch": 1, "status": "ACTIVE", "control_end_time": None,
    })
    assert controller.control_deadline(store, c, 1) is None
    atomic_json(store.root / "service/endpoint.json", {
        "epoch": 1, "status": "ACTIVE", "control_end_time": 12345,
    })
    assert controller.control_deadline(store, c, 1) == 12345


def speculative_config(*, burst=200):
    c = config(burst=burst)
    c.policy.speculative_partition_workers = True
    for name, floor in [("gpu-train", 1), ("gpu-bf", 3)]:
        c.budgets[name] = Budget(partitions=[name], running_target=200, fill_running_target=True,
                                 min_pending=floor, max_submitted=200 + floor)
        c.profiles[name] = c.profiles["gpu"].model_copy(update={"partition": name, "budget": name,
                                                               "max_workers": 200 + floor})
    c.policy.budget_priority = ["gpu", "gpu-train", "gpu-bf"]
    return c


def assert_unique_partition_anchors(c, plan):
    by_partition = {}
    for action in plan["actions"]:
        pool = by_partition.setdefault(c.profiles[action["profile"]].partition, [])
        pool.extend(action["covered_task_ids"])
    for pool in by_partition.values():
        assert len(pool) == len(set(pool))


def test_speculation_fills_other_partitions_without_duplicating_tasks():
    assert Policy().speculative_partition_workers is False
    c = speculative_config()
    work = tasks(96)
    first = capacity_plan(c, work, [], [])
    assert [a["profile"] for a in first["actions"]] == ["gpu"] * 96 + ["gpu-train"] * 96 + ["gpu-bf"] * 8
    assert first["speculative_partition_workers"] is True
    assert first["speculative_anchor_reuses"] == 104
    assert_unique_partition_anchors(c, first)
    owned, jobs = [], []
    for i, action in enumerate(first["actions"]):
        reservation, job = allocation(i, "PENDING")
        reservation["profile"] = job["partition"] = action["profile"]
        owned.append(reservation)
        jobs.append(job)
    second = capacity_plan(c, work, jobs, owned)
    assert [a["profile"] for a in second["actions"]] == ["gpu-bf"] * 88
    assert_unique_partition_anchors(c, second)
    assert len(work) == 96 and len({t["task_id"] for t in work}) == 96  # Planning never creates tasks.


def test_speculation_shares_anchors_between_profiles_in_same_partition():
    c = speculative_config(burst=500)
    c.profiles["gpu-other"] = c.profiles["gpu"].model_copy()
    plan = capacity_plan(c, tasks(96), [], [])
    assert Counter(c.profiles[a["profile"]].partition for a in plan["actions"]) == {
        "gpu": 96, "gpu-train": 96, "gpu-bf": 96,
    }
    assert plan["speculative_anchor_reuses"] == 192
    assert_unique_partition_anchors(c, plan)


def test_speculation_keeps_unknown_reservations_globally_binding_regardless_of_order():
    c = speculative_config(burst=500)
    owned, jobs = zip(*(allocation(i, "PENDING") for i in range(76)))
    unknown = [{"backend": "worker", "token": f"unknown{i}", "profile": "gpu", "state": "unknown"}
               for i in range(20)]
    for all_owned in [list(owned) + unknown, unknown + list(owned)]:
        plan = capacity_plan(c, tasks(96), list(jobs), all_owned)
        assert Counter(a["profile"] for a in plan["actions"]) == {"gpu-train": 76, "gpu-bf": 76}
        assert plan["pending_reserves"]["gpu"]["uncertain_reservations"] == 20
        assert_unique_partition_anchors(c, plan)


@pytest.mark.parametrize("external_count", [0, 10])
def test_speculation_respects_aggregate_caps_and_external_jobs(external_count):
    c = speculative_config(burst=500)
    c.budgets["all"] = Budget(partitions=["gpu", "gpu-train", "gpu-bf"], max_submitted=100)
    jobs = [{"job_id": f"external{i}", "partition": "gpu", "state": "RUNNING"}
            for i in range(external_count)]
    plan = capacity_plan(c, tasks(96), jobs, [])
    assert len(plan["actions"]) == 100 - external_count
    assert plan["budget_usage_after_plan"]["all"] == 100
    assert_unique_partition_anchors(c, plan)


def test_speculation_cannot_override_task_routes_or_control_deadline():
    c = speculative_config(burst=500)
    work = [{**task, "allowed_profiles": ["gpu"]} for task in tasks(96)]
    plan = capacity_plan(c, work, [], [])
    assert [a["profile"] for a in plan["actions"]] == ["gpu"] * 96
    assert not capacity_plan(c, work, [], [], remaining_control_seconds=3600)["actions"]
    assert not capacity_plan(c, [], [], [])["actions"]


def test_speculation_requires_single_partition_profiles_but_default_still_allows_lists():
    c = config()
    c.profiles["gpu"].partition = "gpu,gpu-bf"
    assert Config.model_validate(c.model_dump()).profiles["gpu"].partition == "gpu,gpu-bf"
    c.policy.speculative_partition_workers = True
    with pytest.raises(ValueError, match="one Slurm partition per profile; split: gpu"):
        Config.model_validate(c.model_dump())


@pytest.mark.parametrize("state", ["PENDING", "RUNNING"])
@pytest.mark.parametrize("identity", ["wrong_comment", "missing", "wrong_id", "name_fallback"])
def test_speculation_requires_positive_slurm_ownership_to_relax_reservations(state, identity):
    c = speculative_config(burst=500)
    owned, job = allocation(1, state)
    if identity == "wrong_comment":
        job["comment"] = "hhq:unrelated"
    elif identity == "missing":
        job.pop("comment")
    elif identity == "wrong_id":
        job["job_id"] = "2"
    else:
        job.pop("comment")
        job["name"] = "hhq-job-worker1"
    plan = capacity_plan(c, tasks(3), [job], [owned])
    status = plan["pending_reserves"]["gpu"]
    counts = Counter(a["profile"] for a in plan["actions"])
    if identity == "name_fallback":
        assert status["uncertain_reservations"] == 0
        assert counts["gpu-train"] == counts["gpu-bf"] == 3
        assert shared_budget_usage(c, [job], [owned])["gpu"] == 1
    else:
        assert status["uncertain_reservations"] == 1
        assert status["observed_running"] == status["observed_pending"] == 0
        assert counts["gpu-train"] == counts["gpu-bf"] == 2
        assert shared_budget_usage(c, [job], [owned])["gpu"] == 2
        c.budgets["all"] = Budget(partitions=["gpu", "gpu-train", "gpu-bf"], max_submitted=2)
        capped = capacity_plan(c, tasks(3), [job], [owned])
        assert not capped["actions"] and capped["budget_usage_after_plan"]["all"] == 2
    assert_unique_partition_anchors(c, plan)


def test_growth_precedes_reserves_and_train_pending_cap():
    c = config(target=128, cap=136, burst=300)
    c.policy.pending_reserves_after_growth = True
    c.policy.speculative_partition_workers = True
    for name, target, floor, cap, max_pending in [('train', 2, 1, 3, 1), ('bf', 50, 3, 53, None)]:
        c.budgets[name] = Budget(partitions=[name], running_target=target, fill_running_target=True,
                                 min_pending=floor, max_submitted=cap, max_pending=max_pending)
        c.profiles[name] = c.profiles['gpu'].model_copy(update={'partition': name, 'budget': name, 'max_workers': cap})
    c.policy.budget_priority = ['gpu', 'train', 'bf']
    plan = capacity_plan(c, tasks(400), [], [])
    assert [a['profile'] for a in plan['actions']] == ['gpu'] * 128 + ['train'] + ['bf'] * 50
    assert plan['pending_reserves']['train']['blocked_by'] == ['MAX_PENDING:train']
    # Regular capacity is already running: train/backfill growth precedes GPU reserve.
    owned, jobs = zip(*(allocation(i) for i in range(128)))
    plan = capacity_plan(c, tasks(400), list(jobs), list(owned))
    assert [a['profile'] for a in plan['actions']] == ['train'] + ['bf'] * 50 + ['gpu'] * 8
    # An existing unknown training submission occupies its one pending slot.
    owned = [*owned, {'backend': 'worker', 'token': 'unknown_train', 'profile': 'train', 'state': 'unknown'}]
    plan = capacity_plan(c, tasks(400), list(jobs), owned)
    assert not any(a['profile'] == 'train' for a in plan['actions'])


def test_max_pending_also_limits_non_eager_capacity_packing():
    c = config(eager=False, floor=0)
    c.budgets['gpu'].max_pending = 1
    plan = capacity_plan(c, tasks(400), [], [])
    assert len(plan['actions']) == 1
    assert plan['pending_reserves']['gpu']['reserved_after_plan'] == 1
