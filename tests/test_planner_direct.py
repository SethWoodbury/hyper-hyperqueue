import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hyper_hyperqueue import api
from hyper_hyperqueue.backends.direct import resource_args
from hyper_hyperqueue.models import Budget, Config, Policy, Profile, Resources
from hyper_hyperqueue.planner import array_plan, capacity_plan, compatible, make_plan
from hyper_hyperqueue.state import StateStore


def cpu_config(tmp_path):
    return Config(
        state_dir=str(tmp_path / "state"),
        profiles={"cpu": Profile(partition="cpu", cpus=2, memory_mib=2048, allocation_seconds=3600)},
        policy=Policy(max_array_size=128, max_array_tasks=128, max_direct_submitted=1000),
    )


def test_thousand_commands_split_with_true_campaign_cap(tmp_path):
    config = cpu_config(tmp_path)
    arrays, unsent = array_plan([str(i) for i in range(1000)], config, max_parallel=10)
    assert len(arrays) == 8 and not unsent
    assert sum(x["max_parallel"] for x in arrays) == 10
    assert [tid for a in arrays for row in a["mapping"] for tid in row] == [str(i) for i in range(1000)]
    arrays, unsent = array_plan([str(i) for i in range(1000)], config, max_parallel=2)
    assert len(arrays) == 2 and len(unsent) == 744
    assert sum(x["max_parallel"] for x in arrays) == 2


def test_auto_and_impossible_requests(tmp_path):
    c = cpu_config(tmp_path)
    assert make_plan(10, Resources(runtime_seconds=1800), c, ["cpu"]).mode == "direct"
    assert make_plan(10, Resources(runtime_seconds=10), c, ["cpu"]).mode == "pooled"
    assert make_plan(10, Resources(), c, ["cpu"]).mode == "direct"
    with pytest.raises(ValueError, match="TASK_WILL_NOT_FIT"):
        make_plan(1, Resources(runtime_seconds=4000), c, ["cpu"], mode="pooled")
    assert not compatible(Resources(ranks=2), c.profiles["cpu"], c)[0]


def test_gpu_subclasses_share_budget_and_backlog():
    config = Config(
        budgets={"gpu": Budget(partitions=["gpu"], running_target=2, pending_reserve=1, max_submitted=3)},
        profiles={
            n: Profile(
                partition="gpu", budget="gpu", gres=f"gpu:{n}:1", memory_mib=2048, allocation_seconds=3600
            )
            for n in ["small", "large", "h200"]
        },
    )
    tasks = [
        {
            "task_id": str(i),
            "resources": Resources(gpus=1, runtime_seconds=3500).model_dump(),
            "allowed_profiles": list(config.profiles),
        }
        for i in range(10)
    ]
    jobs = [
        {"job_id": "external1", "partition": "gpu", "state": "RUNNING"},
        {"job_id": "external2", "partition": "gpu", "state": "PENDING"},
    ]
    plan = capacity_plan(config, tasks, jobs, [])
    assert len(plan["actions"]) == 1
    assert plan["budget_usage_after_plan"]["gpu"] == 3
    # One task cannot create three allocations merely because all pools fit it.
    plan = capacity_plan(config, tasks[:1], [], [])
    assert len(plan["actions"]) == 1


def test_cpu_tasks_never_fill_gpu_workers_by_default():
    config = Config(profiles={"gpu": Profile(partition="gpu", gres="gpu:1", memory_mib=4096)})
    assert compatible(Resources(), config.profiles["gpu"], config)[1] == "CPU_GPU_POOL_SEPARATION"


class FakeSlurm:
    """Models acceptance with a lost reply and changing observed scheduler state."""

    def __init__(self, lose_reply=False):
        self.jobs = []
        self.calls = []
        self.lose_reply = lose_reply

    def query_jobs(self, **kwargs):
        return list(self.jobs)

    def submit(self, script, token, job_name, extra_args):
        jid = str(100 + len(self.calls))
        self.calls.append((str(script), token, extra_args))
        self.jobs.append(
            {
                "job_id": jid,
                "name": job_name,
                "comment": "hhq:" + token,
                "state": "PENDING",
                "partition": "cpu",
            }
        )
        return {
            "status": "unknown" if self.lose_reply else "accepted",
            "job_id": None if self.lose_reply else jid,
        }

    def find_submission(self, token, since):
        return [j for j in self.jobs if j["comment"] == "hhq:" + token]

    def accounting(self, **kwargs):
        return []


def test_direct_identity_uncertain_submit_does_not_duplicate(tmp_path):
    c = cpu_config(tmp_path)
    source = tmp_path / "commands file.txt"
    source.write_text("printf '%s\\n' 'a; b'\nprintf '%s\\n' 'a; b'")
    fake = FakeSlurm(lose_reply=True)
    result = api.submit(source, c, profile="cpu", mode="direct", slurm=fake, cwd=tmp_path, import_key="test")
    store = StateStore(c.state_dir)
    assert len(store.task_ids(result.campaign_id)) == 2
    assert len(fake.calls) == 1
    from hyper_hyperqueue.backends.direct import submit_ready

    with store.lock():
        again = submit_ready(store, c, result.campaign_id, fake)
    assert not again["submitted_arrays"] and len(fake.calls) == 1
    from hyper_hyperqueue.controller import reconcile_slurm

    with store.lock():
        reconcile_slurm(store, fake, fake.query_jobs())
    assert store.intents()[0]["job_id"] == "100"
    script = Path(fake.calls[0][0]).read_text()
    assert "-m hyper_hyperqueue.backends.direct run-array" in script
    assert "worker start" not in script and "server start" not in script


def test_plan_does_not_mutate_state_or_slurm(tmp_path, monkeypatch):
    c = cpu_config(tmp_path)
    source = tmp_path / "commands"
    source.write_text("echo hello\n")
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("plan made subprocess call"))
    result = api.plan(source, c, profile="cpu", mode="direct")
    assert result.task_count == 1 and not Path(c.state_dir).exists()


def test_extra_sbatch_rejects_conflicts():
    for flag in ["--mem=20G", "--array=1-2", "--wrap=oops", "--comment=x", "--job-name=x", "-ofoo"]:
        with pytest.raises(ValueError):
            resource_args(Profile(partition="cpu", extra_sbatch=[flag]))
    assert "--nice=100" in resource_args(Profile(partition="cpu", extra_sbatch=["--nice=100"]))


def test_config_rejects_unknown_fields():
    with pytest.raises(ValueError):
        Config.model_validate({"memroy": 12})


def test_control_lifetime_must_fit_pooled_task():
    from hyper_hyperqueue.planner import compatible

    config = Config()
    config.control.walltime_seconds = 900
    profile = Profile(partition="cpu", cpus=1, memory_mib=2048, allocation_seconds=3600)
    task = Resources(runtime_seconds=1000)
    assert compatible(task, profile, config)[1].startswith("CONTROL_LIFETIME_TOO_SHORT")
    assert compatible(task, profile, config, pooled=False)[0]


def test_cli_plan_no_hq_import(tmp_path):
    c = cpu_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(c.model_dump_json())
    source = tmp_path / "cmd"
    source.write_text("echo ok")
    from hyper_hyperqueue.cli import app

    result = CliRunner().invoke(
        app, ["--config", str(config_path), "plan", str(source), "--mode", "direct", "--profile", "cpu"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["mode"] == "direct"


def invoke_plan(tmp_path, config, flags):
    from hyper_hyperqueue.cli import app

    config_path = tmp_path / "cli-config.json"
    config_path.write_text(config.model_dump_json())
    commands = tmp_path / "cli-commands"
    commands.write_text("true\n")
    result = CliRunner().invoke(app, ["--config", str(config_path), "plan", str(commands), *flags])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_cli_direct_overrides_and_declared_mpi_layout(tmp_path):
    config = cpu_config(tmp_path)
    result = invoke_plan(
        tmp_path,
        config,
        [
            "--profile",
            "cpu",
            "--mode",
            "direct",
            "--cpus",
            "4",
            "--mem",
            "4GiB",
            "--nodes",
            "2",
            "--ranks",
            "8",
        ],
    )
    assert result["requests"][0]["cpus"] == 4
    assert result["requests"][0]["memory_mib"] == 4096
    assert result["requests"][0]["nodes"] == 2 and result["requests"][0]["ranks"] == 8
    assert "DIRECT_MPI_REQUIRES_EXPLICIT_LAUNCHER_IN_COMMAND" in result["warnings"]
    args = resource_args(Profile(partition="cpu", cpus=4), Resources(nodes=2, ranks=8))
    assert args[args.index("--nodes") + 1] == "2"
    assert args[args.index("--ntasks") + 1] == "8"


def test_cli_partition_auto_short_cpu_includes_worker_headroom(tmp_path):
    result = invoke_plan(
        tmp_path, cpu_config(tmp_path), ["--partition", "cpu", "--runtime", "5", "--mem", "2GiB"]
    )
    assert result["mode"] == "pooled"
    assert result["requests"][0]["memory_mib"] == 2048 + 256
    assert result["profiles"][0].startswith("explicit_")


def test_cli_pooled_flags_do_not_silently_enlarge_worker(tmp_path):
    from hyper_hyperqueue.cli import app

    config = cpu_config(tmp_path)
    p = tmp_path / "config"
    p.write_text(config.model_dump_json())
    source = tmp_path / "source"
    source.write_text("true")
    result = CliRunner().invoke(
        app,
        ["--config", str(p), "plan", str(source), "--profile", "cpu", "--mode", "pooled", "--mem", "8GiB"],
    )
    assert result.exit_code != 0
    assert not Path(config.state_dir).exists()


def test_direct_shared_budget_counts_unknown_workers_and_external_arrays(tmp_path):
    from hyper_hyperqueue.backends.direct import submit_ready
    from hyper_hyperqueue.ingest import ingest_file

    c = cpu_config(tmp_path)
    c.budgets["shared"] = Budget(partitions=["cpu"], running_target=4, pending_reserve=0, max_submitted=4)
    c.profiles["cpu"].budget = "shared"
    store = StateStore(c.state_dir)
    source = tmp_path / "cmd"
    source.write_text("true\n" * 5)
    campaign = ingest_file(store, source, mode="direct", profile_names=["cpu"])
    store.put_intent(
        {"token": "unknown_worker", "backend": "worker", "state": "unknown", "profile": "cpu", "job_id": None}
    )
    fake = FakeSlurm()
    fake.jobs = [
        {"job_id": "900_1", "array_job_id": "900", "partition": "cpu", "state": "RUNNING"},
        {"job_id": "900_2", "array_job_id": "900", "partition": "cpu", "state": "PENDING"},
    ]
    with store.lock():
        result = submit_ready(store, c, campaign["campaign_id"], fake)
    assert result["submitted_arrays"][0]["elements"] == 1
    assert result["unsent_tasks"] == 4
    assert result["submitted_arrays"][0]["created_at"]


def test_shared_budget_does_not_double_count_observed_owned_array():
    from hyper_hyperqueue.planner import shared_budget_usage

    c = Config(
        budgets={"shared": Budget(partitions=["cpu"], max_submitted=10)},
        profiles={"cpu": Profile(partition="cpu", budget="shared")},
    )
    intent = {
        "backend": "direct",
        "token": "ownedtoken",
        "state": "submitted",
        "profile": "cpu",
        "job_id": "123",
        "elements": 3,
    }
    jobs = [
        {"job_id": "123_0", "array_job_id": "123", "state": "RUNNING", "partition": "cpu"},
        {"job_id": "123_1", "array_job_id": "123", "state": "PENDING", "partition": "cpu"},
    ]
    assert shared_budget_usage(c, jobs, [intent])["shared"] == 3


def test_resume_refuses_profile_shrunk_below_recorded_resources(tmp_path):
    from hyper_hyperqueue.backends.direct import submit_ready
    from hyper_hyperqueue.ingest import ingest_file

    c = cpu_config(tmp_path)
    store = StateStore(c.state_dir)
    source = tmp_path / "cmd"
    source.write_text("true")
    campaign = ingest_file(
        store, source, mode="direct", profile_names=["cpu"], resources=Resources(memory_mib=2048)
    )
    c.profiles["cpu"].memory_mib = 1024
    fake = FakeSlurm()
    with store.lock():
        result = submit_ready(store, c, campaign["campaign_id"], fake)
    assert not fake.calls
    assert result["validation_errors"][0]["reason"] == "NO_PROFILE_FITS_CURRENT_TASK_RESOURCES"


def test_pooled_config_mismatch_rejected_before_new_import(tmp_path):
    from hyper_hyperqueue.state import atomic_json

    c = cpu_config(tmp_path)
    root = Path(c.state_dir)
    service = root / "service"
    service.mkdir(parents=True)
    saved = service / "config.json"
    atomic_json(saved, c.model_dump())
    atomic_json(service / "deployment.json", {"config": str(saved), "desired_state": "RUNNING"})
    source = tmp_path / "cmd"
    source.write_text("true")
    c.profiles["cpu"].memory_mib = 4096
    with pytest.raises(ValueError, match="frozen configuration"):
        api.submit(source, c, profile="cpu", mode="pooled", start_service=False)
    assert not (root / "campaigns").exists()


def test_append_preserves_jsonl_and_task_metadata(tmp_path):
    c = cpu_config(tmp_path)
    source = tmp_path / "cmd.jsonl"
    source.write_text('{"argv":["echo","first"]}\n')
    result = api.submit(
        source,
        c,
        profile="cpu",
        mode="direct",
        format="jsonl",
        slurm=FakeSlurm(),
        workload="chemistry",
        priority=4,
        input_fingerprint="inputs-v2",
        software_fingerprint="git:abc",
    )
    more = tmp_path / "append.jsonl"
    more.write_text('{"argv":["echo","second"]}\n')
    api.append(result.campaign_id, more, c, slurm=FakeSlurm())
    store = StateStore(c.state_dir)
    tasks = [store.task(result.campaign_id, t) for t in store.task_ids(result.campaign_id)]
    assert [t.argv for t in tasks] == [["echo", "first"], ["echo", "second"]]
    assert all(
        t.workload == "chemistry"
        and t.priority == 4
        and t.input_fingerprint == "inputs-v2"
        and t.software_fingerprint == "git:abc"
        for t in tasks
    )


def test_direct_bundle_runtime_is_whole_allocation_not_one_command(tmp_path):
    c = cpu_config(tmp_path)
    with pytest.raises(ValueError, match="BUNDLE_WILL_NOT_FIT"):
        make_plan(4, Resources(runtime_seconds=1500), c, ["cpu"], mode="direct", commands_per_job=3)


def test_explicit_cli_profile_survives_later_resume_without_cli_overrides(tmp_path, monkeypatch):
    from hyper_hyperqueue import cli
    from hyper_hyperqueue.backends.direct import submit_ready

    base = cpu_config(tmp_path)
    base.policy.max_array_size = 1
    monkeypatch.setattr(cli, "cfg", lambda: base.model_copy(deep=True))
    configured, profile, resources = cli.options(None, "cpu", 1, "2GiB", None, None, None, None, "direct")
    source = tmp_path / "commands"
    source.write_text("true\ntrue\n")
    fake = FakeSlurm()
    result = api.submit(
        source, configured, profile=profile, mode="direct", resources=resources, max_parallel=1, slurm=fake
    )
    store = StateStore(base.state_dir)
    intent = store.intents()[0]
    intent["state"] = "terminal"
    store.put_intent(intent)
    store.update_task_state(result.campaign_id, intent["task_ids"][0], {"state": "succeeded"})
    fake.jobs = []
    with store.lock():
        next_result = submit_ready(store, base, result.campaign_id, fake, max_parallel=1)
    assert len(next_result["submitted_arrays"]) == 1
    assert next_result["submitted_arrays"][0]["profile"] == profile


def test_shared_budget_uses_recorded_placement_after_profile_reconfiguration():
    from hyper_hyperqueue.planner import shared_budget_usage

    c = Config(budgets={"old": Budget(partitions=["old"])}, profiles={"worker": Profile(partition="new")})
    intent = {
        "backend": "worker",
        "state": "unknown",
        "profile": "worker",
        "token": "unknown1",
        "placement": {"partition": "old", "account": None, "qos": None},
    }
    assert shared_budget_usage(c, [], [intent])["old"] == 1
