import datetime as dt
import json

import pytest

from hyper_hyperqueue.backends.slurm import (
    CommandResult,
    SlurmClient,
    SlurmError,
    number,
    parse_accounting_job,
    parse_queue_job,
)
from hyper_hyperqueue.discovery import parse_partition, summarize


class Fake:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, argv, timeout):
        self.calls.append(argv)
        response = self.responses.pop(0)
        if isinstance(response, dict) and "returncode" not in response:
            return CommandResult(argv, 0, json.dumps(response))
        return CommandResult(argv, **response)


def job(**kwargs):
    return dict(
        job_id=123,
        name="hhq-token123",
        user_name="alice",
        job_state=["RUNNING"],
        partition="gpu",
        comment="hhq:token123",
        **kwargs,
    )


def test_wrappers_and_resources():
    row = job(
        cpus={"set": True, "number": 4},
        array_job_id={"set": True, "number": 100},
        array_task_id={"set": True, "number": 3},
        tres_req_str="cpu=4,mem=8G,gres/gpu=2,gres/gpu:large=2",
        time_limit={"set": True, "number": 60},
    )
    parsed = parse_queue_job(row)
    assert (parsed.job_id, parsed.gpus, parsed.memory_mib, parsed.time_limit_seconds) == (
        "100_3",
        2,
        8192,
        3600,
    )
    assert number({"set": False, "number": 0}) is None
    assert number({"set": True, "infinite": True, "number": 999}) is None
    assert number({"set": True, "number": 0}) == 0


def test_accounting_parser_separate_schema():
    row = {
        "job_id": 123,
        "name": "n",
        "user": "alice",
        "state": {"current": ["COMPLETED"], "reason": "None"},
        "comment": {"job": "hhq:token123"},
        "partition": "gpu",
        "time": {
            "submission": 10,
            "start": 20,
            "end": 50,
            "elapsed": 30,
            "limit": {"set": True, "number": 60},
        },
        "array": {"job_id": 100, "task_id": {"set": True, "number": 2}},
        "tres": {
            "allocated": [
                {"type": "gres", "name": "gpu", "count": 1},
                {"type": "gres", "name": "gpu:large", "count": 1},
                {"type": "mem", "name": "", "count": 8192},
            ]
        },
    }
    parsed = parse_accounting_job(row)
    assert parsed.job_id == "100_2"
    assert parsed.gpus == 1
    assert parsed.memory_mib == 8192
    assert parsed.terminal


def test_json_fallback_only_unsupported():
    sep = "\x1f"
    line = sep.join(
        [
            "123",
            "name",
            "alice",
            "PENDING",
            "gpu",
            "lab",
            "normal",
            "hhq:token123",
            "Priority",
            "",
            "2",
            "4G",
            "2026-01-01T00:00:00",
            "N/A",
            "N/A",
            "1:00:00",
            "gpu:large:1",
        ]
    )
    fake = Fake(
        {"returncode": 1, "stderr": "squeue: unrecognized option --json"},
        {"returncode": 0, "stdout": line + "\n"},
        {"returncode": 0, "stdout": ""},
    )
    client = SlurmClient("alice", fake)
    parsed = client.query_jobs()
    assert parsed[0].memory_mib == 4096
    assert parsed[0].gpus == 1
    assert client.query_jobs() == []
    assert sum("--json" in call for call in fake.calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        {"returncode": 1, "stderr": "Unable to contact slurm controller"},
        {"returncode": 0, "stdout": "malformed"},
        {"returncode": 0, "stdout": '{"errors": [{"error": "permission"}]}'},
        {"returncode": 0, "stdout": "{}"},
    ],
)
def test_failed_queries_never_become_empty_queue(response):
    fake = Fake(response)
    with pytest.raises(SlurmError):
        SlurmClient("alice", fake).query_jobs()
    assert len(fake.calls) == 1


def test_submit_persists_identity_and_never_retries_timeout(tmp_path):
    script = tmp_path / "batch.sh"
    script.write_text("#!/bin/bash\ntrue\n")
    fake = Fake({"returncode": None, "timed_out": True, "stdout": "123\n"})
    result = SlurmClient("alice", fake).submit(script, "token123")
    assert result.status == "unknown"
    assert len(fake.calls) == 1
    assert "hhq:token123" in fake.calls[0]
    assert "hhq-token123" in fake.calls[0]


@pytest.mark.parametrize(
    "response,status",
    [
        ({"returncode": 0, "stdout": "123;cluster\n"}, "accepted"),
        (
            {
                "returncode": 1,
                "stderr": "sbatch: error: Batch job submission failed: Invalid partition name specified",
            },
            "rejected",
        ),
        (
            {
                "returncode": 1,
                "stderr": "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation",
            },
            "unknown",
        ),
        ({"returncode": 0, "stdout": "Submission acknowledged vaguely"}, "unknown"),
    ],
)
def test_submission_classification(tmp_path, response, status):
    script = tmp_path / "s"
    script.write_text("x")
    assert SlurmClient("alice", Fake(response)).submit(script, "token123").status == status


def test_no_identity_override(tmp_path):
    for extra in [["--comment=other"], ["--job-name", "other"], ["--wrap=x"]]:
        with pytest.raises(ValueError):
            SlurmClient("alice", Fake()).submit(tmp_path / "none", "token123", extra_args=extra)


def test_find_submission_checks_accounting_after_empty_live_queue():
    accounting_row = {
        "job_id": 123,
        "name": "hhq-token123",
        "user": "alice",
        "partition": "cpu",
        "comment": {"job": "hhq:token123"},
        "state": {"current": ["COMPLETED"]},
    }
    fake = Fake({"jobs": []}, {"jobs": [accounting_row]})
    matches = SlurmClient("alice", fake).find_submission(
        "token123", dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
    )
    assert matches[0].job_id == "123"
    assert "--starttime" in fake.calls[1] and "--endtime" in fake.calls[1]


def test_accounting_bounded():
    c = SlurmClient("alice", Fake())
    with pytest.raises(ValueError):
        c.accounting()
    with pytest.raises(ValueError):
        c.accounting(since=1, until=1000000)


def test_cancel_requires_exact_owner_and_token():
    row = job()
    fake = Fake({"jobs": [row]}, {"returncode": 0})
    SlurmClient("alice", fake).cancel(["123"], "token123")
    assert fake.calls[-1] == ["scancel", "123"]
    for changed in [dict(row, user_name="bob"), dict(row, comment="hhq:other123")]:
        fake = Fake({"jobs": [changed]})
        with pytest.raises(SlurmError):
            SlurmClient("alice", fake).cancel(["123"], "token123")
        assert len(fake.calls) == 1


def test_cancel_array_parent_checks_every_expanded_element():
    rows = [
        job(array_job_id=100, array_task_id=1),
        dict(job(array_job_id=100, array_task_id=2), comment="foreign"),
    ]
    fake = Fake({"jobs": rows})
    with pytest.raises(SlurmError):
        SlurmClient("alice", fake).cancel(["100"], "token123")
    assert len(fake.calls) == 1


def test_discovery_unknown_is_not_infinity_or_empty_user_queue():
    raw = {
        "config_text": {
            "returncode": 0,
            "stdout": "MaxArraySize = 250000\nSchedulerParameters = bf_max_job_user=1000\n",
        },
        "qos_text": {"returncode": 0, "stdout": "gpu|0||128||||||gres/gpu=192|||cluster\n"},
    }
    snapshot = summarize(raw, "alice", [], queue_error="permission denied")
    assert snapshot["current_user"]["submitted"] is None
    assert snapshot["limits"]["effective_user_submit_limit"]["status"] == "unknown"
    assert snapshot["qos"][0]["limits"]["max_jobs_per_user"]["value"] == 128
    assert snapshot["qos"][0]["limits"]["max_submit_per_user"]["status"] == "not_set_at_this_scope"
    assert snapshot["limits"]["max_array_tasks"]["value"] == 250000


def test_partition_time_limit_and_preemption():
    parsed = parse_partition(
        "PartitionName=gpu-bf QoS=N/A MaxTime=1-00:00:00 PreemptMode=REQUEUE State=UP TotalNodes=12 AllowAccounts=ALL"
    )
    assert parsed["max_time_seconds"] == 86400
    assert parsed["preempt_mode"] == "REQUEUE"
    assert parsed["qos"] is None


def test_real_site_accounting_drops_comment_but_preserves_full_name_token(tmp_path):
    script = tmp_path / "script"
    script.write_text("#!/bin/bash\ntrue\n")
    fake = Fake({"returncode": 0, "stdout": "123\n"})
    submitted = SlurmClient("alice", fake).submit(script, "full_token_123", job_name="control-primary")
    assert submitted.job_name == "control-primary-full_token_123"
    fake = Fake(
        {"jobs": []},
        {
            "jobs": [
                {
                    "job_id": 123,
                    "name": submitted.job_name,
                    "user": "alice",
                    "comment": {"job": ""},
                    "state": {"current": ["COMPLETED"]},
                },
                {
                    "job_id": 124,
                    "name": submitted.job_name + "x",
                    "user": "alice",
                    "comment": {"job": ""},
                    "state": {"current": ["COMPLETED"]},
                },
                {
                    "job_id": 125,
                    "name": submitted.job_name,
                    "user": "bob",
                    "comment": {"job": ""},
                    "state": {"current": ["COMPLETED"]},
                },
            ]
        },
    )
    result = SlurmClient("alice", fake).find_submission(
        "full_token_123", dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
    )
    assert [j.job_id for j in result] == ["123"]


def test_query_expired_id_filters_successful_user_snapshot():
    fake = Fake({"jobs": [job()]})
    assert SlurmClient("alice", fake).query_jobs(ids=["999"]) == []
    assert "--jobs" not in fake.calls[0]


def test_exact_seven_day_rolling_window_rounding():
    fake = Fake({"jobs": []})
    assert SlurmClient("alice", fake).accounting(since=dt.datetime.now(dt.UTC) - dt.timedelta(days=7)) == []


def test_unfamiliar_return_127_not_assumed_local_rejection(tmp_path):
    script = tmp_path / "script"
    script.write_text("#!/bin/bash\ntrue\n")
    assert (
        SlurmClient("alice", Fake({"returncode": 127, "stderr": "wrapper failure"}))
        .submit(script, "token123")
        .status
        == "unknown"
    )
    assert (
        SlurmClient("alice", Fake({"returncode": 127, "stderr": "not found", "launch_error": True}))
        .submit(script, "token123")
        .status
        == "rejected"
    )


def test_config_placement_requires_resources_on_same_eligible_gpu_node():
    from hyper_hyperqueue.discovery import validate_config_against_snapshot
    from hyper_hyperqueue.models import Config

    config = Config.model_validate(
        {"profiles": {"x": {"partition": "gpu", "gres": "gpu:large:1", "cpus": 32, "memory_mib": 4096}}}
    )
    snapshot = {
        "partitions": [{"name": "gpu", "state": "UP", "max_time_seconds": 3600}],
        "nodes_summary": [
            {
                "partitions": ["gpu"],
                "gres": "gpu:large:1",
                "features": ["big"],
                "cpus": 8,
                "memory_mib": 65536,
            },
            {
                "partitions": ["gpu"],
                "gres": "gpu:small:1",
                "features": ["small"],
                "cpus": 64,
                "memory_mib": 65536,
            },
        ],
    }
    result = validate_config_against_snapshot(config, snapshot)
    assert not result["valid_observable_constraints"]
    config.profiles["x"].cpus = 8
    assert validate_config_against_snapshot(config, snapshot)["valid_observable_constraints"]
    config.profiles["x"].constraint = "imaginary"
    assert not validate_config_against_snapshot(config, snapshot)["valid_observable_constraints"]


def test_json_pending_array_range_expands_even_if_json_ignores_array_flag():
    row = job(array_job_id=100, array_task_id={"set": False, "number": 0}, array_task_string="0-8:2,11%2")
    jobs = SlurmClient("alice", Fake({"jobs": [row]})).query_jobs()
    assert [j.job_id for j in jobs] == ["100_0", "100_2", "100_4", "100_6", "100_8", "100_11"]
    assert sum(j.array_task_count for j in jobs) == 6


def test_large_json_arrays_keep_exact_count_without_unbounded_expansion():
    row = job(array_job_id=100, array_task_id={"set": False}, array_task_string="0-249999%2")
    client = SlurmClient("alice", Fake({"jobs": [row]}))
    jobs = client.query_jobs()
    assert len(jobs) == 1 and jobs[0].array_task_count == 250000
    assert jobs[0].job_id == "100_[0-249999]"
    selected = SlurmClient("alice", Fake({"jobs": [row]})).query_jobs(ids=["100_42"])
    assert len(selected) == 1 and selected[0].job_id == "100_42" and selected[0].array_task_count == 1


def test_truncated_array_fails_closed_instead_of_undercount():
    row = job(array_job_id=100, array_task_id={"set": False}, array_task_string="1-100,...")
    with pytest.raises(SlurmError):
        SlurmClient("alice", Fake({"jobs": [row]})).query_jobs()


def test_cancel_one_child_of_large_compressed_array_checks_membership_and_identity():
    row = job(array_job_id=100, array_task_id={"set": False}, array_task_string="0-249999%2")
    fake = Fake({"jobs": [row]}, {"returncode": 0})
    SlurmClient("alice", fake).cancel(["100_42"], "token123")
    assert fake.calls[-1] == ["scancel", "100_42"]
    for target in ["100_250000", "101_42"]:
        fake = Fake({"jobs": [row]})
        with pytest.raises(SlurmError, match="ownership not proven"):
            SlurmClient("alice", fake).cancel([target], "token123")
        assert len(fake.calls) == 1
