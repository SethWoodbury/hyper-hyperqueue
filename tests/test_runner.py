import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Resources
from hyper_hyperqueue.runner import authorize_retry, run_task
from hyper_hyperqueue.state import StateStore, atomic_json
from hyper_hyperqueue.telemetry import ResourceMismatch, assigned_gpu_context


def imported(tmp_path, command=None, argv=None, **kwargs):
    path = tmp_path / "commands.jsonl"
    path.write_text(json.dumps({"command": command} if command is not None else {"argv": argv}))
    store = StateStore(tmp_path / "state")
    campaign = ingest_file(store, path, mode="direct", cwd=tmp_path, format="jsonl", **kwargs)
    return store, campaign["campaign_id"], campaign["task_ids"][0]


def cli(store, campaign, task):
    return [
        sys.executable,
        "-m",
        "hyper_hyperqueue.runner",
        "--state-dir",
        str(store.root),
        "--campaign",
        campaign,
        "--task",
        task,
    ]


def wait_until(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.03)
    raise AssertionError("Condition did not become true")


def test_preserves_shell_and_argv_no_implicit_pipefail_and_replay(tmp_path):
    store, campaign, task = imported(
        tmp_path,
        command="printf '%s' '$HOME ; $(false)' >> result; false | true",
        output={"expected_paths": ["result"]},
    )
    assert run_task(store.root, campaign, task) == 0
    assert run_task(store.root, campaign, task) == 0
    assert (tmp_path / "result").read_text() == "$HOME ; $(false)"
    attempts = store.attempts(campaign, task)
    assert len(attempts) == 1
    assert attempts[0]["result"]["validation"]["kind"] == "expected_paths"
    assert store.task_state(campaign, task)["state"] == "succeeded"
    separate = tmp_path / "argv"
    separate.mkdir()
    store, campaign, task = imported(
        separate, argv=[sys.executable, "-c", "import sys;print(sys.argv[1])", "a;$HOME"]
    )
    assert run_task(store.root, campaign, task) == 0
    assert (Path(store.attempts(campaign, task)[0]["attempt_dir"]) / "stdout").read_text() == "a;$HOME\n"


def test_nonzero_and_zero_byte_output_are_failures(tmp_path):
    store, campaign, task = imported(tmp_path, command="touch empty", output={"expected_paths": ["empty"]})
    assert run_task(store.root, campaign, task) == 65
    assert store.task_state(campaign, task)["reason"] == "output_validation"
    assert not list((store.campaign_dir(campaign) / "completions").glob("*.json"))
    separate = tmp_path / "failure"
    separate.mkdir()
    store, campaign, task = imported(separate, command="echo failure >&2; exit 7")
    assert run_task(store.root, campaign, task) == 7
    attempt = store.attempts(campaign, task)[0]
    assert attempt["result"]["reason"] == "application_exit"
    assert "failure" in (Path(attempt["attempt_dir"]) / "stderr").read_text()


def test_timeout_reaps_group_even_if_shell_trap_exits_zero(tmp_path):
    command = f"{sys.executable} -c 'import os,time;open(\"child\",\"w\").write(str(os.getpid()));time.sleep(60)' & trap 'exit 0' TERM; wait"
    store, campaign, task = imported(tmp_path, command=command, resources={"timeout_seconds": 0.3})
    assert run_task(store.root, campaign, task) == 124
    result = store.attempts(campaign, task)[0]["result"]
    assert result["reason"] == "timeout"
    assert result["execution"]["timed_out"]
    pid = int((tmp_path / "child").read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_real_signal_forward_duplicate_lock_and_cancellation(tmp_path):
    command = f"{sys.executable} -c 'import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);open(\"child\",\"w\").write(str(os.getpid()));time.sleep(60)' & trap 'exit 0' TERM; wait"
    store, campaign, task = imported(
        tmp_path,
        command=command,
        retry={"max_attempts": 2, "reasons": ["user_cancellation"]},
        output={"repeatable": True, "mode": "in_place"},
    )
    runner = subprocess.Popen(cli(store, campaign, task))
    try:
        wait_until(lambda: (tmp_path / "child").exists())
        assert store.task_state(campaign, task)["state"] == "running"
        assert run_task(store.root, campaign, task) == 75
        runner.send_signal(signal.SIGTERM)
        assert runner.wait(8) == 143
        result = store.attempts(campaign, task)[0]["result"]
        assert result["status"] == "canceled" and result["execution"]["returncode"] == 0
        assert run_task(store.root, campaign, task) == 75
        assert len(store.attempts(campaign, task)) == 1
        pid = int((tmp_path / "child").read_text())
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()


def test_sigkill_leaves_unknown_and_never_blindly_replays(tmp_path):
    store, campaign, task = imported(
        tmp_path,
        argv=[
            sys.executable,
            "-c",
            "import os,time;open('child','w').write(str(os.getpid()));time.sleep(60)",
        ],
        retry={"max_attempts": 2},
        output={"repeatable": True, "mode": "in_place"},
    )
    runner = subprocess.Popen(cli(store, campaign, task))
    try:
        wait_until(lambda: (tmp_path / "child").exists())
        runner.kill()
        runner.wait(5)
        assert run_task(store.root, campaign, task) == 75
        assert store.task_state(campaign, task)["state"] == "unknown"
        with pytest.raises(ValueError, match="fencing"):
            authorize_retry(store, campaign, task, manual=True)
        assert len(store.attempts(campaign, task)) == 1
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()
        if (tmp_path / "child").exists():
            try:
                os.killpg(int((tmp_path / "child").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_bounded_retry_manual_retry_and_completed_result_adoption(tmp_path):
    command = "if test -f first; then echo done; else touch first; exit 9; fi"
    store, campaign, task = imported(tmp_path, command=command, retry={"max_attempts": 2})
    assert run_task(store.root, campaign, task) == 9
    assert run_task(store.root, campaign, task) == 75
    authorize_retry(store, campaign, task, manual=True)
    assert run_task(store.root, campaign, task) == 0
    completion = store.campaign_dir(campaign) / "completions" / f"{task}.json"
    completion.unlink()  # Emulate crash immediately before the marker write.
    assert run_task(store.root, campaign, task) == 0
    assert len(store.attempts(campaign, task)) == 2
    assert completion.exists()


def test_isolated_output_and_collision_claim(tmp_path):
    source = tmp_path / "commands"
    source.write_text(
        'echo result > "$HHQ_TASK_OUTPUT_DIR/result"\necho another > "$HHQ_TASK_OUTPUT_DIR/result"'
    )
    store = StateStore(tmp_path / "state")
    campaign = ingest_file(
        store,
        source,
        mode="direct",
        cwd=tmp_path,
        output={
            "mode": "isolated",
            "expected_paths": ["{attempt_dir}/output/result"],
            "collision_key": "published_target",
        },
    )
    first, second = campaign["task_ids"]
    assert run_task(store.root, campaign["campaign_id"], first) == 0
    assert run_task(store.root, campaign["campaign_id"], second) == 75
    assert not store.attempts(campaign["campaign_id"], second)


def test_validator_failure_never_commits_completion(tmp_path):
    store, campaign, task = imported(
        tmp_path, command="true", output={"validator_argv": [sys.executable, "-c", "raise SystemExit(3)"]}
    )
    assert run_task(store.root, campaign, task) == 65
    assert store.attempts(campaign, task)[0]["result"]["validation"]["execution"]["returncode"] == 3


def test_explicit_cancellation_overrides_failed_result(tmp_path):
    store, campaign, task = imported(tmp_path, command="exit 2", retry={"max_attempts": 3})
    assert run_task(store.root, campaign, task) == 2
    store.update_task_state(campaign, task, {"state": "canceled", "reason": "user_cancelled"})
    assert store.task_state(campaign, task)["state"] == "canceled"
    with pytest.raises(ValueError, match="not retryable"):
        authorize_retry(store, campaign, task, manual=True)
    assert run_task(store.root, campaign, task) == 130


def test_stale_epoch_and_bound_unknown_retry(tmp_path, monkeypatch):
    store, campaign, task = imported(
        tmp_path, command="true", retry={"max_attempts": 2}, output={"repeatable": True, "mode": "in_place"}
    )
    store.update_task_state(campaign, task, {"server_epoch": 5})
    monkeypatch.setenv("HHQ_SERVER_EPOCH", "4")
    assert run_task(store.root, campaign, task) == 75
    assert store.attempts(campaign, task) == []
    monkeypatch.setenv("HHQ_SERVER_EPOCH", "5")
    assert run_task(store.root, campaign, task) == 0
    attempt = store.attempts(campaign, task)[0]
    # Simulate loss of final records in a disposable test generation.
    (store.campaign_dir(campaign) / "completions" / f"{task}.json").unlink()
    (Path(attempt["attempt_dir"]) / "result.json").unlink()
    proof = tmp_path / "fenced.json"
    atomic_json(proof, {"quiescent": True, "previous_epoch": 4})
    with pytest.raises(ValueError, match="epoch"):
        authorize_retry(store, campaign, task, terminal_evidence=proof)
    atomic_json(proof, {"quiescent": True, "previous_epoch": 5})
    authorize_retry(store, campaign, task, terminal_evidence=proof)
    assert store.task_state(campaign, task)["state"] == "retry_wait"
    # Changed evidence fails closed even if the JSON still looks plausible.
    atomic_json(proof, {"quiescent": True, "previous_epoch": 5, "changed": True})
    assert run_task(store.root, campaign, task) == 75


@pytest.mark.parametrize("variable", ["HHQ_WORKER_EPOCH", "HHQ_SERVER_EPOCH"])
def test_task_environment_cannot_spoof_epoch_provenance(tmp_path, monkeypatch, variable):
    monkeypatch.setenv(variable, "6")
    store, campaign, task = imported(tmp_path, command="touch must_not_exist", environment={variable: "999"})
    assert run_task(store.root, campaign, task) == 78
    assert not (tmp_path / "must_not_exist").exists()


def test_assigned_gpu_limits_do_not_query_other_devices():
    queried = []

    def query(identifier):
        queried.append(identifier)
        return {"uuid": "GPU-test", "model": "A100", "memory_mib": 40960}

    resources = Resources(gpus=1, gpu_models=["A100"], min_vram_mib=40000)
    env = {"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "3"}
    record = assigned_gpu_context(resources, environment=env, query=query)
    assert queried == [0] and record["devices"][0]["visible_identifier"] == "0"
    assert record["devices"][0]["slurm_job_gpus"] == "3"
    with pytest.raises(ResourceMismatch, match="allowlist"):
        assigned_gpu_context(
            resources.model_copy(update={"gpu_models": ["H100"]}), environment=env, query=query
        )
    with pytest.raises(ResourceMismatch, match="VRAM"):
        assigned_gpu_context(
            resources.model_copy(update={"min_vram_mib": 80000}), environment=env, query=query
        )
    with pytest.raises(ResourceMismatch, match="explicitly assigned"):
        assigned_gpu_context(resources, environment={}, query=query)


def test_resource_mismatch_records_attempt_without_launching_command(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    store, campaign, task = imported(tmp_path, command="touch must_not_exist", resources={"gpus": 1})
    assert run_task(store.root, campaign, task) == 78
    assert not (tmp_path / "must_not_exist").exists()
    assert store.attempts(campaign, task)[0]["result"]["reason"] == "resource_mismatch"


def test_gpu_task_cannot_change_cuda_device_order(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    store, campaign, task = imported(
        tmp_path,
        command="touch must_not_exist",
        resources={"gpus": 1},
        environment={"CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )
    assert run_task(store.root, campaign, task) == 78
    assert not (tmp_path / "must_not_exist").exists()
    assert "CUDA_DEVICE_ORDER" in store.attempts(campaign, task)[0]["result"]["error"]


def test_thread_defaults_preserve_inherited_and_explicit_choices(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    store, campaign, task = imported(
        tmp_path,
        argv=[
            sys.executable,
            "-c",
            'import os;print(os.environ["OMP_NUM_THREADS"],os.environ["MKL_NUM_THREADS"],os.environ["OPENBLAS_NUM_THREADS"])',
        ],
        resources={"cpus": 2},
        environment={"OPENBLAS_NUM_THREADS": "7"},
    )
    assert run_task(store.root, campaign, task) == 0
    attempt = store.attempts(campaign, task)[0]
    assert (Path(attempt["attempt_dir"]) / "stdout").read_text() == "3 2 7\n"


def test_normal_exit_cleans_background_group_without_changing_exit_status(tmp_path):
    command = f'{sys.executable} -c \'import os,time;open("background","w").write(str(os.getpid()));time.sleep(60)\' & while ! test -s background; do sleep .01; done; exit 0'
    store, campaign, task = imported(tmp_path, command=command)
    assert run_task(store.root, campaign, task) == 0
    pid = int((tmp_path / "background").read_text())
    deadline = time.monotonic() + 2
    while True:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        assert time.monotonic() < deadline, "Background process survived group cleanup"
        time.sleep(0.01)
    result = store.attempts(campaign, task)[0]["result"]
    assert result["execution"]["returncode"] == 0
    assert "before leader was reaped" in result["execution"]["process_group_cleanup"]
