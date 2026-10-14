"""Application startup checks fail before HQ can hand a worker any tasks."""
import json
import signal
import sys
import time
from datetime import UTC, datetime

import psutil
import pytest

from hyper_hyperqueue import config as config_module
from hyper_hyperqueue import worker
from hyper_hyperqueue.controller import allocation_gates
from hyper_hyperqueue.models import Config, Profile
from hyper_hyperqueue.startup_probe import OUTPUT_TAIL_BYTES, StartupProbeError, run_startup_probe
from hyper_hyperqueue.state import StateStore, atomic_json, fingerprint, read_json


def profile(code, *, timeout=30):
    return Profile(partition="cpu", startup_probe_argv=[sys.executable, "-c", code],
                   startup_probe_timeout_seconds=timeout)


def test_default_disabled_has_identical_serialized_schema_and_fingerprint(monkeypatch):
    configured = Config(profiles={"cpu": {"partition": "cpu"}})
    document = configured.model_dump()
    assert "startup_probe_argv" not in document["profiles"]["cpu"]
    assert "startup_probe_timeout_seconds" not in document["profiles"]["cpu"]
    assert Config.model_validate(document).model_dump() == document
    assert fingerprint(Config.model_validate(document).model_dump()) == fingerprint(document)
    monkeypatch.setattr("subprocess.Popen", lambda *_a, **_kw: pytest.fail("Disabled probe started a process"))
    assert run_startup_probe(configured.profiles["cpu"]) is None


def test_enabled_probe_is_frozen_and_inherits_gpu_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test-assignment")
    configured = profile("import os; print(os.environ['CUDA_VISIBLE_DEVICES'])")
    record = run_startup_probe(configured)
    assert record["status"] == "passed" and record["returncode"] == 0
    assert record["stdout_tail"] == "GPU-test-assignment\n"
    assert configured.model_dump()["startup_probe_argv"] == configured.startup_probe_argv
    changed = configured.model_copy(update={"startup_probe_timeout_seconds": 31})
    assert fingerprint(changed.model_dump()) != fingerprint(configured.model_dump())


@pytest.mark.parametrize("changes", [
    {"startup_probe_argv": "echo pass"}, {"startup_probe_argv": [""]},
    {"startup_probe_argv": ["echo", "a\0b"]}, {"startup_probe_argv": [123]},
    {"startup_probe_timeout_seconds": 0}, {"startup_probe_timeout_seconds": 301},
    {"startup_probe_timeout_seconds": 1.1}, {"startup_probe_timeout_seconds": True},
])
def test_invalid_probe_rejected(changes):
    with pytest.raises(ValueError):
        Profile(partition="cpu", **changes)


def test_output_capture_is_bounded_and_drains_both_streams():
    record = run_startup_probe(profile("import os; os.write(1,b'x'*2000000); os.write(2,b'y'*2000000)"))
    for stream, char in (("stdout", "x"), ("stderr", "y")):
        assert record[f"{stream}_tail"] == char * OUTPUT_TAIL_BYTES
        assert record[f"{stream}_bytes"] == 2000000
        assert record[f"{stream}_truncated"] is True
    assert len(json.dumps(record)) < 10000


def test_nonzero_and_missing_executable_are_failed_probes():
    with pytest.raises(StartupProbeError) as caught:
        run_startup_probe(profile("import sys; print('unknown userid',file=sys.stderr); sys.exit(255)"))
    assert caught.value.result["returncode"] == 255
    assert caught.value.result["stderr_tail"] == "unknown userid\n"
    with pytest.raises(StartupProbeError) as caught:
        run_startup_probe(Profile(partition="cpu", startup_probe_argv=["/nonexistent/hhq-probe"]))
    assert caught.value.result["reason"] == "startup_error"
    assert "FileNotFoundError" in caught.value.result["error"]


def test_timeout_kills_process_group_and_preserves_bounded_tail(tmp_path):
    pidfile = tmp_path / "child.pid"
    code = ("import subprocess,sys,time; from pathlib import Path; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
            f"Path({str(pidfile)!r}).write_text(str(child.pid)); "
            "print('started',flush=True); time.sleep(60)")
    start = time.monotonic()
    with pytest.raises(StartupProbeError) as caught:
        run_startup_probe(profile(code, timeout=1))
    assert time.monotonic() - start < 5
    assert caught.value.result["reason"] == "timeout"
    assert caught.value.result["stdout_tail"] == "started\n"
    pid = int(pidfile.read_text())
    for _ in range(50):
        if not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
            break
        time.sleep(0.02)
    else:
        pytest.fail("Probe descendant remained alive after timeout")


def test_worker_signal_aborts_probe_and_restores_signal_handler():
    previous = signal.getsignal(signal.SIGUSR1)
    with pytest.raises(StartupProbeError) as caught:
        run_startup_probe(profile("import os,signal,time; os.kill(os.getppid(), signal.SIGUSR1); time.sleep(60)"))
    assert caught.value.result["reason"] == "signal"
    assert caught.value.result["signal"] == signal.SIGUSR1
    assert caught.value.result["returncode"] == -signal.SIGKILL
    assert signal.getsignal(signal.SIGUSR1) == previous


def setup_worker(tmp_path, monkeypatch, selected):
    configured = Config(state_dir=str(tmp_path), profiles={"cpu": selected})
    monkeypatch.setattr(config_module, "load_config", lambda *_a, **_kw: configured)
    monkeypatch.setenv("SLURM_JOB_ID", "734")
    monkeypatch.setenv("HHQ_ALLOCATION_TOKEN", "probe-token")
    monkeypatch.setattr(worker, "detect_assigned", lambda _: {"cpus": 1, "memory_mib": 1024, "gpu": None})
    atomic_json(tmp_path / "service/endpoint.json", {"epoch": 2, "status": "ACTIVE"})
    return configured, ["--state-dir", str(tmp_path), "--profile", "cpu", "--config", "unused", "--epoch", "2"]


@pytest.mark.parametrize("code, timeout", [("raise SystemExit(255)", 30), ("import time; time.sleep(60)", 1)])
def test_failed_probe_never_registers_or_starts_hq_and_uses_existing_circuit(tmp_path, monkeypatch, code, timeout):
    configured, argv = setup_worker(tmp_path, monkeypatch, profile(code, timeout=timeout))
    monkeypatch.setattr(worker, "worker_argv", lambda *_a, **_kw: pytest.fail("Failed probe reached HQ"))
    with pytest.raises(StartupProbeError):
        worker.main(argv)
    record = read_json(tmp_path / "service/worker_preflight/734.json")
    assert record["status"] == "failed" and record["startup_probe"]["status"] == "failed"
    assert record["allocation_token"] == "probe-token"
    assert not (tmp_path / "service/workers/734.json").exists()
    assert not (tmp_path / "campaigns").exists()
    configured.policy.max_rpc_failures = 1
    intent = {"token": "probe-token", "profile": "cpu", "epoch": 2, "state": "terminal", "job_id": "734",
              "created_at": datetime.now(UTC).isoformat(), "terminal_evidence": [{"state": "FAILED", "elapsed_seconds": 1}]}
    paused, reasons = allocation_gates(StateStore(tmp_path), configured, [intent])
    assert paused == {"cpu"} and any("STARTUP_CIRCUIT_OPEN" in reason for reason in reasons)


@pytest.mark.parametrize("enabled", [False, True])
def test_successful_or_disabled_probe_allows_worker_registration(tmp_path, monkeypatch, enabled):
    _configured, argv = setup_worker(tmp_path, monkeypatch, profile("pass") if enabled else Profile(partition="cpu"))
    monkeypatch.setattr(worker, "allocation_remaining", lambda: None)

    def command(*_args, **_kwargs):
        record = read_json(tmp_path / "service/worker_preflight/734.json")
        assert record["status"] == "passed"
        assert ("startup_probe" in record) == enabled
        return ["hq", "--time-limit", "3s"]

    monkeypatch.setattr(worker, "worker_argv", command)
    monkeypatch.setattr(worker, "check_version", lambda _: None)

    def phase(_command):
        assert (tmp_path / "service/workers/734.json").exists()
        return {"returncode": 0, "termination_reason": "normal_exit"}

    monkeypatch.setattr(worker, "run_worker_phase", phase)
    with pytest.raises(SystemExit) as caught:
        worker.main(argv)
    assert caught.value.code == 0


@pytest.mark.parametrize("field,value", [("startup_probe_argv", ["/bin/false"]), ("startup_probe_timeout_seconds", 31)])
def test_fallback_must_use_identical_probe(field, value):
    first = Profile(partition="cpu", fallback_profiles=["second"], startup_probe_argv=["/bin/true"])
    second = Profile(partition="cpu", startup_probe_argv=["/bin/true"])
    Config(profiles={"first": first, "second": second})
    setattr(second, field, value)
    with pytest.raises(ValueError, match="same.*startup probe"):
        Config(profiles={"first": first, "second": second})


@pytest.mark.integration
def test_real_hq_failed_probe_consumes_nothing_and_healthy_replacement_runs_once(tmp_path):
    import contextlib
    import os
    import subprocess

    from hyper_hyperqueue.backends.hyperqueue import HQBackend, HQError, locate_binary
    from hyper_hyperqueue.config import save_config
    from hyper_hyperqueue.controller import runner_argv
    from hyper_hyperqueue.ingest import ingest_file

    try:
        binary = locate_binary()
        __import__("hyperqueue")
    except (RuntimeError, ImportError):
        pytest.skip("Pinned optional HQ runtime not installed")
    store = StateStore(tmp_path / "state")
    selected = profile("raise SystemExit(255)")
    selected.allocation_seconds = 30
    configured = Config(state_dir=str(store.root), control={"hosting": "local"}, profiles={"cpu": selected},
                        policy={"worker_idle_seconds": 1, "cleanup_seconds": 1})
    commands = tmp_path / "commands"
    commands.write_text("echo once\n")
    campaign = ingest_file(store, commands, mode="pooled", cwd=tmp_path,
                           resources={"memory_mib": 256, "runtime_seconds": 1}, profile_names=["cpu"])
    task = store.task(campaign["campaign_id"], campaign["task_ids"][0])
    atomic_json(store.root / "service/endpoint.json", {"epoch": 1, "status": "ACTIVE"})
    config_path = tmp_path / "config.json"
    save_config(configured, config_path)
    log = (tmp_path / "hq.log").open("w")
    directory = store.root / "service/hq"
    server = subprocess.Popen([binary, "--server-dir", str(directory), "server", "start", "--host", "127.0.0.1"],
                              stdout=log, stderr=log, start_new_session=True)
    process = None
    try:
        backend = HQBackend(directory, binary)
        for _ in range(100):
            try:
                backend.json("server", "info")
                break
            except HQError:
                time.sleep(0.1)
        else:
            pytest.fail("Local HQ server did not start")
        backend.submit([task], "probe_test", lambda item: runner_argv(store, item), epoch=1)
        env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
        argv = [sys.executable, "-m", "hyper_hyperqueue.worker", "--config", str(config_path),
                "--state-dir", str(store.root), "--epoch", "1", "--profile", "cpu"]
        process = subprocess.Popen(argv, env=env, stdout=log, stderr=log, start_new_session=True)
        assert process.wait(timeout=10) != 0
        record = read_json(store.root / f"service/worker_preflight/local-{process.pid}.json")
        assert record["startup_probe"]["returncode"] == 255
        assert backend.workers() == []
        assert store.attempts(task.campaign_id, task.task_id) == []
        assert not (store.root / f"service/workers/local-{process.pid}.json").exists()
        configured.profiles["cpu"].startup_probe_argv = [sys.executable, "-c", "pass"]
        save_config(configured, config_path)
        process = subprocess.Popen(argv, env=env, stdout=log, stderr=log, start_new_session=True)
        assert process.wait(timeout=10) == 0
        assert store.task_state(task.campaign_id, task.task_id)["state"] == "succeeded"
        assert len(store.attempts(task.campaign_id, task.task_id)) == 1
        assert len(backend.workers()) == 1
    finally:
        for child in (process, server):
            if child and child.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
        log.close()
