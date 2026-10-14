"""Real HQ/server lifecycle, synthetic completed work, no Slurm calls."""

import json
import subprocess
import sys
import textwrap

import pytest

from hyper_hyperqueue.backends.hyperqueue import locate_binary
from hyper_hyperqueue.config import save_config
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config
from hyper_hyperqueue.runner import run_task
from hyper_hyperqueue.service.leadership import read_record, same_process
from hyper_hyperqueue.state import StateStore, atomic_json


@pytest.mark.integration
def test_real_supervisor_stops_hq_and_controller_after_completed_work(tmp_path):
    try:
        binary = locate_binary()
        __import__("hyperqueue")
    except (RuntimeError, ImportError):
        pytest.skip("Pinned optional HQ runtime not installed")

    store = StateStore(tmp_path / "state")
    commands = tmp_path / "commands.txt"
    commands.write_text("/bin/true\n")
    campaign = ingest_file(store, commands, mode="pooled", cwd=tmp_path)
    assert run_task(store.root, campaign["campaign_id"], campaign["task_ids"][0]) == 0
    config = Config(
        state_dir=str(store.root),
        control={"hosting": "local", "advertise_host": "127.0.0.1",
                 "hq_binary": binary, "stop_when_complete": True},
        policy={"reconcile_interval_seconds": 5},
    )
    path = tmp_path / "config.json"
    save_config(config, path)
    atomic_json(store.root / "service/deployment.json", {
        "hosting": "local", "deployment_id": "synthetic-completion",
        "desired_state": "RUNNING", "config": str(path), "allocations": [],
    })
    controller = textwrap.dedent("""
        import os, sys, time
        from types import SimpleNamespace
        from hyper_hyperqueue.config import load_config
        from hyper_hyperqueue.controller import tick
        config = load_config(sys.argv[1])
        slurm = SimpleNamespace(query_jobs=lambda: [], user='synthetic')
        tick(config, epoch=int(os.environ['HHQ_SERVER_EPOCH']), slurm=slurm)
        time.sleep(60)
    """)
    launch = textwrap.dedent("""
        import sys
        from pathlib import Path
        from hyper_hyperqueue.config import load_config
        from hyper_hyperqueue.service.supervisor import run_supervisor
        path = Path(sys.argv[1])
        run_supervisor(load_config(path), path, max_seconds=25,
                       controller_argv=[sys.executable, '-c', sys.argv[2], str(path)])
    """)
    log_path = tmp_path / "supervisor.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", launch, str(path), controller],
            stdout=log, stderr=log, start_new_session=True,
        )
        try:
            assert process.wait(timeout=35) == 0, log_path.read_text()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    completion = read_record(store.root / "service/completion.json", {})
    assert completion.get("reason") == "all_registered_work_settled", log_path.read_text()
    assert completion["counts"] == {"succeeded": 1}
    assert read_record(store.root / "service/endpoint.json")["status"] == "STOPPED"
    leader = read_record(store.root / "service/leader.json")
    assert leader["state"] == "STOPPED"
    assert not any(same_process(child) for child in leader.get("children", []))
    deployment = json.loads((store.root / "service/deployment.json").read_text())
    assert deployment["desired_state"] == "STOPPED"
