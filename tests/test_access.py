"""Access hints do not acquire state, poll schedulers, or start a web service."""

import json
import shlex
import subprocess

import pytest
from typer.testing import CliRunner

from hyper_hyperqueue import api
from hyper_hyperqueue.access import access_info, access_lines
from hyper_hyperqueue.cli import app
from hyper_hyperqueue.models import Config


def test_access_quoted_paths_preserve_python_config_and_state_without_writes(tmp_path, monkeypatch):
    state = tmp_path / "state with spaces; $(false)"
    config_path = tmp_path / "config 'quoted'.json"
    python = tmp_path / "venv with spaces/bin/python"

    def forbidden(*args, **kwargs):
        pytest.fail("Access hints must not run external processes")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    before = list(tmp_path.iterdir())
    info = access_info(Config(state_dir=str(state)), config_path=config_path, python=python)
    assert list(tmp_path.iterdir()) == before
    assert info["controller"]["status"] == "UNKNOWN"
    assert "unknown (not published yet)" in "\n".join(access_lines(info))
    for name, tail in {"ui": ["ui"], "service_status": ["service", "status"], "access": ["access"]}.items():
        assert shlex.split(info["commands"][name]) == [str(python), "-m", "hyper_hyperqueue.cli",
                                                      "--config", str(config_path), "--state-dir", str(state), *tail]


def test_cached_service_is_sanitized_and_not_claimed_live(tmp_path):
    service = tmp_path / "service"; service.mkdir()
    (service / "leader.json").write_text(json.dumps({"state": "ACTIVE", "host": "cpu01", "slurm_job_id": "42",
                                                    "epoch": 2, "heartbeat_at": 10, "owner_id": "SECRET_LEADER"}))
    (service / "endpoint.json").write_text(json.dumps({"epoch": 2, "status": "ACTIVE", "server_dir": "/server", "owner_id": "SECRET_LEADER",
        "server": {"host": "cpu01", "port": 5432, "secret_key": "SECRET_HQ"}, "token": "SECRET_ENDPOINT"}))
    (service / "deployment.json").write_text('{"token":"SECRET_DEPLOYMENT"}')
    (tmp_path / "dashboard.json").write_text('{"url":"http://127.0.0.1:8765/?token=SECRET_UI"}')
    info = access_info(Config(state_dir=str(tmp_path)))
    assert info["controller"] == {"status": "ACTIVE", "node": "cpu01", "job_id": "42", "heartbeat_at": 10}
    assert info["hyperqueue"]["host"] == "cpu01"
    assert "not an HTML dashboard" in info["hyperqueue"]["purpose"]
    assert "may be stale" in info["evidence"]
    assert "SECRET" not in json.dumps(info)


def test_pending_controller_has_job_but_no_invented_host_or_url(tmp_path):
    service = tmp_path / "service"; service.mkdir()
    (service / "deployment.json").write_text(json.dumps({"allocations": [{"role": "active", "status": "accepted", "job_id": "123"}]}))
    info = access_info(Config(state_dir=str(tmp_path)))
    assert info["controller"] == {"status": "STARTING_OR_PENDING", "node": None, "job_id": "123", "heartbeat_at": None}
    assert info["hyperqueue"]["status"] == "UNKNOWN"
    assert "http://" not in json.dumps(info)


def test_new_submission_does_not_advertise_old_stopped_endpoint(tmp_path):
    service = tmp_path / "service"; service.mkdir()
    (service / "leader.json").write_text('{"state":"STOPPED","host":"oldnode","slurm_job_id":"1","epoch":1}')
    (service / "endpoint.json").write_text('{"status":"ACTIVE","epoch":1,"server_dir":"/old"}')
    (service / "deployment.json").write_text(json.dumps({"allocations": [{"role": "active", "status": "accepted", "job_id": "2"}]}))
    info = access_info(Config(state_dir=str(tmp_path)))
    assert info["controller"]["job_id"] == "2"
    assert info["controller"]["node"] is None
    assert info["hyperqueue"]["server_directory"] is None


def test_incomplete_cached_record_is_unknown_not_a_launch_error(tmp_path):
    service = tmp_path / "service"; service.mkdir()
    path = service / "leader.json"; path.write_text('{"secret":"do-not-echo",')
    info = access_info(Config(state_dir=str(tmp_path)))
    assert info["warnings"] == ["leader.json: cached record is unavailable or incomplete"]
    assert "do-not-echo" not in json.dumps(info)
    assert path.read_text() == '{"secret":"do-not-echo",'


def test_access_cli_reads_config_and_override_without_state_creation(tmp_path, monkeypatch):
    config = tmp_path / "config chosen.json"; state = tmp_path / "override state"
    config.write_text(json.dumps({"state_dir": str(tmp_path / "unused-state")}))

    def forbidden(*args, **kwargs):
        pytest.fail("Access may not inspect live backend state")

    monkeypatch.setattr(api, "status", forbidden)
    result = CliRunner().invoke(app, ["--config", str(config), "--state-dir", str(state), "access", "--json"])
    assert result.exit_code == 0, result.output
    info = json.loads(result.stdout)
    assert info["state_dir"] == str(state)
    assert shlex.split(info["commands"]["ui"])[-3:] == ["--state-dir", str(state), "ui"]
    assert not state.exists() and not (tmp_path / "unused-state").exists()


def test_generic_direct_submit_keeps_json_stdout_and_hints_do_not_start_service(tmp_path, monkeypatch):
    config = tmp_path / "config.json"; state = tmp_path / "state"
    config.write_text(json.dumps({"state_dir": str(state)}))
    called = []

    def submit(commands, config, **kwargs):
        called.append(kwargs)
        return {"campaign_id": "direct-test", "mode": "direct"}

    monkeypatch.setattr(api, "submit", submit)
    result = CliRunner().invoke(app, ["--config", str(config), "submit", str(tmp_path/'commands'),
                                     "--partition", "cpu", "--mode", "direct"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"campaign_id": "direct-test", "mode": "direct"}
    assert "HHQ state:" in result.stderr and "HTML dashboard:" in result.stderr
    assert "service status" in result.stderr
    assert len(called) == 1 and called[0]["mode"] == "direct"
    assert not state.exists()


def test_fifo_cached_record_does_not_block(tmp_path):
    import os

    service = tmp_path / "service"; service.mkdir()
    os.mkfifo(service / "leader.json")
    info = access_info(Config(state_dir=str(tmp_path)))
    assert info["controller"]["status"] == "UNKNOWN"
    assert info["warnings"] == ["leader.json: cached record is unavailable or incomplete"]


@pytest.mark.parametrize("endpoint", [{}, {"epoch": 1, "owner_id": "different", "status": "ACTIVE"}])
def test_active_leader_needs_matching_published_endpoint(tmp_path, endpoint):
    service = tmp_path / "service"; service.mkdir()
    (service / "leader.json").write_text('{"state":"ACTIVE","epoch":1,"owner_id":"owner"}')
    (service / "endpoint.json").write_text(json.dumps(endpoint))
    info = access_info(Config(state_dir=str(tmp_path)))
    assert info["controller"]["status"] == "UNCONFIRMED"
    assert info["hyperqueue"]["status"] != "ACTIVE"


def test_stopped_leader_overrides_same_epoch_stale_active_endpoint(tmp_path):
    service = tmp_path / "service"; service.mkdir()
    (service / "leader.json").write_text('{"state":"STOPPED","epoch":1,"owner_id":"owner"}')
    (service / "endpoint.json").write_text('{"status":"ACTIVE","epoch":1,"owner_id":"owner","server_dir":"/old"}')
    info = access_info(Config(state_dir=str(tmp_path)))
    assert info["controller"]["status"] == info["hyperqueue"]["status"] == "STOPPED"
    assert info["hyperqueue"]["server_directory"] is None
