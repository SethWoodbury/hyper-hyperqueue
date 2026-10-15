"""Adapter CLI is opt-in and keeps preview distinct from real admission."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hyper_hyperqueue.cli import app
from hyper_hyperqueue.integrations.rfd4 import prepare, submission
from hyper_hyperqueue.integrations.rfd4.cli import _options


def test_rfd4_help_without_native_dependencies():
    result = CliRunner().invoke(app, ["rfd4", "--help"])
    assert result.exit_code == 0, result.output
    assert all(name in result.output for name in ("plan", "prepare", "preview", "launch", "collection-ready"))


def test_python_module_invocation_keeps_explicit_config_in_nested_command(tmp_path):
    config = tmp_path / "chosen-config-does-not-exist.json"
    prepared = tmp_path / "prepared-does-not-exist.json"
    result = subprocess.run([sys.executable, "-m", "hyper_hyperqueue.cli", "--config", str(config),
                             "rfd4", "preview", str(prepared)], capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert str(config) in result.stderr.replace("\n", "")
    assert str(prepared) not in result.stderr.replace("\n", "")


def test_plan_cli_only_reads_and_validates_options(tmp_path, monkeypatch):
    options = tmp_path / "options.yaml"
    options.write_text("runtime: {argv: [/example/python]}\nworkloads: {}\n")
    calls = []

    def fake_plan(manifest, **kwargs):
        calls.append((manifest, kwargs))
        return {"plan_id": "plan", "summary": {"tasks": 1}, "warnings": [], "bundles": []}

    def forbidden(*args, **kwargs):
        pytest.fail("Plan may not prepare or launch")

    monkeypatch.setattr(prepare, "plan_sweep", fake_plan)
    monkeypatch.setattr(prepare, "prepare_sweep", forbidden)
    monkeypatch.setattr(submission, "launch", forbidden)
    manifest = tmp_path / "native.json"
    result = CliRunner().invoke(app, ["rfd4", "plan", str(manifest), str(options)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["summary"] == {"tasks": 1}
    assert calls == [(manifest, {"runtime": {"argv": ["/example/python"]}, "workloads": {}})]


@pytest.mark.parametrize("content", ["[]", "runtime: {}", "runtime: {}\nworkloads: {}\nstart: true"])
def test_options_reject_implicit_or_unknown_execution_controls(tmp_path, content):
    path = tmp_path / "options.yaml"
    path.write_text(content)
    with pytest.raises(ValueError):
        _options(path)


@pytest.mark.parametrize("command", ["preview", "status", "collection-ready", "launch"])
def test_cli_routes_prepared_path_and_explicit_config(tmp_path, monkeypatch, command):
    config_path = tmp_path / "config.json"
    state_path = tmp_path / "no-state-mutation"
    config_path.write_text(json.dumps({"state_dir": str(state_path)}))
    calls = []

    def target(prepared, config, **kwargs):
        calls.append((prepared, config.state_dir, kwargs))
        return {"test_route": command}

    monkeypatch.setattr(submission, command.replace("-", "_"), target)
    prepared_path = tmp_path / "prepared.json"
    result = CliRunner().invoke(app, ["--config", str(config_path), "rfd4", command, str(prepared_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"test_route": command}
    expected_kwargs = {"reconcile": False} if command in {"status", "collection-ready"} else {}
    assert calls == [(Path(prepared_path), str(state_path), expected_kwargs)]
    assert not state_path.exists()
