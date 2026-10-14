import json
import sys
from pathlib import Path

from typer.testing import CliRunner

from hyper_hyperqueue import api, ui
from hyper_hyperqueue.backends.slurm import SubmissionResult
from hyper_hyperqueue.cli import app
from hyper_hyperqueue.config import load_config, save_config
from hyper_hyperqueue.logs import attempt_log_path, log_directory
from hyper_hyperqueue.models import Config, OutputPolicy, Profile, Resources
from hyper_hyperqueue.runner import run_task
from hyper_hyperqueue.service.supervisor import _submit_control
from hyper_hyperqueue.state import StateStore


class FakeSlurm:
    def __init__(self):
        self.calls = []

    def query_jobs(self):
        return []

    def submit(self, script, token, **kwargs):
        self.calls.append((Path(script), kwargs))
        return SubmissionResult("accepted", "123", token, "test", [], "", "", 0, "test")


def test_separate_logs_keep_metadata_outputs_cli_and_ui_working(tmp_path, monkeypatch):
    monkeypatch.delenv("HHQ_STATE_DIR", raising=False)
    monkeypatch.delenv("HHQ_CONFIG", raising=False)
    config = Config(state_dir=str(tmp_path / "slurm"), log_dir=str(tmp_path / "logs"),
                    profiles={"cpu": Profile(partition="cpu", memory_mib=1024)})
    source = tmp_path / "commands"
    source.write_text("printf 'hello logs'; printf 'an error' >&2\n")
    backend = FakeSlurm()
    receipt = api.submit(source, config, mode="direct", profile="cpu", slurm=backend,
                         cwd=tmp_path, resources=Resources(memory_mib=512),
                         output=OutputPolicy(validator_argv=[sys.executable, "-c", "print('valid')"]))
    args = backend.calls[0][1]["extra_args"]
    assert Path(args[args.index("--output") + 1]).is_relative_to(tmp_path / "logs")
    assert backend.calls[0][0].is_relative_to(tmp_path / "slurm")
    store = StateStore(config.state_dir)
    tid = store.task_ids(receipt.campaign_id)[0]
    assert run_task(store.root, receipt.campaign_id, tid) == 0
    attempt = store.attempts(receipt.campaign_id, tid)[0]
    assert Path(attempt["attempt_dir"]).is_relative_to(tmp_path / "slurm")
    assert Path(attempt["log_dir"]).is_relative_to(tmp_path / "logs")
    assert not (Path(attempt["attempt_dir"]) / "stdout").exists()
    assert attempt_log_path(attempt, "stdout").read_text() == "hello logs"
    assert attempt_log_path(attempt, "validator.stdout").read_text() == "valid\n"
    config_path = tmp_path / "config.json"
    save_config(config, config_path)
    result = CliRunner().invoke(app, ["--config", str(config_path), "logs", tid])
    assert result.exit_code == 0 and "hello logs" in result.stdout
    page = ui.detail_page(f"/log/{tid}/{attempt['attempt_id']}/stderr", {}, config)
    assert "an error" in page
    assert run_task(store.root, receipt.campaign_id, tid) == 0
    assert len(store.attempts(receipt.campaign_id, tid)) == 1


def test_control_logs_are_separate_and_default_layout_unchanged(tmp_path):
    config = Config(state_dir=str(tmp_path / "state"), log_dir=str(tmp_path / "logs"))
    directory = tmp_path / "state/service"
    directory.mkdir(parents=True)
    deployment = {"config": str(directory / "config.json"), "allocations": []}
    backend = FakeSlurm()
    _submit_control(config, directory, deployment, "active", backend)
    script, spec = backend.calls[0]
    assert script.parent == directory
    args = spec["extra_args"]
    assert Path(args[args.index("--output") + 1]).parent == tmp_path / "logs/service"
    assert log_directory(Config(), "service", fallback=directory) == directory
    legacy = {"attempt_dir": str(tmp_path / "old")}
    assert attempt_log_path(legacy, "stdout") == tmp_path / "old/stdout"


def test_log_path_normalization_and_planning_does_not_create_logs(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"log_dir": str(tmp_path / "new_logs")}))
    assert load_config(path).log_dir == str(tmp_path / "new_logs")
    assert not (tmp_path / "new_logs").exists()
