"""Recovery preparation does not duplicate successes or discard execution contracts."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hyper_hyperqueue import api, cli
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config, Profile
from hyper_hyperqueue.recovery_export import export_remaining, load_export
from hyper_hyperqueue.runner import authorize_retry, run_task
from hyper_hyperqueue.state import StateStore, atomic_json


def campaign(tmp_path, **settings):
    source = tmp_path / "commands.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"command": "echo success >> successful_runs"},
                {"command": "exit 7"},
                {"argv": ["/bin/echo", "remaining;$HOME"]},
            ]
        )
        + "\n"
    )
    config = Config(
        state_dir=str(tmp_path / "state"), profiles={"cpu": Profile(partition="cpu", memory_mib=8192)}
    )
    store = StateStore(config.state_dir)
    value = ingest_file(
        store,
        source,
        cwd=tmp_path,
        mode="pooled",
        format="jsonl",
        output={"mode": "in_place", "repeatable": True},
        **settings,
    )
    cid = value["campaign_id"]
    assert run_task(store.root, cid, value["task_ids"][0]) == 0
    assert run_task(store.root, cid, value["task_ids"][1]) == 7
    api.campaign_control(config, cid, "pause-import")
    api.campaign_control(config, cid, "pause-dispatch")
    return config, store, value


def test_remaining_export_excludes_success_preserves_commands_and_blocks_old_generation(tmp_path):
    config, store, value = campaign(tmp_path)
    result = export_remaining(config, value["campaign_id"], tmp_path / "remaining")
    assert result["count"] == 2 and result["submitted"] is False
    assert result["excluded"] == [{"task_id": value["task_ids"][0], "state": "succeeded"}]
    records = [json.loads(line) for line in Path(result["commands_file"]).read_text().splitlines()]
    assert records == [{"command": "exit 7"}, {"argv": ["/bin/echo", "remaining;$HOME"]}]
    for tid in value["task_ids"][1:]:
        assert store.task_state(value["campaign_id"], tid)["state"] == "blocked"
        assert run_task(store.root, value["campaign_id"], tid) == 75
        with pytest.raises(ValueError, match="retired into a recovery export"):
            authorize_retry(store, value["campaign_id"], tid, manual=True)
    assert (tmp_path / "successful_runs").read_text() == "success\n"
    assert export_remaining(config, value["campaign_id"], tmp_path / "remaining") == result


@pytest.mark.parametrize("state", ["unknown", "running", "submitted", "retry_wait", "blocked"])
def test_unresolved_source_refuses_export(tmp_path, state):
    config, store, value = campaign(tmp_path)
    store.update_task_state(value["campaign_id"], value["task_ids"][2], {"state": state})
    with pytest.raises(ValueError, match="Cannot export"):
        export_remaining(config, value["campaign_id"], tmp_path / "remaining")
    assert store.task_state(value["campaign_id"], value["task_ids"][1])["state"] == "failed"


def test_submission_uncertainty_or_unpaused_source_refuses_export(tmp_path):
    config, store, value = campaign(tmp_path)
    cid = value["campaign_id"]
    api.campaign_control(config, cid, "resume-dispatch")
    with pytest.raises(ValueError, match="Pause both"):
        export_remaining(config, cid, tmp_path / "remaining")
    api.campaign_control(config, cid, "pause-dispatch")
    store.put_intent(
        {
            "token": "unknown_test",
            "backend": "hq",
            "campaign_id": cid,
            "task_ids": [value["task_ids"][2]],
            "state": "unknown",
        }
    )
    with pytest.raises(ValueError, match="uncertain submission"):
        export_remaining(config, cid, tmp_path / "remaining")


def test_export_recovers_after_interrupted_publication(tmp_path, monkeypatch):
    config, store, value = campaign(tmp_path)
    from hyper_hyperqueue import recovery_export

    original = recovery_export.atomic_bytes
    monkeypatch.setattr(recovery_export, "atomic_bytes", lambda *a: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError, match="disk"):
        export_remaining(config, value["campaign_id"], tmp_path / "remaining")
    assert store.task_state(value["campaign_id"], value["task_ids"][2])["state"] == "blocked"
    monkeypatch.setattr(recovery_export, "atomic_bytes", original)
    result = export_remaining(config, value["campaign_id"], tmp_path / "remaining")
    assert result["count"] == 2


@pytest.mark.parametrize("mutation", ["commands", "manifest", "source"])
def test_tampered_export_rejected(tmp_path, mutation):
    config, store, value = campaign(tmp_path)
    result = export_remaining(config, value["campaign_id"], tmp_path / "remaining")
    path = tmp_path / "remaining/manifest.json"
    if mutation == "commands":
        Path(result["commands_file"]).write_text('{"command":"true"}\n')
    elif mutation == "manifest":
        result["execution_settings"]["output"]["repeatable"] = False
        atomic_json(path, result)
    else:
        store.update_task_state(value["campaign_id"], value["task_ids"][2], {"state": "ready"})
    with pytest.raises(ValueError, match="differ|changed"):
        load_export(path)


def test_recovery_submit_preserves_full_contract_and_changes_only_explicit_resources(tmp_path, monkeypatch):
    source = tmp_path / "commands"
    source.write_text("echo preserved\n")
    config = Config(
        state_dir=str(tmp_path / "state"), profiles={"cpu": Profile(partition="cpu", memory_mib=8192)}
    )
    store = StateStore(config.state_dir)
    value = ingest_file(
        store,
        source,
        cwd=tmp_path,
        mode="pooled",
        environment={"CUSTOM": "a b;$"},
        shell="/bin/bash",
        shell_options=["-o", "pipefail"],
        resources={"cpus": 2, "memory_mib": 1024, "runtime_seconds": 60},
        output={
            "mode": "isolated",
            "expected_paths": ["{attempt_dir}/output/final"],
            "validator_argv": ["/bin/true"],
        },
        input_fingerprint="input",
        software_fingerprint="software",
        retry={"max_attempts": 3, "reasons": ["timeout", "worker_lost"]},
    )
    for action in ("pause-import", "pause-dispatch"):
        api.campaign_control(config, value["campaign_id"], action)
    result = export_remaining(config, value["campaign_id"], tmp_path / "remaining")
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config.model_dump())
    calls = []
    monkeypatch.setattr(cli.api, "submit", lambda *a, **kw: calls.append((a, kw)) or {"test": True})
    args = [
        "--config",
        str(config_path),
        "submit",
        result["commands_file"],
        "--recovery-manifest",
        str(tmp_path / "remaining/manifest.json"),
        "--profile",
        "cpu",
        "--mem",
        "4GiB",
        "--runtime",
        "300",
    ]
    for run_number in range(2):
        if run_number:
            result["note"] = "Human-readable metadata must not create another import identity"
            atomic_json(tmp_path / "remaining/manifest.json", result)
        response = CliRunner().invoke(cli.app, args)
        assert response.exit_code == 0, response.output + str(response.exception)
    settings = calls[0][1]
    assert settings["resources"].memory_mib == 4096
    assert settings["resources"].runtime_seconds == 300
    assert settings["resources"].cpus == 2
    assert settings["environment"] == {"CUSTOM": "a b;$"}
    assert settings["cwd"] == str(tmp_path)
    assert settings["output"].expected_paths == ["{attempt_dir}/output/final"]
    assert settings["output"].validator_argv == ["/bin/true"]
    assert settings["shell_options"] == ["-o", "pipefail"]
    assert settings["retry"].reasons == ["timeout", "worker_lost"]
    assert settings["retry"].max_attempts == 3
    assert settings["input_fingerprint"] == "input" and settings["software_fingerprint"] == "software"
    assert settings["import_key"] == calls[1][1]["import_key"]
    for bad in (
        ["--import-key", "duplicate"],
        ["--cwd", "/tmp"],
        ["--nodes", "2"],
        ["--ranks", "2"],
        ["--output-mode", "in_place"],
        ["--repeatable"],
    ):
        assert CliRunner().invoke(cli.app, args + bad).exit_code != 0
    assert CliRunner().invoke(cli.app, args + ["--max-attempts", "5"]).exit_code == 0
    assert calls[-1][1]["retry"].max_attempts == 5
    assert calls[-1][1]["retry"].reasons == ["timeout", "worker_lost"]


def test_real_recovery_import_is_idempotent_without_submitting_or_reexecuting_success(tmp_path):
    config, store, original = campaign(tmp_path)
    result = export_remaining(config, original["campaign_id"], tmp_path / "remaining")
    config_path = tmp_path / "config.json"
    atomic_json(config_path, config.model_dump())
    args = [
        "--config",
        str(config_path),
        "submit",
        result["commands_file"],
        "--recovery-manifest",
        str(tmp_path / "remaining/manifest.json"),
        "--profile",
        "cpu",
        "--mem",
        "4GiB",
        "--mode",
        "pooled",
        "--no-service",
    ]
    first = CliRunner().invoke(cli.app, args)
    assert first.exit_code == 0, first.output + str(first.exception)
    second = CliRunner().invoke(cli.app, args)
    assert second.exit_code == 0, second.output + str(second.exception)
    cid = json.loads(first.output)["campaign_id"]
    assert cid == json.loads(second.output)["campaign_id"]
    assert len(store.campaigns()) == 2
    assert len(store.task_ids(cid)) == 2
    assert all(store.task(cid, tid).resources.memory_mib == 4096 for tid in store.task_ids(cid))
    assert run_task(store.root, cid, store.task_ids(cid)[1]) == 0
    assert (tmp_path / "successful_runs").read_text() == "success\n"
    changed = CliRunner().invoke(cli.app, args + ["--mem", "5GiB"])
    assert changed.exit_code != 0
    assert len(store.campaigns()) == 2 and not store.intents()
