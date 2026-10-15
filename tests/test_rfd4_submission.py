import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

from hyper_hyperqueue import api
from hyper_hyperqueue.integrations.rfd4 import submission as rfd4
from hyper_hyperqueue.models import Budget, Config, Profile, Resources
from hyper_hyperqueue.planner import capacity_plan, compatible
from hyper_hyperqueue.runner import _validate
from hyper_hyperqueue.state import StateStore, atomic_json, fingerprint, read_json


@pytest.fixture
def config(tmp_path):
    return Config(
        state_dir=str(tmp_path / "state"),
        budgets={"gpu": Budget(partitions=["gpu"], running_target=1, pending_reserve=1, max_submitted=2)},
        profiles={
            "gpu": Profile(
                partition="gpu", gres="gpu:1", budget="gpu", cpus=2, memory_mib=8192, allocation_seconds=3600
            ),
            "cpu": Profile(partition="cpu", cpus=1, memory_mib=1024),
        },
    )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Prepared-file verification itself is covered by the preparation tests."""
    groups, hashes = [], {}
    for method in ("gradient", "beam"):
        path = tmp_path / f"{method}.jsonl"
        path.write_text(json.dumps({"argv": ["/bin/true"]}) + "\n")
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        groups.append(
            {
                "method": method,
                "commands_file": str(path),
                "profile": "gpu",
                "resources": Resources(gpus=1, memory_mib=2048, runtime_seconds=60).model_dump(),
                "max_attempts": 2,
                "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
                "bundle_ids": [method],
            }
        )
    document = {
        "schema": "hhq-rfd4-prepared/1",
        "plan_id": "test-sweep",
        "groups": groups,
        "file_sha256": hashes,
        "manifest_path": str(tmp_path / "prepared.json"),
        "manifest_sha256": "a" * 64,
    }
    atomic_json(document["manifest_path"], document)
    monkeypatch.setattr(rfd4, "_verified", lambda value, **kwargs: document)
    return document


def test_preview_is_detached_and_preflights_all_groups(prepared, config, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Preview/import preflight must not submit")

    monkeypatch.setattr(api, "submit", forbidden)
    result = rfd4.preview(Path(prepared["manifest_path"]), config)
    assert result["submitted"] is False
    assert [g["mode"] for g in result["groups"]] == ["pooled", "pooled"]
    assert not Path(config.state_dir).exists()
    prepared["groups"][1]["resources"]["memory_mib"] = 99999
    with pytest.raises(ValueError, match="INSUFFICIENT_CPU_OR_HOST_RAM"):
        rfd4.launch(prepared, config)
    assert not Path(config.state_dir).exists()


def test_imports_are_idempotent_before_one_service_start(prepared, config, monkeypatch):
    from hyper_hyperqueue.service import supervisor

    starts = []

    def start(c, **kwargs):
        campaigns = StateStore(c.state_dir).campaigns()
        assert len(campaigns) == 2
        starts.append(kwargs)
        return {"state": "submitted", "job_id": "mock-control"}

    monkeypatch.setattr(supervisor, "start_service", start)
    first = rfd4.launch(prepared, config)
    second = rfd4.launch(prepared, config)
    assert first["groups"] == second["groups"]
    assert len(starts) == 1
    store = StateStore(config.state_dir)
    assert len(store.campaigns()) == 2
    for c in store.campaigns():
        task = store.task(c["campaign_id"], c["task_ids"][0])
        assert task.output.repeatable and task.output.mode == "isolated"
        assert task.output.expected_paths == ["{attempt_dir}/output/rfd4_completion.json"]
        assert task.retry.reasons == ["worker_lost", "preempted"]
        assert task.retry.max_attempts == 2
        assert task.input_fingerprint == prepared["manifest_sha256"]
        assert task.environment == {"PYTHONDONTWRITEBYTECODE": "1"}
        assert store.task_state(c["campaign_id"], task.task_id)["state"] == "ready"


def test_method_order_preferences_reach_generic_task_priority(prepared, config):
    prepared["groups"][0]["priority"] = 2
    prepared["groups"][1]["priority"] = 1
    preview = rfd4.preview(prepared, config)
    assert [group["priority"] for group in preview["groups"]] == [2, 1]
    launched = rfd4.launch(prepared, config, start_service=False)
    store = StateStore(config.state_dir)
    for method, priority in (("gradient", 2), ("beam", 1)):
        campaign_id = launched["groups"][method]["campaign_id"]
        campaign = store.campaign(campaign_id)
        assert all(store.task(campaign_id, task_id).priority == priority for task_id in campaign["task_ids"])


def test_explicit_retry_policy_reaches_generic_task(prepared, config):
    group = prepared["groups"][0]
    group["max_attempts"] = 6
    group["retry"] = {"max_attempts": 6, "reasons": ["worker_lost", "preempted", "timeout", "host_oom"],
                      "oom_memory_multipliers": [1.25, 1.5, 2.0]}
    launched = rfd4.launch(prepared, config, start_service=False)
    store = StateStore(config.state_dir)
    campaign_id = launched["groups"][group["method"]]["campaign_id"]
    campaign = store.campaign(campaign_id)
    task = store.task(campaign_id, campaign["task_ids"][0])
    assert task.retry.max_attempts == 6
    assert task.retry.reasons == group["retry"]["reasons"]
    assert task.retry.oom_memory_multipliers == [1.25, 1.5, 2.0]


def test_partial_import_recovery_uses_same_campaign_identity(prepared, config, monkeypatch):
    original = api.submit
    seen = []

    def lose_second_ack(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(result.campaign_id)
        if len(seen) == 2:
            raise TimeoutError("lost import acknowledgement after immutable write")
        return result

    monkeypatch.setattr(api, "submit", lose_second_ack)
    with pytest.raises(TimeoutError):
        rfd4.launch(prepared, config, start_service=False)
    receipt = read_json(rfd4._receipt_path(prepared, config))
    assert receipt["state"] == "partial"
    assert receipt["groups"]["beam"]["state"] == "uncertain_import"
    monkeypatch.setattr(api, "submit", original)
    rfd4.launch(prepared, config, start_service=False)
    assert sorted(c["campaign_id"] for c in StateStore(config.state_dir).campaigns()) == sorted(seen)


def test_uncertain_service_start_is_not_blindly_replayed(prepared, config, monkeypatch):
    from hyper_hyperqueue.service import supervisor

    calls = []

    def lost_reply(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("scheduler reply lost")

    monkeypatch.setattr(supervisor, "start_service", lost_reply)
    with pytest.raises(TimeoutError):
        rfd4.launch(prepared, config)
    with pytest.raises(ValueError, match="Uncertain prior service start"):
        rfd4.launch(prepared, config)
    assert calls == [1]
    receipt = read_json(rfd4._receipt_path(prepared, config))
    assert receipt["service_status"] == "unknown" and len(receipt["groups"]) == 2


def test_frozen_service_config_rejects_before_any_import(prepared, config):
    changed = config.model_copy(deep=True)
    changed.profiles["gpu"].memory_mib += 1
    snapshot = Path(config.state_dir) / "service" / "snapshot.json"
    atomic_json(snapshot, changed.model_dump())
    atomic_json(snapshot.with_name("deployment.json"), {"config": str(snapshot)})
    with pytest.raises(ValueError, match="frozen configuration"):
        rfd4.launch(prepared, config, start_service=False)
    assert StateStore(config.state_dir).campaigns() == []
    assert not rfd4._receipt_path(prepared, config).exists()


def test_launch_receipt_binds_resource_configuration(prepared, config):
    rfd4.launch(prepared, config, start_service=False)
    config.profiles["gpu"].memory_mib += 1
    with pytest.raises(ValueError, match="configuration changed"):
        rfd4.launch(prepared, config, start_service=False)


@pytest.mark.parametrize(
    ("pending_floor", "omitted"),
    [
        (0, ("min_pending",)),
        (0, ("budget_priority",)),
        (0, ("min_pending", "budget_priority")),
        (1, ("budget_priority",)),
        (0, ("stop_when_complete",)),
        (0, ("min_pending", "budget_priority", "stop_when_complete")),
        (0, ("fill_running_target",)),
        (1, ("fill_running_target",)),
        (0, ("auto_renew", "renew_lead_seconds")),
        (0, ("speculative_partition_workers",)),
        (0, ("min_pending", "fill_running_target", "budget_priority", "speculative_partition_workers", "stop_when_complete", "auto_renew", "renew_lead_seconds")),
    ],
)
def test_legacy_launch_receipt_preserves_default_policy_identity(prepared, config, pending_floor, omitted):
    config.budgets["gpu"].min_pending = pending_floor
    first = rfd4.launch(prepared, config, start_service=False)
    path = rfd4._receipt_path(prepared, config)
    legacy = config.model_dump()
    if "min_pending" in omitted:
        for budget in legacy["budgets"].values():
            del budget["min_pending"]
    if "fill_running_target" in omitted:
        for budget in legacy["budgets"].values():
            del budget["fill_running_target"]
    if "budget_priority" in omitted:
        del legacy["policy"]["budget_priority"]
    if "speculative_partition_workers" in omitted:
        del legacy["policy"]["speculative_partition_workers"]
    if "stop_when_complete" in omitted:
        del legacy["control"]["stop_when_complete"]
    for key in ("auto_renew", "renew_lead_seconds"):
        if key in omitted:
            del legacy["control"][key]
    receipt = read_json(path)
    receipt["config_sha256"] = fingerprint(legacy)
    atomic_json(path, receipt)

    replay = rfd4.launch(prepared, config, start_service=False)
    assert replay["config_sha256"] == receipt["config_sha256"]
    assert read_json(path)["config_sha256"] == receipt["config_sha256"]
    assert {
        method: entry["campaign_id"] for method, entry in replay["groups"].items()
    } == {method: entry["campaign_id"] for method, entry in first["groups"].items()}
    assert len(StateStore(config.state_dir).campaigns()) == 2


@pytest.mark.parametrize("change", ["pending_floor", "fill_running_target", "budget_priority", "speculative_partition_workers", "memory", "manifest", "stop_when_complete", "auto_renew", "renew_lead_seconds"])
def test_legacy_launch_receipt_still_rejects_changed_configuration(prepared, config, change):
    rfd4.launch(prepared, config, start_service=False)
    path = rfd4._receipt_path(prepared, config)
    legacy = config.model_dump()
    for budget in legacy["budgets"].values():
        del budget["min_pending"]
        del budget["fill_running_target"]
    del legacy["policy"]["budget_priority"]
    del legacy["policy"]["speculative_partition_workers"]
    del legacy["control"]["stop_when_complete"]
    del legacy["control"]["auto_renew"]
    del legacy["control"]["renew_lead_seconds"]
    receipt = read_json(path)
    receipt["config_sha256"] = fingerprint(legacy)
    atomic_json(path, receipt)
    if change == "pending_floor":
        config.budgets["gpu"].min_pending = 1
    elif change == "fill_running_target":
        config.budgets["gpu"].fill_running_target = True
    elif change == "budget_priority":
        config.policy.budget_priority = ["gpu"]
    elif change == "speculative_partition_workers":
        config.policy.speculative_partition_workers = True
    elif change == "memory":
        config.profiles["gpu"].memory_mib += 1
    elif change == "stop_when_complete":
        config.control.stop_when_complete = True
    elif change == "auto_renew":
        config.control.auto_renew = True
    elif change == "renew_lead_seconds":
        config.control.renew_lead_seconds = 700
    else:
        prepared["manifest_sha256"] = "b" * 64
    before = path.read_bytes()
    with pytest.raises(ValueError, match="configuration changed"):
        rfd4.launch(prepared, config, start_service=False)
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["pending_floor", "fill_running_target", "budget_priority", "speculative_partition_workers", "stop_when_complete", "auto_renew", "renew_lead_seconds"])
def test_service_configuration_normalizes_omitted_policy_defaults(config, change):
    snapshot = Path(config.state_dir) / "service" / "snapshot.json"
    legacy = config.model_dump()
    for budget in legacy["budgets"].values():
        del budget["min_pending"]
        del budget["fill_running_target"]
    del legacy["policy"]["budget_priority"]
    del legacy["policy"]["speculative_partition_workers"]
    del legacy["control"]["stop_when_complete"]
    del legacy["control"]["auto_renew"]
    del legacy["control"]["renew_lead_seconds"]
    atomic_json(snapshot, legacy)
    atomic_json(snapshot.with_name("deployment.json"), {"config": str(snapshot)})
    before = snapshot.read_bytes()
    api.assert_service_config_compatible(config)
    assert snapshot.read_bytes() == before
    if change == "pending_floor":
        config.budgets["gpu"].min_pending = 1
    elif change == "fill_running_target":
        config.budgets["gpu"].fill_running_target = True
    elif change == "speculative_partition_workers":
        config.policy.speculative_partition_workers = True
    elif change == "stop_when_complete":
        config.control.stop_when_complete = True
    elif change == "auto_renew":
        config.control.auto_renew = True
    elif change == "renew_lead_seconds":
        config.control.renew_lead_seconds = 700
    else:
        config.policy.budget_priority = ["gpu"]
    with pytest.raises(ValueError, match="frozen configuration"):
        api.assert_service_config_compatible(config)


def test_completion_placeholder_and_terminal_collection_gate(prepared, config, tmp_path):
    assert rfd4.collection_ready(prepared, config)["counts"] == {"not_imported": 2}
    assert not Path(config.state_dir).exists()
    rfd4.launch(prepared, config, start_service=False)
    assert not rfd4.collection_ready(prepared, config)["ready"]
    store = StateStore(config.state_dir)
    campaigns = store.campaigns()
    cid, tid = campaigns[0]["campaign_id"], campaigns[0]["task_ids"][0]
    task = store.task(cid, tid)
    attempt = tmp_path / "attempt"
    (attempt / "output").mkdir(parents=True)
    assert not _validate(task, attempt, {})["valid"]
    atomic_json(attempt / "output/rfd4_completion.json", {"status": "complete"})
    assert _validate(task, attempt, {})["valid"]
    store.update_task_state(cid, tid, {"state": "succeeded"})
    assert "missing_validated_completion" in {
        b["reason"] for b in rfd4.collection_ready(prepared, config)["blockers"]
    }
    atomic_json(
        store.campaign_dir(cid) / "completions" / f"{tid}.json",
        {"task_fingerprint": fingerprint(task.model_dump(mode="json"))},
    )
    cid2, tid2 = campaigns[1]["campaign_id"], campaigns[1]["task_ids"][0]
    store.update_task_state(cid2, tid2, {"state": "failed", "reason": "worker_lost"})
    assert rfd4.collection_ready(prepared, config)["blockers"][0]["reason"] == "retry_decision_pending"
    store.update_task_state(cid2, tid2, {"state": "failed", "reason": "nonzero_exit"})
    result = rfd4.collection_ready(prepared, config)
    assert result["ready"] and result["successes"] == result["failures"] == 1
    store.update_task_state(cid2, tid2, {"state": "unknown"})
    assert not rfd4.collection_ready(prepared, config)["ready"]


def test_profiles_use_fixed_additive_memory_and_preserve_shared_budget(config):
    before = config.model_dump()
    requests = {
        method: Resources(
            gpus=1, memory_mib=memory, cpus=1, runtime_seconds=3300, runtime_confidence="measured"
        )
        for method, memory in [("gradient", 4096), ("beam", 6144)]
    }
    result = rfd4.method_profiles(config, requests, ["gpu"])
    assert config.model_dump() == before
    assert result.budgets == config.budgets
    for method, request in requests.items():
        name = result.workloads[f"rfd4-{method}"][0]
        clone = result.profiles[name]
        assert clone.memory_mib == request.memory_mib + result.policy.worker_overhead_mib
        assert clone.cpus == 2 and clone.max_workers == config.profiles["gpu"].max_workers
        assert clone.budget == "gpu" and compatible(request, clone, result)[0]
    assert not compatible(requests["beam"], result.profiles["rfd4-gradient-gpu"], result)[0]
    tasks = [
        {
            "task_id": f"{method}{i}",
            "resources": request.model_dump(),
            "allowed_profiles": result.workloads[f"rfd4-{method}"],
        }
        for method, request in requests.items()
        for i in range(3)
    ]
    actions = capacity_plan(result, tasks, [], [])["actions"]
    assert len(actions) == 2  # Shared budget is two, not two per method.


def test_method_routes_can_reserve_training_for_beam(config):
    config.profiles["train"] = config.profiles["gpu"].model_copy(update={"partition": "gpu-train"})
    request = Resources(gpus=1, memory_mib=2048, runtime_seconds=60, runtime_confidence="declared")
    requests = {"gradient": request, "beam": request}
    result = rfd4.method_profiles(config, requests, ["gpu", "train"],
                                 eligible_profiles_by_method={"gradient": ["gpu"]})
    assert result.workloads["rfd4-gradient"] == ["rfd4-gradient-gpu"]
    assert result.workloads["rfd4-beam"] == ["rfd4-beam-gpu", "rfd4-beam-train"]
    assert "rfd4-gradient-train" not in result.profiles
    for bad in ({"unknown": ["gpu"]}, {"gradient": []}, {"gradient": ["missing"]}):
        with pytest.raises(ValueError):
            rfd4.method_profiles(config, requests, ["gpu", "train"], eligible_profiles_by_method=bad)


@pytest.mark.parametrize(
    "bad",
    [
        {"gpus": 1, "memory_mib": 2048},
        {"gpus": 1, "runtime_seconds": 30, "runtime_confidence": "measured"},
        {"gpus": 1, "memory_mib": 2048, "runtime_seconds": 30},
    ],
)
def test_profiles_reject_missing_or_unknown_calibration(config, bad):
    with pytest.raises(ValueError, match="explicit RAM"):
        rfd4.method_profiles(config, {"beam": bad}, ["gpu"])


def test_declared_initial_calibration_is_explicit(config):
    result = rfd4.method_profiles(
        config,
        {"beam": {"gpus": 1, "memory_mib": 2048, "runtime_seconds": 60, "runtime_confidence": "declared"}},
        ["gpu"],
    )
    assert result.profiles["rfd4-beam-gpu"].memory_mib == 2304


def test_service_owned_cancellation_blocks_collection_until_retry_reconciles(prepared, config):
    rfd4.launch(prepared, config, start_service=False)
    store = StateStore(config.state_dir)
    campaigns = store.campaigns()
    cid, tid = campaigns[0]["campaign_id"], campaigns[0]["task_ids"][0]
    second = campaigns[1]
    store.update_task_state(
        second["campaign_id"], second["task_ids"][0], {"state": "failed", "reason": "nonzero_exit"}
    )
    attempt_dir = store.campaign_dir(cid) / "attempts" / tid / "000001_test"
    atomic_json(
        attempt_dir / "attempt.json",
        {
            "attempt_id": "000001_test",
            "attempt_dir": str(attempt_dir),
            "context": {"environment": {"HHQ_WORKER_EPOCH": "2", "SLURM_JOB_ID": "123"}},
        },
    )
    atomic_json(attempt_dir / "result.json", {"status": "canceled", "reason": "signal"})
    assert rfd4.collection_ready(prepared, config)["ready"]
    intent_path = store.root / "service/recovery-intent-2.json"
    atomic_json(
        intent_path,
        {
            "schema": "hhq-service-fencing-intent/1",
            "reason": "generation_recovery",
            "previous_epoch": 2,
            "created_at": time.time(),
            "allocations": [{"job_id": "123", "token": "owned_worker"}],
        },
    )
    assert not rfd4.collection_ready(prepared, config)["ready"]
    store.put_intent(
        {
            "backend": "worker",
            "token": "owned_worker",
            "profile": "gpu",
            "job_id": "123",
            "epoch": 2,
            "state": "terminal",
        }
    )
    atomic_json(
        store.root / "service/recovery-2.json",
        {
            "ready": True,
            "proof": {
                "schema": "hhq-fencing-proof/1",
                "quiescent": True,
                "previous_epoch": 2,
                "intent": {
                    "path": str(intent_path),
                    "sha256": hashlib.sha256(intent_path.read_bytes()).hexdigest(),
                },
                "allocations": [
                    {
                        "job_id": "123",
                        "token": "owned_worker",
                        "owned": True,
                        "live": False,
                        "accounting_state": "CANCELLED",
                    }
                ],
            },
        },
    )
    assert not rfd4.collection_ready(prepared, config)["ready"]
    store.update_task_state(cid, tid, {"state": "canceled", "reason": "user_cancelled"})
    assert rfd4.collection_ready(prepared, config)["ready"]


def test_aggregation_never_runs_before_logical_completion(prepared, config, tmp_path, monkeypatch):
    path = tmp_path / "collect.txt"
    path.write_text("true\n")
    monkeypatch.setattr(api, "submit", lambda *a, **k: pytest.fail("Unexpected submission"))
    with pytest.raises(ValueError, match="not ready"):
        rfd4.submit_aggregation(prepared, config, path, profile="cpu", resources=Resources())


def test_real_preparation_import_contract_and_tamper_gate(tmp_path, config):
    from hyper_hyperqueue.integrations.rfd4.prepare import prepare_sweep

    script = tmp_path / "synthetic_runtime.py"
    script.write_text("# Schema integration fixture; never executed.\n")
    source = tmp_path / "native_manifest.json"
    source.write_text("{}\n")
    ligand = tmp_path / "input.cif"
    ligand.write_text("data_synthetic\n")
    dotenv = tmp_path / ".env.digs"
    dotenv.write_text("EXAMPLE=fixture\n")
    pins = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (script, source, dotenv, ligand)
    }
    source_binding = {"path": str(source), "sha256": pins[str(source)]}
    bundles = []
    for method in ("gradient", "beam"):
        bundle = {
            "schema": "hhq-rfd4-bundle/1",
            "repo": str(tmp_path),
            "repo_commit": "test",
            "source_manifest": source_binding,
            "source_hashes": pins,
            "software_fingerprint": fingerprint(pins),
            "runtime": {
                "argv": [sys.executable],
                "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
                "source_hashes": {},
            },
            "runtime_script": str(script),
            "reuse_model": False,
            "method": method,
            "workload": {
                "profile": "gpu",
                "resources": Resources(
                    gpus=1, memory_mib=2048, runtime_seconds=120, runtime_confidence="declared"
                ).model_dump(),
                "max_attempts": 2,
                "runs_per_task": 1,
            },
            "rows": [
                {
                    "run_id": method,
                    "output_dir": str(tmp_path / "science" / method),
                    "config": str(source),
                    "config_sha256": pins[str(source)],
                    "input": str(ligand),
                    "input_sha256": pins[str(ligand)],
                    "environment": {"CLUSTER": "digs"},
                    "dotenv": {"cluster": "digs", "path": str(dotenv), "sha256": pins[str(dotenv)]},
                }
            ],
        }
        bundle["bundle_id"] = "b_" + fingerprint(bundle)[:24]
        bundles.append(bundle)
    plan = {
        "schema": "hhq-rfd4-plan/1",
        "source_manifest": source_binding,
        "software_fingerprint": fingerprint(pins),
        "bundles": bundles,
        "summary": {"scientific_calls": 2, "tasks": 2},
        "warnings": [],
    }
    plan["plan_id"] = "p_" + fingerprint(plan)[:24]
    prepared = prepare_sweep(plan, tmp_path / "prepared")
    assert rfd4.preview(Path(prepared["manifest_path"]), config)["submitted"] is False
    assert not Path(config.state_dir).exists()
    result = rfd4.launch(prepared, config, start_service=False)
    assert result["state"] == "imported"
    assert rfd4.status(prepared, config)["counts"] == {"ready": 2}
    for live_file in (ligand, script):
        original = live_file.read_bytes()
        live_file.write_bytes(original + b"# routine source update\n")
        assert rfd4.status(prepared, config)["counts"] == {"ready": 2}
        assert rfd4.collection_ready(prepared, config)["counts"] == {"ready": 2}
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            rfd4.preview(prepared, config)
        with pytest.raises(ValueError, match="SHA256 mismatch"):
            rfd4.launch(prepared, config, start_service=False)
        live_file.write_bytes(original)
    command = Path(prepared["groups"][0]["commands_file"])
    command.write_text(command.read_text() + json.dumps({"argv": ["false"]}) + "\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        rfd4.status(prepared, config)
    fresh = config.model_copy(update={"state_dir": str(tmp_path / "never-imported")})
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        rfd4.launch(prepared, fresh, start_service=False)
    assert not Path(fresh.state_dir).exists()


def test_launch_verifies_sources_once_then_imports_all_preflighted_groups(prepared, config, monkeypatch):
    seen = []

    def verify(value, *, check_sources=True):
        seen.append(check_sources)
        return prepared

    monkeypatch.setattr(rfd4, "_verified", verify)
    result = rfd4.launch(prepared, config, start_service=False)
    assert seen == [True]
    assert result["state"] == "imported"
    assert len(StateStore(config.state_dir).campaigns()) == len(prepared["groups"])


def test_public_preview_still_requires_source_verification(prepared, config, monkeypatch):
    def reject(value, *, check_sources=True):
        assert check_sources is True
        raise ValueError("changed native source")

    monkeypatch.setattr(rfd4, "_verified", reject)
    with pytest.raises(ValueError, match="changed native source"):
        rfd4.preview(prepared, config)
    with pytest.raises(ValueError, match="changed native source"):
        rfd4.launch(prepared, config, start_service=False)
    assert not Path(config.state_dir).exists()


def test_pre_recovery_config_receipt_compatibility_is_disabled_defaults_only(config):
    from hyper_hyperqueue.integrations.rfd4.submission import _config_fingerprints

    legacy = config.model_dump()
    for name in ('pending_reserves_after_growth', 'cancel_pending_when_assigned',
                 'allocation_gpu_hours_budget', 'allocation_budget_allowance'):
        legacy['policy'].pop(name)
    for budget in legacy['budgets'].values():
        budget.pop('max_pending')
        budget.pop('max_running')
    for profile in legacy['profiles'].values():
        profile.pop('fallback_profiles')
    expected = fingerprint(legacy)
    assert expected in _config_fingerprints(config)
    config.policy.cancel_pending_when_assigned = True
    assert expected not in _config_fingerprints(config)


def test_rfd4_digs_gpu_guard_is_scoped_and_does_not_widen_custom_constraints(config):
    request = Resources(gpus=1, memory_mib=2048, runtime_seconds=60, runtime_confidence="declared")
    config.cluster = "digs"
    config.profiles["gpu"].constraint = "UW&(A4000|TitanRTX)"
    before = config.model_dump()
    result = rfd4.method_profiles(config, {"beam": request}, ["gpu"])
    assert config.model_dump() == before
    assert result.profiles["gpu"].constraint == "UW&(A4000|TitanRTX)"
    assert result.profiles["rfd4-beam-gpu"].constraint == (
        "(UW&(A4000|TitanRTX))&(Ampere|Lovelace|Hopper|Blackwell)")
    assert rfd4.method_profiles(config, {"beam": request}, ["gpu"], compatibility_constraint=None
                              ).profiles["rfd4-beam-gpu"].constraint == config.profiles["gpu"].constraint


def test_rfd4_default_features_and_explicit_other_cluster_override(config):
    request = Resources(gpus=1, memory_mib=2048, runtime_seconds=60, runtime_confidence="declared")
    config.cluster = "digs"
    guarded = rfd4.method_profiles(config, {"gradient": request}, ["gpu"])
    assert guarded.profiles["rfd4-gradient-gpu"].constraint == rfd4.DIGS_RFD4_GPU_CONSTRAINT
    config.cluster = "different_site"
    assert rfd4.method_profiles(config, {"gradient": request}, ["gpu"]
                              ).profiles["rfd4-gradient-gpu"].constraint is None
    config.profiles["gpu"].constraint = "rack7"
    explicit = rfd4.method_profiles(config, {"gradient": request}, ["gpu"], compatibility_constraint="cuda80|cuda90")
    assert explicit.profiles["rfd4-gradient-gpu"].constraint == "(rack7)&(cuda80|cuda90)"


@pytest.mark.parametrize("constraint", [False, 42, "", "\nA4000"])
def test_rfd4_rejects_invalid_compatibility_override(config, constraint):
    request = Resources(gpus=1, memory_mib=2048, runtime_seconds=60, runtime_confidence="declared")
    with pytest.raises(ValueError, match="compatibility_constraint"):
        rfd4.method_profiles(config, {"gradient": request}, ["gpu"], compatibility_constraint=constraint)


@pytest.mark.parametrize("late_constraint", [None, "TitanRTX", "UW&(A4000|TitanRTX)"])
def test_final_rfd4_guard_cannot_be_removed_by_late_profile_override(config, late_constraint):
    config.cluster = "digs"
    request = Resources(gpus=1, memory_mib=2048, runtime_seconds=60, runtime_confidence="declared")
    configured = rfd4.method_profiles(config, {"beam": request}, ["gpu"])
    configured.profiles["rfd4-beam-gpu"].constraint = late_constraint
    before = configured.model_dump()
    final = rfd4.apply_rfd4_gpu_constraints(configured, ["rfd4-beam-gpu"])
    expected = (f"({late_constraint})&({rfd4.DIGS_RFD4_GPU_CONSTRAINT})" if late_constraint
                else rfd4.DIGS_RFD4_GPU_CONSTRAINT)
    assert final.profiles["rfd4-beam-gpu"].constraint == expected
    assert configured.model_dump() == before
    assert final.profiles["gpu"] == configured.profiles["gpu"]
    assert rfd4.apply_rfd4_gpu_constraints(final, ["rfd4-beam-gpu"]).model_dump() == final.model_dump()
    opted_out = rfd4.apply_rfd4_gpu_constraints(configured, ["rfd4-beam-gpu"], compatibility_constraint=None)
    assert opted_out.model_dump() == before


def test_final_rfd4_gpu_guard_refuses_generic_or_unknown_profiles(config):
    for names in (["gpu"], ["rfd4-missing"], ["rfd4-missing", "rfd4-missing"]):
        with pytest.raises(ValueError, match="RFD4 profile names"):
            rfd4.apply_rfd4_gpu_constraints(config, names)
