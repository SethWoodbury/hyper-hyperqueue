"""Durable evidence, missing data and cache behavior, without scheduler calls."""

import datetime as dt
import json

import pytest

from hyper_hyperqueue.dashboard import (
    DashboardSnapshot,
    DashboardUnavailable,
    gpu_label,
    read_task_state,
)
from hyper_hyperqueue.ingest import ingest_file
from hyper_hyperqueue.models import Config, Resources
from hyper_hyperqueue.state import StateStore, atomic_json, fingerprint


@pytest.fixture
def campaign(tmp_path):
    source = tmp_path / "commands.txt"
    source.write_text("true\nfalse\ntrue\ntrue\n")
    store = StateStore(tmp_path / "state")
    record = ingest_file(
        store,
        source,
        mode="pooled",
        cwd=tmp_path,
        workload="rfd4-beam",
        resources=Resources(gpus=1, memory_mib=8192, cpus=2),
        import_key="dashboard",
    )
    return Config(state_dir=str(store.root)), store, record


def evidence(
    store,
    campaign,
    index,
    *,
    number=1,
    status="succeeded",
    duration=10,
    peak=1024,
    valid=True,
    finished=100,
    complete=True,
):
    cid, tid = campaign["campaign_id"], campaign["task_ids"][index]
    spec = store.task(cid, tid)
    fp = fingerprint(spec.model_dump(mode="json"))
    aid = f"{number:06d}_example"
    directory = store.campaign_dir(cid) / "attempts" / tid / aid
    attempt = {
        "attempt_id": aid,
        "attempt_number": number,
        "task_id": tid,
        "campaign_id": cid,
        "task_fingerprint": fp,
        "started_at": dt.datetime.fromtimestamp(50, dt.UTC).isoformat(),
        "attempt_dir": str(directory),
        "context": {},
    }
    atomic_json(directory / "attempt.json", attempt)
    if status is None:
        return directory
    result = {
        "attempt_id": aid,
        "task_id": tid,
        "campaign_id": cid,
        "task_fingerprint": fp,
        "status": status,
        "reason": "exit_code" if status == "failed" else "completed",
        "finished_at": dt.datetime.fromtimestamp(finished, dt.UTC).isoformat(),
        "validation": {"valid": valid},
        "execution": {
            "telemetry": {
                "wall_seconds": duration,
                "sampled_peak_rss_bytes": peak * 1048576 if peak is not None else None,
            }
        },
        "assigned_gpu": {"devices": [{"model": "Test GPU", "memory_mib": 99999}]},
    }
    atomic_json(directory / "result.json", result)
    if status == "succeeded" and valid and complete:
        atomic_json(
            store.campaign_dir(cid) / "completions" / f"{tid}.json",
            {"task_fingerprint": fp, "validation": {"valid": True}, "attempt_id": aid},
        )
    return directory


def tree(root):
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_empty_dashboard_is_read_only_and_unknown_measurements_stay_null(tmp_path):
    root = tmp_path / "never-created"
    snapshot = DashboardSnapshot(Config(state_dir=str(root))).get()
    assert not root.exists()
    assert snapshot["state_exists"] is False
    assert snapshot["totals"]["tasks"] == 0
    assert snapshot["timing"]["samples"] == 0
    assert snapshot["timing"]["mean_seconds"] is None
    assert snapshot["memory"]["max_peak_mib"] is None
    assert snapshot["health"]["controller"]["status"] == "not_recorded"
    json.dumps(snapshot, allow_nan=False)


def test_pending_reserve_report_uses_cached_controller_evidence_only(tmp_path):
    root = tmp_path / "state"
    report = {"gpu": {"target": 20, "pending": 12, "planned": 4, "deficit": 8}}
    atomic_json(root / "controller_status.json", {
        "state": "healthy", "at": "1970-01-01T00:00:01+00:00",
        "plan": {"pending_reserves": report, "reasons": ["gpu: MAX_SUBMITTED"]},
    })
    before = tree(root)
    snapshot = DashboardSnapshot(Config(state_dir=str(root)), clock=lambda: 200).get()
    assert tree(root) == before
    assert snapshot["pending_reserves"]["budgets"] == report
    assert snapshot["pending_reserves"]["reasons"] == ["gpu: MAX_SUBMITTED"]
    assert snapshot["health"]["controller"]["stale"] is True
    assert "not confirmed pending" in snapshot["pending_reserves"]["note"]


def test_measured_success_runtime_failures_ram_and_unfinished_age(campaign):
    config, store, record = campaign
    evidence(store, record, 0, duration=10, peak=1024)
    evidence(store, record, 1, status="failed", duration=1000, peak=4096)
    evidence(store, record, 2, duration=30, peak=2048, finished=110)
    evidence(store, record, 3, status=None)
    atomic_json(
        store.root / "controller_status.json",
        {"state": "healthy", "at": "1970-01-01T00:00:01+00:00", "epoch": 3},
    )
    atomic_json(store.root / "service/leader.json", {"state": "ACTIVE", "heartbeat_at": 199, "epoch": 3})
    before = tree(store.root)
    result = DashboardSnapshot(config, clock=lambda: 200).get()
    assert tree(store.root) == before
    assert result["totals"]["counts"] == {"succeeded": 2, "failed": 1, "unknown": 1}
    assert result["totals"]["completion_fraction"] == 0.75
    assert result["timing"]["samples"] == 2
    assert result["timing"]["mean_seconds"] == result["timing"]["median_seconds"] == 20
    assert result["timing"]["p90_seconds"] == 30
    assert result["memory"]["samples"] == 3
    assert result["memory"]["max_peak_mib"] == 4096
    assert result["memory"]["mean_requested_mib"] == 8192
    assert result["memory"]["vram_usage_measured"] is False
    assert result["groups"][0]["gpu_model"] == "Test GPU"
    assert result["groups"][0]["timing"]["samples"] == 2
    assert result["active_tasks"][0]["elapsed_seconds"] == 150
    assert result["active_tasks"][0]["state"] == "unknown"
    assert result["active_tasks"][0]["liveness"] == "not_probed"
    assert result["health"]["controller"]["stale"] is True
    assert result["health"]["service"]["stale"] is False
    assert len(result["recent_runs"]) == 2
    assert result["retries"]["failed_attempts"] == 1


def test_cache_is_shared_per_instance_forceable_and_defensive(campaign, monkeypatch):
    config, store, record = campaign
    clock = [200]
    data = DashboardSnapshot(config, clock=lambda: clock[0])
    calls = []
    original = data._scan

    def scan(now):
        calls.append(now)
        return original(now)

    monkeypatch.setattr(data, "_scan", scan)
    first = data.get()
    first["totals"]["tasks"] = -100
    evidence(store, record, 0)
    clock[0] += 3
    cached = data.get()
    assert cached["totals"]["tasks"] == 4 and cached["totals"]["succeeded"] == 0
    assert cached["cache"]["age_seconds"] == 3
    assert len(calls) == 1
    assert data.get(force=True)["totals"]["succeeded"] == 1
    assert len(calls) == 2
    clock[0] += 11
    assert data.get()["cache"]["refreshed"] is True
    assert len(calls) == 3


def test_corrupt_records_and_unvalidated_success_do_not_become_success(campaign):
    config, store, record = campaign
    evidence(store, record, 0, valid=False, duration=99999)
    path = evidence(store, record, 1)
    (path / "result.json").write_text("{partial")
    completion = store.campaign_dir(record["campaign_id"]) / "completions" / f"{record['task_ids'][1]}.json"
    completion.unlink()
    task = store.campaign_dir(record["campaign_id"]) / "tasks" / f"{record['task_ids'][2]}.json"
    task.write_text("[]")
    result = DashboardSnapshot(config, clock=lambda: 200).get()
    assert result["timing"]["samples"] == 0
    assert result["totals"]["succeeded"] == 0
    assert result["diagnostics"]["partial"]
    assert result["diagnostics"]["partial_records"] >= 2
    assert result["totals"]["counts"]["unknown"] >= 3


def test_truncation_keeps_declared_denominator_and_labels_unknown(campaign):
    config, store, record = campaign
    for index in range(4):
        evidence(store, record, index, duration=index + 1, finished=100 + index)
    result = DashboardSnapshot(config, max_tasks=2, max_recent=1).get()
    assert result["totals"]["tasks"] == 4
    assert result["totals"]["observed_tasks"] == 2
    assert result["totals"]["unscanned_tasks"] == 2
    assert result["totals"]["counts"] == {"succeeded": 2, "unknown": 2}
    assert result["diagnostics"]["truncation"]["tasks"] is True
    assert result["diagnostics"]["truncation"]["recent_runs"] == {"available": 2, "shown": 1}
    assert len(result["recent_runs"]) == 1


def test_retry_after_attempt_and_explicit_cancel_precedence(campaign):
    config, store, record = campaign
    evidence(store, record, 0, status="failed")
    evidence(store, record, 0, number=2, status=None)
    evidence(store, record, 1, status=None)
    cid, tid0, tid1 = record["campaign_id"], record["task_ids"][0], record["task_ids"][1]
    store.update_task_state(cid, tid0, {"state": "retry_wait", "after_attempt_id": "000002_example"})
    store.update_task_state(cid, tid1, {"state": "canceled", "reason": "user_cancelled"})
    before = tree(store.root)
    result = DashboardSnapshot(config).get()
    assert result["retries"]["tasks_retried"] == 1
    assert result["retries"]["additional_attempts"] == 1
    assert result["totals"]["counts"]["retry_wait"] == 1
    assert result["totals"]["counts"]["canceled"] == 1
    assert read_task_state(store.root, cid, tid0)["state"] == "retry_wait"
    assert read_task_state(store.root, cid, tid1)["state"] == "canceled"
    assert tree(store.root) == before
    assert not list((store.campaign_dir(cid) / "task_locks").glob("*.lock"))


def test_allocation_intents_are_not_reported_as_live_gpu_counts(campaign):
    config, store, _ = campaign
    store.put_intent({"token": "worker1", "backend": "worker", "state": "submitted", "profile": "gpu-large"})
    store.put_intent({"token": "direct1", "backend": "direct", "state": "running", "profile": "cpu-direct"})
    store.put_intent({"token": "hq1", "backend": "hq", "state": "submitted"})
    result = DashboardSnapshot(config).get()
    assert result["allocations"]["counts"] == {"submitted": 1, "running": 1}
    assert result["allocations"]["hq_admission_counts"] == {"submitted": 1}
    assert "not a live Slurm" in result["allocations"]["note"]


def test_detail_state_rejects_wrong_success_identity(campaign):
    _, store, record = campaign
    evidence(store, record, 0)
    tid, cid = record["task_ids"][0], record["campaign_id"]
    path = store.campaign_dir(cid) / "completions" / f"{tid}.json"
    atomic_json(path, {"task_fingerprint": "wrong", "validation": {"valid": True}})
    with pytest.raises(ValueError, match="completion"):
        read_task_state(store.root, cid, tid)


def test_malformed_nested_telemetry_is_diagnostic_not_a_failed_dashboard(campaign):
    config, store, record = campaign
    path = evidence(store, record, 0)
    result = json.loads((path / "result.json").read_text())
    result["execution"] = ["malformed"]
    result["assigned_gpu"] = {"devices": ["not-a-device"]}
    atomic_json(path / "result.json", result)
    snapshot = DashboardSnapshot(config).get()
    assert snapshot["totals"]["succeeded"] == 1
    assert snapshot["timing"]["mean_seconds"] is None
    assert snapshot["memory"]["mean_peak_mib"] is None
    assert snapshot["diagnostics"]["partial_records"] == 2
    assert snapshot["groups"][0]["gpu_model"] == "unknown"


def test_slow_scan_extends_cache_interval_instead_of_rescan_storm(campaign, monkeypatch):
    config, _, _ = campaign
    clock = [200]
    snapshot = DashboardSnapshot(config, clock=lambda: clock[0])
    original = snapshot._scan
    calls = []

    def slow_scan(now):
        calls.append(now)
        result = original(now)
        clock[0] += 8
        return result

    monkeypatch.setattr(snapshot, "_scan", slow_scan)
    first = snapshot.get()
    assert first["cache"]["scan_seconds"] == 8
    assert first["cache"]["ttl_seconds"] == 40
    clock[0] += 11
    assert snapshot.get()["cache"]["refreshed"] is False
    assert len(calls) == 1
    clock[0] += 30
    assert snapshot.get()["cache"]["refreshed"] is True
    assert len(calls) == 2


def test_unfinished_attempt_is_a_neutral_bucket_not_running_or_attention(campaign):
    config, store, record = campaign
    evidence(store, record, 0, status=None)
    result = DashboardSnapshot(config, clock=lambda: 200).get()
    assert result["totals"]["counts"]["unknown"] == 1  # backend state is unchanged
    assert result["totals"]["unfinished"] == 1
    assert result["totals"]["unknown_other"] == 0
    assert result["campaigns"][0]["unfinished"] == 1
    assert result["active_tasks"][0]["display_state"] == "unfinished"
    assert result["active_tasks"][0]["state"] == "unknown"


def test_multiple_verified_gpu_models_and_effective_ram_override(campaign):
    config, store, record = campaign
    path = evidence(store, record, 0)
    result = json.loads((path / "result.json").read_text())
    result["assigned_gpu"] = {"devices": [{"model": "A4000"}, {"model": "L40"}, {"model": "A4000"}]}
    atomic_json(path / "result.json", result)
    attempt = json.loads((path / "attempt.json").read_text())
    attempt["resources"] = {"memory_mib": 16384}
    atomic_json(path / "attempt.json", attempt)
    snapshot = DashboardSnapshot(config).get()
    assert snapshot["groups"][0]["gpu_model"] == "2× A4000, L40"
    row = snapshot["recent_runs"][0]
    assert (row["requested_mib"], row["declared_mib"], row["ram_override"]) == (16384, 8192, True)
    assert gpu_label([{"model": "A4000"}, {}], 2) == "unknown"
    assert gpu_label([], 0) == "cpu"


def test_failed_scan_keeps_cached_data_visible_and_backs_off(campaign, monkeypatch):
    config, _, _ = campaign
    clock = [200.0]
    snapshot = DashboardSnapshot(config, clock=lambda: clock[0])
    original, calls, broken = snapshot._scan, [], [True]

    def scan(now):
        calls.append(now)
        if broken[0]:
            raise OSError("shared storage unavailable")
        return original(now)

    monkeypatch.setattr(snapshot, "_scan", scan)
    with pytest.raises(DashboardUnavailable):
        snapshot.get()
    clock[0] += 1
    with pytest.raises(DashboardUnavailable):
        snapshot.get()
    assert len(calls) == 1  # backoff: no scan storm
    clock[0] += 5
    broken[0] = False
    assert snapshot.get()["cache"]["error"] is None
    broken[0] = True
    clock[0] += 11
    stale = snapshot.get()
    assert stale["totals"]["tasks"] == 4
    assert stale["cache"]["error"]["kind"] == "OSError"
    assert stale["cache"]["refreshed"] is False
    count = len(calls)
    clock[0] += 2
    assert snapshot.get()["cache"]["error"]["kind"] == "OSError"
    assert len(calls) == count


def test_later_unreadable_attempt_does_not_report_older_failure_as_final(campaign):
    config, store, record = campaign
    evidence(store, record, 0, status="failed")
    later = evidence(store, record, 0, number=2, status=None)
    (later / "attempt.json").write_text("{broken")
    result = DashboardSnapshot(config).get()
    assert result["totals"]["failed"] == 0
    assert result["totals"]["counts"]["unknown"] == 1
    assert result["diagnostics"]["partial_records"] == 1
