"""Explicit RFD4 launch boundary; all scheduling and retries belong to HHQ."""

from __future__ import annotations

import hashlib
from collections import Counter
from copy import deepcopy
from pathlib import Path

from ... import api
from ...ingest import import_campaign_id
from ...models import Config, OutputPolicy, Resources, RetryPolicy
from ...planner import compatible, gpu_count
from ...state import StateStore, atomic_json, file_lock, fingerprint, identifier, read_json, utc_now

DIGS_RFD4_GPU_CONSTRAINT = "Ampere|Lovelace|Hopper|Blackwell"


def _rfd4_constraint(existing, requested, cluster):
    # Feature labels belong to the deployment; generic HHQ is unrestricted.
    guard = DIGS_RFD4_GPU_CONSTRAINT if requested == "auto" and cluster == "digs" else requested
    if guard in (None, "auto"):
        return existing
    if not isinstance(guard, str) or not guard.strip() or any(c in guard for c in "\n\r\x00"):
        raise ValueError("compatibility_constraint must be 'auto', None, or a nonempty Slurm constraint")
    guard = guard.strip()
    if not existing or existing == guard:
        return guard
    return f"({existing})&({guard})"


def _verified(prepared, *, check_sources=True):
    from .prepare import verify_prepared

    return verify_prepared(prepared, check_sources=check_sources)


def _key(document, group):
    return f"rfd4:{document['plan_id']}:{group['method']}"


def _receipt_path(document, config):
    return (
        Path(config.state_dir).expanduser()
        / "integrations/rfd4"
        / identifier(document["plan_id"])
        / "launch.json"
    )


def _config_fingerprints(config: Config) -> set[str]:
    """Accept omitted disabled policy additions, preserving exact prior receipt hashes."""
    resolved = config.model_dump()
    variants = [resolved]
    if all(budget["min_pending"] == 0 for budget in resolved["budgets"].values()):
        legacy = deepcopy(resolved)
        for budget in legacy["budgets"].values():
            del budget["min_pending"]
        variants.append(legacy)
    if all(budget["fill_running_target"] is False for budget in resolved["budgets"].values()):
        for value in list(variants):
            legacy = deepcopy(value)
            for budget in legacy["budgets"].values():
                del budget["fill_running_target"]
            variants.append(legacy)
    for name, default in (("budget_priority", []), ("speculative_partition_workers", False)):
        if resolved["policy"].get(name) == default:
            for value in list(variants):
                legacy = deepcopy(value)
                del legacy["policy"][name]
                variants.append(legacy)
    for name, default in (("stop_when_complete", False), ("auto_renew", False), ("renew_lead_seconds", 600)):
        if resolved["control"].get(name) == default:
            for value in list(variants):
                legacy = deepcopy(value)
                del legacy["control"][name]
                variants.append(legacy)
    # These additions shipped together. Keep historical schemas without an
    # exponential expansion of hypothetical partially-written configurations.
    added_policy = {"pending_reserves_after_growth": False, "cancel_pending_when_assigned": False,
                    "allocation_gpu_hours_budget": None, "allocation_budget_allowance": 0.5}
    if (all(resolved["policy"].get(k) == v for k, v in added_policy.items())
            and all(b.get("max_pending") is None and b.get("max_running") is None
                    for b in resolved["budgets"].values())
            and all(not p.get("fallback_profiles") for p in resolved["profiles"].values())):
        for value in list(variants):
            legacy = deepcopy(value)
            for key in added_policy:
                del legacy["policy"][key]
            for budget in legacy["budgets"].values():
                del budget["max_pending"]
                del budget["max_running"]
            for profile in legacy["profiles"].values():
                del profile["fallback_profiles"]
            variants.append(legacy)
    return {fingerprint(value) for value in variants}


def preview(prepared, config: Config) -> dict:
    """Verify immutable files and plan every group. Never import or start work."""
    return _preview_verified(_verified(prepared), config)


def _preview_verified(document: dict, config: Config) -> dict:
    """Plan groups from the caller's already verified immutable package."""
    groups = document["groups"]
    if not groups or len({g["method"] for g in groups}) != len(groups):
        raise ValueError("Prepared methods must be nonempty and unique")
    plans = []
    for group in groups:
        RetryPolicy.model_validate(group.get("retry", {"max_attempts": group["max_attempts"]}))
        resources = Resources.model_validate(group["resources"])
        if resources.gpus != 1:
            raise ValueError("RFD4 pooled bundles require exactly one GPU")
        plan = api.plan(
            group["commands_file"],
            config,
            profile=group["profile"],
            mode="pooled",
            resources=resources,
            format="jsonl",
        )
        if plan.task_count != len(group["bundle_ids"]):
            raise ValueError("Prepared commands/bundle count mismatch")
        plans.append({"method": group["method"], "import_key": _key(document, group),
                      "priority": group.get("priority", 0), **plan.model_dump()})
    return {
        "schema": "hhq-rfd4-preview/1",
        "plan_id": document["plan_id"],
        "groups": plans,
        "submitted": False,
        "note": "Preview only. Launch imports runnable work into the shared backlog.",
    }


def launch(prepared, config: Config, *, start_service=True) -> dict:
    """Import all groups with stable keys, then start one service when requested.

    Imports are not a distributed transaction: an already running service can
    execute an earlier group before a later import fails. The receipt preserves
    partial progress; rerunning uses the same immutable import identities.
    """
    document = _verified(prepared)
    _preview_verified(document, config)  # All groups must fit before any import occurs.
    api.assert_service_config_compatible(config)
    path = _receipt_path(document, config)
    with file_lock(path.with_suffix(".lock")):
        binding = {
            "plan_id": document["plan_id"],
            "manifest_sha256": document["manifest_sha256"],
            "config_sha256": fingerprint(config.model_dump()),
        }
        receipt = (
            read_json(path)
            if path.exists()
            else {
                "schema": "hhq-rfd4-launch/1",
                **binding,
                "created_at": utc_now(),
                "groups": {},
                "service_status": "not_requested",
            }
        )
        if (
            any(receipt.get(key) != binding[key] for key in ("plan_id", "manifest_sha256"))
            or receipt.get("config_sha256") not in _config_fingerprints(config)
        ):
            raise ValueError("Prepared launch/configuration changed; create a new explicit plan")
        if receipt["service_status"] in {"starting", "unknown"} and start_service:
            raise ValueError(f"Uncertain prior service start; reconcile it before another start: {path}")
        receipt["state"] = "importing"
        atomic_json(path, receipt)
        for group in document["groups"]:
            method, key = group["method"], _key(document, group)
            entry = receipt["groups"].setdefault(
                method, {"import_key": key, "campaign_id": import_campaign_id(key)}
            )
            entry["state"] = "importing"
            atomic_json(path, receipt)
            try:
                result = api.submit(
                    group["commands_file"],
                    config,
                    profile=group["profile"],
                    mode="pooled",
                    resources=Resources.model_validate(group["resources"]),
                    format="jsonl",
                    cwd=str(Path(document["manifest_path"]).parent),
                    name=f"rfd4-{document['plan_id']}-{method}",
                    retry=RetryPolicy.model_validate(group.get("retry", {"max_attempts": group["max_attempts"]})),
                    output=OutputPolicy(
                        mode="isolated",
                        repeatable=True,
                        expected_paths=["{attempt_dir}/output/rfd4_completion.json"],
                    ),
                    import_key=key,
                    start_service=False,
                    environment=group.get("environment", {}),
                    priority=group.get("priority", 0),
                    workload=f"rfd4-{method}",
                    input_fingerprint=document["manifest_sha256"],
                    software_fingerprint=document.get("software_fingerprint")
                    or fingerprint(document["file_sha256"]),
                )
                entry.update(state="imported", submission=result.model_dump())
            except Exception as error:
                entry.update(state="uncertain_import", error=f"{type(error).__name__}: {error}")
                receipt["state"] = "partial"
                atomic_json(path, receipt)
                raise
            atomic_json(path, receipt)
        receipt["state"] = "imported"
        atomic_json(path, receipt)
        if start_service and receipt["service_status"] != "started":
            from ...service.supervisor import start_service as start

            receipt["service_status"] = "starting"
            atomic_json(path, receipt)
            try:
                receipt["service"] = start(
                    config, hosting=config.control.hosting, standby=config.control.standby_replicas
                )
                receipt["service_status"] = "started"
            except Exception as error:
                receipt.update(service_status="unknown", service_error=f"{type(error).__name__}: {error}")
                atomic_json(path, receipt)
                raise
            atomic_json(path, receipt)
        return {
            **receipt,
            "receipt_path": str(path),
            "note": "Imports are runnable immediately on an active service; this is not dormant preparation.",
        }


def _cancellation_recovery_pending(store, task, attempts):
    """Block collection while an exact service fence may still authorize replay."""
    from ...runner import attempt_epoch, internal_fencing_evidence

    if not attempts or len(attempts) >= task.retry.max_attempts:
        return False
    previous = attempts[-1]
    epoch = attempt_epoch(previous)
    if epoch is None:
        return False
    raw = read_json(store.campaign_dir(task.campaign_id) / "task_states" / f"{task.task_id}.json")
    if raw.get("state", raw.get("status")) == "canceled":
        return False  # Explicit coordinator/user cancellation outranks recovery.
    proof = store.root / "service" / f"recovery-{epoch}.json"
    if proof.exists():
        return bool(internal_fencing_evidence(store, task, previous, proof))
    intent = store.root / "service" / f"recovery-intent-{epoch}.json"
    if not intent.exists():
        return False
    record = read_json(intent)
    job = previous.get("context", {}).get("environment", {}).get("SLURM_JOB_ID")
    return (
        record.get("schema") == "hhq-service-fencing-intent/1"
        and record.get("reason") == "generation_recovery"
        and str(record.get("previous_epoch")) == str(epoch)
        and job is not None
        and any(str(a.get("job_id")) == str(job) for a in record.get("allocations", []))
    )


def status(prepared, config: Config, *, reconcile=False, slurm=None) -> dict:
    """Read logical tasks; scheduler reconciliation is explicit and optional."""
    document = _verified(prepared, check_sources=False)
    if reconcile:
        api.status(config, reconcile=True, slurm=slurm)
    root = Path(config.state_dir).expanduser()
    store = StateStore(root) if root.exists() else None
    groups, blockers, total = [], [], Counter()
    for group in document["groups"]:
        cid = import_campaign_id(_key(document, group))
        counts = Counter()
        if store is None or not (store.campaign_dir(cid) / "campaign.json").exists():
            counts["not_imported"] = len(group["bundle_ids"])
            blockers.append({"method": group["method"], "reason": "not_imported"})
        else:
            campaign = store.campaign(cid)
            if len(campaign["task_ids"]) != len(group["bundle_ids"]):
                raise ValueError("Imported campaign/prepared bundle count mismatch")
            for tid in campaign["task_ids"]:
                task = store.task(cid, tid)
                if task.input_fingerprint != document["manifest_sha256"]:
                    raise ValueError("Imported task/prepared manifest identity mismatch")
                current = store.task_state(cid, tid)
                value = current.get("state", "unknown")
                counts[value] += 1
                reason = current.get("reason")
                attempts = store.attempts(cid, tid)
                retry_pending = (
                    value in {"failed", "interrupted"}
                    and reason in task.retry.reasons
                    and len(attempts) < task.retry.max_attempts
                )
                if value == "canceled":
                    retry_pending = _cancellation_recovery_pending(store, task, attempts)
                missing_completion = value == "succeeded" and "completion" not in current
                if value not in {"succeeded", "failed", "canceled"} or retry_pending or missing_completion:
                    blockers.append(
                        {
                            "method": group["method"],
                            "task_id": tid,
                            "state": value,
                            "reason": "retry_decision_pending"
                            if retry_pending
                            else "missing_validated_completion"
                            if missing_completion
                            else reason or value,
                        }
                    )
        total.update(counts)
        groups.append({"method": group["method"], "campaign_id": cid, "counts": dict(counts)})
    return {
        "schema": "hhq-rfd4-status/1",
        "plan_id": document["plan_id"],
        "observed_at": utc_now(),
        "groups": groups,
        "counts": dict(total),
        "blockers": blockers,
        "all_terminal": not blockers,
        "note": "Logical task snapshot; worker allocations may still be live.",
    }


def collection_ready(prepared, config: Config, *, reconcile=False, slurm=None) -> dict:
    """Point-in-time terminal barrier; a later manual retry is a separate action."""
    result = status(prepared, config, reconcile=reconcile, slurm=slurm)
    return {
        **result,
        "ready": result["all_terminal"],
        "successes": result["counts"].get("succeeded", 0),
        "failures": result["counts"].get("failed", 0),
        "canceled": result["counts"].get("canceled", 0),
    }


def submit_aggregation(
    prepared,
    config: Config,
    commands_file,
    *,
    profile: str,
    resources: Resources,
    output: OutputPolicy | None = None,
    format="lines",
    slurm=None,
) -> dict:
    """Explicit CPU postprocessing; no automatic DAG or scientific reconstruction."""
    readiness = collection_ready(prepared, config)
    if not readiness["ready"]:
        raise ValueError(f"Collection is not ready: {readiness['blockers']}")
    if resources.gpus:
        raise ValueError("Aggregation requires a CPU resource request")
    digest = hashlib.sha256(Path(commands_file).read_bytes()).hexdigest()
    key = fingerprint(
        {
            "plan_id": readiness["plan_id"],
            "sha256": digest,
            "profile": profile,
            "resources": resources.model_dump(),
            "format": format,
            "output": output.model_dump() if output else None,
        }
    )
    result = api.submit(
        commands_file,
        config,
        profile=profile,
        mode="direct",
        resources=resources,
        format=format,
        output=output,
        import_key=f"rfd4-collection:{key}",
        start_service=False,
        slurm=slurm,
        workload="rfd4-collection",
    )
    return {"readiness": readiness, "submission": result.model_dump()}


def method_profiles(
    base_config: Config, requests_by_method: dict, eligible_profile_names: list[str],
    *, eligible_profiles_by_method: dict | None = None, compatibility_constraint: str | None = "auto",
) -> Config:
    """Clone explicit GPU placements with task RAM plus the fixed worker reserve.

    Supply explicit task RAM with the desired fixed application headroom already
    added (for example peak + 2048 MiB). No percentages or automatic escalation.
    The returned config exposes workload aliases ``rfd4-<method>``. Build it before
    starting a service; changing a running service's frozen config is rejected.
    Declared estimates are operator assumptions, not measurements.
    DIGS RFD4 profiles default to modern GPU features, AND-ed with custom Slurm
    constraints. Explicit None disables this application guard; other clusters
    retain their placements unless a site-specific guard is supplied.
    """
    config = base_config.model_copy(deep=True)
    if not requests_by_method or not eligible_profile_names:
        raise ValueError("Explicit methods and eligible profiles are required")
    selected_by_method = eligible_profiles_by_method or {}
    if set(selected_by_method) - set(requests_by_method):
        raise ValueError("Per-method profile selection names an unknown method")
    for method, raw in requests_by_method.items():
        identifier(method)
        request = raw if isinstance(raw, Resources) else Resources.model_validate(raw)
        if (
            not {"memory_mib", "runtime_seconds"}.issubset(request.model_fields_set)
            or request.runtime_confidence not in {"declared", "measured"}
            or request.runtime_seconds is None
        ):
            raise ValueError("Provide explicit RAM and declared or measured runtime for each method")
        if request.gpus != 1 or request.nodes != 1 or request.ranks != 1:
            raise ValueError("RFD4 profiles require one GPU, one node and one rank")
        names = []
        selected = selected_by_method.get(method, eligible_profile_names)
        if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
                or set(selected) - set(eligible_profile_names)):
            raise ValueError("Per-method profiles must be a nonempty unique subset of eligible profiles")
        for original in selected:
            base = base_config.profiles[original]
            if gpu_count(base) != 1 or not base.budget or base.budget not in config.budgets:
                raise ValueError("Eligible profiles require one GPU and an explicit shared budget")
            name = identifier(f"rfd4-{method}-{original}")
            if name in config.profiles or f"rfd4-{method}" in base_config.workloads:
                raise ValueError(f"Resource profile/alias already exists: {name}")
            clone = base.model_copy(
                update={
                    "memory_mib": request.memory_mib + config.policy.worker_overhead_mib,
                    "cpus": max(base.cpus, request.cpus),
                    "execution_mode": "pooled",
                    "constraint": _rfd4_constraint(base.constraint, compatibility_constraint, config.cluster),
                }
            )
            valid, reason = compatible(request, clone, config)
            if not valid:
                raise ValueError(f"{method}/{original}: {reason}")
            config.profiles[name] = clone
            names.append(name)
        config.workloads[f"rfd4-{method}"] = names
    return config
