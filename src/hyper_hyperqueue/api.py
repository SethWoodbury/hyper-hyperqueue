"""Notebook-friendly public API. Importing this module never imports HyperQueue."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .config import profile_names
from .ingest import ingest_file, parse_file
from .models import Config, Model, OutputPolicy, Plan, Resources, RetryPolicy
from .planner import make_plan
from .state import StateStore, atomic_json


class Submission(Model):
    schema_version: int = 1
    campaign_id: str
    state_path: str
    mode: str
    task_count: int
    backend_result: dict


def assert_service_config_compatible(config: Config) -> None:
    """An existing service owns its frozen placement policy until explicitly restarted."""
    path = Path(config.state_dir).expanduser() / "service" / "deployment.json"
    if not path.exists():
        return
    deployment = json.loads(path.read_text())
    saved = Config.model_validate(json.loads(Path(deployment["config"]).read_text())).model_dump()
    requested = config.model_dump()
    old_state, new_state = Path(saved["state_dir"]).expanduser(), Path(requested["state_dir"]).expanduser()
    if old_state.resolve() == new_state.resolve() or (
        old_state.exists() and new_state.exists() and old_state.samefile(new_state)
    ):
        requested["state_dir"] = saved["state_dir"]
    if saved != requested:
        changed = sorted(key for key in saved if saved[key] != requested.get(key))
        raise ValueError(
            "Existing service uses a frozen configuration; drain/stop and explicitly restart with the new configuration before importing pooled work. Changed fields: "
            + ", ".join(changed)
        )


def reopen_completed_service(config, *, slurm=None):
    """Allow a new import only after automatic shutdown is positively finished.

    No service is started before import. This is scoped to our automatic closure;
    explicit operator stop/drain semantics are unchanged.
    """
    from .service.leadership import read_record
    from .service.supervisor import _terminal_generation_proof

    store = StateStore(config.state_dir)
    directory = store.root / "service"
    with store.lock():
        completion = read_record(directory / "completion.json", {})
        deployment = read_record(directory / "deployment.json", {})
        if (not completion or completion.get("deployment_id") != deployment.get("deployment_id")
                or deployment.get("desired_state") != "STOPPED"):
            return False
        previous = read_record(directory / "leader.json", {})
        if not previous or previous.get("epoch") != completion.get("epoch"):
            raise ValueError("Auto-completed service ownership is unresolved; new import blocked")
        try:
            _terminal_generation_proof(config, previous, client=slurm)
        except (OSError, RuntimeError) as exc:
            raise ValueError(
                "Automatic completion cleanup is not yet positively terminal; no new work was imported. "
                "Wait for owned allocations/accounting, then repeat submit or append."
            ) from exc
        completion.update(phase="stopped")
        atomic_json(directory / "completion.json", completion)
        return True


def default_resources(config, names):
    gpu = any(config.profiles[n].gres for n in names)
    return Resources(gpus=int(gpu), memory_mib=12288 if gpu else 1024)


def plan(
    path: str | Path,
    config: Config,
    *,
    profile: str,
    mode="auto",
    resources: Resources | None = None,
    literal=False,
    format="lines",
    max_parallel=None,
    commands_per_job=1,
) -> Plan:
    records = parse_file(path, literal=literal, format=format)
    if not records:
        raise ValueError("No commands found")
    names = profile_names(config, profile)
    preferences = {config.profiles[n].execution_mode for n in names}
    chosen_mode = (
        next(iter(preferences))
        if mode == "auto" and len(preferences) == 1 and "auto" not in preferences
        else mode
    )
    decision = make_plan(
        len(records),
        resources or default_resources(config, names),
        config,
        names,
        mode=chosen_mode,
        max_parallel=max_parallel,
        commands_per_job=commands_per_job,
    )
    if mode == "auto" and chosen_mode != "auto":
        decision.reasons = ["PROFILE_MODE_PREFERENCE"]
    if config.discovery_snapshot:
        from .discovery import load_snapshot, validate_config_against_snapshot

        observed = validate_config_against_snapshot(config, load_snapshot(config.discovery_snapshot))
        if observed["errors"]:
            raise ValueError("; ".join(observed["errors"]))
        decision.warnings.extend(observed["warnings"])
    return decision


def submit(
    path: str | Path,
    config: Config,
    *,
    profile: str,
    mode="auto",
    resources: Resources | None = None,
    cwd=None,
    name=None,
    literal=False,
    format="lines",
    max_parallel=None,
    commands_per_job=1,
    retry: RetryPolicy | None = None,
    output: OutputPolicy | None = None,
    environment=None,
    import_key=None,
    start_service=True,
    slurm=None,
    workload="generic",
    priority=0,
    input_fingerprint=None,
    software_fingerprint=None,
    shell="/bin/bash",
    shell_options=None,
) -> Submission:
    resources = resources or default_resources(config, profile_names(config, profile))
    decision = plan(
        path,
        config,
        profile=profile,
        mode=mode,
        resources=resources,
        literal=literal,
        format=format,
        max_parallel=max_parallel,
        commands_per_job=commands_per_job,
    )
    if decision.mode == "pooled":
        assert_service_config_compatible(config)
        reopen_completed_service(config, slurm=slurm)
    store = StateStore(config.state_dir)
    metadata = ingest_file(
        store,
        path,
        mode=decision.mode,
        resources=resources,
        profile_names=decision.profiles,
        cwd=cwd,
        name=name,
        literal=literal,
        format=format,
        retry=retry,
        output=output,
        environment=environment,
        import_key=import_key,
        workload=workload,
        priority=priority,
        input_fingerprint=input_fingerprint,
        software_fingerprint=software_fingerprint,
        shell=shell,
        shell_options=shell_options,
    )
    with store.lock():
        metadata.update(
            max_parallel=max_parallel,
            commands_per_job=commands_per_job,
            resolved_config=config.model_dump(),
            plan=decision.model_dump(),
        )
        store.save_campaign(metadata)
        atomic_json(store.root / "resolved_config.json", config.model_dump())
        result = {"state": "ready", "service_required": decision.service_required}
        if decision.mode == "direct":
            from .backends.direct import submit_ready
            from .backends.slurm import SlurmClient

            result = submit_ready(
                store,
                config,
                metadata["campaign_id"],
                slurm or SlurmClient(),
                max_parallel=max_parallel,
                commands_per_job=commands_per_job,
            )
    if decision.mode == "pooled" and start_service:
        from .service.supervisor import start_service as start

        result = start(config, hosting=config.control.hosting, standby=config.control.standby_replicas)
    return Submission(
        campaign_id=metadata["campaign_id"],
        state_path=str(store.campaign_dir(metadata["campaign_id"])),
        mode=decision.mode,
        task_count=len(metadata["task_ids"]),
        backend_result=result,
    )


def append(campaign_id: str, path: str | Path, config: Config, *, import_key=None, slurm=None) -> Submission:
    store = StateStore(config.state_dir)
    campaign = store.campaign(campaign_id)
    if campaign.get("pause_import"):
        raise ValueError("Campaign imports are paused")
    if campaign["mode"] == "pooled":
        assert_service_config_compatible(config)
        restart_completed = reopen_completed_service(config, slurm=slurm)
    exemplar = store.task(campaign_id, campaign["task_ids"][0])
    import_settings = campaign["imports"][0]["settings"]
    metadata = ingest_file(
        store,
        path,
        campaign_id=campaign_id,
        mode=campaign["mode"],
        resources=exemplar.resources,
        profile_names=exemplar.allowed_profiles,
        cwd=exemplar.cwd,
        environment=exemplar.environment,
        retry=exemplar.retry,
        output=exemplar.output,
        import_key=import_key,
        literal=import_settings["literal"],
        format=import_settings["format"],
        shell=exemplar.shell,
        shell_options=exemplar.shell_options,
        workload=exemplar.workload,
        priority=exemplar.priority,
        input_fingerprint=exemplar.input_fingerprint,
        software_fingerprint=exemplar.software_fingerprint,
    )
    result = {"state": "ready", "service_required": campaign["mode"] == "pooled"}
    if campaign["mode"] == "direct":
        from .backends.direct import submit_ready
        from .backends.slurm import SlurmClient

        with store.lock():
            result = submit_ready(
                store,
                config,
                campaign_id,
                slurm or SlurmClient(),
                max_parallel=campaign.get("max_parallel"),
                commands_per_job=campaign.get("commands_per_job", 1),
            )
    elif restart_completed:
        from .service.supervisor import start_service

        result = start_service(config, hosting=config.control.hosting,
                               standby=config.control.standby_replicas, slurm=slurm)
    return Submission(
        campaign_id=campaign_id,
        state_path=str(store.campaign_dir(campaign_id)),
        mode=campaign["mode"],
        task_count=len(metadata["task_ids"]),
        backend_result=result,
    )


def status(config: Config, *, campaign_id=None, reconcile=False, slurm=None) -> dict:
    store = StateStore(config.state_dir)
    if reconcile:
        from .backends.slurm import SlurmClient
        from .controller import reconcile_slurm

        slurm = slurm or SlurmClient()
        with store.lock():
            reconcile_slurm(store, slurm, slurm.query_jobs())
    rows = []
    for c in store.campaigns():
        if campaign_id and c["campaign_id"] != campaign_id:
            continue
        counts = Counter(store.task_state(c["campaign_id"], t).get("state", "ready") for t in c["task_ids"])
        rows.append(
            {
                "campaign_id": c["campaign_id"],
                "name": c["name"],
                "mode": c["mode"],
                "tasks": len(c["task_ids"]),
                "counts": dict(counts),
                "draining": c.get("draining", False),
            }
        )
    health_path = store.root / "controller_status.json"
    health = json.loads(health_path.read_text()) if health_path.exists() else {"state": "not_started"}
    return {
        "schema_version": 1,
        "campaigns": rows,
        "controller": health,
        "state_dir": str(store.root),
        "note": "ready is unsent; submitted is backend-admitted, not necessarily running. Attempt results determine success.",
    }


def find_task(store, task_id):
    matches = [c["campaign_id"] for c in store.campaigns() if task_id in c["task_ids"]]
    if len(matches) != 1:
        raise ValueError("Task ID must identify exactly one imported task")
    return matches[0]


def explain(task_id: str, config: Config) -> dict:
    store = StateStore(config.state_dir)
    cid = find_task(store, task_id)
    task = store.task(cid, task_id)
    current = store.task_state(cid, task_id)
    result = {
        "schema_version": 1,
        "task_id": task_id,
        "campaign_id": cid,
        "source_line": task.source_line,
        "backend": task.backend,
        "resources": task.resources.model_dump(),
        "allowed_profiles": task.allowed_profiles,
        "state": current,
        "attempts": store.attempts(cid, task_id),
        "submissions": [i for i in store.intents() if task_id in i.get("task_ids", [])],
    }
    if task.backend == "pooled" and current.get("hq_job_id") is not None:
        try:
            from .backends.hyperqueue import HQBackend

            result["live_hq"] = HQBackend(store.root / "service/hq", config.control.hq_binary).explain(
                current["hq_job_id"], current["hq_task_id"]
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            result["live_hq_unavailable"] = str(exc)
    return result


def campaign_control(config, campaign_id, action):
    store = StateStore(config.state_dir)
    options = {
        "drain": ("draining", True),
        "resume": ("draining", False),
        "pause-dispatch": ("pause_dispatch", True),
        "resume-dispatch": ("pause_dispatch", False),
        "pause-import": ("pause_import", True),
        "resume-import": ("pause_import", False),
    }
    if action not in options:
        raise ValueError("Unknown control action")
    with store.lock():
        c = store.campaign(campaign_id)
        k, v = options[action]
        c[k] = v
        store.save_campaign(c)
        store.event("campaign_control", campaign_id=campaign_id, action=action)
    return {
        "campaign_id": campaign_id,
        "action": action,
        "semantics": "Stops/resumes new hhq admission only. Already admitted HQ tasks and Slurm jobs continue; no computations are killed.",
    }
