"""User CLI; direct-only commands never require HyperQueue."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import api
from .config import load_config, profile_names
from .integrations.rfd4.cli import app as rfd4_app
from .models import OutputPolicy, Profile, Resources, RetryPolicy

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
config_app = typer.Typer()
service_app = typer.Typer()
campaign_app = typer.Typer()
pool_app = typer.Typer()
app.add_typer(config_app, name="config")
app.add_typer(service_app, name="service")
app.add_typer(campaign_app, name="campaign")
app.add_typer(pool_app, name="pool")
app.add_typer(rfd4_app, name="rfd4")
console = Console(stderr=True)
settings = {}


@app.callback()
def main(
    config: Annotated[Path | None, typer.Option("--config", envvar="HHQ_CONFIG")] = None,
    state_dir: Annotated[Path | None, typer.Option("--state-dir", envvar="HHQ_STATE_DIR")] = None,
    debug: bool = False,
):
    settings.update(config=config, state_dir=state_dir, debug=debug)


def cfg():
    value = load_config(settings.get("config"))
    if settings.get("state_dir"):
        value.state_dir = str(settings["state_dir"].expanduser().absolute())
    return value


def output(value):
    typer.echo(
        json.dumps(value.model_dump() if hasattr(value, "model_dump") else value, indent=2, default=str)
    )


def print_access(config):
    from .access import access_info, access_lines

    for line in access_lines(access_info(config, config_path=settings.get("config"))):
        typer.echo(line, err=True)


@app.command("access")
def access(json_output: bool = typer.Option(False, "--json")):
    """Show cached controller information and exact dashboard/status commands."""
    from .access import access_info, access_lines

    info = access_info(cfg(), config_path=settings.get("config"))
    if json_output:
        output(info)
    else:
        typer.echo("\n".join(access_lines(info)))


def parse_time(value):
    parts = [int(x) for x in value.split(":")]
    if len(parts) != 3 or parts[1] >= 60 or parts[2] >= 60 or any(x < 0 for x in parts):
        raise typer.BadParameter("Time must be HH:MM:SS")
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_memory(value):
    import re

    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(MiB|GiB|M|G)?", value, re.IGNORECASE)
    if not m:
        raise typer.BadParameter("Memory must be e.g. 512MiB or 12GiB")
    return int(float(m[1]) * (1024 if (m[2] or "").lower() in ("g", "gib") else 1))


def options(profile, partition, cpus, mem, gpus, time, runtime, timeout, mode, nodes=1, ranks=1):
    """Direct flags override allocation shape; pooled flags describe one task."""
    from .planner import choose_mode

    config = cfg()
    if partition and profile:
        raise typer.BadParameter("Choose --profile or --partition, not both")
    if not profile and not partition:
        raise typer.BadParameter("Choose --profile or --partition")
    if cpus is not None and cpus < 1:
        raise typer.BadParameter("--cpus must be positive")
    if partition:
        profile = "explicit"
        inferred_gpu = bool(gpus)
        names = [profile]
    else:
        names = profile_names(config, profile)
        inferred_gpu = any(config.profiles[name].gres for name in names)
    resources = Resources(
        cpus=cpus or 1,
        memory_mib=parse_memory(mem or ("12GiB" if inferred_gpu else "1GiB")),
        gpus=int(inferred_gpu) if gpus is None else gpus,
        nodes=nodes,
        ranks=ranks,
        runtime_seconds=runtime,
        runtime_confidence="declared" if runtime else "unknown",
        timeout_seconds=timeout,
    )
    selected_mode = mode
    if not partition and mode == "auto":
        preferences = {config.profiles[name].execution_mode for name in names}
        if len(preferences) == 1 and "auto" not in preferences:
            selected_mode = next(iter(preferences))
    selected_mode = choose_mode(selected_mode, resources, 1, config)[0]
    if partition:
        config.profiles[profile] = Profile(
            partition=partition,
            account=config.control.account,
            cpus=resources.cpus,
            memory_mib=resources.memory_mib
            + (config.policy.worker_overhead_mib if selected_mode == "pooled" else 0),
            gres=f"gpu:{resources.gpus}" if resources.gpus else None,
            allocation_seconds=parse_time(time or "01:00:00"),
            budget=partition if partition in config.budgets else None,
        )
    if selected_mode == "direct":
        for name in names:
            allocation = config.profiles[name]
            if cpus is not None:
                allocation.cpus = resources.cpus
            if mem is not None:
                allocation.memory_mib = resources.memory_mib
            if gpus is not None:
                if not gpus:
                    allocation.gres = None
                elif allocation.gres:
                    import re

                    match = re.fullmatch(r"gpu(?::([A-Za-z0-9_-]+))?:\d+", allocation.gres)
                    if not match:
                        raise typer.BadParameter("GPU count override needs a simple GPU GRES profile")
                    allocation.gres = f"gpu:{match[1] + ':' if match[1] else ''}{gpus}"
                else:
                    allocation.gres = f"gpu:{gpus}"
    if time:
        for name in names:
            config.profiles[name].allocation_seconds = parse_time(time)
    if partition:
        import hashlib

        definition = config.profiles[profile].model_dump_json()
        unique = "explicit_" + hashlib.sha256(definition.encode()).hexdigest()[:16]
        config.profiles[unique] = config.profiles.pop(profile)
        profile = unique
    return config, profile, resources


@app.command()
def discover(
    output_dir: Annotated[Path, typer.Option()] = Path(".local/discovery"), raw_dir: Path | None = None
):
    """Read cluster configuration; no submissions or cancellations."""
    from .discovery import discover as run

    output(run(output_dir=output_dir, raw_dir=raw_dir))


@app.command()
def doctor():
    from .diagnostics import doctor as run

    output(run(cfg()))


@config_app.command("show")
def config_show(resolved: bool = typer.Option(True, "--resolved/--unresolved")):
    output(cfg())


@app.command()
def plan(
    commands: Path,
    profile: str | None = None,
    partition: str | None = None,
    mode: str = "auto",
    cpus: int | None = None,
    mem: str | None = None,
    gpus: int | None = None,
    time: str | None = None,
    runtime: float | None = None,
    timeout: float | None = None,
    max_parallel: int | None = None,
    commands_per_job: int = 1,
    literal: bool = False,
    format: str = "lines",
    nodes: int = 1,
    ranks: int = 1,
):
    """Preview only. CPU/mem flags override direct allocations; pooled flags describe tasks.

    MPI layouts need an explicit launcher such as srun in the command itself.
    """
    c, p, r = options(profile, partition, cpus, mem, gpus, time, runtime, timeout, mode, nodes, ranks)
    output(
        api.plan(
            commands,
            c,
            profile=p,
            mode=mode,
            resources=r,
            max_parallel=max_parallel,
            commands_per_job=commands_per_job,
            literal=literal,
            format=format,
        )
    )


@app.command()
def submit(
    commands: Path,
    profile: str | None = None,
    partition: str | None = None,
    mode: str = "auto",
    cpus: int | None = None,
    mem: str | None = None,
    gpus: int | None = None,
    time: str | None = None,
    runtime: float | None = None,
    timeout: float | None = None,
    max_parallel: int | None = None,
    commands_per_job: int = 1,
    literal: bool = False,
    format: str = "lines",
    name: str | None = None,
    cwd: Path | None = None,
    import_key: str | None = None,
    repeatable: bool | None = None,
    max_attempts: int | None = None,
    output_mode: str | None = None,
    no_service: bool = False,
    dry_run: bool = False,
    nodes: int | None = None,
    ranks: int | None = None,
    workload: str = "generic",
    priority: int = 0,
    input_fingerprint: str | None = None,
    software_fingerprint: str | None = None,
    recovery_manifest: Path | None = None,
):
    """Import once; direct arrays continue after this CLI exits.

    Direct --cpus/--mem override allocation resources. In pooled mode they are
    task requirements that must fit a configured worker. MPI needs an explicit
    launcher in each command; --nodes/--ranks only reserve the Slurm layout.
    """
    c, p, r = options(profile, partition, cpus, mem, gpus, time, runtime, timeout, mode,
                      1 if nodes is None else nodes, 1 if ranks is None else ranks)
    recovered = None
    if recovery_manifest is not None:
        from .recovery_export import load_export
        from .state import fingerprint

        recovered = load_export(recovery_manifest)
        if Path(c.state_dir).expanduser().resolve() != Path(recovered["source_state_dir"]):
            raise typer.BadParameter("Recovery imports must use the original state directory for duplicate protection")
        if commands.resolve() != Path(recovered["commands_file"]):
            raise typer.BadParameter("Commands path must match the recovery manifest")
        if not recovered["count"]:
            raise typer.BadParameter("There are no remaining commands to submit")
        execution = recovered["execution_settings"]
        for label, supplied, original in (
            ("nodes", nodes, execution["resources"]["nodes"]),
            ("ranks", ranks, execution["resources"]["ranks"]),
            ("output-mode", output_mode, execution["output"]["mode"]),
            ("repeatable", repeatable, execution["output"]["repeatable"]),
        ):
            if supplied is not None and supplied != original:
                raise typer.BadParameter(f"Recovery preserves {label}={original}; changing it requires a separately reviewed plan")
        if cwd is not None and cwd.resolve() != Path(execution["cwd"]):
            raise typer.BadParameter("Recovery preserves the original working directory")
        if input_fingerprint is not None or software_fingerprint is not None:
            raise typer.BadParameter("Recovery preserves original input/software fingerprints")
        key = "remaining:" + fingerprint({
            "manifest": str(recovery_manifest.expanduser().resolve()),
            "source_state_dir": recovered["source_state_dir"], "campaign_id": recovered["campaign_id"],
        })
        if import_key is not None and import_key != key:
            raise typer.BadParameter("Recovery uses one stable import key; do not duplicate the exported generation")
        import_key, format, literal = key, "jsonl", False
        saved_resources = execution["resources"]
        supplied = {"cpus": cpus, "memory_mib": mem, "gpus": gpus,
                    "runtime_seconds": runtime, "timeout_seconds": timeout}
        r = Resources.model_validate({
            **saved_resources,
            **{key: getattr(r, key) for key, value in supplied.items() if value is not None},
            **({"runtime_confidence": "declared"} if runtime is not None else {}),
        })
    if dry_run:
        output(
            api.plan(
                commands,
                c,
                profile=p,
                mode=mode,
                resources=r,
                max_parallel=max_parallel,
                commands_per_job=commands_per_job,
                literal=literal,
                format=format,
            )
        )
        return
    output(
        api.submit(
            commands,
            c,
            profile=p,
            mode=mode,
            resources=r,
            max_parallel=max_parallel,
            commands_per_job=commands_per_job,
            literal=literal,
            format=format,
            name=name,
            cwd=recovered["execution_settings"]["cwd"] if recovered else cwd,
            import_key=import_key,
            retry=RetryPolicy.model_validate({
                **recovered["execution_settings"]["retry"],
                **({"max_attempts": max_attempts} if max_attempts is not None else {}),
            }) if recovered else RetryPolicy(max_attempts=1 if max_attempts is None else max_attempts),
            output=OutputPolicy.model_validate(recovered["execution_settings"]["output"])
            if recovered else OutputPolicy(mode=output_mode or "manual_on_ambiguity", repeatable=bool(repeatable)),
            environment=recovered["execution_settings"]["environment"] if recovered else None,
            start_service=not no_service,
            workload=recovered["execution_settings"]["workload"] if recovered else workload,
            priority=recovered["execution_settings"]["priority"] if recovered else priority,
            input_fingerprint=recovered["execution_settings"]["input_fingerprint"] if recovered else input_fingerprint,
            software_fingerprint=recovered["execution_settings"]["software_fingerprint"] if recovered else software_fingerprint,
            shell=recovered["execution_settings"]["shell"] if recovered else "/bin/bash",
            shell_options=recovered["execution_settings"]["shell_options"] if recovered else None,
        )
    )
    print_access(c)


@app.command("export-remaining")
def export_remaining(campaign_id: str, destination: Path):
    """Retire quiescent unfinished occurrences and prepare exact commands; submit nothing."""
    from .recovery_export import export_remaining as prepare

    result = prepare(cfg(), campaign_id, destination)
    output({
        "manifest": str(destination.expanduser().resolve() / "manifest.json"),
        "commands_file": result["commands_file"], "count": result["count"],
        "excluded": result["excluded"], "submitted": False,
        "source_occurrences": "blocked; ordinary resume cannot execute exported occurrences",
        "next": "hhq submit COMMANDS --recovery-manifest MANIFEST --profile PROFILE --mem NEW_MEMORY --runtime SECONDS",
    })


@app.command()
def append(campaign_id: str, commands: Path, import_key: str | None = None):
    output(api.append(campaign_id, commands, cfg(), import_key=import_key))


@app.command()
def resume(campaign_id: str):
    from .backends.direct import submit_ready
    from .backends.slurm import SlurmClient
    from .state import StateStore

    c = cfg()
    store = StateStore(c.state_dir)
    campaign = store.campaign(campaign_id)
    api.campaign_control(c, campaign_id, "resume")
    if campaign["mode"] == "direct":
        api.status(c, reconcile=True)
        with store.lock():
            output(
                submit_ready(
                    store,
                    c,
                    campaign_id,
                    SlurmClient(),
                    max_parallel=campaign.get("max_parallel"),
                    commands_per_job=campaign.get("commands_per_job", 1),
                )
            )
    else:
        output({"campaign_id": campaign_id, "state": "admission_resumed",
                "next": "An active controller admits work on its next cycle. If stopped, explicitly run hhq service start."})


@app.command()
def status(json_output: bool = typer.Option(False, "--json"), reconcile: bool = False):
    result = api.status(cfg(), reconcile=reconcile)
    if json_output:
        output(result)
        return
    table = Table("Campaign", "Name", "Mode", "Tasks", "States")
    for c in result["campaigns"]:
        table.add_row(c["campaign_id"], c["name"], c["mode"], str(c["tasks"]), str(c["counts"]))
    Console().print(table)
    typer.echo(result["note"])
    typer.echo("Controller: " + str(result["controller"]))


@app.command()
def explain(task_id: str):
    output(api.explain(task_id, cfg()))


@app.command()
def metrics(campaign_id: str | None = None, minimum_samples: int = 5):
    """Summarize successful measurements; never change resource requests automatically."""
    from .metrics import measurements

    if minimum_samples < 1:
        raise typer.BadParameter("minimum-samples must be positive")
    output(measurements(cfg(), campaign_id=campaign_id, minimum_samples=minimum_samples))


@app.command()
def logs(task_id: str, attempt: str = "latest", stderr: bool = False, lines: int = 80):
    from .logs import attempt_log_path
    from .state import StateStore

    store = StateStore(cfg().state_dir)
    cid = api.find_task(store, task_id)
    items = store.attempts(cid, task_id)
    if not items:
        raise typer.BadParameter("Task has not started")
    item = items[-1] if attempt == "latest" else next((x for x in items if x["attempt_id"] == attempt), None)
    if item is None:
        raise typer.BadParameter("Unknown attempt")
    path = attempt_log_path(item, "stderr" if stderr else "stdout")
    if not path.exists():
        matches = list((store.campaign_dir(cid) / "attempts" / task_id).glob("*/attempt.json"))
        for m in matches:
            if json.loads(m.read_text())["attempt_id"] == item["attempt_id"]:
                path = m.parent / path.name
                break
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 1024 * 1024))
        text = stream.read(1024 * 1024).decode("utf-8", errors="replace")
    typer.echo("\n".join(text.splitlines()[-max(1, min(lines, 10000)) :]))


@app.command()
def retry(campaign_id: str, failed: bool = typer.Option(False, "--failed")):
    """Authorize bounded retries of known failures; successes/cancellations stay intact."""
    if not failed:
        raise typer.BadParameter("Specify --failed; ambiguous attempts need explicit fencing evidence")
    from .runner import authorize_retry
    from .state import StateStore

    store = StateStore(cfg().state_dir)
    accepted = []
    blocked = {}
    for tid in store.task_ids(campaign_id):
        if store.task_state(campaign_id, tid).get("state") not in ("failed", "interrupted"):
            continue
        try:
            authorize_retry(store, campaign_id, tid, manual=True)
            accepted.append(tid)
        except (ValueError, RuntimeError) as exc:
            blocked[tid] = str(exc)
    output(
        {
            "authorized": accepted,
            "blocked": blocked,
            "next": "hhq resume CAMPAIGN for direct; pooled work needs an active controller. If stopped, explicitly run hhq service start.",
        }
    )


@app.command()
def cancel(campaign_id: str, pending: bool = False, running: bool = False):
    """Cancel only this campaign's positively identified backend jobs."""
    if not (pending or running):
        raise typer.BadParameter("Choose --pending, --running, or both")
    from .backends.slurm import SlurmClient
    from .controller import mapping
    from .state import StateStore

    c = cfg()
    store = StateStore(c.state_dir)
    slurm = SlurmClient()
    canceled = []
    with store.lock():
        campaign = store.campaign(campaign_id)
        if campaign["mode"] == "pooled" and not (pending and running):
            raise typer.BadParameter(
                "HQ cancellation is job-wide: specify --pending --running; use campaign drain to preserve admitted tasks"
            )
        campaign["draining"] = True
        store.save_campaign(campaign)
        intents = [
            i
            for i in store.intents()
            if i.get("campaign_id") == campaign_id and i.get("state") in ("submitted", "running", "unknown")
        ]
        if campaign["mode"] == "pooled":
            if not (pending and running):
                raise typer.BadParameter(
                    "HQ cancellation is job-wide: specify --pending --running; use campaign drain to preserve admitted tasks"
                )
            # Submit a durable control request. Only the leader mutates HQ.
            from .state import atomic_json

            path = store.root / "requests" / f"cancel-{campaign_id}.json"
            atomic_json(path, {"action": "cancel_hq_campaign", "campaign_id": campaign_id})
            output({"queued_control_request": str(path), "campaign_id": campaign_id})
            return
        jobs = [mapping(j) for j in slurm.query_jobs()]
        for intent in intents:
            ids = [
                j["job_id"]
                for j in jobs
                if str(j.get("array_job_id") or j["job_id"]).split("_")[0] == str(intent.get("job_id"))
                and (
                    (pending and j["state"] == "PENDING")
                    or (running and j["state"] in ("RUNNING", "CONFIGURING", "COMPLETING"))
                )
            ]
            if not ids:
                continue
            # Preserve cancellation intent before scheduler mutation; individual
            # array outcome reconciliation determines which commands were affected.
            intent["cancel_requested_ids"] = ids
            store.put_intent(intent)
            slurm.cancel(ids, expected_token=intent["token"])
            canceled.extend(ids)
        store.event("user_cancel", campaign_id=campaign_id, job_ids=canceled)
    output({"campaign_id": campaign_id, "canceled_job_ids": canceled, "unsent_work": "drained"})


@service_app.command("start")
def service_start(hosting: str | None = None, partition: str | None = None, standby: int | None = None):
    from .service.supervisor import start_service

    c = cfg()
    if partition:
        c.control.partition = partition
    output(
        start_service(
            c,
            hosting=hosting or c.control.hosting,
            standby=c.control.standby_replicas if standby is None else standby,
        )
    )


@service_app.command("status")
def service_status():
    from .service.supervisor import service_status as run

    output(run(cfg()))


@service_app.command("stop")
def service_stop(drain: bool = True):
    from .service.supervisor import stop_service

    output(stop_service(cfg(), drain=drain))


@campaign_app.command("drain")
def campaign_drain(campaign_id: str):
    output(api.campaign_control(cfg(), campaign_id, "drain"))


@campaign_app.command("control")
def campaign_control(campaign_id: str, action: str):
    output(api.campaign_control(cfg(), campaign_id, action))


@pool_app.command("pause")
def pool_pause(profile: str, scope: str = "allocations"):
    if scope != "allocations":
        raise typer.BadParameter("Pool control only pauses new allocations; existing workers continue")
    from .state import StateStore, atomic_json

    c = cfg()
    if profile not in c.profiles:
        raise typer.BadParameter("Unknown pool")
    store = StateStore(c.state_dir)
    with store.lock():
        path = store.root / "controls.json"
        d = json.loads(path.read_text()) if path.exists() else {}
        d["paused_pools"] = sorted(set(d.get("paused_pools", [])) | {profile})
        atomic_json(path, d)
    output({"profile": profile, "scope": "new_allocations", "existing_work": "continues"})


@pool_app.command("resume")
def pool_resume(profile: str, reset_startup_failures: bool = False):
    """Resume allocations; explicitly reset a startup circuit only after repairing its cause."""
    from .state import StateStore, atomic_json, utc_now

    config = cfg()
    if profile not in config.profiles:
        raise typer.BadParameter("Unknown pool")
    store = StateStore(config.state_dir)
    with store.lock():
        path = store.root / "controls.json"
        d = json.loads(path.read_text()) if path.exists() else {}
        d["paused_pools"] = sorted(set(d.get("paused_pools", [])) - {profile})
        if reset_startup_failures:
            d.setdefault("startup_reset_after", {})[profile] = utc_now()
        atomic_json(path, d)
        store.event("pool_resumed", profile=profile, reset_startup_failures=reset_startup_failures)
    output(
        {
            "profile": profile,
            "scope": "new_allocations",
            "state": "resumed",
            "startup_failures_reset": reset_startup_failures,
        }
    )


@app.command()
def ui(
    host: str = "127.0.0.1",
    port: int = 8765,
    read_only: bool = typer.Option(True, "--read-only/--allow-submit"),
    export: Annotated[
        Path | None, typer.Option("--export", help="Save standalone HTML instead of hosting.")
    ] = None,
):
    """View live progress locally, or export a private HTML snapshot."""
    from .ui import export_dashboard, serve

    if export is not None:
        typer.echo(str(export_dashboard(cfg(), export)))
    else:
        serve(cfg(), host=host, port=port, read_only=read_only)


@app.command("support-bundle")
def support_bundle(destination: Path):
    from .diagnostics import support_bundle as run

    output(run(cfg(), destination))


def entrypoint():
    try:
        app()
    except (ValueError, RuntimeError, OSError) as exc:
        if settings.get("debug"):
            raise
        console.print(f"[red]Error:[/red] {exc}", markup=True, highlight=False)
        raise SystemExit(2)


if __name__ == "__main__":
    # Nested commands import .cli.cfg; keep the -m invocation's parsed settings
    # instead of importing a second module with an empty settings dictionary.
    sys.modules["hyper_hyperqueue.cli"] = sys.modules[__name__]
    entrypoint()
