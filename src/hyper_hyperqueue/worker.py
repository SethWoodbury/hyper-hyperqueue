"""Start one HQ worker using only the resources assigned to this allocation."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from .backends.hyperqueue import check_version, locate_binary, resource_label
from .gpu import query_assigned_devices
from .startup_probe import run_startup_probe


def detect_assigned(profile, *, environment=None, affinity=None, gpu_query=None) -> dict:
    env = os.environ if environment is None else environment
    allowed = sorted(os.sched_getaffinity(0)) if affinity is None else sorted(affinity)
    cpus = min(profile.cpus, len(allowed))
    if env.get("SLURM_CPUS_PER_TASK"):
        cpus = min(cpus, int(env["SLURM_CPUS_PER_TASK"]))
    if cpus < 1:
        raise ValueError("No assigned CPUs available")
    memory = profile.memory_mib
    if env.get("SLURM_MEM_PER_NODE"):
        memory = min(memory, int(env["SLURM_MEM_PER_NODE"]))
    elif env.get("SLURM_MEM_PER_CPU"):
        memory = min(memory, int(env["SLURM_MEM_PER_CPU"]) * cpus)
    gpu = None
    if profile.gres:
        match = re.fullmatch(r"gpu(?::([^:]+))?:(\d+)", profile.gres)
        if not match or int(match[2]) != 1:
            raise ValueError("Pooled worker currently requires GRES gpu:1 or gpu:TYPE:1")
        gpu = query_assigned_devices(1, environment=env, query=gpu_query)[0]
        gpu["gres_type"] = match[1]
        if profile.gpu_models and gpu["model"] not in profile.gpu_models:
            raise ValueError(f"Assigned GPU {gpu['model']} is outside the profile model allowlist")
        if profile.min_vram_mib and gpu["memory_mib"] < profile.min_vram_mib:
            raise ValueError("Assigned GPU has insufficient VRAM for this profile")
    return {
        "cpus": cpus,
        "memory_mib": memory,
        "affinity": allowed,
        "gpu": gpu,
        "source": "profile bounded by affinity and Slurm allocation environment",
    }


def allocation_remaining(*, environment=None, now=None, query=None):
    """Read allocated time, never estimate it from queue/submission elapsed time."""
    env = os.environ if environment is None else environment
    now = time.time() if now is None else now
    if not env.get("SLURM_JOB_ID"):
        return None
    if env.get("SLURM_JOB_END_TIME"):
        end = float(env["SLURM_JOB_END_TIME"])
    else:
        if query is None:
            from .backends.slurm import SlurmClient

            query = SlurmClient().query_jobs
        rows = [r for r in query() if str(r.job_id) == env["SLURM_JOB_ID"]]
        if len(rows) != 1 or rows[0].state != "RUNNING":
            raise ValueError("Cannot prove this allocation's actual remaining lifetime")
        row = rows[0]
        end = row.end_time or (
            row.start_time + row.time_limit_seconds if row.start_time and row.time_limit_seconds else None
        )
        if end is None:
            raise ValueError("Slurm did not report a bounded allocation end time")
    remaining = int(end - now)
    if remaining <= 0:
        raise ValueError("Allocation has expired")
    return remaining


def worker_argv(config, profile_name: str, assigned: dict, *, remaining_seconds: int | None = None):
    profile = config.profiles[profile_name]
    binary = locate_binary(config.control.hq_binary)
    memory = assigned["memory_mib"] - config.policy.worker_overhead_mib
    if memory <= 0:
        raise ValueError("Worker overhead leaves no usable host memory")
    lifetime = min(
        profile.allocation_seconds,
        profile.allocation_seconds if remaining_seconds is None else remaining_seconds,
    )
    lifetime -= config.policy.cleanup_seconds
    if lifetime <= 0:
        raise ValueError("Allocation has no remaining useful lifetime")
    resources = {"hhq/memory": memory, resource_label("profile", profile_name): assigned["cpus"]}
    if assigned["gpu"]:
        gpu = assigned["gpu"]
        resources.update({"hhq/gpu": 1, "hhq/vram": gpu["memory_mib"]})
        resources[resource_label("model", gpu["model"])] = 1
        if gpu["gres_type"]:
            resources[resource_label("gpu_type", gpu["gres_type"])] = 1
    else:
        resources["hhq/cpu"] = assigned["cpus"]
    argv = [
        binary,
        "--server-dir",
        str(Path(config.state_dir) / "service/hq"),
        "worker",
        "start",
        "--detect-resources=none",
        "--cpus",
        str(assigned["cpus"]),
        "--manager=none",
        "--on-server-lost=stop",
        "--time-limit",
        f"{lifetime}s",
        "--idle-timeout",
        f"{config.policy.worker_idle_seconds}s",
    ]
    for name, value in sorted(resources.items()):
        argv.extend(["--resource", f"{name}=sum({value})"])
    return argv


def run_worker_phase(command, *, clock=time.time):
    """Keep a small parent to distinguish natural lifetime expiry from signals."""
    started = clock()
    lifetime = int(command[command.index("--time-limit") + 1][:-1])
    received, previous = [], {}
    process = subprocess.Popen(command)

    def forward(signum, _frame):
        received.append(signum)
        if process.poll() is None:
            try:
                process.send_signal(signum)
            except ProcessLookupError:
                pass

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1):
        previous[signum] = signal.signal(signum, forward)
    try:
        code = process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    finished = clock()
    return {"started": started, "configured_end_time": started + lifetime,
            "finished": finished, "returncode": code, "received_signals": received,
            "termination_reason": ("signal" if received or code < 0 else
                                   "time_limit" if finished >= started + lifetime else "normal_exit")}


def select_fallback(store, config, primary, assigned, epoch, token, remaining, *, clock=None):
    """One conservative phase switch after idle exit; admitted HQ tasks never change.

    Caller holds StateStore.lock. A compatible running/unknown primary worker
    blocks fallback. Pending allocations are not available worker capacity yet.
    """
    from datetime import datetime

    from .adaptive_recovery import effective_task
    from .models import Resources
    from .planner import ACTIVE_INTENT_STATES, compatible
    from .service.completion import work_status
    from .state import read_json

    clock = time.time() if clock is None else clock
    try:
        endpoint = read_json(store.root / "service/endpoint.json")
        status = read_json(store.root / "controller_status.json")
        deployment = read_json(store.root / "service/deployment.json")
        controls = read_json(store.root / "controls.json") if (store.root / "controls.json").exists() else {}
    except (OSError, ValueError):
        return None
    if deployment.get("desired_state") != "RUNNING":
        return None
    if (endpoint.get("epoch") != epoch or endpoint.get("status") != "ACTIVE"
            or status.get("epoch") != epoch or status.get("state") != "healthy"):
        return None
    try:
        age = clock - datetime.fromisoformat(status["at"]).timestamp()
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= age <= max(30, 2 * config.policy.reconcile_interval_seconds):
        return None
    if endpoint.get("control_end_time") is not None and endpoint["control_end_time"] <= clock + config.policy.cleanup_seconds:
        return None
    blockers = work_status(store, status.get("retry_decisions", []))["blockers"]
    if any("token" in row for row in blockers):
        return None
    blocked = {row.get("task_id") for row in blockers}
    waiting, primary_seen = [], False
    for campaign in store.campaigns():
        if campaign["mode"] != "pooled":
            continue
        for tid in store.task_ids(campaign["campaign_id"]):
            task = effective_task(store, store.task(campaign["campaign_id"], tid))
            state = store.task_state(task.campaign_id, tid).get("state")
            if not task.allowed_profiles or primary in task.allowed_profiles:
                primary_seen = True
                if tid in blocked or state not in {"succeeded", "failed", "canceled"}:
                    return None
            if not campaign.get("draining") and not campaign.get("pause_dispatch") and state == "submitted":
                waiting.append(task)
    if not primary_seen:
        return None
    observed = {row["token"]: row for row in status.get("worker_capacity", [])}
    active = [row for row in store.intents() if row.get("backend") == "worker"
              and row.get("state") in ACTIVE_INTENT_STATES and row["token"] != token]
    for target in config.profiles[primary].fallback_profiles:
        if primary in controls.get("paused_pools", []) or target in controls.get("paused_pools", []):
            continue
        placement = ("partition", "account", "qos", "gres", "constraint", "exclude", "gpu_models", "min_vram_mib",
                     "startup_probe_argv", "startup_probe_timeout_seconds")
        if any(getattr(config.profiles[target], key) != getattr(config.profiles[primary], key) for key in placement):
            continue  # An alias cannot bypass a task's permitted hardware or queue.
        physical = config.profiles[primary].model_copy(update={
            "cpus": assigned["cpus"], "memory_mib": assigned["memory_mib"],
            "allocation_seconds": max(1, int(remaining)),
        })
        candidates = [task for task in waiting if (not task.allowed_profiles or target in task.allowed_profiles)
                      and compatible(Resources.model_validate(task.resources), physical, config)[0]]
        if not candidates:
            continue
        # All routes already allowed for the waiting tasks count as primary
        # capacity; a fresh phase claim also blocks another opportunistic worker.
        routes = {p for task in candidates for p in task.allowed_profiles}
        blocked_capacity = False
        for allocation in active:
            route = allocation.get("dispatch_profile", allocation["profile"])
            if routes and route not in routes:
                continue
            evidence = observed.get(allocation["token"])
            if allocation.get("dispatch_profile") or not evidence or evidence["state"] != "PENDING":
                blocked_capacity = True
                break
        if not blocked_capacity:
            return target
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    args = parser.parse_args(argv)
    import socket

    from .config import load_config
    from .service.leadership import process_identity
    from .state import atomic_json

    config = load_config(args.config, overrides={"state_dir": args.state_dir})
    endpoint = json.loads((Path(config.state_dir) / "service/endpoint.json").read_text())
    if endpoint.get("epoch") != args.epoch or endpoint.get("status") != "ACTIVE":
        raise SystemExit("Server generation is not active; worker will not connect to a stale generation")
    identity = os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")
    preflight_path = Path(config.state_dir) / "service/worker_preflight" / f"{identity}.json"
    preflight = {
        "status": "starting",
        "epoch": args.epoch,
        "profile": args.profile,
        "gres": config.profiles[args.profile].gres,
        "host": socket.gethostname(),
        "allocation_token": os.environ.get("HHQ_ALLOCATION_TOKEN"),
        "started": time.time(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {
            key: os.environ[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "SLURM_JOB_GPUS",
                "SLURM_STEP_GPUS",
                "SLURM_JOB_ID",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "SLURM_MEM_PER_CPU",
            )
            if key in os.environ
        },
    }
    atomic_json(preflight_path, preflight)
    try:
        assigned = detect_assigned(config.profiles[args.profile])
        probe = run_startup_probe(config.profiles[args.profile])
        if probe is not None:
            preflight["startup_probe"] = probe
    except Exception as error:
        if hasattr(error, "result"):
            preflight["startup_probe"] = error.result
        atomic_json(
            preflight_path,
            {
                **preflight,
                "status": "failed",
                "finished": time.time(),
                "error": f"{type(error).__name__}: {error}"[-4000:],
            },
        )
        raise
    atomic_json(
        preflight_path, {**preflight, "status": "passed", "finished": time.time(), "assigned": assigned}
    )
    remaining = allocation_remaining()
    if endpoint.get("control_end_time"):
        control_remaining = int(endpoint["control_end_time"] - time.time())
        remaining = min(remaining, control_remaining) if remaining is not None else control_remaining
    command = worker_argv(config, args.profile, assigned, remaining_seconds=remaining)
    check_version(command[0])
    path = Path(config.state_dir) / "service/workers" / f"{identity}.json"
    atomic_json(
        path,
        {
            "epoch": args.epoch,
            "profile": args.profile,
            "assigned": assigned,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "pid": os.getpid(),
            "process": process_identity(os.getpid()),
            "host": socket.gethostname(),
            "allocation_token": os.environ.get("HHQ_ALLOCATION_TOKEN"),
            "started": time.time(),
            "argv": command,
        },
    )
    os.environ["HHQ_SERVER_EPOCH"] = str(args.epoch)
    # This allocation epoch survives restoration of older HQ dispatch records.
    os.environ["HHQ_WORKER_EPOCH"] = str(args.epoch)
    os.environ["HHQ_WORKER_PROFILE"] = args.profile
    # Do not alter CUDA_VISIBLE_DEVICES or container visibility. A single
    # logical GPU slot prevents concurrent GPU tasks without VRAM assumptions.
    work = tempfile.mkdtemp(prefix="hhq-worker-")
    command.extend(["--work-dir", work])
    from .state import StateStore

    store = StateStore(config.state_dir)
    token = os.environ.get("HHQ_ALLOCATION_TOKEN")
    record = {"schema_version": 1, "epoch": args.epoch, "profile": args.profile,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "allocation_token": token}
    first = run_worker_phase(command)
    phases = [{"dispatch_profile": args.profile, **first}]
    # One phase change only, after a normal exit lasting at least the idle grace.
    # A time limit, signal, lost/stale controller or unresolved work never switches.
    if (config.profiles[args.profile].fallback_profiles and first["returncode"] == 0
            and first["termination_reason"] == "normal_exit"
            and first["finished"] - first["started"] >= config.policy.worker_idle_seconds):
        remaining = allocation_remaining()
        bound = first["configured_end_time"] - time.time() + config.policy.cleanup_seconds
        remaining = min(remaining, bound) if remaining is not None else bound
        with store.lock():
            target = select_fallback(store, config, args.profile, assigned, args.epoch, token, remaining)
            if target:
                owned = [i for i in store.intents() if i.get("token") == token and i.get("epoch") == args.epoch]
                if len(owned) != 1:
                    target = None
                else:
                    intent = owned[0]
                    intent.update(dispatch_profile=target, fallback_started=time.time())
                    store.put_intent(intent)
        if target:
            # A server lost between the last controller report and this phase
            # switch is not permission to create a new worker generation.
            from .backends.hyperqueue import HQBackend, HQError

            try:
                HQBackend(store.root / "service/hq", command[0]).json("server", "info")
            except (HQError, OSError):
                target = None
        if target:
            os.environ["HHQ_WORKER_PROFILE"] = target
            phase_config = config.model_copy(deep=True)
            phase_config.profiles[target].allocation_seconds = config.profiles[args.profile].allocation_seconds
            second = worker_argv(phase_config, target, assigned, remaining_seconds=int(remaining))
            second.extend(["--work-dir", work])
            phases.append({"dispatch_profile": target, **run_worker_phase(second)})
    atomic_json(store.root / "service/worker_exits" / f"{identity}.json",
                {**record, **phases[-1], "phases": phases})
    code = phases[-1]["returncode"]
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
