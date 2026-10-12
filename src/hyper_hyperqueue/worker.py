"""Start one HQ worker using only the resources assigned to this allocation."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path

from .backends.hyperqueue import check_version, locate_binary, resource_label
from .gpu import query_assigned_devices


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
    except Exception as error:
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
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
