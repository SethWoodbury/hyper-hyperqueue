"""Cheap process-tree measurements; no all-node GPU sampling or peak guarantees."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

import psutil

from .gpu import query_assigned_devices


class ResourceMismatch(ValueError):
    pass


def assigned_gpu_context(resources, *, environment=None, query=None) -> dict:
    """Inspect only devices positively assigned to this task; no node-wide query."""
    env = os.environ if environment is None else environment
    if resources.gpus == 0:
        return {"devices": [], "source": "CPU-only task; no GPU inspection"}
    try:
        devices = query_assigned_devices(resources.gpus, environment=env, query=query)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise ResourceMismatch(f"Assigned GPU identity could not be verified: {exc}") from exc
    for device in devices:
        if resources.gpu_models and device["model"] not in resources.gpu_models:
            raise ResourceMismatch(f"Assigned GPU model {device['model']} is outside task allowlist")
        if resources.min_vram_mib and device["memory_mib"] < resources.min_vram_mib:
            raise ResourceMismatch("Assigned GPU has less VRAM than the task requires")
    return {
        "devices": devices,
        "source": "CUDA driver metadata under inherited visibility; no context created",
    }


def allocation_context() -> dict:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    names = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_CPUS_PER_TASK",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "HQ_JOB_ID",
        "HQ_TASK_ID",
        "HQ_WORKER_ID",
        "HHQ_SERVER_EPOCH",
        "HHQ_WORKER_EPOCH",
        "HHQ_WORKER_PROFILE",
    )
    return {
        "hostname": __import__("socket").gethostname(),
        "cpu_affinity": affinity,
        "environment": {name: os.environ[name] for name in names if name in os.environ},
        "gpu_telemetry": "assigned-UUID VRAM sampling, when available, is recorded in execution telemetry",
    }


def job_cgroup(environment, *, proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
    """Locate only this Slurm job's v2 cgroup; never report a whole-node group."""
    job_id = environment.get("SLURM_JOB_ID", "")
    if not str(job_id).isdigit():
        return None
    try:
        lines = (proc_root / "self/cgroup").read_text().splitlines()
        relative = next(line[3:] for line in lines if line.startswith("0::"))
        parts = Path(relative).parts
        if ".." in parts:
            return None
        index = parts.index(f"job_{job_id}")
        candidate = cgroup_root.joinpath(*[p for p in parts[:index + 1] if p != "/"])
        if not candidate.resolve().is_relative_to(cgroup_root.resolve()):
            return None
        return candidate if (candidate / "memory.current").is_file() else None
    except (OSError, ValueError, StopIteration):
        return None


class AllocationSampler:
    """Best-effort allocation measurements; no scheduler calls or CUDA contexts."""
    def __init__(self, devices=(), *, environment=None, query=subprocess.run,
                 proc_root=Path("/proc"), cgroup_root=Path("/sys/fs/cgroup")):
        self.cgroup = job_cgroup(os.environ if environment is None else environment,
                                 proc_root=proc_root, cgroup_root=cgroup_root)
        self.cgroup_current_peak = None
        self.cgroup_lifetime_peak = None
        self.cgroup_start_peak = None
        self.uuids = [d["uuid"] for d in devices]
        if any(not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", value) for value in self.uuids):
            self.uuids = []
        self.query = query
        self.gpu_peak_mib = {}
        self.gpu_samples = 0
        self.gpu_error = None
        self.next_gpu_sample = 0.0

    def sample(self):
        if self.cgroup is not None:
            for filename, attribute in (("memory.current", "cgroup_current_peak"),
                                        ("memory.peak", "cgroup_lifetime_peak")):
                try:
                    value = int((self.cgroup / filename).read_text().strip())
                    if value < 0:
                        continue
                    if filename == "memory.peak" and self.cgroup_start_peak is None:
                        self.cgroup_start_peak = value
                    setattr(self, attribute, max(getattr(self, attribute) or 0, value))
                except (OSError, ValueError):
                    pass  # Unsupported/inaccessible telemetry never changes task status.
        now = time.monotonic()
        if not self.uuids or self.gpu_error is not None or now < self.next_gpu_sample:
            return
        self.next_gpu_sample = now + 10.0
        try:
            result = self.query(["nvidia-smi", f"--id={','.join(self.uuids)}",
                                 "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=1, check=False)
            if result.returncode:
                raise ValueError(f"nvidia-smi exited {result.returncode}")
            values = {}
            for line in result.stdout.splitlines():
                identity, memory = [v.strip() for v in line.split(",")]
                values[identity] = int(memory)
            if set(values) != set(self.uuids) or any(v < 0 for v in values.values()):
                raise ValueError("VRAM response did not match assigned GPU UUIDs")
            for identity, memory in values.items():
                self.gpu_peak_mib[identity] = max(self.gpu_peak_mib.get(identity, 0), memory)
            self.gpu_samples += 1
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.gpu_error = str(exc)[:300]  # Disable a failing query, rather than repeated overhead.

    def summary(self):
        return {
            "cgroup_path": str(self.cgroup) if self.cgroup else None,
            "sampled_job_memory_current_peak_bytes": self.cgroup_current_peak,
            "job_memory_lifetime_peak_bytes": self.cgroup_lifetime_peak,
            "job_memory_lifetime_peak_at_start_bytes": self.cgroup_start_peak,
            "assigned_gpu_sampled_peak_mib": self.gpu_peak_mib,
            "gpu_samples": self.gpu_samples, "gpu_interval_seconds": 10,
            "gpu_query_error": self.gpu_error,
            "scope": "Job cgroup includes all job processes/cache and concurrent tasks; memory.peak includes earlier tasks. GPU VRAM is device usage, sampled only on assigned UUIDs; brief peaks may be missed. Missing values are unknown, not zero.",
        }


class ProcessSampler:
    def __init__(self, pid: int, interval_seconds: float = 0.2, *, devices=(), environment=None):
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.samples = 0
        self.peak_rss_bytes: int | None = None
        self.cpu_seconds: float | None = None
        self._cpu_by_identity: dict[tuple[int, float], float] = {}
        self.started = time.monotonic()
        self.allocation = AllocationSampler(devices, environment=environment)

    def sample(self) -> None:
        self.allocation.sample()
        try:
            parent = psutil.Process(self.pid)
            processes = [parent, *parent.children(recursive=True)]
        except psutil.Error:
            return
        rss = 0
        observed = False
        for process in processes:
            try:
                identity = (process.pid, process.create_time())
                rss += process.memory_info().rss
                cpu = process.cpu_times()
                self._cpu_by_identity[identity] = cpu.user + cpu.system
                observed = True
            except psutil.Error:
                pass
        if observed:
            self.samples += 1
            self.peak_rss_bytes = max(self.peak_rss_bytes or 0, rss)
            self.cpu_seconds = sum(self._cpu_by_identity.values())

    def summary(self) -> dict:
        return {
            "wall_seconds": time.monotonic() - self.started,
            "samples": self.samples,
            "sample_interval_seconds": self.interval_seconds,
            "sampled_peak_rss_bytes": self.peak_rss_bytes,
            "sampled_cpu_seconds": self.cpu_seconds,
            "allocation": self.allocation.summary(),
            "limitations": "Sampled process-tree RSS sums shared pages and can miss short-lived children/peaks and detached processes. Separate allocation telemetry has wider scope; do not equate RSS with cgroup memory or GPU VRAM.",
        }
