"""Cheap process-tree measurements; no all-node GPU sampling or peak guarantees."""

from __future__ import annotations

import os
import subprocess
import time

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
        "gpu_telemetry": "not sampled; GPU visibility is inherited unchanged",
    }


class ProcessSampler:
    def __init__(self, pid: int, interval_seconds: float = 0.2):
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.samples = 0
        self.peak_rss_bytes: int | None = None
        self.cpu_seconds: float | None = None
        self._cpu_by_identity: dict[tuple[int, float], float] = {}
        self.started = time.monotonic()

    def sample(self) -> None:
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
            "limitations": "Sampled process-tree RSS sums shared pages and can miss short-lived children/peaks; detached processes and GPU VRAM are not measured. Not Slurm cgroup peak memory.",
        }
