"""Small, versioned boundary around upstream HyperQueue; no import in direct mode."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

HQ_VERSION = "0.26.2"


class HQError(RuntimeError):
    pass


class HQSubmissionUnknown(HQError):
    """The server may have accepted the job: reconcile its name before retrying."""


def resource_label(kind: str, value: str) -> str:
    kind = kind.replace("_", "")
    return f"hhq/{kind}/{hashlib.sha256(value.encode()).hexdigest()[:20]}"


def locate_binary(explicit: str | None = None) -> str:
    candidate = explicit or os.environ.get("HHQ_HQ_BINARY") or shutil.which("hq")
    if not candidate:
        default = Path.home() / f".local/share/hyper-hyperqueue/hq/{HQ_VERSION}/hq"
        if default.is_file():
            candidate = str(default)
    if not candidate:
        raise HQError(
            f"HQ {HQ_VERSION} is not installed. Direct mode remains available; see docs/decisions/001-hyperqueue-boundary.md"
        )
    return str(Path(candidate).expanduser().resolve())


def check_version(binary: str, *, python: bool = False) -> dict:
    result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10, check=True)
    if result.stdout.strip() != f"hyperqueue v{HQ_VERSION}":
        raise HQError(f"Expected HQ {HQ_VERSION}; got {result.stdout.strip()!r}")
    info = {"binary": binary, "version": HQ_VERSION}
    if python:
        try:
            actual = importlib.metadata.version("hyperqueue")
        except importlib.metadata.PackageNotFoundError as exc:
            raise HQError("Install the optional hq extra: uv sync --extra hq") from exc
        if actual != HQ_VERSION:
            raise HQError(f"HQ Python package {actual} differs from pinned {HQ_VERSION}")
        info["python_version"] = actual
    return info


def task_resources(task, *, default_runtime_seconds: float = 300) -> list[dict]:
    r = task.resources
    if r.nodes != 1 or r.ranks != 1:
        raise ValueError("Pooled tasks currently require one node and one rank; use direct mode")
    if r.gpus not in (0, 1):
        raise ValueError("Pooled mode currently supports zero or one GPU per task")
    base = {"hhq/memory": r.memory_mib}
    if r.gpus:
        base["hhq/gpu"] = 1
        if r.min_vram_mib:
            base["hhq/vram"] = r.min_vram_mib
        if r.gpu_type:
            base[resource_label("gpu_type", r.gpu_type)] = 1
    else:
        base["hhq/cpu"] = 1
    variants = []
    for profile in task.allowed_profiles or [None]:
        for model in r.gpu_models or [None]:
            resources = dict(base)
            if profile:
                resources[resource_label("profile", profile)] = 1
            if model:
                resources[resource_label("model", model)] = 1
            variants.append(
                {
                    "cpus": r.cpus,
                    "resources": resources,
                    "min_time_seconds": r.runtime_seconds or default_runtime_seconds,
                }
            )
    return variants


class HQBackend:
    def __init__(self, server_dir: str | Path, binary: str | None = None, *, timeout: float = 30,
                 log_dir: str | Path | None = None):
        self.server_dir = Path(server_dir)
        self.binary = locate_binary(binary)
        self.timeout = timeout
        self.log_dir = Path(log_dir) if log_dir is not None else self.server_dir.parent / "task_logs"
        check_version(self.binary)

    def json(self, *args: str):
        p = subprocess.run(
            [self.binary, "--server-dir", str(self.server_dir), "--output-mode=json", *args],
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if p.returncode:
            raise HQError(f"HQ {' '.join(args[:2])}: {p.stderr.strip() or p.stdout.strip()}")
        try:
            return json.loads(p.stdout) if p.stdout.strip() else None
        except json.JSONDecodeError as exc:
            raise HQError("HQ JSON schema/output mismatch; expected pinned 0.26.2") from exc

    def jobs(self):
        return self.json("job", "list", "--all")

    def workers(self):
        return self.json("worker", "list", "--all")

    def tasks(self, job_id: int):
        return self.json("task", "list", str(int(job_id)))

    def job_info(self, job_id: int):
        return self.json("job", "info", str(int(job_id)))

    def explain(self, job_id: int, task_id: int):
        return self.json("task", "explain", str(int(job_id)), str(int(task_id)))

    def find_intent(self, intent_token: str):
        name = self.intent_name(intent_token)
        matches = [j for j in self.jobs() if j.get("name") == name]
        if len(matches) > 1:
            raise HQSubmissionUnknown(f"More than one HQ job carries intent {intent_token}; stop admission")
        return matches[0] if matches else None

    @staticmethod
    def intent_name(intent_token: str):
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", intent_token):
            raise ValueError("Intent token must be a short safe identifier")
        return "hhq:" + intent_token

    def submit(
        self,
        tasks: Iterable,
        intent_token: str,
        runner_argv_factory: Callable,
        *,
        default_runtime_seconds: float = 300,
        epoch: int | None = None,
    ) -> dict:
        """Caller must persist intent before calling, and must own mutation admission.

        Submission uses upstream Python API in a bounded subprocess. A timeout is
        unknown, never an automatic retry. Reconciliation uses exact job name.
        """
        check_version(self.binary, python=True)
        tasks = list(tasks)
        if not tasks:
            raise ValueError("Cannot submit an empty HQ job")
        existing = self.find_intent(intent_token)
        if existing:
            raise HQSubmissionUnknown(
                "Intent already exists; adopt its mappings rather than submitting again"
            )
        payload = {"server_dir": str(self.server_dir), "name": self.intent_name(intent_token), "tasks": []}
        logs = self.log_dir / intent_token
        logs.mkdir(parents=True, exist_ok=True, mode=0o700)
        for index, t in enumerate(tasks):
            env = {"HHQ_CAMPAIGN_ID": t.campaign_id, "HHQ_TASK_ID": t.task_id}
            if epoch is not None:
                env["HHQ_SERVER_EPOCH"] = str(epoch)
            payload["tasks"].append(
                {
                    "task_id": t.task_id,
                    "argv": list(runner_argv_factory(t)),
                    "cwd": t.cwd,
                    "env": env,
                    "priority": t.priority,
                    "stdout": str(logs / f"{index}.stdout"),
                    "stderr": str(logs / f"{index}.stderr"),
                    "resources": task_resources(t, default_runtime_seconds=default_runtime_seconds),
                }
            )
        try:
            p = subprocess.run(
                [sys.executable, "-m", "hyper_hyperqueue.backends.hyperqueue", "submit"],
                input=json.dumps(payload),
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise HQSubmissionUnknown(
                f"HQ submission acknowledgement unavailable for {intent_token}"
            ) from exc
        if p.returncode:
            raise HQSubmissionUnknown(f"HQ submission unresolved for {intent_token}: {p.stderr[-1500:]}")
        try:
            result = json.loads(p.stdout)
        except ValueError as exc:
            raise HQSubmissionUnknown(f"Invalid HQ acknowledgement for {intent_token}") from exc
        return {
            "intent_token": intent_token,
            "job_id": result["job_id"],
            "tasks": {t.task_id: i for i, t in enumerate(tasks)},
        }

    def cancel_owned(self, job_id: int, intent_token: str):
        found = self.find_intent(intent_token)
        if not found or int(found["id"]) != int(job_id):
            raise HQError("Refusing to cancel an HQ job without exact intent ownership")
        return self.json("job", "cancel", str(int(job_id)))


def _submit(payload):
    # Deliberately lazy: importing this adapter must not import HyperQueue.
    from hyperqueue import Client, Job
    from hyperqueue.ffi.protocol import ResourceRequest

    if importlib.metadata.version("hyperqueue") != HQ_VERSION:
        raise HQError("HQ Python package version mismatch")
    job = Job(name=payload["name"], max_fails=None)
    for task in payload["tasks"]:
        requests = [
            ResourceRequest(
                cpus=r["cpus"], resources=r["resources"], min_time=dt.timedelta(seconds=r["min_time_seconds"])
            )
            for r in task["resources"]
        ]
        job.program(
            task["argv"],
            name=task["task_id"],
            cwd=task["cwd"],
            env=task["env"],
            resources=requests,
            priority=task["priority"],
            crash_limit=0,
            stdout=task.get("stdout"),
            stderr=task.get("stderr"),
        )
    submitted = Client(server_dir=payload["server_dir"]).submit(job)
    return {"job_id": submitted.id}


if __name__ == "__main__":
    if sys.argv[1:] != ["submit"]:
        raise SystemExit("Internal API: submit JSON on stdin")
    print(json.dumps(_submit(json.load(sys.stdin))))
