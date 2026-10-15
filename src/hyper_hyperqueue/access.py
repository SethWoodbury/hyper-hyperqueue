"""Read-only access hints from cached records; never start services or query Slurm."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from .dashboard import read_json_object


def _record(path, warnings):
    try:
        return read_json_object(path, maximum=1024 * 1024)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError):
        warnings.append(f"{path.name}: cached record is unavailable or incomplete")
        return {}


def access_info(config, *, config_path=None, python=None):
    """Describe access without writes, scheduler/process probes, or secret fields.

    Pass the config file used by the CLI when supplied. The resolved state path
    is always explicit, including when it overrides that file. Python's venv path
    is retained rather than resolving its executable symlink to system Python.
    """
    state = Path(config.state_dir).expanduser().absolute()
    config_path = config_path or os.environ.get("HHQ_CONFIG")
    if config_path is None:
        default = Path.home() / ".config/hyper-hyperqueue/config.yaml"
        config_path = default if default.is_file() else None
    argv = [str(Path(python or sys.executable).expanduser().absolute()), "-m", "hyper_hyperqueue.cli"]
    if config_path is not None:
        config_path = str(Path(config_path).expanduser().absolute())
        argv += ["--config", config_path]
    argv += ["--state-dir", str(state)]
    commands = {name: shlex.join([*argv, *tail]) for name, tail in {
        "service_status": ["service", "status"], "ui": ["ui"], "access": ["access"],
    }.items()}

    warnings = []
    service = state / "service"
    leader = _record(service / "leader.json", warnings)
    endpoint = _record(service / "endpoint.json", warnings)
    deployment = _record(service / "deployment.json", warnings)
    allocations = deployment.get("allocations", [])
    allocations = allocations if isinstance(allocations, list) else []
    active = [item for item in allocations if isinstance(item, dict) and item.get("role") in ("active", "renewal")]
    latest = active[-1] if active else {}
    controller = {"status": "UNKNOWN", "node": None, "job_id": None, "heartbeat_at": None}
    newer_submission = (leader.get("state") == "STOPPED" and latest.get("job_id")
                        and latest.get("job_id") != leader.get("slurm_job_id"))
    if leader and not newer_submission:
        controller.update(status=leader.get("state", "UNKNOWN"), node=leader.get("host"),
                          job_id=leader.get("slurm_job_id"), heartbeat_at=leader.get("heartbeat_at"))
    elif latest:
        controller.update(status="STARTING_OR_PENDING" if latest.get("status") in ("accepted", "submitting")
                          else "UNKNOWN", job_id=latest.get("job_id"))
    matching_active = (leader.get("state") == endpoint.get("status") == "ACTIVE"
                       and leader.get("epoch") is not None and endpoint.get("epoch") == leader["epoch"]
                       and leader.get("owner_id") is not None and endpoint.get("owner_id") == leader["owner_id"])
    if controller["status"] == "ACTIVE" and not matching_active:
        controller["status"] = "UNCONFIRMED"
    if newer_submission or (leader and endpoint.get("epoch") != leader.get("epoch")):
        endpoint = {}  # Never advertise another controller generation's endpoint.
    if controller["status"] == "STOPPED":
        endpoint = {"status": "STOPPED"}
    elif deployment.get("desired_state") == "STOPPED":
        controller["status"] = "STOP_REQUESTED"
        endpoint = {"status": "STOP_REQUESTED"}
    elif endpoint.get("status") == "ACTIVE" and not matching_active:
        endpoint = {"status": "UNCONFIRMED"}
    server = endpoint.get("server", {})
    server = server if isinstance(server, dict) else {}
    hq = {"status": endpoint.get("status", "UNKNOWN"), "record_path": str(service / "endpoint.json"),
          "server_directory": endpoint.get("server_dir"),
          "host": server.get("host") or server.get("hostname"),
          "port": server.get("port"),
          "purpose": "Internal HyperQueue connection; not an HTML dashboard."}
    return {"schema": "hhq-access/1", "state_dir": str(state), "config_path": config_path,
            "controller": controller, "hyperqueue": hq, "commands": commands,
            "dashboard": "Not started or probed by this command. Run ui for its authenticated browser URL.",
            "evidence": "Cached records only; timestamps may be stale. No scheduler or liveness checks performed.",
            "warnings": warnings}


def access_lines(info):
    """Compact human text; launch commands write these lines to stderr."""
    controller = info["controller"]
    node = controller["node"] or "unknown (not published yet)"
    job = controller["job_id"] or "unknown / no Slurm controller recorded"
    return [f"HHQ state: {info['state_dir']}",
            f"Controller (cached): {controller['status']}; job {job}; node {node}",
            f"HQ connection (not HTML): {info['hyperqueue']['status']}; {info['hyperqueue']['record_path']}",
            "HTML dashboard: start explicitly on the host where you run the ui command.",
            f"  {info['commands']['ui']}",
            f"Service status: {info['commands']['service_status']}",
            info['evidence'], *[f"Warning: {warning}" for warning in info["warnings"]]]
