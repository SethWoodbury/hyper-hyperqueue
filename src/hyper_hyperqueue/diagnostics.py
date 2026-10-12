from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def doctor(config):
    checks = {name: shutil.which(name) for name in ("sbatch", "squeue", "sacct", "scancel", "bash")}
    checks["state_parent_exists"] = Path(config.state_dir).parent.exists()
    checks["profiles"] = list(config.profiles)
    try:
        from .backends.hyperqueue import check_version, locate_binary

        checks["hyperqueue"] = check_version(locate_binary(config.control.hq_binary))
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        checks["hyperqueue"] = {"available": False, "reason": str(exc), "direct_mode_affected": False}
    try:
        from .service.leadership import verify_readiness

        checks["standby_readiness"] = verify_readiness(
            config.control.verification_receipt, Path(config.state_dir)
        )
    except (RuntimeError, ValueError, OSError) as exc:
        checks["standby_readiness"] = {"ready": False, "reason": str(exc)}
    free = shutil.disk_usage(
        Path(config.state_dir).parent if Path(config.state_dir).parent.exists() else Path.home()
    )
    return {
        "schema_version": 1,
        "checks": checks,
        "free_bytes": free.free,
        "note": "No jobs submitted. Worker connectivity and cross-node locking require actual allocated probes.",
    }


def support_bundle(config, destination):
    """Allowlist only: no task commands, environments, credentials, or HQ access files."""
    from .api import status

    destination = Path(destination)
    if destination.exists():
        raise ValueError("Refusing to overwrite support bundle")
    with tempfile.TemporaryDirectory(prefix="hhq-support-") as directory:
        root = Path(directory)
        health = status(config)
        # Arbitrary exception strings and campaign names may contain command data.
        safe = {
            "schema_version": 1,
            "campaigns": [
                {"mode": c["mode"], "tasks": c["tasks"], "counts": c["counts"]} for c in health["campaigns"]
            ],
            "controller_state": health["controller"].get("state", "unknown"),
        }
        (root / "summary.json").write_text(json.dumps(safe, indent=2) + "\n")
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(root / "summary.json", arcname="summary.json")
    return {
        "path": str(destination),
        "contents": ["summary.json"],
        "redaction": "Allowlist; no commands, paths, names, environments or HQ authentication files",
    }
