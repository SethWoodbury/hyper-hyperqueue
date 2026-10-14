"""Snapshot command imports; repeated occurrences are intentional independent tasks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .models import OutputPolicy, Resources, RetryPolicy, TaskSpec
from .state import StateStore, atomic_bytes, fingerprint, identifier, immutable_json, utc_now


def import_campaign_id(import_key: str) -> str:
    """Stable initial-import identity shared by callers and ingestion."""
    return "c_" + hashlib.sha256(import_key.encode()).hexdigest()[:24]


def parse_bytes(data: bytes, *, literal: bool = False, format: str = "lines") -> list[dict]:
    text = data.decode("utf-8")
    records = []
    if format not in {"lines", "jsonl"}:
        raise ValueError("Format must be lines or jsonl")
    if not text:
        return []
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    for line_number, line in enumerate(lines, 1):
        # Treat CRLF as a line terminator, retaining other literal control characters.
        line = line.removesuffix("\r")
        if not literal and (not line.strip() or line.lstrip().startswith("#")):
            continue
        if format == "jsonl":
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) - {"command", "argv"}:
                raise ValueError(f"JSONL line {line_number}: only command or argv is supported")
            if (value.get("command") is None) == (value.get("argv") is None):
                raise ValueError(f"JSONL line {line_number}: supply exactly command or argv")
        else:
            # An odd trailing backslash escapes the newline even inside double quotes.
            trailing = len(line) - len(line.rstrip("\\"))
            if trailing % 2:
                raise ValueError(
                    f"Line {line_number} ends in a shell continuation. Use JSONL with one escaped multiline command or put the whole command on one line."
                )
            value = {"command": line}
        records.append({**value, "source_line": line_number, "source_index": len(records)})
    return records


def parse_file(path: str | Path, *, literal: bool = False, format: str = "lines") -> list[dict]:
    return parse_bytes(Path(path).read_bytes(), literal=literal, format=format)


def ingest_file(
    store: StateStore,
    path: str | Path,
    *,
    mode: str,
    resources: Resources | dict | None = None,
    profile_names: list[str] | None = None,
    cwd: str | Path | None = None,
    name: str | None = None,
    environment: dict[str, str] | None = None,
    retry: RetryPolicy | dict | None = None,
    output: OutputPolicy | dict | None = None,
    campaign_id: str | None = None,
    literal: bool = False,
    import_key: str | None = None,
    format: str = "lines",
    shell: str = "/bin/bash",
    shell_options: list[str] | None = None,
    workload: str = "generic",
    priority: int = 0,
    input_fingerprint: str | None = None,
    software_fingerprint: str | None = None,
) -> dict:
    if mode not in {"direct", "pooled"}:
        raise ValueError("Resolve auto mode before importing immutable tasks")
    source = Path(path).expanduser().absolute()
    data = source.read_bytes()
    records = parse_bytes(data, literal=literal, format=format)
    if not records:
        raise ValueError("No commands found")
    directory = str(Path(cwd or Path.cwd()).expanduser().resolve())
    if not Path(directory).is_dir():
        raise ValueError(f"Working directory does not exist: {directory}")
    resources = Resources.model_validate(resources or {})
    retry = RetryPolicy.model_validate(retry or {})
    output = OutputPolicy.model_validate(output or {})
    settings = {
        "mode": mode,
        "resources": resources.model_dump(),
        "profile_names": profile_names or [],
        "cwd": directory,
        "environment": environment or {},
        "retry": retry.model_dump(),
        "output": output.model_dump(),
        "literal": literal,
        "format": format,
        "shell": shell,
        "shell_options": shell_options or [],
        "workload": workload,
        "priority": priority,
        "input_fingerprint": input_fingerprint,
        "software_fingerprint": software_fingerprint,
    }
    digest = hashlib.sha256(data).hexdigest()
    key = import_key or fingerprint({"sha256": digest, "settings": settings})
    # For initial API retries a stable explicit import key also identifies the campaign.
    campaign_id = identifier(campaign_id or import_campaign_id(key))
    import_id = "i_" + hashlib.sha256((campaign_id + "\0" + key).encode()).hexdigest()[:24]
    with store.lock():
        campaign_path = store.campaign_dir(campaign_id) / "campaign.json"
        old = store.campaign(campaign_id) if campaign_path.exists() else None
        imports = list(old.get("imports", [])) if old else []
        for previous in imports:
            if previous["import_id"] == import_id:
                if previous["sha256"] != digest or previous["settings"] != settings:
                    raise ValueError(
                        "Import key already exists with different command bytes/settings; choose an explicit new append key"
                    )
                return old
        if mode == "pooled":
            # The supervisor commits completion under this same mutation lock.
            # Do not accept new work behind that barrier while its CPU allocation
            # is still closing. API submission reopens after exact terminal proof.
            service = store.root / "service"
            completion_path = service / "completion.json"
            deployment_path = service / "deployment.json"
            if completion_path.exists() and deployment_path.exists():
                completion = json.loads(completion_path.read_text())
                deployment = json.loads(deployment_path.read_text())
                if (completion.get("deployment_id") == deployment.get("deployment_id")
                        and deployment.get("desired_state") == "STOPPED"
                        and completion.get("phase") != "stopped"):
                    raise ValueError(
                        "Service completion cleanup is still in progress; no new work was imported. "
                        "Wait for its owned allocations to terminate, then repeat submit or append."
                    )
        if old and old["mode"] != mode:
            raise ValueError("Cannot change the backend of an existing campaign")
        tasks = []
        for record in records:
            occurrence = record["source_index"]
            task_id = (
                "t_" + hashlib.sha256(f"{campaign_id}\0{import_id}\0{occurrence}".encode()).hexdigest()[:28]
            )
            command_payload = {k: record[k] for k in ("command", "argv") if k in record}
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    campaign_id=campaign_id,
                    **command_payload,
                    shell=shell,
                    shell_options=shell_options or [],
                    cwd=directory,
                    environment=environment or {},
                    source_index=occurrence,
                    source_line=record["source_line"],
                    import_id=import_id,
                    command_digest=fingerprint(command_payload),
                    resources=resources,
                    allowed_profiles=profile_names or [],
                    retry=retry,
                    output=output,
                    backend=mode,
                    workload=workload,
                    priority=priority,
                    input_fingerprint=input_fingerprint,
                    software_fingerprint=software_fingerprint,
                )
            )
        snapshot = store.campaign_dir(campaign_id) / "imports" / import_id / "source"
        if snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
            raise ValueError("Immutable command snapshot differs")
        if not snapshot.exists():
            atomic_bytes(snapshot, data)
        manifest = {
            "schema_version": 1,
            "import_id": import_id,
            "import_key": key,
            "sha256": digest,
            "source_path": str(source),
            "snapshot": str(snapshot),
            "settings": settings,
            "task_ids": [t.task_id for t in tasks],
        }
        immutable_json(snapshot.parent / "manifest.json", manifest)
        imports.append(manifest)
        metadata = {
            **(old or {}),
            "schema_version": 1,
            "campaign_id": campaign_id,
            "name": (old or {}).get("name", name or source.stem),
            "mode": mode,
            "cwd": directory,
            "created_at": (old or {}).get("created_at", utc_now()),
            "updated_at": utc_now(),
            "status": (old or {}).get("status", "ready"),
            "imports": imports,
            "task_ids": (old or {}).get("task_ids", []) + manifest["task_ids"],
            "resources": resources.model_dump(),
            "profile_names": profile_names or [],
        }
        return store.create_campaign(metadata, tasks)
