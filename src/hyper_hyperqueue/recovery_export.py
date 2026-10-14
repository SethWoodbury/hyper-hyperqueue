"""Detached, explicit transfer of a quiescent campaign's unfinished commands.

This is not retry authority or a scheduler. Successful occurrences stay in the
old campaign; exported occurrences are blocked there before a new import exists.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

from .state import StateStore, atomic_bytes, atomic_json, file_lock, fingerprint, read_json

SCHEMA = "hhq-remaining-export/1"
REASON = "exported_to_new_generation"
ACTIVE = {"intent", "unknown", "submitted", "running"}
SETTINGS = (
    "cwd",
    "environment",
    "shell",
    "shell_options",
    "output",
    "resources",
    "workload",
    "priority",
    "input_fingerprint",
    "software_fingerprint",
    "retry",
)


def _quiescent(store, campaign_id):
    campaign = store.campaign(campaign_id)
    if not (campaign.get("pause_import") and campaign.get("pause_dispatch")):
        raise ValueError("Pause both imports and dispatch for the source campaign before exporting")
    for intent in store.intents():
        if intent.get("campaign_id") == campaign_id and intent.get("state") in ACTIVE:
            raise ValueError("Source has active or uncertain submission ownership; reconcile first")
    return campaign


def export_remaining(config, campaign_id: str, destination: str | Path) -> dict:
    """Prepare exact remaining commands, retiring their old occurrences; submit nothing.

    Unknown/canceled attempts are deliberately not converted to new work. This
    narrow interface supports one homogeneous execution envelope per campaign.
    """
    store = StateStore(config.state_dir)
    destination = Path(destination).expanduser().resolve()
    manifest_path = destination / "manifest.json"
    with store.lock(), contextlib.ExitStack() as locks:
        campaign = _quiescent(store, campaign_id)
        if manifest_path.exists():
            existing = _load_export(manifest_path, store=store)
            if existing["campaign_id"] != campaign_id:
                raise ValueError("Export destination belongs to another campaign")
            return existing
        if destination.exists() and {p.name for p in destination.iterdir()} - {"commands.jsonl"}:
            raise ValueError("Export destination contains unrelated files; choose a new directory")
        rows, excluded, settings = [], [], None
        for tid in campaign["task_ids"]:
            task = store.task(campaign_id, tid)
            locks.enter_context(
                file_lock(store.campaign_dir(campaign_id) / "task_locks" / f"{tid}.lock", blocking=False)
            )
            current = store.task_state(campaign_id, tid)
            state = current.get("state", "unknown")
            if state == "succeeded":
                if "completion" not in current:
                    raise ValueError(f"Success lacks a validated completion: {tid}")
                excluded.append({"task_id": tid, "state": state})
                continue
            if state == "canceled":
                excluded.append({"task_id": tid, "state": state})
                continue
            if state == "blocked" and current.get("reason") == REASON:
                if current.get("recovery_export") != str(manifest_path):
                    raise ValueError(f"Task already belongs to a different recovery export: {tid}")
                state = current["exported_source_state"]
            if state not in {"ready", "failed", "interrupted"}:
                raise ValueError(f"Cannot export {tid} in state {state}; reconcile before exporting")
            attempts = store.attempts(campaign_id, tid)
            if attempts:
                result = attempts[-1].get("result", {})
                if state not in {"failed", "interrupted"} or result.get("status") not in {
                    "failed",
                    "interrupted",
                }:
                    raise ValueError(f"Attempt lacks a final failure record: {tid}")
                if not (task.output.repeatable or task.output.mode == "isolated"):
                    raise ValueError(f"Failed task output policy does not authorize replay: {tid}")
            if task.output.collision_key:
                raise ValueError(f"Output collision ownership needs manual transfer: {tid}")
            spec = task.model_dump(mode="json")
            envelope = {key: spec[key] for key in SETTINGS}
            if settings is not None and envelope != settings:
                raise ValueError(
                    "Recovery export requires homogeneous execution settings; split the campaign manually"
                )
            settings = envelope
            rows.append(
                {"task_id": tid, "source_state": state, "task_fingerprint": fingerprint(spec), "task": spec}
            )

        commands = b"".join(
            (
                json.dumps(
                    {key: row["task"][key] for key in ("command", "argv") if row["task"][key] is not None}
                )
                + "\n"
            ).encode()
            for row in rows
        )
        document = {
            "schema": SCHEMA,
            "source_state_dir": str(store.root),
            "campaign_id": campaign_id,
            "commands_file": str(destination / "commands.jsonl"),
            "commands_sha256": hashlib.sha256(commands).hexdigest(),
            "execution_settings": settings,
            "tasks": rows,
            "excluded": excluded,
            "count": len(rows),
            "submitted": False,
        }
        # Retire first: a crash can leave resumable blocked occurrences, never
        # published replacement commands plus runnable old occurrences.
        for row in rows:
            store.update_task_state(
                campaign_id,
                row["task_id"],
                {
                    "state": "blocked",
                    "reason": REASON,
                    "recovery_export": str(manifest_path),
                    "exported_source_state": row["source_state"],
                    "exported_task_fingerprint": row["task_fingerprint"],
                },
            )
        atomic_bytes(destination / "commands.jsonl", commands)
        atomic_json(manifest_path, document)
        store.event(
            "remaining_exported", campaign_id=campaign_id, count=len(rows), manifest=str(manifest_path)
        )
        return document


def _load_export(path: Path, *, store=None) -> dict:
    document = read_json(path)
    if document.get("schema") != SCHEMA:
        raise ValueError("Unsupported recovery export manifest")
    store = store or StateStore(document["source_state_dir"])
    if str(store.root) != document["source_state_dir"]:
        raise ValueError("Recovery source state directory differs")
    _quiescent(store, document["campaign_id"])
    commands = Path(document["commands_file"])
    if (
        commands.parent != path.parent
        or hashlib.sha256(commands.read_bytes()).hexdigest() != document["commands_sha256"]
    ):
        raise ValueError("Recovery command bytes/path differ from the export")
    expected_commands = []
    if len({row["task_id"] for row in document["tasks"]}) != len(document["tasks"]):
        raise ValueError("Recovery export repeats a source occurrence")
    retired = {
        tid
        for tid in store.task_ids(document["campaign_id"])
        if store.task_state(document["campaign_id"], tid).get("recovery_export") == str(path)
    }
    if retired != {row["task_id"] for row in document["tasks"]}:
        raise ValueError("Recovery export does not contain exactly its retired source occurrences")
    for row in document["tasks"]:
        task = store.task(document["campaign_id"], row["task_id"])
        spec = task.model_dump(mode="json")
        current = store.task_state(document["campaign_id"], row["task_id"])
        if (
            fingerprint(spec) != row["task_fingerprint"]
            or spec != row["task"]
            or current.get("state") != "blocked"
            or current.get("reason") != REASON
            or current.get("recovery_export") != str(path)
            or {key: spec[key] for key in SETTINGS} != document["execution_settings"]
        ):
            raise ValueError("Recovery source ownership/settings changed; do not replay")
        expected_commands.append({key: spec[key] for key in ("command", "argv") if spec[key] is not None})
    actual_commands = [json.loads(line) for line in commands.read_text().splitlines()]
    if actual_commands != expected_commands or document["count"] != len(expected_commands):
        raise ValueError("Recovery commands do not match the retired source occurrences")
    return document


def load_export(path: str | Path, *, store: StateStore | None = None) -> dict:
    """Verify detached export before an explicit new import; never start services."""
    path = Path(path).expanduser().resolve()
    document = read_json(path)
    store = store or StateStore(document["source_state_dir"])
    with store.lock():
        return _load_export(path, store=store)
