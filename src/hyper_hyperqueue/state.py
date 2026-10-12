"""Durable JSON metadata. Controller mutations and task execution use separate locks.

Filesystem locking must be qualified across the actual hosts before enabling failover.
No wall-clock timeout, heartbeat age, or missing PID permits deleting a lock.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from .models import TaskSpec


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value) or value in {".", ".."}:
        raise ValueError(f"Unsafe identifier: {value!r}")
    return value


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read_json(path: str | Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_directory(path: str | Path) -> Path:
    """Bounded NFS negative-dentry recovery; never replace a file or unsafe link."""
    path = Path(path)
    for attempt in range(10):
        try:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            return path
        except (FileExistsError, FileNotFoundError):
            # A concurrent mkdir can succeed remotely while is_dir still sees
            # a negative cache entry. Refresh parent listings before retrying.
            for parent in (path.parent, path.parent.parent):
                try:
                    with os.scandir(parent) as entries:
                        for _ in entries:
                            pass
                except FileNotFoundError:
                    continue
            if path.is_dir():
                return path
            if attempt == 9:
                raise
            time.sleep(0.1)
    raise AssertionError("unreachable")


def atomic_bytes(path: str | Path, data: bytes, *, mode: int = 0o600) -> None:
    path = Path(path)
    ensure_directory(path.parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: str | Path, value: object) -> None:
    atomic_bytes(path, json_bytes(value))


def immutable_json(path: str | Path, value: object) -> None:
    """Caller owns the relevant mutation lock. Reject reuse with changed contents."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable metadata differs: {path}")
    else:
        atomic_json(path, value)


@contextlib.contextmanager
def file_lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    ensure_directory(path.parent)
    with path.open("a+b") as handle:
        os.chmod(path, 0o600)
        # Explicit nonblocking acquisition avoids unbounded kernel/NFS grant waits
        # after a contending owner releases. Retrying never breaks another lock.
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not blocking:
                    raise
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class StateStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        checkout = Path(__file__).resolve().parents[2]
        if (checkout / "pyproject.toml").exists() and self.root.is_relative_to(checkout):
            raise ValueError("Runtime state must live outside the source checkout")
        ensure_directory(self.root)
        self._thread_lock = threading.RLock()
        self._depth = 0

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Short mutation lock shared by CLI/controller; never the leadership lock."""
        with self._thread_lock:
            if self._depth:
                self._depth += 1
                try:
                    yield
                finally:
                    self._depth -= 1
            else:
                with file_lock(self.root / "mutation.lock"):
                    self._depth = 1
                    try:
                        yield
                    finally:
                        self._depth = 0

    def campaign_dir(self, campaign_id: str) -> Path:
        return self.root / "campaigns" / identifier(campaign_id)

    def campaign(self, campaign_id: str) -> dict:
        return read_json(self.campaign_dir(campaign_id) / "campaign.json")

    def campaigns(self) -> list[dict]:
        return [read_json(p) for p in sorted((self.root / "campaigns").glob("*/campaign.json"))]

    def save_campaign(self, metadata: dict) -> None:
        with self.lock():
            path = self.campaign_dir(metadata["campaign_id"]) / "campaign.json"
            if path.exists():
                old = read_json(path)
                if old["campaign_id"] != metadata["campaign_id"]:
                    raise ValueError("Campaign identity is immutable")
            atomic_json(path, metadata)
            self.event("campaign_saved", campaign_id=metadata["campaign_id"], metadata=metadata)

    def create_campaign(self, metadata: dict, tasks: list[TaskSpec | dict]) -> dict:
        campaign_id = identifier(metadata["campaign_id"])
        validated = [t if isinstance(t, TaskSpec) else TaskSpec.model_validate(t) for t in tasks]
        if len({t.task_id for t in validated}) != len(validated):
            raise ValueError("Duplicate task IDs")
        if any(t.campaign_id != campaign_id for t in validated):
            raise ValueError("Task belongs to a different campaign")
        with self.lock():
            for task in validated:
                self.add_task(task)
            self.save_campaign(metadata)
        return metadata

    def add_task(self, task: TaskSpec) -> None:
        with self.lock():
            directory = self.campaign_dir(task.campaign_id)
            # Stable shared parents are created by the single import writer,
            # before direct array elements or remote HQ workers start together.
            for relative in (
                "tasks",
                "task_states",
                "task_locks",
                "completions",
                f"attempts/{identifier(task.task_id)}",
            ):
                ensure_directory(directory / relative)
            for relative in ("output_locks", "output_claims"):
                ensure_directory(self.root / relative)
            immutable_json(
                directory / "tasks" / f"{identifier(task.task_id)}.json", task.model_dump(mode="json")
            )
            state = directory / "task_states" / f"{task.task_id}.json"
            if not state.exists():
                atomic_json(
                    state,
                    {
                        "status": "ready",
                        "state": "ready",
                        "task_id": task.task_id,
                        "campaign_id": task.campaign_id,
                    },
                )

    def task(self, campaign_id: str, task_id: str) -> TaskSpec:
        return TaskSpec.model_validate(
            read_json(self.campaign_dir(campaign_id) / "tasks" / f"{identifier(task_id)}.json")
        )

    def task_ids(self, campaign_id: str) -> list[str]:
        return list(self.campaign(campaign_id)["task_ids"])

    def attempts(self, campaign_id: str, task_id: str) -> list[dict]:
        directory = self.campaign_dir(campaign_id) / "attempts" / identifier(task_id)
        records = []
        for path in sorted(directory.glob("*/attempt.json")):
            item = read_json(path)
            result = path.parent / "result.json"
            if result.exists():
                item["result"] = read_json(result)
            records.append(item)
        return records

    def task_state(self, campaign_id: str, task_id: str) -> dict:
        directory = self.campaign_dir(campaign_id)
        state = read_json(directory / "task_states" / f"{identifier(task_id)}.json")
        completion = directory / "completions" / f"{task_id}.json"
        if completion.exists():
            record = read_json(completion)
            if record["task_fingerprint"] != fingerprint(
                self.task(campaign_id, task_id).model_dump(mode="json")
            ):
                raise ValueError("Completion/task identity mismatch")
            return {**state, "status": "succeeded", "state": "succeeded", "completion": record}
        if state.get("state", state.get("status")) in {"canceled", "blocked"}:
            explicit = state.get("state", state.get("status"))
            return {**state, "status": explicit, "state": explicit}
        attempts = self.attempts(campaign_id, task_id)
        if attempts:
            latest = attempts[-1]
            if "result" in latest:
                result = latest["result"]
                # Explicit coordinator decisions after this attempt take precedence.
                if state.get("after_attempt_id") == latest["attempt_id"]:
                    return {**state, "latest_attempt": latest}
                return {
                    **state,
                    "status": result["status"],
                    "state": result["status"],
                    "reason": result["reason"],
                    "latest_attempt": latest,
                }
            try:
                with file_lock(directory / "task_locks" / f"{task_id}.lock", blocking=False):
                    active = False
            except BlockingIOError:
                active = True
            if not active and state.get("after_attempt_id") == latest["attempt_id"]:
                return {**state, "latest_attempt": latest}
            status = "running" if active else "unknown"
            return {
                **state,
                "status": status,
                "state": status,
                "reason": "active_attempt" if active else "attempt_has_no_final_record",
                "latest_attempt": latest,
            }
        return state

    def update_task_state(self, campaign_id: str, task_id: str, update: dict) -> dict:
        with self.lock():
            self.task(campaign_id, task_id)
            path = self.campaign_dir(campaign_id) / "task_states" / f"{identifier(task_id)}.json"
            update = dict(update)
            if "state" in update or "status" in update:
                if "state" in update and "status" in update and update["state"] != update["status"]:
                    raise ValueError("Conflicting state/status values")
                update["state"] = update["status"] = update.get("state", update.get("status"))
            state = {**read_json(path), **update, "updated_at": utc_now()}
            atomic_json(path, state)
            self.event("task_state", campaign_id=campaign_id, task_id=task_id, update=update)
            return state

    def event(self, kind: str, **fields) -> dict:
        with self.lock():
            snapshot_path = self.root / "snapshot.json"
            snapshot = (
                read_json(snapshot_path)
                if snapshot_path.exists()
                else {"last_sequence": 0, "journal_offset": 0}
            )
            journal = self.root / "events.jsonl"
            # Recover a durable append whose snapshot replace was interrupted.
            with journal.open("a+b") as handle:
                os.fchmod(handle.fileno(), 0o600)
                if os.fstat(handle.fileno()).st_size < snapshot["journal_offset"]:
                    raise ValueError(
                        "Event journal is shorter than committed snapshot; manual recovery required"
                    )
                handle.seek(snapshot["journal_offset"])
                while line := handle.readline():
                    if not line.endswith(b"\n"):
                        handle.seek(-len(line), os.SEEK_CUR)
                        handle.truncate()
                        break
                    entry = json.loads(line)
                    if entry["sequence"] != snapshot["last_sequence"] + 1:
                        raise ValueError("Non-contiguous event journal; manual recovery required")
                    snapshot = {"last_sequence": entry["sequence"], "journal_offset": handle.tell()}
                entry = {
                    **fields,
                    "schema_version": 1,
                    "kind": kind,
                    "at": utc_now(),
                    "sequence": snapshot["last_sequence"] + 1,
                }
                handle.seek(0, os.SEEK_END)
                handle.write(
                    (json.dumps(entry, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode()
                )
                handle.flush()
                os.fsync(handle.fileno())
                atomic_json(
                    snapshot_path, {"last_sequence": entry["sequence"], "journal_offset": handle.tell()}
                )
            return entry

    def put_intent(self, intent: dict) -> dict:
        key = intent.get("intent_id") or intent.get("token")
        if not key:
            raise ValueError("Submission intent requires intent_id or token")
        with self.lock():
            path = self.root / "intents" / f"{identifier(key)}.json"
            if path.exists():
                prior = read_json(path)
                for name in (
                    "campaign_id",
                    "token",
                    "backend",
                    "task_ids",
                    "placement",
                    "epoch",
                    "profile",
                    "config_snapshot",
                ):
                    if name in prior and name in intent and prior[name] != intent[name]:
                        raise ValueError(f"Submission intent identity changed: {name}")
            atomic_json(path, intent)
            self.event("submission_intent", intent=intent)
        return intent

    def intents(self) -> list[dict]:
        return [read_json(p) for p in sorted((self.root / "intents").glob("*.json"))]
