"""Lifetime ownership, not a heartbeat lease; network failover is readiness-gated."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import time
import uuid
from pathlib import Path


class LeadershipError(RuntimeError):
    pass


def atomic_record(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def read_record(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def process_identity(pid: int) -> dict:
    text = Path(f"/proc/{pid}/stat").read_text()
    # comm may contain spaces or parentheses. Fields after final ')' begin at3.
    fields = text.rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def same_process(identity: dict) -> bool:
    try:
        return process_identity(identity["pid"]) == identity
    except (FileNotFoundError, ProcessLookupError):
        return False


def verify_readiness(receipt_path: str | None, state_dir: Path) -> dict:
    if not receipt_path:
        raise LeadershipError("Automatic standby disabled: no cross-node storage/fencing readiness receipt")
    path = Path(receipt_path)
    raw = path.read_bytes()
    receipt = json.loads(raw)
    required = ["cross_node_exclusion", "lock_owner_loss", "exact_slurm_fencing", "worker_connectivity"]
    if receipt.get("schema") != "hhq-failover-readiness/1" or any(
        receipt.get(k) is not True for k in required
    ):
        raise LeadershipError(
            "Standby readiness requires completed real cross-node exclusion, owner-loss, fencing and connectivity tests"
        )
    if len(set(receipt.get("tested_nodes", []))) < 2:
        raise LeadershipError("Standby readiness must identify two distinct tested nodes")
    if not state_dir.samefile(receipt["state_dir"]):
        raise LeadershipError("Standby receipt describes another state directory")
    if float(receipt.get("expires_at", 0)) <= time.time():
        raise LeadershipError("Standby readiness evidence has expired")
    for evidence in receipt.get("evidence", []):
        if hashlib.sha256(Path(evidence["path"]).read_bytes()).hexdigest() != evidence["sha256"]:
            raise LeadershipError("Standby readiness evidence changed")
    if not receipt.get("evidence"):
        raise LeadershipError("Standby readiness has no evidence bindings")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "receipt": receipt}


class LeadershipGuard:
    """Only the owner holding this open file description may own HQ/controller.

    The lock file is never deleted or replaced. Children inherit the descriptor
    so a surviving child retains exclusion if its supervisor dies unexpectedly.
    """

    def __init__(self, service_dir: str | Path, *, owner_id: str | None = None):
        self.directory = Path(service_dir)
        self.owner_id = owner_id or uuid.uuid4().hex
        self.fd = None
        self.record = None

    def acquire(self, *, takeover_proof: dict | None = None):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock = self.directory / "leadership.lock"
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise LeadershipError("Another control process or owned child still holds leadership") from exc
        self.fd = fd
        try:
            previous = read_record(self.directory / "leader.json")
            if (
                previous
                and previous.get("state") != "STOPPED"
                and (
                    not takeover_proof
                    or takeover_proof.get("previous_owner") != previous.get("owner_id")
                    or takeover_proof.get("previous_epoch") != previous.get("epoch")
                    or takeover_proof.get("quiescent") is not True
                )
            ):
                raise LeadershipError(
                    "Previous generation needs exact fencing/quiescence proof; heartbeat age is insufficient"
                )
            epoch_record = read_record(self.directory / "epoch.json", {"epoch": 0})
            epoch = max(epoch_record["epoch"], (previous or {}).get("epoch", 0)) + 1
            atomic_record(self.directory / "epoch.json", {"epoch": epoch})
            self.record = {
                "schema": "hhq-leader/1",
                "owner_id": self.owner_id,
                "epoch": epoch,
                "host": socket.gethostname(),
                "process": process_identity(os.getpid()),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "state": "STARTING",
                "heartbeat_at": time.time(),
                "lock_device": os.fstat(fd).st_dev,
                "lock_inode": os.fstat(fd).st_ino,
            }
            atomic_record(self.directory / "leader.json", self.record)
            return self
        except BaseException:
            os.close(fd)
            self.fd = None
            raise

    @property
    def epoch(self):
        return self.record["epoch"]

    def assert_owned(self):
        if self.fd is None or self.record is None:
            raise LeadershipError("No leadership descriptor")
        current_stat = (self.directory / "leadership.lock").stat()
        held = os.fstat(self.fd)
        if (held.st_dev, held.st_ino) != (current_stat.st_dev, current_stat.st_ino):
            raise LeadershipError("Leadership lock inode changed; fail closed")
        current = read_record(self.directory / "leader.json")
        if not current or (current.get("epoch"), current.get("owner_id")) != (self.epoch, self.owner_id):
            raise LeadershipError("Leadership generation changed; fail closed")

    def heartbeat(self, state="ACTIVE", **details):
        self.assert_owned()
        self.record.update(state=state, heartbeat_at=time.time(), **details)
        atomic_record(self.directory / "leader.json", self.record)

    def close(self, *, children_quiescent=False):
        if self.fd is not None:
            if children_quiescent:
                self.heartbeat("STOPPED")
            # Never LOCK_UN here: inherited child descriptions must continue
            # holding exclusion until the children really exit.
            os.close(self.fd)
            self.fd = None


def assert_epoch(service_dir: str | Path, epoch: int):
    """Controller subprocess admission check, paired with inherited lifetime FD."""
    directory = Path(service_dir)
    record = read_record(directory / "leader.json")
    descriptor = os.environ.get("HHQ_LEADER_FD")
    if not descriptor:
        raise LeadershipError("Controller has no inherited leadership descriptor")
    held = os.fstat(int(descriptor))
    on_disk = (directory / "leadership.lock").stat()
    if (held.st_dev, held.st_ino) != (on_disk.st_dev, on_disk.st_ino):
        raise LeadershipError("Controller ownership descriptor no longer names the lock")
    if not record or record.get("epoch") != epoch or record.get("state") != "ACTIVE":
        raise LeadershipError("Controller generation is not active")
    return record
