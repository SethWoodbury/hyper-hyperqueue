"""Supervise one journal writer and a separately restartable controller."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import fcntl
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from ..backends.hyperqueue import HQBackend, HQError, check_version, locate_binary
from ..logs import log_directory
from .leadership import (
    LeadershipError,
    LeadershipGuard,
    atomic_record,
    process_identity,
    read_record,
    same_process,
    verify_readiness,
)
from .recovery import fence_slurm_generation, local_quiescence


def _child_setup():
    # Linux only. The inherited lock descriptor also prevents a child that
    # survives supervisor death from leaving an unguarded journal writer.
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    os.setsid()
    if os.getppid() != parent:
        os.kill(os.getpid(), signal.SIGTERM)


def _terminate(process, timeout=10):
    if process and process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def service_status(config) -> dict:
    directory = Path(config.state_dir) / "service"
    leader = read_record(directory / "leader.json")
    endpoint = read_record(directory / "endpoint.json")
    deployment = read_record(directory / "deployment.json")
    return {
        "schema_version": 1,
        "leader": leader,
        "endpoint": endpoint,
        "deployment": deployment,
        "heartbeat_age_seconds": time.time() - leader["heartbeat_at"] if leader else None,
        "standby_readiness": read_record(directory / "readiness_binding.json"),
        "note": "A heartbeat is diagnostic evidence only; it does not authorize takeover.",
    }


def _config_snapshot(config, directory):
    content = json.dumps(config.model_dump(), sort_keys=True).encode()
    path = directory / ("config-" + hashlib.sha256(content).hexdigest()[:16] + ".json")
    if not path.exists():
        atomic_record(path, config.model_dump())
    return path


def _submit_control(config, directory, deployment, role, client, *, exclude_host=None, renewal_from=None):
    """Persist exactly one control intent before its one external submit call."""
    token = uuid.uuid4().hex
    script = directory / f"control-{token}.sh"
    args = [
        sys.executable,
        "-m",
        "hyper_hyperqueue.service.supervisor",
        "run",
        "--config",
        deployment["config"],
        "--role",
        role,
        "--deployment-token",
        token,
    ]
    script.write_text("#!/bin/bash\nexec " + shlex.join(args) + "\n")
    script.chmod(0o700)
    allocation = {
        "token": token,
        "role": role,
        "script": str(script),
        "status": "submitting",
        "created_at": time.time(),
    }
    if renewal_from is not None:
        allocation["renewal_from"] = renewal_from
    deployment["allocations"].append(allocation)
    atomic_record(directory / "deployment.json", deployment)
    control = config.control
    logs = log_directory(config, "service", fallback=directory)
    extra = [
        "--partition",
        control.partition,
        "--cpus-per-task",
        str(control.cpus),
        "--mem",
        str(control.memory_mib),
        "--time",
        str(max(1, (control.walltime_seconds + 59) // 60)),
        "--signal=B:USR1@90",
        "--no-requeue",
        "--output",
        str(logs / f"control-{token}.out"),
        "--error",
        str(logs / f"control-{token}.err"),
    ]
    if control.account:
        extra += ["--account", control.account]
    if exclude_host:
        extra += ["--exclude", exclude_host]
    result = client.submit(script, token, job_name=f"hhq-control-{role}", extra_args=extra)
    allocation.update(result.to_dict())
    atomic_record(directory / "deployment.json", deployment)
    return result


def maintain_standby(config, guard, *, client=None):
    """Keep two control allocations at most; terminal proof releases a slot."""
    if config.control.hosting != "slurm" or not config.control.standby_replicas:
        return {"status": "disabled"}
    verify_readiness(config.control.verification_receipt, Path(config.state_dir))
    from ..backends.slurm import SlurmClient

    client = client or SlurmClient()
    directory = Path(config.state_dir) / "service"
    guard.assert_owned()
    with (directory / "bootstrap.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        guard.assert_owned()
        deployment = read_record(directory / "deployment.json")
        if not deployment or deployment["hosting"] != "slurm":
            raise LeadershipError("No owned Slurm control deployment")
        if deployment.get("desired_state") == "STOPPED":
            return {"status": "stop_requested"}
        occupied = []
        for allocation in deployment["allocations"]:
            if allocation.get("status") == "rejected":
                continue
            if allocation.get("terminal_evidence"):
                continue  # Positively terminal allocations never consume a new slot.
            found = client.find_submission(
                allocation["token"], since=allocation.get("created_at", deployment["created_at"])
            )
            if not found:
                return {"status": "blocked_unknown_control", "token": allocation["token"]}
            active = [j for j in found if not j.terminal]
            if active:
                occupied.extend(active)
            else:
                allocation["terminal_evidence"] = [j.to_dict() for j in found]
                allocation["terminal_verified_at"] = time.time()
        atomic_record(directory / "deployment.json", deployment)
        failures = 0
        for allocation in reversed(deployment["allocations"]):
            records = allocation.get("terminal_evidence", [])
            if not records:
                continue
            if all(r.get("state") in {"FAILED", "BOOT_FAIL", "OUT_OF_MEMORY", "NODE_FAIL"} for r in records):
                failures += 1
            else:
                break
        if failures >= config.policy.max_rpc_failures:
            return {
                "status": "blocked_control_startup_failures",
                "consecutive_failures": failures,
                "next_action": "Inspect exact control stderr, repair, then explicitly restart the deployment",
            }
        if len(occupied) >= 2:
            return {"status": "capacity_present", "jobs": [j.job_id for j in occupied]}
        if len(occupied) != 1 or str(occupied[0].job_id) != str(guard.record.get("slurm_job_id")):
            raise LeadershipError("Active control identity does not match the sole owned allocation")
        result = _submit_control(
            config,
            directory,
            deployment,
            "standby",
            client,
            exclude_host=os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        )
        return {"status": "replacement_" + result.status, "submission": result.to_dict()}


def start_service(config, hosting=None, standby=None, *, slurm=None) -> dict:
    """Launch once and return; does not tie the service to a notebook process."""
    config = config.model_copy(deep=True)
    if hosting is not None:
        config.control.hosting = hosting
    if standby is not None:
        config.control.standby_replicas = standby
    from .renewal import validate_settings

    validate_settings(config)
    directory = Path(config.state_dir) / "service"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    check_version(locate_binary(config.control.hq_binary), python=True)
    if config.control.standby_replicas:
        if config.control.hosting != "slurm":
            raise LeadershipError(
                "Standby is supported only with positively fenced Slurm control allocations"
            )
        verified = verify_readiness(config.control.verification_receipt, Path(config.state_dir))
        atomic_record(directory / "readiness_binding.json", verified)
    with (directory / "bootstrap.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = read_record(directory / "deployment.json")
        if previous:
            if (
                previous["hosting"] == "local"
                and previous.get("process")
                and same_process(previous["process"])
            ):
                return service_status(config)
            if previous["hosting"] == "local" and not previous.get("process"):
                raise LeadershipError(
                    "Local supervisor launch acknowledgement is missing; reconcile before another start"
                )
            if previous["hosting"] == "slurm":
                from ..backends.slurm import SlurmClient

                slurm = slurm or SlurmClient()
                for allocation in previous.get("allocations", []):
                    if allocation.get("terminal_evidence") or allocation.get("status") == "rejected":
                        continue
                    found = slurm.find_submission(allocation["token"], since=previous["created_at"])
                    if any(not item.terminal for item in found):
                        return service_status(config)
                    if allocation.get("status") != "rejected" and not found:
                        raise LeadershipError(
                            "Control submission remains unproven; reconcile before another start"
                        )
            atomic_record(directory / "deployments" / (previous["deployment_id"] + ".json"), previous)
        path = _config_snapshot(config, directory)
        token = uuid.uuid4().hex
        deployment = {
            "schema": "hhq-service-deployment/1",
            "deployment_id": token,
            "hosting": config.control.hosting,
            "created_at": time.time(),
            "config": str(path),
            "status": "STARTING",
            "desired_state": "RUNNING",
            "allocations": [],
        }
        atomic_record(directory / "deployment.json", deployment)
        base = [sys.executable, "-m", "hyper_hyperqueue.service.supervisor", "run", "--config", str(path)]
        if config.control.hosting == "local":
            logs = log_directory(config, "service", fallback=directory)
            with (logs / "supervisor.log").open("ab") as log:
                process = subprocess.Popen(
                    base,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    close_fds=True,
                )
            deployment["process"] = process_identity(process.pid)
            atomic_record(directory / "deployment.json", deployment)
        else:
            from ..backends.slurm import SlurmClient

            slurm = slurm or SlurmClient()
            for ordinal in range(1 + config.control.standby_replicas):
                exclude_host = None
                if ordinal:
                    # A bounded observation improves placement when the first
                    # control starts promptly; a pending primary is not forecast.
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        leader = read_record(directory / "leader.json", {})
                        if leader.get("deployment_token") == deployment["allocations"][0]["token"]:
                            exclude_host = leader["host"]
                            break
                        time.sleep(0.2)
                result = _submit_control(
                    config,
                    directory,
                    deployment,
                    "active" if ordinal == 0 else "standby",
                    slurm,
                    exclude_host=exclude_host,
                )
                if result.status != "accepted":
                    break
        return service_status(config)


def stop_service(config, drain=True) -> dict:
    directory = Path(config.state_dir) / "service"
    if not directory.exists():
        return {"status": "not_started"}
    with (directory / "bootstrap.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        leader = read_record(directory / "leader.json")
        deployment = read_record(directory / "deployment.json")
        if not leader and not deployment:
            return {"status": "not_started"}
        request = {
            "action": "drain" if drain else "stop",
            "epoch": leader.get("epoch") if leader else None,
            "owner_id": leader.get("owner_id") if leader else None,
            "deployment_id": deployment.get("deployment_id") if deployment else None,
            "requested_at": time.time(),
        }
        if deployment:
            deployment["desired_state"] = "STOPPED"
            atomic_record(directory / "deployment.json", deployment)
        atomic_record(directory / "control_request.json", request)
    return {
        "status": "requested",
        **request,
        "semantics": "Stops future controls before leadership; drain stops admissions and lets existing HQ backlog finish; stop interrupts HQ",
    }


def _generation_allocations(config, previous, client):
    """Resolve every exact old-generation control/worker ownership token."""
    directory = Path(config.state_dir) / "service"
    deployment = read_record(directory / "deployment.json", {})
    candidates = [deployment] + [read_record(p) for p in (directory / "deployments").glob("*.json")]
    deployment = next(
        (
            d
            for d in candidates
            if any(a.get("job_id") == previous.get("slurm_job_id") for a in d.get("allocations", []))
        ),
        deployment,
    )
    allocations = [
        a for a in deployment.get("allocations", []) if a.get("job_id") == previous.get("slurm_job_id")
    ]
    # Worker receipts are immutable ownership hints. A job's exact token must
    # also match Slurm before a cancellation is admitted.
    for path in (directory / "workers").glob("*.json"):
        worker = read_record(path)
        if worker.get("epoch") == previous["epoch"] and worker.get("slurm_job_id"):
            if not worker.get("allocation_token"):
                return None
            allocations.append({"job_id": worker["slurm_job_id"], "token": worker["allocation_token"]})
    from ..state import StateStore

    # Include workers that were accepted but have not yet written a startup
    # receipt, so takeover cannot overlook a pending or unacknowledged worker.
    for intent in StateStore(config.state_dir).intents():
        if intent.get("backend") != "worker" or intent.get("epoch") != previous["epoch"]:
            continue
        if intent.get("state") == "rejected":
            continue
        found = client.find_submission(
            intent["token"], since=intent.get("created_at", deployment["created_at"])
        )
        if not found:
            return None
        for row in found:
            allocations.append({"job_id": row.job_id, "token": intent["token"]})
    unique = {}
    for allocation in allocations:
        key = str(allocation["job_id"])
        if key in unique and unique[key]["token"] != allocation["token"]:
            raise LeadershipError("Conflicting ownership tokens for recovery allocation")
        unique[key] = allocation
    allocations = list(unique.values())
    if previous.get("slurm_job_id") and not any(
        str(a["job_id"]) == str(previous["slurm_job_id"]) for a in allocations
    ):
        return None
    return deployment, allocations


def _terminal_generation_proof(config, previous, *, client=None):
    """An explicit restart can adopt already-terminal resources without fencing.

    This read-only path neither cancels jobs nor enables automatic failover.
    A live/unknown prior allocation still blocks replay.
    """
    from ..backends.slurm import TERMINAL_STATES, SlurmClient, matches_submission

    client = client or SlurmClient()
    resolved = _generation_allocations(config, previous, client)
    if resolved is None:
        raise LeadershipError("Previous generation contains an unresolved ownership/submission intent")
    deployment, allocations = resolved
    if not previous.get("slurm_job_id"):
        local = local_quiescence(previous, host=socket.gethostname(), children=previous.get("children", []))
        if not local.ready:
            raise LeadershipError("Previous local owner/children are not positively quiescent")
    live = client.query_jobs() if allocations else []
    accounting = (
        client.accounting(
            [a["job_id"] for a in allocations], since=deployment.get("created_at", time.time() - 86400)
        )
        if allocations
        else []
    )
    observed = []
    for allocation in allocations:
        current = [j for j in live if str(j.job_id) == str(allocation["job_id"])]
        historical = [j for j in accounting if str(j.job_id) == str(allocation["job_id"])]
        if (
            current
            or not historical
            or any(j.state not in TERMINAL_STATES for j in historical)
            or not all(matches_submission(j, allocation["token"], client.user) for j in historical)
        ):
            raise LeadershipError(
                "Previous allocations are live or terminal ownership is unproven; stop only the exact owned jobs and retry after accounting confirms termination"
            )
        observed.append(
            {
                "job_id": str(allocation["job_id"]),
                "token": allocation["token"],
                "owned": True,
                "live": False,
                "accounting": [j.to_dict() for j in historical],
            }
        )
    proof = {
        "schema": "hhq-manual-restart-proof/1",
        "previous_owner": previous["owner_id"],
        "previous_epoch": previous["epoch"],
        "quiescent": True,
        "allocations": observed,
        "basis": "read-only terminal reconciliation; no cancellation or standby activation",
        "verified_at": time.time(),
    }
    atomic_record(Path(config.state_dir) / "service" / f"manual-restart-{previous['epoch']}.json", proof)
    return proof


def _slurm_recovery(config, previous):
    from ..backends.slurm import SlurmClient, matches_submission

    directory = Path(config.state_dir) / "service"
    ready = verify_readiness(config.control.verification_receipt, Path(config.state_dir))
    binding = read_record(directory / "readiness_binding.json")
    if binding and binding["sha256"] != ready["sha256"]:
        raise LeadershipError("Failover readiness changed since deployment")
    client = SlurmClient()
    resolved = _generation_allocations(config, previous, client)
    if resolved is None:
        return None
    deployment, allocations = resolved
    intent_path = directory / f"recovery-intent-{previous['epoch']}.json"
    intent = {
        "schema": "hhq-service-fencing-intent/1",
        "reason": "generation_recovery",
        "previous_owner": previous["owner_id"],
        "previous_epoch": previous["epoch"],
        "allocations": sorted(
            [{"job_id": str(a["job_id"]), "token": a["token"]} for a in allocations],
            key=lambda a: a["job_id"],
        ),
        "readiness": {"path": ready["path"], "sha256": ready["sha256"]},
    }
    existing = read_record(intent_path)
    if existing and {k: v for k, v in existing.items() if k != "created_at"} != intent:
        raise LeadershipError("Recovery intent changed; manual reconciliation required")
    if not existing:
        intent["created_at"] = time.time()
        atomic_record(intent_path, intent)
    since = deployment.get("created_at", time.time() - 86400)

    def query(allocation):
        live = [j for j in client.query_jobs() if j.job_id == allocation["job_id"]]
        accounting = [
            j
            for j in client.accounting([allocation["job_id"]], since=since)
            if j.job_id == allocation["job_id"]
        ]
        rows = live or accounting
        owned = bool(rows) and all(matches_submission(j, allocation["token"], client.user) for j in rows)
        return {
            "job_id": allocation["job_id"],
            "token": allocation["token"],
            "owned": owned,
            "live": bool(live),
            "accounting_state": accounting[0].state if accounting else None,
        }

    result = fence_slurm_generation(
        previous,
        allocations,
        query=query,
        cancel=lambda a: client.cancel([a["job_id"]], expected_token=a["token"]),
        readiness={"path": ready["path"], "sha256": ready["sha256"]},
    )
    if result.proof:
        result.proof["intent"] = {
            "path": str(intent_path),
            "sha256": hashlib.sha256(intent_path.read_bytes()).hexdigest(),
        }
    atomic_record(directory / f"recovery-{previous['epoch']}.json", result.__dict__)
    return result.proof if result.ready else None


def _verify_old_workers(config, previous, *, terminal_proof=None):
    """Even graceful HQ stop does not prove old worker child processes ended."""
    if (
        terminal_proof
        and terminal_proof.get("quiescent") is True
        and terminal_proof.get("previous_owner") == previous.get("owner_id")
        and terminal_proof.get("previous_epoch") == previous.get("epoch")
    ):
        return
    directory = Path(config.state_dir) / "service"
    workers = [read_record(p) for p in (directory / "workers").glob("*.json")]
    workers = [w for w in workers if w.get("epoch") == previous.get("epoch")]
    from ..state import StateStore

    intents = [
        i
        for i in StateStore(config.state_dir).intents()
        if i.get("backend") == "worker"
        and i.get("epoch") == previous.get("epoch")
        and i.get("state") != "rejected"
    ]
    if not workers and not intents:
        return
    if intents or any(w.get("slurm_job_id") for w in workers):
        if previous.get("state") == "STOPPED":
            _terminal_generation_proof(config, previous)
            return
        if not _slurm_recovery(config, previous):
            raise LeadershipError("Old Slurm workers are not positively fenced; journal replay blocked")
        return
    for worker in workers:
        if worker.get("host") != socket.gethostname() or not worker.get("process"):
            raise LeadershipError("Old worker process identity unavailable; journal replay blocked")
        if same_process(worker["process"]):
            raise LeadershipError("Old worker is still alive; wait for stop or fence before journal replay")


def finish_completed_service(config, guard, backend, shutdown, *, client=None):
    """Final atomic completion barrier. New imports cannot race the decision.

    Only a fresh controller reconciliation can begin this path. It never cancels
    running workers; they have already idled out with positive terminal evidence.
    Running standbys observe desired_state=STOPPED and exit their wait loop.
    """
    if not config.control.stop_when_complete:
        return False
    from ..backends.slurm import SlurmClient, matches_submission
    from ..state import StateStore
    from .completion import hq_settled, work_status

    store = StateStore(config.state_dir)
    directory = store.root / "service"
    report = read_record(store.root / "controller_status.json", {})
    completion = report.get("completion", {})
    if (report.get("state") != "healthy" or not completion.get("ready")
            or completion.get("epoch") != guard.epoch
            or time.time() - completion.get("observed_at", 0)
            > 2 * config.policy.reconcile_interval_seconds + 30):
        return False
    with store.lock():
        guard.assert_owned()
        status = work_status(store, report.get("retry_decisions", []))
        if not status["settled"] or not hq_settled(store, backend):
            return False
        if any(worker.get("ended") is None for worker in backend.workers()):
            return False
        # Serializes standby maintenance and explicit service stop/start.
        with (directory / "bootstrap.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            guard.assert_owned()
            deployment = read_record(directory / "deployment.json", {})
            if deployment.get("desired_state") == "STOPPED":
                return False  # Explicit operator drain/stop retains its own meaning.
            if deployment.get("hosting") == "slurm":
                client = client or SlurmClient()
                for allocation in deployment.get("allocations", []):
                    if (str(allocation.get("job_id")) == str(guard.record.get("slurm_job_id"))
                            or allocation.get("status") == "rejected"
                            or allocation.get("terminal_evidence")):
                        continue
                    found = client.find_submission(
                        allocation["token"], since=allocation.get("created_at", deployment["created_at"])
                    )
                    if not found or not all(
                        matches_submission(row, allocation["token"], client.user) for row in found
                    ):
                        return False
                    pending = [row for row in found if row.state == "PENDING"]
                    if pending:
                        allocation["completion_cancel_requested_at"] = time.time()
                        atomic_record(directory / "deployment.json", deployment)
                        client.cancel_pending([row.job_id for row in pending], allocation["token"])
                        return False  # Confirm on a later observation, not the scancel response.
                    if all(row.terminal for row in found):
                        allocation["terminal_evidence"] = [row.to_dict() for row in found]
                    elif any(row.state != "RUNNING" for row in found):
                        return False
                    # An already-running standby exits on desired_state=STOPPED.
            receipt = {"schema": "hhq-completion-shutdown/1", "phase": "closing", "epoch": guard.epoch,
                       "owner_id": guard.owner_id, "deployment_id": deployment.get("deployment_id"),
                       "committed_at": time.time(), "counts": status["counts"],
                       "reason": "all_registered_work_settled"}
            deployment.update(desired_state="STOPPED", completion=receipt)
            atomic_record(directory / "deployment.json", deployment)
            atomic_record(directory / "completion.json", receipt)
            # Keep the mutation barrier through actual child shutdown. An import
            # waiting for this lock must explicitly restart the completed service.
            shutdown()
            receipt.update(phase="children_stopped", children_stopped_at=time.time())
            atomic_record(directory / "completion.json", receipt)
            return True


def run_supervisor(
    config,
    config_path: Path,
    *,
    role="active",
    deployment_token=None,
    controller_argv=None,
    no_controller=False,
    max_seconds=None,
):
    from .renewal import (
        RenewalNotNeeded,
        announce_waiting,
        cancel_if_unneeded,
        drain_workers,
        early_renewal_due,
        prepare_successor,
        record_handoff,
        successor_ready,
        takeover_proof,
        validate_settings,
    )

    validate_settings(config)
    if threading.active_count() != 1:
        raise LeadershipError("Supervisor must run in its own single-threaded process")
    directory = Path(config.state_dir) / "service"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    guard = LeadershipGuard(directory)
    hq = controller = None
    stopped = {"value": False, "drain": False}
    successor = handoff_workers = None
    renewal_wait_until = time.monotonic() + config.control.renew_lead_seconds + 300
    if role == "renewal":
        from ..worker import allocation_remaining

        own_remaining = allocation_remaining()
        declared = [a for a in read_record(directory / "deployment.json", {}).get("allocations", [])
                    if a.get("token") == deployment_token]
        predecessor_end = (declared[0].get("renewal_from", {}).get("control_end_time") if declared else None)
        predecessor_end = predecessor_end or read_record(directory / "endpoint.json", {}).get("control_end_time")
        bounds = [renewal_wait_until]
        if own_remaining is not None:
            bounds.append(time.monotonic() + max(0, own_remaining - 60))
        if predecessor_end is not None:
            bounds.append(time.monotonic() + max(0, predecessor_end - time.time()) + 300)
        renewal_wait_until = min(bounds)

    def handle(signum, _frame):
        stopped["value"] = signum != signal.SIGUSR1
        stopped["drain"] = signum == signal.SIGUSR1

    original_signals = {s: signal.signal(s, handle) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    try:
        while True:
            if (
                stopped["value"]
                or stopped["drain"]
                or read_record(directory / "deployment.json", {}).get("desired_state") == "STOPPED"
            ):
                return  # An expiring/stopped standby must not fence a healthy owner.
            previous = read_record(directory / "leader.json")
            proof = None
            if role == "renewal":
                if previous and previous.get("state") == "STOPPED":
                    renewal_wait_until = min(renewal_wait_until, time.monotonic() + 300)
                try:
                    announce_waiting(config, previous, deployment_token, os.environ.get("SLURM_JOB_ID"))
                    proof = takeover_proof(config, previous, deployment_token, os.environ.get("SLURM_JOB_ID"))
                except RenewalNotNeeded:
                    return
                except LeadershipError as exc:
                    atomic_record(directory / f"renewal-wait-{deployment_token}.json",
                                  {"status": "blocked", "reason": str(exc), "at": time.time()})
                    if time.monotonic() >= renewal_wait_until:
                        raise
                    time.sleep(15)
                    continue
            elif previous and role == "active" and not config.control.standby_replicas:
                proof = _terminal_generation_proof(config, previous)
            elif previous and previous.get("state") != "STOPPED":
                if (role == "standby" or config.control.hosting == "slurm") and time.time() - previous.get(
                    "heartbeat_at", 0
                ) < 120:
                    role = "standby"  # The other contender may have acquired the lock first.
                    time.sleep(2)
                    continue
                if previous.get("slurm_job_id"):
                    proof = _slurm_recovery(config, previous)
                else:
                    children = [p for p in previous.get("children", []) if p]
                    proof = local_quiescence(previous, host=socket.gethostname(), children=children).proof
            try:
                if previous:
                    _verify_old_workers(config, previous, terminal_proof=proof)
                guard.acquire(takeover_proof=proof)
                break
            except LeadershipError:
                if role != "standby" or stopped["value"]:
                    raise
                time.sleep(2)
        binary = locate_binary(config.control.hq_binary)
        check_version(binary, python=True)
        server_dir = directory / "hq"
        server_dir.mkdir(mode=0o700, exist_ok=True)
        os.chmod(server_dir, 0o700)
        args = [
            binary,
            "--server-dir",
            str(server_dir),
            "server",
            "start",
            "--journal",
            str(directory / "hq.journal"),
            "--journal-flush-period=1s",
            "--host",
            config.control.advertise_host or socket.gethostname(),
        ]
        if config.control.server_port:
            args += ["--client-port", str(config.control.server_port)]
        if config.control.worker_port:
            args += ["--worker-port", str(config.control.worker_port)]
        env = dict(os.environ, HHQ_LEADER_FD=str(guard.fd), HHQ_SERVER_EPOCH=str(guard.epoch))
        logs = log_directory(config, "service", fallback=directory)
        with (logs / f"hq-{guard.epoch}.log").open("ab") as log:
            hq = subprocess.Popen(
                args,
                stdout=log,
                stderr=log,
                env=env,
                pass_fds=(guard.fd,),
                preexec_fn=_child_setup,  # noqa: PLW1509 - supervisor enforces single-threaded execution.
            )
        guard.heartbeat("STARTING", children=[process_identity(hq.pid)], deployment_token=deployment_token)
        backend = HQBackend(server_dir, binary, timeout=5)
        deadline = time.monotonic() + 30
        while True:
            guard.assert_owned()
            if hq.poll() is not None:
                raise HQError(f"HQ server failed; inspect hq-{guard.epoch}.log")
            try:
                info = backend.json("server", "info")
                break
            except (HQError, subprocess.TimeoutExpired):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
        from ..worker import allocation_remaining

        control_remaining = allocation_remaining()
        endpoint = {
            "schema": "hhq-endpoint/1",
            "epoch": guard.epoch,
            "status": "ACTIVE",
            "owner_id": guard.owner_id,
            "server_dir": str(server_dir),
            "server": info,
            "published_at": time.time(),
            "control_end_time": time.time() + control_remaining if control_remaining is not None else None,
        }
        guard.heartbeat("ACTIVE")
        atomic_record(directory / "endpoint.json", endpoint)
        command = controller_argv or [
            sys.executable,
            "-m",
            "hyper_hyperqueue.controller",
            "--state-dir",
            config.state_dir,
            "--config",
            str(config_path),
            "--epoch",
            str(guard.epoch),
        ]
        failures, next_restart = 0, 0.0
        started = time.monotonic()
        next_standby_check = 0.0
        next_completion_check = 0.0
        next_renewal_check = 0.0
        while not stopped["value"]:
            guard.assert_owned()
            request = read_record(directory / "control_request.json", {})
            if request.get("epoch") == guard.epoch and request.get("owner_id") == guard.owner_id:
                stopped["drain"] |= request.get("action") == "drain"
                stopped["value"] |= request.get("action") == "stop"
            if hq.poll() is not None:
                guard.heartbeat("SUSPECT")
                raise HQError("HQ server exited; stop admission and fence old workers before recovery")
            if not stopped["drain"] and time.monotonic() >= next_completion_check:
                try:
                    if finish_completed_service(
                        config, guard, backend, lambda controller=controller: (_terminate(controller), _terminate(hq))
                    ):
                        break
                except (OSError, RuntimeError, TypeError, subprocess.TimeoutExpired) as exc:
                    atomic_record(directory / "completion_error.json",
                                  {"epoch": guard.epoch, "error": str(exc), "at": time.time()})
                next_completion_check = time.monotonic() + config.policy.reconcile_interval_seconds
            completion = read_record(Path(config.state_dir) / "controller_status.json", {}).get("completion", {})
            finishing = (completion.get("epoch") == guard.epoch
                         and completion.get("work", {}).get("settled") is True)
            remaining = endpoint["control_end_time"] - time.time() if endpoint["control_end_time"] else None
            if (config.control.auto_renew and not successor and not finishing
                    and read_record(directory / "deployment.json", {}).get("desired_state") == "RUNNING"
                    and (stopped["drain"] or remaining is not None and remaining <= config.control.renew_lead_seconds
                         or early_renewal_due(config, guard))
                    and time.monotonic() >= next_renewal_check):
                try:
                    planned = prepare_successor(config, guard)
                    atomic_record(directory / "renewal_status.json", {**planned, "at": time.time()})
                    if planned["status"] in {"accepted", "unknown", "submitting"}:
                        successor = planned
                except (LeadershipError, OSError, RuntimeError) as exc:
                    atomic_record(directory / "renewal_status.json",
                                  {"status": "blocked", "reason": str(exc), "at": time.time()})
                next_renewal_check = time.monotonic() + 30
            if successor and not stopped["drain"] and time.monotonic() >= next_renewal_check:
                try:
                    ready = successor_ready(config, guard, successor)
                    atomic_record(directory / "renewal_readiness.json", {**ready, "at": time.time()})
                    stopped["drain"] = ready["ready"]
                except (LeadershipError, OSError, RuntimeError) as exc:
                    atomic_record(directory / "renewal_readiness.json",
                                  {"ready": False, "reason": str(exc), "at": time.time()})
                next_renewal_check = time.monotonic() + 15
            if not stopped["drain"] and not finishing and time.monotonic() >= next_standby_check:
                try:
                    renewal = maintain_standby(config, guard)
                    atomic_record(directory / "standby_maintenance.json", {**renewal, "at": time.time()})
                except (LeadershipError, OSError, RuntimeError) as exc:
                    atomic_record(
                        directory / "standby_maintenance.json",
                        {"status": "blocked", "reason": str(exc), "at": time.time()},
                    )
                next_standby_check = time.monotonic() + 30
            if stopped["drain"]:
                _terminate(controller)
                endpoint["status"] = "DRAINING"
                atomic_record(directory / "endpoint.json", endpoint)
                if successor:
                    if time.monotonic() >= next_renewal_check:
                        try:
                            cancel_if_unneeded(config, guard, successor, backend)
                            workers = drain_workers(config, guard)
                            atomic_record(directory / "renewal_drain.json", workers)
                            if workers["settled"] and not any(w.get("ended") is None for w in backend.workers()):
                                backend.json("journal", "flush")
                                workers["journal_flushed_at"] = time.time()
                                handoff_workers = workers
                                break
                        except (LeadershipError, OSError, RuntimeError) as exc:
                            atomic_record(directory / "renewal_drain.json",
                                          {"settled": False, "reason": str(exc), "at": time.time()})
                        next_renewal_check = time.monotonic() + 15
                elif not any(w.get("ended") is None for w in backend.workers()):
                    break
            elif not no_controller and time.monotonic() >= next_restart:
                if controller is None or controller.poll() is not None:
                    if controller is not None:
                        failures += 1
                    if failures >= config.policy.max_rpc_failures:
                        raise LeadershipError("Repeated controller exits; admission circuit breaker opened")
                    with (logs / f"controller-{guard.epoch}.log").open("ab") as log:
                        controller = subprocess.Popen(
                            command,
                            stdout=log,
                            stderr=log,
                            env=env,
                            pass_fds=(guard.fd,),
                            preexec_fn=_child_setup,  # noqa: PLW1509 - supervisor enforces single-threaded execution.
                        )
                    next_restart = time.monotonic() + 2**failures
            children = [process_identity(p.pid) for p in (hq, controller) if p and p.poll() is None]
            guard.heartbeat("DRAINING" if stopped["drain"] else "ACTIVE", children=children)
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                break
            time.sleep(0.5)
    finally:
        if successor and guard.fd is not None:
            with contextlib.suppress(OSError, RuntimeError):
                # Explicit stop can arrive between drain polls. It still vetoes
                # the successor before this control exits.
                cancel_if_unneeded(config, guard, successor, backend,
                                   abort=stopped["value"] or handoff_workers is None)
        _terminate(controller)
        _terminate(hq)
        if guard.fd is not None:
            try:
                if successor and handoff_workers and not stopped["value"]:
                    record_handoff(config, guard, successor, handoff_workers)
            finally:
                with contextlib.suppress(OSError, LeadershipError):
                    atomic_record(
                        directory / "endpoint.json",
                        {"epoch": guard.epoch, "status": "STOPPED", "stopped_at": time.time()},
                    )
                    guard.close(children_quiescent=True)
                guard.close()
        for signum, handler in original_signals.items():
            signal.signal(signum, handler)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run"])
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--role", choices=["active", "standby", "renewal"], default="active")
    p.add_argument("--deployment-token")
    p.add_argument("--no-controller", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    from ..config import load_config

    run_supervisor(
        load_config(args.config),
        args.config,
        role=args.role,
        deployment_token=args.deployment_token,
        no_controller=args.no_controller,
    )


if __name__ == "__main__":
    main()
