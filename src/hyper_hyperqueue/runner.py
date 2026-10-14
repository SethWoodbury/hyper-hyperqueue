"""Common direct/HQ command runner. Completion is owned by immutable task identity."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from .adaptive_recovery import (
    classify_memory_failure,
    effective_task,
    prepare_resource_override,
    verified_scheduler_proof,
)
from .logs import log_directory
from .models import TaskSpec
from .state import StateStore, atomic_json, ensure_directory, file_lock, fingerprint, read_json, utc_now
from .telemetry import ProcessSampler, ResourceMismatch, allocation_context, assigned_gpu_context


def task_argv(task: TaskSpec) -> list[str]:
    if task.argv is not None:
        return list(task.argv)
    # The original command is exactly one argument: no eval, no implicit pipefail.
    return [task.shell, *task.shell_options, "-c", task.command]


def _signal_group(process: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _execute(
    argv: list[str],
    *,
    cwd: str,
    env: dict[str, str],
    stdout,
    stderr,
    timeout: float | None,
    grace: float = 2.0,
    devices=(),
) -> dict:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Task runner must execute in the main thread to forward signals safely")
    received: list[int] = []
    process = None
    handlers = {}

    def forward(signum, _frame):
        received.append(signum)
        if process is not None:
            _signal_group(process, signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1):
        handlers[signum] = signal.signal(signum, forward)
    started = time.monotonic()
    sampler = None
    timed_out = False
    deadline = None
    try:
        # Popen creates a new session in the child before exec; its PID is the PGID.
        process = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True
        )
        for signum in received:
            _signal_group(process, signum)
        sampler = ProcessSampler(process.pid, devices=devices, environment=env)
        next_sample = started
        while True:
            now = time.monotonic()
            if now >= next_sample:
                sampler.sample()
                next_sample = now + sampler.interval_seconds
            if received and deadline is None:
                deadline = now + grace
            if deadline is None:
                # Check completion before timeout; a delayed runner poll must not
                # label an already completed short command as timed out.
                # Observe without reaping: the zombie leader keeps its PID/PGID
                # reserved until same-group background descendants are killed.
                exited = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if exited is not None:
                    _signal_group(process, signal.SIGKILL)
                    break
                if timeout is not None and now - started >= timeout:
                    timed_out = True
                    _signal_group(process, signal.SIGTERM)
                    deadline = now + grace
            if deadline is not None and now >= deadline:
                # Descendants may still be alive even if the shell already exited.
                # Do not reap the leader before this kill: its PID reserves PGID.
                _signal_group(process, signal.SIGKILL)
                break
            time.sleep(0.05)
        returncode = process.wait()
        return {
            "returncode": returncode,
            "application_signal": -returncode if returncode < 0 else None,
            "forwarded_signals": received,
            "timed_out": timed_out,
            "process_group_cleanup": "remaining same-group descendants killed before leader was reaped",
            "telemetry": sampler.summary(),
        }
    except BaseException:
        if process is not None:
            _signal_group(process, signal.SIGKILL)
            process.wait()
        raise
    finally:
        for signum, previous in handlers.items():
            signal.signal(signum, previous)


def _output_path(path: str, task: TaskSpec, attempt_dir: Path) -> Path:
    # Output declarations may refer explicitly to runner-provided paths. Commands are unchanged.
    for key, value in {
        "{attempt_dir}": str(attempt_dir),
        "{task_id}": task.task_id,
        "{campaign_id}": task.campaign_id,
    }.items():
        path = path.replace(key, value)
    value = Path(path)
    return value if value.is_absolute() else Path(task.cwd) / value


def _validate(task: TaskSpec, attempt_dir: Path, env: dict[str, str]) -> dict:
    checks = []
    for expected in task.output.expected_paths:
        path = _output_path(expected, task, attempt_dir)
        try:
            stat = path.stat()
            valid = path.is_file() and stat.st_size >= task.output.min_bytes
            checks.append({"path": str(path), "valid": valid, "size_bytes": stat.st_size})
        except OSError as error:
            checks.append({"path": str(path), "valid": False, "error": str(error)})
    if any(not item["valid"] for item in checks):
        return {"valid": False, "kind": "expected_paths", "paths": checks}
    if task.output.validator_argv:
        logs = Path(env.get("HHQ_TASK_LOG_DIR", str(attempt_dir)))
        with (
            (logs / "validator.stdout").open("wb") as out,
            (logs / "validator.stderr").open("wb") as err,
        ):
            result = _execute(
                task.output.validator_argv,
                cwd=task.cwd,
                env=env,
                stdout=out,
                stderr=err,
                timeout=task.resources.timeout_seconds or 300.0,
            )
        return {
            "valid": result["returncode"] == 0
            and not result["timed_out"]
            and not result["forwarded_signals"],
            "kind": "validator_argv",
            "paths": checks,
            "execution": result,
        }
    return {"valid": True, "kind": "expected_paths" if checks else "exit_code_only", "paths": checks}


def attempt_epoch(previous: dict) -> str | None:
    """Physical worker generation, with compatibility for pre-worker-epoch records.

    HQ restores immutable tasks with their original dispatch epoch. Their actual
    worker may belong to a newer generation, which owns fencing of that attempt.
    """
    environment = previous.get("context", {}).get("environment", {})
    return environment.get("HHQ_WORKER_EPOCH", environment.get("HHQ_SERVER_EPOCH"))


def _recovery_allows(task: TaskSpec, previous: dict, coordinator: dict) -> bool:
    """Explicit terminal/fencing evidence, never heartbeat-age or a PID absence heuristic."""
    proof = coordinator.get("recovery_proof", {})
    expected_epoch = attempt_epoch(previous)
    intent_path = proof.get("fencing_intent_path")
    intent_matches = not intent_path or (
        Path(intent_path).is_file()
        and hashlib.sha256(Path(intent_path).read_bytes()).hexdigest() == proof.get("fencing_intent_sha256")
    )
    return (
        task.output.repeatable
        and task.output.mode != "manual_on_ambiguity"
        and intent_matches
        and proof.get("attempt_id") == previous["attempt_id"]
        and proof.get("terminal_verified") is True
        and proof.get("processes_fenced") is True
        and bool(proof.get("evidence_path"))
        and Path(proof["evidence_path"]).is_file()
        and hashlib.sha256(Path(proof["evidence_path"]).read_bytes()).hexdigest()
        == proof.get("evidence_sha256")
        and proof.get("server_epoch") == expected_epoch
    )


def internal_fencing_evidence(
    store: StateStore, task: TaskSpec, previous: dict, path: str | Path
) -> dict | None:
    """Attribute cancellation only to a positively bound, predeclared service fence."""
    record = read_json(path)
    proof = record.get("proof", {})
    if (
        record.get("ready") is not True
        or proof.get("schema") != "hhq-fencing-proof/1"
        or proof.get("quiescent") is not True
    ):
        return None
    env = previous.get("context", {}).get("environment", {})
    epoch, job_id = attempt_epoch(previous), env.get("SLURM_JOB_ID")
    if epoch is None or job_id is None or str(proof.get("previous_epoch")) != epoch:
        return None
    binding = proof.get("intent", {})
    intent_path = Path(binding.get("path", "/nonexistent"))
    if not intent_path.is_file() or hashlib.sha256(intent_path.read_bytes()).hexdigest() != binding.get(
        "sha256"
    ):
        return None
    intention = read_json(intent_path)
    if (
        intention.get("schema") != "hhq-service-fencing-intent/1"
        or intention.get("reason") != "generation_recovery"
        or str(intention.get("previous_epoch")) != epoch
        or not intention.get("created_at")
    ):
        return None
    owners = [
        i
        for i in store.intents()
        if i.get("backend") == "worker" and str(i.get("job_id")) == job_id and str(i.get("epoch")) == epoch
    ]
    if len(owners) != 1:
        return None
    token = owners[0]["token"]
    if not any(
        str(a.get("job_id")) == job_id and a.get("token") == token for a in intention.get("allocations", [])
    ):
        return None
    valid = [
        a
        for a in proof.get("allocations", [])
        if str(a.get("job_id")) == job_id
        and a.get("token") == token
        and a.get("owned") is True
        and a.get("live") is False
        and a.get("accounting_state")
        in {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
        }
    ]
    if len(valid) != 1:
        return None
    raw = read_json(store.campaign_dir(task.campaign_id) / "task_states" / f"{task.task_id}.json")
    if raw.get("state", raw.get("status")) == "canceled":
        return None
    # A durable user cancel request outranks service recovery, including an
    # already processed request whose filename ends in .done.json.
    for request_path in (store.root / "requests").glob("cancel-*.json"):
        request = read_json(request_path)
        if request.get("campaign_id") == task.campaign_id and request.get("action") == "cancel_hq_campaign":
            return None
    result = previous.get("result", {})
    if result.get("finished_at"):
        started = intention["created_at"]
        created = (
            float(started)
            if isinstance(started, (int, float))
            else datetime.fromisoformat(started).timestamp()
        )
        if datetime.fromisoformat(result["finished_at"]).timestamp() < created:
            return None  # Cancellation preceded this service-owned fencing operation.
    return {**proof, "fencing_intent_path": str(intent_path), "fencing_intent_sha256": binding["sha256"]}


def authorize_retry(
    store: StateStore,
    campaign_id: str,
    task_id: str,
    *,
    manual: bool = False,
    terminal_evidence: str | Path | None = None,
) -> dict:
    """Authorize an existing failed task without widening immutable retry bounds.

    An unknown attempt additionally requires a durable fencing proof produced by
    the recovery service. Even a manual retry cannot turn uncertainty into safety.
    """
    with store.lock():
        task = store.task(campaign_id, task_id)
        state = store.task_state(campaign_id, task_id)
        attempts = store.attempts(campaign_id, task_id)
        if state.get("reason") == "exported_to_new_generation":
            raise ValueError("Task was retired into a recovery export; retry the new generation only")
        internal = (
            internal_fencing_evidence(store, task, attempts[-1], terminal_evidence)
            if terminal_evidence is not None and attempts
            else None
        )
        scheduler = (verified_scheduler_proof(store, task, attempts[-1], terminal_evidence)
                     if terminal_evidence is not None and attempts else None)
        if state["state"] in {"succeeded", "running", "submitted", "ready"} or (
            state["state"] == "canceled" and not (internal or scheduler)
        ):
            raise ValueError(f"Task is not retryable: {state['state']}")
        if len(attempts) >= task.retry.max_attempts:
            raise ValueError("Retry attempt limit exhausted; import a new generation to change policy")
        if not attempts:
            raise ValueError("No attempt evidence; reconcile submission ownership before retry")
        latest = attempts[-1]
        token = state.get("submission_token")
        if token:
            active = [
                item
                for item in store.intents()
                if item.get("token") == token
                and item.get("state") in ("intent", "unknown", "submitted", "running")
            ]
            env = latest.get("context", {}).get("environment", {})
            for intent in active:
                observed = (
                    env.get("HQ_JOB_ID")
                    if intent.get("backend") == "hq"
                    else env.get("SLURM_ARRAY_JOB_ID") or env.get("SLURM_JOB_ID")
                )
                assigned = intent.get("hq_job_id") if intent.get("backend") == "hq" else intent.get("job_id")
                if observed is None or assigned is None or str(observed) != str(assigned):
                    raise ValueError(
                        "Active or uncertain submission does not belong to the last attempt; reconcile before retry"
                    )
        update = {
            "state": "retry_wait",
            "after_attempt_id": latest["attempt_id"],
            "retry_authorized": True,
            "manual_retry": manual,
        }
        if "result" not in latest or internal or scheduler:
            if terminal_evidence is None:
                raise ValueError("Missing final result requires terminal/fencing evidence")
            path = Path(terminal_evidence).resolve()
            data = path.read_bytes()
            proof = internal or scheduler or __import__("json").loads(data)
            epoch = attempt_epoch(latest)
            if proof.get("quiescent") is not True or str(proof.get("previous_epoch")) != str(epoch):
                raise ValueError("Fencing proof does not bind this attempt epoch")
            if epoch is None and proof.get("attempt_id") != latest["attempt_id"]:
                raise ValueError("A direct-task fencing proof must name the exact attempt_id")
            update["recovery_proof"] = {
                "attempt_id": latest["attempt_id"],
                "terminal_verified": True,
                "processes_fenced": True,
                "server_epoch": epoch,
                "evidence_path": str(path),
                "evidence_sha256": hashlib.sha256(data).hexdigest(),
            }
            if internal:
                if "worker_lost" not in task.retry.reasons and not manual:
                    raise ValueError("Internal generation recovery is not allowed by retry reason policy")
                update["recovery_proof"].update(
                    kind="generation_recovery",
                    fencing_intent_path=internal["fencing_intent_path"],
                    fencing_intent_sha256=internal["fencing_intent_sha256"],
                )
            if scheduler:
                if scheduler["reason"] not in task.retry.reasons and not manual:
                    raise ValueError("Scheduler termination reason is not allowed by retry policy")
                update["recovery_proof"].update(kind="scheduler_termination", reason=scheduler["reason"])
            if not _recovery_allows(task, latest, update):
                raise ValueError("Output policy prohibits ambiguous replay")
        else:
            result = latest["result"]
            if result["status"] == "canceled":
                raise ValueError("User cancellations are never automatically retried")
            if not manual and (
                result["reason"] not in task.retry.reasons
                or not (task.output.repeatable or task.output.mode == "isolated")
            ):
                raise ValueError("Failure reason/output policy does not allow automatic retry")
        reason = (scheduler["reason"] if scheduler else "worker_lost" if internal
                  else latest.get("result", {}).get("reason"))
        override = prepare_resource_override(store, task, latest, reason)
        if override:
            update["resource_override"] = override
        return store.update_task_state(campaign_id, task_id, update)


def run_task(state_dir: str | Path, campaign_id: str, task_id: str) -> int:
    store = StateStore(state_dir)
    task = store.task(campaign_id, task_id)
    directory = store.campaign_dir(campaign_id)
    task_fp = fingerprint(task.model_dump(mode="json"))
    completion_path = directory / "completions" / f"{task.task_id}.json"
    # A duplicated dispatch never waits behind an active task and executes afterward.
    try:
        with file_lock(directory / "task_locks" / f"{task.task_id}.lock", blocking=False):
            if completion_path.exists():
                completion = read_json(completion_path)
                if completion.get("task_fingerprint") != task_fp:
                    raise ValueError("Completion fingerprint differs from immutable task")
                return 0
            coordinator = read_json(directory / "task_states" / f"{task.task_id}.json")
            if coordinator.get("status") in {"canceled", "blocked"}:
                return 130 if coordinator["status"] == "canceled" else 75
            assigned_epoch = coordinator.get("server_epoch")
            dispatched_epoch = os.environ.get("HHQ_SERVER_EPOCH")
            if assigned_epoch is not None and str(assigned_epoch) != dispatched_epoch:
                return 75  # A delayed old-generation dispatch is not a fresh attempt.
            attempts = store.attempts(campaign_id, task_id)
            if attempts:
                previous = attempts[-1]
                if previous["task_fingerprint"] != task_fp:
                    raise ValueError("Attempt fingerprint differs from immutable task")
                if "result" not in previous:
                    if not _recovery_allows(task, previous, coordinator):
                        return 75
                else:
                    result = previous["result"]
                    if result["status"] == "succeeded":
                        # Crash between result and completion publication: adopt exact validated result.
                        if result.get("task_fingerprint") != task_fp or not result.get("validation", {}).get(
                            "valid"
                        ):
                            raise ValueError("Unqualified success record")
                        atomic_json(
                            completion_path,
                            {
                                "schema_version": 1,
                                "task_id": task_id,
                                "campaign_id": campaign_id,
                                "task_fingerprint": task_fp,
                                "attempt_id": previous["attempt_id"],
                                "result_path": str(Path(previous["attempt_dir"]) / "result.json"),
                                "completed_at": result["finished_at"],
                                "validation": result["validation"],
                            },
                        )
                        return 0
                    manually_authorized = (
                        coordinator.get("manual_retry") is True
                        and coordinator.get("after_attempt_id") == previous["attempt_id"]
                    )
                    internal_retry = coordinator.get("recovery_proof", {}).get(
                        "kind"
                    ) in {"generation_recovery", "scheduler_termination"} and _recovery_allows(task, previous, coordinator)
                    if (result["status"] == "canceled" and not internal_retry) or (
                        not manually_authorized
                        and not internal_retry
                        and (
                            result["reason"] not in task.retry.reasons
                            or not (task.output.repeatable or task.output.mode == "isolated")
                        )
                    ):
                        return 75
            if len(attempts) >= task.retry.max_attempts:
                return 75
            if (attempts and task.retry.oom_memory_multipliers
                    and attempts[-1].get("result", {}).get("reason") == "host_oom"
                    and not (coordinator.get("retry_authorized")
                             and coordinator.get("after_attempt_id") == attempts[-1]["attempt_id"])):
                return 75  # Only the retry owner may choose an escalated attempt envelope.
            task = effective_task(store, task)
            with contextlib.ExitStack() as locks:
                collision = task.output.collision_key
                if collision:
                    key = hashlib.sha256(collision.encode()).hexdigest()
                    locks.enter_context(
                        file_lock(store.root / "output_locks" / f"{key}.lock", blocking=False)
                    )
                    claim_path = store.root / "output_claims" / f"{key}.json"
                    claim = {"campaign_id": campaign_id, "task_id": task_id, "task_fingerprint": task_fp}
                    if claim_path.exists() and read_json(claim_path) != claim:
                        return 75
                    atomic_json(claim_path, claim)
                attempt_id = f"{len(attempts) + 1:06d}_{uuid.uuid4().hex[:12]}"
                attempt_dir = directory / "attempts" / task_id / attempt_id
                ensure_directory(attempt_dir)
                (attempt_dir / "output").mkdir(mode=0o700)
                logs = log_directory(
                    store.campaign(campaign_id).get("resolved_config", {}),
                    "campaigns", campaign_id, task_id, attempt_id, fallback=attempt_dir,
                )
                for stream in ("stdout", "stderr"):
                    (logs / stream).touch(mode=0o600)
                attempt = {
                    "schema_version": 1,
                    "attempt_id": attempt_id,
                    "attempt_number": len(attempts) + 1,
                    "task_id": task_id,
                    "campaign_id": campaign_id,
                    "task_fingerprint": task_fp,
                    "attempt_dir": str(attempt_dir),
                    "log_dir": str(logs),
                    "started_at": utc_now(),
                    "runner_pid": os.getpid(),
                    "context": allocation_context(),
                    "argv": task_argv(task),
                    "resources": task.resources.model_dump(),
                    "resource_override": coordinator.get("resource_override"),
                }
                atomic_json(attempt_dir / "attempt.json", attempt)
                env = {
                    **os.environ,
                    **task.environment,
                    "HHQ_TASK_ID": task_id,
                    "HHQ_CAMPAIGN_ID": campaign_id,
                    "HHQ_ATTEMPT_ID": attempt_id,
                    "HHQ_ATTEMPT_DIR": str(attempt_dir),
                    "HHQ_TASK_OUTPUT_DIR": str(attempt_dir / "output"),
                    "HHQ_TASK_LOG_DIR": str(logs),
                }
                for variable in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    env.setdefault(variable, str(task.resources.cpus))
                result = {
                    "schema_version": 1,
                    "attempt_id": attempt_id,
                    "task_id": task_id,
                    "campaign_id": campaign_id,
                    "task_fingerprint": task_fp,
                    "status": "failed",
                    "reason": "runner_error",
                    "exit_code": 70,
                    "validation": {"valid": False, "kind": "not_run"},
                }
                try:
                    for variable in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
                        if variable in task.environment and task.environment[variable] != os.environ.get(
                            variable
                        ):
                            raise ResourceMismatch(f"Task environment cannot replace allocated {variable}")
                    if (
                        task.resources.gpus
                        and "CUDA_DEVICE_ORDER" in task.environment
                        and task.environment["CUDA_DEVICE_ORDER"] != os.environ.get("CUDA_DEVICE_ORDER")
                    ):
                        raise ResourceMismatch("Task environment cannot replace allocated CUDA_DEVICE_ORDER")
                    for variable in ("HHQ_WORKER_EPOCH", "HHQ_SERVER_EPOCH"):
                        if variable in task.environment and task.environment[variable] != os.environ.get(
                            variable
                        ):
                            raise ResourceMismatch(f"Task environment cannot replace provenance {variable}")
                    result["assigned_gpu"] = assigned_gpu_context(task.resources)
                    with (
                        (logs / "stdout").open("wb") as out,
                        (logs / "stderr").open("wb") as err,
                    ):
                        execution = _execute(
                            task_argv(task),
                            cwd=task.cwd,
                            env=env,
                            stdout=out,
                            stderr=err,
                            timeout=task.resources.timeout_seconds,
                            devices=result["assigned_gpu"]["devices"],
                        )
                        os.fsync(out.fileno())
                        os.fsync(err.fileno())
                    result["execution"] = execution
                    if execution["forwarded_signals"]:
                        signum = execution["forwarded_signals"][0]
                        result.update(
                            status="canceled" if signum in (signal.SIGTERM, signal.SIGINT) else "interrupted",
                            reason="user_cancellation"
                            if signum in (signal.SIGTERM, signal.SIGINT)
                            else "interrupted",
                            exit_code=128 + signum,
                        )
                    elif execution["timed_out"]:
                        result.update(status="failed", reason="timeout", timeout_scope="task", exit_code=124)
                    elif execution["returncode"] != 0:
                        rc = execution["returncode"]
                        result.update(
                            reason="application_signal" if rc < 0 else "application_exit",
                            exit_code=(128 - rc) if rc < 0 else rc,
                        )
                        memory_reason = classify_memory_failure(logs / "stderr")
                        if memory_reason:
                            result["reason"] = memory_reason
                    else:
                        validation = _validate(task, attempt_dir, env)
                        result["validation"] = validation
                        if validation["valid"]:
                            result.update(status="succeeded", reason="validated_success", exit_code=0)
                        elif validation.get("execution", {}).get("forwarded_signals"):
                            signum = validation["execution"]["forwarded_signals"][0]
                            result.update(
                                status="canceled", reason="user_cancellation", exit_code=128 + signum
                            )
                        else:
                            result.update(reason="output_validation", exit_code=65)
                except ResourceMismatch as error:
                    result.update(reason="resource_mismatch", exit_code=78, error=str(error))
                except (OSError, ValueError, RuntimeError) as error:
                    result["error"] = f"{type(error).__name__}: {error}"
                result["finished_at"] = utc_now()
                atomic_json(attempt_dir / "result.json", result)
                if result["status"] == "succeeded":
                    atomic_json(
                        completion_path,
                        {
                            "schema_version": 1,
                            "task_id": task_id,
                            "campaign_id": campaign_id,
                            "task_fingerprint": task_fp,
                            "attempt_id": attempt_id,
                            "result_path": str(attempt_dir / "result.json"),
                            "completed_at": result["finished_at"],
                            "validation": result["validation"],
                        },
                    )
                return result["exit_code"]
    except BlockingIOError:
        return 75


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    raise SystemExit(run_task(args.state_dir, args.campaign, args.task))


if __name__ == "__main__":
    main()
