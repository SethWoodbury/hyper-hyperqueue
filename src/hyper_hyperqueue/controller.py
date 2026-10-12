"""Small reconciler. HQ dispatches tasks; this module only supplies allocations."""

from __future__ import annotations

import argparse
import dataclasses
import json
import shlex
import sys
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .backends.direct import resource_args, submit_ready
from .config import load_config
from .models import Config, TaskSpec
from .planner import capacity_plan
from .runner import attempt_epoch, authorize_retry, internal_fencing_evidence
from .state import StateStore, atomic_json

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}


def terminal_state(value):
    return str(value).split("+")[0].split()[0]


def owned_record(record, intent):
    """Positive scheduler identity, not merely a job number reused after retention."""
    token = intent["token"]
    return record.get("comment") == f"hhq:{token}" or (
        not record.get("comment") and record.get("name", "").endswith(f"-{token}")
    )


def mapping(value):
    return value if isinstance(value, dict) else dataclasses.asdict(value)


def now():
    return datetime.now(UTC).isoformat()


def assert_epoch(store, epoch):
    if epoch is None:
        return
    from .service.leadership import assert_epoch as guarded_epoch

    guarded_epoch(store.root / "service", epoch)


def reconcile_slurm(store, slurm, jobs):
    by_base = {}
    for j in jobs:
        d = mapping(j)
        by_base.setdefault(str(d.get("array_job_id") or d["job_id"]).split("_")[0], []).append(d)
    active = [
        i
        for i in store.intents()
        if i.get("backend") in ("direct", "worker")
        and i.get("state") in ("intent", "unknown", "submitted", "running")
    ]
    absent = [str(i["job_id"]) for i in active if i.get("job_id") and str(i["job_id"]) not in by_base]
    accounting = []
    if absent:
        accounting = [
            mapping(j)
            for j in slurm.accounting(ids=absent, since=datetime.now(UTC) - timedelta(days=6, hours=23))
        ]
    for intent in active:
        if not intent.get("job_id"):
            matches = slurm.find_submission(
                intent["token"], since=datetime.now(UTC) - timedelta(days=6, hours=23)
            )
            parents = {
                str(mapping(item).get("array_job_id") or mapping(item)["job_id"]).split("_")[0]
                for item in matches
            }
            if len(parents) == 1:
                intent.update(
                    job_id=parents.pop(),
                    state="submitted",
                    job_name=mapping(matches[0]).get("name", intent.get("job_name")),
                )
                store.put_intent(intent)
            elif len(parents) > 1:
                raise RuntimeError(f"DUPLICATE_SUBMISSION_IDENTITY {intent['token']}")
            continue  # absence is uncertainty, never permission to resubmit
        jid = str(intent["job_id"])
        if jid in by_base:
            intent["state"] = "running" if any(j["state"] == "RUNNING" for j in by_base[jid]) else "submitted"
            timed = [j for j in by_base[jid] if j.get("time_limit_seconds")]
            if timed:
                current_time = time.time()
                # Pending start times are Slurm forecasts, not extra allocation
                # lifetime. Neither pending nor running work gains time beyond
                # its declared limit when Slurm reports a future start.
                intent["remaining_seconds"] = min(
                    max(
                        0,
                        min(
                            j["time_limit_seconds"], j["start_time"] + j["time_limit_seconds"] - current_time
                        ),
                    )
                    if j["state"] in {"RUNNING", "COMPLETING"} and j.get("start_time")
                    else j["time_limit_seconds"]
                    for j in timed
                )
            store.put_intent(intent)
            continue
        records = [j for j in accounting if str(j.get("array_job_id") or j["job_id"]).split("_")[0] == jid]
        if not records or not all(
            terminal_state(j["state"]) in TERMINAL and owned_record(j, intent) for j in records
        ):
            continue
        intent.update(state="terminal", terminal_evidence=records, terminal_at=now())
        store.put_intent(intent)
        if intent["backend"] != "direct":
            continue
        for tid in intent["task_ids"]:
            current = store.task_state(intent["campaign_id"], tid)
            if current.get("state") in ("succeeded", "failed", "interrupted", "canceled"):
                continue
            # An array sibling's cancellation does not cancel this task. Bind
            # individual slots when available; absent identity remains unknown.
            attempt = current.get("latest_attempt", {})
            environment = attempt.get("context", {}).get("environment", {})
            slot = environment.get("SLURM_ARRAY_TASK_ID")
            applicable = (
                [j for j in records if str(j.get("array_task_id")) == str(slot)] if slot is not None else []
            )
            cancellation = bool(applicable) and all(
                terminal_state(j["state"]) == "CANCELLED" for j in applicable
            )
            store.update_task_state(
                intent["campaign_id"],
                tid,
                {
                    "state": "canceled" if cancellation else "unknown",
                    "reason": "SLURM_CANCELLED" if cancellation else "TERMINAL_ALLOCATION_WITHOUT_COMPLETION",
                    "terminal_evidence": applicable or records,
                },
            )


def runner_argv(store, task):
    return [
        sys.executable,
        "-m",
        "hyper_hyperqueue.runner",
        "--state-dir",
        str(store.root),
        "--campaign",
        task.campaign_id,
        "--task",
        task.task_id,
    ]


def admit_hq(store, config, hq, campaign, epoch):
    if campaign.get("draining") or campaign.get("pause_dispatch"):
        return
    existing = [
        i for i in store.intents() if i.get("backend") == "hq" and i["campaign_id"] == campaign["campaign_id"]
    ]
    if any(i.get("state") in ("intent", "unknown") for i in existing):
        return False
    assigned = {
        tid
        for i in existing
        if i.get("state") in ("intent", "unknown", "submitted")
        for tid in i["task_ids"]
        if tid not in i.get("retired_task_ids", [])
    }
    tasks = []
    for tid in store.task_ids(campaign["campaign_id"]):
        if tid in assigned:
            continue
        if store.task_state(campaign["campaign_id"], tid).get("state", "ready") in ("ready", "retry_wait"):
            task = store.task(campaign["campaign_id"], tid)
            tasks.append(task if isinstance(task, TaskSpec) else TaskSpec.model_validate(task))
    if not tasks:
        return
    # Bounded submissions and durable identity; append creates another HQ job on
    # the same server/backlog, without repartitioning old tasks or workers.
    for offset in range(0, min(len(tasks), config.policy.max_submissions_per_cycle * 1000), 1000):
        batch = tasks[offset : offset + 1000]
        token = uuid.uuid4().hex
        intent = {
            "token": token,
            "backend": "hq",
            "campaign_id": campaign["campaign_id"],
            "task_ids": [t.task_id for t in batch],
            "state": "intent",
            "epoch": epoch,
            "created_at": now(),
        }
        store.put_intent(intent)
        assert_epoch(store, epoch)
        try:
            result = hq.submit(
                batch,
                token,
                lambda t: runner_argv(store, t),
                epoch=epoch,
                default_runtime_seconds=config.policy.unknown_runtime_seconds,
            )
        except Exception as exc:  # noqa: BLE001 -- any lost acknowledgement is uncertain, never replayed
            intent.update(state="unknown", error=str(exc))
            store.put_intent(intent)
            store.event("hq_submission_unknown", token=token, error=str(exc))
            return False
        intent.update(state="submitted", hq_job_id=result["job_id"], hq_task_ids=result["tasks"])
        store.put_intent(intent)
        for task in batch:
            store.update_task_state(
                task.campaign_id,
                task.task_id,
                {
                    "state": "submitted",
                    "hq_job_id": result["job_id"],
                    "hq_task_id": result["tasks"][task.task_id],
                    "submission_token": token,
                    "server_epoch": epoch,
                },
            )


def reconcile_hq(store, hq):
    for intent in [
        i
        for i in store.intents()
        if i.get("backend") == "hq" and i.get("state") in ("intent", "unknown", "submitted")
    ]:
        if not intent.get("hq_job_id"):
            found = hq.find_intent(intent["token"])
            if found:
                intent.update(
                    state="submitted",
                    hq_job_id=found["id"],
                    hq_task_ids={t: i for i, t in enumerate(intent["task_ids"])},
                )
                store.put_intent(intent)
            continue
        states = [
            "retired"
            if t in intent.get("retired_task_ids", [])
            else store.task_state(intent["campaign_id"], t).get("state", "unknown")
            for t in intent["task_ids"]
        ]
        if all(s in ("succeeded", "failed", "interrupted", "canceled", "blocked", "retired") for s in states):
            intent.update(state="terminal", terminal_at=now())
            store.put_intent(intent)
            continue
        info = hq.tasks(intent["hq_job_id"])
        if isinstance(info, dict):
            info = info.get(str(intent["hq_job_id"]), info.get("tasks", []))
        for item in info:
            status = item.get("state", item.get("status", ""))
            if isinstance(status, dict):
                status = status.get("state", status.get("type", ""))
            if str(status).lower() not in ("failed", "canceled", "cancelled"):
                continue
            task_index = int(item.get("id", item.get("task_id", -1)))
            if 0 <= task_index < len(intent["task_ids"]):
                tid = intent["task_ids"][task_index]
                if tid in intent.get("retired_task_ids", []):
                    continue
                current = store.task_state(intent["campaign_id"], tid)
                if current.get("submission_token") not in (None, intent["token"]):
                    continue
                if current.get("state") not in ("succeeded", "failed", "canceled", "interrupted"):
                    canceled = str(status).lower() in ("canceled", "cancelled")
                    change = {
                        "state": "canceled" if canceled else "interrupted",
                        "reason": "user_cancelled" if canceled else "worker_lost",
                        "hq_terminal_evidence": item,
                        "server_epoch": intent.get("epoch"),
                    }
                    if current.get("latest_attempt"):
                        change["after_attempt_id"] = current["latest_attempt"]["attempt_id"]
                    store.update_task_state(intent["campaign_id"], tid, change)


def retry_ready_failures(store, config, campaigns):
    """One retry owner. Resultless work requires exact terminal Slurm ownership."""
    decisions = []
    intents = store.intents()
    for campaign in campaigns:
        if campaign.get("draining") or campaign.get("pause_dispatch"):
            continue
        cid = campaign["campaign_id"]
        for tid in store.task_ids(cid):
            state = store.task_state(cid, tid)
            if state.get("state") not in ("failed", "interrupted", "unknown", "canceled"):
                continue
            task = store.task(cid, tid)
            attempt = state.get("latest_attempt")
            if not attempt:
                continue  # No task execution identity: operator/backend reconciliation required.
            reason = attempt.get("result", {}).get("reason") or state.get("reason")
            proof_path = None
            old_epoch = attempt_epoch(attempt)
            service_proof = store.root / "service" / f"recovery-{old_epoch}.json"
            internal = (
                internal_fencing_evidence(store, task, attempt, service_proof)
                if old_epoch is not None and service_proof.exists()
                else None
            )
            if state.get("state") == "canceled" and not internal:
                continue
            if internal:
                proof_path = service_proof
                reason = "worker_lost"
            elif "result" not in attempt:
                env = attempt.get("context", {}).get("environment", {})
                jid = env.get("SLURM_ARRAY_JOB_ID") or env.get("SLURM_JOB_ID")
                exact = [
                    i
                    for i in intents
                    if i.get("backend") in ("worker", "direct")
                    and str(i.get("job_id")) == str(jid)
                    and i.get("state") == "terminal"
                ]
                if len(exact) != 1:
                    continue
                allocation = exact[0]
                if allocation.get("backend") == "worker" and str(allocation.get("epoch")) != old_epoch:
                    continue
                records = allocation.get("terminal_evidence", [])
                if not records or not all(
                    owned_record(r, allocation) and terminal_state(r["state"]) in TERMINAL for r in records
                ):
                    continue
                if any(terminal_state(r["state"]) == "CANCELLED" for r in records):
                    continue
                terminal_states = {terminal_state(r["state"]) for r in records}
                reason = (
                    "host_oom"
                    if "OUT_OF_MEMORY" in terminal_states
                    else "timeout"
                    if "TIMEOUT" in terminal_states
                    else "preempted"
                    if "PREEMPTED" in terminal_states
                    else "worker_lost"
                )
                if reason not in task.retry.reasons:
                    continue
                proof_path = store.root / "recovery" / f"{attempt['attempt_id']}-{tid}.json"
                atomic_json(
                    proof_path,
                    {
                        "schema_version": 1,
                        "quiescent": True,
                        "previous_epoch": old_epoch,
                        "attempt_id": attempt["attempt_id"],
                        "campaign_id": cid,
                        "task_id": tid,
                        "allocation_token": allocation["token"],
                        "terminal_evidence": records,
                        "verified_at": now(),
                        "basis": "exact owned allocation absent from live query and positively terminal in accounting",
                    },
                )
            try:
                authorize_retry(store, cid, tid, terminal_evidence=proof_path)
            except ValueError as exc:
                decisions.append({"task_id": tid, "decision": "retry_blocked", "reason": str(exc)})
                continue
            # The old HQ job may contain unrelated long-running siblings. Retire
            # only this exact terminal attempt rather than wait for the whole job.
            for intent in intents:
                if (
                    intent.get("backend") == "hq"
                    and intent.get("campaign_id") == cid
                    and tid in intent.get("task_ids", [])
                ):
                    retired = set(intent.get("retired_task_ids", []))
                    retired.add(tid)
                    intent["retired_task_ids"] = sorted(retired)
                    store.put_intent(intent)
            decisions.append({"task_id": tid, "decision": "retry_authorized", "reason": reason})
    return decisions


def allocation_gates(store, config, owned, *, clock=None, reset_after=None):
    """Bounded startup failure circuit plus per-profile submission cooldown."""
    clock = time.time() if clock is None else clock
    reset_after = reset_after or {}
    paused, reasons = set(), []
    for name in config.profiles:
        history = sorted(
            (
                i
                for i in owned
                if i.get("profile") == name and i.get("created_at", "") >= reset_after.get(name, "")
            ),
            key=lambda i: i.get("created_at", ""),
        )
        failures = 0
        for intent in reversed(history):
            if intent.get("state") in ("intent", "unknown", "submitted", "running"):
                continue
            if intent.get("state") == "rejected":
                failures += 1
                continue
            records = intent.get("terminal_evidence", [])
            if intent.get("state") != "terminal" or not records:
                continue
            duration = max((r.get("elapsed_seconds") or 0 for r in records), default=0)
            receipt = store.root / "service/workers" / f"{intent.get('job_id')}.json"
            started = False
            if receipt.exists():
                item = json.loads(receipt.read_text())
                started = item.get("allocation_token") == intent["token"] and item.get("epoch") == intent.get(
                    "epoch"
                )
            states = {terminal_state(r["state"]) for r in records}
            if states & {"CANCELLED", "PREEMPTED", "NODE_FAIL"}:
                continue  # User actions/site preemption do not prove a broken startup configuration.
            if not started or (states - {"COMPLETED", "TIMEOUT"} and duration < 180):
                failures += 1
            else:
                break
        if failures >= config.policy.max_rpc_failures:
            paused.add(name)
            reasons.append(
                f"{name}: STARTUP_CIRCUIT_OPEN ({failures} failed starts; after repair set controls.startup_reset_after[{name!r}] to UTC reset time)"
            )
        if history:
            try:
                last = datetime.fromisoformat(history[-1]["created_at"]).timestamp()
            except (ValueError, KeyError):
                continue
            if clock - last < config.policy.submit_cooldown_seconds:
                paused.add(name)
                reasons.append(f"{name}: SUBMISSION_COOLDOWN")
    return paused, reasons


def direct_admission_gate(store, config, campaign_id):
    history = sorted(
        (i for i in store.intents() if i.get("backend") == "direct" and i.get("campaign_id") == campaign_id),
        key=lambda i: i.get("created_at", ""),
    )
    if any(i.get("state") in ("intent", "unknown") for i in history):
        return "DIRECT_SUBMISSION_UNKNOWN: reconcile exact submission before growth"
    failures = 0
    for intent in reversed(history):
        if intent.get("state") != "rejected":
            break
        failures += 1
    if failures >= config.policy.max_rpc_failures:
        return "DIRECT_SUBMISSION_CIRCUIT_OPEN: fix rejected submission and explicitly resume with CLI"
    if history and history[-1].get("created_at"):
        elapsed = time.time() - datetime.fromisoformat(history[-1]["created_at"]).timestamp()
        if elapsed < config.policy.submit_cooldown_seconds:
            return "DIRECT_SUBMISSION_COOLDOWN"
    return None


def tick(config: Config, *, epoch=None, slurm=None, hq=None):
    from .backends.slurm import SlurmClient

    store = StateStore(config.state_dir)
    slurm = slurm or SlurmClient()
    with store.lock():
        assert_epoch(store, epoch)
        jobs = slurm.query_jobs()
        reconcile_slurm(store, slurm, jobs)
        campaigns = store.campaigns()
        pooled = [c for c in campaigns if c["mode"] == "pooled"]
        if pooled:
            if epoch is None:
                raise RuntimeError("Pooled reconciliation requires the supervised leadership descriptor")
            from .backends.hyperqueue import HQBackend

            hq = hq or HQBackend(store.root / "service/hq", config.control.hq_binary)
            for request_path in sorted((store.root / "requests").glob("cancel-*.json")):
                if request_path.name.endswith(".done.json"):
                    continue
                request = json.loads(request_path.read_text())
                if request.get("action") != "cancel_hq_campaign":
                    continue
                cid = request["campaign_id"]
                for intent in store.intents():
                    if (
                        intent.get("backend") != "hq"
                        or intent.get("campaign_id") != cid
                        or intent.get("state") != "submitted"
                    ):
                        continue
                    assert_epoch(store, epoch)
                    hq.cancel_owned(intent["hq_job_id"], intent["token"])
                    for tid in intent["task_ids"]:
                        if store.task_state(cid, tid).get("state") != "succeeded":
                            store.update_task_state(
                                cid, tid, {"state": "canceled", "reason": "user_cancelled"}
                            )
                    intent["state"] = "canceled"
                    store.put_intent(intent)
                request_path.rename(request_path.with_suffix(".done.json"))
            reconcile_hq(store, hq)
        retry_decisions = retry_ready_failures(store, config, campaigns)
        ready = []
        direct_gate_reasons = []
        hq_admission_blocked = False
        for campaign in campaigns:
            if campaign["mode"] == "direct":
                if campaign.get("managed", False):
                    blocked = direct_admission_gate(store, config, campaign["campaign_id"])
                    if blocked:
                        direct_gate_reasons.append(f"{campaign['campaign_id']}: {blocked}")
                        continue
                    assert_epoch(store, epoch)
                    submit_ready(
                        store,
                        config,
                        campaign["campaign_id"],
                        slurm,
                        max_parallel=campaign.get("max_parallel"),
                        commands_per_job=campaign.get("commands_per_job", 1),
                    )
                continue
            if admit_hq(store, config, hq, campaign, epoch) is False:
                hq_admission_blocked = True
            if campaign.get("draining"):
                continue
            for tid in store.task_ids(campaign["campaign_id"]):
                state = store.task_state(campaign["campaign_id"], tid)
                if state.get("state", "ready") in ("ready", "submitted", "retry_wait"):
                    task = store.task(campaign["campaign_id"], tid)
                    ready.append(task.model_dump() if hasattr(task, "model_dump") else task)
        owned = [i for i in store.intents() if i.get("backend") == "worker"]
        controls_path = store.root / "controls.json"
        controls = json.loads(controls_path.read_text()) if controls_path.exists() else {}
        gated, gate_reasons = allocation_gates(
            store, config, owned, reset_after=controls.get("startup_reset_after")
        )
        plan = capacity_plan(
            config,
            ready,
            jobs,
            owned,
            set(controls.get("paused_pools", [])) | gated,
            reservations=[
                i
                for i in store.intents()
                if i.get("backend") in ("worker", "direct")
                and i.get("state") in ("intent", "unknown", "submitted", "running")
            ],
        )
        plan["reasons"].extend(gate_reasons + direct_gate_reasons)
        if hq_admission_blocked:
            plan["actions"] = []
            plan["reasons"].append("HQ_SUBMISSION_UNKNOWN: reconcile exact intent before new worker growth")
        config_path = store.root / "resolved_config.json"
        atomic_json(config_path, config.model_dump())
        for action in plan["actions"]:
            assert_epoch(store, epoch)
            profile = config.profiles[action["profile"]]
            token = uuid.uuid4().hex
            directory = store.root / "submissions" / token
            directory.mkdir(parents=True)
            worker_config = directory / "config.json"
            atomic_json(worker_config, config.model_dump())
            script = directory / "worker.sh"
            argv = [
                sys.executable,
                "-m",
                "hyper_hyperqueue.worker",
                "--state-dir",
                str(store.root),
                "--config",
                str(worker_config),
                "--profile",
                action["profile"],
                "--epoch",
                str(epoch or 0),
            ]
            script.write_text(
                "#!/bin/bash\nexport HHQ_ALLOCATION_TOKEN="
                + shlex.quote(token)
                + "\nexec "
                + shlex.join(argv)
                + "\n"
            )
            extra = resource_args(profile) + [
                "--output",
                str(directory / "slurm.out"),
                "--error",
                str(directory / "slurm.err"),
            ]
            intent = {
                "token": token,
                "backend": "worker",
                "profile": action["profile"],
                "epoch": epoch,
                "placement": {"partition": profile.partition, "account": profile.account, "qos": profile.qos},
                "config_snapshot": str(worker_config),
                "state": "intent",
                "created_at": now(),
                "script": str(script),
                "extra_args": extra,
                "job_name": f"hhq-{token[:16]}",
                "reason": action["reason"],
            }
            store.put_intent(intent)
            assert_epoch(store, epoch)
            result = mapping(slurm.submit(script, token=token, job_name=intent["job_name"], extra_args=extra))
            intent.update(
                response=result,
                job_id=result.get("job_id"),
                requested_job_name=intent["job_name"],
                job_name=result.get("job_name") or intent["job_name"],
                state="submitted" if result["status"] == "accepted" else result["status"],
            )
            store.put_intent(intent)
            if result["status"] != "accepted":
                break
        status = {
            "schema_version": 1,
            "at": now(),
            "epoch": epoch,
            "state": "healthy",
            "plan": plan,
            "retry_decisions": retry_decisions,
        }
        atomic_json(store.root / "controller_status.json", status)
        return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-dir")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.state_dir:
        config.state_dir = args.state_dir
    failures = 0
    while True:
        try:
            tick(config, epoch=args.epoch)
            failures = 0
        except Exception as exc:
            failures += 1
            atomic_json(
                Path(config.state_dir) / "controller_status.json",
                {
                    "schema_version": 1,
                    "at": now(),
                    "state": "degraded",
                    "error": str(exc),
                    "consecutive_errors": failures,
                    "epoch": args.epoch,
                },
            )
            print(str(exc), file=sys.stderr, flush=True)
            if args.once:
                raise
        if args.once:
            return
        time.sleep(min(300, config.policy.reconcile_interval_seconds * max(1, 2 ** min(failures, 3))))


if __name__ == "__main__":
    main()
