"""Small, fail-closed Slurm CLI adapter; importing it never contacts Slurm.

The caller must durably record a submission intent before ``submit``. An UNKNOWN
result is a reconciliation obligation, never permission to resubmit.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import getpass
import json
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal

TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)
ACTIVE_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "SUSPENDED",
        "RESIZING",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "SPECIAL_EXIT",
        "STOPPED",
    }
)
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
JOB_RE = re.compile(r"^\d+(?:_(?:\d+|\[[\d,%:\-]+\]))?(?:\+\d+)?$")


@dataclasses.dataclass
class CommandResult:
    argv: list[str]
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    launch_error: bool = False


@dataclasses.dataclass
class SlurmJob:
    job_id: str
    name: str
    user: str
    state: str
    partition: str
    account: str = ""
    qos: str = ""
    comment: str = ""
    array_job_id: str | None = None
    array_task_id: int | None = None
    array_task_expression: str | None = None
    reason: str = ""
    nodes: str = ""
    cpus: int | None = None
    memory_mib: int | None = None
    gpus: int | None = None
    submit_time: int | None = None
    start_time: int | None = None
    end_time: int | None = None
    elapsed_seconds: float | None = None
    time_limit_seconds: int | None = None
    exit_code: str | None = None
    source: str = "squeue"
    array_task_count: int = 1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


@dataclasses.dataclass
class SubmissionResult:
    status: Literal["accepted", "rejected", "unknown"]
    job_id: str | None
    token: str
    job_name: str
    argv: list[str]
    stdout: str
    stderr: str
    returncode: int | None
    reason: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class SlurmError(RuntimeError):
    def __init__(self, message: str, result: CommandResult | None = None):
        super().__init__(message)
        self.result = result


def run_command(argv: list[str], timeout: float = 30) -> CommandResult:
    env = dict(os.environ, LC_ALL="C", TZ="UTC", SLURM_BITSTR_LEN="0")
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, check=False)
        return CommandResult(argv, r.returncode, r.stdout, r.stderr)
    except subprocess.TimeoutExpired as e:

        def text(v):
            return v.decode(errors="replace") if isinstance(v, bytes) else v or ""

        return CommandResult(argv, None, text(e.stdout), text(e.stderr), True)
    except OSError as e:
        # A missing local executable is known not to have submitted anything.
        return CommandResult(argv, 127, "", str(e), launch_error=True)


def number(value, default=None):
    """Slurm data-parser numeric wrappers distinguish unset/infinite from zero."""
    if isinstance(value, dict):
        if not value.get("set", True) or value.get("infinite", False):
            return default
        return value.get("number", default)
    return default if value is None else value


def _state(value) -> str:
    if isinstance(value, list):
        # Flags such as COMPLETING may follow the base state.
        return str(value[0]).upper() if value else "UNKNOWN"
    return str(value or "UNKNOWN").split()[0].rstrip("+").upper()


def _time(value) -> int | None:
    value = number(value)
    if not value or value in {"Unknown", "N/A", "None"}:
        return None
    if isinstance(value, (int, float)) or str(value).isdigit():
        return int(value)
    try:
        parsed = dt.datetime.fromisoformat(str(value))
        return int(parsed.replace(tzinfo=parsed.tzinfo or dt.UTC).timestamp())
    except ValueError:
        return None


def duration(value) -> int | None:
    if value in {None, "", "UNLIMITED", "NOT_SET", "N/A", "Unknown", "INVALID"}:
        return None
    if isinstance(value, (int, float)) or str(value).isdigit():
        return int(value)
    text = str(value)
    days, _, rest = text.partition("-")
    total = int(days) * 86400 if rest else 0
    parts = [int(x) for x in (rest or days).split(":")]
    for index, n in enumerate(reversed(parts)):
        total += n * (60**index)
    return total


def _memory(value) -> int | None:
    if not value:
        return None
    m = re.fullmatch(r"([\d.]+)([KMGTPE]?)(?:[cn])?", str(value), re.IGNORECASE)
    if not m:
        return None
    scale = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024**2, "P": 1024**3, "E": 1024**4}
    return int(float(m[1]) * scale[m[2].upper()])


def _tres(text: str) -> tuple[int | None, int | None, int | None]:
    fields = dict(x.split("=", 1) for x in (text or "").split(",") if "=" in x)
    # Generic total avoids double-counting generic and typed GPU TRES.
    gpu = fields.get("gres/gpu")
    if gpu is None:
        typed = [int(v) for k, v in fields.items() if k.startswith("gres/gpu:")]
        gpu = sum(typed) if typed else 0
    return int(fields["cpu"]) if "cpu" in fields else None, _memory(fields.get("mem")), int(gpu)


def _exit(value) -> str | None:
    if not isinstance(value, dict):
        return str(value) if value else None
    code = number(value.get("return_code"))
    sig = number(value.get("signal", {}).get("id"), 0)
    return f"{code}:{sig}" if code is not None else None


def parse_queue_job(row: dict) -> SlurmJob:
    aid = number(row.get("array_job_id"), 0) or 0
    task = number(row.get("array_task_id"))
    expression = row.get("array_task_string") or None
    jid = str(row["job_id"])
    if aid and task is not None:
        jid = f"{aid}_{task}"
    elif aid and expression:
        jid = f"{aid}_[{expression}]"
    cpus, memory, gpus = _tres(row.get("tres_alloc_str") or row.get("tres_req_str", ""))
    mem_node = number(row.get("memory_per_node"))
    mem_cpu = number(row.get("memory_per_cpu"))
    cpu_count = number(row.get("cpus"), cpus)
    return SlurmJob(
        job_id=jid,
        name=row.get("name", ""),
        user=row.get("user_name", ""),
        state=_state(row.get("job_state")),
        partition=row.get("partition", ""),
        account=row.get("account", ""),
        qos=row.get("qos", ""),
        comment=row.get("comment", "") or "",
        array_job_id=str(aid) if aid else None,
        array_task_id=task,
        array_task_expression=expression,
        reason=row.get("state_reason", "") or "",
        nodes=row.get("nodes", "") or "",
        cpus=cpu_count,
        memory_mib=memory or mem_node or (mem_cpu * cpu_count if mem_cpu and cpu_count else None),
        gpus=gpus,
        submit_time=_time(row.get("submit_time")),
        start_time=_time(row.get("start_time")),
        end_time=_time(row.get("end_time")),
        time_limit_seconds=(
            number(row.get("time_limit")) * 60 if number(row.get("time_limit")) is not None else None
        ),
        exit_code=_exit(row.get("exit_code")),
        source="squeue",
    )


def parse_accounting_job(row: dict) -> SlurmJob:
    array = row.get("array", {})
    aid = number(array.get("job_id"), 0) or 0
    task = number(array.get("task_id"))
    jid = f"{aid}_{task}" if aid and task is not None else str(row["job_id"])
    tres = row.get("tres", {}).get("allocated") or row.get("tres", {}).get("requested", [])
    ttext = ",".join(
        f"{r['type']}{('/' + r['name']) if r.get('name') else ''}={r['count']}"
        for r in tres
        if r.get("count", -1) >= 0
    )
    cpus, memory, gpus = _tres(ttext)
    time = row.get("time", {})
    state = row.get("state", {})
    comment = row.get("comment", {})
    return SlurmJob(
        job_id=jid,
        name=row.get("name", ""),
        user=row.get("user", ""),
        state=_state(state.get("current") if isinstance(state, dict) else state),
        partition=row.get("partition", ""),
        account=row.get("account", ""),
        qos=row.get("qos", ""),
        comment=comment.get("job", "") if isinstance(comment, dict) else comment or "",
        array_job_id=str(aid) if aid else None,
        array_task_id=task,
        array_task_expression=array.get("task") or None,
        reason=state.get("reason", "") if isinstance(state, dict) else "",
        nodes=row.get("nodes", "") or "",
        cpus=cpus or row.get("required", {}).get("CPUs"),
        memory_mib=memory,
        gpus=gpus,
        submit_time=_time(time.get("submission")),
        start_time=_time(time.get("start")),
        end_time=_time(time.get("end")),
        elapsed_seconds=number(time.get("elapsed")),
        time_limit_seconds=number(time.get("limit")) * 60 if number(time.get("limit")) is not None else None,
        exit_code=_exit(row.get("exit_code")),
        source="sacct",
    )


def matches_submission(job: SlurmJob, token: str, user: str) -> bool:
    """Exact durable identity, including sites that omit comments from accounting."""
    if not TOKEN_RE.fullmatch(token) or job.user != user:
        return False
    return job.comment == f"hhq:{token}" or (not job.comment and job.name.endswith(f"-{token}"))


def _array_ranges(expression: str) -> list[range]:
    expression = expression.strip("[]").split("%", 1)[0]
    ranges = []
    for part in expression.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+)(?::(\d+))?)?", part)
        if not match:
            raise SlurmError(f"Unparseable or truncated Slurm array expression: {expression}")
        first, last, step = int(match[1]), int(match[2] or match[1]), int(match[3] or 1)
        if last < first or step < 1:
            raise SlurmError("Invalid array range")
        ranges.append(range(first, last + 1, step))
    return ranges


def _expand_jobs(jobs: list[SlurmJob], expand: bool) -> list[SlurmJob]:
    result = []
    for job in jobs:
        if not job.array_job_id or job.array_task_id is not None or not job.array_task_expression:
            result.append(job)
            continue
        ranges = _array_ranges(job.array_task_expression)
        count = sum(len(segment) for segment in ranges)
        if expand and count <= 10000:
            for segment in ranges:
                for index in segment:
                    result.append(
                        dataclasses.replace(
                            job,
                            job_id=f"{job.array_job_id}_{index}",
                            array_task_id=index,
                            array_task_expression=None,
                            array_task_count=1,
                        )
                    )
        else:
            # Bound controller RAM while preserving the exact scheduler footprint.
            expr = job.array_task_expression.strip("[]").split("%", 1)[0]
            result.append(
                dataclasses.replace(job, job_id=f"{job.array_job_id}_[{expr}]", array_task_count=count)
            )
    return result


def _filter_jobs(jobs: list[SlurmJob], ids) -> list[SlurmJob]:
    if ids is None:
        return jobs
    wanted = {str(i) for i in ids}
    result = []
    for job in jobs:
        if job.job_id in wanted or job.array_job_id in wanted:
            result.append(job)
        elif job.array_task_expression and job.array_job_id:
            ranges = _array_ranges(job.array_task_expression)
            for jid in wanted:
                match = re.fullmatch(r"(\d+)_(\d+)", jid)
                if (
                    match
                    and match[1] == job.array_job_id
                    and any(int(match[2]) in segment for segment in ranges)
                ):
                    result.append(
                        dataclasses.replace(
                            job,
                            job_id=jid,
                            array_task_id=int(match[2]),
                            array_task_expression=None,
                            array_task_count=1,
                        )
                    )
    return result


def _utc(value) -> dt.datetime:
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value, dt.UTC)
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value)
    if not isinstance(value, dt.datetime):
        raise TypeError("An explicit datetime, ISO timestamp, or epoch is required")
    return value.replace(tzinfo=value.tzinfo or dt.UTC).astimezone(dt.UTC)


class SlurmClient:
    def __init__(self, user: str | None = None, runner: Callable | None = None, timeout: float = 30):
        self.user = user or getpass.getuser()
        self.runner = runner or run_command
        self.timeout = timeout
        self.json_capabilities: dict[str, bool] = {}

    def _run(self, argv: list[str]) -> CommandResult:
        result = self.runner(argv, self.timeout)
        return result

    def _checked(self, argv: list[str]) -> CommandResult:
        result = self._run(argv)
        if result.returncode != 0 or result.timed_out:
            raise SlurmError(f"{argv[0]} query failed: {result.stderr[-2000:]}", result)
        return result

    def _json(self, argv: list[str]) -> dict | None:
        if self.json_capabilities.get(argv[0]) is False:
            return None
        result = self._run(argv)
        unsupported = re.search(
            r"(?:unrecognized|unknown|invalid) (?:option|argument).*json|JSON.*(?:not supported|unavailable)|unable to find.*data_parser|unable to load.*data_parser",
            result.stderr,
            re.IGNORECASE,
        )
        if result.returncode != 0 and unsupported:
            self.json_capabilities[argv[0]] = False
            return None
        if result.returncode != 0 or result.timed_out:
            raise SlurmError(f"{argv[0]} query failed: {result.stderr[-2000:]}", result)
        try:
            obj = json.loads(result.stdout)
        except (ValueError, TypeError) as e:
            raise SlurmError(f"Malformed {argv[0]} JSON; no empty-state assumption", result) from e
        if not isinstance(obj, dict) or obj.get("errors"):
            raise SlurmError(
                f"{argv[0]} returned JSON errors: {obj.get('errors') if isinstance(obj, dict) else 'not an object'}",
                result,
            )
        self.json_capabilities[argv[0]] = True
        return obj

    @staticmethod
    def _ids(ids) -> list[str]:
        items = [str(i) for i in ids or []]
        if any(not JOB_RE.fullmatch(i) for i in items):
            raise ValueError("Invalid Slurm job ID")
        return items

    def query_jobs(self, ids=None, user: str | None = None, expand_arrays: bool = True) -> list[SlurmJob]:
        if ids is not None and not ids:
            return []
        args = ["--user", user or self.user]
        if expand_arrays:
            args.append("--array")
        if ids is not None:
            self._ids(ids)
        # Query the complete user queue then filter: Slurm rejects expired IDs
        # rather than returning an empty result for --jobs. RPC errors stay errors.
        obj = self._json(["squeue", "--json", *args])
        if obj is not None:
            if not isinstance(obj.get("jobs"), list):
                raise SlurmError("squeue JSON is missing its jobs array")
            return _filter_jobs(
                _expand_jobs([parse_queue_job(row) for row in obj["jobs"]], expand_arrays), ids
            )
        # Unit separator prevents ordinary commas/spaces in job names/comments
        # from corrupting columns. This fallback requests all ownership fields.
        sep = "\x1f"
        fmt = sep.join(
            [
                "%i",
                "%j",
                "%u",
                "%T",
                "%P",
                "%a",
                "%q",
                "%k",
                "%r",
                "%N",
                "%C",
                "%m",
                "%V",
                "%S",
                "%e",
                "%l",
                "%b",
            ]
        )
        result = self._checked(["squeue", "--noheader", "--format", fmt, *args])
        jobs = []
        for line in result.stdout.splitlines():
            fields = line.split(sep)
            if len(fields) != 17:
                raise SlurmError("Malformed squeue delimited output", result)
            (
                jid,
                name,
                owner,
                state,
                partition,
                account,
                qos,
                comment,
                reason,
                nodes,
                cpus,
                mem,
                submit,
                start,
                end,
                limit,
                gres,
            ) = fields
            match = re.fullmatch(r"(\d+)_(\d+|\[.*\])", jid)
            gpu_counts = [int(m[1]) for m in re.finditer(r"(?:^|,)gpu(?::[^,:()]+)?:(\d+)", gres)]
            jobs.append(
                SlurmJob(
                    jid,
                    name,
                    owner,
                    _state(state),
                    partition,
                    account,
                    qos,
                    comment,
                    array_job_id=match[1] if match else None,
                    array_task_id=int(match[2]) if match and match[2].isdigit() else None,
                    array_task_expression=match[2][1:-1] if match and match[2].startswith("[") else None,
                    reason=reason,
                    nodes=nodes,
                    cpus=int(cpus) if cpus.isdigit() else None,
                    memory_mib=_memory(mem),
                    gpus=sum(gpu_counts),
                    submit_time=_time(submit),
                    start_time=_time(start),
                    end_time=_time(end),
                    time_limit_seconds=duration(limit),
                )
            )
        return _filter_jobs(_expand_jobs(jobs, expand_arrays), ids)

    def accounting(self, ids=None, since=None, until=None, max_window_days: int = 7) -> list[SlurmJob]:
        if since is None:
            raise ValueError("Accounting requires an explicit bounded start time")
        start, end = _utc(since), _utc(until or dt.datetime.now(dt.UTC))
        if (
            end <= start
            or end - start > dt.timedelta(days=max_window_days, seconds=1)
            or not 0 < max_window_days <= 31
        ):
            raise ValueError(
                "Accounting window must be positive and at most max_window_days (hard maximum 31)"
            )
        if ids is not None and not ids:
            return []
        args = [
            "--allocations",
            "--user",
            self.user,
            "--starttime",
            start.strftime("%Y-%m-%dT%H:%M:%S"),
            "--endtime",
            end.strftime("%Y-%m-%dT%H:%M:%S"),
        ]
        if ids is not None:
            args += ["--jobs", ",".join(self._ids(ids))]
        obj = self._json(["sacct", "--json", *args])
        if obj is not None:
            if not isinstance(obj.get("jobs"), list):
                raise SlurmError("sacct JSON is missing its jobs array")
            return [parse_accounting_job(row) for row in obj["jobs"]]
        fmt = "JobID,JobName%128,User,State,Partition,Account,QOS,Comment%256,NodeList,AllocCPUS,AllocTRES,Submit,Start,End,ElapsedRaw,TimelimitRaw,ExitCode"
        result = self._checked(["sacct", "--noheader", "--parsable2", "--format", fmt, *args])
        jobs = []
        for line in result.stdout.splitlines():
            f = line.split("|")
            if len(f) != 17:
                raise SlurmError("Malformed sacct delimited output", result)
            (
                jid,
                name,
                owner,
                state,
                part,
                account,
                qos,
                comment,
                nodes,
                cpu,
                tres,
                submit,
                start,
                end,
                elapsed,
                limit,
                exitcode,
            ) = f
            cpus, mem, gpus = _tres(tres)
            match = re.fullmatch(r"(\d+)_(\d+)", jid)
            jobs.append(
                SlurmJob(
                    jid,
                    name,
                    owner,
                    _state(state),
                    part,
                    account,
                    qos,
                    comment,
                    array_job_id=match[1] if match else None,
                    array_task_id=int(match[2]) if match else None,
                    nodes=nodes,
                    cpus=cpus or (int(cpu) if cpu.isdigit() else None),
                    memory_mib=mem,
                    gpus=gpus,
                    submit_time=_time(submit),
                    start_time=_time(start),
                    end_time=_time(end),
                    elapsed_seconds=float(elapsed) if elapsed.isdigit() else None,
                    time_limit_seconds=int(limit) * 60 if limit.isdigit() else None,
                    exit_code=exitcode,
                    source="sacct",
                )
            )
        return jobs

    def submit(
        self, script: str | Path, token: str, job_name: str | None = None, extra_args=()
    ) -> SubmissionResult:
        if not TOKEN_RE.fullmatch(token):
            raise ValueError("Submission token must contain 8–64 ASCII alphanumerics, underscores or hyphens")
        name = job_name or f"hhq-{token}"
        if job_name and not name.endswith(f"-{token}"):
            name = f"{name[: 127 - len(token)]}-{token}"
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name):
            raise ValueError("Unsafe job name")
        # Never let an escape hatch replace our durable identity or script.
        for arg in extra_args:
            if str(arg).startswith("-J") or str(arg).split("=", 1)[0] in {
                "--comment",
                "--job-name",
                "-J",
                "--wrap",
                "--parsable",
            }:
                raise ValueError("extra_args cannot override submission identity")
        script = str(Path(script).expanduser().resolve(strict=True))
        argv = [
            "sbatch",
            "--parsable",
            "--comment",
            f"hhq:{token}",
            "--job-name",
            name,
            *map(str, extra_args),
            script,
        ]
        result = self._run(argv)
        match = re.fullmatch(r"\s*(\d+)(?:;[^\s;]+)?\s*", result.stdout)
        if result.returncode == 0 and not result.timed_out and match:
            return SubmissionResult(
                "accepted",
                match[1],
                token,
                name,
                argv,
                result.stdout,
                result.stderr,
                result.returncode,
                "Slurm acknowledged job ID; execution is not yet confirmed",
            )
        known_rejection = re.search(
            r"Batch job submission failed: (?:Invalid |Requested node configuration is not available|Job violates accounting/QOS policy|Access/permission denied|Job submit/allocate failed|Socket timed out).*",
            result.stderr,
        )
        # A transport timeout can follow acceptance, even when prefixed 'failed'.
        if known_rejection and "Socket timed out" in known_rejection[0]:
            known_rejection = None
        locally_invalid = re.search(
            r"sbatch: (?:error: )?(?:unrecognized option|invalid option|Unable to open file|This does not look like a batch script)",
            result.stderr,
        )
        rejected = not result.timed_out and (bool(known_rejection or locally_invalid) or result.launch_error)
        return SubmissionResult(
            "rejected" if rejected else "unknown",
            None,
            token,
            name,
            argv,
            result.stdout,
            result.stderr,
            result.returncode,
            "Explicit rejection before allocation"
            if rejected
            else "Acceptance ambiguous; reconcile token in squeue and bounded sacct before any action",
        )

    def find_submission(self, token: str, since) -> list[SlurmJob]:
        if not TOKEN_RE.fullmatch(token):
            raise ValueError("Invalid submission token")
        # Always query accounting as well: empty live queue does not prove absence.
        live = self.query_jobs()
        historical = self.accounting(since=since)
        found = {j.job_id: j for j in historical if matches_submission(j, token, self.user)}
        found.update({j.job_id: j for j in live if matches_submission(j, token, self.user)})
        return list(found.values())

    def cancel(self, job_ids, expected_token: str) -> CommandResult:
        if not TOKEN_RE.fullmatch(expected_token):
            raise ValueError("Invalid submission token")
        ids = self._ids(job_ids)
        if not ids:
            raise ValueError("Cancellation needs explicit job IDs")
        live = self.query_jobs(expand_arrays=True)
        for jid in ids:
            matching = _filter_jobs(live, [jid])
            if not matching or any(
                j.user != self.user or j.comment != f"hhq:{expected_token}" for j in matching
            ):
                raise SlurmError(f"Refusing cancellation: ownership not proven for {jid}")
        return self._checked(["scancel", *ids])
