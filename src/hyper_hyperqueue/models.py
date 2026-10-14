from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Resources(Model):
    cpus: int = Field(1, ge=1)
    memory_mib: int = Field(1024, ge=1)
    gpus: int = Field(0, ge=0)
    gpu_type: str | None = None
    gpu_models: list[str] = Field(default_factory=list)
    min_vram_mib: int | None = Field(None, ge=1)
    nodes: int = Field(1, ge=1)
    ranks: int = Field(1, ge=1)
    runtime_seconds: float | None = Field(None, gt=0)
    runtime_confidence: Literal["declared", "measured", "unknown"] = "unknown"
    timeout_seconds: float | None = Field(None, gt=0)


class RetryPolicy(Model):
    max_attempts: int = Field(1, ge=1, le=100)
    reasons: list[str] = Field(default_factory=lambda: ["worker_lost", "preempted"])
    oom_memory_multipliers: list[float] = Field(default_factory=list)
    # Retained for schema-1 manifest compatibility; resource-changing retries
    # require a new immutable task generation in this release.
    memory_increment_mib: int = Field(2048, ge=0, deprecated=True)
    max_memory_mib: int | None = Field(None, deprecated=True)

    @model_validator(mode="after")
    def reject_unimplemented_escalation(self):
        if self.__dict__["max_memory_mib"] is not None or self.__dict__["memory_increment_mib"] != 2048:
            raise ValueError(
                "Use oom_memory_multipliers for bounded host-memory retries; legacy increment fields are inert"
            )
        import math
        values = self.oom_memory_multipliers
        if any(not math.isfinite(v) or v <= 1 for v in values) or values != sorted(set(values)):
            raise ValueError("OOM memory multipliers must be finite, increasing, unique values > 1")
        return self

    @model_serializer(mode="wrap")
    def preserve_legacy_default(self, handler):
        value = handler(self)
        if not self.oom_memory_multipliers:
            value.pop("oom_memory_multipliers", None)
        return value


class OutputPolicy(Model):
    mode: Literal["isolated", "in_place", "manual_on_ambiguity"] = "manual_on_ambiguity"
    repeatable: bool = False
    expected_paths: list[str] = Field(default_factory=list)
    min_bytes: int = Field(1, ge=0)
    validator_argv: list[str] | None = None
    collision_key: str | None = None


class TaskSpec(Model):
    schema_version: int = 1
    task_id: str
    campaign_id: str
    command: str | None = None
    argv: list[str] | None = None
    shell: str = "/bin/bash"
    shell_options: list[str] = Field(default_factory=list)
    cwd: str
    environment: dict[str, str] = Field(default_factory=dict)
    source_index: int = Field(0, ge=0)
    source_line: int | None = None
    import_id: str
    command_digest: str
    input_fingerprint: str | None = None
    software_fingerprint: str | None = None
    resources: Resources = Field(default_factory=Resources)
    allowed_profiles: list[str] = Field(default_factory=list)
    priority: int = 0
    workload: str = "generic"
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    output: OutputPolicy = Field(default_factory=OutputPolicy)
    backend: Literal["direct", "pooled"]

    @model_validator(mode="after")
    def command_xor_argv(self):
        if (self.command is None) == (self.argv is None):
            raise ValueError("Supply exactly one of command or argv")
        if self.argv is not None and not self.argv:
            raise ValueError("argv cannot be empty")
        return self


class Profile(Model):
    partition: str
    account: str | None = None
    qos: str | None = None
    budget: str | None = None
    gres: str | None = None
    constraint: str | None = None
    exclude: list[str] = Field(default_factory=list)
    cpus: int = Field(1, ge=1)
    memory_mib: int = Field(1024, ge=1)
    allocation_seconds: int = Field(3600, ge=1)
    gpu_models: list[str] = Field(default_factory=list)
    min_vram_mib: int | None = None
    execution_mode: Literal["auto", "direct", "pooled"] = "auto"
    weight: float = Field(1.0, gt=0)
    min_workers: int = Field(0, ge=0)
    max_workers: int = Field(4, ge=0)
    extra_sbatch: list[str] = Field(default_factory=list)
    fallback_profiles: list[str] = Field(default_factory=list)
    startup_probe_argv: list[str] = Field(default_factory=list)
    startup_probe_timeout_seconds: int = Field(30, ge=1, le=300, strict=True)

    @model_validator(mode="after")
    def valid_startup_probe(self):
        if self.startup_probe_argv and (not self.startup_probe_argv[0]
                                       or any("\0" in arg for arg in self.startup_probe_argv)):
            raise ValueError("startup_probe_argv requires a nonempty executable and no NUL bytes")
        return self

    @model_serializer(mode="wrap")
    def preserve_legacy_probe_default(self, handler):
        value = handler(self)
        if not self.startup_probe_argv and self.startup_probe_timeout_seconds == 30:
            value.pop("startup_probe_argv", None)
            value.pop("startup_probe_timeout_seconds", None)
        return value


class Budget(Model):
    partitions: list[str] = Field(default_factory=list)
    account: str | None = None
    qos: str | None = None
    running_target: int = Field(1, ge=0)
    fill_running_target: bool = False
    pending_reserve: int = Field(1, ge=0)
    min_pending: int = Field(0, ge=0)
    max_pending: int | None = Field(None, ge=0)
    max_running: int | None = Field(None, ge=1)
    max_submitted: int = Field(2, ge=0)
    include_external_user_jobs: bool = True
    source: str = "explicit operator configuration"

    @model_validator(mode="after")
    def pending_bounds(self):
        if self.max_pending is not None and self.min_pending > self.max_pending:
            raise ValueError("min_pending must not exceed max_pending")
        if self.max_running is not None and (self.max_pending is None or self.max_pending > 1):
            raise ValueError("max_running requires max_pending <= 1 for a serial pending reserve")
        return self


class Policy(Model):
    reconcile_interval_seconds: int = Field(15, ge=5)
    max_submissions_per_cycle: int = Field(4, ge=1)
    budget_priority: list[str] = Field(default_factory=list)
    speculative_partition_workers: bool = False
    pending_reserves_after_growth: bool = False
    cancel_pending_when_assigned: bool = False
    allocation_gpu_hours_budget: float | None = Field(None, gt=0, allow_inf_nan=False)
    allocation_budget_allowance: float = Field(0.5, ge=0, allow_inf_nan=False)
    max_array_size: int = Field(1000, ge=1)
    max_array_tasks: int = Field(1000, ge=1)
    max_direct_submitted: int = Field(1000, ge=1)
    short_cpu_seconds: int = Field(300, ge=1)
    unknown_runtime_seconds: int = Field(300, ge=1)
    worker_overhead_mib: int = Field(256, ge=0)
    cleanup_seconds: int = Field(60, ge=0)
    worker_idle_seconds: int = Field(60, ge=1)
    max_rpc_failures: int = Field(3, ge=1)
    submit_cooldown_seconds: int = Field(30, ge=0)


class Control(Model):
    stop_when_complete: bool = False
    auto_renew: bool = False
    renew_lead_seconds: int = Field(600, ge=90)
    hosting: Literal["local", "slurm"] = "slurm"
    partition: str = "cpu"
    account: str | None = None
    cpus: int = Field(1, ge=1)
    memory_mib: int = Field(1024, ge=256)
    walltime_seconds: int = Field(14400, ge=120)
    standby_replicas: int = Field(0, ge=0, le=1)
    require_verified_failover: bool = True
    verification_receipt: str | None = None
    hq_binary: str | None = None
    advertise_host: str | None = None
    server_port: int | None = None
    worker_port: int | None = None


class Config(Model):
    schema_version: Literal[1] = 1
    state_dir: str = "~/.local/state/hyper-hyperqueue"
    log_dir: str | None = None
    cluster: str = "unconfigured"
    control: Control = Field(default_factory=Control)
    policy: Policy = Field(default_factory=Policy)
    budgets: dict[str, Budget] = Field(default_factory=dict)
    profiles: dict[str, Profile] = Field(default_factory=dict)
    workloads: dict[str, list[str]] = Field(default_factory=dict)
    discovery_snapshot: str | None = None

    @model_validator(mode="after")
    def speculative_partition_profiles(self):
        for name, profile in self.profiles.items():
            if name in profile.fallback_profiles or set(profile.fallback_profiles) - self.profiles.keys():
                raise ValueError(f"Profile {name}: fallback profiles must name other configured profiles")
            for target in profile.fallback_profiles:
                other = self.profiles[target]
                placement = ("partition", "account", "qos", "gres", "constraint", "exclude", "gpu_models",
                             "min_vram_mib", "startup_probe_argv", "startup_probe_timeout_seconds")
                if any(getattr(profile, field) != getattr(other, field) for field in placement):
                    raise ValueError(f"Profile {name}: fallback must use the same partition, account, QoS, hardware restrictions and startup probe")
        if self.policy.speculative_partition_workers:
            invalid = [name for name, profile in self.profiles.items() if "," in profile.partition]
            if invalid:
                raise ValueError("Speculative workers require one Slurm partition per profile; split: "
                                 + ", ".join(invalid))
        return self


class Plan(Model):
    schema_version: int = 1
    mode: Literal["direct", "pooled"]
    task_count: int
    profiles: list[str]
    reasons: list[str]
    arrays: list[dict] = Field(default_factory=list)
    unsent_tasks: int = 0
    service_required: bool = False
    warnings: list[str] = Field(default_factory=list)
    requests: list[dict] = Field(default_factory=list)
