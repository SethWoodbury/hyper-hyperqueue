# Current limits and deliberate boundaries

This is a usable initial implementation, not a globally optimal scheduler or an exactly-once transaction system. [Validation](validation.md) separates real allocation tests from mocked scheduler boundaries.

## Operational boundaries

- Production standby is off until its exact state directory is qualified. A readiness receipt for a validation directory is not permission to use a different path. NFS cache delays and scheduler outages can delay recovery; uncertainty fails closed.
- A single CPU control job has a finite walltime. Opt-in `control.auto_renew` supports planned, verified handoffs to prestarted waiting successor jobs; it cannot guarantee a gapless renewal or recover automatically from an unproven crash. The waiting successor uses part of its own walltime while the old service drains. Pending standby is not a ready standby. Shared Slurm, filesystem, authentication or network failures can affect both controls.
- A new HQ epoch restores the journal but requires replacement workers. Recovery can interrupt commands. Checkpoint/restart semantics belong to the application; hhq preserves attempts and enforces bounded replay policy.
- Direct arrays continue without HQ. Unsent direct chunks require explicit `resume`; the optional internal managed-feed path is not exposed as a supported turnkey command yet. The CLI makes the remainder visible.
- Direct MPI layouts are accepted, with an explicit user-supplied launcher. MPI was not run as a cluster test. Pooled multi-node, multiple ranks and multiple GPUs are rejected.
- Unknown submission acknowledgements can require operator investigation. The system intentionally has no force-duplicate toggle that erases uncertainty.
- A finalized generic signal can be retried only with exact owned terminal accounting and temporal attribution to that allocation failure, or a qualified internal service-fencing proof. Natural HQ lifetime expiry also needs its exact exit receipt. User cancellation wins; missing timing/ownership remains manual. An unrelated earlier application failure is never relabeled by a later worker failure.

## Resources and scheduling

- GPU execution currently targets NVIDIA/CUDA and is serial per assigned GPU. Actual model and total VRAM are checked; ROCm/MIG qualification, model-specific concurrent sharing, utilization sampling, general GPU OOM recovery, AF3/JAX preallocation tuning and persistent inference servers are not implemented.
- Host RAM and GPU VRAM are distinct. Process-tree RSS is sampled and can miss short peaks or double-count shared pages. Detached children are outside its guarantee. Slurm's cgroup/accounting evidence remains useful separately.
- Resource summaries are advisory (successful measurements, minimum sample count, P90 runtime plus a fixed 60 seconds, observed RAM plus a fixed 2 GiB). They never silently update requests. A shell command does not establish input size or hardware compatibility. Unknown software/cache/concurrency stays unknown.
- Auto initially uses declared runtime/profile preferences and an explicit unknown-runtime fallback. It does not automatically learn task families from shell text or apply cross-GPU speed estimates. Use measurement summaries as explicit hints. Workload plugins and adaptive weighting are follow-up work.
- Profile weights, worker bounds, shared budgets, outstanding reservations, cooldowns and startup circuits are implemented. Queue-delay forecasting, automatic budget growth, learned interruption risk, and dynamic priority aging of already admitted HQ tasks are not. Static task priority is passed to HQ; equal-priority work follows its scheduler. Avoid continuously submitting higher-priority work if lower-priority completion matters.
- `min_pending` is a per-budget replenishment target, not an instantaneous guarantee: it is bounded by eligible waiting work, hard budgets, profile caps, control lifetime, and submission/reconciliation limits. Uncertain submissions occupy reserve slots until reconciled. A depleted backlog tapers the target; external jobs consume hard allowance but do not count as HHQ reserve workers. `running_target` is a target. Slurm can start multiple pending jobs between polls. `max_submitted` and direct aggregate array throttles are conservative admission bounds, not overrides of the site's accounting/QoS/fair-share rules. Policy limits hidden in submission plugins remain unknown.
- `fill_running_target` can queue a large worker burst without waiting for starts; it does not discover idle nodes or promise immediate scheduling. Submission calls remain sequential, bounded and ownership-recorded. A finite backlog of 184 compatible tasks cannot provide 200 simultaneous useful workers. Increasing burst limits does not waive profile or shared-budget caps.
- Opt-in `speculative_partition_workers` may reserve the same waiting-work capacity in several Slurm partitions. Actual HQ commands remain single, but simultaneous starts can produce idle excess allocations until their idle timeout. This is a deliberate latency-versus-allocation tradeoff; default global capacity conservation remains available. Unknown submissions remain globally binding.
- Optional `retry.oom_memory_multipliers` permits bounded host-RAM increases after positive host-OOM evidence, using append-only attempt overrides. Empty disables it; immutable task identity and successes remain unchanged. GPU VRAM shortages do not trigger host-RAM increases. Pooled CPU co-location prevents attributing job-level OOM to an individual command without its own diagnostic. Legacy increment/max-memory fields stay inert; use the multiplier policy for new imports.

## Outputs and interfaces

- Generic isolated output mode creates per-attempt paths; the command must use those paths. There is no universal atomic publication of arbitrary output directories. A validated completion references the accepted attempt.
- Existing old-system completion markers are not imported automatically. Arbitrary side effects are not made transactional by task IDs. User cancellations remain canceled; internally fenced recovery has its own hash-bound proof requirements.
- The browser is a loopback convenience, read-only by default, with an explicitly enabled command preview/submit form. It shows cached, bounded durable records; unfinished attempts are not proven live, large or partial scans are labeled, and scan cost can lengthen refresh intervals. GPU utilization/VRAM and inference-internal progress are not measured. Advanced controls use the CLI/API. It is not a multi-user hosted web application, and its CPU hosting template does not provide automatic permanent hosting. HTML exports and command logs contain private metadata. See the [dashboard guide](dashboard.md).
- Campaign/task views paginate rows; aggregate status still reads task state files. This was tested on bounded workloads and thousand-task planning, not a million-task metadata benchmark. NFS metadata throughput is a scaling limit to measure before a very large campaign.
- State retention/backup uses external filesystem tools during quiescence. A supported export/restore/compaction CLI and general plugin registry are future work. The optional [RFD4 adapter](rfd4.md) supports a specific saved-manifest/runtime contract, not arbitrary RFD4 command parsing or automatic scientific migration. Actual single-GPU RFD4+AF3 execution and persistent per-bundle model reuse have been measured; exact RNG parity, continuous GPU residency, specialized multi-GPU/AF2 execution and unobserved runtime/hardware combinations remain unqualified. See the dated [measured reference](rfd4-measured-resources.md). The sanitized support bundle is diagnostic, not a restorable backup.
- Early generic validation did not run scientific workloads. Later explicitly authorized RFD4+AF3 controls and comparisons did, and the HHQ repository was published. Those dated qualifications do not authorize changing model weights, altering frozen campaign records or launching unrelated workloads.

These limits are reflected in errors/defaults rather than claims of untested functionality. The code is structured so workload estimators or model-sharing policies can be added behind the existing runner/profile interfaces without replacing Slurm or HQ.

- Automatic completion shutdown is opt-in (`control.stop_when_complete`). It requires
  authoritative completion/retry and scheduler evidence; an unknown allocation or
  paused unfinished command intentionally prevents shutdown. It does not convert
  failures into successes, cancel unrelated jobs, or stop workers doing active work.
- Resource changes do not mutate existing task identities. Bounded host-OOM retries
  use attempt overrides; other changes use the documented remaining-only export.
  Reimporting the whole command list with different resources can repeat successes.
- The optional state-wide allocation GPU-hour ceiling conservatively reserves full
  requested time for unresolved allocations. It is not a billing/utilization meter
  and can temporarily underuse the allowance. A zero extension still allows work
  inside the base budget; an unspecified ceiling imposes no inferred spend cap.
- Strict `max_running` uses one dependency-gated pending reserve and exact owned
  worker evidence. It can underfill during reconciliation and does not cap unrelated
  manually submitted jobs. Optional fallback phases must preserve placement
  restrictions and actual hardware fit; they are not cross-partition migration.

Application failures do not currently quarantine a healthy worker automatically.
An optional profile `startup_probe_argv` can reject a broken application environment
before HQ registration; it consumes no task and uses the existing startup circuit.
It only tests the supplied command, not later model execution or GPU kernels.
The allocation circuit breaker detects worker startup failures and rejected Slurm
submissions; it is not a consecutive command-failure limit. A started HQ worker
can therefore consume multiple commands that all fail during application/model
startup, including failures on an unsupported GPU model. Qualify broad hardware
routes with bounded first runs before releasing a large backlog, and pause an
affected route when a systemic fault appears. Scientific designs that miss quality
cutoffs but complete their declared output validation remain successful executions.
