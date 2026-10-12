# Current limits and deliberate boundaries

This is a usable initial implementation, not a globally optimal scheduler or an exactly-once transaction system. [Validation](validation.md) separates real allocation tests from mocked scheduler boundaries.

## Operational boundaries

- Production standby is off until its exact state directory is qualified. A readiness receipt for a validation directory is not permission to use a different path. NFS cache delays and scheduler outages can delay recovery; uncertainty fails closed.
- A single CPU control job has a finite walltime. There is no unlimited automatic service lifetime without qualified rollover. Pending standby is not a ready standby. Shared Slurm, filesystem, authentication or network failures can affect both controls.
- A new HQ epoch restores the journal but requires replacement workers. Recovery can interrupt commands. Checkpoint/restart semantics belong to the application; hhq preserves attempts and enforces bounded replay policy.
- Direct arrays continue without HQ. Unsent direct chunks require explicit `resume`; the optional internal managed-feed path is not exposed as a supported turnkey command yet. The CLI makes the remainder visible.
- Direct MPI layouts are accepted, with an explicit user-supplied launcher. MPI was not run as a cluster test. Pooled multi-node, multiple ranks and multiple GPUs are rejected.
- Unknown submission acknowledgements can require operator investigation. The system intentionally has no force-duplicate toggle that erases uncertainty.
- A finalized attempt reporting `SIGTERM` remains canceled unless an exact internal service-fencing intent attributes that interruption. Slurm `PREEMPTED`/OOM/timeout evidence can authorize bounded replay of a missing-final attempt; it does not currently override a finalized cancellation. Some preemptions therefore require manual inspection/new execution generation. This conservative distinction avoids replaying an earlier user cancellation merely because its worker allocation later ended.

## Resources and scheduling

- GPU execution currently targets NVIDIA/CUDA and is serial per assigned GPU. Actual model and total VRAM are checked; ROCm/MIG qualification, model-specific concurrent sharing, utilization sampling, GPU OOM recognition from arbitrary stderr, AF3/JAX preallocation tuning and persistent inference servers are not implemented.
- Host RAM and GPU VRAM are distinct. Process-tree RSS is sampled and can miss short peaks or double-count shared pages. Detached children are outside its guarantee. Slurm's cgroup/accounting evidence remains useful separately.
- Resource summaries are advisory (successful measurements, minimum sample count, P90 runtime plus a fixed 60 seconds, observed RAM plus a fixed 2 GiB). They never silently update requests. A shell command does not establish input size or hardware compatibility. Unknown software/cache/concurrency stays unknown.
- Auto initially uses declared runtime/profile preferences and an explicit unknown-runtime fallback. It does not automatically learn task families from shell text or apply cross-GPU speed estimates. Use measurement summaries as explicit hints. Workload plugins and adaptive weighting are follow-up work.
- Profile weights, worker bounds, shared budgets, outstanding reservations, cooldowns and startup circuits are implemented. Queue-delay forecasting, automatic budget growth, learned interruption risk, and dynamic priority aging of already admitted HQ tasks are not. Static task priority is passed to HQ; equal-priority work follows its scheduler. Avoid continuously submitting higher-priority work if lower-priority completion matters.
- `running_target` is a target. Slurm can start multiple pending jobs between polls. `max_submitted` and direct aggregate array throttles are conservative admission bounds, not overrides of the site's accounting/QoS/fair-share rules. Policy limits hidden in submission plugins remain unknown.
- OOM/time failures are recorded but resource escalation does not mutate an existing task. Import a new generation with explicit resources after inspection. Legacy schema-1 retry memory fields accept only their original inert defaults for fingerprint compatibility; configurable escalation is rejected.

## Outputs and interfaces

- Generic isolated output mode creates per-attempt paths; the command must use those paths. There is no universal atomic publication of arbitrary output directories. A validated completion references the accepted attempt.
- Existing old-system completion markers are not imported automatically. Arbitrary side effects are not made transactional by task IDs. User cancellations remain canceled; internally fenced recovery has its own hash-bound proof requirements.
- The browser is a loopback convenience with preview/submit and read-only detail pages. Advanced controls and resource options use the CLI/API. It is not a multi-user hosted web application. HQ credentials and private command logs must not be shared in public bundles.
- Campaign/task views paginate rows; aggregate status still reads task state files. This was tested on bounded workloads and thousand-task planning, not a million-task metadata benchmark. NFS metadata throughput is a scaling limit to measure before a very large campaign.
- State retention/backup uses external filesystem tools during quiescence. A supported export/restore/compaction CLI and application-specific validators/plugins are future work. The sanitized support bundle is diagnostic, not a restorable backup.
- No scientific AF3/RFD workloads, weights, existing campaigns or remote Git repositories were changed for validation.

These limits are reflected in errors/defaults rather than claims of untested functionality. The code is structured so workload estimators or model-sharing policies can be added behind the existing runner/profile interfaces without replacing Slurm or HQ.
