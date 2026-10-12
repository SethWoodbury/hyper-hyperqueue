# Cluster discovery and Slurm boundaries

`hhq` discovers placement facts; Slurm decides when admitted jobs run. A snapshot
is evidence for validation and shared user budgets, not a prediction of future
free nodes. Discovery never submits jobs.

Private snapshots belong under `.local/discovery/` or the external state folder.
Raw node inventories, usernames, internal endpoints and job commands must not be
committed. `discovery.discover(output_dir, raw_dir=...)` records command arguments,
timestamps, hashes and failures, and saves a normalized `snapshot.json` plus a
readable `cluster-report.md` with observed limits, unknowns and provenance. It runs
bounded read-only CLI queries with four concurrent subprocesses at most.

## Facts and unknowns

The snapshot includes partitions, walltimes, preemption, partition QOS, user
associations, QOS limits, configured GRES/features, and all current-user jobs.
Array ranges are counted exactly, including jobs submitted outside hhq. Up to
10,000 elements are expanded; larger ranges retain an exact count without
creating hundreds of thousands of Python objects. Explicit child queries and
cancellation still verify membership in these ranges.
An empty successful queue is different from a failed query: failed observation
has an error and unknown aggregate count, never a fabricated zero.

A blank `MaxSubmitJobs` field means no explicit limit at that queried scope. It
does not establish an unlimited user budget. Parent associations, partition QOS,
and site submission plugins may constrain jobs. The discovered `MaxJobCount`
applies to the whole cluster. A backfill inspection limit such as
`bf_max_job_user` is not a submitted-job limit.

[Slurm's resource-limit hierarchy](https://slurm.schedmd.com/resource_limits.html)
explains why independently measured scopes cannot simply be added together.
Profiles that share a user/account/QOS restriction must share a configured budget.
Conservative bootstrap targets in the private local configuration are operator
policy, explicitly labeled as such, rather than alleged discovered limits.

`validate_config_against_snapshot(config, snapshot)` rejects observable placement
mismatches: unknown partitions, disallowed observed account/QOS pairs, excessive
walltime, absent typed GRES/counts and array bounds. It does not certify application
compatibility or infer GPU VRAM from a marketing name. A GPU node feature and a
GPU GRES type are separate scheduler concepts; use only observed expressions.

## CLI adapter behavior

`backends.slurm.SlurmClient` has no import-time side effects. Its injectable runner
supports deterministic tests. `query_jobs` and `accounting` return `SlurmJob`
dataclasses; `submit` returns a `SubmissionResult`. Each has `to_dict()`.

The adapter prefers Slurm JSON. On explicitly unsupported JSON options/plugins,
it falls back to selected delimited fields. Authentication/transport errors,
malformed JSON, and Slurm JSON `errors` fail closed. They never trigger an empty
queue assumption. JSON numeric wrappers preserve the difference between zero,
unset and infinite values. Normal queries are batched; accounting always has
explicit start/end timestamps and a default seven-day bound.

Every submission uses `--parsable` and a durable `--comment=hhq:TOKEN`. All
job names also carry the full token, including when a caller supplies a name prefix. The caller journals its intent before calling
`submit`. A successfully acknowledged job ID means accepted, not running. A
transport timeout, malformed response or unfamiliar failure remains **unknown**.
There is no automatic resubmission. `find_submission` searches both live jobs and
bounded accounting for the exact user/token. Sites may omit comments from
accounting; the full token suffix in the job name provides the historical identity
in that case. Even no matches may reflect delayed accounting. Reconciliation policy must preserve that uncertainty.

`cancel` checks current ownership against the complete token and user before
canceling explicit IDs. For an array parent it verifies every observed child.
The wrapper never cancels arbitrary user jobs to create headroom.

See the upstream [sbatch](https://slurm.schedmd.com/sbatch.html),
[squeue](https://slurm.schedmd.com/squeue.html), and
[sacct](https://slurm.schedmd.com/sacct.html) references for CLI semantics.

## Shared storage and controller recovery

An NFS mount and successful local `flock` do not prove cross-node failover safety.
Before standby activation, a bounded disposable test must establish:

1. Atomic file publication visible from two distinct allocated nodes.
2. Exclusive lock contention from those nodes.
3. Positive identification and terminal Slurm evidence for the retired owner.
4. Lock acquisition only after that owner is quiescent.
5. Connectivity of the actual HyperQueue worker to the control endpoint.

Record exact paths, job identities, timestamps, hashes and the tested state
directory. A generic TCP echo proves transport reachability only; it is not an
HQ worker integration test. Never enable standby from partial evidence, missing
heartbeats, stale mtimes, or a job's absence from `squeue` alone. Production state
must live outside the checkout, and SQLite WAL is not used on shared NFS.

The initial private discovery on this installation found Slurm 25.11.5 and NFS
home/scratch mounts. GPU runtime/device-isolation probes and failover receipts
are separate integration evidence. Keep observed results and unvalidated
capabilities distinct in the operations record.
