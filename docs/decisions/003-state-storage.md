# Durable JSON state with distinct mutation and execution ownership

Status: implemented; shared-host failover is disabled until deployment locking and
fencing tests qualify the actual filesystem. Local tests are not cross-node proof.

Authoritative runtime state lives outside the checkout on an operator-selected
durable shared filesystem. Scratch is appropriate for disposable tests and caches,
not the only copy of command snapshots, campaign metadata or completion evidence.
No SQLite WAL, network database, or additional service is required.

Every import snapshots the exact bytes and immutable settings. Immutable task JSON
is indexed by task ID, avoiding a scan of a million-line manifest in each array
element. A task identifies a campaign/import occurrence, so identical commands are
distinct. Reusing an import key is idempotent only with identical bytes/settings;
an explicit new append key intentionally adds new occurrences. Shell commands
remain one `bash -c` argument. No implicit `set -e`/`pipefail` is introduced.

The short `mutation.lock` serializes coordinator/CLI metadata writes; it is separate
from the service's lifetime leadership lock. JSON replacement fsyncs the file and
parent directory. Coordinator events append to `events.jsonl` under the same lock,
fsync, then atomically update an offset/sequence snapshot. Recovery reads entries
after the snapshot and discards only an incomplete final journal line; invalid
complete records fail closed. Entity files remain authoritative if interrupted
between an entity write and diagnostic event. External submission intents must be
persisted before `sbatch` or HQ submission; accepted-but-unacknowledged mutations
remain unknown until reconciled. A log is not a transactional scheduler API.

On the development NFS scratch mount, a blocking kernel flock waiter continued
waiting after another writer finished. Acquisition therefore uses LOCK_NB and
100 ms retries, retaining kernel exclusion and never deleting a lock. A real
three-process concurrent journal test passed with contiguous records. This is
local-host evidence only; cross-node failure tests remain a deployment gate.

Workers never append the shared event journal or modify coordinator task state.
They publish independent attempt/result files and an identity-bound completion
receipt under a per-task execution flock. A duplicate live dispatch returns 75;
completed identity suppresses replay. A crash after a validated result but before
the completion marker can be adopted without executing again. A missing result is
unknown, not failure. Automatic ambiguous replay requires a declared repeatable
in-place task, exact prior-attempt and epoch matching, and coordinator-supplied
terminal/fencing evidence. `manual_on_ambiguity` never permits ambiguous replay.
An epoch or a stale heartbeat alone is not fencing. This layer trusts the verified
coordinator evidence; deployment service recovery owns scheduler/cgroup fencing.

Service-owned generation recovery is a narrowly attributed exception to ordinary
cancellation: its pre-cancel intent names the old epoch and exact allocation
tokens, and the completed proof binds that intent's SHA and positively terminal
owned allocations. A canceled attempt must have finished after the intent was
created. A coordinator cancellation or durable campaign cancel request always
blocks this exception. Old results are retained unchanged; a new bounded attempt
requires declared repeatability, an eligible retry reason, and unchanged proof
and intent hashes. Cancellation before the service intent remains manual.

Controller retries do not escalate resource requests. Host OOM, timeout,
preemption, application failure, and lost worker are separate categories; changed
resource requests require a new immutable task generation. Unknown submission
acknowledgements stop further automatic growth until reconciled. Startup failure
circuits and submission cooldowns bound repeated infrastructure failures; an
operator can explicitly reset a repaired profile's circuit without deleting
history. Direct and pooled intents share budget reservations. Every queued pooled
worker reads its own frozen configuration snapshot, so later profile edits do
not change an already submitted allocation's advertised capabilities.

Optional collision keys own outputs across tasks, including after completion.
Generic commands are never rewritten to make isolation work: an isolated task
must explicitly use `HHQ_TASK_OUTPUT_DIR` or `HHQ_ATTEMPT_DIR`. The runner retains
attempt outputs and does not invent a scientific publication step. Validators and
expected files qualify completion; absence of either is explicitly exit-code-only
success. Input changes require new immutable tasks, not reused completion markers.

Attempts capture assigned-resource environment, stdout/stderr, signal and timeout
outcomes, validation and sampled process-tree RSS/CPU. No telemetry is advertised
as a true peak or full GPU measurement. SIGKILL can prevent finalization and is
reconciled conservatively. Directory backups must include imports, tasks, intents,
attempts/results, completions, journals and ownership/configuration records; copy
while mutations are stopped or from a consistent filesystem snapshot.

Pooled attempts record both immutable HQ dispatch epoch (`HHQ_SERVER_EPOCH`) and
physical worker generation (`HHQ_WORKER_EPOCH`). A waiting task restored from the
journal can retain its original dispatch epoch while running on a new worker.
Dispatch guards compare the former with coordinator admission; terminal and
fencing proofs bind the latter to the actual Slurm worker allocation. Older
attempt records without a worker epoch retain the original single-epoch fallback.
User task environments cannot replace either provenance value.

On normal shell exit the runner observes status with Linux `waitid(WNOWAIT)`,
cleans the remaining process group, then reaps the leader. Retaining the zombie
leader until cleanup prevents a reused PID from naming an unrelated group.
Commands must wait for background work they intend to retain; deliberately
detached daemon/session processes are unsupported.

GPU tasks verify only positively assigned visible device identifiers before
launch and enforce declared model/VRAM constraints. CPU tasks do not query GPUs.
The shared worker/runner probe initializes the CUDA driver in a bounded fresh
process under unchanged `CUDA_VISIBLE_DEVICES`, checks the visible count, and
reads visible-device UUID, name, and memory without creating a context. Slurm
GPU indices are recorded as provenance, never assumed to be NVML indices.
This handles the documented cgroup renumbering described by
[Slurm](https://slurm.schedmd.com/gres.html#GPU_Management) using NVIDIA's
[device metadata API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html).
MIG assignments remain unqualified and explicitly identified MIG devices are
rejected. Startup preflight receipts retain allowlisted environment identifiers
and bounded errors separately from successful worker registration.
Thread defaults use the declared CPU count while preserving inherited or explicit
OMP/MKL/OpenBLAS/NumExpr choices. GPU matching unit tests use an injected device
query; real allocation evidence is recorded separately in the validation notes.
