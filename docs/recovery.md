# Recovery and ownership

`hhq service status` reports the current epoch, owner, endpoint, deployment intents, and heartbeat age. An old heartbeat never alone permits takeover.

| Failure | Behavior |
| --- | --- |
| Controller exits | Supervisor restarts it within a bounded circuit breaker; HQ and its journal stay live. |
| Application returns nonzero | The attempt remains failed; HQ does not cancel unrelated tasks in that job. |
| Worker disappears | HQ marks its affected tasks failed (`crash_limit=0`); hhq alone owns bounded retries after positive allocation termination and output-policy checks. |
| HQ or control allocation dies | Stop admission; fence exact old control and workers; acquire the lifetime lock; restore journal in a new epoch. Workers do not survive HQ restart. |
| Control walltime approaches | `USR1` stops admissions and drains. Workers advertise a lifetime bounded by the control allocation, including cleanup reserve. Opt-in planned renewal prestarts one waiting successor and requires a graceful handoff plus terminal ownership proof; qualified standby is a separate recovery option. |
| Shared storage, Slurm, or ownership unavailable | Block automatic takeover/replay and report uncertainty. Do not delete a lock or edit an epoch to bypass it. |
| Accepted submission loses acknowledgement | Reconcile its exact durable token; do not resubmit blindly. |
| A validated output already exists | The shared runner verifies its fingerprint and preserves success. |
| Interrupted arbitrary side effect | Manual handling unless repeatability, output policy, retry bound, and fencing evidence permit retry. |

Local hosting is explicit. It can restart after positively identified local processes are gone, but automatic remote failover is not promised. Shared file locks are held through the full lifetime of the owner and inherited HQ/controller children; local PID identity includes start ticks and boot ID.

A Slurm standby readiness receipt must be `hhq-failover-readiness/1`, identify the exact existing state directory, two tested nodes, expiration, hash-bound evidence, and true results for `cross_node_exclusion`, `lock_owner_loss`, `exact_slurm_fencing`, and `worker_connectivity`. Partial probes deliberately keep unsupported fields false. Readiness remains disabled on an ordinary untested state directory.

Local real-HQ tests cover two-worker CPU packing, independent command failure, explicit append, journal restart, worker loss without hidden HQ retries, and controller restart while the same HQ server stays alive. Unit tests additionally exercise unknown submissions, ownership/epoch conflicts, missing readiness, terminal accounting, and the two-control replenishment bound. Cluster evidence belongs outside Git and should state actual nodes, allocation IDs, versions, result hashes, and cleanup status; do not infer live standby readiness from these unit tests.

Drain stops new controller admission and lets already admitted HQ work finish on existing workers. It is not an atomic pause of an HQ backlog. Stop interrupts the service; outputs and attempt records remain available. User cancellation is not automatically retried. A running direct Slurm array continues independently while HQ/control service is stopped.

## Cluster validation performed on 2026-10-12

The bounded validation used one 512 MiB CPU control initially, then two 512 MiB CPU controls for standby, and at most one 1 GiB/two-CPU worker. Control and worker connected across distinct compute nodes. Eight initial commands succeeded, an intentional exit 7 stayed failed, and two appended commands succeeded on the same worker. Killing only the recorded controller process restarted it while the same HQ server and epoch remained active.

A planned `USR1` handoff restored the journal in a new epoch and replenished a healthy standby. A separate suspended-owner drill stopped the exact old supervisor, HQ, and controller while one repeatable synthetic command ran. After the 120-second suspicion threshold, standby recorded its intent, positively fenced the old control and worker allocations, acquired leadership, and retried the affected command. Its second attempt passed the output validator **156.39 seconds after suspension**. The original successful commands were not executed again. No more than two control allocations and one worker were live in this drill.

The first replenishment probe exposed an incorrect CLI module name in generated replacement scripts; three tiny controls failed before doing work. The corrected constant module name, startup-failure circuit, and healthy replacement were retested. Those failed allocations and their stderr remain in the private evidence. Every CPU validation allocation was terminal at cleanup.

This qualification is specific to the tested state directory and filesystem. It does not enable standby for the default or another production state directory. The final private `readiness-verified.json` binds the storage, connectivity, fencing, planned-handoff, interrupted-task, and cleanup receipts. This was a suspended-owner/fencing test, not an exhaustive network-partition proof.

Internal fencing differs from user cancellation. A retry can use the exact prior epoch, job ID, allocation token, terminal observations, and a hash-bound pre-cancel intent. An explicit user cancellation still blocks replay. A finalized canceled result that predates that intent remains manual; the system does not retroactively call it an internal cancellation. Missing final records can be retried only after positive quiescence, repeatability, output-policy, and attempt-limit checks.

## Explicit restart without standby readiness

A normal `standby_replicas: 0` service can be stopped and restarted without qualifying automatic failover. Restart performs **read-only** reconciliation of the exact previous control allocation and every old worker intent/receipt. Every allocation must be absent from the live queue and positively terminal in accounting under its original token and user. The new supervisor then acquires the unchanged lifetime lock. This also allows an explicit restart after abrupt control loss once all old resources are positively terminal; it does not enable automatic takeover or silently cancel old work.

Start with the supported commands:

```bash
uv run hhq service stop --no-drain
uv run hhq service status
uv run hhq status --reconcile
```

If an old control or worker is still alive, inspect `service/deployment.json`, archived `service/deployments/`, and `intents/` for its exact job ID and full token. A dead controller cannot itself carry out a queued stop request. An operator can cancel the positively identified allocation using the same ownership-checking adapter; replace the two placeholders only after inspecting the record:

```python
from hyper_hyperqueue.backends.slurm import SlurmClient
client = SlurmClient()
client.cancel(["EXACT_JOB_ID"], expected_token="EXACT_INTENT_TOKEN")
```

The adapter refuses an unrelated or unproven job. Repeat only for the exact old allocations needing termination, wait for terminal accounting, then run `hhq service start --hosting slurm --standby 0`. A `manual-restart-EPOCH.json` receipt records the read-only proof. User-canceled task attempts remain canceled; restarting the service is not permission to replay them.

Missing accounting, ambiguous submission acknowledgements, or unknown ownership still blocks restart. Resolve those identities with the scheduler/operator; do not delete a lock, alter an epoch, or invent terminal evidence. A pending control stopped before it acquires leadership will exit before starting HQ when Slurm starts it. Explicitly cancel that exact pending allocation if it should also leave the Slurm queue immediately.

## Finishing a partially completed command file

Commands have independent attempt records and validated completion markers.
Resuming the same campaign skips successful occurrences; it does not rerun the
whole command file. Pooled workers replenish while eligible work remains and the
controller is alive. Worker walltime is not a fixed total command quota.

For a known failure with unchanged resources and an unused retry allowance:

```bash
hhq retry CAMPAIGN --failed
hhq resume CAMPAIGN
```

`resume` only resumes admission. If the pooled service has stopped (including
automatic completion shutdown), explicitly run `hhq service start` after status
and ownership reconciliation. Neither command increases the immutable attempt
limit. Unknown/live attempts and user cancellations are not retry permission.

For resource changes or an exhausted retry allowance, prepare a **remaining-only
execution generation**. First pause imports and dispatch, allow admitted work to
finish, and reconcile its terminal status:

```bash
hhq campaign control CAMPAIGN pause-import
hhq campaign control CAMPAIGN pause-dispatch
hhq status --reconcile
hhq export-remaining CAMPAIGN /shared/path/recovery_01
```

Export submits nothing. It excludes successful and user-canceled occurrences,
preserves exact shell/argv, working directory, environment, output validators,
retry policy and input/software identities, and writes `commands.jsonl` plus
`manifest.json`. Exported occurrences become `blocked` in the original campaign
with reason `exported_to_new_generation`; ordinary resume or retry cannot run
them again. Original attempts and outputs remain untouched. Repeating the same
export is idempotent, including recovery from interrupted export publication.

The exporter refuses active/uncertain submissions, running or unknown attempts,
nonrepeatable failed side effects, output collision claims, and heterogeneous
execution settings. Resolve those cases explicitly instead of exporting them
blindly. A source campaign must remain paused. These checks are deliberately
stricter than copying every command whose status is not `succeeded`.

Use the verified manifest when explicitly importing the replacement work:

```bash
hhq submit /shared/path/recovery_01/commands.jsonl \
  --recovery-manifest /shared/path/recovery_01/manifest.json \
  --profile gpu --mem 24GiB --runtime 1800 --dry-run
# Remove --dry-run to import/submit after reviewing the plan.
```

Keep the **same HHQ state directory**. A stable recovery import key prevents a
repeated submission from creating duplicate command occurrences; changed settings
after that import require another explicit recovery generation. Only supplied
resource flags change; original validators and execution settings stay in force.
`--max-attempts` can explicitly change the fresh generation's attempt bound while
preserving its retry reasons. Do not submit the JSONL without
`--recovery-manifest`: ordinary command import does not restore that contract.

For pooled work, `--mem` describes one task; choose/update a worker profile whose
RAM covers it plus worker overhead. It does not automatically enlarge an existing
Slurm worker. Changed pool configuration requires the existing supported explicit
stop/restart procedure; an active service keeps its frozen configuration. Resource
summaries remain advisory. The optional bounded host-OOM policy below can enlarge only future attempt/allocation requests.

For RFD4, exact completed row receipts inside a bundle are revalidated and reused
even if another row needs another attempt. Unfinished diffusion may restart;
this is not mid-trajectory RNG restoration. Preserve the original bundle, source
pins and scientific output paths. A Git/runtime/seed change is a different
scientific execution and is intentionally rejected by old row ownership.


## Automatic interruption and host-memory retries

Configure the immutable task retry policy when importing a new campaign:

```python
RetryPolicy(
    max_attempts=6,
    reasons=["worker_lost", "preempted", "allocation_timeout", "host_oom"],
    oom_memory_multipliers=[1.25, 1.5, 2.0],
)
```

The empty multiplier list disables memory escalation. `max_attempts` bounds all
attempts, including the first; the multiplier list separately limits host-OOM
increases. Each increase is relative to the **original** RAM request, rounded up
in MiB. A 16 GiB task requests 20, 24, then 32 GiB after successive confirmed host
OOMs. An OOM at the largest step stops automatic retry. The original task/config
stays unchanged; hash-bound, append-only resource decisions and each attempt's
actual request record the changes. Future worker allocations request enough RAM
for their covered tasks; old workers retain their actual smaller capacity.

A terminal Python `MemoryError` diagnostic or exact Slurm `OUT_OF_MEMORY` evidence
can identify host OOM. Exit 137 alone cannot. Job-level OOM cannot identify which
of several pooled CPU commands caused it: without an individual diagnostic these
victims can retry as `worker_lost` at the same RAM. Serial GPU allocations and direct
single-command allocations do not have this CPU co-location ambiguity. Explicit
CUDA/torch OOM is recorded as `gpu_oom`; more host RAM is not an automatic remedy.

Slurm `TIMEOUT` is `allocation_timeout`; a command's own `timeout_seconds` remains
`timeout` with `timeout_scope="task"`. Only include the latter retry reason when
repeating that command timeout is intentional. Positive owned terminal accounting
can resolve a finalized generic SIGTERM/SIGKILL result when it coincides with the
allocation end (within 120 seconds). An earlier specific application/GPU-memory/
validation failure is never relabeled because its worker failed later. Natural
HQ worker time-limit expiry additionally requires the exact worker exit receipt
and terminal accounting. Raw `COMPLETED` alone does not explain a finalized
cancellation. Explicit task/campaign cancellation always wins; uncertain ownership
or timing requires inspection. Exhausted retries remain visible failures.

One existing controller owns retry and refill decisions. Direct arrays work
without HQ; unattended direct retries currently require the internal managed-feed
controller path, while ordinary CLI direct campaigns use explicit retry/resume.
RFD4 bundle retries reuse validated completed rows and restart only unfinished
rows, using their original inputs, seeds and output identity.

## Optional cumulative allocation budget

`policy.allocation_gpu_hours_budget` defaults to `null` (no cumulative ceiling).
When explicitly set, the ceiling is that value times
`1 + policy.allocation_budget_allowance`; allowance defaults to `0.5`. Setting
allowance to zero permits no extension, but still permits retry/refill inside the
base budget. This is independent of Slurm allocation walltime, command timeout,
queue concurrency and the finite service lifetime.

Before each new allocation, HHQ reserves its full requested GPU-hours, including
pending/running/unknown allocations. Positive terminal accounting replaces that
reservation with recorded elapsed allocation time times GPU count. Missing or
invalid evidence retains the conservative reservation. CPU control time is not
included. These are allocation GPU-hours, not GPU utilization or site billing.
A blocked budget prevents new allocations; it does not terminate already running
commands or silently enlarge the ceiling. Conservative reservations can leave
unused budget until jobs finish. Use a separate state directory for a separate
budget scope, or intentionally share the state-wide allowance.

The new retry ladder, attribution, budget and pending-drain paths have bounded
CPU/local-HQ and mocked scheduler tests. On October14 a live direct CPU probe
confirmed actual Slurm host-OOM recovery: job29197837 requested512MiB and ended
OUT_OF_MEMORY after1s; the controller automatically submitted job29197854 at640MiB
(1.25×), which completed and validated its output after3s. Both were terminal;
no GPU was used. Original task resources stayed512MiB and the retry override was
hash-bound. The full ladder/exhaustion remains synthetic-tested, not a live GPU
qualification. Shared filesystem metadata delayed attempt visibility in an earlier
256MiB probe; absence from an early directory listing was not proof of no execution.
Private evidence: AChE V3 scratch `agent_scratchwork/hhq_recovery_qualification_01/
direct_cpu_oom_attempt03/qualification.json`.
