# Recovery and ownership

`hhq service status` reports the current epoch, owner, endpoint, deployment intents, and heartbeat age. An old heartbeat never alone permits takeover.

| Failure | Behavior |
| --- | --- |
| Controller exits | Supervisor restarts it within a bounded circuit breaker; HQ and its journal stay live. |
| Application returns nonzero | The attempt remains failed; HQ does not cancel unrelated tasks in that job. |
| Worker disappears | HQ marks its affected tasks failed (`crash_limit=0`); hhq alone owns bounded retries after positive allocation termination and output-policy checks. |
| HQ or control allocation dies | Stop admission; fence exact old control and workers; acquire the lifetime lock; restore journal in a new epoch. Workers do not survive HQ restart. |
| Control walltime approaches | `USR1` stops admissions and drains. Workers advertise a lifetime bounded by the control allocation, including cleanup reserve. Verified standby takes over and replenishes one control allocation. |
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
