# Validation record — 2026-10-12

Implementation and validation were performed on an actual Slurm login host, with Python 3.12, Slurm 25.11.5 and separately installed HyperQueue 0.26.2. Tests used synthetic shell/Python/CUDA commands. Existing scientific jobs and campaigns were untouched. Private evidence and failed exploratory runs were retained rather than overwritten.

## What has been exercised

| Path / failure | Evidence and outcome |
| --- | --- |
| Direct CPU, no running HQ | Real three-element Slurm array: two successes and deliberate exit 7. Failed-only retry submitted just the failed task. Final counts: three successes; attempt counts 1, 1, 2. |
| HQ optional dependency absent | Separate wheel installation in a scratch virtualenv without the `hyperqueue` package, with an unavailable HQ binary: direct plan, runner and status passed; no service was created. |
| Real local HQ | Two real workers, concurrent CPU packing, multiple commands per worker, independent command failure, append, worker termination, journal restart, and controller restart tests. |
| Real CPU pooled service | CPU control and a different CPU worker host: eight initial successes, one intentional failure, then two appended successes on the same worker. |
| Controller-only restart | Killed the recorded controller child; supervisor restarted it while the same HQ PID and epoch stayed alive. No duplicate initial tasks. |
| Shared storage | Two distinct allocated CPU nodes proved flock exclusion, bidirectional atomic publication, exact owned-job fencing, and lock acquisition after verified owner termination. |
| Planned control handoff | Sent the pre-walltime `USR1` signal to the exact owner. Standby restored the journal in a new epoch; replacement standby became healthy; at most two control allocations. This tests the signal path, not a full multi-hour timeout. |
| Suspended owner with running task | Suspended the owned supervisor/HQ/controller while a repeatable synthetic command ran. After the 120-second suspicion threshold, standby wrote a fencing intent, fenced exact old control and worker, proved termination/exclusion, restored journal and retried only that task. Attempt 2 passed its output validator about **156.39 seconds** after suspension. Earlier ten successes and one intentional failure retained one attempt each. |
| GPU allocation | One NVIDIA L40 worker ran three commands serially. Each allocated, wrote, synchronized and freed 1 MiB of device memory and passed its output validator. Slurm GPU index 4 / CUDA-visible ordinal 0 resolved to the same UUID in the worker and every task. The allocation completed in 32 seconds. |
| Browser and diagnostics | Real local HTTP requests check status, escaped task/log pages, bounded log reads, rejection of cross-site hosts and missing mutation token. CLI plans and diagnostics are tested without scheduler mutations. |
| Budget/failure simulation | Shared GPU subclasses, external jobs, compressed arrays, uncertain accepted submissions, pure plans, array throttles, worker startup circuits, output validation, explicit cancellation and stale epochs are tested through modeled adapter boundaries. These are not real infrastructure outages. |

Direct array IDs were `29054372` and failed-only retry `29054408`. CPU pooled control/worker were `29054419` / `29054420`. The interrupted-owner drill fenced `29054619` / `29054644`. All thirteen controls/workers created across the CPU service drills were positively terminal at cleanup; the desired deployment and leader were both stopped. Storage and direct-array probes also finished. Exact node names, scripts, outputs, timestamps, checksums and scheduler receipts remain in private evidence.

The passing GPU control/worker were `29056924` / `29056926`. Earlier sequential probes preserved useful negative evidence: one typed GPU allocation was canceled while pending (zero GPU execution); a three-second startup exposed an incorrect device-index lookup; after repair, a separate NVIDIA L40S correctly registered and dispatched all three commands but CUDA context creation returned `CUDA_ERROR_ECC_UNCORRECTABLE` (214). Only the final validation profile excluded that node. No production exclusion was added. These probes used at most one GPU allocation at a time; the final success qualifies this L40/CUDA path, not every GPU or container. All eight GPU-probe control/worker jobs are terminal, with zero live owned jobs and all deployments stopped. Total GPU allocation time, including failures, was **62 seconds (0.0172 GPU-hours)**.

## Bugs discovered by real tests

- Concurrent first writes on NFS exposed negative directory-cache races. Import now precreates shared/task parents under the mutation lock; bounded directory refresh handles remote creation visibility. A fresh direct array passed afterward. The original failed probe remains available.
- This site's accounting omits Slurm comments. Every owned job now also carries its full intent token in the job name; historical reconciliation checks the exact user/name token.
- A replacement control launched from a module CLI initially rendered `python -m __main__`. The module name is now explicit, a regression test executes that rendering context, and a startup circuit bounds repeated replacement failures. The corrected replacement became healthy in the real drill.
- GPU startup originally treated a Slurm GPU index as an NVML query index. Assigned-device metadata now comes from the CUDA driver under unchanged inherited visibility, using the UUID shared by worker and task telemetry. Startup writes a bounded preflight receipt even on failure. A real L40S proved identity/dispatch despite its CUDA ECC error; a real L40 then passed device operations. Regression tests reject task overrides of device visibility and ordering.

## Automated checks

Use the pinned environment:

```bash
uv sync --all-extras --locked
uv run pytest -q --basetemp /net/scratch/woodbuse/hyper-hyperqueue/pytest-final
uv run ruff check src tests examples
uv build
```

The final integrated run passed **118 tests in 21.37 seconds**, with no skips, including real local HQ processes and CPU packing. Lint, formatting, lockfile consistency, wheel/source builds and documentation-link checks passed. Built archives contain no private runtime/discovery files. Local HQ integration tests skip explicitly if the pinned executable is unavailable; a skip is not counted as a demonstrated HQ path. The separate direct-only environment validates that optional imports are real.

## Standby scope and unresolved infrastructure limits

Automatic standby was **operational in the qualified validation state directory**. The readiness receipt binds that exact directory path, two tested nodes, expiring evidence and verified Slurm/HQ connectivity. The ordinary production configuration remains `standby_replicas: 0`, without a readiness receipt. It cannot silently enable redundancy on a different path. Replacing/restoring the directory requires requalification.

The suspended-owner test is a controlled failure drill, not a physical host power cut or an NFS/Slurm service outage. Mocked tests cover unavailable storage/accounting and racing candidates by failing closed. No claim is made that every infrastructure partition, GPU model/container combination, MPI program, multi-day rollover, or production-scale metadata load has been validated. Site submission-plugin policy and storage purge/backup guarantees remain administrator-owned unknowns.

## Private evidence index on this installation

- Discovery and resolved profile: repository `.local/discovery/cluster-report.md`, `.local/cluster.yaml`.
- Storage, pooled CPU, controller restart, planned handoff, interrupted recovery and cleanup: `~/.local/state/hyper-hyperqueue-validation/20261012T231029Z/` (`readiness-verified.json`, `standby_recovery_result.json`, `hq_validation_cleanup.json`, `manual_restart_qualification.json`).
- Direct array / failed-only retry: `~/.local/state/hyper-hyperqueue-validation/direct-20261012-v2/result.json`.
- Original failed NFS probe: `~/.local/state/hyper-hyperqueue-validation/direct-20261012/`.
- Direct-only wheel environment: `/net/scratch/woodbuse/hyper-hyperqueue/direct-only-env-validation/result.json`.
- Passing GPU probe: `~/.local/state/hyper-hyperqueue-gpu-validation/20261012T233953Z_healthy_device/` (`execution_pass.json`, `cleanup.json`, `integration_result.json`).
- Earlier GPU probes: sibling directories `20261012T232302Z`, `20261012T233033Z_flex`, and `20261012T233401Z_cuda_visibility`, with separate failure and cleanup evidence.
- Final test report: `/net/scratch/woodbuse/hyper-hyperqueue/pytest-final-20261012T2341.xml`; private handoff index: repository `.local/final-validation.json`.

Runtime metadata is outside the checkout; scratch holds disposable test environments and bulk output. No remote Git repository was created or pushed.
