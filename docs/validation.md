# Validation record — 2026-10-12

## October 14 — planned renewal and rapid multi-partition refill

The combined suite passed **466 tests, zero skipped, in 109.05 seconds** on shared
scratch storage. `uv run ruff check src tests examples` also passed. Three agents
implemented/reviewed planned renewal, burst planning and ownership boundaries.
No Slurm or scientific workloads were launched for these changes.

Real local HyperQueue restored waiting work from a flushed journal in a new
physical worker epoch without repeating the already completed command. Mocked
Slurm tests cover prestarted successor readiness, exact normal predecessor and
worker termination, bounded failed takeover, explicit stop/completion, uncertain
submission acknowledgements and early renewal of quiescent lifetime-blocked work.
Actual multi-node Slurm rollover remains untested for this feature.

Planner/controller tests cover 200 submissions in one cycle, no per-worker
15-second sleep, actual controller remaining-time checks, hard/profile caps,
external jobs and optional cross-partition reservations. Same-partition profiles
share capacity; unknown/wrong-token ownership remains globally reserved. These
are synthetic capacity tests, not a live 200-GPU benchmark.

A detached 96-bundle application preparation was also checked read-only: all
resource envelopes fit the chosen 8/5/12-hour routes, with no execution state
created. Speculative mode may reserve capacity for those tasks in several
partitions; it does not create additional HQ tasks. Worker/model processes restart
at a planned service handoff, and canceled pending workers lose their queue place.

The earlier October 12 implementation and validation below were performed on an actual Slurm login host, with Python 3.12, Slurm 25.11.5 and separately installed HyperQueue 0.26.2. Tests used synthetic shell/Python/CUDA commands. Existing scientific jobs and campaigns were untouched. Private evidence and failed exploratory runs were retained rather than overwritten.

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

## RFD4 adapter extension — October 12 local / October 13 UTC

The optional [RFD4 adapter](rfd4.md) was added without modifying the native RFD4 checkout, the reference AChE notebook/campaign, or their environments. No cluster or scientific jobs were submitted for this extension.

The added tests exercise native-manifest parsing, preservation of seeds/configurations, source/environment pins, preparation idempotency, resource-profile budgets, explicit import/launch, partial acknowledgement recovery, a recovery-aware collection barrier, output ownership, tampering, incomplete batch coverage, early native exit, completed-row reuse and interruption. A standalone subprocess test runs the adapter against a **synthetic native module**, including its real CLI, scoped Lightning adapter, completion receipts and repeat invocation. It does not simulate the RFD4 model's scientific behavior.

The final integrated run passed **207 tests in 38.26 seconds, with no skips**. The standalone marimo notebook passed `marimo check` and HTML export with blank controls: opening it performed no preparation or submission. Lint, formatting, source/wheel builds, private-file exclusion and local documentation links passed. The earlier 118-test result describes the original generic HHQ implementation, not the adapter extension.

A CPU-only `--check` probe used the **actual** RFD4 checkout and its selected `.venv`. It failed during native import with `ModuleNotFoundError: peft`. Read-only inspection found other declared dependencies absent and a Torch version different from the checkout's requirement. The selected runtime is stale/incomplete; it was not repaired or silently substituted. The failure and command remain in scratch evidence. No model weights, generation, Trainer, scientific output directory or Slurm allocation were created. The small resource values in this preflight fixture are placeholders and are **not** RFD4 measurements.

Therefore, actual RFD4 native composition/execution, RFD4/AF3 GPU and container behavior, model reuse parity, throughput and memory remain unqualified. The earlier generic synthetic L40 test does not qualify them. Use a complete declared native runtime for the next CPU preflight and bounded GPU qualification before a scientific rollout.

A follow-up source audit distinguished declared Slurm resources, native Lightning requeue, completed-output reuse and explicit beam continuation; the findings are in the RFD4 guide. Small code cleanup removed duplicate full-source verification on launch, centralized the unchanged campaign-ID formula, and deleted an unused retry-policy helper. The active retry authority and recovery semantics were preserved. The subsequent full suite passed **210 tests in 39.25 seconds, with no skips**; lint, formatting, builds, documentation links and package privacy checks passed. This audit did not run native inference or a new cluster recovery experiment. Generated marimo session metadata was archived to scratch and is now ignored by Git.

## Dashboard extension — October 12 local / October 13 UTC

The live overview and standalone HTML export passed **231 tests in 18.07 seconds,
with no skips** in the integrated suite. The 24 dashboard/data/HTTP tests passed
again after final frontend adjustments. Lint, formatting, JavaScript syntax,
builds, local documentation links and packaged static assets were checked.

Chrome tested the real synthetic CPU records on desktop and a 390-pixel mobile
viewport: automatic refresh, search/filter, campaign/task/log navigation, offline
failure with retained data, reconnect, no page overflow and no JavaScript errors.
The exported HTML rendered both charts without a network request. The demo is
explicitly synthetic: 28 imported tasks, 22 successes, one intentional exit-7
failure and five unexecuted tasks. An initial threaded test harness was rejected
by the runner's main-thread signal guard; its evidence is retained. The corrected
harness used separate processes. No scientific or RFD4 workload was launched.

Slurm job **29096101** ran the unchanged dashboard hosting template inside a
bounded test wrapper on `cpu` with one CPU, 512 MiB and a five-minute limit. It
completed successfully; HTTP/assets/API, caching, disabled submission, unchanged
state files and server cleanup passed. Observed cgroup/accounting peak was
**42.75 MiB**. The preceding probe **29096079** reached HTTP successfully but failed
an outdated product-title assertion in the probe; both its failure and cleanup
remain recorded. Neither test job remains queued or running. Compute-node SSH
forwarding and automatic permanent hosting were not tested.

The 28-task local snapshot scan took 0.108 seconds. A separate **metadata-only**
1,000-task fixture, never submitted, took 4.46 seconds and used about 36 MiB reader
RSS; the adaptive cache extended its interval to 22.3 seconds. These observations
do not qualify very large state directories or worst-case memory. Full scans on
shared storage can be slow; the viewer limits scans and labels truncated results.
GPU utilization/VRAM, scientific pass rates and within-inference progress are not
measured by this dashboard. The RFD4 qualification boundary above is unchanged.

## Circuit dashboard polish — October 13

The visual update adds an SVG circuit diagram and a restrained teal/warm-neutral
theme. A separate Claude design consultation informed the palette, trace/pad
treatment and motion limits; local agents implemented and independently reviewed
the frontend. Scheduling, task ownership and backend polling were outside this
change. Existing concurrent backend work was preserved.

The **24 dashboard/data/HTTP tests passed in 1.46 seconds**. Relevant Ruff checks,
JavaScript syntax and browser checks passed. Chrome verified exact counts from
the existing synthetic CPU records, existing filters/detail links, desktop and
390-pixel mobile layout, zero and six-digit counts, partial/unknown/canceled
states, keyboard pause, reduced motion, hidden-page handling, stale snapshots and
heartbeats, offline retention, and static exports with no network requests.
Actual SVG dash offsets advanced during the moving test; paused and exported
views had no CSS animation. The three-second video is explicitly a **browser
fixture**, not observed cluster traffic. No jobs were submitted for this polish;
owned preview/browser processes were stopped afterward. Earlier harness-only
failures (keyboard injection and an outdated rectangle assertion) were retained.

Evidence and the final HTML preview are under
`/net/scratch/woodbuse/hyper-hyperqueue/dashboard-polish-20261013/`.
`browser/result.json` binds tested asset/export hashes;
repository `.local/dashboard-polish-validation.json` indexes checks, the Claude
consultation and cleanup. These frontend checks do not expand RFD4 runtime or
large-workspace qualification.

## Private evidence index on this installation

- Discovery and resolved profile: repository `.local/discovery/cluster-report.md`, `.local/cluster.yaml`.
- Storage, pooled CPU, controller restart, planned handoff, interrupted recovery and cleanup: `~/.local/state/hyper-hyperqueue-validation/20261012T231029Z/` (`readiness-verified.json`, `standby_recovery_result.json`, `hq_validation_cleanup.json`, `manual_restart_qualification.json`).
- Direct array / failed-only retry: `~/.local/state/hyper-hyperqueue-validation/direct-20261012-v2/result.json`.
- Original failed NFS probe: `~/.local/state/hyper-hyperqueue-validation/direct-20261012/`.
- Direct-only wheel environment: `/net/scratch/woodbuse/hyper-hyperqueue/direct-only-env-validation/result.json`.
- Passing GPU probe: `~/.local/state/hyper-hyperqueue-gpu-validation/20261012T233953Z_healthy_device/` (`execution_pass.json`, `cleanup.json`, `integration_result.json`).
- Earlier GPU probes: sibling directories `20261012T232302Z`, `20261012T233033Z_flex`, and `20261012T233401Z_cuda_visibility`, with separate failure and cleanup evidence.
- Final test report: `/net/scratch/woodbuse/hyper-hyperqueue/pytest-final-20261012T2341.xml`; private handoff index: repository `.local/final-validation.json`.
- RFD4 adapter extension: `/net/scratch/woodbuse/hyper-hyperqueue/rfd4-adapter-validation/` contains `pytest-final-v2.xml`, the exported `notebook.html`, and `native-preflight-result.json` with its stdout/stderr and reproducible preparation script. Repository `.local/rfd4-validation.json` records final source hashes and validation scope. Earlier failed intermediate tests remain separately recorded.
- Follow-up resource/requeue audit and cleanup: the same scratch directory contains `pytest-review.xml` and archived `marimo-session/`; repository `.local/rfd4-review-validation.json` records the reviewed source hashes.
- Dashboard: `/net/scratch/woodbuse/hyper-hyperqueue/dashboard-validation/` contains `pytest.xml`, `browser/` screenshots and browser checks, `demo-v2/dashboard.html`, local measurements, `metadata-only-1000/benchmark.json`, and both Slurm hosting probes. Repository `.local/dashboard-validation.json` indexes the final evidence and source hashes. Preview servers and browser processes were stopped after validation.

Runtime metadata is outside the checkout; scratch holds disposable test environments and bulk output. No remote Git repository was created or pushed.

The preceding publication statement describes the original October 12 validation.
On October 14 the owner authorized public publication; the repository was published
at https://github.com/SethWoodbury/hyper-hyperqueue after the checks recorded below.


## October 14 — finite-batch cleanup and remaining-only recovery

`uv run pytest -q`: **347 passed** (27.61 s); `uv run ruff check src tests examples`:
clean. The live AChE marimo II.F ran without cell errors; `marimo check` passed.
Three agents implemented lifecycle/recovery and independently reviewed races.

Evidence includes a real pinned HQ server + local CPU supervisor stopping its
controller and HQ subprocess after a synthetic completed command
(`tests/test_completion_integration.py`). No Slurm calls occur in that test.
Mocked Slurm tests verify positive ownership, server-side pending-only cancellation,
pending-to-running races, lost acknowledgements, terminal accounting requirements,
and unregistered/paused/unknown work preventing shutdown. Separate threaded tests
exercise imports racing with final shutdown. These are not live Slurm cancellation
or production-scale qualification.

Recovery tests exercise unfinished-only export, successful-command exclusion,
retirement of original occurrences, interrupted publication, preserved validators,
environment/shell/retry settings, explicit resource overrides and rejected tampering.
An actual no-service import repeats idempotently and rejects subsequent resource
changes under the same recovery identity; successful originals execute once.

Read-only existing beta observation: 26/26 commands succeeded; controller 29105296
COMPLETED after 23:58:11 at 2026-10-14T00:49:43, no queued user jobs. It ran the previous
configuration and ended near its walltime; this does not validate automatic cleanup.
No scientific jobs were submitted, canceled or rerun, no live service was restarted,
and frozen beta snapshots were not changed. Future private/example configurations
and notebook controls use 15 s polling with stop_when_complete enabled.


## October 14 authenticated dashboard browser check

Independent review added per-launch access tokens, separate CSRF checks, private
exports, bounded detail reads, safe handling of malformed records and neutral
unfinished-task display. Focused security/dashboard tests: 36 passed. A real
headless Chrome check then verified authenticated data/detail access, removal of
the token from the address bar, hidden submission controls in read-only mode, no
JavaScript exceptions, and desktop/mobile layouts without horizontal overflow.
The screenshot in `docs/images/dashboard.png` uses preserved synthetic records
(28 tasks, 22 succeeded, one failed, five ready). This check neither queried Slurm
nor submitted jobs.


## October 14 — live CPU dispatch and automatic stop

A bounded synthetic Slurm qualification completed **1,000/1,000 validated tasks**,
with **1,000 unique executions and zero duplicates**. Controller **29198031** used
66 one-CPU workers (32 `cpu`, 34 `cpu-bf`); 65 received tasks. The two-second
commands ran over **44.84 seconds**. Median gap from one runner's completion to
the next runner's start on the same worker was **0.171 seconds** (P90 **0.202 s**);
this includes filesystem, validation and runner overhead, not pure HQ latency.

All 67 allocations completed, no owned jobs remained, and HQ/controller stopped
automatically **81.9 seconds after the final command**, including the configured
60-second worker idle grace and accounting/reconciliation. Recorded consumption
was **1.78 worker core-hours + 0.038 controller core-hours**. All workers started
promptly, so this run did not exercise live surplus-pending cancellation; that
path remains covered by ownership/race tests. These are CPU dispatch measurements,
not RFD4/GPU throughput or scientific qualification.

Two preceding probe revisions are preserved as failures: the first task fixture
used an incorrect output environment variable, and the second validator used a
relative path (resolved from task cwd) instead of `{attempt_dir}/output/result.json`.
The third revision first passed the exact runner and validator locally, then ran
the batch. All three revisions consumed about **5.59 allocated CPU core-hours**
in total; no GPU hours. The first revision also exposed a real NFS timing bug:
HQ observed process exit before the final result appeared. The fix allows a
matching final result to supersede only provisional `interrupted/worker_lost`
evidence; explicit cancellation, retry and terminal decisions remain authoritative.
The focused late-result/state/recovery suite passed **60 tests** and independent
review passed 35 relevant cases.

Evidence: AChE V3 `slurm/scheduler_comparison_30/qualification/` contains the three
`cpu_throughput*` namespaces, `summary.json`, batched Slurm accounting and the
reusable `analyze_cpu_probe.py`. The corrected result is `cpu_throughput_r03`;
failed revisions were not overwritten.

## October 14 pre-launch source checks

The combined source passed **561 tests** and `ruff check src tests examples`;
`git diff --check` was clean. Both source distribution and wheel built with
`uv build`. These checks include the CLI module invocation/configuration fix,
late-NFS result reconciliation, bounded OOM recovery, strict pending dependency
controls, same-placement worker fallback, and the RFD4 interrupted-row contract.
This count is unit/local-integration evidence, not a throughput or GPU-quality
claim. Real Slurm CPU evidence is recorded separately above.

## October 14 — actual pending-only cancellation

A held CPU job, **29199098**, exercised the actual `cleanup_workers()` path and
Slurm `scancel --ctld --state=PENDING`. A deliberately wrong ownership token was
refused first; the correctly owned pending job was then canceled. Accounting
confirmed `CANCELLED`, zero elapsed allocation time, and no node assigned.
This qualifies the real pending cancellation RPC and ownership check. It does not
replace mocked pending-to-running race tests or demonstrate a whole live HQ
backlog draining to pending cancellation.

The small probe initially requested less than this site's 15-minute minimum and
was rejected without a job ID. Its cancellation then succeeded, but the probe's
final accounting call omitted the required bounded start time; that fixture was
corrected and terminal accounting verified separately. All evidence is preserved
under AChE V3 `slurm/scheduler_comparison_30/qualification/pending_drain_real/`.

## October 14 — allocation telemetry qualification

The subsequent source passed **566 tests in 39.89 seconds**, with Ruff clean and
independent review. Added telemetry locates the exact Slurm job's cgroup-v2
ancestor and reports sampled `memory.current` separately from lifetime
`memory.peak` (which includes earlier tasks). It queries VRAM only for verified
assigned GPU UUIDs, at most once per ten seconds with a one-second query timeout.
CPU tasks make no GPU query. Missing/unsupported measurements remain unknown and
cannot fail a scientific command. Device usage is not process-specific memory,
and brief VRAM peaks can be missed. These source tests precede live RFD4
qualification of the new measurements.
