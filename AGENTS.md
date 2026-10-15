# hyper-hyperqueue

## Scope and ownership

Keep this a thin configurable Slurm layer around pinned HyperQueue 0.26.2.
Slurm allocates resources, HQ dispatches tasks, and HHQ is the only allocation
owner. Direct arrays must work without importing or starting HQ. Keep application
logic under optional integrations; generic scheduling must not import RFD4.

Read README.md, docs/architecture.md, docs/recovery.md and docs/limitations.md
before changing ownership, replay, resources or output semantics. Preserve
immutable task/configuration snapshots. Never use heartbeat age alone for takeover,
remove leadership locks, or treat accepted-but-unacknowledged submission as failure.
Keep raw cluster discovery, credentials, runtime state and private command files
outside Git. Do not use SQLite WAL on shared network filesystems.

## Current authorization and validation

October 14: the owner explicitly authorized public GitHub publication, bounded
CPU/GPU qualification, and the prepared AChE comparison after checks pass. This is
application authorization for that campaign, not blanket permission to launch or
migrate other campaigns. Preserve existing results and preparation receipts.

Use bounded synthetic tests for generic recovery/dispatch; only cancel positively
identified owned test jobs. Test resource-heavy scientific paths separately and
label evidence accurately. Unit tests, real local HQ tests and real Slurm/GPU
qualification are different. Record live evidence in docs/validation.md; failed
probes remain evidence. Native comparison scope is documented in
[docs/native-rfd4-comparison.md](docs/native-rfd4-comparison.md); never claim
universal superiority without measurements.

Run `uv run ruff check src tests examples`, focused tests, then the full suite for
ownership/recovery changes. Workers must not install dependencies at startup.
HQ integration tests require the separately installed pinned executable.

## Scheduling and recovery invariants

- Queue profiles, hard caps, pending floors and optional running targets are
  separate. Site limits are discovered facts; notebook fleet targets are choices.
  The reference deployment prefers gpu, then optional gpu-train, then gpu-bf.
- Reconciliation defaults to 15 seconds; idle workers default to 60 seconds.
  Optional speculative workers reserve multiple queues for one shared backlog,
  never duplicate application tasks. Unknown submissions still reserve capacity.
- `pending_reserves_after_growth` fills running targets before reserve floors.
  A strict owned running cap can use an OR-afterany dependency for a pending reserve;
  do not imply it changes site-wide limits or other independent campaigns.
- Pending-only early cleanup requires actual task assignment and exact job ownership.
  Never cancel a running worker to clear a pending reserve. Finite-batch shutdown
  waits for settled tasks, retries and owned worker termination. Unknown work blocks.
- Planned CPU renewal is opt-in. Successor leadership requires graceful handoff,
  flushed journal, old worker terminal proofs and the lifetime lock; no unplanned
  stale-heartbeat takeover. Stop/completion veto renewal.
- Retry only unfinished tasks after positive termination proof. Explicit user cancel
  wins. Task timeouts differ from allocation expiry. Generic exit 137 is not OOM proof.
- Host OOM escalation uses immutable original RAM multipliers and attempt bounds;
  GPU VRAM OOM must not inflate host RAM. Pooled CPU allocation OOM does not identify
  a specific culprit without its own evidence. Oversized retries must not inflate
  unrelated baseline allocations.
- Optional GPU-hour ceilings reserve full requested time for unresolved allocations.
  This is conservative admission accounting, not consumed time or Slurm billing.
- One-phase fallback on an existing allocation must satisfy placement, resources,
  task eligibility and current controller ownership; no new allocation owner.

## RFD4 integration

Read docs/rfd4.md. Scientific notebook and native repository are external. Preserve
row input/config/seed/output mapping; do not silently introduce scientific settings.
Preparation never admits READY tasks. Disable native Lightning Slurm requeue: HHQ
owns retries. Hash pins require stable source or a dedicated frozen checkout.

Use docs/rfd4-measured-resources.md and its aggregate recipe CSV as dated
calibration, not universal defaults. Current application follow-up uses integral
22 GiB TOTAL allocations for gradient/beam, with worker reserve inside. Future
application bundles target about 240 minutes using explicit recipe/device costs;
the older two-hour preference is historical. Generic memory stays MiB and generic
hardware profiles stay neutral. RFD4 method profiles apply the site-specific
modern-GPU compatibility guard unless explicitly overridden. Do not infer
AF3-only Titan failure from the demonstrated RFD4 attention incompatibility.

Native model caching is not proof of continuous GPU residency or exact RNG parity.
Qualify warm/cold behavior and actual AF3 inference. Completed row receipts require
artifact validation. Preserve unfinished row evidence before restarting that row.
Beam `resume_from`/`resume_seed` creates a separate scientific continuation; never
choose a newest checkpoint automatically or call it exact interrupted recovery.

The host oracle bridge owns foreground children and private per-task scratch;
never adopt an unrelated daemon or background a competing owner. Native multi-GPU,
AF2 launcher and aggregation behavior must not be claimed qualified by the current
single-GPU AF3 comparison.

## Dashboard

The optional dashboard reads cached state and never polls Slurm on browser refresh,
probes leadership locks or decides recovery. Bind loopback, require the per-launch
session token, and retain separate CSRF protection for opt-in controls. Exports
contain private campaign metadata and must be private files. Preserve unknown
liveness and missing measurements; never call unfinished work successful.
