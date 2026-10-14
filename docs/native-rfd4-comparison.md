# Native RFD4 and HHQ: capability audit

Reviewed against the inspected RFD4 `main` checkout on 2026-10-14. HHQ is an
allocation/dispatch layer, not a replacement for RFD4 scientific APIs. A universal
speed or priority advantage has not been demonstrated.

## What the comparison measures

The large comparison uses identical scientific settings and different seeds in
both arms: pooled HHQ workers versus static Slurm arrays rendered using native
RFD4 submission helpers. **Both arms use the same warm scientific driver.** This
isolates scheduling and allocation effects; it does not compare the unchanged
native launcher against HHQ end to end. A separate direct-native control checks
that distinction. Report allocation GPU-hours, wall time, GPU models and actual
billing TRES separately. Fair-share snapshots are context, not a causal estimate
of the effect of one campaign.

## Features and remaining differences

| Capability | Native RFD4 | HHQ RFD4 path |
|---|---|---|
| Loaded model reuse | One model over a multi-input dataset and replicates | Cached model across compatible rows within a bundle; per-row Trainer teardown may move it to CPU |
| Compilation reuse | Length buckets and persistent compiler cache | Native settings remain available; shorter row passes can amortize setup less well |
| AF3 daemon | Launcher can start and own it | Owned host bridge, native prewarm hook and daemon reuse within a bundle |
| AF2 daemon | Dedicated launcher support | Not qualified in the specialized warm adapter; excluded from this comparison |
| Beam continuation | Saved states, explicit new resume seed | Native configuration can request it; pin the state file and use new outputs; never automatic retry |
| Interruption | Native requeue/skip-existing; not exact diffusion checkpointing | Generic retry; validate and skip completed row receipts; preserve partial row outputs and restart that row |
| Host OOM | No measured adaptive RAM ladder in the inspected launcher | Optional bounded original-RAM multipliers, backed by specific failure evidence |
| Multiple queues | Explicit launcher resource requests | Shared backlog across configured partitions and optional pending reserves |
| Multi-GPU allocation | Native fan-out across GPUs | Generic commands can request GPUs; specialized warm RFD4 adapter currently qualifies one GPU per worker |
| Aggregation | Dependent per-run and sweep concat jobs | Existing collector callable through CPU submission after logical completion; not automatic scientific DAG creation |
| Foldseek inputs | Launcher can enable chain-of-interest saves and clustering | Set scientific saves explicitly and call native aggregation; no silent injection into frozen configs |
| Source isolation | Optional copied code checkout | Source hash pins fail if modified; a separate frozen checkout avoids interrupting active work |
| Resume provenance | Native output checks | Immutable task identity, ownership and artifact checks; stronger checks can require more recomputation |

Ordinary native inference commands and configurations can also be submitted
through generic HHQ. An unqualified feature in the specialized warm adapter is
not removal of that native capability. Do not put native **Slurm-submitting**
launch scripts inside pooled tasks: HHQ should own the allocation, and the task
should execute the inference command inside it.

An actual repaired warm control completed six gradient roots and four beam
roots across different inputs, with real AF3 outputs and complete saved motif
mapping. A separate stock-inference control also completed. The first warm beam
row took about 517 seconds versus about 530 seconds for the standalone native
control process; this is not evidence of a general speedup. Their generated
sequences differed despite equal declared seeds, and these are only small
controls. Repeated beam proposals can share sequences: report unique sequences
separately from structure and AF3 file counts.

## Qualification criteria

1. Check input/config/seed/output mapping across cold and warm rows, including
   different inputs, methods and settings. Equal seed does **not** guarantee equal
   structures between these paths: skipping model initialization can change the
   RNG stream before diffusion. Record this rather than calling it exact replay.
2. Interrupt a multi-row bundle after a completed row. On retry, verify the
   completed row is not recomputed, partial evidence survives, and only unfinished
   work runs. Reject tampered completion artifacts.
3. Verify AF3 actually executes on an eligible design and reports the intended
   ligand/sequence. Merely starting a daemon is not an AF3 inference test.
4. Check native/container environment, bindings, threads and allocator settings.
5. Distinguish unit/simulated tests, local real-HQ tests and real Slurm/GPU tests
   in the validation record. A feature is not production-qualified by inspection.

Native implementations inspected: `scripts/launch_generate.sh`,
`scripts/launch_sweep.sh`, `scripts/common.sh`, `src/rfproteina/generate.py`,
beam-search and oracle implementations. Native `launch_search.sh` is a separate
workflow; this comparison uses generated native inference configurations, not
that launcher's two-phase interface.
