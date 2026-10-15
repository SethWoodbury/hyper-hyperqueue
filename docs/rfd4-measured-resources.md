# Measured RFD4 resource reference

These are descriptive measurements from one completed-work snapshot on
2026-10-14 at 22:35 UTC, using a pinned RFD4 development runtime and the same
warm scientific driver under pooled workers and static arrays. They are starting
points for similar jobs, not portable performance guarantees or backend rankings.
Only aggregate measurements are published; private inputs and execution records
remain outside this repository.

## Workload and interpretation

- Each command requested **two replicates**, with 60 diffusion steps, medium/high
  thinking, backbone noise 0.3 and latent noise 0.1. Gradient used late backbone
  or backbone+latent guidance at strengths 1e-5/1e-4. Beam used widths 2/4,
  branches 2/4, and early/late AF3 schedules. The requested protein length range was
  120–260 residues. Runtime depends on the sampled length and AF3 admission.
- AF3 used one sample, five recycles, an explicit fixed prediction seed, and no
  supplied MSA/template. It ran only when the configured admission accepted a
  design. No AF2 or vanilla-generation timing is included in this dataset.
- Tables use **1,405 completed rows** linked to successful attempts with a known
  GPU model. Twenty other completed rows lacked this successful-attempt join and
  are excluded. Rows share allocations and are not independent benchmark repeats.
- “First” means the first completed row in a bundle attempt; “later” means a
  subsequent row in the same process. These are cold-start-including versus warm
  observations, not isolated model-loading measurements. Different inputs,
  lengths, AF3 call counts and hardware mix prevent a causal cold/warm speedup.
- Memory is the **whole-job cgroup lifetime peak**, including file cache and
  earlier tasks, not process RSS or GPU VRAM. Native gradient allocations could
  contain two bundles, so memory sample counts can be smaller than first-row
  counts. Summed process RSS double-counts shared pages. Percentiles below use
  nearest rank (`ceil(p*n)`); p90 from few samples is particularly uncertain.

## GPU and method measurements

Times are seconds **per two-replicate command**, including any admitted AF3 work.
Different recipes are mixed in this summary; use the recipe table for planning.

| GPU | Method | First rows: n / median s | Later rows: n / median / p90 s | Memory samples: n / p90 / max GiB |
|---|---|---:|---:|---:|
| A100 80 GB | beam | 13 / 391 | 59 / 285 / 784 | 13 / 19.25 / 19.25 |
| A100 80 GB | gradient | 3 / 114 | 47 / 30 / 52 | 3 / 19.54 / 19.54 |
| L40 | beam | 5 / 531 | 25 / 335 / 484 | 5 / 16.08 / 16.08 |
| L40 | gradient | 1 / 244 | 9 / 40 / 67 | 1 / 17.68 / 17.68 |
| L40S | beam | 13 / 383 | 63 / 407 / 648 | 13 / 17.78 / 19.25 |
| L40S | gradient | 4 / 117 | 56 / 30 / 49 | 4 / 17.97 / 17.97 |
| RTX 4000 Ada | beam | 7 / 743 | 35 / 585 / 1701 | 7 / 16.90 / 16.90 |
| RTX 4000 Ada | gradient | 2 / 169 | 28 / 56 / 91 | 1 / 16.92 / 16.92 |
| RTX A4000 | beam | 53 / 1084 | 258 / 923 / 2453 | 53 / 17.00 / 18.37 |
| RTX A4000 | gradient | 11 / 261 | 159 / 97 / 144 | 8 / 17.02 / 17.02 |
| RTX A5000 | beam | 2 / 860 | 8 / 781 / 1552 | 2 / 17.45 / 17.45 |
| RTX A6000 | beam | 44 / 689 | 220 / 552 / 1232 | 44 / 16.47 / 16.84 |
| RTX A6000 | gradient | 8 / 193 | 112 / 54 / 78 | 4 / 16.71 / 16.71 |
| RTX PRO 4000 Blackwell | beam | 20 / 599 | 100 / 486 / 1083 | 20 / 16.92 / 17.76 |
| RTX PRO 4000 Blackwell | gradient | 3 / 187 | 37 / 44 / 71 | 3 / 16.67 / 16.67 |

The [recipe/thinking CSV](rfd4-measured-recipes.csv) separates GPU model, thinking,
AF3 stage schedule, beam width/branches and gradient modality/strength. It includes
86 groups with at least six observed rows, retaining separate first/later counts,
medians and p90s. Six rows can be just one bundle: a listed p90 is an observation,
not a reliable tail bound. Missing combinations have no measured estimate here.

## Memory requests

The historical comparison requested **20.25 GiB total for gradient and 19.25 GiB
for beam** (application RAM plus 256 MiB worker reserve). Whole-job observed
maxima were 19.54 and 19.25 GiB respectively. The beam maximum equals its limit;
no recorded host OOM does not rule out reclaim pressure. That comparison did not
preserve `memory.events` counters.

The current bounded follow-up uses **22 GiB total and two CPUs for either method**.
The worker reserve is inside that total; do not add another 0.25 GiB to the Slurm
request. Notebook fractional total-GiB inputs round down as requested by its
operator. Choose an adequate integer first: rounding a limit below a measured
working footprint is not an optimization. Generic HHQ task resources remain MiB;
the profile must fit task RAM plus its reserve. Generic OOM multipliers can still
produce non-whole-GiB requests unless the caller applies an explicit policy.

Keep memory and time estimates separate. CPU cgroup usage, process RSS and
assigned-GPU VRAM are different measurements. Record pressure/reclaim counters
on further controls before reducing RAM from the median alone. Requalify higher
lengths, batch sizes, beam widths, oracle settings or new software revisions.

## Planning longer warm bundles

The current campaign preference is a **240-minute target per warm bundle**,
superseding its earlier two-hour target. This is a configurable application
planning target, not an HHQ timeout, mandatory duration or global scheduler
setting. A shorter recipe tail should finish early. Worker walltime can remain
8 hours on regular GPU, 5 hours on backfill and 12 hours on opt-in training;
those are ceilings, not consumed GPU-hours.

Use a source-conditioned estimate: the same method, recipe/thinking, GPU class,
length range, replicate count and AF3 protocol/admission policy. Estimate a bundle
as first-row cost plus later-row costs, then allow uncertainty and cleanup before
the worker's expiry. A four-hour target on a five-hour backfill allocation leaves
only one hour for that uncertainty and cleanup. Do not multiply an all-recipe
median by a command count and call the resulting tail safe.

For example, expensive early beam at width 4 / branches 4 can take tens of
minutes per command on A4000; lighter beam or gradient on a faster GPU can take
much less. Six historical beam commands reached 313 minutes in one successful
allocation, exceeding the five-hour backfill ceiling even though beam p90 was
140 minutes. Choose fewer expensive rows and more cheap rows. Longer bundles
reduce reloads but expose more unfinished work to preemption; completed-row
receipts preserve finished rows, not a partly completed trajectory.

If one shared task pool serves heterogeneous GPUs without speed-specific routing,
size each recipe against a conservative qualified GPU estimate and let faster
workers consume more bundles. They reload at bundle boundaries today. Automatic
cross-GPU timing adaptation is not implemented; update future preparations from
measurements and keep published plans unchanged. The recipe CSV is a calibration
reference, not a scheduler input or automatic packer.

The current campaign uses the A4000 median for each exact recipe with at least
three observed rows, a 10% margin, and an explicit startup allowance. This avoids
treating a fast-GPU-only measurement as evidence for slower eligible GPUs. Missing
A4000 recipes use declared, unmeasured fallback costs of 300 seconds for gradient,
3,600 for beam, and 600 for vanilla generation. These are editable campaign
assumptions, not generic HHQ defaults or measured vanilla performance. The plan
prints which recipes use fallback; changing the timing GPU does not restrict
Slurm eligibility. A4000 medians still do not guarantee tail fit.

## Warm execution, matching and hardware

Actual daemon logs show one AF3 worker reused across many predictions per bundle;
RFD4 caches compatible model objects between rows. Caches end with the bundle.
Per-row Trainer teardown may offload RFD4 weights to CPU, so reuse does not prove
continuous GPU residency. First/later AF3 timings also depend on shape compilation
and admitted workloads.

Six earlier GPU-model/recipe/input-matched beam pairs used different seeds and
performed 857 versus 818 AF3 predictions. One pair contributed about 98% of their
aggregate timing difference. They do not establish a 5–7% scheduler penalty.
A small same-declared-seed follow-up also produced different sequences and AF3
admissions; its timing must not be presented as equal-work inference speed.
RNG initialization, realized lengths, branches and oracle calls need checking
in addition to the seed field.

The observed working models are those in the table. The pinned RFD4 attention
path failed on Titan RTX despite simpler BF16/TransformerEngine probes passing.
Use the RFD4-specific qualified generation constraint; generic HHQ remains
hardware-neutral. A separate real AF3-only Titan test also failed in cuDNN SDPA
with an explicit SM80-or-newer requirement, before producing a structure. It used
8.20 GiB of 18 GiB host RAM and sampled 4,284 MiB of 24,022 MiB GPU memory; this
was an attention-architecture failure, not evidence of OOM. The inspected AF3
image was `af3-open-2026-08-19.sif`; alternate attention backends remain untested.
H200 is unmeasured here, not inherently
unsupported. Temporary unhealthy nodes require separate, revisitable exclusions.
See [site-specific compatibility and billing](site-example-digs.md) and the
[native feature comparison](native-rfd4-comparison.md).
