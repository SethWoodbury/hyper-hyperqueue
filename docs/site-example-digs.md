# DIGS site example: limits and accounting

Read-only observation on **2026-10-14**. These are site settings, not portable
HHQ defaults. Recheck them before a new large campaign; Slurm remains authoritative.

| Partition | Visible per-user running-job limit | Other relevant limits |
|---|---:|---|
| `gpu` | 128 through partition QoS `gpu` | 192 allocated GPUs; additional model-specific TRES limits |
| `gpu-train` | 8 through partition QoS `train` | 64 allocated GPUs |
| `gpu-bf` | No explicit limit found for the inspected user/account | Preemptible; available resources and other site policy still apply |

The inspected backfill association uses QoS `normal`, which has no per-user job
or GPU TRES cap. Its user, parent account and root associations also showed no
job/TRES cap. This does **not** mean unlimited available GPUs or a guaranteed
allocation. A site submission plugin is present; not all policy is discoverable.

An actual bounded submission exposed an additional rule: regular `gpu` rejects
untyped `--gres=gpu:1` requests. Use `gpu:small:1` and `gpu:large:1` profiles sharing
one regular-GPU budget to cover both currently advertised classes, without a GPU
model constraint. Inventory alone did not reveal this rule. On the same date,
`sbatch --test-only` accepted generic `gpu:1` for `gpu-bf` (five hours) and
`gpu-train` (twelve hours); these checks allocated no GPUs. A successful planning
check is not model/kernel compatibility qualification on every physical GPU.

For the inspected RFD4 image specifically, a real Titan RTX (SM75) passed BF16
matrix multiplication and TransformerEngine linear operations but failed the
native pair-biased attention path with its configured backends (`Invalid backend`).
The application comparison therefore adds `Ampere|Lovelace|Hopper|Blackwell` to
both backends' routes, covering the site's currently observed newer generations.
This is an application compatibility constraint, not a generic HHQ restriction.
Titan nodes can belong to regular `gpu` as well as backfill; selecting `small`
alone does not guarantee this workload's hardware compatibility.

The application's two-running/one-pending training setting is intentionally lower
than the site limit. A running target is not a Slurm hard throttle: pending jobs
can start between reconciliations. Shared outstanding caps include other jobs by
the same user, including a concurrent native benchmark.

## Accounting

`priority/multifactor` is enabled, with fair-share weight 100000 and historical
usage half-life two days. The visible `gpu` and `gpu-train` billing weights are
`CPU=1.0,Mem=0.25G,GRES/gpu=4.0,GRES/gpu:large=8.0,GRES/gpu:xlarge=12.0`.
Backfill has no explicit weights and therefore uses Slurm's default CPU billing.
The observed job QoS usage factors are 1.0. Use each allocation's actual
`AllocTRES` billing value rather than guessing the GPU-class weighting.

Pending cancellations consume no allocated time. Started idle workers consume
allocation time; finishing an eight-hour request after two hours records two
hours of allocated usage. Neither GPU utilization nor process peak RAM determines
the allocation billing weight. Requested resources and elapsed allocation time do.

Distributing work between partitions can change weighted usage and waiting time.
This user has partition-specific associations; their usage/fair-share values can
differ, while parent-account activity remains shared. Before/after snapshots must
include the partition and also reflect other activity and decay. They cannot
attribute a priority change uniquely to HHQ versus another backend.

```bash
scontrol show partition -o
scontrol show config
sacctmgr show qos format=Name,Flags,MaxJobsPU,MaxSubmitJobsPU,MaxTRESPU,UsageFactor
sacctmgr show assoc where user="$USER" format=Account,User,Partition,QOS,MaxJobs,MaxSubmitJobs,GrpTRES
python -m hyper_hyperqueue.accounting --jobs 12345 12346 \
  --qos-factor normal=1 --qos-factor gpu=1 --fairshare-user "$USER" \
  --output /shared/private/allocation-report.json
```

The report retains partial/missing accounting, separates each partition's GPU-hours,
CPU-hours, allocated-memory GiB-hours and billing-unit-hours, and never submits or
cancels work. For actual GPU model and UUID, join worker/task telemetry; a Slurm
type such as `small` is not the physical model. Compare queue delay, completion
time, failures and model-loading overhead alongside compute usage.

References: [Slurm resource limits](https://slurm.schedmd.com/resource_limits.html),
[TRES billing](https://slurm.schedmd.com/tres.html),
[multifactor priority](https://slurm.schedmd.com/priority_multifactor.html).
