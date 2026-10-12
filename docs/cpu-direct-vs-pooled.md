# CPU: direct or pooled

Both paths use the same immutable tasks, runner, logs and completion checks.

| Situation | Useful initial choice | Reason |
| --- | --- | --- |
| Many independent, long CPU commands | Direct | Slurm handles each command; no control service lifetime dependency. |
| Many short commands | Pooled | Reuse allocations and pack CPU/RAM across commands. |
| CPU runtime unknown | Auto initially selects direct | Avoid assuming unknown work is short. Override explicitly if appropriate. |
| MPI or multiple nodes/ranks | Direct | Allocation layout is explicit; the command must include its own `srun`/MPI launcher. |
| GPU campaign | Pooled | Long allocations consume many individually tracked commands. Direct GPU is also allowed. |

Auto uses a configurable CPU duration threshold (`policy.short_cpu_seconds`, initially 300). A profile's explicit `execution_mode` is honored before the generic auto rule. An explicit CLI mode always wins if feasible. `hhq plan` returns the decision reason and rejected profiles. Measurement recommendations can be supplied as runtime/resource hints; the initial planner does not silently fit a model to arbitrary shell commands.

## Direct array limits

A 1,000-line file means 1,000 command occurrences, not 1,000 `sbatch` RPCs. Arrays split on shape, array bounds and outstanding budget. The default is one command per element. An array `%N` is an array-local running throttle; hhq divides a requested `max_parallel` across all its outstanding arrays. It does **not** copy `%N` to every split array.

This conservative reservation can temporarily underfill as parts of arrays finish. `resume CAMPAIGN_ID` reconciles and submits eligible remainder. The receipt reports `unsent_tasks`; unsent work is never represented as scheduler-pending. A terminal/CLI/HQ outage cannot stop arrays already accepted by Slurm.

## CPU packing

Workers request real CPUs and RAM, reserve worker RAM, then advertise the remaining capacity. A four-CPU worker can run four one-CPU commands if RAM permits. Larger tasks consume more slots. OpenMP/BLAS thread defaults follow the declared task CPUs only when the variable is absent; explicit user and inherited settings remain untouched. Oversubscribing via an explicit thread setting is the user's choice.

Pooled one-node/one-rank execution is the current boundary. Unsupported pooled MPI/multi-GPU requests fail before submission; use direct mode for explicit rank layouts. Resource limits do not insert an MPI launcher into commands.

## What the measurements show

Bounded real tests demonstrate both paths. The direct test ran three commands as three elements, preserved two successes and retried only the deliberate failure. Local HQ tests used two actual workers, and the cluster CPU pool reused one worker for initial and appended tasks. These are correctness tests, not enough evidence to declare one path globally faster or to set workload-specific memory for scientific software. See [validation](validation.md).
