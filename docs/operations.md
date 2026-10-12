# Operations

## Configuration and preview

`--config PATH` / `HHQ_CONFIG` selects YAML (JSON is valid YAML). Otherwise hhq checks `~/.config/hyper-hyperqueue/config.yaml`. Explicit CLI task/resource overrides win; campaign manifests snapshot resolved settings. Unknown fields are rejected. Portable examples are in `examples/`; the real local profile and full discovery evidence are gitignored under `.local/`.

```bash
uv run hhq discover --output-dir .local/new-discovery
uv run hhq --config .local/cluster.yaml doctor
uv run hhq --config .local/cluster.yaml config show --resolved
uv run hhq --config .local/cluster.yaml plan commands.txt --mode direct --profile cpu-direct
```

Discovery does not submit application work. Plan is a feasibility/configuration preview, not a promise that Slurm will accept or start the job immediately. Final admission observes existing user jobs and durable unknown submissions. An unavailable scheduler is not interpreted as an empty queue.

Resource units in YAML are explicit integer MiB and seconds. CLI memory accepts `512MiB`, `12GiB`, `512M`, `12G`; time is `HH:MM:SS`. Direct `--cpus/--mem` set the job shape. In pooled mode they describe one command; the configured worker must fit it plus overhead. `--runtime` is an expected duration; `--timeout` is an enforced task timeout. `--time` changes allocation walltime. Direct `--nodes` and `--ranks` request the allocation; your command supplies its explicit launcher.

One active service uses an immutable configuration snapshot. Stop/drain and restart it explicitly before changing pool definitions, global budgets or control configuration. Imports with incompatible active service configuration are rejected rather than silently using new definitions that the controller cannot see. Campaign tasks remain immutable. Pool pause/resume is a durable live control.

For production runs, keep the installed orchestrator environment and workload software fixed until their jobs finish. An editable checkout is convenient for development, but a queued job will execute the code installed at its start; the configuration snapshot does not freeze Python source files. Record a commit/version in workload fingerprints and deploy a new environment for software upgrades.

## Imports and outputs

By default each nonblank line that does not begin with `#` after whitespace is a command. Whitespace/quoting are preserved. A final newline is optional. `--literal` retains blank/comment lines as intentional shell commands. An odd trailing backslash is rejected: use `--format jsonl` with `{"command":"...\n..."}` for multiline shell, or `{"argv":[...]}` for execution without shell parsing. JSONL deliberately accepts only command/argv fields; resource policy is campaign-level through the API/configuration.

Initial import identity binds the snapshot bytes and task settings. Same import, same IDs, no duplicate work. An explicit `--import-key` makes uncertain caller retries identifiable. Reusing that key with different contents/settings fails. Duplicate command lines are separate occurrences. `append` imports a new immutable file; it does not watch edits to the old file. To intentionally execute an identical second batch, provide a fresh append key.

Commands run unchanged in the declared working directory, with original shell syntax; hhq does not enable `set -e` or `pipefail`. The API exposes explicit environment/shell settings. The runner adds `HHQ_TASK_ID`, `HHQ_CAMPAIGN_ID`, `HHQ_ATTEMPT_ID`, `HHQ_ATTEMPT_DIR`, and `HHQ_TASK_OUTPUT_DIR`.

Output modes:

- `manual_on_ambiguity` is the default: unresolved side effects block replay.
- `in_place` preserves declared paths. Automatic interrupted replay requires repeatability, retry bounds and exact fencing evidence.
- `isolated` supplies a fresh per-attempt output directory. The command must actually use `HHQ_TASK_OUTPUT_DIR`; hhq cannot rewrite arbitrary shell paths. Validated completion points to the accepted attempt. Generic atomic publication of application directory trees is not implemented.

The API accepts expected output files, minimum bytes, a validator argv and a collision key. Completion requires exit zero plus every configured validator. Exit-code-only success is recorded explicitly. A collision key protects cooperating hhq tasks, not arbitrary external writers. Do not claim exactly-once external side effects.

## Start and stop

```bash
# Usually submit starts this automatically for pooled work.
uv run hhq service start --hosting slurm --partition cpu --standby 0
uv run hhq service status

uv run hhq service stop             # drain admissions, then stop
uv run hhq service stop --no-drain  # interrupt service; task records remain
```

For local hosting, use a separate selected configuration with `control.hosting: local` and `control.standby_replicas: 0`, then run `hhq service start` and submit with that same configuration. The host must have Slurm access and worker connectivity. Standalone `service start --hosting/--partition/--standby` overrides do not rewrite your YAML; subsequent submissions must match the service snapshot, so put lasting overrides in the selected configuration first.

Direct-only submission does not start a service. A CPU control job never silently switches to `cpu-bf`. A notebook is a client; closing it does not own the lifecycle of submitted jobs. Local hosting is only suitable where the declared endpoint is reachable by workers; no automatic SSH tunnels or public binding are created.

An idle running service continues to consume its modest control allocation until stopped or its walltime ends. Workers exit after the configured idle timeout. Long campaigns need qualified standby/rollover or an explicit recovery/restart plan; a single control job does not provide unattended indefinite service.

## Controls have distinct scopes

| Action | Effect |
| --- | --- |
| `pool pause NAME --scope allocations` | Prevents new workers for that profile; existing workers continue. |
| `campaign drain ID` | Stops new admissions for this campaign; admitted HQ tasks and Slurm jobs continue. |
| `campaign control ID pause-import` | Blocks append imports. |
| `campaign control ID pause-dispatch` | Legacy spelling for pause **admission**, not suspension of already admitted HQ work. |
| `cancel ID --pending` | Direct: cancel positively identified pending elements; unsent work is drained. |
| `cancel ID --running` | Direct: cancel positively identified running elements. |
| `cancel ID --pending --running` | Direct or pooled campaign cancellation; pooled HQ cancellation is job-wide. |
| `retry ID --failed` | Authorizes only eligible known failures within original attempt bounds. |
| `resume ID` | Resumes admission; direct mode also feeds unsent chunks. |

User cancellation does not authorize automatic retries. Repeatable tasks interrupted by *recorded internal generation fencing* can recover only after exact owned allocation quiescence is proved. This distinction is recorded rather than guessed from `SIGTERM` alone.

`submit` defaults to one total attempt. Choose `--max-attempts 2` (or an explicit API `RetryPolicy`) at import if known failures should be eligible for retry. `retry --failed` never increases this immutable limit. Direct mode then uses `resume` to submit the eligible failed-only work; pooled admission is handled by the running controller.

## Diagnose without guessing

```bash
uv run hhq status --json
uv run hhq status --reconcile        # bounded Slurm/accounting query
uv run hhq explain TASK_ID
uv run hhq logs TASK_ID --attempt latest --stderr --lines 80
uv run hhq metrics
uv run hhq support-bundle /absolute/path/support.tar.gz
```

`ready` means unsent; `submitted` means backend-admitted and can still be pending; `running` requires an active attempt. Controller snapshots carry timestamps and decision reasons. `explain` connects immutable source, resources, profile, backend IDs and attempt results; it adds live HQ task explanations when available. Application stderr is separate from scheduler stderr and the exact rendered script.

Unknown submissions reserve capacity. First inspect the intent token, exact Slurm job name, recent accounting and backend logs. Do not delete an intent just to clear a queue. Repeated RPC/startup failures pause growth rather than creating unlimited replacement jobs. Manual override should change an explicit budget, profile or pause control—not erase ownership evidence.

After repairing a repeated worker startup error, use `hhq pool resume PROFILE --reset-startup-failures` to acknowledge the repair and reset that circuit's counting window. Old evidence stays intact. This does not clear an unknown submission or override a budget.

## Browser

```bash
uv run hhq ui --port 8765
# From your workstation, substituting the actual login host:
ssh -N -L 8765:127.0.0.1:8765 USER@CLUSTER_HOST
```

Open `http://127.0.0.1:8765`. The server accepts loopback hostnames, protects mutations with a session token and origin checks, and refuses public binding. Local users with account access are trusted; this is not a multi-user authenticated web application. Detailed logs and commands are private; support bundles use an allowlist and omit them.

## Retention and migration

Keep the whole authoritative state directory on durable shared storage. Stop the service and ensure attempts are quiescent before making a consistent backup with normal filesystem backup tools. Include manifests, completions, intents, HQ journal, epochs and lock inode paths; protect access keys. Restore to a new path only after requalifying readiness and reconciling old allocation identities. No automated compaction or backup/restore CLI is provided yet. Do not copy a readiness receipt and edit its path.

Old notebook command generation can stay unchanged:

| Existing knob | Replacement |
| --- | --- |
| `commands_file_path` | `api.plan` / `api.submit`; immutable import snapshot |
| `cmds_per_job` | Unneeded pooled; explicit `commands_per_job` direct |
| `qtime` | Profile allocation time, distinct from task runtime/timeout |
| Queue/GPU class/constraint/exclude | Allowed concrete profiles and workload aliases |
| CPU/RAM/GPU/ranks | Explicit task resources and worker allocation capacity |
| `max_restarts` | Retry reasons and total attempt bound |
| `extra_sbatch` | Validated structured extras; conflicting ownership/resource flags rejected |
| `force_redo` | New explicit import generation and safe output policy |

The original aKG scheduler is untouched. Migrating it is separate work: start with a small new commands batch, inspect its manifest and output validator, then increase explicit budgets. Old completion markers are not trusted or imported automatically.
