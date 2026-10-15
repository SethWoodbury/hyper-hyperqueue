# Live dashboard

The dashboard reads the selected HHQ state directory. It is a viewer, not a second
scheduler: closing the browser, losing an SSH tunnel, or stopping its process does
not cancel tasks or stop the separate pooled service. Direct Slurm jobs remain
independent of that service too.

Use the same configuration as the campaign. Preparing an RFD4 sweep alone does not
import it, so it will appear after explicit launch. Native RFD4 jobs submitted
outside HHQ do not become HHQ campaigns automatically. Submission starts the
requested scheduler service, not this separate HTML viewer. Start `hhq ui`
explicitly; the viewer can run on a different host that can read the shared state.

`hhq --config /shared/hhq/config.yaml access` prints a concise cached controller
location and the exact viewer/status commands; add `--json` for a machine-readable
record. Generic submission and RFD4 launch also print these hints to stderr while
preserving their JSON stdout. `service status` reads the saved controller
deployment, leader and HQ endpoint. Immediately after Slurm submission, the
controller job ID can be known while its node and endpoint are still unavailable.
Those fields become available once the controller actually starts; an HQ endpoint
is not an HTML dashboard URL.

An inherited `HHQ_STATE_DIR` overrides the configuration's state directory, as does
the global `--state-dir` option. Check that override when a viewer appears empty
or shows another workspace; select the intended state explicitly or unset the
environment variable before starting the viewer.

## Start the viewer

Install HHQ before starting the server. Use the installed executable from your
existing environment; workers and viewer allocations should not install packages.

```bash
/absolute/path/to/venv/bin/hhq --config /shared/hhq/config.yaml ui --port 8765
```

The default is read-only. `--allow-submit` explicitly enables the generic command
preview/submit form; the Slurm template keeps `--read-only`. Submission permissions
and resource choices are never inferred from merely opening the viewer.

The viewer prints its actual hostname, selected state directory, access mode,
and an SSH-forwarding example, followed by access instructions. It prints one private URL such as
`http://127.0.0.1:8765/?token=...`. Open exactly that URL. Every page, asset,
API route and form post requires this random per-launch secret, including in
read-only mode; without it the server answers `401` and reveals no records.
Opening the URL stores the secret in an `HttpOnly`, `SameSite=Strict` session
cookie and redirects to the same page without the token, so later address bars,
bookmarks, and `Referer` headers (always `no-referrer`) do not carry it.
Restarting the viewer issues a new secret; reopen the newly printed URL. Cookie
names include a per-launch identifier, so separate viewers using the same remote
port can coexist through different local SSH forwards without logging each other
out.
The submission form keeps its own separate CSRF token.

The server does not log request lines and HHQ does not write the secret to disk,
but it is printed to the viewer's console. Under `examples/dashboard.sbatch`
that means the job log, created with `umask 077`; keep that log directory private
and treat the URL like a password. Scripted local clients may instead send
`Authorization: Bearer <secret>`.

For a server on a login host, run this from your own computer and then open the
printed URL:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8765:127.0.0.1:8765 USER@LOGIN_HOST
```

Keep the viewer and its configuration private. Loopback binding and SSH forwarding
protect transport; the session secret keeps other users/processes that share the
host's loopback interface from reading records, provided they cannot read your
console or job log. Detailed task pages can expose command arguments, paths, and
logs. Do not expose the port publicly or include it in a public reverse proxy.

## What the screen means

The page separates task outcomes from allocation status. A Slurm allocation that
started or completed is not proof that every command succeeded.

The task-flow diagram groups saved counts as **Ready / retry → Admitted →
Started → Succeeded**, with a separate **Needs attention** branch.
Ready includes `retry_wait`; admitted means `submitted` to Slurm or HQ. Started
is recorded running plus unfinished attempts. Attention includes failed, blocked,
interrupted, and unknown records other than unfinished attempts. Canceled tasks and
any other states are stated separately below the diagram; partial coverage is
also labeled. These nodes are current recorded buckets, not measured transitions,
rates, or a prediction of which task will run next.

Subtle motion illustrates the lifecycle only when fresh admitted/running records
exist. It does not measure GPU activity or prove a process is alive. Motion stops
for stale or undated snapshots, known-stale controller/service heartbeats, failed
connections, hidden tabs, saved exports, and reduced-motion preferences. The
keyboard-accessible **Pause motion** control
stops the decoration without stopping data refresh or any jobs. Ready-only and
completed-only workspaces remain static. On narrow screens, the diagram scrolls
inside its own focusable region; the page still fits the viewport.

| State | Meaning |
| --- | --- |
| Ready | Imported but not yet admitted to a backend; it is not pending in Slurm. |
| Submitted | Admitted to Slurm or HQ; it may still be waiting for resources. |
| Running | Last coordinator-reported state, where available; the viewer does not independently prove the process is alive. |
| Succeeded | The task completed and its configured output checks passed. |
| Failed or canceled | Inspect attempts and reasons; these are not automatically successful retries. |
| Unfinished | The latest attempt has no result yet. Shown neutrally under **Started**, not as running or failed and not under attention. The stored state stays `unknown` with reason `unfinished_attempt_liveness_not_probed`; the viewer does not probe locks/processes and attempt age is not proof of ongoing execution. |
| Unknown | Any other uncertain outcome (unreadable or unvalidated records, truncated scans); shown under **Needs attention**. |

Watch the displayed snapshot age as well as the counts. Browser refresh reads
durable HHQ records; it does not call Slurm repeatedly or infer when nodes will
become free. Summary reads are cached for at least ten seconds; slower scans can
extend that interval to reduce shared-storage load. The pooled controller updates
its own snapshot on its configured cadence. For a direct campaign whose scheduler
state needs updating, an explicit
`hhq status --reconcile` is separate from viewing the dashboard. Large state trees
and shared-storage latency can affect refresh speed.

The first response waits for its scan. A bounded 1,000-task metadata-only test
took about 4.46 seconds on this installation; the adaptive cache then waited at
least 22.3 seconds between scans. Very large workspaces may take substantially
longer. The overview caps scans at 50,000 tasks and 100,000 attempts by default
and explicitly labels partial results rather than treating unscanned tasks as
successes. See [validation](validation.md) for the measured scope.

If a scan of the records fails (for example, shared storage is briefly
unavailable), the overview keeps showing the previous snapshot with a visible
notice and waits a few seconds before scanning again rather than rescanning on
every request. Unreadable or malformed individual records, including a malformed
controller plan, are counted under Snapshot details instead of failing the page.

Use campaign, task, allocation, event, and measurement details to investigate a
problem. Detail pages read bounded records directly: the latest 50 attempts,
at most 1,000 attempt or submission files scanned, 30 allocation records per
page, and the last 128 KiB of a log. Logs, scripts and events that are not regular
files (for example a FIFO) are refused rather than read. Resource summaries are
advisory measurements, not automatically changed Slurm requests. Missing
measurements mean unknown, not zero.

GPU hardware is labeled from the devices recorded for each attempt; multi-GPU
attempts list every model (for example `2× A4000, L40`), and any device without a
recorded model makes the label unknown. **Requested** host RAM is the attempt's
effective request: the task's declared value, or a recorded retry RAM override.
Chart tooltips name the declared value when an override applied. RFD4 scientific
quality, candidate classifications, and AF3 ligand correctness remain in the
campaign's own metrics and review tools.

For an offline point-in-time view, export a self-contained HTML file:

```bash
hhq --config /shared/hhq/config.yaml ui --export /shared/private/hhq-report.html
```

Export starts no web server or Slurm job and contains no session secret. It is
created with mode `0600` and never overwrites an existing file. The file does not
update live; regenerate it when needed. It contains private campaign metadata, so share it deliberately
and do not publish it automatically. Progress counts track tasks, not percentage
completion inside diffusion; absent GPU utilization or VRAM observations stay
unknown.

## Host it in a small CPU allocation

[dashboard.sbatch](../examples/dashboard.sbatch) runs just the viewer: one CPU,
512 MiB host RAM, no GPU, and a 12-hour limit. The memory value is a small **initial
declared request**, not a measured guarantee for every campaign size. The site's
partition/account are intentionally unset. Choose an allowed CPU partition and
account explicitly; the viewer's queue choice is independent of the HHQ worker
profiles in the selected configuration.

Create a private shared log directory first. Substitute real paths and site values:

```bash
mkdir -p /shared/private/hhq-dashboard
chmod 700 /shared/private/hhq-dashboard
sbatch --partition=CPU_PARTITION --account=ACCOUNT \
  --chdir=/shared/private/hhq-dashboard \
  --output=/shared/private/hhq-dashboard/dashboard-%j.log \
  /path/to/hyper-hyperqueue/examples/dashboard.sbatch \
  /absolute/path/to/venv/bin/hhq /shared/hhq/config.yaml 8765
```

Omit `--account` if your site does not use it. Standard `sbatch` flags can explicitly
override the template's memory or walltime. The executable, configuration, and
state directory must be reachable from the chosen compute node. Keep authoritative
HHQ records on durable shared storage according to your site's retention policy;
the viewer's disposable logs can live in scratch.

After the job starts, its log prints the compute hostname. If your site permits
SSH to nodes allocated to you, tunnel **to that node through the login host**:

```bash
ssh -N -o ExitOnForwardFailure=yes -J USER@LOGIN_HOST \
  -L 127.0.0.1:8765:127.0.0.1:8765 USER@COMPUTE_HOST
```

The last host in that command must be the compute host: forwarding from the login
host directly to `COMPUTE_HOST:8765` cannot reach a server bound to the compute
node's loopback. If compute-node SSH or forwarding is prohibited, use the site's
approved interactive-app/tunnel mechanism or host the loopback viewer where SSH
is supported. Public binding is not a fallback. If local port 8765 is occupied,
change the first port in `-L` and open that local port instead.

The viewer is available for the allocation's lifetime. The template does not
requeue itself, submit replacement jobs, or qualify automatic long-term hosting.
At timeout or preemption, submit a fresh viewer allocation and reconnect to its
new node; campaign records remain in the selected state directory. Only cancel
the dashboard's own known job ID when shutting it down.

Inspect actual resource usage before changing the request:

```bash
sacct -j DASHBOARD_JOB_ID --format=JobID,State,Elapsed,ReqMem,MaxRSS
```

Check the batch/step rows for MaxRSS; the top-level allocation can have no RSS
value. Keep modest fixed headroom over representative measured peaks. Browsing
a tiny empty state directory does not qualify memory use for a large campaign.
The bounded hosting test with 28 synthetic task records observed 42.75 MiB peak;
512 MiB remains an initial request, not a guarantee for every workspace.

## Simple task inputs

For ordinary commands, keep one command per line and preview before submission:

```bash
hhq --config /shared/hhq/config.yaml plan /shared/commands.txt --profile cpu-direct --mode direct
hhq --config /shared/hhq/config.yaml submit /shared/commands.txt --profile cpu-direct --mode direct
```

Select a configured pooled CPU or GPU profile for pooled execution; see the
[quick start](../README.md) and [direct versus pooled guide](cpu-direct-vs-pooled.md).
Set explicit per-task resources when profile defaults do not describe the work.
The dashboard is optional: the notebook/API and command line continue working
without it.

For RFD4, keep the notebook's saved native sweep and use the
[RFD4 adapter](rfd4.md). It adds configuration/source checks, explicit method
resources, ownership, and per-row completion receipts. Passing arbitrary RFD4
shell commands as generic tasks does not provide those adapter guarantees.
The adapter's real model/AF3 GPU execution and cross-config model reuse still
need separate qualification. Native preflights have encountered environment
requirements, including `libcuda.so.1` during container imports on a login host.
Dashboard availability does not change that readiness boundary. Exact tested
and blocked paths are recorded in [validation](validation.md).
