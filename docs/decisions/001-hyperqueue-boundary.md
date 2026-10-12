# ADR 001: pin HyperQueue and keep the boundary small

Accepted 2026-10-12. HyperQueue binary and optional Python package are both pinned to **0.26.2**, the available upstream release checked during implementation. The executable remains a separately installed MIT-licensed dependency; this repository does not vendor or relicense it.

Task submission uses upstream `hyperqueue.Client`, `Job.program`, and `ResourceRequest`. A small subprocess bounds submission latency; a timeout or missing acknowledgement creates an unknown intent requiring reconciliation by the unique job name. It never authorizes an automatic second submission. Inspection and cancellation use the version-matched JSON CLI, with explicit ownership checks. Importing this adapter does not import HyperQueue, so direct Slurm execution works without the optional package or executable.

Verified 0.26.2 schemas include `task list ID` returning `{str(ID): [tasks]}`, a singular `worker` integer on task records, and `worker list --all` returning `ended=null` for a connected worker. Resource names use slash-separated labels because this release rejects underscores in resource names. The adapter checks both versions before submission.

Install the Python integration once with `uv sync --extra hq`. Install the shared executable once before submitting workers; workers never download or build it. The tested Linux x86-64 artifact is:

- [Upstream release](https://github.com/It4innovations/hyperqueue/releases/tag/v0.26.2)
- Archive `hq-v0.26.2-linux-x64.tar.gz`, SHA256 `e15dae9113e1a307a97a66bfe90f74f78c6016239436b5d9f1e4efec480e84b5`
- Extracted `hq` SHA256 `c5f7c152e10ed61e12338053d0b637fbdc1b2c1e09844a5e9e8ea8f4f9c74c6b`

Set `control.hq_binary`, `HHQ_HQ_BINARY`, or install at `$HOME/.local/share/hyper-hyperqueue/hq/0.26.2/hq`. Other platforms need their own verified upstream artifact; these hashes apply only to Linux x86-64.

The persistent HQ journal restores tasks, not live workers. This upstream behavior is part of the recovery design, rather than a promise of uninterrupted execution. See [server documentation](https://it4innovations.github.io/hyperqueue/stable/deployment/server/) and [Python submission](https://it4innovations.github.io/hyperqueue/stable/python/submit/).
