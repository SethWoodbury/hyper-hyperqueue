# hyper-hyperqueue

Keep this a thin configurable layer around the pinned HyperQueue runtime.
Direct Slurm arrays must work without importing or starting HyperQueue.
Never migrate existing scientific campaigns or submit application workloads as tests.
Use bounded synthetic jobs and cancel only positively identified test resources.
Preserve uncertainty: accepted-but-unacknowledged submission is not a retry signal.
Never remove leadership locks or use heartbeat age alone to authorize takeover.
Keep raw cluster discovery, command contents, credentials and runtime state out of Git.
Do not use SQLite WAL on shared network filesystems. Runtime state lives outside this checkout.
Document real integration evidence separately from unit tests and untested features.
Use `uv sync --all-extras` and `uv run pytest`; workers must not install at startup.
Do not publish a remote or push without explicit user authorization.

Read README.md, docs/architecture.md, docs/recovery.md and docs/limitations.md
before changing ownership, replay, resource or output semantics. Keep direct CPU
independent of HQ and preserve immutable task/configuration snapshots. Slurm
controls allocations; HQ dispatches tasks. Do not add a second allocation owner.

Use scratch for test environments and bulk output. Keep authoritative manifests,
journals and completion records outside this checkout on durable shared storage.
Real validation evidence is indexed in docs/validation.md; failed exploratory
probes are evidence and should not be silently overwritten. Production readiness
is not inherited from the separate validation directory.

Run `uv run ruff check src tests examples`, relevant pytest tests, then the full
suite for ownership/recovery changes. HQ integration tests need the separately
installed pinned executable. Record skipped/blocked probes explicitly. Never
run scientific workloads or migrate old campaigns as an integration test.
