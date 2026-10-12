# ADR 005: make both CPU paths first-class

Accepted 2026-10-12. Long independent CPU commands default to direct arrays; short tasks can pack into HQ workers. Unknown CPU runtime defaults to direct and is explained in the plan. Explicit mode wins, and profile preferences provide a convenient default. The initial short/long threshold is configurable at 300 seconds.

Slurm arrays avoid a control-service dependency and express many identical resource requests with one submission RPC. Pooled CPU avoids repeated allocation startup and can share real allocated CPU/RAM across commands. Neither is inherently best for every workload. Synthetic tests validate correctness; they do not justify universal throughput claims.

Direct uses one command per element by default. All split-array throttles together reserve the requested campaign maximum; when that cannot admit every chunk immediately, the remainder is explicit. Pooled uses individual tasks with no fixed command batches. Both execute through the same attempt/validation runner.
