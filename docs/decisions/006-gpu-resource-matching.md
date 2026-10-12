# ADR 006: match actual assigned GPU resources

Accepted 2026-10-12. A Slurm GRES type is a placement class, not a guarantee of model, VRAM or application compatibility. `--constraint` remains separate. GPU workers inspect only assigned device identifiers and advertise actual resources to HQ. The shared runner also checks declared GPU model/VRAM requirements before executing, including direct GPU tasks.

The initial supported pooled shape is one node, one rank, one GPU, one simultaneous GPU task. Host RAM is independent of total GPU VRAM; neither a logical HQ counter nor a fractional GPU request isolates device memory. CPU tasks use CPU pools by default and never require a fake GPU.

Serial execution is intentionally retained until a workload/model-specific concurrent policy has real aggregate RAM/VRAM, visibility and throughput tests. GPU models, concurrency and cache/software context must remain separate measurement groups. No AF3/JAX memory policy is invented by this wrapper.

Shared GPU subclasses consume the same configured budget. Faster assigned GPUs naturally execute more tasks during a long allocation. The planner does not need to predict a node's availability or pre-partition commands into fixed per-GPU batches.
