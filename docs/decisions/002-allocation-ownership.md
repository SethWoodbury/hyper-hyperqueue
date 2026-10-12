# ADR 002: hhq owns allocations; HQ dispatches commands

Accepted 2026-10-12. One hhq controller owns Slurm worker submission intents. One HQ server dispatches compatible individual commands across all workers. HQ's automatic allocator is not enabled, avoiding competing resource controllers.

A durable random token is recorded before each external submission. Slurm receives it in the comment and complete job-name suffix. Live and historical ownership checks require that exact token and the user; the name fallback is necessary because this site's accounting does not retain comments. Unknown acknowledgements occupy capacity until reconciled. Cancellation only touches positively identified allocations.

CPU workers expose assigned CPUs and host RAM after overhead. GPU workers expose exactly one assigned GPU, its measured model/VRAM, and one logical GPU slot. Automatic node-wide resource detection is disabled. CPU and GPU eligibility are separate resources so CPU-only tasks cannot consume GPU allocations accidentally. The current GPU baseline intentionally runs one task at a time. CPU workers may run multiple tasks when CPU and memory resources fit.

Workers read actual allocation end time from Slurm, not queue age, and cap useful lifetime by the current control server allocation's end time. A cleanup reserve is subtracted. Tasks declare a minimum remaining duration to HQ, which prevents starting tasks that are known not to fit. Incorrect or uncertain runtime hints remain visible limitations.
