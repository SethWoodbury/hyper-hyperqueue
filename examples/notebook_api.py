"""Import these calls into a notebook; execution below only previews a plan."""

from pathlib import Path

from hyper_hyperqueue import api
from hyper_hyperqueue.config import load_config
from hyper_hyperqueue.models import OutputPolicy, Resources, RetryPolicy


def preview(commands: Path, config_path: Path):
    config = load_config(config_path)
    return api.plan(
        commands,
        config,
        profile="cpu-pool",
        mode="pooled",
        resources=Resources(cpus=1, memory_mib=1024, runtime_seconds=10),
    )


def launch_repeatable(commands: Path, config_path: Path):
    config = load_config(config_path)
    return api.submit(
        commands,
        config,
        profile="cpu-pool",
        mode="pooled",
        resources=Resources(cpus=1, memory_mib=1024, runtime_seconds=10),
        retry=RetryPolicy(max_attempts=2, reasons=["worker_lost", "preempted"]),
        output=OutputPolicy(mode="in_place", repeatable=True),
        name="notebook_example",
        workload="synthetic_sleep",
        software_fingerprint="example-v1",
    )


if __name__ == "__main__":
    here = Path(__file__).parent
    print(preview(here / "cpu_commands.txt", here / "cluster.yaml").model_dump_json(indent=2))
