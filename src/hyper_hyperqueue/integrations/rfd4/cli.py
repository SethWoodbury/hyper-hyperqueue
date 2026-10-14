"""Optional RFD4 commands; importing this module does not import model libraries."""

from pathlib import Path

import typer
import yaml

app = typer.Typer(no_args_is_help=True, help="Prepare and execute saved native RFD4 sweeps.")


def _options(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or set(value) - {"runtime", "workloads", "reuse_model"}:
        raise ValueError("Options must contain runtime, workloads, and optional reuse_model")
    if not {"runtime", "workloads"} <= value.keys():
        raise ValueError("Options require explicit runtime and workload resource settings")
    return value


@app.command("plan")
def plan_command(manifest: Path, options: Path):
    """Verify a native sweep and preview execution groups; no writes or submission."""
    from ...cli import output
    from .prepare import plan_sweep

    plan = plan_sweep(manifest, **_options(options))
    output(
        {
            "plan_id": plan["plan_id"],
            "summary": plan.get("summary"),
            "warnings": plan.get("warnings", []),
            "bundles": [
                {
                    "bundle_id": bundle["bundle_id"],
                    "runs": len(bundle["rows"]),
                    "workload": bundle["workload"],
                }
                for bundle in plan["bundles"]
            ],
        }
    )


@app.command("prepare")
def prepare_command(manifest: Path, options: Path, directory: Path):
    """Write immutable adapter files outside active HHQ state; does not launch work."""
    from ...cli import output
    from .prepare import plan_sweep, prepare_sweep

    output(prepare_sweep(plan_sweep(manifest, **_options(options)), directory))


@app.command("preview")
def preview_command(prepared: Path):
    """Check resource placement for a prepared sweep without importing tasks."""
    from ...cli import cfg, output
    from .submission import preview

    output(preview(prepared, cfg()))


@app.command("launch")
def launch_command(prepared: Path):
    """Explicitly admit prepared scientific work and start its pooled service."""
    from ...cli import cfg, output
    from .submission import launch

    output(launch(prepared, cfg()))


@app.command("status")
def status_command(prepared: Path, reconcile: bool = False):
    """Inspect only the child campaigns belonging to this sweep."""
    from ...cli import cfg, output
    from .submission import status

    output(status(prepared, cfg(), reconcile=reconcile))


@app.command("collection-ready")
def collection_ready_command(prepared: Path, reconcile: bool = False):
    """Report task completion and failures before downstream CPU collection."""
    from ...cli import cfg, output
    from .submission import collection_ready

    output(collection_ready(prepared, cfg(), reconcile=reconcile))
