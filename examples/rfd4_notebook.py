"""Standalone HHQ controls to adapt into the scientific notebook's submission section.

Opening this notebook does not prepare files, import tasks, or submit jobs.
Install marimo in the notebook environment separately; it is not an HHQ dependency.
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import yaml

    from hyper_hyperqueue.config import load_config, save_config
    from hyper_hyperqueue.integrations.rfd4.prepare import plan_sweep, prepare_sweep
    from hyper_hyperqueue.integrations.rfd4.submission import (
        collection_ready,
        launch,
        method_profiles,
        preview,
    )

    return (
        Path,
        collection_ready,
        launch,
        load_config,
        method_profiles,
        mo,
        plan_sweep,
        prepare_sweep,
        preview,
        save_config,
        yaml,
    )


@app.cell
def _(mo):
    mo.md("""
    # RFD4 execution with HHQ

    Keep the scientific notebook's input selection, filters, seeds, configurations,
    and **write sweep** cell. Point these controls at its saved `sweep_manifest.json`.
    HHQ executes those rows without changing their scientific settings.

    ## 1. Choose files and execution resources

    Copy `examples/rfd4_options.yaml` and fill in the runtime, task RAM and duration
    for each method actually used. Remove unused methods. RAM is **host memory**;
    add a fixed headroom to the measured peak. An initial estimate is `declared`,
    not `measured`. Runtime describes the whole bundle. Start with one row per task.
    Use a durable shared preparation directory outside the source repository.
    """)


@app.cell
def _(mo):
    native_manifest = mo.ui.text(label="Saved native sweep manifest", full_width=True)
    cluster_path = mo.ui.text(label="HHQ cluster configuration", full_width=True)
    options_path = mo.ui.text(label="Completed RFD4 resource/options YAML", full_width=True)
    prepared_dir = mo.ui.text(label="New shared preparation directory", full_width=True)
    placement_names = mo.ui.text(
        value="gpu-small,gpu-large,gpu-backfill",
        label="Eligible base profiles (comma separated; add gpu-train if intended)",
        full_width=True,
    )
    plan_button = mo.ui.run_button(label="Check files and plan — no submission")
    mo.vstack([native_manifest, cluster_path, options_path, prepared_dir, placement_names, plan_button])
    return cluster_path, native_manifest, options_path, placement_names, plan_button, prepared_dir


@app.cell
def _(
    Path,
    cluster_path,
    load_config,
    method_profiles,
    mo,
    native_manifest,
    options_path,
    placement_names,
    plan_button,
    plan_sweep,
    yaml,
):
    mo.stop(not plan_button.value, mo.md("Fill the paths and resource options, then check the plan."))
    _options = yaml.safe_load(Path(options_path.value).expanduser().read_text())
    _base = load_config(Path(cluster_path.value).expanduser())
    _requests = {name: spec["resources"] for name, spec in _options["workloads"].items()}
    execution_config = method_profiles(
        _base, _requests, [name.strip() for name in placement_names.value.split(",") if name.strip()]
    )
    scientific_plan = plan_sweep(Path(native_manifest.value).expanduser(), **_options)
    mo.vstack(
        [
            mo.md("## 2. Review and prepare\nThis has not imported tasks or started a service."),
            mo.json(scientific_plan["summary"]),
            mo.json(scientific_plan["warnings"]),
            mo.json(
                {
                    name: profile.model_dump()
                    for name, profile in execution_config.profiles.items()
                    if name.startswith("rfd4-")
                }
            ),
        ]
    )
    return execution_config, scientific_plan


@app.cell
def _(mo, scientific_plan):
    prepare_button = mo.ui.run_button(
        label=f"Prepare {scientific_plan['summary']['tasks']} tasks — no submission"
    )
    mo.vstack([prepare_button])
    return (prepare_button,)


@app.cell
def _(
    Path,
    execution_config,
    load_config,
    mo,
    prepare_button,
    prepare_sweep,
    prepared_dir,
    preview,
    save_config,
    scientific_plan,
):
    mo.stop(not prepare_button.value, mo.md("Review the plan, then prepare its immutable execution files."))
    _directory = Path(prepared_dir.value).expanduser()
    _configuration = _directory / "hhq-config.json"
    if _configuration.exists() and load_config(_configuration) != execution_config:
        raise ValueError("Existing preparation has a different HHQ config; choose a new directory")
    prepared = prepare_sweep(scientific_plan, _directory)
    if not _configuration.exists():
        save_config(execution_config, _configuration)
    resource_preview = preview(prepared, execution_config)
    mo.vstack(
        [
            mo.json(resource_preview),
            mo.md(f"Saved manifest: `{prepared['manifest_path']}`\n\nHHQ config: `{_configuration}`"),
        ]
    )
    return (prepared,)


@app.cell
def _(mo, prepared):
    launch_button = mo.ui.run_button(label=f"Launch prepared sweep {prepared['plan_id']}")
    mo.vstack(
        [
            mo.md("""
        ## 3. Launch explicitly

        Launch imports runnable work. An existing HHQ service can begin immediately.
        It starts one shared service if needed; it does not invoke RFD4's Slurm launcher.
        Repeating an identical launch uses the same task identities.
        """),
            launch_button,
        ]
    )
    return (launch_button,)


@app.cell
def _(execution_config, launch, launch_button, mo, prepared):
    mo.stop(not launch_button.value, mo.md("Prepared; waiting for an explicit launch."))
    mo.json(launch(prepared, execution_config))


@app.cell
def _(mo, prepared):
    refresh_button = mo.ui.run_button(label=f"Refresh task status for {prepared['plan_id']}")
    mo.vstack(
        [
            mo.md("""
        ## 4. Inspect completion and collect

        Check logical tasks, not whether worker jobs have ended. `ready` means all
        tasks are terminal; inspect failures too. Native early stopping can produce
        no retained designs and still complete correctly. Scientific pass rates
        belong to the native results, not the HHQ completion count.

        Run the existing collector on CPU only after this barrier. Keep AF3 inputs,
        candidate filters and visualizations in the scientific notebook.
        """),
            refresh_button,
        ]
    )
    return (refresh_button,)


@app.cell
def _(collection_ready, execution_config, mo, prepared, refresh_button):
    mo.stop(not refresh_button.value, mo.md("Refresh to inspect this sweep only."))
    mo.json(collection_ready(prepared, execution_config, reconcile=True))


if __name__ == "__main__":
    app.run()
