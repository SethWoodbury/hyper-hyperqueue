"""Model-free adapter tests: preserve native calls and fail closed on drift."""

import hashlib
import json
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from hyper_hyperqueue.integrations.rfd4 import prepare


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def sweep(tmp_path, monkeypatch):
    repo = tmp_path / "native"
    (repo / "src/rfproteina").mkdir(parents=True)
    (repo / "configs").mkdir()
    (repo / "src/rfproteina/generate.py").write_text("raise AssertionError('never execute native code')\n")
    (repo / "configs/inference.yaml").write_text("native: true\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    (repo / ".env.digs").write_text("EXAMPLE=fixture\n")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    ligand = tmp_path / "input with space.cif"
    ligand.write_text("data_dummy\n")
    rows = []
    for index, method in enumerate(("generate", "generate", "generate", "beam", "gradient")):
        run_id = f"native_{index}"
        config = config_dir / f"{run_id}.yaml"
        output = tmp_path / "science" / run_id
        config.write_text(
            yaml.safe_dump(
                {
                    "inputs": [str(ligand)],
                    "seed": 42 + index,
                    "num_replicates": 3,
                    "out_dir": str(output),
                    "run_name_": run_id,
                    "gen_njobs": 1,
                    "job_id": 0,
                    "search": {"width": 4} if method == "beam" else None,
                    "guidance": {"strength": 0.1} if method == "gradient" else None,
                }
            )
        )
        command = shlex.join(
            [
                "env",
                "-u",
                "EXEC_SKIP_AF3_DAEMON",
                "CLUSTER=digs",
                "PYTHONPATH=/custom code",
                "OMP_NUM_THREADS=1",
                str(repo / "src/rfproteina/generate.py"),
                "--config-path",
                str(config_dir),
                "--config-name",
                run_id,
            ]
        )
        rows.append(
            {
                "input": str(ligand),
                "method": method,
                "variant": "test",
                "thinking": "medium",
                "bb_noise": 0.3,
                "latent_noise": 0.1,
                "seed": 42 + index,
                "requested_seed": 42 + index,
                "replicates": 3,
                "steps": 200,
                "trunk_passes": 4,
                "run_id": run_id,
                "config": str(config),
                "output_dir": str(output),
                "command": command,
            }
        )
    commands_file = tmp_path / "native_commands.txt"
    commands_file.write_text("\n".join(row["command"] for row in rows) + "\n")
    manifest = {
        "schema_version": 1,
        "status": "prepared_not_submitted",
        "name": "test",
        "repo": str(repo),
        "repo_commit": revision,
        "output_dir": str(tmp_path / "science"),
        "commands_file": str(commands_file),
        "rows": rows,
        "warnings": [],
        "counts": {"commands": 5},
        "input_sha256": {str(ligand): sha(ligand)},
        "file_sha256": {
            **{row["config"]: sha(row["config"]) for row in rows},
            str(commands_file): sha(commands_file),
        },
        "native_config_sha256": {str(repo / "configs/inference.yaml"): sha(repo / "configs/inference.yaml")},
    }
    path = tmp_path / "native_manifest.json"
    path.write_text(json.dumps(manifest))
    runtime_script = tmp_path / "runtime.py"
    runtime_script.write_text("raise AssertionError('not executed in preparation')\n")
    monkeypatch.setattr(prepare, "RUNTIME_SCRIPT", runtime_script)
    resources = {
        "cpus": 2,
        "memory_mib": 8192,
        "gpus": 1,
        "runtime_seconds": 1200,
        "runtime_confidence": "declared",
        "nodes": 1,
        "ranks": 1,
    }
    workloads = {
        method: {"profile": "test_gpu", "resources": resources, "runs_per_task": 2}
        for method in ("generate", "beam", "gradient")
    }
    return {
        "path": path,
        "manifest": manifest,
        "runtime": {"argv": [sys.executable]},
        "workloads": workloads,
        "root": tmp_path,
        "repo": repo,
    }


def plan(sweep, **kwargs):
    return prepare.plan_sweep(sweep["path"], runtime=sweep["runtime"], workloads=sweep["workloads"], **kwargs)


def rewrite(sweep):
    sweep["path"].write_text(json.dumps(sweep["manifest"]))


def refresh_config(sweep, row):
    sweep["manifest"]["file_sha256"][row["config"]] = sha(row["config"])
    rewrite(sweep)


def refresh_commands(sweep):
    source = sweep["manifest"]
    Path(source["commands_file"]).write_text("\n".join(row["command"] for row in source["rows"]) + "\n")
    source["file_sha256"][source["commands_file"]] = sha(source["commands_file"])
    rewrite(sweep)


def test_explicit_beam_continuation_requires_its_external_state_pin(sweep):
    row = next(row for row in sweep["manifest"]["rows"] if row["method"] == "beam")
    state = sweep["root"] / "saved-beam.pt"
    state.write_bytes(b"fixture: deliberate continuation state")
    config = Path(row["config"])
    document = yaml.safe_load(config.read_text())
    document["search"]["resume_from"] = str(state)
    config.write_text(yaml.safe_dump(document))
    refresh_config(sweep, row)
    with pytest.raises(ValueError, match="state must be pinned"):
        plan(sweep)
    sweep["runtime"]["source_hashes"] = {str(state): sha(state)}
    result = plan(sweep)
    assert all(str(state) in bundle["source_hashes"] for bundle in result["bundles"])
    state.write_bytes(b"changed state")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        plan(sweep)


@pytest.mark.parametrize("fallback", [False, True])
def test_safe_yaml_parser_preserves_values_aliases_and_recipe_identity(monkeypatch, fallback):
    if fallback:
        monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    text = """defaults: [inference, _self_]
inputs: [/one.cif]
out_dir: /scratch/one
seed: 173
run_name_: one
metrics: &metrics
  threshold: 1.25
  accept: true
  missing: null
  input: '${inputs.0}'
guidance: {metrics: *metrics}
"""
    expected = yaml.safe_load(text)
    actual = prepare._safe_yaml_load(text)
    assert actual == expected
    assert actual["metrics"] is actual["guidance"]["metrics"]
    assert prepare.recipe_identity(actual, "gradient", {}, []) == prepare.recipe_identity(expected, "gradient", {}, [])
    with pytest.raises(yaml.constructor.ConstructorError):
        prepare._safe_yaml_load("!!python/object/apply:os.system ['false']")


def test_plan_is_read_only_preserves_native_rows_and_operational_bundles(sweep):
    before = sorted(str(path.relative_to(sweep["root"])) for path in sweep["root"].rglob("*"))
    result = plan(sweep)
    after = sorted(str(path.relative_to(sweep["root"])) for path in sweep["root"].rglob("*"))
    assert before == after
    assert result == plan(sweep)
    assert result["summary"]["scientific_calls"] == 5
    assert result["summary"]["tasks"] == 4
    assert [len(b["rows"]) for b in result["bundles"]] == [2, 1, 1, 1]
    restored = [row for bundle in result["bundles"] for row in bundle["rows"]]
    for actual, original in zip(restored, sweep["manifest"]["rows"]):
        assert {key: actual[key] for key in original} == original
        assert actual["job_id"] == 0
        assert actual["config_sha256"] == sha(actual["config"])
        assert actual["environment"]["PYTHONPATH"] == "/custom code"
        assert actual["unset_environment"] == ["EXEC_SKIP_AF3_DAEMON"]
    assert all(bundle["workload"]["resources"]["runtime_seconds"] == 1200 for bundle in result["bundles"])
    assert not (sweep["root"] / "science").exists()


def test_prepare_is_idempotent_verified_and_refuses_conflicts(sweep):
    planned = plan(sweep)
    output = sweep["root"] / "prepared"
    result = prepare.prepare_sweep(planned, output)
    assert result == prepare.prepare_sweep(planned, output)
    assert prepare.verify_prepared(result) == result
    assert prepare.verify_prepared(result["manifest_path"]) == result
    group = result["groups"][0]
    commands = [json.loads(line) for line in Path(group["commands_file"]).read_text().splitlines()]
    bundle = result["bundle_files"][group["bundle_ids"][0]]
    assert commands[0]["argv"][-4:] == ["--bundle", bundle["path"], "--fingerprint", bundle["sha256"]]
    Path(group["commands_file"]).write_text("corruption\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.verify_prepared(result)
    with pytest.raises(FileExistsError, match="Preserving"):
        prepare.prepare_sweep(planned, output)


def test_prepare_creates_exact_empty_runtime_sidecars_but_not_scientific_runs(sweep, monkeypatch):
    from hyper_hyperqueue.integrations.rfd4.runtime import identity

    planned = plan(sweep)
    calls = []
    original = prepare.ensure_directory

    def create(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(prepare, "ensure_directory", create)
    prepared = prepare.prepare_sweep(planned, sweep["root"] / "prepared")
    for row in sweep["manifest"]["rows"]:
        output = Path(row["output_dir"]).resolve()
        parent = output.parent / ".hhq-rfd4"
        sidecar = parent / identity({"output_dir": str(output)})
        assert sidecar.is_dir()
        assert list(sidecar.iterdir()) == []  # No owner, lock or success claim.
        assert not output.exists()
        assert calls.index(output.parent) < calls.index(parent) < calls.index(sidecar)
    assert len(calls) == len(set(calls)) == 7
    assert not any(path.is_file() for path in (sweep["root"] / "science").rglob("*"))
    assert prepare.verify_prepared(prepared) == prepared


def test_reprepare_restores_missing_sidecar_and_preserves_existing_scientific_evidence(sweep):
    from hyper_hyperqueue.integrations.rfd4.runtime import identity

    planned = plan(sweep)
    destination = sweep["root"] / "prepared"
    first = prepare.prepare_sweep(planned, destination)
    row = sweep["manifest"]["rows"][0]
    output = Path(row["output_dir"])
    sidecars = output.parent / ".hhq-rfd4"
    sidecar = sidecars / identity({"output_dir": str(output)})
    owner = sidecar / "owner.json"
    owner.write_text('{"fixture": "existing runtime evidence"}\n')
    output.mkdir()
    model = output / "model.cif"
    model.write_text("data_existing\n")
    other = next(path for path in sidecars.iterdir() if path != sidecar)
    other.rmdir()

    assert prepare.prepare_sweep(planned, destination) == first
    assert other.is_dir()
    assert owner.read_text() == '{"fixture": "existing runtime evidence"}\n'
    assert model.read_text() == "data_existing\n"


def test_sidecar_directory_conflict_prevents_package_publication(sweep):
    planned = plan(sweep)
    science = sweep["root"] / "science"
    science.mkdir()
    conflict = science / ".hhq-rfd4"
    conflict.write_text("preserve real file\n")
    destination = sweep["root"] / "prepared"
    with pytest.raises((FileExistsError, NotADirectoryError)):
        prepare.prepare_sweep(planned, destination)
    assert not (destination / "prepared.json").exists()
    assert not (destination / "bundles").exists()
    assert conflict.read_text() == "preserve real file\n"


def test_sidecar_precreation_uses_runtime_unicode_path_identity(tmp_path):
    from hyper_hyperqueue.integrations.rfd4.runtime import identity

    output = tmp_path / "protéina" / "design"
    prepare._prepare_output_sidecars([{"rows": [{"output_dir": str(output)}]}])
    expected = output.parent / ".hhq-rfd4" / identity({"output_dir": str(output.resolve())})
    assert expected.is_dir()
    assert not output.exists()


def test_native_source_and_input_drift_fail_before_any_writes(sweep):
    planned = plan(sweep)
    Path(sweep["manifest"]["rows"][0]["input"]).write_text("changed\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        plan(sweep)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.prepare_sweep(planned, sweep["root"] / "never_created")
    assert not (sweep["root"] / "never_created").exists()
    assert not (sweep["root"] / "science").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"gen_njobs": 2},
        {"job_id": 1},
        {"seed": 991},
        {"inputs": []},
        {"num_replicates": 2},
        {"search": {"bad": True}},
    ],
)
def test_hash_valid_but_semantically_changed_config_rejected(sweep, change):
    row = sweep["manifest"]["rows"][0]
    document = yaml.safe_load(Path(row["config"]).read_text())
    document.update(change)
    Path(row["config"]).write_text(yaml.safe_dump(document))
    refresh_config(sweep, row)
    with pytest.raises(ValueError):
        plan(sweep)


@pytest.mark.parametrize(
    "value",
    [
        "sbatch native.sh",
        "bash -c 'echo dangerous'",
        "env CUDA_VISIBLE_DEVICES=0 x",
        "env SLURM_JOB_ID=fake x",
        "env X=$(touch never) x",
    ],
)
def test_unsupported_shell_and_resource_overrides_rejected(sweep, value):
    sweep["manifest"]["rows"][0]["command"] = value
    refresh_commands(sweep)
    with pytest.raises(ValueError):
        plan(sweep)
    assert not (sweep["root"] / "never").exists()


def test_environment_differences_prevent_bundling_and_runtime_conflicts_rejected(sweep):
    row = sweep["manifest"]["rows"][1]
    row["command"] = row["command"].replace("OMP_NUM_THREADS=1", "OMP_NUM_THREADS=2")
    refresh_commands(sweep)
    bundles = [b for b in plan(sweep)["bundles"] if b["method"] == "generate"]
    assert [len(b["rows"]) for b in bundles] == [2, 1]
    assert [r["run_id"] for r in bundles[0]["rows"]] == ["native_0", "native_2"]
    sweep["runtime"]["environment"] = {"OMP_NUM_THREADS": "3"}
    with pytest.raises(ValueError, match="conflicts"):
        plan(sweep)


def test_opt_in_recipe_bundles_keep_same_science_and_seeds_together(sweep):
    rows = sweep["manifest"]["rows"]
    changed = rows[1]
    document = yaml.safe_load(Path(changed["config"]).read_text())
    document["precision"] = "32-true"
    Path(changed["config"]).write_text(yaml.safe_dump(document))
    refresh_config(sweep, changed)
    result = plan(sweep, group_by_recipe=True, execution_order=["gradient", "beam", "generate"])
    assert result["summary"]["recipes_by_method"] == {"generate": 2, "beam": 1, "gradient": 1}
    assert result["summary"]["scientific_calls"] == 5
    assert result["summary"]["execution_order"] == ["gradient", "beam", "generate"]
    groups = [bundle for bundle in result["bundles"] if bundle["method"] == "generate"]
    assert sorted(sorted(row["run_id"] for row in bundle["rows"]) for bundle in groups) == [
        ["native_0", "native_2"], ["native_1"]
    ]
    restored = {row["run_id"]: row for bundle in result["bundles"] for row in bundle["rows"]}
    for original in rows:
        assert {key: restored[original["run_id"]][key] for key in original} == original
    assert [bundle["workload"]["priority"] for bundle in result["bundles"]] == [3, 2, 1, 1]
    prepared = prepare.prepare_sweep(result, sweep["root"] / "ordered")
    assert [group["priority"] for group in prepared["groups"]] == [3, 2, 1]
    assert prepare.verify_prepared(prepared) == prepared


def test_legacy_grouping_has_no_new_identity_fields_and_explicit_priority_wins(sweep):
    legacy = plan(sweep)
    assert "execution" not in legacy
    assert "recipes" not in legacy["summary"]
    assert all("recipe_id" not in bundle and "priority" not in bundle["workload"] for bundle in legacy["bundles"])
    assert legacy == plan(sweep, group_by_recipe=False, execution_order=None)
    sweep["workloads"]["beam"]["priority"] = 100
    result = plan(sweep, execution_order=["gradient", "beam", "generate"])
    assert [bundle["method"] for bundle in result["bundles"]] == ["gradient", "beam", "generate", "generate"]
    assert result["bundles"][1]["workload"]["priority"] == 100


@pytest.mark.parametrize("order", [[], "gradient", ["beam"], ["beam", "beam"], ["gradient", "beam", "unknown"]])
def test_invalid_execution_order_refused(sweep, order):
    with pytest.raises(ValueError, match="execution_order"):
        plan(sweep, execution_order=order)


@pytest.mark.parametrize("field,value", [("search", {"width": 3}), ("metrics", {"different": True}),
                                         ("compile_lengths", [128, 256]), ("reference", "/other/input.cif")])
def test_recipe_identity_never_merges_distinct_scientific_settings(field, value):
    base = {"inputs": ["first"], "out_dir": "one", "run_name_": "first", "seed": 1, "num_replicates": 2}
    same = {**base, "inputs": ["second"], "out_dir": "two", "run_name_": "second", "seed": 2}
    identity = prepare.recipe_identity(base, "generate", {"CLUSTER": "digs"}, [])
    assert identity == prepare.recipe_identity(same, "generate", {"CLUSTER": "digs"}, [])
    assert identity != prepare.recipe_identity({**same, field: value}, "generate", {"CLUSTER": "digs"}, [])
    assert identity != prepare.recipe_identity(same, "generate", {"CLUSTER": "elsewhere"}, [])


def test_recipe_identity_normalizes_only_exact_generated_hydra_output_binding():
    first = {"out_dir": "/scratch/one", "hydra": {"run": {"dir": "/scratch/one/hydra"}}}
    second = {"out_dir": "/scratch/two", "hydra": {"run": {"dir": "/scratch/two/hydra"}}}
    identity = prepare.recipe_identity(first, "beam", {}, [])
    assert identity == prepare.recipe_identity(second, "beam", {}, [])
    assert first["hydra"]["run"]["dir"] == "/scratch/one/hydra"  # No live scientific config mutation.
    second["hydra"]["run"]["dir"] = "/custom/logs"
    assert identity != prepare.recipe_identity(second, "beam", {}, [])


def test_recipe_homogeneity_is_rechecked_even_with_rehashed_plan(sweep):
    result = plan(sweep, group_by_recipe=True)
    result["bundles"][0]["recipe_id"] = "a" * 64
    for bundle in result["bundles"]:
        bundle["bundle_id"] = "b_" + prepare.fingerprint({key: value for key, value in bundle.items() if key != "bundle_id"})[:24]
    result["plan_id"] = "p_" + prepare.fingerprint({key: value for key, value in result.items() if key != "plan_id"})[:24]
    with pytest.raises(ValueError, match="incompatible scientific recipes"):
        prepare.prepare_sweep(result, sweep["root"] / "never-written")
    assert not (sweep["root"] / "never-written").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"gpus": 0},
        {"gpus": 2},
        {"nodes": 2},
        {"ranks": 2},
        {"runtime_seconds": None},
        {"runtime_confidence": "unknown"},
    ],
)
def test_explicit_single_gpu_calibrated_resources_required(sweep, change):
    sweep["workloads"]["generate"]["resources"] = {**sweep["workloads"]["generate"]["resources"], **change}
    with pytest.raises(ValueError):
        plan(sweep)


def test_plan_mutation_and_runtime_source_mutation_rejected(sweep):
    planned = plan(sweep)
    changed = deepcopy(planned)
    changed["bundles"][0]["rows"][0]["seed"] += 1
    with pytest.raises(ValueError, match="Plan contents changed"):
        prepare.prepare_sweep(changed, sweep["root"] / "rejected")
    prepare.RUNTIME_SCRIPT.write_text("changed runtime\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.prepare_sweep(planned, sweep["root"] / "rejected")


def test_extra_source_hashes_and_protected_runtime_environment(sweep):
    extra = sweep["root"] / "custom_metric.py"
    extra.write_text("CUSTOM = 1\n")
    sweep["runtime"]["source_hashes"] = {str(extra): sha(extra)}
    result = plan(sweep)
    assert all(bundle["source_hashes"][str(extra)] == sha(extra) for bundle in result["bundles"])
    sweep["runtime"]["environment"] = {"HHQ_SERVER_EPOCH": "999"}
    with pytest.raises(ValueError, match="ownership"):
        plan(sweep)


def test_launchers_are_not_accepted_as_runtime(sweep):
    launcher = sweep["root"] / "launch_generate.sh"
    launcher.write_text("#!/bin/sh\n")
    sweep["runtime"]["argv"] = [str(launcher)]
    with pytest.raises(ValueError, match="no scheduler"):
        plan(sweep)


def test_dotenv_profile_selection_is_exact_and_hash_bound(sweep):
    result = plan(sweep)
    row = result["bundles"][0]["rows"][0]
    assert row["dotenv"]["path"] == str(sweep["repo"] / ".env.digs")
    assert result["bundles"][0]["source_hashes"][row["dotenv"]["path"]] == row["dotenv"]["sha256"]
    (sweep["repo"] / ".env.digs").write_text("EXAMPLE=changed\n")
    with pytest.raises(ValueError, match="profile selection changed|SHA256 mismatch"):
        prepare.prepare_sweep(result, sweep["root"] / "rejected")
    private = sweep["repo"] / ".ipd"
    private.mkdir()
    (private / ".env.digs").write_text("EXAMPLE=private\n")
    assert plan(sweep)["bundles"][0]["rows"][0]["dotenv"]["path"] == str(private / ".env.digs")


@pytest.mark.parametrize("name", ["AF3_LAUNCHER_SOCKET", "AF3_SOCKET_DIR", "LOCAL_RANK", "WORLD_SIZE"])
def test_additional_native_ownership_environment_rejected(sweep, name):
    sweep["runtime"]["environment"] = {name: "override"}
    with pytest.raises(ValueError, match="ownership"):
        plan(sweep)


def test_new_higher_priority_dotenv_profile_rejected_before_preparation(sweep):
    result = plan(sweep)
    private = sweep["repo"] / ".ipd"
    private.mkdir()
    (private / ".env.digs").write_text("OTHER=changed\n")
    with pytest.raises(ValueError, match="profile selection changed"):
        prepare.prepare_sweep(result, sweep["root"] / "rejected")
    assert not (sweep["root"] / "rejected").exists()


def test_local_absent_dotenv_profile_recorded_and_later_creation_rejected(sweep):
    for row in sweep["manifest"]["rows"]:
        row["command"] = row["command"].replace("CLUSTER=digs", "CLUSTER=local")
    refresh_commands(sweep)
    result = plan(sweep)
    assert result["bundles"][0]["rows"][0]["dotenv"] == {"cluster": "local", "path": None, "sha256": None}
    saved = prepare.prepare_sweep(result, sweep["root"] / "prepared")
    (sweep["repo"] / ".env.local").write_text("OTHER=changed\n")
    with pytest.raises(ValueError, match="profile selection changed"):
        prepare.verify_prepared(saved)


def test_lexical_input_alias_preserved_for_native_hydra_identity(sweep):
    original = Path(sweep["manifest"]["rows"][0]["input"])
    alias = sweep["root"] / "input_alias.cif"
    alias.symlink_to(original)
    sweep["manifest"]["input_sha256"][str(alias)] = sha(alias)
    for row in sweep["manifest"]["rows"]:
        row["input"] = str(alias)
        config = Path(row["config"])
        document = yaml.safe_load(config.read_text())
        document["inputs"] = [str(alias)]
        config.write_text(yaml.safe_dump(document))
        sweep["manifest"]["file_sha256"][str(config)] = sha(config)
    rewrite(sweep)
    result = plan(sweep)
    assert all(row["input"] == str(alias) for bundle in result["bundles"] for row in bundle["rows"])
    saved = prepare.prepare_sweep(result, sweep["root"] / "prepared")
    assert prepare.verify_prepared(saved) == saved


def test_runtime_cannot_reintroduce_explicit_native_unset(sweep):
    sweep["runtime"]["environment"] = {"EXEC_SKIP_AF3_DAEMON": "1"}
    with pytest.raises(ValueError, match="conflicts with native unset"):
        plan(sweep)


def test_runtime_environment_exposed_to_task_startup(sweep):
    sweep["runtime"]["environment"] = {"PYTHONHASHSEED": "7", "LD_PRELOAD": "/explicit/operator/library.so"}
    result = plan(sweep)
    saved = prepare.prepare_sweep(result, sweep["root"] / "prepared")
    assert all(group["environment"] == sweep["runtime"]["environment"] for group in saved["groups"])
    assert prepare.verify_prepared(saved) == saved


def test_bundle_pins_are_linear_in_own_rows_but_plan_verifies_all_rows(sweep):
    """Thousands of native rows must not become thousands of pins per worker."""
    for index, row in enumerate(sweep["manifest"]["rows"]):
        ligand = sweep["root"] / f"distinct_input_{index}.cif"
        ligand.write_text(f"data_input_{index}\n")
        row["input"] = str(ligand)
        sweep["manifest"]["input_sha256"][str(ligand)] = sha(ligand)
        config = Path(row["config"])
        document = yaml.safe_load(config.read_text())
        document["inputs"] = [str(ligand)]
        config.write_text(yaml.safe_dump(document))
        sweep["manifest"]["file_sha256"][str(config)] = sha(config)
    rewrite(sweep)
    result = plan(sweep)
    all_row_paths = {row[key] for row in sweep["manifest"]["rows"] for key in ("config", "input")}
    for bundle in result["bundles"]:
        own_paths = {row[key] for row in bundle["rows"] for key in ("config", "input")}
        assert set(bundle["source_hashes"]) & all_row_paths == own_paths
        assert str(sweep["path"]) in bundle["source_hashes"]
        assert sweep["manifest"]["commands_file"] not in bundle["source_hashes"]
    assert all_row_paths <= result["preparation_source_hashes"].keys()
    saved = prepare.prepare_sweep(result, sweep["root"] / "prepared")
    outside_first_bundle = result["bundles"][-1]["rows"][0]["config"]
    assert outside_first_bundle not in result["bundles"][0]["source_hashes"]
    with Path(outside_first_bundle).open("a") as handle:
        handle.write("# changed after planning\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.prepare_sweep(result, sweep["root"] / "refused")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.verify_prepared(saved)


def test_global_commands_binding_is_checked_once_without_worker_fanout(sweep):
    result = plan(sweep)
    commands = Path(sweep["manifest"]["commands_file"])
    assert all(str(commands) not in bundle["source_hashes"] for bundle in result["bundles"])
    assert result["preparation_source_hashes"][str(commands)] == sha(commands)
    commands.write_text("modified global commands\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.prepare_sweep(result, sweep["root"] / "refused")


def test_historical_verification_keeps_snapshots_but_not_live_source_requirements(sweep):
    saved = prepare.prepare_sweep(plan(sweep), sweep["root"] / "prepared")
    Path(sweep["manifest"]["rows"][0]["input"]).unlink()
    prepare.RUNTIME_SCRIPT.write_text("a later runtime version\n")
    assert prepare.verify_prepared(saved, check_sources=False) == saved
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare.verify_prepared(saved)
    bundle_path = next(iter(saved["bundle_files"].values()))["path"]
    with Path(bundle_path).open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare.verify_prepared(saved, check_sources=False)


def test_optional_retry_policy_survives_packaging_and_validation(sweep):
    policy = {"max_attempts": 6, "reasons": ["worker_lost", "preempted", "timeout", "host_oom"],
              "oom_memory_multipliers": [1.25, 1.5, 2.0]}
    sweep["workloads"]["generate"]["retry"] = policy
    planned = plan(sweep)
    bundles = [b for b in planned["bundles"] if b["method"] == "generate"]
    assert all(b["workload"]["max_attempts"] == 6 for b in bundles)
    assert all(b["workload"]["retry"]["oom_memory_multipliers"] == [1.25, 1.5, 2.0] for b in bundles)
    saved = prepare.prepare_sweep(planned, sweep["root"] / "retry_prepared")
    assert prepare.verify_prepared(saved) == saved
    group = next(g for g in saved["groups"] if g["method"] == "generate")
    assert group["retry"] == bundles[0]["workload"]["retry"]
    assert all("retry" not in g for g in saved["groups"] if g["method"] != "generate")


@pytest.mark.parametrize("change", [
    {"max_attempts": 2, "retry": {"max_attempts": 3}},
    {"retry": {"oom_memory_multipliers": [2.0, 1.25]}},
    {"retry": []},
])
def test_invalid_or_conflicting_retry_policy_rejected(sweep, change):
    sweep["workloads"]["generate"].update(change)
    with pytest.raises(ValueError):
        plan(sweep)
