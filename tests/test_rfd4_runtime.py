import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyper_hyperqueue.integrations.rfd4 import runtime as rt


def test_owned_tmpdir_is_recreated_and_tempfile_cache_scoped_between_rows(tmp_path, monkeypatch):
    import shutil

    owned = tmp_path / "owned"
    stale = tmp_path / "stale"
    monkeypatch.setenv("HHQ_RFD4_TMPDIR", str(owned))
    monkeypatch.setenv("TMPDIR", str(stale))
    previous = tempfile.tempdir
    try:
        tempfile.tempdir = str(stale)
        for _ in range(2):
            with rt.row_environment({"TMPDIR": str(stale)}):
                assert tempfile.gettempdir() == str(owned)
                with tempfile.TemporaryDirectory() as work:
                    assert Path(work).parent == owned
                shutil.rmtree(owned)
            assert tempfile.tempdir == str(stale)
            assert os.environ["TMPDIR"] == str(stale)
    finally:
        tempfile.tempdir = previous


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


class Omega:
    @staticmethod
    def to_container(value, **kwargs):
        return copy.deepcopy(dict(value))


class Plugin:
    def __init__(self, auto_requeue=True):
        self.auto_requeue = auto_requeue


@pytest.fixture
def fixture(tmp_path):
    config = tmp_path / "input.yaml"
    source = tmp_path / "input.cif"
    config.write_text("native-config\n")
    source.write_text("data_input\n")
    row = {
        "run_id": "example",
        "config": str(config),
        "config_sha256": rt.digest(config),
        "input": str(source),
        "input_sha256": rt.digest(source),
        "seed": 123,
        "replicates": 2,
        "output_dir": str(tmp_path / "outputs"),
        "environment": {},
        "unset_environment": [],
    }
    bundle = {
        "schema": "hhq-rfd4-bundle/1",
        "bundle_id": "fixture",
        "repo": str(tmp_path),
        "source_manifest": {"path": str(config), "sha256": rt.digest(config)},
        "runtime": {"argv": [sys.executable], "environment": {}},
        "rows": [row],
        "source_hashes": {str(config): rt.digest(config), str(source): rt.digest(source)},
    }
    cfg = Config(
        seed=123,
        inputs=[str(source)],
        out_dir=row["output_dir"],
        job_id=0,
        hardware={"ngpus_per_node_": 1, "nnodes_": 1},
        precision="32",
        skip_existing=True,
        hf_ckpt_path="pinned/model",
        hf_ckpt_revision="test-revision",
    )
    state = SimpleNamespace(
        batches=[["root_rep0"], ["root_rep1"]],
        expected=["root_rep0", "root_rep1"],
        mode="normal",
        loads=[],
        trainers=[],
        cfgs=[],
        files=True,
    )
    native = SimpleNamespace()
    native.get_dataset_from_cfg = lambda cfg, job_id: list(state.expected)
    native._get_example_ids_from_dataset = lambda dataset: dataset

    def load(cfg, model=None):
        state.loads.append(model)
        return model if model is not None else object()

    def trainer(**kwargs):
        state.trainers.append(kwargs)
        return SimpleNamespace(sanity_checking=False, **kwargs)

    native.load_ckpt_n_configure_inference = load
    native.L = SimpleNamespace(Trainer=trainer, seed_everything=lambda *args, **kwargs: None)

    def run(cfg, job_id=0, config_name=None, *, model=None):
        state.cfgs.append(cfg)
        cfg.seed += job_id
        native.load_ckpt_n_configure_inference(cfg, model=model)
        native.get_dataset_from_cfg(cfg, job_id)
        trained = native.L.Trainer(callbacks=[])
        for index, names in enumerate(state.batches):
            for callback in trained.callbacks:
                callback.on_validation_batch_end(trained, None, None, {"example_id_batch": names}, index)
        output = Path(cfg.out_dir)
        if state.files:
            output.mkdir(exist_ok=True)
            (output / "metrics.csv").write_text("example_id,value\nroot_rep0,1\n")
        if state.mode == "exit":
            raise SystemExit(0)
        if state.mode == "fail":
            raise ValueError("synthetic failed application")
        if state.mode == "signal":
            raise InterruptedError("synthetic interruption")

    native.run_generation = run
    execute = lambda **kwargs: rt.execute_row(
        bundle,
        row,
        "bundle-fingerprint",
        native=native,
        compose_fn=lambda row: (cfg, Omega),
        environment_class=Plugin,
        callback_class=object,
        **kwargs,
    )
    return SimpleNamespace(bundle=bundle, row=row, cfg=cfg, state=state, native=native, execute=execute)


def test_scoped_real_plugin_contract_coverage_and_seed(fixture, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-allocated-uuid")
    original = (
        fixture.native.L,
        fixture.native.get_dataset_from_cfg,
        fixture.native.load_ckpt_n_configure_inference,
    )
    result = fixture.execute()
    receipt = rt.read_json(result["path"])
    assert receipt["coverage"]["processed_roots"] == 2
    assert receipt["effective_seed"] == 123
    assert fixture.cfg.seed == 123
    assert fixture.state.trainers[0]["plugins"][0].auto_requeue is False
    assert os.environ["SLURM_JOB_ID"] == "123"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-allocated-uuid"
    assert original == (
        fixture.native.L,
        fixture.native.get_dataset_from_cfg,
        fixture.native.load_ckpt_n_configure_inference,
    )
    assert fixture.execute()["reused"] is True
    assert len(fixture.state.cfgs) == 1


def test_zero_selected_structures_is_processed_completion(fixture):
    fixture.state.files = False
    receipt = rt.read_json(fixture.execute()["path"])
    assert receipt["artifact_count"] == 0
    assert receipt["coverage"]["processed_roots"] == 2


@pytest.mark.parametrize("batches", [[["root_rep0"]], [["root_rep0"], ["root_rep0"], ["root_rep1"]]])
def test_missing_or_duplicate_processed_roots_fail(fixture, batches):
    fixture.state.batches = batches
    with pytest.raises(rt.RuntimeContractError, match="Incomplete native root coverage"):
        fixture.execute()
    assert len(list(Path(fixture.row["output_dir"]).parent.glob(".hhq-rfd4/*/failures/*.json"))) == 1


@pytest.mark.parametrize(
    "mode, error", [("exit", rt.RuntimeContractError), ("fail", ValueError), ("signal", InterruptedError)]
)
def test_native_early_exit_exception_interruption_not_completion(fixture, mode, error):
    fixture.state.mode = mode
    original = fixture.native.L
    with pytest.raises(error):
        fixture.execute()
    assert fixture.native.L is original
    assert not list(Path(fixture.row["output_dir"]).parent.glob(".hhq-rfd4/*/completion.json"))
    fixture.state.mode = "normal"
    assert fixture.execute()["reused"] is False  # matching partial outputs are preserved, then regenerated
    assert fixture.state.cfgs[-1].seed == 123


def test_interrupted_row_cannot_reuse_nonempty_truncated_native_output(fixture):
    fixture.state.mode = "signal"
    with pytest.raises(InterruptedError):
        fixture.execute()
    output = Path(fixture.row["output_dir"])
    partial = output / "inference_samples" / "root_rep0" / "root_rep0_generated.cif"
    partial.parent.mkdir(parents=True)
    partial.write_text("data_incomplete\n_atom_site.")
    original = fixture.native.run_generation

    def no_partial_inputs(cfg, **kwargs):
        assert not Path(cfg.out_dir).exists(), "Interrupted native outputs must never reach skip_existing"
        return original(cfg, **kwargs)

    fixture.native.run_generation = no_partial_inputs
    fixture.state.mode = "normal"
    result = fixture.execute()
    receipt = rt.read_json(result["path"])
    archive = Path(receipt["preserved_interrupted_outputs"])
    assert (archive / partial.relative_to(output)).read_text() == "data_incomplete\n_atom_site."
    assert not partial.exists()
    assert fixture.execute()["reused"] is True
    assert len(list(archive.parent.iterdir())) == 1


def test_three_row_bundle_signal_restarts_only_unfinished_rows(fixture, monkeypatch):
    bundle = copy.deepcopy(fixture.bundle)
    bundle["reuse_model"] = True
    rows = []
    for number in range(3):
        row = dict(fixture.row, run_id=f"row{number}", seed=123 + number,
                   output_dir=str(Path(fixture.row["output_dir"]).with_name(f"row{number}")))
        rows.append(row)
    bundle["rows"] = rows
    path = Path(bundle["repo"]) / "three-rows.json"
    rt.atomic_json(path, bundle)
    digest = rt.digest(path)
    task_output = Path(bundle["repo"]) / "task-output"
    monkeypatch.setenv("HHQ_TASK_OUTPUT_DIR", str(task_output))
    monkeypatch.setattr(rt, "load_native", lambda *args: (fixture.native, Plugin, object))

    def compose(row):
        cfg = copy.deepcopy(fixture.cfg)
        cfg.seed, cfg.out_dir = row["seed"], row["output_dir"]
        return cfg, Omega

    monkeypatch.setattr(rt, "compose_native", compose)
    original = fixture.native.run_generation
    seen = []

    def run(cfg, **kwargs):
        seen.append(cfg.seed)
        original(cfg, **kwargs)
        if cfg.seed == 124 and seen.count(124) == 1:
            os.kill(os.getpid(), signal.SIGTERM)

    fixture.native.run_generation = run
    with pytest.raises(InterruptedError, match="signal"):
        rt.run_bundle(path, digest)
    assert seen == [123, 124]
    assert not (task_output / "rfd4_completion.json").exists()
    rt.run_bundle(path, digest)
    completion = rt.read_json(task_output / "rfd4_completion.json")
    assert seen == [123, 124, 124, 125]
    assert [row["reused"] for row in completion["rows"]] == [True, False, False]
    interrupted = rt.read_json(completion["rows"][1]["path"])["preserved_interrupted_outputs"]
    assert (Path(interrupted) / "metrics.csv").is_file()
    (Path(rows[0]["output_dir"]) / "metrics.csv").write_text("tampered")
    with pytest.raises(rt.RuntimeContractError, match="outputs changed"):
        rt.run_bundle(path, digest)
    assert seen == [123, 124, 124, 125]


def test_refuse_foreign_output_owner_and_changed_completed_artifact(fixture):
    output = Path(fixture.row["output_dir"])
    output.mkdir()
    (output / "foreign.cif").write_text("data_foreign\n")
    with pytest.raises(rt.RuntimeContractError, match="unrelated"):
        fixture.execute()
    (output / "foreign.cif").unlink()
    fixture.execute()
    (output / "metrics.csv").write_text("example_id,value\nroot_rep0,9\n")
    with pytest.raises(rt.RuntimeContractError, match="outputs changed"):
        fixture.execute()


def test_changed_seed_cannot_adopt_prior_owned_outputs(fixture):
    fixture.execute()
    fixture.row["seed"] = 124
    fixture.cfg.seed = 124
    with pytest.raises(rt.RuntimeContractError, match="another immutable row"):
        fixture.execute()


def test_explicit_seed_single_gpu_and_oracle_transport(fixture, monkeypatch):
    fixture.cfg.seed = None
    with pytest.raises(rt.RuntimeContractError, match="integer seed"):
        rt.validate_config(fixture.cfg, fixture.row)
    fixture.cfg.seed = 123
    fixture.cfg.hardware["ngpus_per_node_"] = 2
    with pytest.raises(rt.RuntimeContractError, match="one GPU"):
        rt.validate_config(fixture.cfg, fixture.row)
    fixture.cfg.metrics = {"refolding_oracles": {"oracle": {}}}
    for visible in ("GPU-assigned-uuid", "0,1", ""):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        with pytest.raises(rt.RuntimeContractError, match="numeric"):
            rt.check_gpu_transport(fixture.cfg)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")
    rt.check_gpu_transport(fixture.cfg)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6"


@pytest.mark.parametrize(
    "name",
    [
        "CUDA_VISIBLE_DEVICES",
        "SLURM_JOB_NAME",
        "NVIDIA_VISIBLE_DEVICES",
        "HHQ_ATTEMPT_ID",
        "HQ_TASK_ID",
        "AF3_LAUNCHER_SOCKET_TEMPLATE",
        "HIP_VISIBLE_DEVICES",
    ],
)
def test_no_allocation_environment_override(name):
    with (
        pytest.raises(rt.RuntimeContractError, match="Cannot override"),
        rt.row_environment({name: "pretend"}),
    ):
        pytest.fail("protected environment accepted")
    with pytest.raises(rt.RuntimeContractError, match="Cannot override"), rt.row_environment({}, [name]):
        pytest.fail("protected environment unset accepted")


def test_model_reuse_compatible_no_adapter_and_fresh_configs(fixture):
    cache = {}
    fixture.execute(model_cache=cache)
    first_model = cache["model"]
    fixture.row["run_id"] = "second"
    fixture.row["output_dir"] = str(Path(fixture.row["output_dir"]).with_name("second"))
    fixture.cfg.out_dir = fixture.row["output_dir"]
    fixture.execute(model_cache=cache)
    assert fixture.state.loads == [None, first_model]
    assert fixture.state.cfgs[0] is not fixture.state.cfgs[1]
    fixture.cfg.adapter_path = "adapter"
    with pytest.raises(rt.RuntimeContractError, match="LoRA"):
        rt.reuse_key(fixture.cfg)
    fixture.cfg.adapter_path = None
    fixture.cfg.precision = "16-mixed"
    with (
        rt.native_hooks(
            fixture.native,
            rt.Coverage(),
            environment_class=Plugin,
            callback_class=object,
            model_cache=cache,
            model_key=rt.reuse_key(fixture.cfg),
        ),
        pytest.raises(rt.RuntimeContractError, match="Incompatible"),
    ):
        fixture.native.load_ckpt_n_configure_inference(fixture.cfg)


def test_pin_tampering_is_detected_before_science(fixture, tmp_path):
    path = tmp_path / "bundle.json"
    rt.atomic_json(path, fixture.bundle)
    fingerprint = rt.digest(path)
    rt.check_pins(fixture.bundle, path, fingerprint)
    Path(fixture.row["input"]).write_text("changed\n")
    with pytest.raises(rt.RuntimeContractError, match="Pinned source changed"):
        rt.check_pins(fixture.bundle, path, fingerprint)
    with pytest.raises(rt.RuntimeContractError, match="Bundle fingerprint"):
        rt.check_pins(fixture.bundle, path, "wrong")


def test_pin_changed_during_science_has_no_completion(fixture):
    original = fixture.native.run_generation

    def changed(*args, **kwargs):
        original(*args, **kwargs)
        Path(fixture.row["input"]).write_text("different\n")

    fixture.native.run_generation = changed
    with pytest.raises(rt.RuntimeContractError, match="changed during generation"):
        fixture.execute()


def test_standalone_has_no_hhq_or_science_import_for_help():
    result = subprocess.run(
        [sys.executable, "-S", str(Path(rt.__file__)), "--help"], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--check" in result.stdout


def test_standalone_cli_pin_failure_records_failed_attempt(fixture, tmp_path):
    path = tmp_path / "bundle.json"
    rt.atomic_json(path, fixture.bundle)
    output = tmp_path / "attempt"
    result = subprocess.run(
        [sys.executable, "-S", str(Path(rt.__file__)), "--bundle", str(path), "--fingerprint", "wrong"],
        check=False,
        env={**os.environ, "HHQ_TASK_OUTPUT_DIR": str(output)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert rt.read_json(output / "rfd4_failure.json")["status"] == "failed"
    assert not (output / "rfd4_completion.json").exists()


def test_lock_rejects_concurrent_duplicate(tmp_path):
    with (
        rt.exclusive_lock(tmp_path / "execution.lock"),
        pytest.raises(rt.RuntimeContractError, match="Another execution owns"),
        rt.exclusive_lock(tmp_path / "execution.lock"),
    ):
        pytest.fail("duplicate lock accepted")


def test_reevaluation_union_and_sanity_batches():
    coverage = rt.Coverage()
    coverage.set_dataset(["new", "existing"])
    coverage.batch({"example_id_batch": ["new"]})
    coverage.batch({"example_id_batch": ["existing"]})
    assert coverage.finish()["processed_roots"] == 2


def test_preflight_no_output_creation(fixture, tmp_path, monkeypatch):
    fixture.native.run_generation = lambda cfg, job_id=0, *, model=None: pytest.fail("science ran")
    monkeypatch.setattr(rt, "load_native", lambda bundle, row: (fixture.native, Plugin, object))
    monkeypatch.setattr(rt, "compose_native", lambda row: (fixture.cfg, Omega))
    path = tmp_path / "bundle.json"
    rt.atomic_json(path, fixture.bundle)
    result = rt.preflight(path, rt.digest(path))
    assert result["models_loaded"] is False
    assert result["generation_run"] is False
    assert result["gpu_runtime_qualified"] is False
    assert not Path(fixture.row["output_dir"]).exists()
    assert not (tmp_path / ".hhq-rfd4").exists()


def test_exact_native_dotenv_selection_and_absence_pin(fixture, monkeypatch):
    monkeypatch.setenv("CLUSTER", "local")
    fixture.row["dotenv"] = {"cluster": "local", "path": None, "sha256": None}
    rt.validate_dotenv_binding(fixture.bundle, fixture.row)
    env_file = Path(fixture.bundle["repo"]) / ".env.local"
    env_file.write_text("SYNTHETIC_TEST=1\n")
    with pytest.raises(rt.RuntimeContractError, match="dotenv selection changed"):
        rt.validate_dotenv_binding(fixture.bundle, fixture.row)
    fixture.row["dotenv"] = {"cluster": "local", "path": str(env_file), "sha256": rt.digest(env_file)}
    rt.validate_dotenv_binding(fixture.bundle, fixture.row)
    monkeypatch.setenv("CLUSTER", "missing")
    with pytest.raises(rt.RuntimeContractError, match="profile is missing"):
        rt.validate_dotenv_binding(fixture.bundle, fixture.row)


def test_shared_storage_mkdir_negative_cache_retries(tmp_path, monkeypatch):
    original = Path.mkdir
    target = tmp_path / "sidecar"
    calls = []

    def delayed(path, *args, **kwargs):
        if path == target:
            calls.append(path)
            if len(calls) == 1:
                raise FileExistsError("simulated remote mkdir negative dentry")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", delayed)
    monkeypatch.setattr(rt.time, "sleep", lambda duration: None)
    assert rt.ensure_directory(target) == target
    assert len(calls) == 2


def test_actual_standalone_cli_with_synthetic_native_modules(tmp_path):
    """Exercise executable imports/hooks/receipts without Torch, HQ, or a GPU."""
    repo = tmp_path / "synthetic_native"
    modules = {
        "rfproteina/__init__.py": "",
        "rfproteina/generate.py": """
import os
from pathlib import Path
import lightning as L
def load_dotenv(): pass
def get_dataset_from_cfg(cfg, job_id): return ["root0", "root1"]
def _get_example_ids_from_dataset(dataset): return dataset
def load_ckpt_n_configure_inference(cfg, model=None): return model or object()
def run_generation(cfg, job_id=0, config_name=None, *, model=None):
    if os.environ.get("SYNTHETIC_EARLY_EXIT"): raise SystemExit(0)
    cfg.seed += job_id
    model = load_ckpt_n_configure_inference(cfg, model=model)
    dataset = get_dataset_from_cfg(cfg, job_id)
    trainer = L.Trainer(callbacks=[])
    assert trainer.plugins[0].auto_requeue is False
    trainer.validate(model, dataset)
    output = Path(cfg.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.csv").write_text("example_id,metric\\nroot0,1\\nroot1,2\\n")
""",
        "lightning/__init__.py": """
class Trainer:
    def __init__(self, callbacks, plugins):
        self.callbacks, self.plugins, self.sanity_checking = callbacks, plugins, False
    def validate(self, model, dataset):
        for i, item in enumerate(dataset):
            for callback in self.callbacks:
                callback.on_validation_batch_end(self, model, None, {"example_id_batch": [item]}, i)
""",
        "lightning/fabric/__init__.py": "",
        "lightning/fabric/plugins/__init__.py": "",
        "lightning/fabric/plugins/environments.py": """
class SLURMEnvironment:
    def __init__(self, auto_requeue=True): self.auto_requeue = auto_requeue
""",
        "lightning/pytorch/__init__.py": "",
        "lightning/pytorch/callbacks.py": "class Callback: pass\n",
        "omegaconf.py": """
class Config(dict):
    def __getattr__(self, key):
        try: return self[key]
        except KeyError as error: raise AttributeError(key) from error
    def __setattr__(self, key, value): self[key] = value
class OmegaConf:
    @staticmethod
    def to_container(cfg, **kwargs): return dict(cfg)
""",
        "hydra/__init__.py": """
from contextlib import contextmanager
import json
from pathlib import Path
from omegaconf import Config
directory = None
@contextmanager
def initialize_config_dir(config_dir, version_base):
    global directory
    directory = config_dir
    try: yield
    finally: directory = None
def compose(config_name):
    return Config(json.loads((Path(directory) / (config_name + ".yaml")).read_text()))
""",
    }
    for name, contents in modules.items():
        target = repo / "src" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    source = repo / "input.cif"
    source.write_text("data_synthetic\n")
    output = tmp_path / "scientific_outputs"
    config = repo / "run.yaml"
    config.write_text(
        json.dumps(
            {
                "seed": 42,
                "inputs": [str(source)],
                "out_dir": str(output),
                "hardware": {"ngpus_per_node_": 1, "nnodes_": 1},
                "job_id": 0,
            }
        )
    )
    manifest = repo / "manifest.json"
    manifest.write_text("{}\n")
    row = {
        "run_id": "run",
        "config": str(config),
        "config_sha256": rt.digest(config),
        "input": str(source),
        "input_sha256": rt.digest(source),
        "seed": 42,
        "output_dir": str(output),
        "dotenv": {"cluster": "local", "path": None, "sha256": None},
        "environment": {"CLUSTER": "local"},
    }
    bundle = {
        "schema": "hhq-rfd4-bundle/1",
        "bundle_id": "synthetic",
        "repo": str(repo),
        "rows": [row],
        "runtime": {"argv": [sys.executable], "environment": {}},
        "source_manifest": {"path": str(manifest), "sha256": rt.digest(manifest)},
        "source_hashes": {str(p): rt.digest(p) for p in repo.rglob("*") if p.is_file()},
    }
    path = tmp_path / "bundle.json"
    rt.atomic_json(path, bundle)
    argv = [sys.executable, "-S", rt.__file__, "--bundle", str(path), "--fingerprint", rt.digest(path)]
    check = subprocess.run([*argv, "--check"], capture_output=True, text=True, check=False)
    assert check.returncode == 0, check.stderr
    assert json.loads(check.stdout)["generation_run"] is False
    assert not output.exists()
    environment = {**os.environ, "HHQ_TASK_OUTPUT_DIR": str(tmp_path / "attempt"), "SLURM_JOB_ID": "567"}
    first = subprocess.run(argv, capture_output=True, text=True, check=False, env=environment)
    assert first.returncode == 0, first.stderr
    completion = rt.read_json(tmp_path / "attempt/rfd4_completion.json")
    assert completion["row_count"] == 1
    assert rt.read_json(completion["rows"][0]["path"])["coverage"]["processed_roots"] == 2
    second = subprocess.run(argv, capture_output=True, text=True, check=False, env=environment)
    assert second.returncode == 0, second.stderr
    assert rt.read_json(tmp_path / "attempt/rfd4_completion.json")["rows"][0]["reused"] is True
