"""Execute pinned native RFD4 configurations inside their explicitly selected runtime.

This file is also an executable: top-level imports deliberately use only stdlib.
It does not import HHQ, submit jobs, invoke private RFD4 shell wrappers, or edit
scientific configurations. Native dependencies are imported only after pin checks.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib
import inspect
import json
import os
import signal
import sys
import tempfile
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


class RuntimeContractError(RuntimeError):
    """An execution cannot safely satisfy its immutable bundle contract."""


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ensure_directory(path):
    """Bound shared-storage negative-dentry retries; never remove anything."""
    path = Path(path)
    for attempt in range(10):
        try:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            return path
        except (FileExistsError, FileNotFoundError):
            for parent in (path.parent, path.parent.parent):
                try:
                    with os.scandir(parent) as entries:
                        for _ in entries:
                            pass
                except FileNotFoundError:
                    continue
            if path.is_dir():
                return path
            if attempt == 9:
                raise
            time.sleep(0.1)
    raise AssertionError("unreachable")


def atomic_json(path, value):
    path = Path(path)
    ensure_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeContractError(f"Another execution owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def protected_environment(name):
    name = name.removeprefix("APPTAINERENV_").removeprefix("SINGULARITYENV_")
    return name.startswith(
        ("SLURM_", "HHQ_", "HQ_", "CUDA_", "NVIDIA_", "ROCR_", "HIP_", "AF3_LAUNCHER_")
    ) or name in {"AF3_SOCKET_DIR", "LOCAL_RANK", "RANK", "WORLD_SIZE"}


@contextmanager
def row_environment(environment, unset=()):
    names = set(environment) | set(unset)
    forbidden = sorted(name for name in names if protected_environment(name))
    if forbidden:
        raise RuntimeContractError(f"Cannot override assigned execution environment: {forbidden}")
    old = dict(os.environ)
    old_path = list(sys.path)
    old_tempdir = tempfile.tempdir
    try:
        for name in unset:
            os.environ.pop(name, None)
        os.environ.update({name: str(value) for name, value in environment.items()})
        if owned_tmp := old.get("HHQ_RFD4_TMPDIR"):
            ensure_directory(owned_tmp)
            os.environ.update({name: owned_tmp for name in ("TMPDIR", "TMP", "TEMP")})
            # tempfile caches its first result across rows in a warm bundle.
            tempfile.tempdir = owned_tmp
        # Setting PYTHONPATH after interpreter startup alone does not update imports.
        sys.path[:0] = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        yield
    finally:
        # A daemon started by native Oracle.setup belongs to this task and may be
        # reused by its next row. All other row-specific settings are restored.
        daemon = {k: v for k, v in os.environ.items() if k.startswith("AF3_LAUNCHER_")}
        os.environ.clear()
        os.environ.update(old)
        os.environ.update(daemon)
        sys.path[:] = old_path
        tempfile.tempdir = old_tempdir


def check_pins(bundle, bundle_path, fingerprint):
    if digest(bundle_path) != fingerprint:
        raise RuntimeContractError("Bundle fingerprint mismatch")
    if bundle.get("schema") != "hhq-rfd4-bundle/1" or not bundle.get("rows"):
        raise RuntimeContractError("Unsupported or empty RFD4 bundle")
    pins = dict(bundle.get("source_hashes", {}))
    manifest = bundle["source_manifest"]
    pins[manifest["path"]] = manifest["sha256"]
    for row in bundle["rows"]:
        pins[row["config"]] = row["config_sha256"]
        pins[row["input"]] = row["input_sha256"]
    for filename, expected in pins.items():
        if digest(filename) != expected:
            raise RuntimeContractError(f"Pinned source changed: {filename}")
    return pins


def environment_for_row(bundle, row):
    base = bundle["runtime"].get("environment", {})
    per_row = row.get("environment", {})
    conflicts = {name for name in base.keys() & per_row.keys() if str(base[name]) != str(per_row[name])}
    conflicts.update(base.keys() & set(row.get("unset_environment", [])))
    if conflicts:
        raise RuntimeContractError(
            f"Runtime environment conflicts with original native command: {sorted(conflicts)}"
        )
    return {**base, **per_row}


class Coverage:
    """Count processed roots, independently of scientific output selection."""

    def __init__(self):
        self.expected = Counter()
        self.observed = Counter()
        self.dataset_calls = 0
        self.batches = 0

    def set_dataset(self, names):
        self.dataset_calls += 1
        if self.dataset_calls != 1:
            raise RuntimeContractError("Native dataset was constructed more than once")
        self.expected.update(str(name) for name in names)
        if not self.expected:
            raise RuntimeContractError("Native dataset is empty")

    def batch(self, batch):
        names = batch.get("example_id_batch")
        if not isinstance(names, (tuple, list)) or not names:
            raise RuntimeContractError("Native batch lacks example_id_batch")
        self.observed.update(str(name) for name in names)
        self.batches += 1

    def finish(self):
        if self.dataset_calls != 1 or not self.expected or self.expected != self.observed:
            raise RuntimeContractError(
                f"Incomplete native root coverage: missing={dict(self.expected - self.observed)}, "
                f"unexpected={dict(self.observed - self.expected)}"
            )
        return {
            "expected": dict(self.expected),
            "observed": dict(self.observed),
            "batches": self.batches,
            "processed_roots": sum(self.observed.values()),
        }


def reuse_key(cfg):
    if cfg.get("adapter_path"):
        raise RuntimeContractError("Model reuse with a LoRA adapter is not supported")
    fields = (
        "hf_ckpt_path",
        "hf_ckpt_revision",
        "ckpt_path",
        "ckpt_name",
        "autoencoder_ckpt_path",
        "precision",
    )
    return identity({name: cfg.get(name) for name in fields})


@contextmanager
def native_hooks(native, coverage, *, environment_class, callback_class, model_cache=None, model_key=None):
    """Patch only this imported generate module; preserve native scientific code."""
    original_lightning = native.L
    original_dataset = native.get_dataset_from_cfg
    original_load = native.load_ckpt_n_configure_inference

    class ProcessedBatch(callback_class):
        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            # Lightning's optional sanity validation must not count as generation.
            if not getattr(trainer, "sanity_checking", False):
                coverage.batch(batch)

    class LightningProxy:
        def __getattr__(self, name):
            return getattr(original_lightning, name)

        def Trainer(self, *args, **kwargs):
            if kwargs.get("plugins"):
                raise RuntimeContractError("Native Trainer plugins changed; adapter needs qualification")
            kwargs["plugins"] = [environment_class(auto_requeue=False)]
            kwargs["callbacks"] = [*kwargs.get("callbacks", []), ProcessedBatch()]
            return original_lightning.Trainer(*args, **kwargs)

    def dataset(*args, **kwargs):
        result = original_dataset(*args, **kwargs)
        coverage.set_dataset(native._get_example_ids_from_dataset(result))
        return result

    def load(cfg, model=None):
        if model_cache is not None:
            if model_cache.get("key") not in (None, model_key):
                raise RuntimeContractError("Incompatible model reuse group")
            model = model_cache.get("model")
        result = original_load(cfg, model=model)
        if model_cache is not None:
            model_cache.update(key=model_key, model=result)
        return result

    native.L = LightningProxy()
    native.get_dataset_from_cfg = dataset
    native.load_ckpt_n_configure_inference = load
    try:
        yield
    finally:
        native.L = original_lightning
        native.get_dataset_from_cfg = original_dataset
        native.load_ckpt_n_configure_inference = original_load


def artifact_manifest(output_dir):
    """Hash scientific summaries/structures; stat large trajectories/checkpoints."""
    root = Path(output_dir)
    entries = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise RuntimeContractError(f"Output symlink is not an owned artifact: {path}")
            if not path.is_file():
                continue
            stat = path.stat()
            relative = str(path.relative_to(root))
            scientific = path.name.endswith((".csv", ".json", ".cif", ".cif.gz", ".cif.zst", ".pdb"))
            if scientific and stat.st_size == 0:
                raise RuntimeContractError(f"Empty scientific output: {path}")
            entry = {"path": relative, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if scientific:
                entry["sha256"] = digest(path)
            entries.append(entry)
    return entries


def validate_artifacts(output_dir, entries):
    if artifact_manifest(output_dir) != entries:
        raise RuntimeContractError(f"Previously completed outputs changed: {output_dir}")


def oracle_enabled(cfg):
    # Deliberately conservative: an oracle definition can be selected by search,
    # custom provider names, or native metric tags. No class-name guessing.
    metrics = cfg.get("metrics") or {}
    return bool(metrics.get("refolding_oracles"))


def check_gpu_transport(cfg):
    if not oracle_enabled(cfg):
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdecimal():
        raise RuntimeContractError(
            "Native oracle transport requires exactly one numeric CUDA_VISIBLE_DEVICES; "
            "UUID, multiple, or missing visibility needs a separately qualified native transport"
        )


def compose_native(row):
    # Hydra's own composition resolves defaults/searchpath exactly as native CLI.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    path = Path(row["config"])
    with initialize_config_dir(config_dir=str(path.parent.resolve()), version_base="1.3"):
        cfg = compose(config_name=path.stem)
    return copy.deepcopy(cfg), OmegaConf


def validate_dotenv_binding(bundle, row):
    repo = Path(bundle["repo"]).resolve()
    cluster = os.environ.get("CLUSTER")
    if not cluster:
        raise RuntimeContractError("An explicit CLUSTER profile is required")
    if cluster == "local":
        selected = repo / ".env.local"
        if not selected.exists():
            selected = repo / ".env"
        if not selected.exists():
            selected = None
    else:
        selected = repo / ".ipd" / f".env.{cluster}"
        if not selected.exists():
            selected = repo / f".env.{cluster}"
        if not selected.is_file():
            raise RuntimeContractError(f"Native environment profile is missing: {selected}")
    selected = selected.resolve() if selected is not None else None
    current = {
        "cluster": cluster,
        "path": str(selected) if selected is not None else None,
        "sha256": digest(selected) if selected is not None else None,
    }
    if row.get("dotenv") != current:
        raise RuntimeContractError("Native dotenv selection changed or was not pinned during preparation")
    return current


def load_native(bundle, row):
    """Resolve the pinned native module, never a PYTHONPATH replacement."""
    repo = Path(bundle["repo"]).resolve()
    # Read-only source checkouts must not gain Python bytecode from adapter use.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo / "src"))
    validate_dotenv_binding(bundle, row)
    before = {k: v for k, v in os.environ.items() if protected_environment(k)}
    native = importlib.import_module("rfproteina.generate")
    if Path(native.__file__).resolve() != repo / "src/rfproteina/generate.py":
        raise RuntimeContractError("Imported generate module is outside the pinned repository")
    # rfproteina.__init__ performs the first call. Repeat native main's call for
    # each config; its CLUSTER-aware implementation, not generic python-dotenv.
    native.load_dotenv()
    after = {k: v for k, v in os.environ.items() if protected_environment(k)}
    if before != after:
        raise RuntimeContractError("Native dotenv changed protected allocation variables")
    from lightning.fabric.plugins.environments import SLURMEnvironment
    from lightning.pytorch.callbacks import Callback

    return native, SLURMEnvironment, Callback


def preflight(bundle_path, fingerprint):
    """CPU-only composition and interface check, without output/owner creation."""
    bundle = read_json(bundle_path)
    pins = check_pins(bundle, bundle_path, fingerprint)
    reports = []
    reuse_identity = None
    old_cwd, old_path = Path.cwd(), list(sys.path)
    try:
        os.chdir(bundle["repo"])
        for row in bundle["rows"]:
            environment = environment_for_row(bundle, row)
            with row_environment(environment, row.get("unset_environment", [])):
                native, environment_class, _ = load_native(bundle, row)
                signature = inspect.signature(native.run_generation)
                if not {"cfg", "job_id", "model"}.issubset(signature.parameters):
                    raise RuntimeContractError("Native run_generation interface changed")
                if not callable(getattr(native, "_get_example_ids_from_dataset", None)):
                    raise RuntimeContractError("Native dataset identity seam is unavailable")
                if "auto_requeue" not in inspect.signature(environment_class).parameters:
                    raise RuntimeContractError("Lightning cannot disable native Slurm requeue")
                cfg, omega = compose_native(row)
                job_id = validate_config(cfg, row)
                if bundle.get("reuse_model"):
                    key = reuse_key(cfg)
                    if reuse_identity is not None and key != reuse_identity:
                        raise RuntimeContractError("Incompatible model reuse group")
                    reuse_identity = key
                reports.append(
                    {
                        "run_id": row["run_id"],
                        "base_seed": cfg.seed,
                        "job_id": job_id,
                        "oracle_defined": oracle_enabled(cfg),
                        "composed_config_sha256": identity(omega.to_container(cfg, resolve=True)),
                    }
                )
        check_pins(bundle, bundle_path, fingerprint)
        return {
            "schema": "hhq-rfd4-preflight/1",
            "status": "PASS",
            "bundle_sha256": fingerprint,
            "row_count": len(reports),
            "rows": reports,
            "source_pins_checked": len(pins),
            "slurm_auto_requeue": False,
            "models_loaded": False,
            "generation_run": False,
            "gpu_runtime_qualified": False,
            "outputs_created": False,
        }
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def validate_config(cfg, row):
    seed = cfg.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != row["seed"]:
        raise RuntimeContractError("A row requires its original explicit integer seed")
    if Path(str(cfg.get("out_dir"))).resolve() != Path(row["output_dir"]).resolve():
        raise RuntimeContractError("Composed output directory differs from immutable row")
    if list(cfg.get("inputs", [])) != [row["input"]]:
        raise RuntimeContractError("Composed input differs from immutable row")
    hardware = cfg.get("hardware") or {}
    if hardware.get("ngpus_per_node_") != 1 or hardware.get("nnodes_") != 1:
        raise RuntimeContractError("RFD4 adapter supports one GPU on one node per task")
    job_id = cfg.get("job_id")
    job_id = 0 if job_id is None else job_id
    if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id != 0 or cfg.get("gen_njobs", 1) != 1:
        raise RuntimeContractError("RFD4 row requires original job_id=0 and gen_njobs=1")
    return job_id


def execute_row(
    bundle,
    row,
    fingerprint,
    *,
    native=None,
    compose_fn=None,
    environment_class=None,
    callback_class=None,
    model_cache=None,
):
    output_dir = Path(row["output_dir"]).resolve()
    key = identity({"output_dir": str(output_dir)})
    sidecar = output_dir.parent / ".hhq-rfd4" / key
    row_fingerprint = identity(
        {
            "row": row,
            "repo": bundle["repo"],
            "runtime": bundle["runtime"],
            "source_hashes": bundle["source_hashes"],
            "reuse_model": bool(bundle.get("reuse_model", False)),
        }
    )
    with exclusive_lock(sidecar / "owner.lock"):
        owner_path = sidecar / "owner.json"
        if owner_path.exists():
            if read_json(owner_path).get("row_fingerprint") != row_fingerprint:
                raise RuntimeContractError(f"Output directory belongs to another immutable row: {output_dir}")
        else:
            if output_dir.exists() and any(output_dir.iterdir()):
                raise RuntimeContractError(
                    f"Refusing to adopt nonempty unrelated output directory: {output_dir}"
                )
            atomic_json(
                owner_path,
                {
                    "schema": "hhq-rfd4-owner/1",
                    "row_fingerprint": row_fingerprint,
                    "output_dir": str(output_dir),
                    "created_at": time.time(),
                },
            )
        receipt_path = sidecar / "completion.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt.get("row_fingerprint") != row_fingerprint or receipt.get("status") != "complete":
                raise RuntimeContractError("Invalid prior row completion identity")
            validate_artifacts(output_dir, receipt["artifacts"])
            return {"path": str(receipt_path), "sha256": digest(receipt_path), "reused": True}

        started = time.time()
        attempt = os.environ.get("HHQ_ATTEMPT_ID", f"local-{os.getpid()}-{time.time_ns()}")
        failure_path = sidecar / "failures" / f"{identity(attempt)[:24]}.json"
        interrupted_outputs = None
        if output_dir.exists() and any(output_dir.iterdir()):
            # Ownership is proven above, but no completion receipt exists. Native
            # skip_existing checks filenames/size, not atomic completion. Preserve
            # all partial evidence and restart this row in an empty directory.
            archive_root = ensure_directory(sidecar / "interrupted_outputs")
            if sum(1 for _ in archive_root.iterdir()) >= 9000:
                raise RuntimeContractError("Interrupted-output archive exceeds the directory entry limit")
            interrupted_outputs = archive_root / f"attempt_{time.time_ns()}_{identity(attempt)[:12]}"
            output_dir.rename(interrupted_outputs)
            atomic_json(sidecar / "last_interrupted_output.json", {
                "schema": "hhq-rfd4-interrupted-output/1", "row_fingerprint": row_fingerprint,
                "path": str(interrupted_outputs), "retry_attempt_id": attempt,
                "preserved_at": time.time(), "reason": "owned row has no completion receipt",
            })
            print(f"[rfd4] Preserved unfinished row outputs: {interrupted_outputs}", file=sys.stderr, flush=True)
        try:
            environment = environment_for_row(bundle, row)
            with row_environment(environment, row.get("unset_environment", [])):
                if native is None:
                    native, environment_class, callback_class = load_native(bundle, row)
                cfg, omega = (compose_fn or compose_native)(row)
                cfg = copy.deepcopy(cfg)  # Native run_generation mutates cfg.seed.
                job_id = validate_config(cfg, row)
                check_gpu_transport(cfg)
                if cfg.get("data_path") is not None:
                    os.environ["DATA_PATH"] = str(cfg.data_path)
                composed = omega.to_container(cfg, resolve=True, throw_on_missing=False)
                model_key = reuse_key(cfg) if model_cache is not None else None
                coverage = Coverage()
                with native_hooks(
                    native,
                    coverage,
                    environment_class=environment_class,
                    callback_class=callback_class,
                    model_cache=model_cache,
                    model_key=model_key,
                ):
                    try:
                        native.run_generation(cfg, job_id=job_id, config_name=Path(row["config"]).stem)
                    except SystemExit as error:
                        raise RuntimeContractError(
                            f"Native generation exited early ({error.code}); normal completion is required"
                        ) from error
                complete_coverage = coverage.finish()
                artifacts = artifact_manifest(output_dir)
                for filename, expected in bundle["source_hashes"].items():
                    if digest(filename) != expected:
                        raise RuntimeContractError(f"Pinned source changed during generation: {filename}")
                receipt = {
                    "schema": "hhq-rfd4-row-completion/1",
                    "status": "complete",
                    "bundle_sha256": fingerprint,
                    "row_fingerprint": row_fingerprint,
                    "run_id": row["run_id"],
                    "config_sha256": row["config_sha256"],
                    "input_sha256": row["input_sha256"],
                    "composed_config_sha256": identity(composed),
                    "base_seed": row["seed"],
                    "job_id": job_id,
                    "effective_seed": row["seed"] + job_id,
                    "coverage": complete_coverage,
                    "artifacts": artifacts,
                    "artifact_count": len(artifacts),
                    "output_dir": str(output_dir),
                    "started_at": started,
                    "finished_at": time.time(),
                    "attempt_id": attempt,
                    "model_reuse": model_cache is not None,
                    "native_skip_existing": bool(cfg.get("skip_existing", False)),
                    "resume_semantics": "completed rows reused; unfinished rows restart; not bitwise RNG continuation",
                    "preserved_interrupted_outputs": str(interrupted_outputs) if interrupted_outputs else None,
                }
                atomic_json(receipt_path, receipt)
                return {"path": str(receipt_path), "sha256": digest(receipt_path), "reused": False}
        except BaseException as error:
            atomic_json(
                failure_path,
                {
                    "schema": "hhq-rfd4-row-failure/1",
                    "status": "failed",
                    "row_fingerprint": row_fingerprint,
                    "bundle_sha256": fingerprint,
                    "run_id": row["run_id"],
                    "started_at": started,
                    "finished_at": time.time(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "attempt_id": attempt,
                },
            )
            raise


def run_bundle(bundle_path, fingerprint):
    """Give direct-host execution the same task-local temp ownership as AF3 mode."""
    bundle = read_json(bundle_path)
    if bundle["runtime"].get("host_af3"):
        return _run_bundle(bundle_path, fingerprint)
    old = os.environ.get("HHQ_RFD4_TMPDIR")
    if old:
        raise RuntimeContractError("Refusing inherited RFD4 temporary-directory ownership")
    try:
        with tempfile.TemporaryDirectory(prefix="hhq-rfd4-", dir="/tmp") as directory:
            os.environ["HHQ_RFD4_TMPDIR"] = directory
            return _run_bundle(bundle_path, fingerprint)
    finally:
        os.environ.pop("HHQ_RFD4_TMPDIR", None)


def _run_bundle(bundle_path, fingerprint):
    bundle_path = Path(bundle_path).resolve()
    bundle = read_json(bundle_path)
    check_pins(bundle, bundle_path, fingerprint)
    validate_host_daemon(bundle, fingerprint)
    task_output = Path(os.environ["HHQ_TASK_OUTPUT_DIR"])
    ensure_directory(task_output)
    completion = task_output / "rfd4_completion.json"
    old_cwd, old_path = Path.cwd(), list(sys.path)
    original_handlers = {}

    def interrupted(signum, frame):
        raise InterruptedError(f"RFD4 task interrupted by signal {signum}")

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1):
        original_handlers[number] = signal.signal(number, interrupted)
    try:
        os.chdir(bundle["repo"])
        sys.path.insert(0, str(Path(bundle["repo"]) / "src"))
        with exclusive_lock(bundle_path.parent / ".runtime-locks" / f"{fingerprint}.lock"):
            completion.unlink(missing_ok=True)
            model_cache = {} if bundle.get("reuse_model", False) else None
            results = [
                execute_row(bundle, row, fingerprint, model_cache=model_cache) for row in bundle["rows"]
            ]
            # Detect source/config/input edits made while computation was in flight.
            check_pins(bundle, bundle_path, fingerprint)
            for result in results:
                if digest(result["path"]) != result["sha256"]:
                    raise RuntimeContractError("Row completion changed before bundle publication")
            atomic_json(
                completion,
                {
                    "schema": "hhq-rfd4-completion/1",
                    "status": "complete",
                    "bundle_id": bundle["bundle_id"],
                    "bundle_sha256": fingerprint,
                    "row_count": len(results),
                    "rows": results,
                    "finished_at": time.time(),
                },
            )
        return completion
    finally:
        for number, handler in original_handlers.items():
            signal.signal(number, handler)
        os.chdir(old_cwd)
        sys.path[:] = old_path


def validate_host_daemon(bundle, fingerprint):
    """Accept only the task wrapper's pinned, live, explicitly opted-in daemon."""
    template = os.environ.get("AF3_LAUNCHER_SOCKET_TEMPLATE")
    context_path = os.environ.get("HHQ_RFD4_HOST_CONTEXT")
    if not template and not context_path and not os.environ.get("AF3_SOCKET_DIR"):
        if bundle["runtime"].get("host_af3"):
            raise RuntimeContractError("Owned host AF3 must run through its host wrapper")
        return
    if (not template or not context_path or os.environ.get("AF3_SOCKET_DIR")
            or not bundle["runtime"].get("host_af3")):
        raise RuntimeContractError("Refusing inherited AF3 daemon ownership")
    context = read_json(context_path)
    if (context.get("schema") != "hhq-owned-af3/1" or context.get("bundle_sha256") != fingerprint
            or context.get("socket_template") != template):
        raise RuntimeContractError("Host AF3 ownership does not match this bundle")
    owned_tmp = context.get("temporary_directory")
    if not owned_tmp or os.environ.get("HHQ_RFD4_TMPDIR") != owned_tmp:
        raise RuntimeContractError("Host AF3 temporary-directory ownership does not match this bundle")
    for key in ("pid", "daemon_pid"):
        pid = context.get(key)
        if type(pid) is not int or pid <= 1:
            raise RuntimeContractError("Invalid host AF3 owner process")
        os.kill(pid, 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument(
        "--check", action="store_true", help="CPU-only pinned composition/interface preflight"
    )
    args = parser.parse_args(argv)
    try:
        if args.check:
            print(json.dumps(preflight(args.bundle, args.fingerprint), indent=2, sort_keys=True))
        else:
            run_bundle(args.bundle, args.fingerprint)
        return 0
    except BaseException as error:  # noqa: BLE001 -- executable boundary records interruptions and exits nonzero
        output = os.environ.get("HHQ_TASK_OUTPUT_DIR")
        if output and not args.check:
            atomic_json(
                Path(output) / "rfd4_failure.json",
                {
                    "schema": "hhq-rfd4-failure/1",
                    "status": "failed",
                    "bundle": args.bundle,
                    "bundle_sha256": args.fingerprint,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "finished_at": time.time(),
                },
            )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
