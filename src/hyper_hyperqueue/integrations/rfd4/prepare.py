"""Read a saved native sweep and package its calls without changing the science.

Only the native notebook helper's emitted command grammar is accepted. This
module never imports RFD4, composes Hydra, loads models, or submits work.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shlex
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path

import yaml

from ...models import Resources, RetryPolicy
from ...state import atomic_bytes, ensure_directory, file_lock, fingerprint, json_bytes
from .runtime import identity

RUNTIME_SCRIPT = Path(__file__).with_name("runtime.py")
HOST_SCRIPT = Path(__file__).with_name("host_runtime.py")
_METHODS = {"generate", "gradient", "beam"}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_PROTECTED_PREFIXES = ("SLURM_", "HHQ_", "CUDA_", "NVIDIA_", "ROCR_", "HIP_", "HQ_", "AF3_LAUNCHER_")
_PROTECTED = {
    "AF3_DAEMON_SOCKET",
    "AF2_DAEMON_SOCKET",
    "AF3_SOCKET",
    "AF2_SOCKET",
    "AF3_SOCKET_DIR",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
}


def _safe_yaml_load(text):
    """Use LibYAML's safe parser when installed; retain the pure-Python fallback."""
    return yaml.load(text, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


def _path(value) -> Path:
    result = Path(value).expanduser()
    if not result.is_absolute():
        raise ValueError(f"Explicit absolute path required: {value}")
    return result.resolve()


def _sha(path) -> str:
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise ValueError(f"Source changed while hashing: {path}")
    return digest.hexdigest()


def _bind(bindings: dict, path, digest: str) -> None:
    path = str(_path(path))
    if not isinstance(digest, str) or not _SHA.fullmatch(digest):
        raise ValueError(f"Invalid SHA256 for {path}")
    if path in bindings and bindings[path] != digest:
        raise ValueError(f"Conflicting source hashes for {path}")
    bindings[path] = digest


def _verify(bindings: dict) -> None:
    for path, expected in bindings.items():
        if _sha(path) != expected:
            raise ValueError(f"SHA256 mismatch: {path}")


def _environment(values: dict) -> dict:
    if not isinstance(values, dict):
        raise TypeError("Environment must be a mapping of strings")
    result = {}
    for name, value in values.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name) or not isinstance(value, str):
            raise ValueError("Environment names and values must be valid strings")
        effective_name = name.removeprefix("APPTAINERENV_").removeprefix("SINGULARITYENV_")
        if (effective_name.startswith(_PROTECTED_PREFIXES) or effective_name in _PROTECTED
                or "DAEMON_SOCKET" in effective_name):
            raise ValueError(f"Cannot override allocated resources or daemon ownership: {name}")
        if "\x00" in value:
            raise ValueError(f"NUL in environment value: {name}")
        result[name] = value
    return result


def _runtime(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) - {"argv", "environment", "source_hashes", "host_af3"}:
        raise ValueError("Runtime accepts argv, environment, source_hashes, and optional host_af3")
    result = deepcopy(value)
    argv = result.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(v, str) or not v for v in argv):
        raise ValueError("runtime.argv must be an explicit nonempty argument vector")
    executable = _path(argv[0])
    if not executable.is_file():
        raise ValueError(f"Runtime executable does not exist: {executable}")
    python_name = re.compile(r"python(?:[0-9]+(?:\.[0-9]+)*)?\Z")
    if python_name.fullmatch(Path(argv[0]).name):
        if len(argv) != 1:
            raise ValueError("Native Python runtime argv must contain only its explicit executable")
    elif Path(argv[0]).name == "apptainer":
        if len(argv) < 4 or argv[1] != "exec" or not python_name.fullmatch(Path(argv[-1]).name):
            raise ValueError("Container runtime must be explicit apptainer exec ... python")
        _path(argv[-1])
        forbidden = {"--cleanenv", "-e", "--env", "--env-file", "--containall", "-C"}
        if any(arg.split("=", 1)[0] in forbidden for arg in argv[2:-1]):
            raise ValueError("Container runtime must preserve the allocated environment")
        for arg in argv[2:-1]:
            if "=" in arg:
                name, val = arg.split("=", 1)
                if _NAME.fullmatch(name):
                    _environment({name: val})
    else:
        raise ValueError("Use an explicit Python or apptainer exec runtime; no scheduler or native launcher")
    if any("\x00" in arg or "\n" in arg for arg in argv):
        raise ValueError("Invalid runtime argv")
    result["environment"] = _environment(result.get("environment", {}))
    extras = {}
    for path, expected in result.get("source_hashes", {}).items():
        _bind(extras, path, expected)
    result["source_hashes"] = extras
    if "host_af3" in result:
        host = result["host_af3"]
        if not isinstance(host, dict) or "python" not in host or set(host) - {"python", "adapter"}:
            raise ValueError("host_af3 requires host python and an optional qualified adapter")
        executable = _path(host["python"])
        if not executable.is_file() or not python_name.fullmatch(Path(host["python"]).name):
            raise ValueError("host_af3.python must be an existing host Python executable")
        result["host_af3"] = {"python": str(host["python"])}
        if "adapter" in host:
            adapter = host["adapter"]
            required = {"daemon_script", "batch_script", "environment", "source_hashes"}
            optional = {"bind_paths", "log_root", "config", "config_sha256"}
            if (not isinstance(adapter, dict) or not required <= set(adapter)
                    or set(adapter) - required - optional):
                raise ValueError("Host adapter requires daemon_script, batch_script, environment and source_hashes")
            adapter = deepcopy(adapter)
            adapter["environment"] = _environment(adapter["environment"])
            pins = {}
            for path, expected in adapter["source_hashes"].items():
                _bind(pins, path, expected)
            for name in ("daemon_script", "batch_script"):
                adapter[name] = str(_path(adapter[name]))
                if adapter[name] not in pins:
                    raise ValueError(f"Host adapter {name} is not source-pinned")
            for path in adapter.get("bind_paths", []):
                _path(path)
            if adapter.get("config") and pins.get(str(_path(adapter["config"]))) != adapter.get("config_sha256"):
                raise ValueError("Host adapter config is not source-pinned")
            _verify(pins)
            adapter["source_hashes"] = pins
            result["host_af3"]["adapter"] = adapter
            for path, expected in pins.items():
                _bind(result["source_hashes"], path, expected)
    return result


def execution_argv(bundle, bundle_path, digest):
    """Keep the optional host daemon inside the generic runner's process group."""
    host = bundle["runtime"].get("host_af3")
    entry = (
        [host["python"], bundle["host_script"]]
        if host else [*bundle["runtime"]["argv"], bundle["runtime_script"]]
    )
    return [*entry, "--bundle", str(bundle_path), "--fingerprint", digest]


def _command(command: str, *, repo: Path, config: Path) -> tuple[dict, list[str]]:
    try:
        tokens = shlex.split(command)
    except (ValueError, TypeError) as exc:
        raise ValueError("Malformed native command") from exc
    # This is the precise writer's grammar, not a shell interpreter. Requiring
    # its canonical serialization also rejects expansion/operators in raw shell.
    if shlex.join(tokens) != command or not tokens or tokens.pop(0) != "env":
        raise ValueError("Unsupported command; use the saved native inference_sweep writer")
    environment, unset = {}, []
    while tokens and tokens[0] == "-u":
        if len(tokens) < 2 or tokens[1] in unset:
            raise ValueError("Malformed or duplicate env -u")
        _environment({tokens[1]: ""})
        unset.append(tokens[1])
        del tokens[:2]
    while tokens and "=" in tokens[0]:
        name, value = tokens.pop(0).split("=", 1)
        if name in environment:
            raise ValueError(f"Duplicate environment setting: {name}")
        environment.update(_environment({name: value}))
    expected = [
        str(repo / "src/rfproteina/generate.py"),
        "--config-path",
        str(config.parent),
        "--config-name",
        config.stem,
    ]
    if len(tokens) != 5 or tokens[1] != expected[1] or tokens[3:] != expected[3:]:
        raise ValueError("Only a single native generate.py --config-path --config-name call is supported")
    if _path(tokens[0]) != _path(expected[0]) or _path(tokens[2]) != config.parent:
        raise ValueError("Native command repository/config does not match the saved row")
    return environment, unset


def _workload(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) - {"profile", "resources", "runs_per_task", "max_attempts", "priority", "retry"}:
        raise ValueError("Unknown workload setting")
    if not isinstance(value.get("profile"), str) or not value["profile"].strip():
        raise ValueError("Each workload requires an explicit profile")
    raw = value.get("resources", {})
    required = {"cpus", "memory_mib", "gpus", "runtime_seconds", "runtime_confidence"}
    if not isinstance(raw, dict) or not required <= raw.keys():
        raise ValueError("Explicit CPU, RAM, GPU and calibrated runtime resources are required")
    resources = Resources.model_validate(raw)
    if (resources.gpus, resources.nodes, resources.ranks) != (1, 1, 1):
        raise ValueError("RFD4 adapter requires exactly one GPU, node and rank")
    if resources.runtime_confidence == "unknown" or resources.runtime_seconds is None:
        raise ValueError("Supply measured or explicitly declared bundle runtime")
    if not math.isfinite(resources.runtime_seconds):
        raise ValueError("Runtime must be finite")
    result = {"profile": value["profile"], "resources": resources.model_dump()}
    for key, upper in (("runs_per_task", 100000), ("max_attempts", 100)):
        number = value.get(key, 1)
        if type(number) is not int or not 1 <= number <= upper:
            raise ValueError(f"{key} must be a positive integer <= {upper}")
        result[key] = number
    if "retry" in value:
        if not isinstance(value["retry"], dict):
            raise ValueError("Workload retry must be a mapping")
        retry = RetryPolicy.model_validate({"max_attempts": result["max_attempts"], **value["retry"]})
        if "max_attempts" in value and retry.max_attempts != value["max_attempts"]:
            raise ValueError("Workload max_attempts conflicts with retry.max_attempts")
        result["max_attempts"] = retry.max_attempts
        result["retry"] = retry.model_dump()
    if "priority" in value:
        priority = value["priority"]
        if type(priority) is not int or not -(2**31) <= priority < 2**31:
            raise ValueError("Workload priority must be a signed 32-bit integer")
        result["priority"] = priority
    return result


def _native_sources(repo: Path, expected_commit: str | None) -> tuple[str, dict]:
    def git(*args):
        return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout

    commit = git("rev-parse", "HEAD").decode().strip()
    if expected_commit is not None and commit != expected_commit:
        raise ValueError("Native repository commit differs from the saved sweep")
    sources = {}
    for raw in git("ls-files", "-z", "--", "src", "configs").split(b"\x00"):
        if not raw:
            continue
        path = repo / raw.decode()
        if path.suffix in {".py", ".yaml", ".yml"}:
            _bind(sources, path, _sha(path))
    generate = repo / "src/rfproteina/generate.py"
    _bind(sources, generate, _sha(generate))
    return commit, sources


def _dotenv(repo: Path, environment: dict) -> dict:
    cluster = environment.get("CLUSTER")
    if not isinstance(cluster, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", cluster):
        raise ValueError(
            "An explicit simple CLUSTER name is required; inherited runtime profiles are not pinned"
        )
    candidates = (
        [repo / ".env.local", repo / ".env"]
        if cluster == "local"
        else [repo / ".ipd" / f".env.{cluster}", repo / f".env.{cluster}"]
    )
    selected = next((path for path in candidates if path.exists()), None)
    if selected is None and cluster != "local":
        raise ValueError(f"No native dotenv profile exists for CLUSTER={cluster}")
    return {
        "cluster": cluster,
        "path": str(selected.resolve()) if selected else None,
        "sha256": _sha(selected) if selected else None,
    }


def recipe_identity(document: dict, method: str, environment: dict, unset: list[str]) -> str:
    """Conservative identity: retain every scientific setting and explicit environment.

    The four per-row bindings and their exact generated Hydra output path may
    differ. Native interpolations remain intact; references are never rewritten.
    """
    config = deepcopy({key: value for key, value in document.items() if key not in {"inputs", "seed", "out_dir", "run_name_"}})
    hydra = config.get("hydra")
    if (isinstance(hydra, dict) and isinstance(hydra.get("run"), dict)
            and isinstance(document.get("out_dir"), str)
            and hydra["run"].get("dir") == str(Path(document["out_dir"]) / "hydra")):
        hydra["run"]["dir"] = "${out_dir}/hydra"
    return fingerprint({"method": method, "config": config, "environment": environment, "unset": unset})


def plan_sweep(manifest_path, *, runtime: dict, workloads: dict, reuse_model=False,
               group_by_recipe=False, execution_order=None) -> dict:
    """Verify a native saved sweep; return an immutable plan without writing files.

    Resource runtimes describe one *bundle*, not one scientific call. External
    model weights/dependencies/custom modules need additional operator pins.
    """
    path = _path(manifest_path)
    original_sha = _sha(path)
    source = json.loads(path.read_text())
    if source.get("schema_version") != 1 or source.get("status") != "prepared_not_submitted":
        raise ValueError("Expected a prepared schema-1 inference_sweep manifest")
    if type(reuse_model) is not bool:
        raise ValueError("reuse_model must be boolean")
    if type(group_by_recipe) is not bool:
        raise ValueError("group_by_recipe must be boolean")
    if execution_order is not None and (
        not isinstance(execution_order, list) or not execution_order
        or any(not isinstance(method, str) or method not in _METHODS for method in execution_order)
        or len(set(execution_order)) != len(execution_order)
    ):
        raise ValueError("execution_order must be a nonempty unique list of generate, gradient and/or beam")
    runtime = _runtime(runtime)
    if not isinstance(workloads, dict) or set(workloads) - _METHODS:
        raise ValueError("Workloads must be keyed by generate, gradient or beam")
    workload_specs = {method: _workload(spec) for method, spec in workloads.items()}
    repo = _path(source["repo"])
    bindings = {str(path): original_sha}
    required_maps = ("input_sha256", "file_sha256", "native_config_sha256")
    normalized_maps = {}
    for name in required_maps:
        if not isinstance(source.get(name), dict) or not source[name]:
            raise ValueError(f"Saved sweep is missing {name}")
        normalized_maps[name] = {}
        for name_path, expected in source[name].items():
            _bind(bindings, name_path, expected)
            _bind(normalized_maps[name], name_path, expected)
    _verify(bindings)
    commit, native = _native_sources(repo, source.get("repo_commit"))
    for mapping in (native, runtime["source_hashes"]):
        for name_path, expected in mapping.items():
            _bind(bindings, name_path, expected)
    runtime_path = _path(RUNTIME_SCRIPT)
    _bind(bindings, runtime_path, _sha(runtime_path))
    software_sources = dict(native)
    for source_path, digest in runtime["source_hashes"].items():
        _bind(software_sources, source_path, digest)
    _bind(software_sources, runtime_path, bindings[str(runtime_path)])
    if runtime.get("host_af3"):
        _bind(bindings, HOST_SCRIPT, _sha(HOST_SCRIPT))
        _bind(software_sources, HOST_SCRIPT, bindings[str(_path(HOST_SCRIPT))])
    rows = source.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Saved sweep has no rows")
    commands_path = _path(source["commands_file"])
    if str(commands_path) not in normalized_maps["file_sha256"]:
        raise ValueError("Saved commands file is not hash-bound")
    if commands_path.read_text() != "\n".join(row["command"] for row in rows) + "\n":
        raise ValueError("Saved command file and ordered rows disagree")
    methods = {row["method"] for row in rows}
    if not methods <= set(workload_specs):
        raise ValueError(f"Missing explicit resource workload for methods: {sorted(methods - set(workload_specs))}")
    if execution_order is not None:
        if not methods <= set(execution_order):
            raise ValueError("execution_order must include every method in the saved sweep")
        for method in methods:
            workload_specs[method].setdefault("priority", len(execution_order) - execution_order.index(method))
    groups, run_ids, outputs, recipes = {}, set(), set(), {}
    for raw in rows:
        row = deepcopy(raw)
        method = row["method"]
        if method not in workload_specs:
            raise ValueError(f"Missing explicit resource workload for method: {method}")
        config, ligand = _path(row["config"]), _path(row["input"])
        if str(config) not in normalized_maps["file_sha256"]:
            raise ValueError(f"Row config is not hash-bound: {config}")
        if str(ligand) not in normalized_maps["input_sha256"]:
            raise ValueError(f"Row input is not hash-bound: {ligand}")
        document = _safe_yaml_load(config.read_text())
        if not isinstance(document, dict):
            raise TypeError("Config must be a YAML mapping")
        inputs = document.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != 1 or _path(inputs[0]) != ligand:
            raise ValueError("Exactly one matching input per native config is required")
        output = _path(row["output_dir"])
        if (
            document.get("seed") != row["seed"]
            or type(row["seed"]) is not int
            or document.get("num_replicates") != row["replicates"]
            or type(row["replicates"]) is not int
            or row["replicates"] < 1
            or _path(document["out_dir"]) != output
            or document.get("run_name_") != row["run_id"]
            or config.stem != row["run_id"]
            or document.get("gen_njobs") != 1
            or document.get("job_id") != 0
        ):
            raise ValueError("Saved row and native config disagree on seed, output, identity or sharding")
        if bool(document.get("search")) != (method == "beam") or bool(document.get("guidance")) != (
            method == "gradient"
        ):
            raise ValueError("Saved method disagrees with native search/guidance config")
        resume_from = (document.get("search") or {}).get("resume_from")
        if resume_from is not None:
            if not isinstance(resume_from, str) or "${" in resume_from or not Path(resume_from).is_absolute():
                raise ValueError("Explicit beam resume_from requires a literal absolute state path")
            state_path = str(_path(resume_from))
            if state_path not in runtime["source_hashes"]:
                raise ValueError("Explicit beam continuation state must be pinned in runtime.source_hashes")
        if row["run_id"] in run_ids or str(output) in outputs:
            raise ValueError("Repeated run identity or scientific output directory")
        run_ids.add(row["run_id"])
        outputs.add(str(output))
        environment, unset = _command(row["command"], repo=repo, config=config)
        for key, value in runtime["environment"].items():
            if key in unset:
                raise ValueError(f"Runtime environment conflicts with native unset: {key}")
            if key in environment and environment[key] != value:
                raise ValueError(f"Runtime environment conflicts with native command: {key}")
        dotenv = _dotenv(repo, {**runtime["environment"], **environment})
        if dotenv["path"] is not None:
            _bind(bindings, dotenv["path"], dotenv["sha256"])
            _bind(software_sources, dotenv["path"], dotenv["sha256"])
        row.update(
            dotenv=dotenv,
            job_id=0,
            config_path=str(Path(row["config"]).parent),
            config_name=config.stem,
            config_sha256=bindings[str(config)],
            input_sha256=bindings[str(ligand)],
            environment=environment,
            unset_environment=unset,
        )
        recipe = recipe_identity(document, method, environment, unset)
        key = (method, recipe if group_by_recipe else fingerprint({"environment": environment, "unset": unset}))
        groups.setdefault(key, []).append(row)
        recipes.setdefault(recipe, {"method": method, "recipe_id": recipe, "scientific_calls": 0})["scientific_calls"] += 1
    _verify(bindings)
    software_fingerprint = fingerprint({"commit": commit, "sources": software_sources, "runtime": runtime})
    bundles = []
    grouped = list(groups.items())
    if group_by_recipe or execution_order is not None:
        order = execution_order or list(dict.fromkeys(row["method"] for row in rows))

        def group_key(item):
            (method, identity), values = item
            first = values[0]
            recipe_order = tuple(str(first.get(field, "")) for field in ("variant", "thinking", "bb_noise", "latent_noise"))
            return (order.index(method), recipe_order, identity) if group_by_recipe else (order.index(method),)

        grouped.sort(key=group_key)
    for (method, recipe), group in grouped:
        if group_by_recipe:
            group = sorted(group, key=lambda row: (row["input"], row["run_id"]))
        workload = workload_specs[method]
        for start in range(0, len(group), workload["runs_per_task"]):
            chosen_rows = group[start : start + workload["runs_per_task"]]
            needed = {
                *software_sources,
                *normalized_maps["native_config_sha256"],
                str(path),
            }
            # Workers verify only their own scientific rows plus common code.
            # Whole-sweep declarations are verified once by preparation below.
            needed.update(str(_path(row[key])) for row in chosen_rows for key in ("config", "input"))
            bundle = {
                "schema": "hhq-rfd4-bundle/1",
                "repo": str(repo),
                "repo_commit": commit,
                "source_manifest": {"path": str(path), "sha256": original_sha},
                "source_hashes": {name: bindings[name] for name in sorted(needed)},
                "runtime": runtime,
                "software_fingerprint": software_fingerprint,
                "runtime_script": str(runtime_path),
                "reuse_model": reuse_model,
                "method": method,
                "workload": workload,
                "rows": chosen_rows,
            }
            if runtime.get("host_af3"):
                bundle["host_script"] = str(_path(HOST_SCRIPT))
            if group_by_recipe:
                bundle["recipe_id"] = recipe
            bundle["bundle_id"] = "b_" + fingerprint(bundle)[:24]
            bundles.append(bundle)
    warnings = list(source.get("warnings", [])) + [
        "External dependencies, model weights and custom PYTHONPATH modules are not frozen unless explicitly pinned in runtime.source_hashes.",
        "Resource runtime_seconds is an explicit estimate for a whole bundle; no native campaign timings are assumed.",
        "Native output paths remain unchanged; use a new source sweep/output directory for different scientific runs.",
    ]
    if reuse_model:
        warnings.append("Cross-config model reuse is opt-in and requires workload-specific qualification.")
    if execution_order is not None:
        warnings.append("execution_order is a soft HQ task preference, not a completion barrier or Slurm scheduling priority.")
    plan = {
        "schema": "hhq-rfd4-plan/1",
        "software_fingerprint": software_fingerprint,
        "preparation_source_hashes": dict(sorted(bindings.items())),
        "source_manifest": {"path": str(path), "sha256": original_sha},
        "bundles": bundles,
        "summary": {
            "scientific_calls": len(rows),
            "tasks": len(bundles),
            "calls_by_method": dict(Counter(row["method"] for row in rows)),
            "tasks_by_method": dict(Counter(bundle["method"] for bundle in bundles)),
            "native_counts": deepcopy(source.get("counts", {})),
        },
        "warnings": warnings,
    }
    if group_by_recipe or execution_order is not None:
        plan["execution"] = {"group_by_recipe": group_by_recipe, "execution_order": execution_order}
        plan["summary"].update(
            recipes=len(recipes), recipes_by_method=dict(Counter(value["method"] for value in recipes.values())),
            recipe_groups=[recipes[key] for key in dict.fromkeys(bundle["recipe_id"] for bundle in bundles)]
            if group_by_recipe else list(recipes.values()),
            execution_order=list(dict.fromkeys(bundle["method"] for bundle in bundles)),
        )
    plan["plan_id"] = "p_" + fingerprint(plan)[:24]
    return plan


def _check_bundle_sources(bundle: dict) -> None:
    repo = _path(bundle["repo"])
    pins = bundle["source_hashes"]
    if bundle["runtime_script"] not in pins:
        raise ValueError("Runtime entry point is not source-pinned")
    if bundle["runtime"].get("host_af3") and bundle.get("host_script") not in pins:
        raise ValueError("Host AF3 entry point is not source-pinned")
    manifest = bundle["source_manifest"]
    if pins.get(manifest["path"]) != manifest["sha256"]:
        raise ValueError("Source manifest is not source-pinned")
    for row in bundle["rows"]:
        for name in ("config", "input"):
            if pins.get(str(_path(row[name]))) != row[f"{name}_sha256"]:
                raise ValueError(f"Row {name} is not source-pinned")
        effective = {**bundle["runtime"]["environment"], **row["environment"]}
        if _dotenv(repo, effective) != row["dotenv"]:
            raise ValueError("Native dotenv profile selection changed since preparation")
        if "recipe_id" in bundle:
            document = _safe_yaml_load(Path(row["config"]).read_text())
            if recipe_identity(document, bundle["method"], row["environment"], row["unset_environment"]) != bundle["recipe_id"]:
                raise ValueError("Bundle mixes incompatible scientific recipes")


def _validate_plan(plan: dict) -> None:
    if plan.get("schema") != "hhq-rfd4-plan/1":
        raise ValueError("Unsupported RFD4 plan")
    content = {key: value for key, value in plan.items() if key != "plan_id"}
    if plan.get("plan_id") != "p_" + fingerprint(content)[:24]:
        raise ValueError("Plan contents changed after planning")
    sources = {}
    for path, expected in plan.get("preparation_source_hashes", {}).items():
        _bind(sources, path, expected)
    for bundle in plan["bundles"]:
        _check_bundle_sources(bundle)
        content = {key: value for key, value in bundle.items() if key != "bundle_id"}
        if bundle["bundle_id"] != "b_" + fingerprint(content)[:24]:
            raise ValueError("Bundle contents changed after planning")
        for path, expected in bundle["source_hashes"].items():
            _bind(sources, path, expected)
    _verify(sources)


def _prepare_output_sidecars(bundles):
    """Create shared parents centrally, without claiming or opening scientific outputs."""
    directories = set()
    for bundle in bundles:
        for row in bundle["rows"]:
            output = _path(row["output_dir"])
            sidecars = output.parent / ".hhq-rfd4"
            directories.update((output.parent, sidecars, sidecars / identity({"output_dir": str(output)})))
    # Establish every shared parent before its children. Workers otherwise race
    # recursive mkdir across NFS clients with independently cached missing paths.
    for path in sorted(directories, key=lambda value: (len(value.parts), str(value))):
        ensure_directory(path)


def prepare_sweep(plan: dict, directory) -> dict:
    """Write pinned bundles and empty output sidecars; never import or submit tasks."""
    _validate_plan(plan)
    directory = _path(directory)
    payloads, bundle_files, grouped = {}, {}, {}
    for bundle in plan["bundles"]:
        path = directory / "bundles" / f"{bundle['bundle_id']}.json"
        data = json_bytes(bundle)
        digest = hashlib.sha256(data).hexdigest()
        payloads[str(path)] = data
        bundle_files[bundle["bundle_id"]] = {"path": str(path), "sha256": digest}
        grouped.setdefault(bundle["method"], []).append(bundle)
    groups = []
    for method, bundles in grouped.items():
        command_file = directory / "commands" / f"{method}.jsonl"
        commands = [
            {
                "argv": execution_argv(bundle, bundle_files[bundle["bundle_id"]]["path"],
                                       bundle_files[bundle["bundle_id"]]["sha256"])
            }
            for bundle in bundles
        ]
        payloads[str(command_file)] = (
            "\n".join(json.dumps(row, sort_keys=True) for row in commands) + "\n"
        ).encode()
        spec = bundles[0]["workload"]
        groups.append(
            {
                "method": method,
                "commands_file": str(command_file),
                "profile": spec["profile"],
                "resources": spec["resources"],
                "max_attempts": spec["max_attempts"],
                "environment": dict(bundles[0]["runtime"]["environment"]),
                "bundle_ids": [bundle["bundle_id"] for bundle in bundles],
            }
        )
        if "priority" in spec:
            groups[-1]["priority"] = spec["priority"]
        if "retry" in spec:
            groups[-1]["retry"] = spec["retry"]
    manifest = directory / "prepared.json"
    hashes = {path: hashlib.sha256(data).hexdigest() for path, data in payloads.items()}
    for bundle in plan["bundles"]:
        _bind(hashes, bundle["runtime_script"], bundle["source_hashes"][bundle["runtime_script"]])
        if bundle.get("host_script"):
            _bind(hashes, bundle["host_script"], bundle["source_hashes"][bundle["host_script"]])
    prepared = {
        "schema": "hhq-rfd4-prepared/1",
        "plan_id": plan["plan_id"],
        "manifest_path": str(manifest),
        "source_manifest": plan["source_manifest"],
        "software_fingerprint": plan["software_fingerprint"],
        "preparation_source_hashes": plan.get("preparation_source_hashes", {}),
        "groups": groups,
        "bundle_files": bundle_files,
        "file_sha256": hashes,
        "summary": plan["summary"],
        "warnings": plan["warnings"],
    }
    payloads[str(manifest)] = json_bytes(prepared)
    with file_lock(directory / ".prepare.lock"):
        for path, data in payloads.items():
            existing = Path(path)
            if existing.exists() and (not existing.is_file() or existing.read_bytes() != data):
                raise FileExistsError(f"Preserving different existing file: {path}; choose a new directory")
        _prepare_output_sidecars(plan["bundles"])
        for path, data in payloads.items():
            if not Path(path).exists():
                atomic_bytes(path, data)
    return {**prepared, "manifest_sha256": hashlib.sha256(payloads[str(manifest)]).hexdigest()}


def verify_prepared(prepared: dict | str | Path, *, check_sources=True) -> dict:
    """Verify immutable package identity; optionally require live source equality.

    Historical status uses check_sources=False: saved bundles and commands are
    still checked, while later native/input/runtime changes do not hide results.
    Launch and preview always use the default full verification.
    """
    if type(check_sources) is not bool:
        raise ValueError("check_sources must be boolean")
    expected = prepared.get("manifest_sha256") if isinstance(prepared, dict) else None
    path = _path(prepared["manifest_path"] if isinstance(prepared, dict) else prepared)
    digest = _sha(path)
    if expected is not None and digest != expected:
        raise ValueError("Prepared manifest SHA256 mismatch")
    saved = json.loads(path.read_text())
    if saved.get("schema") != "hhq-rfd4-prepared/1" or _path(saved["manifest_path"]) != path:
        raise ValueError("Unsupported or relocated prepared manifest")
    if isinstance(prepared, dict) and {k: v for k, v in prepared.items() if k != "manifest_sha256"} != saved:
        raise ValueError("Prepared mapping differs from saved manifest")
    snapshot_paths = {entry["path"] for entry in saved["bundle_files"].values()}
    snapshot_paths.update(group["commands_file"] for group in saved["groups"])
    if not snapshot_paths <= saved["file_sha256"].keys():
        raise ValueError("Prepared payload is missing its snapshot hash")
    _verify({path: saved["file_sha256"][path] for path in snapshot_paths})
    sources, observed, runtime_paths = {}, set(), set()
    if check_sources:
        for source_path, source_digest in saved.get("preparation_source_hashes", {}).items():
            _bind(sources, source_path, source_digest)
    for group in saved["groups"]:
        commands = [json.loads(line) for line in Path(group["commands_file"]).read_text().splitlines()]
        if len(commands) != len(group["bundle_ids"]):
            raise ValueError("Prepared commands and bundles disagree")
        for identifier, command in zip(group["bundle_ids"], commands):
            if identifier in observed:
                raise ValueError("Bundle appears more than once")
            observed.add(identifier)
            entry = saved["bundle_files"][identifier]
            if saved["file_sha256"].get(entry["path"]) != entry["sha256"]:
                raise ValueError("Bundle digest is not bound by prepared manifest")
            bundle = json.loads(Path(entry["path"]).read_text())
            content = {key: value for key, value in bundle.items() if key != "bundle_id"}
            if (
                bundle.get("schema") != "hhq-rfd4-bundle/1"
                or bundle["bundle_id"] != identifier
                or identifier != "b_" + fingerprint(content)[:24]
                or bundle["method"] != group["method"]
            ):
                raise ValueError("Prepared bundle identity mismatch")
            runtime_path = bundle["runtime_script"]
            runtime_paths.add(runtime_path)
            runtime_sha = bundle["source_hashes"].get(runtime_path)
            if not runtime_sha or saved["file_sha256"].get(runtime_path) != runtime_sha:
                raise ValueError("Prepared runtime pin differs from bundle")
            if bundle.get("host_script"):
                host_path = bundle["host_script"]
                runtime_paths.add(host_path)
                if saved["file_sha256"].get(host_path) != bundle["source_hashes"].get(host_path):
                    raise ValueError("Prepared host runtime pin differs from bundle")
            if check_sources:
                _check_bundle_sources(bundle)
            if bundle["source_manifest"] != saved["source_manifest"]:
                raise ValueError("Prepared source manifest differs from bundle")
            if bundle["runtime"]["environment"] != group["environment"]:
                raise ValueError("Prepared startup environment differs from bundle")
            for key in ("profile", "resources", "max_attempts"):
                if bundle["workload"][key] != group[key]:
                    raise ValueError(f"Prepared workload mismatch: {key}")
            if bundle["workload"].get("priority", 0) != group.get("priority", 0):
                raise ValueError("Prepared workload mismatch: priority")
            if bundle["workload"].get("retry") != group.get("retry"):
                raise ValueError("Prepared workload mismatch: retry")
            argv = execution_argv(bundle, entry["path"], entry["sha256"])
            if command != {"argv": argv}:
                raise ValueError("Prepared execution argv differs from its bundle")
            if check_sources:
                for source_path, source_digest in bundle["source_hashes"].items():
                    _bind(sources, source_path, source_digest)
    if observed != set(saved["bundle_files"]):
        raise ValueError("Unreferenced bundle in prepared manifest")
    if set(saved["file_sha256"]) != snapshot_paths | runtime_paths:
        raise ValueError("Unexpected prepared file pin outside snapshots and runtime")
    if check_sources:
        _verify(sources)
    if _sha(path) != digest:
        raise ValueError("Prepared manifest changed during verification")
    return {**saved, "manifest_sha256": digest}
