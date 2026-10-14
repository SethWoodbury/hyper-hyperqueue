"""Own a native host AF3 daemon while the RFD4 process runs inside its container.

No detached processes, scheduler calls, or replacement AF3 implementation. The
generic HHQ runner owns this complete process group and interruption/replay.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _runtime(path):
    spec = importlib.util.spec_from_file_location("hhq_rfd4_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_environment(bundle, runtime):
    """Source the pinned native shell profile exactly as the native wrapper does."""
    row = bundle["rows"][0]
    environment = dict(os.environ)
    if any(environment.get(k) for k in ("AF3_LAUNCHER_SOCKET_TEMPLATE", "AF3_SOCKET_DIR",
                                        "HHQ_RFD4_HOST_CONTEXT", "HHQ_RFD4_TMPDIR")):
        raise runtime.RuntimeContractError("Refusing inherited AF3 daemon ownership")
    protected = {k: v for k, v in environment.items() if runtime.protected_environment(k)}
    environment.update(runtime.environment_for_row(bundle, row))
    for name in row.get("unset_environment", []):
        environment.pop(name, None)
    profile = row["dotenv"]["path"]
    if profile:
        result = subprocess.run(
            ["/bin/bash", "-c", 'set -ae; source "$1" >/dev/null; /usr/bin/env -0',
             "hhq-native-profile", profile],
            env=environment, check=True, stdout=subprocess.PIPE,
        )
        environment = dict(item.decode().split("=", 1) for item in result.stdout.split(b"\0") if item)
    after = {k: v for k, v in environment.items() if runtime.protected_environment(k)}
    if protected != after:
        raise runtime.RuntimeContractError("Native host profile changed protected allocation variables")
    adapter = bundle["runtime"].get("host_af3", {}).get("adapter", {})
    for name, value in adapter.get("environment", {}).items():
        if runtime.protected_environment(name):
            raise runtime.RuntimeContractError("Host adapter cannot replace allocation/daemon ownership")
        environment[name] = value
    for name in ("AF3_IMAGE", "AF3_MODEL_DIR", "AF3_JAX_CACHE_DIR"):
        if not environment.get(name):
            raise runtime.RuntimeContractError(f"Native host profile must provide {name}")
    return environment


def _request(sock, command, timeout=1):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(sock))
        client.sendall(json.dumps({"cmd": command}).encode() + b"\n")
        data = client.recv(4096)
    return json.loads(data)


def _stop(process, *, sock=None):
    if process is None:
        return
    if sock is not None and process.poll() is None:
        try:
            _request(sock, "shutdown", timeout=20)
        except (OSError, ValueError):
            pass
    if process.poll() is None:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def container_command(bundle, bundle_path, fingerprint, environment, socket_dir=None, *, check=False):
    argv = list(bundle["runtime"]["argv"])
    if Path(argv[0]).name == "apptainer":
        # Native exec_core.sh does these binds/pass-throughs. The socket must
        # be the HOST socket even if the image has private /tmp defaults.
        additions = ["--bind", f"{bundle['repo']}:{bundle['repo']}"]
        for spec in environment.get("APPTAINER_BIND_PATHS", "").split(";"):
            if spec:
                additions += ["--bind", spec]
        for path in bundle["runtime"].get("host_af3", {}).get("adapter", {}).get("bind_paths", []):
            additions += ["--bind", f"{path}:{path}"]
        if socket_dir is not None:
            additions += ["--bind", f"{socket_dir}:{socket_dir}"]
        if environment.get("PYTHONPATH"):
            additions += ["--env", f"PYTHONPATH={environment['PYTHONPATH']}"]
        for name in ("AF3_LAUNCHER_SOCKET_TEMPLATE", "HHQ_RFD4_HOST_CONTEXT", "HHQ_RFD4_TMPDIR",
                     "TMPDIR", "TMP", "TEMP", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
                     "NCCL_P2P_DISABLE", "NCCL_DEBUG", "PYTHONFAULTHANDLER",
                     "TORCH_COMPILE_CACHE_BLOB", "TORCHINDUCTOR_FX_GRAPH_CACHE"):
            if environment.get(name):
                additions += ["--env", f"{name}={environment[name]}"]
        argv[2:2] = additions
    argv += [bundle["runtime_script"], "--bundle", str(bundle_path), "--fingerprint", fingerprint]
    if check:
        argv.append("--check")
    return argv


def run_host(bundle_path, fingerprint, *, check=False):
    bundle_path = Path(bundle_path).resolve()
    import hashlib
    if hashlib.sha256(bundle_path.read_bytes()).hexdigest() != fingerprint:
        raise RuntimeError("Bundle fingerprint mismatch")
    bundle = json.loads(bundle_path.read_text())
    # Import only a pinned standard-library module; no model/native imports here.
    runtime_path = bundle["runtime_script"]
    if hashlib.sha256(Path(runtime_path).read_bytes()).hexdigest() != bundle["source_hashes"][runtime_path]:
        raise RuntimeError("Pinned scientific runtime changed")
    runtime = _runtime(runtime_path)
    runtime.check_pins(bundle, bundle_path, fingerprint)
    if not bundle["runtime"].get("host_af3"):
        raise runtime.RuntimeContractError("Bundle did not authorize owned host AF3")
    environment = host_environment(bundle, runtime)
    if check:
        # CPU preflight composes native configs; it starts no daemon or worker.
        result = subprocess.run(container_command(bundle, bundle_path, fingerprint, environment, check=True),
                                env=environment, cwd=bundle["repo"], check=False)
        return result.returncode
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.isdecimal():
        raise runtime.RuntimeContractError("Host AF3 requires exactly one numeric CUDA_VISIBLE_DEVICES")
    adapter = bundle["runtime"].get("host_af3", {}).get("adapter", {})
    daemon_script = Path(adapter.get("daemon_script") or
                         (Path(bundle["repo"]) / "src/rfproteina/af3/af3_daemon.py"))
    if str(daemon_script) not in bundle["source_hashes"]:
        raise runtime.RuntimeContractError("Native AF3 daemon is not source-pinned")
    output = Path(environment["HHQ_TASK_OUTPUT_DIR"])
    runtime.ensure_directory(output)
    daemon = child = None
    handlers = {}

    def interrupted(signum, frame):
        raise InterruptedError(f"Host RFD4/AF3 interrupted by signal {signum}")

    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1):
        handlers[number] = signal.signal(number, interrupted)
    try:
        with tempfile.TemporaryDirectory(prefix="hhq-af3-", dir="/tmp") as temporary:
            socket_dir = Path(temporary)
            sock = socket_dir / "rank0.sock"
            context = socket_dir / "owner.json"
            # Never share Slurm's job-wide TMPDIR across scientific tasks. Site
            # step cleanup or a native library can remove that directory. Keep
            # scratch separate from the socket/owner files, and bind both into
            # the scientific container through this private parent directory.
            scratch = socket_dir / "scratch"
            scratch.mkdir(mode=0o700)
            environment.update({name: str(scratch) for name in ("TMPDIR", "TMP", "TEMP")})
            environment["HHQ_RFD4_TMPDIR"] = str(scratch)
            # serve runs in foreground, and every descendant stays in the HHQ
            # runner's process group. No --background, setsid, or shell exec wrapper.
            logs = Path(os.environ.get("HHQ_TASK_LOG_DIR", str(output)))
            logs.mkdir(parents=True, exist_ok=True, mode=0o700)
            with (logs / "native_af3_daemon.log").open("ab") as log:
                daemon = subprocess.Popen(
                    [bundle["runtime"]["host_af3"]["python"], str(daemon_script), "serve",
                     "--socket-path", str(sock), "--gpu-device", visible, "--lazy-warmup",
                     *(["--batch-script", adapter["batch_script"]] if adapter else [])],
                    env=environment, stdout=log, stderr=subprocess.STDOUT,
                )
            deadline = time.monotonic() + 30
            while True:
                if daemon.poll() is not None:
                    raise runtime.RuntimeContractError(f"Native host AF3 daemon exited: {daemon.returncode}")
                try:
                    if _request(sock, "ping").get("status") == "ok":
                        break
                except (OSError, ValueError):
                    pass
                if time.monotonic() >= deadline:
                    raise runtime.RuntimeContractError("Native host AF3 daemon did not become ready in 30s")
                time.sleep(0.1)
            template = str(socket_dir / "rank{rank}.sock")
            runtime.atomic_json(context, {"schema": "hhq-owned-af3/1", "bundle_sha256": fingerprint,
                                          "pid": os.getpid(), "daemon_pid": daemon.pid,
                                          "socket_template": template,
                                          "temporary_directory": str(scratch)})
            environment["AF3_LAUNCHER_SOCKET_TEMPLATE"] = template
            environment["HHQ_RFD4_HOST_CONTEXT"] = str(context)
            try:
                child = subprocess.Popen(container_command(bundle, bundle_path, fingerprint,
                                                           environment, socket_dir),
                                         env=environment, cwd=bundle["repo"])
                returncode = child.wait()
                runtime.check_pins(bundle, bundle_path, fingerprint)
                return returncode
            finally:
                _stop(child)
                _stop(daemon, sock=sock)
    finally:
        # Also handles bootstrap failures before the child/finally block.
        _stop(child)
        _stop(daemon)
        for number, previous in handlers.items():
            signal.signal(number, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return run_host(args.bundle, args.fingerprint, check=args.check)


if __name__ == "__main__":
    sys.exit(main())
