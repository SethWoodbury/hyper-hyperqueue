"""Owned host-daemon lifecycle without GPU, AF3 weights, or Slurm."""

import json
import os
import sys
from pathlib import Path

import pytest
from test_rfd4_prepare import sweep  # noqa: F401 -- shared native-file fixture

from hyper_hyperqueue.integrations.rfd4 import host_runtime as host
from hyper_hyperqueue.integrations.rfd4 import prepare, runtime


def test_host_mode_is_pinned_and_prepared_through_host_python(sweep):  # noqa: F811
    sweep["runtime"]["host_af3"] = {"python": sys.executable}
    plan = prepare.plan_sweep(sweep["path"], runtime=sweep["runtime"], workloads=sweep["workloads"])
    prepared = prepare.prepare_sweep(plan, sweep["root"] / "prepared-host")
    assert prepare.verify_prepared(prepared) == prepared
    command = json.loads(Path(prepared["groups"][0]["commands_file"]).read_text().splitlines()[0])
    assert command["argv"][:2] == [sys.executable, str(prepare.HOST_SCRIPT)]
    assert all(bundle["host_script"] in bundle["source_hashes"] for bundle in plan["bundles"])


def test_optional_host_adapter_is_bound_and_rejects_gpu_environment_override(sweep):  # noqa: F811
    daemon = sweep["root"] / "frozen-daemon.py"
    worker = sweep["root"] / "frozen-worker.py"
    config = sweep["root"] / "frozen-config.json"
    for path in (daemon, worker, config):
        path.write_text("{}\n")
    adapter = {"daemon_script": str(daemon), "batch_script": str(worker),
               "environment": {"RFD4_AF3_CACHE_CONFIG": str(config)},
               "source_hashes": {str(p): runtime.digest(p) for p in (daemon, worker, config)},
               "config": str(config), "config_sha256": runtime.digest(config),
               "bind_paths": [str(sweep["root"])]}
    sweep["runtime"]["host_af3"] = {"python": sys.executable, "adapter": adapter}
    planned = prepare.plan_sweep(sweep["path"], runtime=sweep["runtime"], workloads=sweep["workloads"])
    assert all(str(config) in bundle["source_hashes"] for bundle in planned["bundles"])
    adapter["environment"]["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = "other-device"
    with pytest.raises(ValueError, match="allocated resources"):
        prepare.plan_sweep(sweep["path"], runtime=sweep["runtime"], workloads=sweep["workloads"])


@pytest.fixture
def host_bundle(tmp_path, monkeypatch):
    repo = tmp_path / "native"
    af3 = repo / "src/rfproteina/af3"
    af3.mkdir(parents=True)
    daemon = af3 / "af3_daemon.py"
    daemon.write_text('''import argparse, json, os, socket
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("action"); p.add_argument("--socket-path")
p.add_argument("--gpu-device"); p.add_argument("--lazy-warmup", action="store_true")
a=p.parse_args()
Path(os.environ["PROBE_RECORD"]).write_text(json.dumps({"pid":os.getpid(), "pgid":os.getpgrp(), "gpu":a.gpu_device}))
s=socket.socket(socket.AF_UNIX); s.bind(a.socket_path); s.listen()
while True:
 c,_=s.accept(); req=json.loads(c.recv(4096)); c.sendall(b'{"status":"ok"}\\n'); c.close()
 if req["cmd"]=="shutdown": break
s.close()
''')
    profile = repo / ".env.digs"
    profile.write_text("AF3_IMAGE=/fake/image.sif\nAF3_MODEL_DIR=/fake/models\nAF3_JAX_CACHE_DIR=/fake/cache\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    inp = tmp_path / "input.cif"
    inp.write_text("data_test\n")
    record = tmp_path / "daemon-record.json"
    row = {"config": str(config), "config_sha256": runtime.digest(config),
           "input": str(inp), "input_sha256": runtime.digest(inp),
           "environment": {"CLUSTER": "digs", "PROBE_RECORD": str(record)},
           "unset_environment": [], "dotenv": {"path": str(profile)}}
    paths = [Path(runtime.__file__), Path(host.__file__), daemon, profile, manifest, config, inp]
    bundle = {"schema": "hhq-rfd4-bundle/1", "rows": [row], "repo": str(repo),
              "runtime_script": str(Path(runtime.__file__)), "host_script": str(Path(host.__file__)),
              "source_manifest": {"path": str(manifest), "sha256": runtime.digest(manifest)},
              "source_hashes": {str(p): runtime.digest(p) for p in paths},
              "runtime": {"argv": [sys.executable], "environment": {},
                          "host_af3": {"python": sys.executable}}}
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("HHQ_TASK_OUTPUT_DIR", str(tmp_path / "output"))
    for name in ("AF3_LAUNCHER_SOCKET_TEMPLATE", "AF3_SOCKET_DIR", "HHQ_RFD4_HOST_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("HHQ_RFD4_TMPDIR", raising=False)
    return path, runtime.digest(path), record


@pytest.mark.parametrize("exit_code", [0, 7])
def test_foreground_daemon_has_allocated_device_and_stops_on_child_exit(host_bundle, monkeypatch, exit_code):
    path, fingerprint, record = host_bundle
    def child(*args, **kwargs):
        return [sys.executable, "-c", f"raise SystemExit({exit_code})"]
    monkeypatch.setattr(host, "container_command", child)
    assert host.run_host(path, fingerprint) == exit_code
    evidence = json.loads(record.read_text())
    assert evidence["gpu"] == "3"
    assert evidence["pgid"] == os.getpgrp()  # never detached from runner's task group
    with pytest.raises(ProcessLookupError):
        os.kill(evidence["pid"], 0)


def test_check_never_starts_daemon(host_bundle, monkeypatch):
    path, fingerprint, record = host_bundle
    monkeypatch.setattr(host, "container_command", lambda *a, **k: [sys.executable, "-c", "pass"])
    assert host.run_host(path, fingerprint, check=True) == 0
    assert not record.exists()
    assert not Path(os.environ["HHQ_TASK_OUTPUT_DIR"]).exists()


def test_foreign_daemon_and_uuid_visibility_fail_closed(host_bundle, monkeypatch):
    path, fingerprint, _ = host_bundle
    monkeypatch.setenv("AF3_LAUNCHER_SOCKET_TEMPLATE", "/somebody/else.sock")
    with pytest.raises(Exception, match="inherited AF3"):
        host.run_host(path, fingerprint)
    monkeypatch.delenv("AF3_LAUNCHER_SOCKET_TEMPLATE")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-uuid")
    with pytest.raises(Exception, match="numeric CUDA_VISIBLE_DEVICES"):
        host.run_host(path, fingerprint)


def test_host_profile_cannot_replace_allocation(host_bundle):
    path, _fingerprint, _ = host_bundle
    bundle = json.loads(path.read_text())
    profile = Path(bundle["rows"][0]["dotenv"]["path"])
    profile.write_text(profile.read_text() + "CUDA_VISIBLE_DEVICES=0\n")
    bundle["source_hashes"][str(profile)] = runtime.digest(profile)
    path.write_text(json.dumps(bundle))
    with pytest.raises(Exception, match="protected allocation"):
        host.run_host(path, runtime.digest(path))


def test_inner_runtime_requires_matching_live_owner(host_bundle, monkeypatch, tmp_path):
    path, fingerprint, _ = host_bundle
    bundle = json.loads(path.read_text())
    with pytest.raises(runtime.RuntimeContractError, match="host wrapper"):
        runtime.validate_host_daemon(bundle, fingerprint)
    context = tmp_path / "owner.json"
    context.write_text(json.dumps({"schema": "hhq-owned-af3/1", "bundle_sha256": fingerprint,
                                   "pid": os.getpid(), "daemon_pid": os.getpid(),
                                   "socket_template": "/temporary/rank{rank}.sock",
                                   "temporary_directory": str(tmp_path / "scratch")}))
    monkeypatch.setenv("HHQ_RFD4_HOST_CONTEXT", str(context))
    monkeypatch.setenv("AF3_LAUNCHER_SOCKET_TEMPLATE", "/temporary/rank{rank}.sock")
    monkeypatch.setenv("HHQ_RFD4_TMPDIR", str(tmp_path / "scratch"))
    runtime.validate_host_daemon(bundle, fingerprint)
    with pytest.raises(runtime.RuntimeContractError, match="does not match"):
        runtime.validate_host_daemon(bundle, "different-bundle")


def test_two_sequential_children_remove_only_their_private_tmpdir(host_bundle, monkeypatch, tmp_path):
    path, fingerprint, _ = host_bundle
    stale = tmp_path / "missing-slurm-job-tmp"
    monkeypatch.setenv("TMPDIR", str(stale))
    evidence = tmp_path / "child-temp.jsonl"

    def child(*args, **kwargs):
        return [sys.executable, "-c", """
import json, os, shutil, tempfile
from pathlib import Path
root = Path(tempfile.gettempdir())
assert str(root) == os.environ['HHQ_RFD4_TMPDIR']
assert root.is_dir()
with tempfile.TemporaryDirectory() as work:
    Path(work, 'probe').write_text('ok')
with open(os.environ['TEMP_RECORD'], 'a') as out:
    out.write(json.dumps({'tmpdir': str(root), 'owner': os.environ['HHQ_RFD4_HOST_CONTEXT']}) + '\\n')
# Emulate native/site scratch cleanup before the next task starts.
shutil.rmtree(root)
assert Path(os.environ['HHQ_RFD4_HOST_CONTEXT']).exists()
"""]

    monkeypatch.setenv("TEMP_RECORD", str(evidence))
    monkeypatch.setattr(host, "container_command", child)
    assert host.run_host(path, fingerprint) == 0
    assert host.run_host(path, fingerprint) == 0
    rows = [json.loads(line) for line in evidence.read_text().splitlines()]
    assert rows[0]["tmpdir"] != rows[1]["tmpdir"]
    assert not stale.exists()
    assert all(not Path(row["tmpdir"]).exists() for row in rows)
    assert os.environ["TMPDIR"] == str(stale)


def test_container_explicitly_binds_and_exports_owned_tmpdir(host_bundle):
    path, fingerprint, _ = host_bundle
    bundle = json.loads(path.read_text())
    bundle["runtime"]["argv"] = ["/usr/bin/apptainer", "exec", "--nv", "/image.sif", "/usr/bin/python3"]
    env = {"HHQ_RFD4_TMPDIR": "/tmp/owned/scratch", "TMPDIR": "/tmp/owned/scratch",
           "TMP": "/tmp/owned/scratch", "TEMP": "/tmp/owned/scratch"}
    argv = host.container_command(bundle, path, fingerprint, env, Path("/tmp/owned"))
    assert "/tmp/owned:/tmp/owned" in argv
    assert "TMPDIR=/tmp/owned/scratch" in argv
    assert "HHQ_RFD4_TMPDIR=/tmp/owned/scratch" in argv


def test_native_profile_binds_and_allocator_reach_container_explicitly(host_bundle):
    path, fingerprint, _ = host_bundle
    bundle = json.loads(path.read_text())
    bundle["runtime"]["argv"] = ["apptainer", "exec", "--nv", "/image.sif", "/usr/bin/python3"]
    environment = {"APPTAINER_BIND_PATHS": "/host data:/data:ro;;/models",
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "NCCL_P2P_DISABLE": "1"}
    argv = host.container_command(bundle, path, fingerprint, environment)
    assert "/host data:/data:ro" in argv and "/models" in argv
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in argv
    assert "NCCL_P2P_DISABLE=1" in argv
    assert "PYTHONFAULTHANDLER=1" not in argv  # Do not invent unspecified settings.
