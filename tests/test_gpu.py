import ctypes
import subprocess

import pytest

from hyper_hyperqueue.gpu import cuda_visible_devices, query_assigned_devices


class Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class Driver:
    def __init__(self, count=1):
        self.calls = []
        self.cuInit = Function(lambda flags: 0)
        self.cuDeviceGetCount = Function(lambda out: self.write(out, count))
        self.cuDeviceGet = Function(self.get)
        self.cuDeviceGetName = Function(self.name)
        self.cuDeviceTotalMem_v2 = Function(lambda out, device: self.write(out, 40 * 1024**3))
        self.cuDeviceGetUuid_v2 = Function(self.identity)

    @staticmethod
    def write(out, value):
        out._obj.value = value
        return 0

    def get(self, out, ordinal):
        self.calls.append(ordinal)
        return self.write(out, ordinal)

    @staticmethod
    def name(out, size, device):
        out.value = b"NVIDIA A100"
        return 0

    @staticmethod
    def identity(out, device):
        ctypes.memmove(out, bytes(range(1, 17)), 16)
        return 0


def test_cuda_driver_reads_only_visible_ordinals_without_context():
    driver = Driver()
    records = cuda_visible_devices(1, library=driver)
    assert driver.calls == [0]
    assert records == [
        {
            "uuid": "GPU-01020304-0506-0708-090a-0b0c0d0e0f10",
            "model": "NVIDIA A100",
            "memory_mib": 40960,
            "cuda_ordinal": 0,
            "identity_source": "cuDeviceGetUuid_v2",
        }
    ]


def test_cuda_driver_count_mismatch_fails_before_identity_queries():
    driver = Driver(count=8)
    with pytest.raises(ValueError, match="differs from assigned"):
        cuda_visible_devices(1, library=driver)
    assert driver.calls == []


def test_driver_errors_are_explicit():
    driver = Driver()
    driver.cuInit = Function(lambda flags: 100)
    with pytest.raises(ValueError, match="cuInit failed with CUDA driver error 100"):
        cuda_visible_devices(1, library=driver)


def test_numeric_slurm_id_is_metadata_not_a_query_index():
    queried = []
    env = {"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "6", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    original = dict(env)

    def query(ordinal):
        queried.append(ordinal)
        return {"uuid": "GPU-selected", "model": "A100", "memory_mib": 40960}

    result = query_assigned_devices(1, environment=env, query=query)
    assert queried == [0] and env == original
    assert result[0]["assigned_identifier"] == "GPU-selected"
    assert result[0]["slurm_job_gpus"] == "6"


def test_probe_subprocess_preserves_visibility_and_reports_bounded_stderr(monkeypatch):
    env = {"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "6"}

    def run(argv, **kwargs):
        assert kwargs["env"] == env and kwargs["timeout"] == 20
        assert argv[-2:] == ["--count", "1"]
        return subprocess.CompletedProcess(argv, 1, "", "driver diagnostic " + "x" * 3000)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ValueError, match="query exited 1") as caught:
        query_assigned_devices(1, environment=env)
    assert len(str(caught.value)) < 2100


def test_mig_explicitly_rejected_before_probe():
    with pytest.raises(ValueError, match="MIG"):
        query_assigned_devices(1, environment={"CUDA_VISIBLE_DEVICES": "MIG-abc"})


def test_worker_preserves_failed_preflight_diagnostics(tmp_path, monkeypatch):
    from hyper_hyperqueue import config, worker
    from hyper_hyperqueue.models import Config
    from hyper_hyperqueue.state import atomic_json, read_json

    configured = Config(state_dir=str(tmp_path), profiles={"gpu": {"partition": "gpu", "gres": "gpu:1"}})
    monkeypatch.setattr(config, "load_config", lambda *_args, **_kwargs: configured)
    monkeypatch.setenv("SLURM_JOB_ID", "734")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_JOB_GPUS", "6")

    def fail(_profile):
        raise ValueError("assigned CUDA identity failed: diagnostic")

    monkeypatch.setattr(worker, "detect_assigned", fail)
    atomic_json(tmp_path / "service/endpoint.json", {"epoch": 2, "status": "ACTIVE"})
    with pytest.raises(ValueError, match="identity failed"):
        worker.main(["--state-dir", str(tmp_path), "--profile", "gpu", "--config", "unused", "--epoch", "2"])
    receipt = read_json(tmp_path / "service/worker_preflight/734.json")
    assert receipt["status"] == "failed" and "diagnostic" in receipt["error"]
    assert receipt["environment"]["SLURM_JOB_GPUS"] == "6"
    assert receipt["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert not (tmp_path / "service/workers/734.json").exists()
