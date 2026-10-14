from pathlib import Path
from types import SimpleNamespace

from hyper_hyperqueue.telemetry import AllocationSampler, job_cgroup

GPU = "GPU-12345678-1234-1234-1234-123456789abc"


def fixture_cgroup(tmp_path):
    proc, root = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "self").mkdir(parents=True)
    (proc / "self/cgroup").write_text("0::/slurm/uid_1000/job_123/step_batch/task_0\n")
    job = root / "slurm/uid_1000/job_123"
    job.mkdir(parents=True)
    (job / "memory.current").write_text("100\n")
    (job / "memory.peak").write_text("300\n")
    return proc, root, job


def test_job_scope_includes_siblings_and_preserves_previous_peak(tmp_path):
    proc, root, job = fixture_cgroup(tmp_path)
    sampler = AllocationSampler(environment={"SLURM_JOB_ID": "123"},
                                proc_root=proc, cgroup_root=root)
    sampler.sample()
    (job / "memory.current").write_text("200\n")
    (job / "memory.peak").write_text("350\n")
    sampler.sample()
    result = sampler.summary()
    assert result["cgroup_path"] == str(job)
    assert result["sampled_job_memory_current_peak_bytes"] == 200
    assert result["job_memory_lifetime_peak_bytes"] == 350
    assert result["job_memory_lifetime_peak_at_start_bytes"] == 300
    assert result["gpu_samples"] == 0


def test_refuses_wrong_job_or_whole_node_cgroup(tmp_path):
    proc, root, _ = fixture_cgroup(tmp_path)
    for environment in ({}, {"SLURM_JOB_ID": "124"}, {"SLURM_JOB_ID": "../../"}):
        assert job_cgroup(environment, proc_root=proc, cgroup_root=root) is None
    (proc / "self/cgroup").write_text("0::/\n")
    assert job_cgroup({"SLURM_JOB_ID": "123"}, proc_root=proc, cgroup_root=root) is None


def test_gpu_queries_only_assigned_uuids_and_throttles():
    calls = []
    def query(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=f"{GPU}, 123\n")
    sampler = AllocationSampler([{"uuid": GPU}], environment={}, query=query)
    sampler.sample()
    sampler.sample()
    assert len(calls) == 1
    assert calls[0][0][1] == f"--id={GPU}"
    assert calls[0][1]["timeout"] == 1
    assert sampler.summary()["assigned_gpu_sampled_peak_mib"] == {GPU: 123}


def test_gpu_failure_disables_queries_and_does_not_fail_task():
    calls = []
    def query(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="GPU-foreign, 999\n")
    sampler = AllocationSampler([{"uuid": GPU}], environment={}, query=query)
    sampler.sample()
    sampler.next_gpu_sample = 0
    sampler.sample()
    assert len(calls) == 1
    assert sampler.summary()["gpu_query_error"]
    assert sampler.summary()["assigned_gpu_sampled_peak_mib"] == {}


def test_cpu_and_bad_gpu_identity_never_probe_gpu():
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not inspect a nonassigned device")
    for devices in ([], [{"uuid": "0"}]):
        sampler = AllocationSampler(devices, environment={}, query=forbidden,
                                    proc_root=Path("/no/proc"))
        sampler.sample()
        assert sampler.summary()["gpu_samples"] == 0
