"""Application placement tests; no model, GPU, HQ server or Slurm is needed."""

from pathlib import Path

import pytest

from hyper_hyperqueue import api
from hyper_hyperqueue.integrations.af3 import submission as af3
from hyper_hyperqueue.integrations.gpu_compatibility import merge_gpu_constraint
from hyper_hyperqueue.models import Budget, Config, Profile, Resources


@pytest.fixture
def config(tmp_path):
    return Config(
        cluster="digs", state_dir=str(tmp_path / "state"),
        budgets={"gpu": Budget(partitions=["gpu"]), "bf": Budget(partitions=["gpu-bf"])},
        profiles={
            "gpu": Profile(partition="gpu", gres="gpu:1", budget="gpu", cpus=2, memory_mib=22528,
                           allocation_seconds=28800, constraint="UW", exclude=["broken-node"]),
            "bf": Profile(partition="gpu-bf", gres="gpu:1", budget="bf", cpus=2, memory_mib=22528,
                          allocation_seconds=18000),
            "cpu": Profile(partition="cpu", cpus=1, memory_mib=1024),
        },
        workloads={"generic_gpu": ["gpu", "bf"]},
    )


def test_af3_routes_default_guard_preserve_generic_resources_and_aliases(config):
    before = config.model_dump()
    result = af3.af3_profiles(config, ["gpu", "bf"])
    assert config.model_dump() == before
    assert result.workloads == {"generic_gpu": ["gpu", "bf"], "af3": ["af3-gpu", "af3-bf"]}
    for name in config.profiles:
        assert result.profiles[name] == config.profiles[name]
    for original in ("gpu", "bf"):
        generated = result.profiles[f"af3-{original}"].model_dump()
        generated.pop("constraint")
        baseline = config.profiles[original].model_dump()
        baseline.pop("constraint")
        assert generated == baseline
    assert result.profiles["af3-gpu"].constraint == f"(UW)&({af3.DIGS_AF3_GPU_CONSTRAINT})"
    assert result.profiles["af3-bf"].constraint == af3.DIGS_AF3_GPU_CONSTRAINT
    result.profiles["af3-gpu"].exclude.append("new-node")
    assert result.profiles["gpu"].exclude == ["broken-node"]
    assert not Path(config.state_dir).exists()


@pytest.mark.parametrize("mode", ["direct", "pooled"])
def test_af3_alias_plans_generic_commands_without_submitting(config, tmp_path, mode):
    result = af3.af3_profiles(config, ["gpu", "bf"])
    commands = tmp_path / "commands.txt"
    commands.write_text("python my_existing_af3_driver.py --input example.json\n")
    request = Resources(cpus=2, gpus=1, memory_mib=20480, runtime_seconds=300, runtime_confidence="declared")
    plan = api.plan(commands, result, profile="af3", mode=mode, resources=request)
    assert plan.task_count == 1
    assert plan.profiles == ["af3-gpu", "af3-bf"]
    assert plan.mode == mode
    assert not Path(config.state_dir).exists()


def test_af3_explicit_optout_preserves_custom_constraint(config):
    config.profiles["gpu"].constraint = "TitanRTX"
    result = af3.af3_profiles(config, ["gpu"], compatibility_constraint=None)
    assert result.profiles["af3-gpu"].constraint == "TitanRTX"


def test_af3_other_sites_need_their_own_guard(config):
    config.cluster = "another-cluster"
    assert af3.af3_profiles(config, ["gpu"]).profiles["af3-gpu"].constraint == "UW"
    result = af3.af3_profiles(config, ["gpu"], compatibility_constraint="sm80|sm90")
    assert result.profiles["af3-gpu"].constraint == "(UW)&(sm80|sm90)"


@pytest.mark.parametrize("late_constraint", [None, "TitanRTX", "UW&(A4000|TitanRTX)"])
def test_af3_final_guard_after_operator_overrides_is_idempotent(config, late_constraint):
    configured = af3.af3_profiles(config, ["gpu", "bf"])
    configured.profiles["af3-gpu"].constraint = late_constraint
    before = configured.model_dump()
    final = af3.apply_af3_gpu_constraints(configured, configured.workloads["af3"])
    expected = (f"({late_constraint})&({af3.DIGS_AF3_GPU_CONSTRAINT})" if late_constraint
                else af3.DIGS_AF3_GPU_CONSTRAINT)
    assert final.profiles["af3-gpu"].constraint == expected
    assert configured.model_dump() == before
    assert af3.apply_af3_gpu_constraints(final, final.workloads["af3"]).model_dump() == final.model_dump()
    assert af3.apply_af3_gpu_constraints(configured, configured.workloads["af3"],
                                       compatibility_constraint=None).model_dump() == before


@pytest.mark.parametrize("constraint", [False, 42, "", "\nAmpere", "Ampere\x00"])
def test_af3_rejects_invalid_compatibility_setting(config, constraint):
    with pytest.raises(ValueError, match="compatibility_constraint"):
        af3.af3_profiles(config, ["gpu"], compatibility_constraint=constraint)


@pytest.mark.parametrize("names", [[], ["missing"], ["cpu"], ["gpu", "gpu"], [None]])
def test_af3_requires_explicit_unique_gpu_routes(config, names):
    with pytest.raises(ValueError):
        af3.af3_profiles(config, names)


@pytest.mark.parametrize("names", [[], ["gpu"], ["af3-missing"], ["af3-gpu", "af3-gpu"]])
def test_af3_final_guard_cannot_change_generic_or_unknown_profiles(config, names):
    configured = af3.af3_profiles(config, ["gpu"])
    with pytest.raises(ValueError, match="AF3 profile names"):
        af3.apply_af3_gpu_constraints(configured, names)


def test_af3_existing_alias_and_profile_are_not_silently_overwritten(config):
    result = af3.af3_profiles(config, ["gpu"])
    with pytest.raises(ValueError, match="already exists"):
        af3.af3_profiles(result, ["gpu"])
    config.profiles["af3-gpu"] = config.profiles["gpu"].model_copy(deep=True)
    with pytest.raises(ValueError, match="already exists"):
        af3.af3_profiles(config, ["gpu"])


def test_af3_fallbacks_stay_in_guarded_clones(config):
    config.profiles["gpu-spare"] = config.profiles["gpu"].model_copy(deep=True)
    config.profiles["gpu"].fallback_profiles = ["gpu-spare"]
    with pytest.raises(ValueError, match="Include all fallback routes"):
        af3.af3_profiles(config, ["gpu"])
    result = af3.af3_profiles(config, ["gpu", "gpu-spare"])
    assert result.profiles["af3-gpu"].fallback_profiles == ["af3-gpu-spare"]
    assert result.profiles["gpu"].fallback_profiles == ["gpu-spare"]
    result.profiles["af3-gpu"].fallback_profiles = ["gpu-spare"]
    with pytest.raises(ValueError, match="selected AF3 profiles"):
        af3.apply_af3_gpu_constraints(result, result.workloads["af3"])


def test_guard_idempotence_cannot_preserve_or_bypass():
    guard = af3.DIGS_AF3_GPU_CONSTRAINT
    unsafe = f"(TitanRTX)|(UW)&({guard})"
    guarded = merge_gpu_constraint(unsafe, "auto", "digs")
    assert guarded == f"({unsafe})&({guard})"
    assert merge_gpu_constraint(guarded, "auto", "digs") == guarded


def test_af3_late_cpu_override_rejected(config):
    configured = af3.af3_profiles(config, ["gpu"])
    configured.profiles["af3-gpu"].gres = None
    with pytest.raises(ValueError, match="GPU allocation"):
        af3.apply_af3_gpu_constraints(configured, configured.workloads["af3"])
