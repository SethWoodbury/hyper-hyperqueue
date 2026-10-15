"""Guard explicit AF3 routes without interpreting or changing scientific commands."""

from __future__ import annotations

from ...models import Config
from ...planner import gpu_count
from ...state import identifier
from ..gpu_compatibility import DIGS_MODERN_GPU_CONSTRAINT, merge_gpu_constraint

DIGS_AF3_GPU_CONSTRAINT = DIGS_MODERN_GPU_CONSTRAINT


def apply_af3_gpu_constraints(
    config: Config, profile_names, *, compatibility_constraint: str | None = "auto",
) -> Config:
    """Reapply AF3 placement after profile overrides; generic profiles stay intact.

    DIGS ``auto`` requires modern GPU features for the qualified cuDNN attention
    path. Explicit ``None`` leaves the supplied constraints untouched. Other
    clusters need their own feature expression; ``auto`` does not guess one.
    """
    names = list(profile_names)
    if (not names or any(not isinstance(name, str) or not name.startswith("af3-")
                         or name not in config.profiles for name in names)
            or len(names) != len(set(names))):
        raise ValueError("profile_names must contain unique existing AF3 profile names")
    result = config.model_copy(deep=True)
    for name in names:
        profile = result.profiles[name]
        if gpu_count(profile) < 1:
            raise ValueError("AF3 profiles require an explicit GPU allocation")
        profile.constraint = merge_gpu_constraint(profile.constraint, compatibility_constraint, result.cluster)
        if set(profile.fallback_profiles) - set(names):
            raise ValueError("AF3 fallback routes must stay among the selected AF3 profiles")
    return Config.model_validate(result.model_dump())


def af3_profiles(
    base_config: Config, eligible_profile_names, *, compatibility_constraint: str | None = "auto",
) -> Config:
    """Clone explicit GPU routes as ``af3-<route>`` and expose workload alias ``af3``.

    Resource requests, walltimes, modes and budgets remain operator-controlled.
    Call generic ``api.plan/submit(..., profile='af3')`` with the returned config.
    This prepares no commands, starts no service and submits no jobs. Reapply
    ``apply_af3_gpu_constraints`` after any subsequent placement overrides.
    """
    names = list(eligible_profile_names)
    if (not names or any(not isinstance(name, str) or name not in base_config.profiles for name in names)
            or len(names) != len(set(names))):
        raise ValueError("eligible_profile_names must contain unique existing profile names")
    clones = {name: identifier(f"af3-{name}") for name in names}
    if "af3" in base_config.workloads or set(clones.values()) & set(base_config.profiles):
        raise ValueError("AF3 profile/alias already exists; use a fresh preparation")
    result = base_config.model_copy(deep=True)
    for original, name in clones.items():
        base = base_config.profiles[original]
        if gpu_count(base) < 1:
            raise ValueError("AF3 profiles require an explicit GPU allocation")
        if set(base.fallback_profiles) - set(names):
            raise ValueError("Include all fallback routes in eligible_profile_names")
        clone = base.model_copy(deep=True)
        clone.fallback_profiles = [clones[target] for target in base.fallback_profiles]
        result.profiles[name] = clone
    result.workloads["af3"] = list(clones.values())
    return apply_af3_gpu_constraints(result, list(clones.values()), compatibility_constraint=compatibility_constraint)
