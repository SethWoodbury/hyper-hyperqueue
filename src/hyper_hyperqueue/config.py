from __future__ import annotations

import os
from pathlib import Path

import yaml

from .models import Config


def load_config(path: str | Path | None = None, *, overrides: dict | None = None) -> Config:
    path = path or os.environ.get("HHQ_CONFIG")
    if path is None:
        candidate = Path.home() / ".config/hyper-hyperqueue/config.yaml"
        path = candidate if candidate.exists() else None
    data = yaml.safe_load(Path(path).read_text()) if path else {}
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping")  # noqa: TRY004 -- invalid user document

    def merge(base, changes):
        for key, value in changes.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value

    merge(data, overrides or {})
    config = Config.model_validate(data)
    config.state_dir = str(Path(config.state_dir).expanduser().absolute())
    for name, profile in config.profiles.items():
        if profile.budget and profile.budget not in config.budgets:
            raise ValueError(f"Profile {name}: unknown budget {profile.budget}")
    for name, profiles in config.workloads.items():
        unknown = set(profiles) - config.profiles.keys()
        if unknown:
            raise ValueError(f"Workload {name}: unknown profiles {sorted(unknown)}")
    return config


def profile_names(config: Config, name: str) -> list[str]:
    if name in config.workloads:
        return config.workloads[name]
    if name in config.profiles:
        return [name]
    raise ValueError(f"Unknown profile/workload {name!r}; configure it or use --partition")


def save_config(config: Config, path: Path) -> None:
    from .state import atomic_json

    atomic_json(path, config.model_dump())  # JSON is valid YAML too.
