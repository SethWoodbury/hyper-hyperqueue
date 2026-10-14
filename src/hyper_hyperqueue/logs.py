"""Optional console-log layout; durable state and scientific outputs stay put."""

from pathlib import Path

from .state import ensure_directory, identifier


def log_directory(config, *parts: str, fallback: Path) -> Path:
    """Resolve and create a log leaf only when execution actually needs it.

    ``config`` can be an immutable campaign's resolved configuration or Config.
    Unconfigured callers retain the historical layout exactly.
    """
    root = config.get("log_dir") if isinstance(config, dict) else config.log_dir
    directory = (
        Path(root).expanduser().absolute().joinpath(*(identifier(p) for p in parts))
        if root else Path(fallback)
    )
    return ensure_directory(directory)


def attempt_log_path(attempt: dict, stream: str) -> Path:
    if stream not in {"stdout", "stderr", "validator.stdout", "validator.stderr"}:
        raise ValueError("Unknown attempt log stream")
    return Path(attempt.get("log_dir", attempt["attempt_dir"])) / stream
