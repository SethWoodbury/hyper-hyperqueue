"""Small opt-in application placement helpers; generic scheduling stays neutral."""

from __future__ import annotations

DIGS_MODERN_GPU_CONSTRAINT = "Ampere|Lovelace|Hopper|Blackwell"


def _fully_parenthesized(expression: str) -> bool:
    if not expression.startswith("(") or not expression.endswith(")"):
        return False
    depth = 0
    for index, character in enumerate(expression):
        depth += (character == "(") - (character == ")")
        if depth < 0 or (depth == 0 and index != len(expression) - 1):
            return False
    return depth == 0


def merge_gpu_constraint(existing: str | None, requested: str | None, cluster: str) -> str | None:
    """AND an application guard with an operator's Slurm constraint, idempotently.

    ``auto`` uses deployment feature labels only on DIGS. ``None`` leaves the
    existing placement untouched. Call this only from an opted-in application
    integration; neither shell text nor a generic GPU request selects a guard.
    """
    guard = DIGS_MODERN_GPU_CONSTRAINT if requested == "auto" and cluster == "digs" else requested
    if guard in (None, "auto"):
        return existing
    if not isinstance(guard, str) or not guard.strip() or any(c in guard for c in "\n\r\x00"):
        raise ValueError("compatibility_constraint must be 'auto', None, or a nonempty Slurm constraint")
    guard = guard.strip()
    if not existing or existing == guard:
        return guard
    suffix = f"&({guard})"
    if existing.endswith(suffix) and _fully_parenthesized(existing[:-len(suffix)]):
        return existing
    return f"({existing})&({guard})"
