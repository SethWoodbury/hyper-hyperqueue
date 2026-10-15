"""Explicit measured-cost packing for future RFD4 preparations, never live tuning."""

from __future__ import annotations

import math


def _number(value, name, *, minimum=0, strict=True):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value <= minimum if strict else value < minimum)):
        raise ValueError(f"packing.{name} must be finite and {'>' if strict else '>='} {minimum}")
    return value


def validate_packing(value, resources):
    """Keep a common method fit budget while varying only each recipe's row count."""
    required = {"target_seconds", "fit_seconds", "startup_seconds", "safety_factor"}
    allowed = required | {"seconds_per_row", "seconds_per_row_by_recipe"}
    if not isinstance(value, dict) or set(value) - allowed or not required <= value.keys():
        raise ValueError("packing requires target_seconds, fit_seconds, startup_seconds, safety_factor")
    result = dict(value)
    for key in ("target_seconds", "fit_seconds"):
        _number(result[key], key)
    _number(result["startup_seconds"], "startup_seconds", strict=False)
    _number(result["safety_factor"], "safety_factor", minimum=1, strict=False)
    if not result["startup_seconds"] < result["fit_seconds"] <= result["target_seconds"]:
        raise ValueError("packing requires startup_seconds < fit_seconds <= target_seconds")
    if resources.runtime_seconds != result["fit_seconds"]:
        raise ValueError("Set resources.runtime_seconds equal to packing.fit_seconds (whole-bundle fit budget)")
    if result.get("seconds_per_row") is not None:
        _number(result["seconds_per_row"], "seconds_per_row")
    costs = result.get("seconds_per_row_by_recipe", {})
    if not isinstance(costs, dict) or any(not isinstance(k, str) or not k for k in costs):
        raise ValueError("packing.seconds_per_row_by_recipe must map recipe IDs to positive seconds")
    for key, cost in costs.items():
        _number(cost, f"seconds_per_row_by_recipe[{key}]")
    if result.get("seconds_per_row") is None and not costs:
        raise ValueError("packing requires a seconds_per_row fallback or recipe-specific costs")
    result["seconds_per_row_by_recipe"] = dict(costs)
    return result


def recipe_packing(packing, recipe_id, hard_count_cap):
    """Compute an inspectable count from operator-supplied timings in seconds/row.

    A row includes all replicates and configured AF3 work. Timings must describe
    the selected recipe/hardware; these values are never inferred from shell text.
    """
    costs = packing["seconds_per_row_by_recipe"]
    cost = costs.get(recipe_id, packing.get("seconds_per_row"))
    if cost is None:
        raise ValueError(f"No packing timing for recipe {recipe_id}; supply a cost or fallback")
    available = packing["fit_seconds"] - packing["startup_seconds"]
    rows = math.floor(available / (cost * packing["safety_factor"]))
    if rows < 1:
        raise ValueError(f"Recipe {recipe_id}: one row exceeds the packing fit budget; enlarge it or change the recipe")
    return {
        "recipe_id": recipe_id, "target_seconds": packing["target_seconds"],
        "fit_seconds": packing["fit_seconds"], "startup_seconds": packing["startup_seconds"],
        "safety_factor": packing["safety_factor"], "seconds_per_row": cost,
        "timing_source": "recipe" if recipe_id in costs else "fallback",
        "runs_per_task": min(rows, hard_count_cap), "hard_count_cap": hard_count_cap,
    }


def bundle_estimate(recipe, rows):
    """Report actual tail size without pretending its declared fit budget shrinks."""
    return {
        **recipe, "rows": rows,
        "estimated_seconds": recipe["startup_seconds"] + rows * recipe["seconds_per_row"],
        "conservative_estimate_seconds": (
            recipe["startup_seconds"] + rows * recipe["seconds_per_row"] * recipe["safety_factor"]
        ),
    }
