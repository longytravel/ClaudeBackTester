"""Parameter widening: expand numeric param ranges to fill trial budgets.

For numeric ParamDefs, widening produces finer steps AND extended range
using a single `factor` multiplier. Non-numeric params (categoricals,
booleans, day-lists) are left unchanged.

Usage:
    from backtester.strategies.param_widening import apply_widening
    new_space, summary = apply_widening(strategy, mode="2.0", preset_name="standard")
"""

from __future__ import annotations

import math
from typing import Any

from backtester.strategies.base import ParamDef, ParamSpace


def _is_numeric(values: list[Any]) -> bool:
    """Check if all values in a ParamDef are numeric (int or float), not bool."""
    return len(values) >= 2 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    )


def _is_int_param(values: list[Any]) -> bool:
    """Check if all values are integers (not bools)."""
    return all(isinstance(v, int) and not isinstance(v, bool) for v in values)


def widen_param(pdef: ParamDef, factor: float) -> ParamDef:
    """Widen a single numeric ParamDef by the given factor.

    - Finer steps: avg_step / sqrt(factor)
    - Extended range: +(sqrt(factor) - 1) * range / 2 each side
    - Original values always preserved (union)
    - Positive-only if originals were all positive
    - Ints clamped to >= 1
    """
    if factor <= 1.0 or not _is_numeric(pdef.values) or len(pdef.values) < 2:
        return pdef

    vals = sorted(pdef.values)
    orig_min = vals[0]
    orig_max = vals[-1]
    orig_range = orig_max - orig_min
    is_int = _is_int_param(pdef.values)
    all_positive = all(v > 0 for v in vals)

    # Compute average step
    avg_step = orig_range / (len(vals) - 1)

    # New step = avg_step / sqrt(factor) (finer grid)
    sqrt_f = math.sqrt(factor)
    new_step = avg_step / sqrt_f
    if is_int:
        new_step = max(1, round(new_step))

    # Extend range by (sqrt(factor) - 1) * range / 2 each side
    extension = (sqrt_f - 1) * orig_range / 2
    new_min = orig_min - extension
    new_max = orig_max + extension

    if is_int:
        new_min = round(new_min)
        new_max = round(new_max)

    # Enforce positive-only if originals were all positive
    if all_positive:
        if is_int:
            new_min = max(1, new_min)
        else:
            new_min = max(0.0, new_min)

    # Generate new values from new_min to new_max at new_step
    new_vals: list[int | float] = []
    if is_int:
        new_step_int = max(1, int(new_step))
        v = int(new_min)
        while v <= new_max:
            new_vals.append(v)
            v += new_step_int
        # Ensure new_max included
        if new_vals and new_vals[-1] < int(new_max):
            new_vals.append(int(new_max))
    else:
        v = new_min
        while v <= new_max + new_step * 0.01:  # small epsilon for float imprecision
            new_vals.append(round(v, 6))
            v += new_step

    # Union with original values (always preserve author's choices)
    orig_set = set(vals)
    merged_set = set(new_vals) | orig_set
    merged = sorted(merged_set)

    return ParamDef(name=pdef.name, values=merged, group=pdef.group)


def widen_param_space(
    ps: ParamSpace,
    factor: float,
    groups: list[str] | None = None,
) -> ParamSpace:
    """Apply widening to all numeric params in target groups.

    Args:
        ps: Original parameter space
        factor: Widening factor (1.0 = no change)
        groups: Groups to widen (None = all groups)

    Returns:
        New ParamSpace with widened params
    """
    new_params: list[ParamDef] = []
    target_groups = set(groups) if groups else None

    for pdef in ps:
        if target_groups is not None and pdef.group not in target_groups:
            new_params.append(pdef)
        else:
            new_params.append(widen_param(pdef, factor))

    return ParamSpace(new_params)


def group_combos(ps: ParamSpace, group: str) -> int:
    """Count unique combos for a single parameter group."""
    combos = 1
    for pdef in ps:
        if pdef.group == group:
            combos *= len(pdef.values)
    return combos


def compute_fill_factor(
    ps: ParamSpace,
    target_combos: int,
    group: str,
) -> float:
    """Compute the factor needed to reach target_combos for a group.

    Uses nth root where n = number of numeric params in the group,
    so the expansion is distributed evenly across params.
    """
    current = group_combos(ps, group)
    if current <= 0 or current >= target_combos:
        return 1.0

    # Count numeric params in group
    n_numeric = 0
    for pdef in ps:
        if pdef.group == group and _is_numeric(pdef.values):
            n_numeric += 1

    if n_numeric == 0:
        return 1.0

    ratio = target_combos / current
    factor = ratio ** (1.0 / n_numeric)

    # Safety clamp
    return min(max(factor, 1.0), 10.0)


def apply_widening(
    strategy,
    mode: str,
    preset_name: str,
) -> tuple[ParamSpace, dict]:
    """Top-level widening function called from full_run.py.

    Args:
        strategy: Strategy instance (will be monkey-patched)
        mode: "1.0" | "1.5" | "2.0" | "3.0" | "auto"
        preset_name: Preset name for budget lookup (used by "auto" mode)

    Returns:
        (new_param_space, summary_dict) for logging
    """
    if mode == "1.0":
        ps = strategy.param_space()
        return ps, {"mode": "1.0", "groups": {}}

    original_ps = strategy.param_space()
    raw_stages = strategy.optimization_stages()

    # Normalize stages to (name, [groups]) tuples
    stages: list[tuple[str, list[str]]] = []
    for entry in raw_stages:
        if isinstance(entry, str):
            stages.append((entry, [entry]))
        else:
            stages.append((entry[0], list(entry[1])))

    summary: dict[str, Any] = {"mode": mode, "groups": {}}

    if mode == "auto":
        from backtester.optimizer.config import get_preset

        config = get_preset(preset_name)
        new_ps = original_ps  # start from original

        for stage_name, stage_groups in stages:
            # For composite stages, compute combined combo count
            before = 1
            for g in stage_groups:
                before *= group_combos(new_ps, g)
            budget = config.trials_per_stage

            # Auto-cap mirrors staged optimizer: max 10x coverage
            max_coverage = 10

            # Only widen if budget > current combos * max_coverage
            # (i.e., the auto-cap would kick in and waste budget)
            if before > 0 and budget > before * max_coverage:
                target = budget // max_coverage  # aim for max_coverage x coverage
                # Widen each group in the stage
                for g in stage_groups:
                    g_target = max(1, int(target ** (1.0 / len(stage_groups))))
                    g_factor = compute_fill_factor(new_ps, g_target, g)
                    if g_factor > 1.0:
                        new_ps = widen_param_space(new_ps, g_factor, groups=[g])
            else:
                factor = 1.0

            after = 1
            for g in stage_groups:
                after *= group_combos(new_ps, g)
            summary["groups"][stage_name] = {
                "before": before,
                "after": after,
                "factor": round(after / before, 1) if before > 0 else 1.0,
            }
    else:
        factor = float(mode)
        new_ps = widen_param_space(original_ps, factor, groups=None)

        for stage_name, stage_groups in stages:
            before = 1
            for g in stage_groups:
                before *= group_combos(original_ps, g)
            after = 1
            for g in stage_groups:
                after *= group_combos(new_ps, g)
            summary["groups"][stage_name] = {
                "before": before,
                "after": after,
                "factor": round(after / before, 1) if before > 0 else 1.0,
            }

    # Monkey-patch strategy instance so ALL downstream code sees widened space
    new_ps_ref = new_ps  # capture for closure
    strategy.param_space = lambda: new_ps_ref

    return new_ps, summary
