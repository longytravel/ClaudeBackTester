"""Tests for backtester.strategies.param_widening."""

import pytest

from backtester.strategies.base import ParamDef, ParamSpace
from backtester.strategies.param_widening import (
    _is_numeric,
    compute_fill_factor,
    group_combos,
    widen_param,
    widen_param_space,
)


# ---- _is_numeric ----

def test_is_numeric_ints():
    assert _is_numeric([10, 20, 30]) is True


def test_is_numeric_floats():
    assert _is_numeric([0.5, 1.0, 1.5]) is True


def test_is_numeric_mixed():
    assert _is_numeric([1, 2.0, 3]) is True


def test_is_numeric_strings():
    assert _is_numeric(["off", "fixed_pip"]) is False


def test_is_numeric_booleans():
    assert _is_numeric([False, True]) is False


def test_is_numeric_single_value():
    assert _is_numeric([42]) is False


def test_is_numeric_lists():
    assert _is_numeric([[0, 1, 2], [3, 4]]) is False


# ---- widen_param basics ----

def test_widen_param_factor_1():
    """Factor 1.0 returns unchanged param."""
    p = ParamDef("rsi_period", [10, 14, 20, 25, 30], group="signal")
    result = widen_param(p, 1.0)
    assert result.values == p.values
    assert result.name == p.name
    assert result.group == p.group


def test_widen_param_non_numeric():
    """Non-numeric params are unchanged regardless of factor."""
    p = ParamDef("sl_mode", ["fixed_pips", "atr_based", "swing"], group="risk")
    result = widen_param(p, 2.0)
    assert result.values == p.values


def test_widen_param_boolean():
    """Boolean params are unchanged."""
    p = ParamDef("breakeven_enabled", [False, True], group="management")
    result = widen_param(p, 3.0)
    assert result.values == [False, True]


def test_widen_param_preserves_originals():
    """All original values must be present in widened result."""
    p = ParamDef("rsi_period", [10, 14, 20, 25, 30], group="signal")
    result = widen_param(p, 2.0)
    for v in p.values:
        assert v in result.values, f"Original value {v} missing from widened {result.values}"


def test_widen_param_int_more_values():
    """Widening an int param produces more values."""
    p = ParamDef("rsi_period", [10, 20, 30], group="signal")
    result = widen_param(p, 2.0)
    assert len(result.values) > len(p.values)


def test_widen_param_float_more_values():
    """Widening a float param produces more values."""
    p = ParamDef("sl_atr_mult", [0.5, 1.0, 1.5, 2.0], group="risk")
    result = widen_param(p, 2.0)
    assert len(result.values) > len(p.values)


def test_widen_param_int_stays_positive():
    """Positive int params stay >= 1 after widening."""
    p = ParamDef("rsi_period", [5, 10, 15], group="signal")
    result = widen_param(p, 3.0)
    assert all(v >= 1 for v in result.values)


def test_widen_param_float_stays_nonneg():
    """Positive float params stay >= 0 after widening."""
    p = ParamDef("sl_atr_mult", [0.5, 1.0, 1.5], group="risk")
    result = widen_param(p, 3.0)
    assert all(v >= 0.0 for v in result.values)


def test_widen_param_sorted():
    """Widened values are sorted ascending."""
    p = ParamDef("rsi_period", [30, 10, 20], group="signal")
    result = widen_param(p, 2.0)
    assert result.values == sorted(result.values)


def test_widen_param_no_duplicates():
    """No duplicate values after widening."""
    p = ParamDef("rsi_period", [10, 20, 30, 40, 50], group="signal")
    result = widen_param(p, 2.0)
    assert len(result.values) == len(set(result.values))


def test_widen_param_group_preserved():
    """Group assignment is preserved."""
    p = ParamDef("sl_fixed_pips", list(range(10, 51, 5)), group="risk")
    result = widen_param(p, 1.5)
    assert result.group == "risk"


def test_widen_param_name_preserved():
    """Name is preserved."""
    p = ParamDef("trail_distance_pips", [5, 10, 15, 20, 30], group="management")
    result = widen_param(p, 2.0)
    assert result.name == "trail_distance_pips"


def test_widen_param_range_extends():
    """Widened param has an extended range (wider min/max)."""
    p = ParamDef("rsi_period", [10, 20, 30], group="signal")
    result = widen_param(p, 2.0)
    assert min(result.values) <= min(p.values)
    assert max(result.values) >= max(p.values)


# ---- widen_param_space ----

def test_widen_param_space_all_groups():
    """Widening with groups=None widens all numeric params."""
    ps = ParamSpace([
        ParamDef("rsi_period", [10, 14, 20], group="signal"),
        ParamDef("sl_mode", ["fixed_pips", "atr_based"], group="risk"),
        ParamDef("sl_fixed_pips", list(range(10, 51, 10)), group="risk"),
    ])
    ws = widen_param_space(ps, 2.0, groups=None)
    # rsi_period should be widened
    assert len(ws.get("rsi_period").values) > 3
    # sl_mode stays same (non-numeric)
    assert ws.get("sl_mode").values == ["fixed_pips", "atr_based"]
    # sl_fixed_pips should be widened
    assert len(ws.get("sl_fixed_pips").values) > 5


def test_widen_param_space_target_groups():
    """Widening specific groups leaves others unchanged."""
    ps = ParamSpace([
        ParamDef("rsi_period", [10, 14, 20], group="signal"),
        ParamDef("sl_fixed_pips", list(range(10, 51, 10)), group="risk"),
    ])
    ws = widen_param_space(ps, 2.0, groups=["signal"])
    # signal group widened
    assert len(ws.get("rsi_period").values) > 3
    # risk group unchanged
    assert ws.get("sl_fixed_pips").values == list(range(10, 51, 10))


def test_widen_param_space_factor_1():
    """Factor 1.0 produces identical space."""
    ps = ParamSpace([
        ParamDef("rsi_period", [10, 14, 20], group="signal"),
    ])
    ws = widen_param_space(ps, 1.0)
    assert ws.get("rsi_period").values == [10, 14, 20]


# ---- group_combos ----

def test_group_combos():
    ps = ParamSpace([
        ParamDef("a", [1, 2, 3], group="signal"),
        ParamDef("b", [10, 20], group="signal"),
        ParamDef("c", [0.5, 1.0], group="risk"),
    ])
    assert group_combos(ps, "signal") == 6  # 3 * 2
    assert group_combos(ps, "risk") == 2    # 2
    assert group_combos(ps, "management") == 1  # no params in group


# ---- compute_fill_factor ----

def test_compute_fill_factor_already_enough():
    """If current combos >= target, factor is 1.0."""
    ps = ParamSpace([
        ParamDef("a", list(range(100)), group="signal"),
    ])
    assert compute_fill_factor(ps, 50, "signal") == 1.0


def test_compute_fill_factor_needs_expansion():
    """Factor > 1.0 when target > current."""
    ps = ParamSpace([
        ParamDef("a", [1, 2, 3], group="signal"),
    ])
    factor = compute_fill_factor(ps, 30, "signal")
    assert factor > 1.0


def test_compute_fill_factor_capped():
    """Factor is capped at 10.0."""
    ps = ParamSpace([
        ParamDef("a", [1, 2], group="signal"),
    ])
    factor = compute_fill_factor(ps, 1_000_000, "signal")
    assert factor <= 10.0


def test_compute_fill_factor_no_numeric():
    """Non-numeric params can't be widened, factor = 1.0."""
    ps = ParamSpace([
        ParamDef("mode", ["a", "b"], group="signal"),
    ])
    factor = compute_fill_factor(ps, 100, "signal")
    assert factor == 1.0


def test_compute_fill_factor_multi_param():
    """Factor is nth root when multiple numeric params exist."""
    ps = ParamSpace([
        ParamDef("a", [1, 2, 3], group="signal"),      # 3 values
        ParamDef("b", [10, 20, 30], group="signal"),    # 3 values
    ])
    # Current combos = 9, target = 81 → ratio = 9 → sqrt(9) = 3.0
    factor = compute_fill_factor(ps, 81, "signal")
    assert abs(factor - 3.0) < 0.01


# ---- Integration test: round-trip widening ----

def test_widen_param_scaling():
    """Widening by 2x roughly doubles the value count for evenly spaced params."""
    p = ParamDef("x", list(range(10, 110, 10)), group="signal")  # 10 values
    result = widen_param(p, 2.0)
    # sqrt(2) ≈ 1.41 finer steps AND extended range → ~2x values
    assert len(result.values) >= 14  # at least 1.4x more


def test_widen_param_large_factor():
    """Large factor (3.0) produces many more values."""
    p = ParamDef("x", [10, 20, 30, 40, 50], group="signal")
    result = widen_param(p, 3.0)
    assert len(result.values) >= 8
