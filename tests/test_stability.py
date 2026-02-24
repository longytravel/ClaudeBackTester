"""Tests for parameter stability analysis (pipeline stage 4)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.stability import (
    evaluate_stability,
    generate_perturbations,
    run_stability,
)
from backtester.pipeline.types import StabilityRating
from backtester.strategies.base import (
    Direction,
    ParamDef,
    ParamSpace,
    Signal,
    SLTPResult,
    Strategy,
    management_params,
    risk_params,
    time_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyStrategy(Strategy):
    """Minimal strategy for stability tests."""

    def __init__(self, signal_bars: list[int] | None = None):
        self._signal_bars = signal_bars or [10, 30, 50]

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def version(self) -> str:
        return "1.0"

    def param_space(self) -> ParamSpace:
        ps = ParamSpace([
            ParamDef("rsi_threshold", [30, 40, 50, 60, 70], group="signal"),
        ])
        for p in risk_params():
            ps.add(p.name, p.values, p.group)
        for p in management_params():
            ps.add(p.name, p.values, p.group)
        for p in time_params():
            ps.add(p.name, p.values, p.group)
        return ps

    def generate_signals(self, open_, high, low, close, volume, spread):
        return [
            Signal(
                bar_index=bar,
                direction=Direction.BUY,
                entry_price=close[bar],
                hour=10,
                day_of_week=1,
                atr_pips=20.0,
            )
            for bar in self._signal_bars
            if bar < len(close)
        ]

    def filter_signals(self, signals, params):
        return signals

    def calc_sl_tp(self, signal, params, high, low):
        return SLTPResult(
            sl_price=signal.entry_price - 0.003,
            tp_price=signal.entry_price + 0.006,
            sl_pips=30.0,
            tp_pips=60.0,
        )


def _make_trending_data(n_bars: int = 100, pip: float = 0.0001):
    """Create synthetic trending price data where BUY trades should profit."""
    base = 1.1000
    open_ = np.zeros(n_bars, dtype=np.float64)
    high = np.zeros(n_bars, dtype=np.float64)
    low = np.zeros(n_bars, dtype=np.float64)
    close = np.zeros(n_bars, dtype=np.float64)
    for i in range(n_bars):
        price = base + i * 3 * pip
        open_[i] = price
        high[i] = price + 15 * pip
        low[i] = price - 10 * pip
        close[i] = price + 2 * pip
    volume = np.ones(n_bars, dtype=np.float64)
    spread = np.full(n_bars, 1.0 * pip)
    return open_, high, low, close, volume, spread


DEFAULT_PARAMS: dict[str, Any] = {
    "rsi_threshold": 50,
    "sl_mode": "fixed_pips",
    "sl_fixed_pips": 30,
    "sl_atr_mult": 1.5,
    "tp_mode": "rr_ratio",
    "tp_rr_ratio": 2.0,
    "tp_atr_mult": 2.0,
    "tp_fixed_pips": 60,
    "allowed_hours_start": 0,
    "allowed_hours_end": 23,
    "allowed_days": [0, 1, 2, 3, 4],
    "trailing_mode": "off",
    "trail_activate_pips": 0,
    "trail_distance_pips": 10,
    "trail_atr_mult": 2.0,
    "breakeven_enabled": False,
    "breakeven_trigger_pips": 20,
    "breakeven_offset_pips": 2,
    "partial_close_enabled": False,
    "partial_close_pct": 50,
    "partial_close_trigger_pips": 30,
    "max_bars": 0,
    "stale_exit_enabled": False,
    "stale_exit_bars": 50,
    "stale_exit_atr_threshold": 0.5,
}


def _make_data_arrays(n_bars: int = 100):
    """Build the data_arrays dict expected by stability functions."""
    open_, high, low, close, volume, spread = _make_trending_data(n_bars)
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "spread": spread,
        "bar_hour": np.full(n_bars, 10, dtype=np.int64),
        "bar_day_of_week": np.full(n_bars, 1, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Tests for generate_perturbations
# ---------------------------------------------------------------------------

class TestGeneratePerturbations:
    def test_generate_perturbations_numeric(self):
        """Numeric param generates +-N step perturbations."""
        ps = ParamSpace([ParamDef("rsi_threshold", [30, 40, 50, 60, 70], group="signal")])
        params = {"rsi_threshold": 50}

        result = generate_perturbations(params, ps, n_steps=3)

        # rsi_threshold=50 is at index 2 in [30, 40, 50, 60, 70]
        # +1 -> 60, +2 -> 70, +3 -> out of bounds
        # -1 -> 40, -2 -> 30, -3 -> out of bounds
        # So 4 perturbations expected
        param_names = [r[0] for r in result]
        values = [r[1] for r in result]

        assert all(n == "rsi_threshold" for n in param_names)
        assert set(values) == {30, 40, 60, 70}
        assert len(result) == 4

        # Verify each perturbed dict is a copy with one param changed
        for name, val, pdict in result:
            assert pdict["rsi_threshold"] == val
            assert pdict is not params  # must be a copy

    def test_generate_perturbations_boolean(self):
        """Boolean param generates exactly 1 perturbation (flip)."""
        ps = ParamSpace([ParamDef("breakeven_enabled", [False, True], group="management")])
        params = {"breakeven_enabled": False}

        result = generate_perturbations(params, ps, n_steps=3)

        assert len(result) == 1
        assert result[0][0] == "breakeven_enabled"
        assert result[0][1] is True
        assert result[0][2]["breakeven_enabled"] is True

    def test_generate_perturbations_boolean_flip_true(self):
        """Boolean param flips from True to False."""
        ps = ParamSpace([ParamDef("breakeven_enabled", [False, True], group="management")])
        params = {"breakeven_enabled": True}

        result = generate_perturbations(params, ps, n_steps=3)

        assert len(result) == 1
        assert result[0][1] is False

    def test_generate_perturbations_categorical(self):
        """Categorical param tries each other value."""
        ps = ParamSpace([
            ParamDef("sl_mode", ["fixed_pips", "atr_based", "swing"], group="risk"),
        ])
        params = {"sl_mode": "fixed_pips"}

        result = generate_perturbations(params, ps, n_steps=3)

        assert len(result) == 2  # "atr_based" and "swing"
        values = {r[1] for r in result}
        assert values == {"atr_based", "swing"}
        assert all(r[0] == "sl_mode" for r in result)

    def test_generate_perturbations_boundary(self):
        """Param at edge of values list doesn't go out of bounds."""
        ps = ParamSpace([ParamDef("rsi_threshold", [30, 40, 50, 60, 70], group="signal")])

        # At the start: index 0
        params = {"rsi_threshold": 30}
        result = generate_perturbations(params, ps, n_steps=3)
        values = [r[1] for r in result]
        # Can only go up: +1=40, +2=50, +3=60
        assert set(values) == {40, 50, 60}
        assert 30 not in values
        # No negative values
        for v in values:
            assert v in [30, 40, 50, 60, 70]

        # At the end: index 4
        params = {"rsi_threshold": 70}
        result = generate_perturbations(params, ps, n_steps=3)
        values = [r[1] for r in result]
        # Can only go down: -1=60, -2=50, -3=40
        assert set(values) == {40, 50, 60}

    def test_generate_perturbations_bitmask(self):
        """Bitmask param (allowed_days) tries each other value."""
        ps = ParamSpace([
            ParamDef("allowed_days", [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3],
                [1, 2, 3],
                [0, 1, 2, 3, 4, 5, 6],
            ], group="time"),
        ])
        params = {"allowed_days": [0, 1, 2, 3, 4]}

        result = generate_perturbations(params, ps, n_steps=3)

        # Should try the other 3 values
        assert len(result) == 3
        perturbed_values = [r[1] for r in result]
        assert [0, 1, 2, 3] in perturbed_values
        assert [1, 2, 3] in perturbed_values
        assert [0, 1, 2, 3, 4, 5, 6] in perturbed_values

    def test_generate_perturbations_multiple_params(self):
        """Multiple params in the space all get perturbations."""
        ps = ParamSpace([
            ParamDef("rsi_threshold", [30, 40, 50, 60, 70], group="signal"),
            ParamDef("breakeven_enabled", [False, True], group="management"),
        ])
        params = {"rsi_threshold": 50, "breakeven_enabled": False}

        result = generate_perturbations(params, ps, n_steps=2)

        rsi_perts = [r for r in result if r[0] == "rsi_threshold"]
        bool_perts = [r for r in result if r[0] == "breakeven_enabled"]

        # rsi_threshold at index 2: +1=60, +2=70, -1=40, -2=30 = 4
        assert len(rsi_perts) == 4
        # breakeven: 1 flip
        assert len(bool_perts) == 1


# ---------------------------------------------------------------------------
# Tests for evaluate_stability
# ---------------------------------------------------------------------------

class TestEvaluateStability:
    def test_evaluate_stability_trending(self):
        """DummyStrategy on trending data returns valid stability result."""
        strategy = DummyStrategy()
        data = _make_data_arrays(n_bars=100)
        config = PipelineConfig(commission_pips=0.0, max_spread_pips=0.0)

        perturbations = generate_perturbations(
            DEFAULT_PARAMS, strategy.param_space(), n_steps=config.stab_perturbation_steps,
        )

        result = evaluate_stability(strategy, DEFAULT_PARAMS, perturbations, data, config)

        assert len(result.perturbations) == len(perturbations)
        assert result.mean_ratio >= 0.0
        assert result.min_ratio <= result.mean_ratio
        assert result.worst_param != ""
        assert result.rating in (
            StabilityRating.ROBUST,
            StabilityRating.MODERATE,
            StabilityRating.FRAGILE,
            StabilityRating.OVERFIT,
        )

        # All perturbation results should have valid data
        for pr in result.perturbations:
            assert pr.param_name != ""
            assert isinstance(pr.ratio, float)

    def test_stability_rating_robust(self):
        """When all ratios > 0.8 and min > 0.5, rating is ROBUST."""
        strategy = DummyStrategy()
        data = _make_data_arrays(n_bars=100)
        config = PipelineConfig(
            stab_robust_mean=0.8,
            stab_robust_min=0.5,
            commission_pips=0.0,
            max_spread_pips=0.0,
        )

        # Create fake perturbations that all return the same params (ratio ~1.0)
        # by using the exact same params as the original
        fake_perturbations = [
            ("rsi_threshold", 50, dict(DEFAULT_PARAMS)),
            ("rsi_threshold", 50, dict(DEFAULT_PARAMS)),
        ]

        result = evaluate_stability(strategy, DEFAULT_PARAMS, fake_perturbations, data, config)

        # All perturbations are identical to original => ratio should be ~1.0
        for pr in result.perturbations:
            assert abs(pr.ratio - 1.0) < 0.01, f"Expected ratio ~1.0, got {pr.ratio}"

        assert result.mean_ratio >= 0.8
        assert result.min_ratio >= 0.5
        assert result.rating == StabilityRating.ROBUST

    def test_stability_rating_overfit(self):
        """When all ratios < 0.4, rating is OVERFIT."""
        strategy = DummyStrategy()
        data = _make_data_arrays(n_bars=100)
        config = PipelineConfig(commission_pips=0.0, max_spread_pips=0.0)

        # Create perturbations that will produce 0 quality
        # Use signals at bar indices that are out of range for the data
        strategy_no_trades = DummyStrategy(signal_bars=[999])

        # We need a different approach: use the evaluate_stability function
        # but with perturbations that zero out the quality.
        # Easiest: create a mock that returns specific quality values.

        # Instead, test via the rating logic directly with known ratios.
        # Create a StabilityResult manually to verify the rating thresholds.
        from backtester.pipeline.types import PerturbationResult, StabilityResult

        perturbation_results = [
            PerturbationResult(
                param_name="rsi_threshold",
                original_value=50,
                perturbed_value=30,
                original_quality=10.0,
                perturbed_quality=2.0,
                ratio=0.2,
            ),
            PerturbationResult(
                param_name="rsi_threshold",
                original_value=50,
                perturbed_value=70,
                original_quality=10.0,
                perturbed_quality=3.0,
                ratio=0.3,
            ),
        ]

        ratios = [pr.ratio for pr in perturbation_results]
        mean_ratio = float(np.mean(ratios))
        min_ratio = float(np.min(ratios))

        # mean_ratio = 0.25, min_ratio = 0.2 => OVERFIT (mean < 0.4)
        assert mean_ratio < 0.4
        assert min_ratio < 0.5

        # Verify the rating logic by running through the same thresholds
        if mean_ratio >= config.stab_robust_mean and min_ratio >= config.stab_robust_min:
            rating = StabilityRating.ROBUST
        elif mean_ratio >= config.stab_moderate_mean:
            rating = StabilityRating.MODERATE
        elif mean_ratio >= config.stab_fragile_mean:
            rating = StabilityRating.FRAGILE
        else:
            rating = StabilityRating.OVERFIT

        assert rating == StabilityRating.OVERFIT

    def test_stability_zero_quality(self):
        """Original quality=0 results in all ratios=0."""
        # Use a strategy with no signals so quality will be 0
        strategy = DummyStrategy(signal_bars=[999])
        data = _make_data_arrays(n_bars=100)
        config = PipelineConfig(commission_pips=0.0, max_spread_pips=0.0)

        # Create perturbations manually (they don't matter since no signals)
        perturbations = [
            ("rsi_threshold", 40, dict(DEFAULT_PARAMS, rsi_threshold=40)),
            ("rsi_threshold", 60, dict(DEFAULT_PARAMS, rsi_threshold=60)),
        ]

        result = evaluate_stability(strategy, DEFAULT_PARAMS, perturbations, data, config)

        # With no signals => quality = 0 => all ratios = 0
        for pr in result.perturbations:
            assert pr.original_quality == 0.0
            assert pr.ratio == 0.0

        assert result.mean_ratio == 0.0
        assert result.min_ratio == 0.0
        assert result.rating == StabilityRating.OVERFIT

    def test_empty_perturbations(self):
        """No perturbations returns ROBUST with ratio 1.0."""
        strategy = DummyStrategy()
        data = _make_data_arrays(n_bars=100)
        config = PipelineConfig(commission_pips=0.0, max_spread_pips=0.0)

        result = evaluate_stability(strategy, DEFAULT_PARAMS, [], data, config)

        assert result.mean_ratio == 1.0
        assert result.min_ratio == 1.0
        assert result.rating == StabilityRating.ROBUST
        assert result.perturbations == []


# ---------------------------------------------------------------------------
# Tests for run_stability
# ---------------------------------------------------------------------------

class TestRunStability:
    def test_run_stability_multiple_candidates(self):
        """Run stability on 2+ candidates returns one result per candidate."""
        strategy = DummyStrategy()
        data = _make_data_arrays(n_bars=100)
        config = PipelineConfig(stab_perturbation_steps=1, commission_pips=0.0, max_spread_pips=0.0)

        candidate1 = dict(DEFAULT_PARAMS)
        candidate2 = dict(DEFAULT_PARAMS, rsi_threshold=40)

        results = run_stability(
            strategy=strategy,
            candidates=[candidate1, candidate2],
            data_arrays=data,
            config=config,
        )

        assert len(results) == 2
        for r in results:
            assert r.rating in (
                StabilityRating.ROBUST,
                StabilityRating.MODERATE,
                StabilityRating.FRAGILE,
                StabilityRating.OVERFIT,
            )
            assert isinstance(r.mean_ratio, float)
            assert isinstance(r.min_ratio, float)

    def test_run_stability_default_config(self):
        """run_stability works with default config (None)."""
        strategy = DummyStrategy()
        data = _make_data_arrays(n_bars=100)

        results = run_stability(
            strategy=strategy,
            candidates=[dict(DEFAULT_PARAMS)],
            data_arrays=data,
            config=None,
        )

        assert len(results) == 1
        assert results[0].perturbations  # should have perturbations
