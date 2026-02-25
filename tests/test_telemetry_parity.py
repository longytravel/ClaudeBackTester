"""Telemetry vs Batch Evaluator parity tests.

Ensures run_telemetry() produces identical aggregate metrics to
engine.evaluate_single() for EXEC_BASIC and EXEC_FULL modes.

This catches regressions when either the Rust batch evaluator or
the Python telemetry engine is modified independently.
"""

import numpy as np
import pytest

from backtester.core.dtypes import EXEC_BASIC, EXEC_FULL
from backtester.core.engine import BacktestEngine
from backtester.core.telemetry import run_telemetry
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
# Test strategy: many signals in volatile data to exercise all exit paths
# ---------------------------------------------------------------------------

class VolatileStrategy(Strategy):
    """Generates BUY and SELL signals every N bars in volatile data."""

    def __init__(self, every_n: int = 5):
        self._every_n = every_n

    @property
    def name(self) -> str:
        return "volatile_test"

    @property
    def version(self) -> str:
        return "1.0"

    def param_space(self) -> ParamSpace:
        ps = ParamSpace([
            ParamDef("dummy_param", [1], group="signal"),
        ])
        for p in risk_params():
            ps.add(p.name, p.values, p.group)
        for p in management_params():
            ps.add(p.name, p.values, p.group)
        for p in time_params():
            ps.add(p.name, p.values, p.group)
        return ps

    def generate_signals(self, open_, high, low, close, volume, spread):
        signals = []
        for bar in range(10, len(close) - 10, self._every_n):
            direction = Direction.BUY if bar % 2 == 0 else Direction.SELL
            signals.append(Signal(
                bar_index=bar,
                direction=direction,
                entry_price=close[bar],
                hour=bar % 24,
                day_of_week=bar % 5,
                atr_pips=15.0,
            ))
        return signals

    def filter_signals(self, signals, params):
        return signals

    def calc_sl_tp(self, signal, params, high, low):
        return SLTPResult(sl_price=0, tp_price=0, sl_pips=0, tp_pips=0)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _make_volatile_data(n_bars: int = 500, pip: float = 0.0001, seed: int = 42):
    """Volatile price data with trends and reversals to trigger all exit types."""
    rng = np.random.default_rng(seed)
    base = 1.1000
    returns = rng.normal(0, 8 * pip, n_bars)  # ~8 pip std dev per bar
    close = np.cumsum(returns) + base
    high = close + rng.uniform(3 * pip, 20 * pip, n_bars)
    low = close - rng.uniform(3 * pip, 20 * pip, n_bars)
    open_ = close - returns / 2
    volume = np.ones(n_bars, dtype=np.float64)
    spread = np.full(n_bars, 1.0 * pip)
    return open_, high, low, close, volume, spread


def _make_engine(n_bars=500, every_n=5, seed=42):
    """Create engine with volatile strategy."""
    data = _make_volatile_data(n_bars=n_bars, seed=seed)
    open_, high, low, close, volume, spread = data
    bar_hour = np.arange(n_bars, dtype=np.int64) % 24
    bar_dow = np.arange(n_bars, dtype=np.int64) % 5

    strategy = VolatileStrategy(every_n=every_n)
    engine = BacktestEngine(
        strategy, open_, high, low, close, volume, spread,
        pip_value=0.0001, slippage_pips=0.5,
        commission_pips=0.7, max_spread_pips=3.0,
        bar_hour=bar_hour, bar_day_of_week=bar_dow,
    )
    return engine


METRIC_NAMES = [
    "trades", "win_rate", "profit_factor", "sharpe", "sortino",
    "max_dd_pct", "return_pct", "r_squared", "ulcer", "quality_score",
]


def _assert_metrics_match(batch_metrics, telem_result, label=""):
    """Assert all 10 metrics match between batch and telemetry."""
    for name in METRIC_NAMES:
        b = batch_metrics[name]
        t = telem_result.metrics.get(name, 0.0)
        np.testing.assert_allclose(
            t, b, atol=1e-6, rtol=1e-6,
            err_msg=f"{label} metric '{name}' mismatch: batch={b}, telemetry={t}",
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestExecBasicParity:
    """Telemetry matches batch evaluator for EXEC_BASIC."""

    def test_basic_default_params(self):
        engine = _make_engine()
        params = {
            "dummy_param": 1,
            "sl_mode": "fixed_pips", "sl_fixed_pips": 30,
            "sl_atr_mult": 1.5,
            "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0,
            "tp_atr_mult": 2.0, "tp_fixed_pips": 60,
            "allowed_hours_start": 0, "allowed_hours_end": 23,
            "allowed_days": [0, 1, 2, 3, 4],
            "trailing_mode": "off", "trail_activate_pips": 0,
            "trail_distance_pips": 10, "trail_atr_mult": 2.0,
            "breakeven_enabled": False, "breakeven_trigger_pips": 20,
            "breakeven_offset_pips": 2,
            "partial_close_enabled": False, "partial_close_pct": 50,
            "partial_close_trigger_pips": 30,
            "max_bars": 0, "stale_exit_enabled": False,
            "stale_exit_bars": 50, "stale_exit_atr_threshold": 0.5,
        }
        batch = engine.evaluate_single(params, exec_mode=EXEC_BASIC)
        telem = run_telemetry(engine, params, exec_mode=EXEC_BASIC)
        assert batch["trades"] > 0, "Need trades to test parity"
        assert len(telem.trades) == int(batch["trades"])
        _assert_metrics_match(batch, telem, "EXEC_BASIC default")

    def test_basic_atr_sl_fixed_tp(self):
        engine = _make_engine()
        params = {
            "dummy_param": 1,
            "sl_mode": "atr_based", "sl_fixed_pips": 30,
            "sl_atr_mult": 2.0,
            "tp_mode": "fixed_pips", "tp_rr_ratio": 2.0,
            "tp_atr_mult": 2.0, "tp_fixed_pips": 40,
            "allowed_hours_start": 0, "allowed_hours_end": 23,
            "allowed_days": [0, 1, 2, 3, 4],
            "trailing_mode": "off", "trail_activate_pips": 0,
            "trail_distance_pips": 10, "trail_atr_mult": 2.0,
            "breakeven_enabled": False, "breakeven_trigger_pips": 20,
            "breakeven_offset_pips": 2,
            "partial_close_enabled": False, "partial_close_pct": 50,
            "partial_close_trigger_pips": 30,
            "max_bars": 0, "stale_exit_enabled": False,
            "stale_exit_bars": 50, "stale_exit_atr_threshold": 0.5,
        }
        batch = engine.evaluate_single(params, exec_mode=EXEC_BASIC)
        telem = run_telemetry(engine, params, exec_mode=EXEC_BASIC)
        assert batch["trades"] > 0
        _assert_metrics_match(batch, telem, "EXEC_BASIC atr_sl")


class TestExecFullSimpleParity:
    """Telemetry matches batch for EXEC_FULL with trailing + BE."""

    def test_trailing_fixed_pip(self):
        engine = _make_engine()
        params = {
            "dummy_param": 1,
            "sl_mode": "fixed_pips", "sl_fixed_pips": 25,
            "sl_atr_mult": 1.5,
            "tp_mode": "rr_ratio", "tp_rr_ratio": 3.0,
            "tp_atr_mult": 2.0, "tp_fixed_pips": 60,
            "allowed_hours_start": 0, "allowed_hours_end": 23,
            "allowed_days": [0, 1, 2, 3, 4],
            "trailing_mode": "fixed_pip", "trail_activate_pips": 10,
            "trail_distance_pips": 8, "trail_atr_mult": 2.0,
            "breakeven_enabled": False, "breakeven_trigger_pips": 20,
            "breakeven_offset_pips": 2,
            "partial_close_enabled": False, "partial_close_pct": 50,
            "partial_close_trigger_pips": 30,
            "max_bars": 0, "stale_exit_enabled": False,
            "stale_exit_bars": 50, "stale_exit_atr_threshold": 0.5,
        }
        batch = engine.evaluate_single(params, exec_mode=EXEC_FULL)
        telem = run_telemetry(engine, params, exec_mode=EXEC_FULL)
        assert batch["trades"] > 0
        _assert_metrics_match(batch, telem, "EXEC_FULL trailing_fixed")

    def test_breakeven_only(self):
        engine = _make_engine()
        params = {
            "dummy_param": 1,
            "sl_mode": "fixed_pips", "sl_fixed_pips": 25,
            "sl_atr_mult": 1.5,
            "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0,
            "tp_atr_mult": 2.0, "tp_fixed_pips": 60,
            "allowed_hours_start": 0, "allowed_hours_end": 23,
            "allowed_days": [0, 1, 2, 3, 4],
            "trailing_mode": "off", "trail_activate_pips": 0,
            "trail_distance_pips": 10, "trail_atr_mult": 2.0,
            "breakeven_enabled": True, "breakeven_trigger_pips": 10,
            "breakeven_offset_pips": 2,
            "partial_close_enabled": False, "partial_close_pct": 50,
            "partial_close_trigger_pips": 30,
            "max_bars": 0, "stale_exit_enabled": False,
            "stale_exit_bars": 50, "stale_exit_atr_threshold": 0.5,
        }
        batch = engine.evaluate_single(params, exec_mode=EXEC_FULL)
        telem = run_telemetry(engine, params, exec_mode=EXEC_FULL)
        assert batch["trades"] > 0
        _assert_metrics_match(batch, telem, "EXEC_FULL be_only")

    def test_trailing_and_breakeven(self):
        engine = _make_engine()
        params = {
            "dummy_param": 1,
            "sl_mode": "atr_based", "sl_fixed_pips": 30,
            "sl_atr_mult": 2.0,
            "tp_mode": "rr_ratio", "tp_rr_ratio": 2.5,
            "tp_atr_mult": 2.0, "tp_fixed_pips": 60,
            "allowed_hours_start": 0, "allowed_hours_end": 23,
            "allowed_days": [0, 1, 2, 3, 4],
            "trailing_mode": "atr_chandelier", "trail_activate_pips": 12,
            "trail_distance_pips": 10, "trail_atr_mult": 1.5,
            "breakeven_enabled": True, "breakeven_trigger_pips": 8,
            "breakeven_offset_pips": 2,
            "partial_close_enabled": False, "partial_close_pct": 50,
            "partial_close_trigger_pips": 30,
            "max_bars": 0, "stale_exit_enabled": False,
            "stale_exit_bars": 50, "stale_exit_atr_threshold": 0.5,
        }
        batch = engine.evaluate_single(params, exec_mode=EXEC_FULL)
        telem = run_telemetry(engine, params, exec_mode=EXEC_FULL)
        assert batch["trades"] > 0
        _assert_metrics_match(batch, telem, "EXEC_FULL trail+be")


class TestExecFullComplexParity:
    """Telemetry matches batch for EXEC_FULL with ALL features active."""

    def test_all_features_active(self):
        engine = _make_engine(n_bars=800)
        params = {
            "dummy_param": 1,
            "sl_mode": "fixed_pips", "sl_fixed_pips": 25,
            "sl_atr_mult": 1.5,
            "tp_mode": "rr_ratio", "tp_rr_ratio": 3.0,
            "tp_atr_mult": 2.0, "tp_fixed_pips": 60,
            "allowed_hours_start": 0, "allowed_hours_end": 23,
            "allowed_days": [0, 1, 2, 3, 4],
            "trailing_mode": "fixed_pip", "trail_activate_pips": 12,
            "trail_distance_pips": 8, "trail_atr_mult": 2.0,
            "breakeven_enabled": True, "breakeven_trigger_pips": 8,
            "breakeven_offset_pips": 2,
            "partial_close_enabled": True, "partial_close_pct": 50,
            "partial_close_trigger_pips": 15,
            "max_bars": 100, "stale_exit_enabled": True,
            "stale_exit_bars": 30, "stale_exit_atr_threshold": 0.3,
        }
        batch = engine.evaluate_single(params, exec_mode=EXEC_FULL)
        telem = run_telemetry(engine, params, exec_mode=EXEC_FULL)
        assert batch["trades"] > 0
        assert len(telem.trades) == int(batch["trades"])
        _assert_metrics_match(batch, telem, "EXEC_FULL all_features")

    def test_all_features_atr_chandelier(self):
        engine = _make_engine(n_bars=800, seed=123)
        params = {
            "dummy_param": 1,
            "sl_mode": "atr_based", "sl_fixed_pips": 30,
            "sl_atr_mult": 2.5,
            "tp_mode": "rr_ratio", "tp_rr_ratio": 2.5,
            "tp_atr_mult": 2.0, "tp_fixed_pips": 60,
            "allowed_hours_start": 0, "allowed_hours_end": 23,
            "allowed_days": [0, 1, 2, 3, 4],
            "trailing_mode": "atr_chandelier", "trail_activate_pips": 15,
            "trail_distance_pips": 10, "trail_atr_mult": 1.5,
            "breakeven_enabled": True, "breakeven_trigger_pips": 10,
            "breakeven_offset_pips": 3,
            "partial_close_enabled": True, "partial_close_pct": 40,
            "partial_close_trigger_pips": 20,
            "max_bars": 80, "stale_exit_enabled": True,
            "stale_exit_bars": 25, "stale_exit_atr_threshold": 0.4,
        }
        batch = engine.evaluate_single(params, exec_mode=EXEC_FULL)
        telem = run_telemetry(engine, params, exec_mode=EXEC_FULL)
        assert batch["trades"] > 0
        _assert_metrics_match(batch, telem, "EXEC_FULL all_features_atr")

    def test_max_bars_exit(self):
        """Short max_bars to force max_bars exits."""
        engine = _make_engine(n_bars=500)
        params = {
            "dummy_param": 1,
            "sl_mode": "fixed_pips", "sl_fixed_pips": 50,
            "sl_atr_mult": 1.5,
            "tp_mode": "rr_ratio", "tp_rr_ratio": 4.0,
            "tp_atr_mult": 2.0, "tp_fixed_pips": 60,
            "allowed_hours_start": 0, "allowed_hours_end": 23,
            "allowed_days": [0, 1, 2, 3, 4],
            "trailing_mode": "off", "trail_activate_pips": 0,
            "trail_distance_pips": 10, "trail_atr_mult": 2.0,
            "breakeven_enabled": False, "breakeven_trigger_pips": 20,
            "breakeven_offset_pips": 2,
            "partial_close_enabled": False, "partial_close_pct": 50,
            "partial_close_trigger_pips": 30,
            "max_bars": 5, "stale_exit_enabled": False,
            "stale_exit_bars": 50, "stale_exit_atr_threshold": 0.5,
        }
        batch = engine.evaluate_single(params, exec_mode=EXEC_FULL)
        telem = run_telemetry(engine, params, exec_mode=EXEC_FULL)
        assert batch["trades"] > 0
        _assert_metrics_match(batch, telem, "EXEC_FULL max_bars")

    def test_stale_exit(self):
        """Low stale threshold to force stale exits."""
        engine = _make_engine(n_bars=500)
        params = {
            "dummy_param": 1,
            "sl_mode": "fixed_pips", "sl_fixed_pips": 50,
            "sl_atr_mult": 1.5,
            "tp_mode": "rr_ratio", "tp_rr_ratio": 4.0,
            "tp_atr_mult": 2.0, "tp_fixed_pips": 60,
            "allowed_hours_start": 0, "allowed_hours_end": 23,
            "allowed_days": [0, 1, 2, 3, 4],
            "trailing_mode": "off", "trail_activate_pips": 0,
            "trail_distance_pips": 10, "trail_atr_mult": 2.0,
            "breakeven_enabled": False, "breakeven_trigger_pips": 20,
            "breakeven_offset_pips": 2,
            "partial_close_enabled": False, "partial_close_pct": 50,
            "partial_close_trigger_pips": 30,
            "max_bars": 0, "stale_exit_enabled": True,
            "stale_exit_bars": 3, "stale_exit_atr_threshold": 2.0,
        }
        batch = engine.evaluate_single(params, exec_mode=EXEC_FULL)
        telem = run_telemetry(engine, params, exec_mode=EXEC_FULL)
        assert batch["trades"] > 0
        _assert_metrics_match(batch, telem, "EXEC_FULL stale")


class TestTradeCountMatch:
    """Number of trades from telemetry matches batch evaluator exactly."""

    def test_trade_count_basic(self):
        engine = _make_engine()
        params = {
            "dummy_param": 1,
            "sl_mode": "fixed_pips", "sl_fixed_pips": 20,
            "sl_atr_mult": 1.5,
            "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0,
            "tp_atr_mult": 2.0, "tp_fixed_pips": 60,
            "allowed_hours_start": 0, "allowed_hours_end": 23,
            "allowed_days": [0, 1, 2, 3, 4],
            "trailing_mode": "off", "trail_activate_pips": 0,
            "trail_distance_pips": 10, "trail_atr_mult": 2.0,
            "breakeven_enabled": False, "breakeven_trigger_pips": 20,
            "breakeven_offset_pips": 2,
            "partial_close_enabled": False, "partial_close_pct": 50,
            "partial_close_trigger_pips": 30,
            "max_bars": 0, "stale_exit_enabled": False,
            "stale_exit_bars": 50, "stale_exit_atr_threshold": 0.5,
        }
        batch = engine.evaluate_single(params, exec_mode=EXEC_BASIC)
        telem = run_telemetry(engine, params, exec_mode=EXEC_BASIC)
        assert len(telem.trades) == int(batch["trades"])

    def test_trade_count_full(self):
        engine = _make_engine(n_bars=800)
        params = {
            "dummy_param": 1,
            "sl_mode": "fixed_pips", "sl_fixed_pips": 20,
            "sl_atr_mult": 1.5,
            "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0,
            "tp_atr_mult": 2.0, "tp_fixed_pips": 60,
            "allowed_hours_start": 0, "allowed_hours_end": 23,
            "allowed_days": [0, 1, 2, 3, 4],
            "trailing_mode": "fixed_pip", "trail_activate_pips": 10,
            "trail_distance_pips": 8, "trail_atr_mult": 2.0,
            "breakeven_enabled": True, "breakeven_trigger_pips": 8,
            "breakeven_offset_pips": 2,
            "partial_close_enabled": True, "partial_close_pct": 50,
            "partial_close_trigger_pips": 15,
            "max_bars": 100, "stale_exit_enabled": True,
            "stale_exit_bars": 30, "stale_exit_atr_threshold": 0.3,
        }
        batch = engine.evaluate_single(params, exec_mode=EXEC_FULL)
        telem = run_telemetry(engine, params, exec_mode=EXEC_FULL)
        assert len(telem.trades) == int(batch["trades"])
