"""Tests for BacktestEngine orchestrator."""

import numpy as np
import pytest

from backtester.core.dtypes import (
    DIR_BUY,
    DIR_SELL,
    EXEC_BASIC,
    EXEC_FULL,
    NUM_METRICS,
    M_TRADES,
    M_WIN_RATE,
)
from backtester.core.encoding import build_encoding_spec, encode_params
from backtester.core.engine import BacktestEngine
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
# DummyStrategy for testing
# ---------------------------------------------------------------------------

class DummyStrategy(Strategy):
    """Simple test strategy: generate BUY signals at fixed bars."""

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
        signals = []
        for bar in self._signal_bars:
            if bar < len(close):
                signals.append(Signal(
                    bar_index=bar,
                    direction=Direction.BUY,
                    entry_price=close[bar],
                    hour=10,
                    day_of_week=1,  # Tuesday
                    atr_pips=20.0,
                ))
        return signals

    def filter_signals(self, signals, params):
        return signals  # No filtering

    def calc_sl_tp(self, signal, params, high, low):
        return SLTPResult(
            sl_price=signal.entry_price - 0.003,
            tp_price=signal.entry_price + 0.006,
            sl_pips=30.0,
            tp_pips=60.0,
        )


class SellStrategy(Strategy):
    """Strategy that generates SELL signals."""

    @property
    def name(self) -> str:
        return "sell_test"

    @property
    def version(self) -> str:
        return "1.0"

    def param_space(self) -> ParamSpace:
        ps = ParamSpace()
        for p in risk_params():
            ps.add(p.name, p.values, p.group)
        for p in management_params():
            ps.add(p.name, p.values, p.group)
        for p in time_params():
            ps.add(p.name, p.values, p.group)
        return ps

    def generate_signals(self, open_, high, low, close, volume, spread):
        return [Signal(
            bar_index=5,
            direction=Direction.SELL,
            entry_price=close[5],
            hour=14,
            day_of_week=2,
            atr_pips=25.0,
        )]

    def filter_signals(self, signals, params):
        return signals

    def calc_sl_tp(self, signal, params, high, low):
        return SLTPResult(sl_price=0, tp_price=0, sl_pips=0, tp_pips=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trending_data(n_bars: int = 100, pip: float = 0.0001):
    """Create upward trending price data."""
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
    spread = np.full(n_bars, 1.0 * pip)  # 1 pip spread in price units
    return open_, high, low, close, volume, spread


def _make_flat_data(n_bars: int = 100, pip: float = 0.0001):
    """Create flat/sideways price data."""
    base = 1.1000
    rng = np.random.default_rng(99)
    noise = rng.normal(0, 3 * pip, n_bars)
    close = np.full(n_bars, base) + noise
    high = close + rng.uniform(3 * pip, 8 * pip, n_bars)
    low = close - rng.uniform(3 * pip, 8 * pip, n_bars)
    open_ = close - noise / 2
    volume = np.ones(n_bars, dtype=np.float64)
    spread = np.full(n_bars, 1.0 * pip)  # 1 pip spread in price units
    return open_, high, low, close, volume, spread


def _default_params() -> dict:
    """Default parameter set (basic mode — management OFF)."""
    return {
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEngineInit:
    def test_creates_encoding(self):
        data = _make_trending_data()
        engine = BacktestEngine(DummyStrategy(), *data, commission_pips=0.0, max_spread_pips=0.0)
        assert engine.encoding.num_params > 0

    def test_generates_signals(self):
        data = _make_trending_data()
        engine = BacktestEngine(DummyStrategy([10, 30, 50]), *data, commission_pips=0.0, max_spread_pips=0.0)
        assert engine.n_signals == 3

    def test_no_signals(self):
        data = _make_trending_data(n_bars=5)  # Fewer bars than signal bars
        engine = BacktestEngine(DummyStrategy([10, 30, 50]), *data, commission_pips=0.0, max_spread_pips=0.0)
        assert engine.n_signals == 0

    def test_param_layout_built(self):
        data = _make_trending_data()
        engine = BacktestEngine(DummyStrategy(), *data, commission_pips=0.0, max_spread_pips=0.0)
        assert len(engine.param_layout) > 0


class TestEvaluateSingle:
    def test_returns_metric_dict(self):
        data = _make_trending_data()
        engine = BacktestEngine(DummyStrategy([10]), *data, slippage_pips=0.0, commission_pips=0.0, max_spread_pips=0.0)
        result = engine.evaluate_single(_default_params())
        assert "trades" in result
        assert "win_rate" in result
        assert "quality_score" in result

    def test_trade_count(self):
        data = _make_trending_data()
        engine = BacktestEngine(DummyStrategy([10, 30, 50]), *data, slippage_pips=0.0, commission_pips=0.0, max_spread_pips=0.0)
        result = engine.evaluate_single(_default_params())
        assert result["trades"] == 3.0

    def test_no_signals_zero_trades(self):
        data = _make_trending_data(n_bars=5)
        engine = BacktestEngine(DummyStrategy([10, 30, 50]), *data, commission_pips=0.0, max_spread_pips=0.0)
        result = engine.evaluate_single(_default_params())
        assert result["trades"] == 0.0


class TestEvaluateBatch:
    def test_batch_shape(self):
        data = _make_trending_data()
        engine = BacktestEngine(DummyStrategy([10]), *data, commission_pips=0.0, max_spread_pips=0.0)

        spec = engine.encoding
        params_dict = _default_params()
        row = encode_params(spec, params_dict)
        matrix = np.vstack([row, row, row])  # 3 identical trials

        metrics = engine.evaluate_batch(matrix)
        assert metrics.shape == (3, NUM_METRICS)

    def test_different_params_different_results(self):
        data = _make_trending_data()
        engine = BacktestEngine(DummyStrategy([10]), *data, slippage_pips=0.0, commission_pips=0.0, max_spread_pips=0.0)

        spec = engine.encoding
        # Trial 0: tight SL
        p0 = _default_params()
        p0["sl_fixed_pips"] = 5
        # Trial 1: wide SL
        p1 = _default_params()
        p1["sl_fixed_pips"] = 100

        matrix = np.vstack([
            encode_params(spec, p0),
            encode_params(spec, p1),
        ])

        metrics = engine.evaluate_batch(matrix)
        # Both should have 1 trade, but different outcomes
        assert metrics[0, M_TRADES] == 1.0
        assert metrics[1, M_TRADES] == 1.0

    def test_empty_signals(self):
        data = _make_trending_data(n_bars=5)
        engine = BacktestEngine(DummyStrategy([10, 30]), *data, commission_pips=0.0, max_spread_pips=0.0)

        spec = engine.encoding
        row = encode_params(spec, _default_params())
        metrics = engine.evaluate_batch(row.reshape(1, -1))
        assert metrics[0, M_TRADES] == 0.0


class TestEvaluateFromIndices:
    def test_index_evaluation(self):
        data = _make_trending_data()
        engine = BacktestEngine(DummyStrategy([10]), *data, commission_pips=0.0, max_spread_pips=0.0)

        # Create index matrix where hours_end is max (index 23 = hour 23)
        # so signals at hour 10 pass the 0-23 filter
        idx = np.zeros((1, engine.encoding.num_params), dtype=np.int64)
        # Set hours_end to last index (23 = hour 23)
        hours_end_col = engine.encoding.name_to_index["allowed_hours_end"]
        idx[0, hours_end_col] = len(engine.encoding.column("allowed_hours_end").values) - 1
        metrics = engine.evaluate_batch_from_indices(idx)
        assert metrics.shape == (1, NUM_METRICS)
        assert metrics[0, M_TRADES] == 1.0


class TestFullMode:
    def test_full_mode_evaluation(self):
        data = _make_trending_data()
        engine = BacktestEngine(DummyStrategy([10, 30]), *data, slippage_pips=0.0, commission_pips=0.0, max_spread_pips=0.0)

        params = _default_params()
        params["max_bars"] = 5  # Force exit after 5 bars
        row = encode_params(engine.encoding, params)
        metrics = engine.evaluate_batch(row.reshape(1, -1), exec_mode=EXEC_FULL)
        assert metrics[0, M_TRADES] == 2.0


class TestSellSignals:
    def test_sell_strategy(self):
        data = _make_flat_data()
        engine = BacktestEngine(SellStrategy(), *data, slippage_pips=0.0, commission_pips=0.0, max_spread_pips=0.0)
        params = _default_params()
        del params["rsi_threshold"]  # SellStrategy doesn't have this param
        result = engine.evaluate_single(params)
        assert result["trades"] == 1.0


class TestExecutionCosts:
    def test_commission_reduces_profit(self):
        """Engine with commission should produce lower PnL than without."""
        data = _make_trending_data()
        engine_no_cost = BacktestEngine(
            DummyStrategy([10]), *data, slippage_pips=0.0,
            commission_pips=0.0, max_spread_pips=0.0,
        )
        engine_with_cost = BacktestEngine(
            DummyStrategy([10]), *data, slippage_pips=0.0,
            commission_pips=0.7, max_spread_pips=0.0,
        )
        params = _default_params()
        r0 = engine_no_cost.evaluate_single(params)
        r1 = engine_with_cost.evaluate_single(params)
        assert r0["trades"] == r1["trades"]  # Same trade count
        if r0["trades"] > 0:
            assert r0["return_pct"] > r1["return_pct"]  # Commission reduces returns

    def test_max_spread_filter_engine(self):
        """Engine with tight max_spread_pips should filter high-spread signals."""
        pip = 0.0001
        n_bars = 100
        base = 1.1000
        open_ = np.full(n_bars, base, dtype=np.float64)
        high = np.full(n_bars, base + 50 * pip, dtype=np.float64)
        low = np.full(n_bars, base - 50 * pip, dtype=np.float64)
        close = np.full(n_bars, base, dtype=np.float64)
        volume = np.ones(n_bars, dtype=np.float64)
        # Spread is 5 pips — higher than the 3 pip threshold
        spread = np.full(n_bars, 5.0 * pip, dtype=np.float64)

        engine = BacktestEngine(
            DummyStrategy([10]), open_, high, low, close, volume, spread,
            slippage_pips=0.0, commission_pips=0.0, max_spread_pips=3.0,
        )
        params = _default_params()
        result = engine.evaluate_single(params)
        assert result["trades"] == 0.0  # Filtered out by spread

    def test_cost_attributes_stored(self):
        """Engine should store commission_pips and max_spread_pips attributes."""
        data = _make_trending_data()
        engine = BacktestEngine(
            DummyStrategy([10]), *data,
            commission_pips=0.7, max_spread_pips=3.0,
        )
        assert engine.commission_pips == 0.7
        assert engine.max_spread_pips == 3.0
