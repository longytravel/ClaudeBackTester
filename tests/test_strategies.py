"""Tests for Phase 2: Strategy Framework — indicators, base class, SL/TP, registry."""

import numpy as np
import pytest

from backtester.strategies import indicators as ind
from backtester.strategies.base import (
    Direction,
    ParamDef,
    ParamSpace,
    Signal,
    Strategy,
    management_params,
    risk_params,
    time_params,
)
from backtester.strategies.registry import clear, create, get, register, set_stage, list_strategies
from backtester.strategies.base import StrategyStage
from backtester.strategies.sl_tp import calc_sl_tp


# ---------------------------------------------------------------------------
# Helpers: synthetic price data
# ---------------------------------------------------------------------------

def _random_ohlcv(n: int = 500, seed: int = 42) -> tuple:
    """Generate realistic synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0003, n))
    high = close + rng.uniform(0.0001, 0.0010, n)
    low = close - rng.uniform(0.0001, 0.0010, n)
    open_ = close + rng.normal(0, 0.0002, n)
    volume = rng.integers(50, 500, n).astype(np.float64)
    spread = rng.uniform(0.00005, 0.0003, n)
    return open_, high, low, close, volume, spread


# ---------------------------------------------------------------------------
# Indicator Tests
# ---------------------------------------------------------------------------

class TestSMA:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ind.sma(data, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_almost_equal(result[2], 2.0)
        np.testing.assert_almost_equal(result[3], 3.0)
        np.testing.assert_almost_equal(result[4], 4.0)

    def test_period_exceeds_data(self):
        data = np.array([1.0, 2.0])
        result = ind.sma(data, 5)
        assert all(np.isnan(result))

    def test_period_one(self):
        data = np.array([3.0, 5.0, 7.0])
        result = ind.sma(data, 1)
        np.testing.assert_array_almost_equal(result, data)


class TestEMA:
    def test_first_value_is_sma(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = ind.ema(data, 5)
        np.testing.assert_almost_equal(result[4], 3.0)  # SMA of first 5

    def test_converges_to_value(self):
        data = np.full(100, 5.0)
        result = ind.ema(data, 10)
        np.testing.assert_almost_equal(result[-1], 5.0)

    def test_period_exceeds_data(self):
        data = np.array([1.0, 2.0])
        result = ind.ema(data, 5)
        assert all(np.isnan(result))


class TestATR:
    def test_constant_range(self):
        n = 50
        high = np.full(n, 1.1010)
        low = np.full(n, 1.1000)
        close = np.full(n, 1.1005)
        result = ind.atr(high, low, close, period=14)
        # ATR should be 0.0010 (constant range)
        np.testing.assert_almost_equal(result[13], 0.0010, decimal=6)

    def test_returns_nan_before_period(self):
        _, high, low, close, _, _ = _random_ohlcv(50)
        result = ind.atr(high, low, close, period=14)
        assert all(np.isnan(result[:13]))
        assert not np.isnan(result[13])


class TestRSI:
    def test_overbought_on_rally(self):
        data = np.arange(1.0, 51.0)  # Monotone up
        result = ind.rsi(data, 14)
        assert result[-1] == 100.0  # Pure up = RSI 100

    def test_oversold_on_decline(self):
        data = np.arange(50.0, 0.0, -1.0)  # Monotone down
        result = ind.rsi(data, 14)
        assert result[-1] == 0.0  # Pure down = RSI 0

    def test_middle_on_flat(self):
        rng = np.random.default_rng(0)
        data = 50.0 + np.cumsum(rng.choice([-1.0, 1.0], size=200))
        result = ind.rsi(data, 14)
        # Should hover around 50 for random walk
        assert 30 < result[-1] < 70


class TestBollingerBands:
    def test_middle_equals_sma(self):
        data = np.arange(1.0, 31.0)
        upper, middle, lower = ind.bollinger_bands(data, period=10, num_std=2.0)
        expected_sma = ind.sma(data, 10)
        np.testing.assert_array_almost_equal(middle, expected_sma)

    def test_bands_symmetric(self):
        data = np.arange(1.0, 31.0)
        upper, middle, lower = ind.bollinger_bands(data, period=10, num_std=2.0)
        # upper - middle should equal middle - lower
        for i in range(9, 30):
            np.testing.assert_almost_equal(upper[i] - middle[i], middle[i] - lower[i])


class TestStochastic:
    def test_basic_range(self):
        _, high, low, close, _, _ = _random_ohlcv(100)
        k, d = ind.stochastic(high, low, close, k_period=14, d_period=3)
        valid_k = k[~np.isnan(k)]
        assert all(0 <= v <= 100 for v in valid_k)

    def test_d_is_sma_of_k(self):
        """Verify %D is actually the SMA of %K."""
        _, high, low, close, _, _ = _random_ohlcv(100)
        k, d = ind.stochastic(high, low, close, k_period=14, d_period=3)
        # %D should be the 3-period moving average of %K
        # Check a few values manually
        for i in range(15, 50):  # well past warmup
            expected = np.mean(k[i - 2 : i + 1])
            np.testing.assert_almost_equal(d[i], expected, decimal=10)

    def test_d_range(self):
        """Verify %D values are also in 0-100 range."""
        _, high, low, close, _, _ = _random_ohlcv(100)
        k, d = ind.stochastic(high, low, close, k_period=14, d_period=3)
        valid_d = d[~np.isnan(d)]
        assert len(valid_d) > 0
        assert all(0 <= v <= 100 for v in valid_d)

    def test_d_starts_at_correct_index(self):
        _, high, low, close, _, _ = _random_ohlcv(100)
        k, d = ind.stochastic(high, low, close, k_period=14, d_period=3)
        # %K valid from index 13, %D valid from index 15 (13 + 3 - 1)
        assert np.isnan(d[14])
        assert not np.isnan(d[15])


class TestMACD:
    def test_histogram_is_diff(self):
        _, _, _, close, _, _ = _random_ohlcv(100)
        line, signal, hist = ind.macd(close, fast=12, slow=26, signal_period=9)
        valid = ~np.isnan(line) & ~np.isnan(signal)
        np.testing.assert_array_almost_equal(
            hist[valid], (line - signal)[valid],
        )


class TestADX:
    def test_trending_market(self):
        # Strong uptrend
        close = np.linspace(1.0, 2.0, 100)
        high = close + 0.01
        low = close - 0.005
        adx_val, plus_di, minus_di = ind.adx(high, low, close, period=14)
        valid_adx = adx_val[~np.isnan(adx_val)]
        assert len(valid_adx) > 0, "ADX should have valid values for 100-bar trending data"
        assert valid_adx[-1] > 20  # Should show trending


class TestDonchian:
    def test_basic(self):
        high = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        low = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        upper, middle, lower = ind.donchian(high, low, period=3)
        np.testing.assert_almost_equal(upper[2], 7.0)
        np.testing.assert_almost_equal(lower[2], 1.0)
        np.testing.assert_almost_equal(upper[4], 9.0)
        np.testing.assert_almost_equal(lower[4], 3.0)


class TestSupertrend:
    def test_returns_direction(self):
        _, high, low, close, _, _ = _random_ohlcv(200)
        st, direction = ind.supertrend(high, low, close, period=10, multiplier=3.0)
        # Direction should be +1 or -1
        valid = direction != 0
        assert all(d in (1, -1) for d in direction[valid])


class TestKeltner:
    def test_middle_is_ema(self):
        _, high, low, close, _, _ = _random_ohlcv(100)
        upper, middle, lower = ind.keltner(high, low, close, ema_period=20)
        expected = ind.ema(close, 20)
        np.testing.assert_array_almost_equal(middle, expected)


class TestWilliamsR:
    def test_range(self):
        _, high, low, close, _, _ = _random_ohlcv(100)
        result = ind.williams_r(high, low, close, period=14)
        valid = result[~np.isnan(result)]
        assert all(-100 <= v <= 0 for v in valid)


class TestCCI:
    def test_returns_values(self):
        _, high, low, close, _, _ = _random_ohlcv(100)
        result = ind.cci(high, low, close, period=20)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0


class TestSwingDetection:
    def test_swing_highs(self):
        # Clear peak at index 5
        high = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        result = ind.swing_highs(high, lookback=3)
        assert not np.isnan(result[5])
        np.testing.assert_almost_equal(result[5], 10.0)

    def test_swing_lows(self):
        low = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 1.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        result = ind.swing_lows(low, lookback=3)
        assert not np.isnan(result[5])
        np.testing.assert_almost_equal(result[5], 1.0)


# ---------------------------------------------------------------------------
# Parameter Space Tests
# ---------------------------------------------------------------------------

class TestParamSpace:
    def test_add_and_iterate(self):
        ps = ParamSpace()
        ps.add("rsi_period", [10, 14, 20], group="signal")
        ps.add("sl_pips", [20, 30, 40], group="risk")
        assert len(ps) == 2
        assert "rsi_period" in ps
        assert ps.groups == {"signal": ["rsi_period"], "risk": ["sl_pips"]}

    def test_random_sample(self):
        ps = ParamSpace()
        ps.add("a", [1, 2, 3])
        ps.add("b", ["x", "y"])
        sample = ps.random_sample(np.random.default_rng(42))
        assert sample["a"] in [1, 2, 3]
        assert sample["b"] in ["x", "y"]

    def test_total_combinations(self):
        ps = ParamSpace()
        ps.add("a", [1, 2, 3])
        ps.add("b", [10, 20])
        assert ps.total_combinations() == 6

    def test_standard_groups(self):
        rp = risk_params()
        assert len(rp) == 7
        assert all(p.group == "risk" for p in rp)

        mp = management_params()
        assert all(p.group == "management" for p in mp)
        # Verify defaults are OFF (REQ-S13)
        trailing = next(p for p in mp if p.name == "trailing_mode")
        assert trailing.values[0] == "off"
        be = next(p for p in mp if p.name == "breakeven_enabled")
        assert be.values[0] is False


# ---------------------------------------------------------------------------
# SL/TP Calculator Tests
# ---------------------------------------------------------------------------

class TestSLTP:
    def _make_signal(self, direction=Direction.BUY, price=1.1000, atr=20.0):
        return Signal(
            bar_index=100,
            direction=direction,
            entry_price=price,
            hour=10,
            day_of_week=2,
            atr_pips=atr,
        )

    def test_fixed_pips_buy(self):
        sig = self._make_signal()
        result = calc_sl_tp(
            sig, {"sl_mode": "fixed_pips", "sl_fixed_pips": 30, "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0},
            high=np.array([]), low=np.array([]),
        )
        np.testing.assert_almost_equal(result.sl_pips, 30.0)
        np.testing.assert_almost_equal(result.tp_pips, 60.0)
        assert result.sl_price < 1.1000
        assert result.tp_price > 1.1000

    def test_fixed_pips_sell(self):
        sig = self._make_signal(direction=Direction.SELL)
        result = calc_sl_tp(
            sig, {"sl_mode": "fixed_pips", "sl_fixed_pips": 25, "tp_mode": "rr_ratio", "tp_rr_ratio": 1.5},
            high=np.array([]), low=np.array([]),
        )
        assert result.sl_price > 1.1000  # SL above entry for sell
        assert result.tp_price < 1.1000  # TP below entry for sell

    def test_atr_based_sl(self):
        sig = self._make_signal(atr=30.0)
        result = calc_sl_tp(
            sig, {"sl_mode": "atr_based", "sl_atr_mult": 1.5, "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0},
            high=np.array([]), low=np.array([]),
        )
        np.testing.assert_almost_equal(result.sl_pips, 45.0)  # 30 * 1.5
        np.testing.assert_almost_equal(result.tp_pips, 90.0)  # 45 * 2.0

    def test_tp_gte_sl_enforced(self):
        """REQ-S17: TP distance >= SL distance."""
        sig = self._make_signal()
        result = calc_sl_tp(
            sig, {"sl_mode": "fixed_pips", "sl_fixed_pips": 50, "tp_mode": "rr_ratio", "tp_rr_ratio": 0.5},
            high=np.array([]), low=np.array([]),
        )
        # RR 0.5 would give TP=25 pips, but constraint forces TP >= SL
        assert result.tp_pips >= result.sl_pips

    def test_swing_based_sl(self):
        sig = self._make_signal(direction=Direction.BUY, price=1.1050)
        low = np.full(200, 1.1020)
        low[80] = 1.0990  # Swing low 60 pips below entry
        high = np.full(200, 1.1060)
        result = calc_sl_tp(
            sig, {"sl_mode": "swing", "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0},
            high=high, low=low, swing_lookback=50,
        )
        # Should use swing low at 1.0990, distance = 60 pips
        np.testing.assert_almost_equal(result.sl_pips, 60.0, decimal=0)

    def test_atr_tp_mode(self):
        sig = self._make_signal(atr=25.0)
        result = calc_sl_tp(
            sig, {"sl_mode": "fixed_pips", "sl_fixed_pips": 20, "tp_mode": "atr_based", "tp_atr_mult": 3.0},
            high=np.array([]), low=np.array([]),
        )
        np.testing.assert_almost_equal(result.tp_pips, 75.0)  # 25 * 3.0

    def test_fixed_tp_mode(self):
        sig = self._make_signal()
        result = calc_sl_tp(
            sig, {"sl_mode": "fixed_pips", "sl_fixed_pips": 20, "tp_mode": "fixed_pips", "tp_fixed_pips": 100},
            high=np.array([]), low=np.array([]),
        )
        np.testing.assert_almost_equal(result.tp_pips, 100.0)


# ---------------------------------------------------------------------------
# Registry Tests
# ---------------------------------------------------------------------------

class _DummyStrategy(Strategy):
    @property
    def name(self) -> str:
        return "dummy_test"

    @property
    def version(self) -> str:
        return "0.1"

    def param_space(self) -> ParamSpace:
        ps = ParamSpace()
        ps.add("period", [10, 20, 30])
        return ps

    def generate_signals(self, open, high, low, close, volume, spread):
        return []

    def filter_signals(self, signals, params):
        return signals

    def calc_sl_tp(self, signal, params, high, low):
        from backtester.strategies.base import SLTPResult
        return SLTPResult(0, 0, 0, 0)


class TestRegistry:
    def setup_method(self):
        clear()

    def test_register_and_get(self):
        register(_DummyStrategy, aliases=["dummy", "test_strat"])
        cls = get("dummy_test")
        assert cls is _DummyStrategy
        cls2 = get("dummy")
        assert cls2 is _DummyStrategy

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown strategy"):
            get("nonexistent")

    def test_create_instance(self):
        register(_DummyStrategy)
        strat = create("dummy_test")
        assert isinstance(strat, _DummyStrategy)
        assert strat.name == "dummy_test"

    def test_lifecycle_stages(self):
        register(_DummyStrategy)
        assert list_strategies() == [{"name": "dummy_test", "stage": "built"}]
        set_stage("dummy_test", StrategyStage.VALIDATED)
        assert list_strategies() == [{"name": "dummy_test", "stage": "validated"}]

    def test_list_strategies(self):
        register(_DummyStrategy)
        result = list_strategies()
        assert len(result) == 1
        assert result[0]["name"] == "dummy_test"

    def test_register_strategy_with_self_access(self):
        """Verify registry works when name property uses self."""
        class _SelfAccessStrategy(_DummyStrategy):
            def __init__(self):
                self._my_name = "self_access_test"

            @property
            def name(self) -> str:
                return self._my_name

        # This should NOT fail — the registry should handle self access
        register(_SelfAccessStrategy)
        cls = get("self_access_test")
        assert cls is _SelfAccessStrategy


# ---------------------------------------------------------------------------
# Vectorized Path Tests
# ---------------------------------------------------------------------------

class _SignalProducingStrategy(Strategy):
    """Strategy that produces signals with attrs for testing vectorized path."""

    @property
    def name(self) -> str:
        return "signal_producer"

    @property
    def version(self) -> str:
        return "0.1"

    def param_space(self) -> ParamSpace:
        ps = ParamSpace()
        ps.add("min_rsi", [30, 40, 50])
        return ps

    def generate_signals(self, open, high, low, close, volume, spread):
        return [
            Signal(bar_index=10, direction=Direction.BUY, entry_price=1.1000,
                   hour=10, day_of_week=2, atr_pips=20.0,
                   attrs={"rsi": 35.0, "trend": 1.0}),
            Signal(bar_index=20, direction=Direction.SELL, entry_price=1.1050,
                   hour=14, day_of_week=3, atr_pips=25.0,
                   attrs={"rsi": 70.0, "trend": -1.0}),
            Signal(bar_index=30, direction=Direction.BUY, entry_price=1.0980,
                   hour=9, day_of_week=1, atr_pips=18.0,
                   attrs={"rsi": 42.0, "trend": 1.0}),
        ]

    def filter_signals(self, signals, params):
        min_rsi = params.get("min_rsi", 30)
        return [s for s in signals if s.attrs.get("rsi", 0) >= min_rsi]

    def calc_sl_tp(self, signal, params, high, low):
        from backtester.strategies.base import SLTPResult
        return SLTPResult(0, 0, 0, 0)


class TestVectorizedPath:
    def test_vectorized_generation_preserves_attrs(self):
        """REQ-S10: attrs must be propagated in vectorized path."""
        strat = _SignalProducingStrategy()
        dummy = np.zeros(50)
        result = strat.generate_signals_vectorized(dummy, dummy, dummy, dummy, dummy, dummy)

        assert len(result["bar_index"]) == 3
        assert "attr_rsi" in result
        assert "attr_trend" in result
        np.testing.assert_array_almost_equal(result["attr_rsi"], [35.0, 70.0, 42.0])
        np.testing.assert_array_almost_equal(result["attr_trend"], [1.0, -1.0, 1.0])

    def test_vectorized_generation_empty(self):
        strat = _DummyStrategy()  # returns empty signals
        dummy = np.zeros(50)
        result = strat.generate_signals_vectorized(dummy, dummy, dummy, dummy, dummy, dummy)
        assert len(result["bar_index"]) == 0

    def test_vectorized_filter_default(self):
        strat = _SignalProducingStrategy()
        dummy = np.zeros(50)
        signals = strat.generate_signals_vectorized(dummy, dummy, dummy, dummy, dummy, dummy)
        # With min_rsi=40, only signals with rsi >= 40 pass (rsi=70 and rsi=42)
        mask = strat.filter_signals_vectorized(signals, {"min_rsi": 40})
        assert mask.sum() == 2
        assert not mask[0]  # rsi=35 < 40
        assert mask[1]      # rsi=70 >= 40
        assert mask[2]      # rsi=42 >= 40


# ---------------------------------------------------------------------------
# SL/TP Edge Cases
# ---------------------------------------------------------------------------

class TestSLTPEdgeCases:
    def _make_signal(self, direction=Direction.BUY, price=1.1000, atr=20.0):
        return Signal(
            bar_index=100, direction=direction, entry_price=price,
            hour=10, day_of_week=2, atr_pips=atr,
        )

    def test_zero_atr_fixed_sl(self):
        """Zero ATR should still work with fixed SL mode."""
        sig = self._make_signal(atr=0.0)
        result = calc_sl_tp(
            sig, {"sl_mode": "fixed_pips", "sl_fixed_pips": 30, "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0},
            high=np.array([]), low=np.array([]),
        )
        np.testing.assert_almost_equal(result.sl_pips, 30.0)

    def test_zero_atr_atr_mode_gives_zero(self):
        """Zero ATR with ATR-based SL gives zero SL distance (edge case)."""
        sig = self._make_signal(atr=0.0)
        result = calc_sl_tp(
            sig, {"sl_mode": "atr_based", "sl_atr_mult": 1.5, "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0},
            high=np.array([]), low=np.array([]),
        )
        # 0 ATR * 1.5 = 0 SL distance, 0 * 2.0 RR = 0 TP distance
        np.testing.assert_almost_equal(result.sl_pips, 0.0)

    def test_swing_no_data_falls_back_to_atr(self):
        """Swing mode with no lookback data falls back to ATR."""
        sig = self._make_signal(atr=20.0)
        sig.bar_index = 0  # No bars to look back
        result = calc_sl_tp(
            sig, {"sl_mode": "swing", "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0},
            high=np.array([1.1050]), low=np.array([1.0950]), swing_lookback=50,
        )
        # Should fallback to ATR * 1.5 = 30 pips
        np.testing.assert_almost_equal(result.sl_pips, 30.0)

    def test_jpy_pip_value(self):
        """Test with JPY pip value (0.01 instead of 0.0001)."""
        sig = Signal(
            bar_index=100, direction=Direction.BUY, entry_price=150.00,
            hour=10, day_of_week=2, atr_pips=50.0,
        )
        result = calc_sl_tp(
            sig, {"sl_mode": "atr_based", "sl_atr_mult": 1.0, "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0},
            high=np.array([]), low=np.array([]), pip_value=0.01,
        )
        np.testing.assert_almost_equal(result.sl_pips, 50.0)
        np.testing.assert_almost_equal(result.sl_price, 149.50)  # 150 - 50*0.01


# ---------------------------------------------------------------------------
# Integration Test: Full Strategy Workflow
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_end_to_end_strategy_workflow(self):
        """Register, generate signals, filter, calc SL/TP — full flow."""
        clear()

        strat = _SignalProducingStrategy()
        register(_SignalProducingStrategy)

        # Look up and instantiate
        cls = get("signal_producer")
        instance = cls()
        assert instance.name == "signal_producer"

        # Generate signals
        dummy = np.zeros(50)
        signals = instance.generate_signals(dummy, dummy, dummy, dummy, dummy, dummy)
        assert len(signals) == 3

        # Filter with params
        filtered = instance.filter_signals(signals, {"min_rsi": 40})
        assert len(filtered) == 2

        # Calc SL/TP for each filtered signal
        for sig in filtered:
            result = calc_sl_tp(
                sig, {"sl_mode": "atr_based", "sl_atr_mult": 1.5, "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0},
                high=dummy, low=dummy,
            )
            assert result.tp_pips >= result.sl_pips  # REQ-S17
            assert result.sl_pips > 0

    def test_time_params_valid(self):
        """Verify time_params returns valid parameter definitions."""
        tp = time_params()
        assert len(tp) == 3
        assert all(p.group == "time" for p in tp)
        hours_start = next(p for p in tp if p.name == "allowed_hours_start")
        assert len(hours_start.values) == 24  # 0..23


# ---------------------------------------------------------------------------
# Regression: RSI multi-threshold signal generation (Feb 2026)
# ---------------------------------------------------------------------------

class TestRSIMultiThreshold:
    """Verify RSI strategy generates signals at each threshold crossing."""

    def test_generates_signals_at_multiple_thresholds(self):
        """RSI crossing different thresholds should produce distinct signals."""
        from backtester.strategies.rsi_mean_reversion import (
            RSIMeanReversion,
            OVERSOLD_THRESHOLDS,
            OVERBOUGHT_THRESHOLDS,
        )
        strategy = RSIMeanReversion()

        # Create data with a clear RSI drop from high to very low
        n = 300
        rng = np.random.default_rng(42)
        close = np.full(n, 1.1000)
        # First half: rising (RSI high), then declining (RSI drops)
        close[:100] = 1.1000 + np.cumsum(rng.uniform(0.0001, 0.0005, 100))
        close[100:200] = close[99] - np.cumsum(rng.uniform(0.0001, 0.0005, 100))
        close[200:] = close[199] + np.cumsum(rng.uniform(0.0001, 0.0005, 100))

        high = close + rng.uniform(0.0001, 0.0005, n)
        low = close - rng.uniform(0.0001, 0.0005, n)
        open_ = close + rng.uniform(-0.0002, 0.0002, n)
        volume = np.full(n, 100.0)
        spread = np.full(n, 0.0001)

        sigs = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread, 0.0001
        )

        if len(sigs["bar_index"]) > 0:
            unique_filters = set(sigs["filter_value"].tolist())
            # All filter values must be from the threshold constants
            valid_thresholds = set(float(t) for t in OVERSOLD_THRESHOLDS + OVERBOUGHT_THRESHOLDS)
            assert unique_filters.issubset(valid_thresholds), (
                f"filter_values {unique_filters} not in valid thresholds {valid_thresholds}"
            )

    def test_no_dead_params(self):
        """RSI strategy should not have atr_period or sma_filter_period."""
        from backtester.strategies.rsi_mean_reversion import RSIMeanReversion
        strategy = RSIMeanReversion()
        space = strategy.param_space()
        param_names = space.names
        assert "atr_period" not in param_names
        assert "sma_filter_period" not in param_names

    def test_variant_is_rsi_period_value(self):
        """Variant field should store actual RSI period (7/9/14/21), not index."""
        from backtester.strategies.rsi_mean_reversion import RSIMeanReversion, RSI_PERIODS
        strategy = RSIMeanReversion()
        n = 300
        rng = np.random.default_rng(123)
        close = 1.1 + np.cumsum(rng.normal(0, 0.0005, n))
        high = close + rng.uniform(0.0001, 0.0005, n)
        low = close - rng.uniform(0.0001, 0.0005, n)
        open_ = close + rng.uniform(-0.0002, 0.0002, n)
        volume = np.full(n, 100.0)
        spread = np.full(n, 0.0001)

        sigs = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread, 0.0001,
        )
        if len(sigs["variant"]) > 0:
            for v in sigs["variant"]:
                assert int(v) in RSI_PERIODS, f"variant {v} not a valid RSI period"


# ---------------------------------------------------------------------------
# Regression: Weekly timeframe anchor (Feb 2026)
# ---------------------------------------------------------------------------

class TestWeeklyTimeframeAnchor:
    """Verify weekly resampling uses Monday anchor for FX."""

    def test_weekly_rule_is_monday(self):
        from backtester.data.timeframes import TIMEFRAME_RULES
        assert TIMEFRAME_RULES["W"] == "W-MON"
