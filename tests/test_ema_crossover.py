"""Tests for EMA Crossover strategy."""

import numpy as np
import pytest

from backtester.strategies.ema_crossover import (
    EMA_COMBOS,
    EMA_FAST_PERIODS,
    EMA_SLOW_PERIODS,
    EMACrossover,
    decode_combo,
)


@pytest.fixture
def strategy():
    return EMACrossover()


class TestDecodeCombo:
    def test_roundtrip(self):
        for f in EMA_FAST_PERIODS:
            for s in EMA_SLOW_PERIODS:
                if f < s:
                    combo = f * 1000 + s
                    assert decode_combo(combo) == (f, s)

    def test_all_combos_valid(self):
        for combo in EMA_COMBOS:
            fast, slow = decode_combo(combo)
            assert fast < slow
            assert fast in EMA_FAST_PERIODS
            assert slow in EMA_SLOW_PERIODS

    def test_combos_count(self):
        expected = sum(1 for f in EMA_FAST_PERIODS for s in EMA_SLOW_PERIODS if f < s)
        assert len(EMA_COMBOS) == expected


class TestParamSpace:
    def test_has_ema_combo(self, strategy):
        ps = strategy.param_space()
        assert "ema_combo" in ps
        p = ps.get("ema_combo")
        assert p.group == "signal"
        assert p.values == EMA_COMBOS

    def test_has_standard_groups(self, strategy):
        ps = strategy.param_space()
        groups = ps.groups
        assert "signal" in groups
        assert "risk" in groups
        assert "management" in groups
        assert "time" in groups

    def test_total_params(self, strategy):
        ps = strategy.param_space()
        # 1 signal + 7 risk + 14 management + 3 time = 25
        assert len(ps) == 25


class TestSignalGeneration:
    def _make_trending_data(self, n=500):
        """Create trending price data that guarantees EMA crossovers."""
        rng = np.random.default_rng(42)
        # Start flat, then trend up, then trend down, then up again
        close = np.ones(n) * 1.1000
        # Flat phase
        for i in range(1, 100):
            close[i] = close[i - 1] + rng.normal(0, 0.0001)
        # Strong uptrend
        for i in range(100, 250):
            close[i] = close[i - 1] + 0.0005 + rng.normal(0, 0.0001)
        # Strong downtrend
        for i in range(250, 400):
            close[i] = close[i - 1] - 0.0005 + rng.normal(0, 0.0001)
        # Recovery
        for i in range(400, n):
            close[i] = close[i - 1] + 0.0003 + rng.normal(0, 0.0001)

        high = close + rng.uniform(0.0001, 0.0010, n)
        low = close - rng.uniform(0.0001, 0.0010, n)
        open_ = close + rng.normal(0, 0.0002, n)
        volume = rng.uniform(100, 1000, n)
        spread = np.full(n, 0.00015)
        return open_, high, low, close, volume, spread

    def test_generates_signals(self, strategy):
        open_, high, low, close, volume, spread = self._make_trending_data()
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread,
        )
        assert len(result["bar_index"]) > 0
        assert len(result["variant"]) == len(result["bar_index"])

    def test_signals_have_both_directions(self, strategy):
        open_, high, low, close, volume, spread = self._make_trending_data()
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread,
        )
        dirs = result["direction"]
        assert 1 in dirs    # BUY
        assert -1 in dirs   # SELL

    def test_variants_are_valid_combos(self, strategy):
        open_, high, low, close, volume, spread = self._make_trending_data()
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread,
        )
        combo_set = set(EMA_COMBOS)
        for v in result["variant"]:
            assert int(v) in combo_set

    def test_no_filter_value_needed(self, strategy):
        open_, high, low, close, volume, spread = self._make_trending_data()
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread,
        )
        # EMA crossover doesn't use filter_value
        assert "filter_value" not in result

    def test_atr_pips_positive(self, strategy):
        open_, high, low, close, volume, spread = self._make_trending_data()
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread,
        )
        assert np.all(result["atr_pips"] > 0)

    def test_empty_on_flat_data(self, strategy):
        n = 300
        close = np.ones(n) * 1.1000
        high = close + 0.0001
        low = close - 0.0001
        open_ = close.copy()
        volume = np.ones(n) * 100.0
        spread = np.full(n, 0.00015)
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread,
        )
        # On perfectly flat data, EMAs converge to the same value = no crossovers
        assert len(result["bar_index"]) == 0

    def test_bar_indices_in_range(self, strategy):
        open_, high, low, close, volume, spread = self._make_trending_data()
        n = len(close)
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread,
        )
        assert np.all(result["bar_index"] >= 0)
        assert np.all(result["bar_index"] < n)

    def test_many_more_signals_than_rsi(self, strategy):
        """EMA crossover should generate more signals than RSI on same data."""
        open_, high, low, close, volume, spread = self._make_trending_data(n=1000)
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread,
        )
        # With 41 combos and trending data, should have plenty of crossovers
        assert len(result["bar_index"]) > 50


class TestMetadata:
    def test_name(self, strategy):
        assert strategy.name == "ema_crossover"

    def test_version(self, strategy):
        assert strategy.version == "1.0.0"

    def test_optimization_stages(self, strategy):
        assert strategy.optimization_stages() == ["signal", "time", "risk", "management"]


class TestRegistry:
    def test_registered(self):
        from backtester.strategies import registry
        strategies = {s["name"] for s in registry.list_strategies()}
        assert "ema_crossover" in strategies

    def test_create_by_name(self):
        from backtester.strategies import registry
        strat = registry.create("ema_crossover")
        assert strat.name == "ema_crossover"
