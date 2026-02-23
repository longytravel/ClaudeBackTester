"""Tests for Python-side metrics computation."""

import numpy as np
import pytest

from backtester.core.metrics import (
    compute_metrics,
    max_drawdown_pct,
    profit_factor,
    quality_score,
    r_squared,
    return_pct,
    sharpe_ratio,
    sortino_ratio,
    ulcer_index,
    win_rate,
)


# ---------------------------------------------------------------------------
# Win Rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_all_wins(self):
        pnl = np.array([10.0, 20.0, 5.0, 15.0])
        assert win_rate(pnl) == 1.0

    def test_all_losses(self):
        pnl = np.array([-10.0, -20.0, -5.0])
        assert win_rate(pnl) == 0.0

    def test_mixed(self):
        pnl = np.array([10.0, -5.0, 20.0, -10.0])
        assert win_rate(pnl) == 0.5

    def test_empty(self):
        assert win_rate(np.array([])) == 0.0

    def test_single_win(self):
        assert win_rate(np.array([5.0])) == 1.0

    def test_single_loss(self):
        assert win_rate(np.array([-5.0])) == 0.0

    def test_zero_pnl_not_counted_as_win(self):
        pnl = np.array([0.0, 10.0, -5.0])
        assert abs(win_rate(pnl) - 1.0 / 3.0) < 1e-10


# ---------------------------------------------------------------------------
# Profit Factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_basic(self):
        pnl = np.array([30.0, -10.0, 20.0, -10.0])
        # Gross profit = 50, Gross loss = 20
        assert abs(profit_factor(pnl) - 2.5) < 1e-10

    def test_all_wins(self):
        pnl = np.array([10.0, 20.0])
        assert profit_factor(pnl) == 10.0  # Capped at 10

    def test_all_losses(self):
        pnl = np.array([-10.0, -20.0])
        assert profit_factor(pnl) == 0.0

    def test_empty(self):
        assert profit_factor(np.array([])) == 0.0

    def test_breakeven(self):
        pnl = np.array([10.0, -10.0])
        assert abs(profit_factor(pnl) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_positive(self):
        pnl = np.array([10.0, 12.0, 8.0, 11.0, 9.0])
        sh = sharpe_ratio(pnl, trades_per_year=252)
        assert sh > 0

    def test_negative(self):
        pnl = np.array([-10.0, -12.0, -8.0, -11.0, -9.0])
        sh = sharpe_ratio(pnl, trades_per_year=252)
        assert sh < 0

    def test_zero_std(self):
        pnl = np.array([5.0, 5.0, 5.0])
        assert sharpe_ratio(pnl) == 0.0

    def test_single_trade(self):
        assert sharpe_ratio(np.array([10.0])) == 0.0

    def test_empty(self):
        assert sharpe_ratio(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Sortino Ratio
# ---------------------------------------------------------------------------

class TestSortino:
    def test_positive_with_losses(self):
        pnl = np.array([20.0, -5.0, 15.0, -3.0, 10.0])
        so = sortino_ratio(pnl, trades_per_year=252)
        assert so > 0

    def test_all_wins(self):
        pnl = np.array([10.0, 20.0, 15.0])
        so = sortino_ratio(pnl)
        assert so == 10.0  # Capped

    def test_all_losses(self):
        pnl = np.array([-10.0, -20.0])
        so = sortino_ratio(pnl)
        assert so < 0

    def test_empty(self):
        assert sortino_ratio(np.array([])) == 0.0

    def test_single(self):
        assert sortino_ratio(np.array([5.0])) == 0.0


# ---------------------------------------------------------------------------
# Max Drawdown %
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_no_drawdown(self):
        pnl = np.array([10.0, 10.0, 10.0])
        dd = max_drawdown_pct(pnl)
        assert dd == 0.0

    def test_simple_drawdown(self):
        pnl = np.array([10.0, 10.0, -15.0, 10.0])
        dd = max_drawdown_pct(pnl)
        assert dd > 0

    def test_all_losses(self):
        pnl = np.array([-10.0, -10.0, -10.0])
        dd = max_drawdown_pct(pnl)
        assert dd > 0

    def test_empty(self):
        assert max_drawdown_pct(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Return %
# ---------------------------------------------------------------------------

class TestReturnPct:
    def test_positive(self):
        pnl = np.array([10.0, 20.0, -5.0])
        ret = return_pct(pnl, avg_sl_pips=25.0)
        # total = 25, ret = 25/25 * 100 = 100%
        assert abs(ret - 100.0) < 1e-10

    def test_negative(self):
        pnl = np.array([-10.0, -20.0])
        ret = return_pct(pnl, avg_sl_pips=30.0)
        assert ret < 0

    def test_zero_sl(self):
        assert return_pct(np.array([10.0]), avg_sl_pips=0.0) == 0.0

    def test_empty(self):
        assert return_pct(np.array([]), avg_sl_pips=30.0) == 0.0


# ---------------------------------------------------------------------------
# R-squared
# ---------------------------------------------------------------------------

class TestRSquared:
    def test_perfect_linear(self):
        """Constant positive PnL → perfectly linear equity → R²=1."""
        pnl = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        assert abs(r_squared(pnl) - 1.0) < 1e-10

    def test_random_low_rsq(self):
        rng = np.random.default_rng(42)
        pnl = rng.normal(0, 10, size=100)
        rsq = r_squared(pnl)
        assert 0.0 <= rsq <= 1.0

    def test_empty(self):
        assert r_squared(np.array([])) == 0.0

    def test_single(self):
        assert r_squared(np.array([5.0])) == 0.0

    def test_trending_up(self):
        """Steadily increasing PnL with noise → high R²."""
        pnl = np.array([8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 13.0, 11.0, 14.0, 12.0])
        rsq = r_squared(pnl)
        assert rsq > 0.8


# ---------------------------------------------------------------------------
# Ulcer Index
# ---------------------------------------------------------------------------

class TestUlcerIndex:
    def test_no_drawdown(self):
        pnl = np.array([10.0, 10.0, 10.0])
        assert ulcer_index(pnl) == 0.0

    def test_some_drawdown(self):
        pnl = np.array([10.0, -5.0, 10.0])
        u = ulcer_index(pnl)
        assert u > 0

    def test_empty(self):
        assert ulcer_index(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Quality Score
# ---------------------------------------------------------------------------

class TestQualityScore:
    def test_empty(self):
        assert quality_score(np.array([])) == 0.0

    def test_positive_for_good_system(self):
        """A system with consistent wins should have positive quality."""
        pnl = np.array([10.0, 5.0, 15.0, -3.0, 8.0, 12.0, -2.0, 20.0] * 10)
        qs = quality_score(pnl)
        assert qs > 0

    def test_low_for_bad_system(self):
        """A system with mostly losses should have very low quality."""
        pnl = np.array([-10.0, -15.0, 2.0, -8.0, -12.0] * 10)
        qs = quality_score(pnl)
        # Small positive possible due to formula structure, but should be near zero
        assert qs < 1.0


# ---------------------------------------------------------------------------
# compute_metrics (integration)
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_all_keys_present(self):
        pnl = np.array([10.0, -5.0, 20.0, -3.0, 15.0])
        m = compute_metrics(pnl)
        expected_keys = {
            "trades", "win_rate", "profit_factor", "sharpe", "sortino",
            "max_dd_pct", "return_pct", "r_squared", "ulcer", "quality_score",
        }
        assert set(m.keys()) == expected_keys

    def test_trade_count(self):
        pnl = np.array([10.0, -5.0, 20.0])
        m = compute_metrics(pnl)
        assert m["trades"] == 3.0

    def test_empty_returns_zeros(self):
        m = compute_metrics(np.array([]))
        assert m["trades"] == 0.0
        assert m["win_rate"] == 0.0
        assert m["quality_score"] == 0.0

    def test_consistent_with_individual_functions(self):
        pnl = np.array([10.0, -5.0, 20.0, -8.0, 15.0, -2.0, 12.0])
        m = compute_metrics(pnl, avg_sl_pips=20.0)
        assert abs(m["win_rate"] - win_rate(pnl)) < 1e-10
        assert abs(m["profit_factor"] - profit_factor(pnl)) < 1e-10
        assert abs(m["r_squared"] - r_squared(pnl)) < 1e-10
