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
        # mean=10, std~1.58, raw~6.32, annualized~6.32*sqrt(252)~100
        assert 50 < sh < 200

    def test_negative(self):
        pnl = np.array([-10.0, -12.0, -8.0, -11.0, -9.0])
        sh = sharpe_ratio(pnl, trades_per_year=252)
        assert -200 < sh < -50

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
        # equity=[10,20,5,15], peak=20, dd=15, base=max(20,20)=20, dd%=75
        assert 70 < dd < 80

    def test_all_losses(self):
        pnl = np.array([-10.0, -10.0, -10.0])
        dd = max_drawdown_pct(pnl)
        # equity=[-10,-20,-30], peak=0, dd=30, base=max(30,0)=30, dd%=100
        assert 95 < dd <= 100

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
        assert 0 < u < 100  # Meaningful range check

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

    def test_zero_for_losing_system(self):
        """A system with negative Sortino must score exactly zero."""
        pnl = np.array([-10.0, -15.0, 2.0, -8.0, -12.0] * 10)
        qs = quality_score(pnl)
        assert qs == 0.0

    def test_zero_for_consistently_losing(self):
        """Consistently losing strategy must score zero, not false positive.

        Regression test: previously, negative Sortino * negative Return%
        produced a double-negative that gave a large POSITIVE quality score.
        """
        # Simulate a consistently losing strategy (like RSI with bad params)
        pnl = np.array([-30.0, -25.0, 60.0, -30.0, -20.0] * 200)
        qs = quality_score(pnl)
        assert qs == 0.0, f"Losing strategy got quality={qs}, expected 0"

    def test_zero_for_all_losses(self):
        """All-loss strategy must score zero."""
        pnl = np.array([-10.0] * 100)
        qs = quality_score(pnl)
        assert qs == 0.0

    def test_negative_sortino_always_zero(self):
        """Regardless of other metrics, negative Sortino must yield zero quality.

        This is the key guard against double-negative inflation.
        """
        # Strategy with negative mean but high R² (consistent loser)
        pnl = np.array([-5.0, -5.0, -5.0, 10.0, -5.0] * 100)
        so = sortino_ratio(pnl)
        assert so < 0, f"Expected negative Sortino, got {so}"
        qs = quality_score(pnl)
        assert qs == 0.0, f"Negative Sortino should give quality=0, got {qs}"


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
        tpy = 252.0
        m = compute_metrics(pnl, avg_sl_pips=20.0, trades_per_year=tpy)
        assert abs(m["win_rate"] - win_rate(pnl)) < 1e-10
        assert abs(m["profit_factor"] - profit_factor(pnl)) < 1e-10
        assert abs(m["r_squared"] - r_squared(pnl)) < 1e-10
        assert abs(m["sharpe"] - sharpe_ratio(pnl, trades_per_year=tpy)) < 1e-10
        assert abs(m["sortino"] - sortino_ratio(pnl, trades_per_year=tpy)) < 1e-10
