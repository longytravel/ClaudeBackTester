"""Tests for cross-validation objective inside optimizer."""

import numpy as np
import pytest

from backtester.optimizer.cv_objective import (
    BARS_PER_DAY,
    CVFoldConfig,
    aggregate_fold_scores,
    auto_configure_folds,
)


class TestAutoConfigureFolds:
    """Tests for auto_configure_folds()."""

    def test_h1_day_trader(self):
        """H1, 16 years, 200 trades/year → K=7 (capped)."""
        cfg = auto_configure_folds(
            n_bars=96000, bars_per_year=6048, timeframe="H1",
            expected_trades_per_year=200,
        )
        assert cfg.n_folds == 7
        assert len(cfg.fold_boundaries) == 7
        assert cfg.embargo_bars == 5 * 24  # 120

    def test_d1_position_trader_long(self):
        """D1, 16 years, 20 trades/year → K=7 (enough total trades)."""
        cfg = auto_configure_folds(
            n_bars=4032, bars_per_year=252, timeframe="D1",
            expected_trades_per_year=20,
        )
        # 320 total trades / 30 min = 10.6, but 16yr/2 = 8 → capped at 7
        assert cfg.n_folds == 7
        assert cfg.embargo_bars == 5  # 5 days * 1 bar/day

    def test_d1_position_trader_short(self):
        """D1, 5 years, 20 trades/year → K=3 (floor from limited data)."""
        cfg = auto_configure_folds(
            n_bars=1260, bars_per_year=252, timeframe="D1",
            expected_trades_per_year=20,
        )
        assert cfg.n_folds == 3

    def test_m15_scalper(self):
        """M15, 5 years, 2000 trades/year → K=3 (floor from years)."""
        cfg = auto_configure_folds(
            n_bars=175200, bars_per_year=35040, timeframe="M15",
            expected_trades_per_year=2000,
        )
        assert cfg.n_folds == 3
        assert cfg.embargo_bars == 5 * 96  # 480

    def test_h4_swing(self):
        """H4, 16 years, 50 trades/year → K=7 (capped; plenty of trades)."""
        cfg = auto_configure_folds(
            n_bars=24192, bars_per_year=1512, timeframe="H4",
            expected_trades_per_year=50,
        )
        assert cfg.n_folds == 7

    def test_override_folds(self):
        """Manual K override."""
        cfg = auto_configure_folds(
            n_bars=96000, bars_per_year=6048, timeframe="H1",
            expected_trades_per_year=200, n_folds_override=4,
        )
        assert cfg.n_folds == 4

    def test_fold_boundaries_cover_data(self):
        """All bars should be covered by folds (with embargo gaps)."""
        cfg = auto_configure_folds(
            n_bars=10000, bars_per_year=6048, timeframe="H1",
            expected_trades_per_year=100,
        )
        # Last fold should end at n_bars
        assert cfg.fold_boundaries[-1][1] == 10000
        # All folds should have positive size
        for start, end in cfg.fold_boundaries:
            assert end > start

    def test_embargo_days_converted_by_timeframe(self):
        """Embargo in bars depends on timeframe."""
        for tf, bpd in BARS_PER_DAY.items():
            if tf == "W":
                continue  # fractional
            cfg = auto_configure_folds(
                n_bars=100000, bars_per_year=6048, timeframe=tf,
                expected_trades_per_year=200, embargo_days=3,
            )
            assert cfg.embargo_bars == max(1, int(3 * bpd))

    def test_very_short_data(self):
        """2 years of data → K=3 (minimum)."""
        cfg = auto_configure_folds(
            n_bars=12096, bars_per_year=6048, timeframe="H1",
            expected_trades_per_year=100,
        )
        assert cfg.n_folds == 3


class TestAggregateFoldScores:
    """Tests for aggregation functions."""

    def test_mean_std_basic(self):
        """Consistent scores get high aggregate."""
        qualities = np.array([[10.0, 10.0, 10.0]])  # consistent
        trades = np.array([[50, 50, 50]])
        result = aggregate_fold_scores(qualities, trades, "mean_std", 1.0, 30)
        assert result[0] == pytest.approx(10.0, abs=0.1)

    def test_mean_std_penalizes_variance(self):
        """High variance gets penalized."""
        consistent = np.array([[10.0, 10.0, 10.0]])
        variable = np.array([[20.0, 10.0, 0.0]])
        trades = np.array([[50, 50, 50]])
        r_consistent = aggregate_fold_scores(consistent, trades, "mean_std", 1.0, 30)
        r_variable = aggregate_fold_scores(variable, trades, "mean_std", 1.0, 30)
        assert r_consistent[0] > r_variable[0]

    def test_mean_std_zero_fold_not_killed(self):
        """A single zero fold should NOT kill the score (variance penalty handles it)."""
        qualities = np.array([[15.0, 0.0, 12.0]])
        trades = np.array([[50, 50, 50]])
        result = aggregate_fold_scores(qualities, trades, "mean_std", 1.0, 30)
        # mean=9, std=8.19, score=9-8.19=0.81 — small but not zero
        assert result[0] > 0

    def test_insufficient_trades_excluded(self):
        """Folds with < min_trades are excluded from aggregation."""
        qualities = np.array([[10.0, 5.0, 8.0]])
        trades = np.array([[50, 5, 50]])  # fold 1 has only 5 trades
        result = aggregate_fold_scores(qualities, trades, "mean_std", 1.0, 30)
        # Should aggregate folds 0 and 2 only (mean=9, std=1)
        assert result[0] > 0

    def test_too_few_valid_folds(self):
        """< 2 valid folds → score 0."""
        qualities = np.array([[10.0, 5.0, 8.0]])
        trades = np.array([[50, 5, 10]])  # only 1 valid fold
        result = aggregate_fold_scores(qualities, trades, "mean_std", 1.0, 30)
        assert result[0] == 0.0

    def test_cvar(self):
        """CVaR takes worst 40% of folds."""
        qualities = np.array([[1.0, 5.0, 10.0, 15.0, 20.0]])
        trades = np.array([[50, 50, 50, 50, 50]])
        result = aggregate_fold_scores(qualities, trades, "cvar", 1.0, 30)
        # Worst 40% of 5 = worst 2: mean(1, 5) = 3.0
        assert result[0] == pytest.approx(3.0, abs=0.1)

    def test_geometric_mean(self):
        """Geometric mean zeros if any fold is zero."""
        good = np.array([[10.0, 10.0, 10.0]])
        has_zero = np.array([[10.0, 0.0, 10.0]])
        trades = np.array([[50, 50, 50]])
        r_good = aggregate_fold_scores(good, trades, "geometric_mean", 1.0, 30)
        r_zero = aggregate_fold_scores(has_zero, trades, "geometric_mean", 1.0, 30)
        assert r_good[0] > 0
        assert r_zero[0] == 0.0

    def test_batch_processing(self):
        """Multiple trials aggregated in one call."""
        qualities = np.array([
            [10.0, 10.0, 10.0],
            [5.0, 5.0, 5.0],
            [20.0, 0.0, 20.0],
        ])
        trades = np.full((3, 3), 50)
        result = aggregate_fold_scores(qualities, trades, "mean_std", 1.0, 30)
        assert len(result) == 3
        assert result[0] > result[1]  # higher consistent > lower consistent
        assert result[2] < result[0]  # has zero fold, penalized but not necessarily zero
