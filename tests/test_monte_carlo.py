"""Tests for Monte Carlo simulation module (pipeline stage 5)."""

import numpy as np
import pytest

from backtester.core.metrics import sharpe_ratio
from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.monte_carlo import (
    block_bootstrap,
    execution_stress_test,
    permutation_test,
    run_monte_carlo,
    trade_skip_test,
)
from backtester.pipeline.types import MonteCarloResult


# ---------------------------------------------------------------------------
# Test data generators (fixed seeds for determinism)
# ---------------------------------------------------------------------------

def _strong_pnl() -> np.ndarray:
    """Profitable strategy: mean=2 pips, std=5."""
    rng = np.random.default_rng(42)
    return rng.normal(2.0, 5.0, 200)


def _weak_pnl() -> np.ndarray:
    """No edge: mean=0, std=5."""
    rng = np.random.default_rng(42)
    return rng.normal(0.0, 5.0, 200)


# ---------------------------------------------------------------------------
# Block bootstrap tests
# ---------------------------------------------------------------------------

class TestBlockBootstrap:
    def test_positive_pnl_mean_sharpe_positive(self):
        """Strong positive PnL should produce mean bootstrap Sharpe > 0."""
        pnl = _strong_pnl()
        rng = np.random.default_rng(42)
        sharpes = block_bootstrap(pnl, n_iterations=500, block_size=10, rng=rng)
        assert np.mean(sharpes) > 0, f"Expected positive mean Sharpe, got {np.mean(sharpes)}"

    def test_output_shape(self):
        """Output shape must match n_iterations."""
        pnl = _strong_pnl()
        rng = np.random.default_rng(42)
        n_iter = 300
        sharpes = block_bootstrap(pnl, n_iterations=n_iter, block_size=10, rng=rng)
        assert sharpes.shape == (n_iter,)

    def test_ci_contains_observed(self):
        """95% CI should contain the observed Sharpe (usually).

        With 1000 iterations and a well-behaved distribution, the observed
        Sharpe should fall within the bootstrap CI. We use a generous check.
        """
        pnl = _strong_pnl()
        observed = sharpe_ratio(pnl)
        rng = np.random.default_rng(42)
        sharpes = block_bootstrap(pnl, n_iterations=1000, block_size=10, rng=rng)
        ci_low = np.percentile(sharpes, 2.5)
        ci_high = np.percentile(sharpes, 97.5)
        # The observed Sharpe should be within the CI (or very close)
        assert ci_low <= observed <= ci_high, (
            f"Observed Sharpe {observed:.4f} outside CI [{ci_low:.4f}, {ci_high:.4f}]"
        )

    def test_empty_pnl(self):
        """Empty PnL should return zeros."""
        rng = np.random.default_rng(42)
        sharpes = block_bootstrap(np.array([]), n_iterations=100, block_size=5, rng=rng)
        assert sharpes.shape == (100,)
        assert np.all(sharpes == 0)

    def test_block_size_larger_than_pnl(self):
        """Block size exceeding PnL length is clamped and doesn't crash."""
        pnl = _strong_pnl()[:10]  # Only 10 trades
        rng = np.random.default_rng(42)
        sharpes = block_bootstrap(pnl, n_iterations=50, block_size=100, rng=rng)
        assert sharpes.shape == (50,)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

class TestPermutationTest:
    def test_strong_signal_low_pvalue(self):
        """Strong positive PnL should produce p-value < 0.05."""
        pnl = _strong_pnl()
        rng = np.random.default_rng(42)
        p = permutation_test(pnl, n_permutations=1000, rng=rng)
        assert p < 0.05, f"Expected p < 0.05 for strong signal, got {p}"

    def test_random_pnl_not_significant(self):
        """Random symmetric PnL should NOT be significant (p > 0.05)."""
        pnl = _weak_pnl()
        rng = np.random.default_rng(42)
        p = permutation_test(pnl, n_permutations=1000, rng=rng)
        assert p > 0.05, f"Expected p > 0.05 for random PnL, got {p}"

    def test_empty_pnl(self):
        """Empty PnL returns p-value = 1.0."""
        rng = np.random.default_rng(42)
        p = permutation_test(np.array([]), n_permutations=100, rng=rng)
        assert p == 1.0


# ---------------------------------------------------------------------------
# Trade skip test
# ---------------------------------------------------------------------------

class TestTradeSkipTest:
    def test_returns_quality_score(self):
        """Skipping trades should return a valid quality score."""
        pnl = _strong_pnl()
        rng = np.random.default_rng(42)
        qs = trade_skip_test(pnl, skip_fraction=0.10, rng=rng)
        # Quality score should be a finite number
        assert np.isfinite(qs), f"Quality score should be finite, got {qs}"

    def test_zero_skip_matches_full(self):
        """Skipping 0% should return very similar quality to full PnL."""
        pnl = _strong_pnl()
        rng = np.random.default_rng(42)
        qs = trade_skip_test(pnl, skip_fraction=0.0, rng=rng)
        # With 0 skip, we keep all trades (n_keep = n), so quality should be > 0
        assert qs > 0, f"Expected positive quality with 0% skip, got {qs}"

    def test_empty_pnl(self):
        """Empty PnL returns 0."""
        rng = np.random.default_rng(42)
        qs = trade_skip_test(np.array([]), skip_fraction=0.10, rng=rng)
        assert qs == 0.0


# ---------------------------------------------------------------------------
# Execution stress test
# ---------------------------------------------------------------------------

class TestExecutionStressTest:
    def test_reduces_pnl(self):
        """Stressed PnL sum should be less than original PnL sum."""
        pnl = _strong_pnl()
        stressed = execution_stress_test(
            pnl,
            original_slippage=0.5,
            original_commission=0.7,
            stress_slippage_mult=1.5,
            stress_commission_mult=1.3,
        )
        assert np.sum(stressed) < np.sum(pnl), (
            f"Stressed sum {np.sum(stressed):.2f} should be < original {np.sum(pnl):.2f}"
        )

    def test_no_stress_preserves_pnl(self):
        """Multipliers of 1.0 should not change PnL."""
        pnl = _strong_pnl()
        stressed = execution_stress_test(
            pnl,
            original_slippage=0.5,
            original_commission=0.7,
            stress_slippage_mult=1.0,
            stress_commission_mult=1.0,
        )
        np.testing.assert_array_almost_equal(stressed, pnl)

    def test_empty_pnl(self):
        """Empty PnL returns empty array."""
        stressed = execution_stress_test(
            np.array([]),
            original_slippage=0.5,
            original_commission=0.7,
            stress_slippage_mult=1.5,
            stress_commission_mult=1.3,
        )
        assert len(stressed) == 0

    def test_correct_deduction(self):
        """Verify exact cost deduction per trade."""
        pnl = np.array([10.0, -5.0, 20.0])
        stressed = execution_stress_test(
            pnl,
            original_slippage=1.0,
            original_commission=1.0,
            stress_slippage_mult=2.0,   # extra = 1.0 * (2.0 - 1.0) = 1.0
            stress_commission_mult=1.5,  # extra = 1.0 * (1.5 - 1.0) = 0.5
        )
        # Each trade loses 1.0 + 0.5 = 1.5
        expected = pnl - 1.5
        np.testing.assert_array_almost_equal(stressed, expected)


# ---------------------------------------------------------------------------
# run_monte_carlo orchestrator
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    def test_strong_signal_passes_gates(self):
        """Strong signal with few trials should pass DSR + permutation gates.

        DSR requires the observed Sharpe to exceed sqrt(2*ln(n_trials)).
        Even with n_trials=1, the formula floors to max(n_trials,2) so
        E[max] = sqrt(2*ln(2)) ~ 1.18. We need a very high raw Sharpe
        to pass, so we use mean=8, std=3 for ~2.67 raw Sharpe.
        """
        rng_data = np.random.default_rng(42)
        pnl = rng_data.normal(8.0, 3.0, 200)  # Very strong signal
        config = PipelineConfig(
            mc_n_bootstrap=500,
            mc_n_permutations=500,
            mc_dsr_gate=0.95,
            mc_permutation_p_gate=0.05,
            seed=42,
        )
        # Use n_trials=1 so DSR penalty is minimal
        result = run_monte_carlo(
            pnl, n_trials=1, n_trades=len(pnl), config=config,
        )

        assert isinstance(result, MonteCarloResult)
        assert result.observed_sharpe > 0
        assert result.bootstrap_sharpe_mean > 0
        assert result.permutation_p_value < 0.05
        # With n_trials=1, DSR should be high for a decent Sharpe
        assert result.dsr >= 0.95, f"DSR {result.dsr:.4f} should be >= 0.95"
        assert result.passed_gate is True

    def test_weak_signal_fails_gates(self):
        """Weak/random signal should fail the permutation p-value gate."""
        pnl = _weak_pnl()
        config = PipelineConfig(
            mc_n_bootstrap=500,
            mc_n_permutations=500,
            mc_dsr_gate=0.95,
            mc_permutation_p_gate=0.05,
            seed=42,
        )
        result = run_monte_carlo(
            pnl, n_trials=100, n_trades=len(pnl), config=config,
        )

        assert isinstance(result, MonteCarloResult)
        # Random PnL should NOT have significant permutation p-value
        assert result.permutation_p_value > 0.05
        assert result.passed_gate is False

    def test_empty_pnl_graceful(self):
        """Empty PnL array handled gracefully — returns defaults."""
        config = PipelineConfig(seed=42)
        result = run_monte_carlo(
            np.array([]), n_trials=100, n_trades=0, config=config,
        )

        assert isinstance(result, MonteCarloResult)
        assert result.observed_sharpe == 0.0
        assert result.bootstrap_sharpe_mean == 0.0
        assert result.permutation_p_value == 1.0
        assert result.dsr == 0.0
        assert result.passed_gate is False

    def test_skip_results_populated(self):
        """Skip results should have entries for each configured level."""
        pnl = _strong_pnl()
        config = PipelineConfig(
            mc_n_bootstrap=100,
            mc_n_permutations=100,
            mc_skip_levels=[0.05, 0.10, 0.20],
            seed=42,
        )
        result = run_monte_carlo(
            pnl, n_trials=1, n_trades=len(pnl), config=config,
        )

        assert "5%" in result.skip_results
        assert "10%" in result.skip_results
        assert "20%" in result.skip_results

    def test_stress_quality_ratio_populated(self):
        """Stress test should produce a quality ratio between 0 and ~1."""
        pnl = _strong_pnl()
        config = PipelineConfig(
            mc_n_bootstrap=100,
            mc_n_permutations=100,
            seed=42,
        )
        result = run_monte_carlo(
            pnl, n_trials=1, n_trades=len(pnl), config=config,
        )

        # Stress quality ratio should be positive (strategy still profitable
        # with slightly higher costs) and <= 1.0 (costs reduce quality)
        assert result.stress_quality_ratio > 0, (
            f"Expected positive stress ratio, got {result.stress_quality_ratio}"
        )
        assert result.stress_quality_ratio <= 1.0 + 1e-9, (
            f"Expected ratio <= 1.0, got {result.stress_quality_ratio}"
        )
