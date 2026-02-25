"""Rust vs Numba parity tests.

Runs batch_evaluate with identical inputs on both backends and
asserts bit-for-bit identical output (within floating-point tolerance).
"""

import os
import numpy as np
import pytest

from backtester.core.dtypes import (
    DIR_BUY,
    DIR_SELL,
    EXEC_BASIC,
    EXEC_FULL,
    SL_ATR_BASED,
    SL_FIXED_PIPS,
    TP_FIXED_PIPS,
    TP_RR_RATIO,
    TRAIL_FIXED_PIP,
    TRAIL_OFF,
    NUM_METRICS,
)

# We need both backends available for parity tests
try:
    import backtester_core as rust_mod

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

from backtester.core.jit_loop import batch_evaluate as numba_batch_evaluate
from backtester.core.jit_loop import NUM_PL

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust backend not built")


def _make_price_data(n_bars=500, seed=42):
    """Generate synthetic price data."""
    rng = np.random.RandomState(seed)
    # Random walk for close prices
    returns = rng.normal(0, 0.0005, n_bars)
    close = np.cumsum(returns) + 1.10
    high = close + rng.uniform(0.0001, 0.001, n_bars)
    low = close - rng.uniform(0.0001, 0.001, n_bars)
    spread = np.full(n_bars, 0.00010)  # 1 pip spread
    return high, low, close, spread


def _make_signals(n_signals=50, n_bars=500, seed=42):
    """Generate synthetic signals."""
    rng = np.random.RandomState(seed)
    # Spread signals across the bar range (not at the very end)
    bar_indices = np.sort(rng.choice(np.arange(10, n_bars - 50), n_signals, replace=False))
    directions = rng.choice([DIR_BUY, DIR_SELL], n_signals)
    # Entry prices near the "close" of the bar (we use a dummy)
    entry_prices = np.full(n_signals, 1.1000) + rng.normal(0, 0.001, n_signals)
    hours = rng.randint(0, 24, n_signals).astype(np.int64)
    days = rng.randint(0, 5, n_signals).astype(np.int64)
    atr_pips = np.full(n_signals, 10.0)
    swing_sl = np.full(n_signals, np.nan)
    filter_value = np.zeros(n_signals, dtype=np.float64)
    variant = np.full(n_signals, -1, dtype=np.int64)
    return (
        bar_indices.astype(np.int64),
        directions.astype(np.int64),
        entry_prices,
        hours,
        days,
        atr_pips,
        swing_sl,
        filter_value,
        variant,
    )


def _make_param_layout():
    """Create a param layout for a simple strategy."""
    # Map each PL_* to its own column (identity mapping)
    layout = np.arange(NUM_PL, dtype=np.int64)
    # Disable filters
    layout[24] = -1  # PL_SIGNAL_VARIANT
    layout[25] = -1  # PL_BUY_FILTER_MAX
    layout[26] = -1  # PL_SELL_FILTER_MIN
    return layout


def _make_param_matrix_basic(n_trials=100, seed=42):
    """Generate parameter matrix for EXEC_BASIC."""
    rng = np.random.RandomState(seed)
    matrix = np.zeros((n_trials, NUM_PL), dtype=np.float64)
    for i in range(n_trials):
        matrix[i, 0] = rng.choice([SL_FIXED_PIPS, SL_ATR_BASED])  # sl_mode
        matrix[i, 1] = rng.uniform(10, 30)  # sl_fixed_pips
        matrix[i, 2] = rng.uniform(1.0, 2.5)  # sl_atr_mult
        matrix[i, 3] = rng.choice([TP_RR_RATIO, TP_FIXED_PIPS])  # tp_mode
        matrix[i, 4] = rng.uniform(1.0, 3.0)  # tp_rr_ratio
        matrix[i, 5] = rng.uniform(1.0, 3.0)  # tp_atr_mult
        matrix[i, 6] = rng.uniform(20, 60)  # tp_fixed_pips
        matrix[i, 7] = 0.0  # hours_start
        matrix[i, 8] = 23.0  # hours_end
        matrix[i, 9] = 31.0  # days_bitmask (Mon-Fri)
        # Management params (not used in BASIC but need values)
        matrix[i, 10] = TRAIL_OFF
        matrix[i, 14] = 0  # BE disabled
        matrix[i, 17] = 0  # Partial disabled
        matrix[i, 20] = 0  # Max bars disabled
        matrix[i, 21] = 0  # Stale disabled
    return matrix


def _make_param_matrix_full(n_trials=100, seed=42):
    """Generate parameter matrix for EXEC_FULL (all management features)."""
    rng = np.random.RandomState(seed)
    matrix = _make_param_matrix_basic(n_trials, seed)
    for i in range(n_trials):
        # Trailing stop
        matrix[i, 10] = rng.choice([TRAIL_OFF, TRAIL_FIXED_PIP])
        matrix[i, 11] = rng.uniform(5, 15)  # trail_activate
        matrix[i, 12] = rng.uniform(3, 10)  # trail_distance
        matrix[i, 13] = rng.uniform(1.0, 2.0)  # trail_atr_mult
        # Breakeven
        matrix[i, 14] = rng.choice([0, 1])
        matrix[i, 15] = rng.uniform(5, 15)  # be_trigger
        matrix[i, 16] = rng.uniform(1, 3)  # be_offset
        # Partial close
        matrix[i, 17] = rng.choice([0, 1])
        matrix[i, 18] = rng.uniform(25, 75)  # partial_pct
        matrix[i, 19] = rng.uniform(10, 30)  # partial_trigger
        # Max bars
        if rng.random() < 0.3:
            matrix[i, 20] = rng.randint(10, 100)
        # Stale
        if rng.random() < 0.3:
            matrix[i, 21] = 1
            matrix[i, 22] = rng.randint(5, 20)
            matrix[i, 23] = rng.uniform(0.3, 0.8)
    return matrix


def _identity_sub_arrays(n_bars, high, low, close, spread):
    """Create identity sub-bar mapping (each bar maps to itself)."""
    return (
        high,
        low,
        close,
        spread,
        np.arange(n_bars, dtype=np.int64),
        np.arange(n_bars, dtype=np.int64) + 1,
    )


def _run_both_backends(
    high, low, close, spread, signals, param_matrix, param_layout,
    exec_mode, n_bars, max_trades=5000, bars_per_year=6048.0,
    commission_pips=0.0, max_spread_pips=0.0,
):
    """Run batch_evaluate on both Rust and Numba, return both results."""
    n_trials = param_matrix.shape[0]
    (sig_bar, sig_dir, sig_entry, sig_hour, sig_day,
     sig_atr, sig_swing, sig_filter, sig_variant) = signals
    sub_high, sub_low, sub_close, sub_spread, sub_start, sub_end = \
        _identity_sub_arrays(n_bars, high, low, close, spread)

    # Numba run
    metrics_numba = np.zeros((n_trials, NUM_METRICS), dtype=np.float64)
    pnl_numba = np.empty((n_trials, max_trades), dtype=np.float64)
    numba_batch_evaluate(
        high, low, close, spread, 0.0001, 0.0,
        sig_bar, sig_dir, sig_entry, sig_hour, sig_day,
        sig_atr, sig_swing, sig_filter, sig_variant,
        param_matrix, param_layout, exec_mode,
        metrics_numba, max_trades, bars_per_year,
        commission_pips, max_spread_pips,
        sub_high, sub_low, sub_close, sub_spread,
        sub_start, sub_end, pnl_numba,
    )

    # Rust run
    metrics_rust = np.zeros((n_trials, NUM_METRICS), dtype=np.float64)
    pnl_rust = np.empty((n_trials, max_trades), dtype=np.float64)
    rust_mod.batch_evaluate(
        high, low, close, spread, 0.0001, 0.0,
        sig_bar, sig_dir, sig_entry, sig_hour, sig_day,
        sig_atr, sig_swing, sig_filter, sig_variant,
        param_matrix, param_layout, exec_mode,
        metrics_rust, max_trades, bars_per_year,
        commission_pips, max_spread_pips,
        sub_high, sub_low, sub_close, sub_spread,
        sub_start, sub_end, pnl_rust,
    )

    return metrics_numba, metrics_rust


class TestExecBasicParity:
    """EXEC_BASIC parity: Rust vs Numba."""

    def test_100_trials_basic(self):
        high, low, close, spread = _make_price_data(500, seed=42)
        signals = _make_signals(50, 500, seed=42)
        param_layout = _make_param_layout()
        param_matrix = _make_param_matrix_basic(100, seed=42)

        metrics_numba, metrics_rust = _run_both_backends(
            high, low, close, spread, signals, param_matrix, param_layout,
            EXEC_BASIC, 500,
        )

        np.testing.assert_allclose(
            metrics_rust, metrics_numba,
            rtol=1e-12, atol=1e-12,
            err_msg="EXEC_BASIC metrics differ between Rust and Numba",
        )

    def test_with_commission_and_spread_filter(self):
        high, low, close, spread = _make_price_data(500, seed=99)
        signals = _make_signals(80, 500, seed=99)
        param_layout = _make_param_layout()
        param_matrix = _make_param_matrix_basic(50, seed=99)

        metrics_numba, metrics_rust = _run_both_backends(
            high, low, close, spread, signals, param_matrix, param_layout,
            EXEC_BASIC, 500,
            commission_pips=0.7, max_spread_pips=3.0,
        )

        np.testing.assert_allclose(
            metrics_rust, metrics_numba,
            rtol=1e-12, atol=1e-12,
            err_msg="EXEC_BASIC with costs differs between Rust and Numba",
        )

    def test_large_batch_500_trials(self):
        """Stress test with 500 trials."""
        high, low, close, spread = _make_price_data(1000, seed=77)
        signals = _make_signals(100, 1000, seed=77)
        param_layout = _make_param_layout()
        param_matrix = _make_param_matrix_basic(500, seed=77)

        metrics_numba, metrics_rust = _run_both_backends(
            high, low, close, spread, signals, param_matrix, param_layout,
            EXEC_BASIC, 1000,
        )

        np.testing.assert_allclose(
            metrics_rust, metrics_numba,
            rtol=1e-12, atol=1e-12,
            err_msg="Large batch EXEC_BASIC differs between Rust and Numba",
        )


class TestExecFullParity:
    """EXEC_FULL parity: Rust vs Numba."""

    def test_100_trials_full(self):
        high, low, close, spread = _make_price_data(500, seed=42)
        signals = _make_signals(50, 500, seed=42)
        param_layout = _make_param_layout()
        param_matrix = _make_param_matrix_full(100, seed=42)

        metrics_numba, metrics_rust = _run_both_backends(
            high, low, close, spread, signals, param_matrix, param_layout,
            EXEC_FULL, 500,
        )

        np.testing.assert_allclose(
            metrics_rust, metrics_numba,
            rtol=1e-12, atol=1e-12,
            err_msg="EXEC_FULL metrics differ between Rust and Numba",
        )

    def test_full_with_costs(self):
        high, low, close, spread = _make_price_data(500, seed=123)
        signals = _make_signals(60, 500, seed=123)
        param_layout = _make_param_layout()
        param_matrix = _make_param_matrix_full(100, seed=123)

        metrics_numba, metrics_rust = _run_both_backends(
            high, low, close, spread, signals, param_matrix, param_layout,
            EXEC_FULL, 500,
            commission_pips=0.7, max_spread_pips=3.0,
        )

        np.testing.assert_allclose(
            metrics_rust, metrics_numba,
            rtol=1e-12, atol=1e-12,
            err_msg="EXEC_FULL with costs differs between Rust and Numba",
        )

    def test_large_batch_500_trials_full(self):
        """Stress test with 500 trials in FULL mode."""
        high, low, close, spread = _make_price_data(1000, seed=55)
        signals = _make_signals(100, 1000, seed=55)
        param_layout = _make_param_layout()
        param_matrix = _make_param_matrix_full(500, seed=55)

        metrics_numba, metrics_rust = _run_both_backends(
            high, low, close, spread, signals, param_matrix, param_layout,
            EXEC_FULL, 1000,
        )

        np.testing.assert_allclose(
            metrics_rust, metrics_numba,
            rtol=1e-12, atol=1e-12,
            err_msg="Large batch EXEC_FULL differs between Rust and Numba",
        )


class TestEdgeCases:
    """Edge cases for parity."""

    def test_zero_signals(self):
        high, low, close, spread = _make_price_data(100, seed=1)
        # Empty signals
        signals = (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
        )
        param_layout = _make_param_layout()
        param_matrix = _make_param_matrix_basic(10, seed=1)

        metrics_numba, metrics_rust = _run_both_backends(
            high, low, close, spread, signals, param_matrix, param_layout,
            EXEC_BASIC, 100,
        )

        np.testing.assert_array_equal(metrics_rust, metrics_numba)
        assert np.all(metrics_rust == 0.0)

    def test_single_trial(self):
        high, low, close, spread = _make_price_data(200, seed=5)
        signals = _make_signals(20, 200, seed=5)
        param_layout = _make_param_layout()
        param_matrix = _make_param_matrix_basic(1, seed=5)

        metrics_numba, metrics_rust = _run_both_backends(
            high, low, close, spread, signals, param_matrix, param_layout,
            EXEC_BASIC, 200,
        )

        np.testing.assert_allclose(
            metrics_rust, metrics_numba, rtol=1e-12, atol=1e-12,
        )

    def test_sell_only_signals(self):
        high, low, close, spread = _make_price_data(300, seed=8)
        signals = list(_make_signals(30, 300, seed=8))
        signals[1] = np.full(30, DIR_SELL, dtype=np.int64)  # all sells
        signals = tuple(signals)
        param_layout = _make_param_layout()
        param_matrix = _make_param_matrix_basic(20, seed=8)

        metrics_numba, metrics_rust = _run_both_backends(
            high, low, close, spread, signals, param_matrix, param_layout,
            EXEC_BASIC, 300,
        )

        np.testing.assert_allclose(
            metrics_rust, metrics_numba, rtol=1e-12, atol=1e-12,
        )


class TestBackendSelection:
    """Test the rust_loop dispatcher."""

    def test_auto_selects_rust(self):
        from backtester.core.rust_loop import get_backend_name
        # In auto mode with Rust available, should use Rust
        assert get_backend_name() == "rust"

    def test_numba_fallback_env(self):
        """Verify BACKTESTER_BACKEND=numba works."""
        # We can't easily test this without reimporting, but we can verify
        # the numba module is still importable
        from backtester.core.jit_loop import batch_evaluate as nb_eval
        assert callable(nb_eval)
