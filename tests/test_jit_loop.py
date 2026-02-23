"""Tests for JIT-compiled batch evaluator."""

import numpy as np
import pytest

from backtester.core.dtypes import (
    DIR_BUY,
    DIR_SELL,
    EXEC_BASIC,
    EXEC_FULL,
    EXIT_BREAKEVEN,
    EXIT_MAX_BARS,
    EXIT_SL,
    EXIT_STALE,
    EXIT_TP,
    EXIT_TRAILING,
    M_MAX_DD_PCT,
    M_PROFIT_FACTOR,
    M_QUALITY,
    M_R_SQUARED,
    M_RETURN_PCT,
    M_SHARPE,
    M_SORTINO,
    M_TRADES,
    M_ULCER,
    M_WIN_RATE,
    NUM_METRICS,
    SL_FIXED_PIPS,
    TP_RR_RATIO,
    TRAIL_OFF,
    TRAIL_FIXED_PIP,
)
from backtester.core.jit_loop import (
    NUM_PL,
    PL_BREAKEVEN_ENABLED,
    PL_BREAKEVEN_OFFSET,
    PL_BREAKEVEN_TRIGGER,
    PL_BUY_FILTER_MAX,
    PL_DAYS_BITMASK,
    PL_HOURS_END,
    PL_HOURS_START,
    PL_MAX_BARS,
    PL_PARTIAL_ENABLED,
    PL_PARTIAL_PCT,
    PL_PARTIAL_TRIGGER,
    PL_SELL_FILTER_MIN,
    PL_SIGNAL_VARIANT,
    PL_SL_ATR_MULT,
    PL_SL_FIXED_PIPS,
    PL_SL_MODE,
    PL_STALE_ATR_THRESH,
    PL_STALE_BARS,
    PL_STALE_ENABLED,
    PL_TP_ATR_MULT,
    PL_TP_FIXED_PIPS,
    PL_TP_MODE,
    PL_TP_RR_RATIO,
    PL_TRAILING_MODE,
    PL_TRAIL_ACTIVATE,
    PL_TRAIL_ATR_MULT,
    PL_TRAIL_DISTANCE,
    _compute_sl_tp,
    _simulate_trade_basic,
    _simulate_trade_full,
    batch_evaluate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(n_bars: int = 100, base: float = 1.1000, pip: float = 0.0001):
    """Create synthetic price data with known moves."""
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 5 * pip, n_bars)
    close = np.cumsum(returns) + base
    high = close + rng.uniform(0, 10 * pip, n_bars)
    low = close - rng.uniform(0, 10 * pip, n_bars)
    spread = np.full(n_bars, 1.0 * pip)  # 1 pip spread in price units
    return high, low, close, spread


def _make_simple_prices(moves: list[tuple[float, float]], base: float = 1.1000, pip: float = 0.0001):
    """Create prices with known high/low relative to base.

    moves: list of (low_offset_pips, high_offset_pips) per bar.
    """
    n = len(moves) + 1  # +1 for entry bar
    high = np.full(n, base, dtype=np.float64)
    low = np.full(n, base, dtype=np.float64)
    close = np.full(n, base, dtype=np.float64)
    spread = np.zeros(n, dtype=np.float64)

    for i, (lo, hi) in enumerate(moves):
        bar = i + 1  # Bar 0 is entry
        high[bar] = base + hi * pip
        low[bar] = base + lo * pip
        close[bar] = base + (hi + lo) / 2 * pip

    return high, low, close, spread


def _default_param_layout() -> np.ndarray:
    """Create a simple param layout where PL_* maps to column PL_*."""
    return np.arange(NUM_PL, dtype=np.int64)


def _basic_params(sl_pips: float = 30.0, tp_rr: float = 2.0) -> np.ndarray:
    """Create a single trial param row for basic mode."""
    params = np.zeros(NUM_PL, dtype=np.float64)
    params[PL_SL_MODE] = SL_FIXED_PIPS
    params[PL_SL_FIXED_PIPS] = sl_pips
    params[PL_SL_ATR_MULT] = 1.5
    params[PL_TP_MODE] = TP_RR_RATIO
    params[PL_TP_RR_RATIO] = tp_rr
    params[PL_TP_ATR_MULT] = 2.0
    params[PL_TP_FIXED_PIPS] = 60.0
    params[PL_HOURS_START] = 0
    params[PL_HOURS_END] = 23
    params[PL_DAYS_BITMASK] = 31  # Mon-Fri
    params[PL_TRAILING_MODE] = TRAIL_OFF
    params[PL_TRAIL_ACTIVATE] = 0.0
    params[PL_TRAIL_DISTANCE] = 10.0
    params[PL_TRAIL_ATR_MULT] = 2.0
    params[PL_BREAKEVEN_ENABLED] = 0
    params[PL_BREAKEVEN_TRIGGER] = 20.0
    params[PL_BREAKEVEN_OFFSET] = 2.0
    params[PL_PARTIAL_ENABLED] = 0
    params[PL_PARTIAL_PCT] = 50.0
    params[PL_PARTIAL_TRIGGER] = 30.0
    params[PL_MAX_BARS] = 0
    params[PL_STALE_ENABLED] = 0
    params[PL_STALE_BARS] = 50
    params[PL_STALE_ATR_THRESH] = 0.5
    # Signal filter params: -1 = disabled (accept all signals)
    params[PL_SIGNAL_VARIANT] = -1.0
    params[PL_BUY_FILTER_MAX] = -1.0
    params[PL_SELL_FILTER_MIN] = -1.0
    return params


# ---------------------------------------------------------------------------
# _compute_sl_tp tests
# ---------------------------------------------------------------------------

class TestComputeSLTP:
    def test_fixed_pips_buy(self):
        pip = 0.0001
        entry = 1.1000
        sl_p, tp_p, sl_pips, tp_pips = _compute_sl_tp(
            DIR_BUY, entry, 20.0, pip,
            SL_FIXED_PIPS, 30.0, 1.5, np.nan,
            TP_RR_RATIO, 2.0, 2.0, 60.0,
        )
        assert abs(sl_pips - 30.0) < 0.01
        assert abs(tp_pips - 60.0) < 0.01
        assert sl_p < entry
        assert tp_p > entry

    def test_fixed_pips_sell(self):
        pip = 0.0001
        entry = 1.1000
        sl_p, tp_p, sl_pips, tp_pips = _compute_sl_tp(
            DIR_SELL, entry, 20.0, pip,
            SL_FIXED_PIPS, 30.0, 1.5, np.nan,
            TP_RR_RATIO, 2.0, 2.0, 60.0,
        )
        assert sl_p > entry
        assert tp_p < entry

    def test_tp_gte_sl_enforced(self):
        """TP distance must be >= SL distance."""
        pip = 0.0001
        entry = 1.1000
        sl_p, tp_p, sl_pips, tp_pips = _compute_sl_tp(
            DIR_BUY, entry, 20.0, pip,
            SL_FIXED_PIPS, 50.0, 1.5, np.nan,
            TP_RR_RATIO, 0.5, 2.0, 60.0,  # RR=0.5 would make TP < SL
        )
        assert tp_pips >= sl_pips


# ---------------------------------------------------------------------------
# _simulate_trade_basic tests
# ---------------------------------------------------------------------------

class TestSimulateTradeBasic:
    def test_sl_hit_buy(self):
        """BUY trade: price drops to SL."""
        pip = 0.0001
        # Bar 0: entry. Bar 1: low drops 40 pips below entry
        high, low, close, spread = _make_simple_prices(
            [(-40, 5)],  # Bar 1: low at -40 pips, high at +5
            base=1.1000,
        )
        pnl, exit_code = _simulate_trade_basic(
            DIR_BUY, 0, 1.1000,
            sl_price=1.1000 - 30 * pip,  # SL at -30 pips
            tp_price=1.1000 + 60 * pip,  # TP at +60 pips
            high=high, low=low, close=close, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        assert exit_code == EXIT_SL
        assert pnl < 0

    def test_tp_hit_buy(self):
        """BUY trade: price rises to TP."""
        pip = 0.0001
        high, low, close, spread = _make_simple_prices(
            [(- 5, 70)],  # Bar 1: high at +70 pips
            base=1.1000,
        )
        pnl, exit_code = _simulate_trade_basic(
            DIR_BUY, 0, 1.1000,
            sl_price=1.1000 - 30 * pip,
            tp_price=1.1000 + 60 * pip,
            high=high, low=low, close=close, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        assert exit_code == EXIT_TP
        assert pnl > 0

    def test_same_bar_tiebreak_sl_wins(self):
        """When both SL and TP are hit on the same bar, SL wins (conservative)."""
        pip = 0.0001
        high, low, close, spread = _make_simple_prices(
            [(-40, 70)],  # Bar 1: both SL (-30) and TP (+60) hit
            base=1.1000,
        )
        pnl, exit_code = _simulate_trade_basic(
            DIR_BUY, 0, 1.1000,
            sl_price=1.1000 - 30 * pip,
            tp_price=1.1000 + 60 * pip,
            high=high, low=low, close=close, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        assert exit_code == EXIT_SL  # Conservative tiebreak

    def test_sell_sl_hit(self):
        """SELL trade: price rises to SL."""
        pip = 0.0001
        high, low, close, spread = _make_simple_prices(
            [(-5, 40)],  # Bar 1: high at +40 pips
            base=1.1000,
        )
        pnl, exit_code = _simulate_trade_basic(
            DIR_SELL, 0, 1.1000,
            sl_price=1.1000 + 30 * pip,  # SL above for sell
            tp_price=1.1000 - 60 * pip,  # TP below for sell
            high=high, low=low, close=close, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        assert exit_code == EXIT_SL
        assert pnl < 0

    def test_sell_tp_hit(self):
        """SELL trade: price drops to TP."""
        pip = 0.0001
        high, low, close, spread = _make_simple_prices(
            [(-70, 5)],  # Bar 1: low at -70 pips
            base=1.1000,
        )
        pnl, exit_code = _simulate_trade_basic(
            DIR_SELL, 0, 1.1000,
            sl_price=1.1000 + 30 * pip,
            tp_price=1.1000 - 60 * pip,
            high=high, low=low, close=close, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        assert exit_code == EXIT_TP
        assert pnl > 0

    def test_spread_costs(self):
        """Spread should reduce BUY profit."""
        pip = 0.0001
        high, low, close, spread = _make_simple_prices(
            [(-5, 70)],
            base=1.1000,
        )
        # No spread
        pnl_no_spread, _ = _simulate_trade_basic(
            DIR_BUY, 0, 1.1000,
            sl_price=1.0970, tp_price=1.1060,
            high=high, low=low, close=close, spread_arr=np.zeros_like(spread),
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        # With 2 pip spread (in price units)
        spread_2 = np.full_like(spread, 2.0 * pip)
        pnl_with_spread, _ = _simulate_trade_basic(
            DIR_BUY, 0, 1.1000,
            sl_price=1.0970, tp_price=1.1060,
            high=high, low=low, close=close, spread_arr=spread_2,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        assert pnl_with_spread < pnl_no_spread


# ---------------------------------------------------------------------------
# batch_evaluate tests
# ---------------------------------------------------------------------------

class TestBatchEvaluate:
    def test_single_trial_basic(self):
        """Single trial with a BUY signal hitting TP."""
        pip = 0.0001
        high, low, close, spread = _make_simple_prices(
            [(- 5, 70)] * 3,  # 3 bars all with high enough for TP
            base=1.1000,
        )

        # One signal at bar 0
        sig_bar = np.array([0], dtype=np.int64)
        sig_dir = np.array([DIR_BUY], dtype=np.int64)
        sig_entry = np.array([1.1000], dtype=np.float64)
        sig_hour = np.array([10], dtype=np.int64)
        sig_day = np.array([1], dtype=np.int64)  # Tuesday
        sig_atr = np.array([20.0], dtype=np.float64)
        sig_swing = np.array([np.nan], dtype=np.float64)

        params = _basic_params(sl_pips=30.0, tp_rr=2.0).reshape(1, -1)
        layout = _default_param_layout()
        metrics = np.zeros((1, NUM_METRICS), dtype=np.float64)

        batch_evaluate(
            high, low, close, spread, pip, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_BASIC, metrics, 1000, 6048.0,
            0.0, 0.0,
        )

        assert metrics[0, M_TRADES] == 1.0
        assert metrics[0, M_WIN_RATE] == 1.0  # TP hit = win

    def test_no_signals(self):
        """No signals should produce 0 trades."""
        high, low, close, spread = _make_price_data(50)

        sig_bar = np.array([], dtype=np.int64)
        sig_dir = np.array([], dtype=np.int64)
        sig_entry = np.array([], dtype=np.float64)
        sig_hour = np.array([], dtype=np.int64)
        sig_day = np.array([], dtype=np.int64)
        sig_atr = np.array([], dtype=np.float64)
        sig_swing = np.array([], dtype=np.float64)

        params = _basic_params().reshape(1, -1)
        layout = _default_param_layout()
        metrics = np.zeros((1, NUM_METRICS), dtype=np.float64)

        batch_evaluate(
            high, low, close, spread, 0.0001, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_BASIC, metrics, 1000, 6048.0,
            0.0, 0.0,
        )

        assert metrics[0, M_TRADES] == 0.0

    def test_day_filter_blocks_signal(self):
        """Signal on Saturday (day=5) should be blocked by Mon-Fri bitmask."""
        pip = 0.0001
        high, low, close, spread = _make_simple_prices(
            [(-5, 70)],
            base=1.1000,
        )

        sig_bar = np.array([0], dtype=np.int64)
        sig_dir = np.array([DIR_BUY], dtype=np.int64)
        sig_entry = np.array([1.1000], dtype=np.float64)
        sig_hour = np.array([10], dtype=np.int64)
        sig_day = np.array([5], dtype=np.int64)  # Saturday
        sig_atr = np.array([20.0], dtype=np.float64)
        sig_swing = np.array([np.nan], dtype=np.float64)

        params = _basic_params()
        params[PL_DAYS_BITMASK] = 31  # Mon-Fri only
        params = params.reshape(1, -1)
        layout = _default_param_layout()
        metrics = np.zeros((1, NUM_METRICS), dtype=np.float64)

        batch_evaluate(
            high, low, close, spread, pip, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_BASIC, metrics, 1000, 6048.0,
            0.0, 0.0,
        )

        assert metrics[0, M_TRADES] == 0.0  # Filtered out

    def test_hour_filter(self):
        """Signal at hour 22 blocked by 8-17 filter."""
        pip = 0.0001
        high, low, close, spread = _make_simple_prices(
            [(-5, 70)],
            base=1.1000,
        )

        sig_bar = np.array([0], dtype=np.int64)
        sig_dir = np.array([DIR_BUY], dtype=np.int64)
        sig_entry = np.array([1.1000], dtype=np.float64)
        sig_hour = np.array([22], dtype=np.int64)
        sig_day = np.array([1], dtype=np.int64)
        sig_atr = np.array([20.0], dtype=np.float64)
        sig_swing = np.array([np.nan], dtype=np.float64)

        params = _basic_params()
        params[PL_HOURS_START] = 8
        params[PL_HOURS_END] = 17
        params = params.reshape(1, -1)
        layout = _default_param_layout()
        metrics = np.zeros((1, NUM_METRICS), dtype=np.float64)

        batch_evaluate(
            high, low, close, spread, pip, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_BASIC, metrics, 1000, 6048.0,
            0.0, 0.0,
        )

        assert metrics[0, M_TRADES] == 0.0

    def test_multi_trial_parallel(self):
        """Multiple trials should produce different results with different params."""
        pip = 0.0001
        high, low, close, spread = _make_simple_prices(
            [(-5, 15), (-20, 10), (-35, 5), (-5, 70)],
            base=1.1000,
        )

        sig_bar = np.array([0], dtype=np.int64)
        sig_dir = np.array([DIR_BUY], dtype=np.int64)
        sig_entry = np.array([1.1000], dtype=np.float64)
        sig_hour = np.array([10], dtype=np.int64)
        sig_day = np.array([1], dtype=np.int64)
        sig_atr = np.array([20.0], dtype=np.float64)
        sig_swing = np.array([np.nan], dtype=np.float64)

        # Trial 0: tight SL (10 pips) — should get stopped out
        # Trial 1: wide SL (50 pips) — should survive to TP
        p0 = _basic_params(sl_pips=10.0, tp_rr=2.0)
        p1 = _basic_params(sl_pips=50.0, tp_rr=1.0)
        params = np.vstack([p0, p1])
        layout = _default_param_layout()
        metrics = np.zeros((2, NUM_METRICS), dtype=np.float64)

        batch_evaluate(
            high, low, close, spread, pip, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_BASIC, metrics, 1000, 6048.0,
            0.0, 0.0,
        )

        # Both should have 1 trade
        assert metrics[0, M_TRADES] == 1.0
        assert metrics[1, M_TRADES] == 1.0
        # Tight SL should lose (SL hit), wide SL should win (TP hit)
        assert metrics[0, M_WIN_RATE] == 0.0  # SL hit
        assert metrics[1, M_WIN_RATE] == 1.0  # TP hit

    def test_multiple_signals(self):
        """Multiple signals should produce multiple trades."""
        pip = 0.0001
        base = 1.1000
        # Create 20 bars with gradual upward move
        n_bars = 20
        high = np.full(n_bars, base, dtype=np.float64)
        low = np.full(n_bars, base, dtype=np.float64)
        close = np.full(n_bars, base, dtype=np.float64)
        for i in range(n_bars):
            high[i] = base + (i * 5 + 10) * pip
            low[i] = base + (i * 5 - 10) * pip
            close[i] = base + i * 5 * pip
        spread = np.zeros(n_bars, dtype=np.float64)

        # Two buy signals
        sig_bar = np.array([0, 5], dtype=np.int64)
        sig_dir = np.array([DIR_BUY, DIR_BUY], dtype=np.int64)
        sig_entry = np.array([base, base + 25 * pip], dtype=np.float64)
        sig_hour = np.array([10, 14], dtype=np.int64)
        sig_day = np.array([1, 2], dtype=np.int64)
        sig_atr = np.array([20.0, 20.0], dtype=np.float64)
        sig_swing = np.array([np.nan, np.nan], dtype=np.float64)

        params = _basic_params(sl_pips=20.0, tp_rr=1.0).reshape(1, -1)
        layout = _default_param_layout()
        metrics = np.zeros((1, NUM_METRICS), dtype=np.float64)

        batch_evaluate(
            high, low, close, spread, pip, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_BASIC, metrics, 1000, 6048.0,
            0.0, 0.0,
        )

        assert metrics[0, M_TRADES] == 2.0


# ---------------------------------------------------------------------------
# Full mode management tests
# ---------------------------------------------------------------------------

class TestFullMode:
    def test_max_bars_exit(self):
        """Trade should exit at max_bars limit."""
        pip = 0.0001
        # 10 bars with no SL/TP hit (price stays flat)
        n = 10
        base = 1.1000
        high = np.full(n, base + 5 * pip, dtype=np.float64)
        low = np.full(n, base - 5 * pip, dtype=np.float64)
        close = np.full(n, base, dtype=np.float64)
        spread = np.zeros(n, dtype=np.float64)

        sig_bar = np.array([0], dtype=np.int64)
        sig_dir = np.array([DIR_BUY], dtype=np.int64)
        sig_entry = np.array([base], dtype=np.float64)
        sig_hour = np.array([10], dtype=np.int64)
        sig_day = np.array([1], dtype=np.int64)
        sig_atr = np.array([20.0], dtype=np.float64)
        sig_swing = np.array([np.nan], dtype=np.float64)

        params = _basic_params(sl_pips=50.0, tp_rr=3.0)
        params[PL_MAX_BARS] = 3  # Exit after 3 bars
        params = params.reshape(1, -1)
        layout = _default_param_layout()
        metrics = np.zeros((1, NUM_METRICS), dtype=np.float64)

        batch_evaluate(
            high, low, close, spread, pip, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_FULL, metrics, 1000, 6048.0,
            0.0, 0.0,
        )

        assert metrics[0, M_TRADES] == 1.0

    def test_trailing_stop(self):
        """Trailing stop should lock in profits."""
        pip = 0.0001
        base = 1.1000
        # Price goes up 50 pips, then drops back 25 pips
        moves = [
            (-5, 20),   # Bar 1: +20 high
            (-5, 40),   # Bar 2: +40 high (trail activates at 30)
            (-5, 50),   # Bar 3: +50 high (trail moves up)
            (-30, 10),  # Bar 4: drops, trail should trigger
        ]
        high, low, close, spread = _make_simple_prices(moves, base=base)

        sig_bar = np.array([0], dtype=np.int64)
        sig_dir = np.array([DIR_BUY], dtype=np.int64)
        sig_entry = np.array([base], dtype=np.float64)
        sig_hour = np.array([10], dtype=np.int64)
        sig_day = np.array([1], dtype=np.int64)
        sig_atr = np.array([20.0], dtype=np.float64)
        sig_swing = np.array([np.nan], dtype=np.float64)

        params = _basic_params(sl_pips=50.0, tp_rr=5.0)  # Wide SL/TP
        params[PL_TRAILING_MODE] = TRAIL_FIXED_PIP
        params[PL_TRAIL_ACTIVATE] = 30.0  # Activate at +30 pips
        params[PL_TRAIL_DISTANCE] = 15.0  # Trail 15 pips behind
        params = params.reshape(1, -1)
        layout = _default_param_layout()
        metrics = np.zeros((1, NUM_METRICS), dtype=np.float64)

        batch_evaluate(
            high, low, close, spread, pip, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_FULL, metrics, 1000, 6048.0,
            0.0, 0.0,
        )

        assert metrics[0, M_TRADES] == 1.0
        # Should have some positive PnL from trailing lock
        assert metrics[0, M_WIN_RATE] == 1.0

    def test_breakeven_lock(self):
        """Breakeven should move SL to entry + offset."""
        pip = 0.0001
        base = 1.1000
        # Price goes up 25 pips (triggers BE at 20), then drops back below entry
        moves = [
            (-5, 25),   # Bar 1: triggers BE at 20 pips
            (-15, 5),   # Bar 2: drops but BE holds
            (-3, -1),   # Bar 3: hovers near entry
        ]
        high, low, close, spread = _make_simple_prices(moves, base=base)

        sig_bar = np.array([0], dtype=np.int64)
        sig_dir = np.array([DIR_BUY], dtype=np.int64)
        sig_entry = np.array([base], dtype=np.float64)
        sig_hour = np.array([10], dtype=np.int64)
        sig_day = np.array([1], dtype=np.int64)
        sig_atr = np.array([20.0], dtype=np.float64)
        sig_swing = np.array([np.nan], dtype=np.float64)

        params = _basic_params(sl_pips=50.0, tp_rr=5.0)
        params[PL_BREAKEVEN_ENABLED] = 1
        params[PL_BREAKEVEN_TRIGGER] = 20.0
        params[PL_BREAKEVEN_OFFSET] = 2.0
        params = params.reshape(1, -1)
        layout = _default_param_layout()
        metrics = np.zeros((1, NUM_METRICS), dtype=np.float64)

        batch_evaluate(
            high, low, close, spread, pip, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_FULL, metrics, 1000, 6048.0,
            0.0, 0.0,
        )

        assert metrics[0, M_TRADES] == 1.0
        # BE offset = 2 pips, so exit should be slightly profitable
        assert metrics[0, M_WIN_RATE] == 1.0


# ---------------------------------------------------------------------------
# Execution cost tests
# ---------------------------------------------------------------------------

class TestExecutionCosts:
    def test_sell_exit_spread_deducted(self):
        """SELL trade PnL should be reduced by exit bar spread."""
        pip = 0.0001
        # SELL TP hit: price drops 70 pips
        high, low, close, spread = _make_simple_prices(
            [(-70, 5)],
            base=1.1000,
        )
        # Zero spread
        pnl_no_spread, _ = _simulate_trade_basic(
            DIR_SELL, 0, 1.1000,
            sl_price=1.1000 + 30 * pip,
            tp_price=1.1000 - 60 * pip,
            high=high, low=low, close=close,
            spread_arr=np.zeros_like(spread),
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        # With 1.5 pip spread (in price units)
        spread_15 = np.full_like(spread, 1.5 * pip)
        pnl_with_spread, _ = _simulate_trade_basic(
            DIR_SELL, 0, 1.1000,
            sl_price=1.1000 + 30 * pip,
            tp_price=1.1000 - 60 * pip,
            high=high, low=low, close=close,
            spread_arr=spread_15,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        # SELL PnL reduced by 1.5 pips (exit spread)
        assert abs((pnl_no_spread - pnl_with_spread) - 1.5) < 0.01

    def test_buy_no_exit_spread(self):
        """BUY trade PnL should NOT be affected by exit bar spread (only entry)."""
        pip = 0.0001
        # BUY TP hit: price rises 70 pips
        high, low, close, spread = _make_simple_prices(
            [(-5, 70)],
            base=1.1000,
        )
        # Zero spread
        pnl_no_spread, _ = _simulate_trade_basic(
            DIR_BUY, 0, 1.1000,
            sl_price=1.1000 - 30 * pip,
            tp_price=1.1000 + 60 * pip,
            high=high, low=low, close=close,
            spread_arr=np.zeros_like(spread),
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        # With 1.5 pip spread — BUY pays at entry, not exit
        spread_15 = np.full_like(spread, 1.5 * pip)
        pnl_with_spread, _ = _simulate_trade_basic(
            DIR_BUY, 0, 1.1000,
            sl_price=1.1000 - 30 * pip,
            tp_price=1.1000 + 60 * pip,
            high=high, low=low, close=close,
            spread_arr=spread_15,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        # BUY spread cost = entry spread only (1.5 pips)
        assert abs((pnl_no_spread - pnl_with_spread) - 1.5) < 0.01

    def test_commission_deducted(self):
        """Both BUY and SELL PnL should be reduced by commission."""
        pip = 0.0001
        # BUY TP hit
        high, low, close, spread = _make_simple_prices(
            [(-5, 70)],
            base=1.1000,
        )
        pnl_no_comm, _ = _simulate_trade_basic(
            DIR_BUY, 0, 1.1000,
            sl_price=1.1000 - 30 * pip,
            tp_price=1.1000 + 60 * pip,
            high=high, low=low, close=close,
            spread_arr=np.zeros_like(spread),
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        pnl_with_comm, _ = _simulate_trade_basic(
            DIR_BUY, 0, 1.1000,
            sl_price=1.1000 - 30 * pip,
            tp_price=1.1000 + 60 * pip,
            high=high, low=low, close=close,
            spread_arr=np.zeros_like(spread),
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.7,
        )
        assert abs((pnl_no_comm - pnl_with_comm) - 0.7) < 0.01

    def test_commission_deducted_sell(self):
        """SELL trade commission also deducted."""
        pip = 0.0001
        high, low, close, spread = _make_simple_prices(
            [(-70, 5)],
            base=1.1000,
        )
        pnl_no_comm, _ = _simulate_trade_basic(
            DIR_SELL, 0, 1.1000,
            sl_price=1.1000 + 30 * pip,
            tp_price=1.1000 - 60 * pip,
            high=high, low=low, close=close,
            spread_arr=np.zeros_like(spread),
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.0,
        )
        pnl_with_comm, _ = _simulate_trade_basic(
            DIR_SELL, 0, 1.1000,
            sl_price=1.1000 + 30 * pip,
            tp_price=1.1000 - 60 * pip,
            high=high, low=low, close=close,
            spread_arr=np.zeros_like(spread),
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            commission_pips=0.7,
        )
        assert abs((pnl_no_comm - pnl_with_comm) - 0.7) < 0.01

    def test_max_spread_filter(self):
        """Signal with spread > max_spread_pips should be skipped."""
        pip = 0.0001
        high, low, close, spread = _make_simple_prices(
            [(-5, 70)] * 3,
            base=1.1000,
        )
        # Set spread to 5 pips (in price units)
        spread[:] = 5.0 * pip

        sig_bar = np.array([0], dtype=np.int64)
        sig_dir = np.array([DIR_BUY], dtype=np.int64)
        sig_entry = np.array([1.1000], dtype=np.float64)
        sig_hour = np.array([10], dtype=np.int64)
        sig_day = np.array([1], dtype=np.int64)
        sig_atr = np.array([20.0], dtype=np.float64)
        sig_swing = np.array([np.nan], dtype=np.float64)

        params = _basic_params().reshape(1, -1)
        layout = _default_param_layout()
        metrics = np.zeros((1, NUM_METRICS), dtype=np.float64)

        # max_spread_pips=3.0 — signal spread is 5 pips, should be filtered
        batch_evaluate(
            high, low, close, spread, pip, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_BASIC, metrics, 1000, 6048.0,
            0.0, 3.0,
        )
        assert metrics[0, M_TRADES] == 0.0  # Filtered out

    def test_max_spread_filter_passes(self):
        """Signal with spread <= max_spread_pips should NOT be skipped."""
        pip = 0.0001
        high, low, close, spread = _make_simple_prices(
            [(-5, 70)] * 3,
            base=1.1000,
        )
        # Set spread to 1 pip (in price units) — below threshold
        spread[:] = 1.0 * pip

        sig_bar = np.array([0], dtype=np.int64)
        sig_dir = np.array([DIR_BUY], dtype=np.int64)
        sig_entry = np.array([1.1000], dtype=np.float64)
        sig_hour = np.array([10], dtype=np.int64)
        sig_day = np.array([1], dtype=np.int64)
        sig_atr = np.array([20.0], dtype=np.float64)
        sig_swing = np.array([np.nan], dtype=np.float64)

        params = _basic_params().reshape(1, -1)
        layout = _default_param_layout()
        metrics = np.zeros((1, NUM_METRICS), dtype=np.float64)

        # max_spread_pips=3.0 — signal spread is 1 pip, should pass
        batch_evaluate(
            high, low, close, spread, pip, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_BASIC, metrics, 1000, 6048.0,
            0.0, 3.0,
        )
        assert metrics[0, M_TRADES] == 1.0  # Not filtered

    def test_sell_vs_buy_symmetric_costs(self):
        """BUY and SELL should have approximately equal total execution costs."""
        pip = 0.0001
        spread_val = 1.2 * pip  # 1.2 pip spread
        commission = 0.7        # 0.7 pip commission

        # BUY TP hit
        high_b, low_b, close_b, spread_b = _make_simple_prices(
            [(-5, 70)], base=1.1000,
        )
        spread_b[:] = spread_val
        pnl_buy, _ = _simulate_trade_basic(
            DIR_BUY, 0, 1.1000,
            sl_price=1.1000 - 30 * pip,
            tp_price=1.1000 + 60 * pip,
            high=high_b, low=low_b, close=close_b, spread_arr=spread_b,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high_b),
            commission_pips=commission,
        )
        pnl_buy_nocost, _ = _simulate_trade_basic(
            DIR_BUY, 0, 1.1000,
            sl_price=1.1000 - 30 * pip,
            tp_price=1.1000 + 60 * pip,
            high=high_b, low=low_b, close=close_b,
            spread_arr=np.zeros_like(spread_b),
            pip_value=pip, slippage_pips=0.0, num_bars=len(high_b),
            commission_pips=0.0,
        )
        buy_total_cost = pnl_buy_nocost - pnl_buy  # spread_entry + commission

        # SELL TP hit
        high_s, low_s, close_s, spread_s = _make_simple_prices(
            [(-70, 5)], base=1.1000,
        )
        spread_s[:] = spread_val
        pnl_sell, _ = _simulate_trade_basic(
            DIR_SELL, 0, 1.1000,
            sl_price=1.1000 + 30 * pip,
            tp_price=1.1000 - 60 * pip,
            high=high_s, low=low_s, close=close_s, spread_arr=spread_s,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high_s),
            commission_pips=commission,
        )
        pnl_sell_nocost, _ = _simulate_trade_basic(
            DIR_SELL, 0, 1.1000,
            sl_price=1.1000 + 30 * pip,
            tp_price=1.1000 - 60 * pip,
            high=high_s, low=low_s, close=close_s,
            spread_arr=np.zeros_like(spread_s),
            pip_value=pip, slippage_pips=0.0, num_bars=len(high_s),
            commission_pips=0.0,
        )
        sell_total_cost = pnl_sell_nocost - pnl_sell  # spread_exit + commission

        # Both should have total cost = spread (1.2) + commission (0.7) = 1.9 pips
        expected_total = 1.2 + 0.7
        assert abs(buy_total_cost - expected_total) < 0.01
        assert abs(sell_total_cost - expected_total) < 0.01
        # BUY and SELL costs should be approximately equal
        assert abs(buy_total_cost - sell_total_cost) < 0.1

    def test_full_mode_commission(self):
        """Full mode should also apply commission and SELL spread."""
        pip = 0.0001
        base = 1.1000
        # SELL trade: price drops 70 pips (hits TP at -60)
        moves = [(-70, 5)]
        high, low, close, spread = _make_simple_prices(moves, base=base)
        spread[:] = 1.0 * pip  # 1 pip spread

        pnl_no_cost, _ = _simulate_trade_full(
            DIR_SELL, 0, base,
            sl_price=base + 30 * pip,
            tp_price=base - 60 * pip,
            atr_pips=20.0,
            high=high, low=low, close=close, spread_arr=np.zeros_like(spread),
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            trailing_mode=TRAIL_OFF,
            trail_activate_pips=0.0, trail_distance_pips=10.0, trail_atr_mult=2.0,
            breakeven_enabled=0, breakeven_trigger_pips=20.0, breakeven_offset_pips=2.0,
            partial_enabled=0, partial_pct=50.0, partial_trigger_pips=30.0,
            max_bars=0, stale_enabled=0, stale_bars=50, stale_atr_thresh=0.5,
            commission_pips=0.0,
        )
        pnl_with_cost, _ = _simulate_trade_full(
            DIR_SELL, 0, base,
            sl_price=base + 30 * pip,
            tp_price=base - 60 * pip,
            atr_pips=20.0,
            high=high, low=low, close=close, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
            trailing_mode=TRAIL_OFF,
            trail_activate_pips=0.0, trail_distance_pips=10.0, trail_atr_mult=2.0,
            breakeven_enabled=0, breakeven_trigger_pips=20.0, breakeven_offset_pips=2.0,
            partial_enabled=0, partial_pct=50.0, partial_trigger_pips=30.0,
            max_bars=0, stale_enabled=0, stale_bars=50, stale_atr_thresh=0.5,
            commission_pips=0.7,
        )
        # Total cost = spread (1.0) + commission (0.7) = 1.7 pips
        assert abs((pnl_no_cost - pnl_with_cost) - 1.7) < 0.01


# ---------------------------------------------------------------------------
# SELL direction management tests
# ---------------------------------------------------------------------------

class TestSellDirectionManagement:
    def test_sell_trailing_stop(self):
        """Trailing stop should lock in profits for SELL trades."""
        pip = 0.0001
        base = 1.1000
        # Price drops 50 pips, then rallies back
        moves = [
            (5, -20),    # Bar 1: -20 low (favorable for SELL)
            (5, -40),    # Bar 2: -40 low (trail activates at 30)
            (5, -50),    # Bar 3: -50 low (trail moves down)
            (30, -10),   # Bar 4: rallies, trail should trigger
        ]
        high, low, close, spread = _make_simple_prices(moves, base=base)
        # For SELL: low is favorable, high is adverse. Fix signs:
        # Actually _make_simple_prices uses offsets from base.
        # For SELL: price going DOWN is good. Let me reconstruct properly.
        n = 5
        high = np.full(n, base, dtype=np.float64)
        low = np.full(n, base, dtype=np.float64)
        close = np.full(n, base, dtype=np.float64)
        spread = np.zeros(n, dtype=np.float64)
        # Bar 0: entry bar
        # Bar 1: price drops 20 pips
        high[1] = base + 5 * pip;  low[1] = base - 20 * pip;  close[1] = base - 10 * pip
        # Bar 2: price drops 40 pips
        high[2] = base + 5 * pip;  low[2] = base - 40 * pip;  close[2] = base - 30 * pip
        # Bar 3: price drops 50 pips
        high[3] = base + 5 * pip;  low[3] = base - 50 * pip;  close[3] = base - 40 * pip
        # Bar 4: price rallies (adverse for sell)
        high[4] = base - 10 * pip; low[4] = base - 30 * pip;  close[4] = base - 15 * pip

        pnl, exit_code = _simulate_trade_full(
            DIR_SELL, 0, base,
            sl_price=base + 50 * pip,  # Wide SL
            tp_price=base - 100 * pip, # Wide TP
            atr_pips=20.0,
            high=high, low=low, close=close, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=n,
            trailing_mode=TRAIL_FIXED_PIP,
            trail_activate_pips=30.0,  # Activate at +30 pips profit
            trail_distance_pips=15.0,  # Trail 15 pips behind
            trail_atr_mult=2.0,
            breakeven_enabled=0, breakeven_trigger_pips=20.0, breakeven_offset_pips=2.0,
            partial_enabled=0, partial_pct=50.0, partial_trigger_pips=30.0,
            max_bars=0, stale_enabled=0, stale_bars=50, stale_atr_thresh=0.5,
            commission_pips=0.0,
        )
        assert exit_code == EXIT_TRAILING
        assert pnl > 20  # Should lock in significant profit

    def test_sell_breakeven_lock(self):
        """Breakeven should work for SELL trades."""
        pip = 0.0001
        base = 1.1000
        n = 4
        high = np.full(n, base, dtype=np.float64)
        low = np.full(n, base, dtype=np.float64)
        close = np.full(n, base, dtype=np.float64)
        spread = np.zeros(n, dtype=np.float64)
        # Bar 0: entry
        # Bar 1: price drops 25 pips (triggers BE at 20)
        high[1] = base + 5 * pip;  low[1] = base - 25 * pip;  close[1] = base - 15 * pip
        # Bar 2: price rallies back above entry (should hit BE stop)
        high[2] = base + 5 * pip;  low[2] = base - 5 * pip;   close[2] = base + 3 * pip

        pnl, exit_code = _simulate_trade_full(
            DIR_SELL, 0, base,
            sl_price=base + 50 * pip,
            tp_price=base - 100 * pip,
            atr_pips=20.0,
            high=high, low=low, close=close, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=n,
            trailing_mode=TRAIL_OFF,
            trail_activate_pips=0.0, trail_distance_pips=10.0, trail_atr_mult=2.0,
            breakeven_enabled=1, breakeven_trigger_pips=20.0, breakeven_offset_pips=2.0,
            partial_enabled=0, partial_pct=50.0, partial_trigger_pips=30.0,
            max_bars=0, stale_enabled=0, stale_bars=50, stale_atr_thresh=0.5,
            commission_pips=0.0,
        )
        assert exit_code == EXIT_BREAKEVEN
        # BE offset = 2 pips below entry for SELL, so small profit
        assert pnl > 0


# ---------------------------------------------------------------------------
# Partial close and stale exit tests
# ---------------------------------------------------------------------------

class TestPartialCloseAndStaleExit:
    def test_partial_close_reduces_position(self):
        """Partial close should realize some PnL and reduce remaining position."""
        pip = 0.0001
        base = 1.1000
        n = 4
        high = np.full(n, base, dtype=np.float64)
        low = np.full(n, base, dtype=np.float64)
        close = np.full(n, base, dtype=np.float64)
        spread = np.zeros(n, dtype=np.float64)
        # Bar 0: entry
        # Bar 1: price goes up 35 pips (triggers partial at 30, closes 50% at bar close)
        high[1] = base + 35 * pip;  low[1] = base - 2 * pip;   close[1] = base + 30 * pip
        # Bar 2: price drops to SL
        high[2] = base + 5 * pip;   low[2] = base - 50 * pip;  close[2] = base - 40 * pip

        pnl_with_partial, _ = _simulate_trade_full(
            DIR_BUY, 0, base,
            sl_price=base - 40 * pip,
            tp_price=base + 100 * pip,
            atr_pips=20.0,
            high=high, low=low, close=close, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=n,
            trailing_mode=TRAIL_OFF,
            trail_activate_pips=0.0, trail_distance_pips=10.0, trail_atr_mult=2.0,
            breakeven_enabled=0, breakeven_trigger_pips=20.0, breakeven_offset_pips=2.0,
            partial_enabled=1, partial_pct=50.0, partial_trigger_pips=30.0,
            max_bars=0, stale_enabled=0, stale_bars=50, stale_atr_thresh=0.5,
            commission_pips=0.0,
        )
        pnl_without_partial, _ = _simulate_trade_full(
            DIR_BUY, 0, base,
            sl_price=base - 40 * pip,
            tp_price=base + 100 * pip,
            atr_pips=20.0,
            high=high, low=low, close=close, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=n,
            trailing_mode=TRAIL_OFF,
            trail_activate_pips=0.0, trail_distance_pips=10.0, trail_atr_mult=2.0,
            breakeven_enabled=0, breakeven_trigger_pips=20.0, breakeven_offset_pips=2.0,
            partial_enabled=0, partial_pct=50.0, partial_trigger_pips=30.0,
            max_bars=0, stale_enabled=0, stale_bars=50, stale_atr_thresh=0.5,
            commission_pips=0.0,
        )
        # Without partial: full SL hit = -40 pips
        # With partial: 50% closed at +30 pips (+15) + 50% hits SL at -40 pips (-20) = -5
        # So partial close should significantly improve the losing trade
        assert pnl_with_partial > pnl_without_partial

    def test_stale_exit_triggers(self):
        """Stale exit should close trade when price barely moves."""
        pip = 0.0001
        base = 1.1000
        # Create 60 bars of tiny range (< atr_thresh * atr_pips)
        n = 65
        high = np.full(n, base + 0.1 * pip, dtype=np.float64)
        low = np.full(n, base - 0.1 * pip, dtype=np.float64)
        close = np.full(n, base, dtype=np.float64)
        spread = np.zeros(n, dtype=np.float64)

        pnl, exit_code = _simulate_trade_full(
            DIR_BUY, 0, base,
            sl_price=base - 50 * pip,
            tp_price=base + 100 * pip,
            atr_pips=20.0,
            high=high, low=low, close=close, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=n,
            trailing_mode=TRAIL_OFF,
            trail_activate_pips=0.0, trail_distance_pips=10.0, trail_atr_mult=2.0,
            breakeven_enabled=0, breakeven_trigger_pips=20.0, breakeven_offset_pips=2.0,
            partial_enabled=0, partial_pct=50.0, partial_trigger_pips=30.0,
            max_bars=0, stale_enabled=1, stale_bars=50, stale_atr_thresh=0.5,
            commission_pips=0.0,
        )
        assert exit_code == EXIT_STALE
        # PnL should be near zero (price barely moved)
        assert abs(pnl) < 5


# ---------------------------------------------------------------------------
# Hour wrap-around filter test
# ---------------------------------------------------------------------------

class TestHourWrapAround:
    def test_hour_wrap_around_filter(self):
        """Hours filter 22-06 should accept signals at hour 2 and reject hour 10."""
        pip = 0.0001
        base = 1.1000
        high, low, close, spread = _make_simple_prices(
            [(-5, 70)] * 3, base=base,
        )

        # Signal at hour 2 (should pass 22-06 filter)
        sig_bar = np.array([0], dtype=np.int64)
        sig_dir = np.array([DIR_BUY], dtype=np.int64)
        sig_entry = np.array([base], dtype=np.float64)
        sig_hour = np.array([2], dtype=np.int64)  # 2 AM
        sig_day = np.array([1], dtype=np.int64)
        sig_atr = np.array([20.0], dtype=np.float64)
        sig_swing = np.array([np.nan], dtype=np.float64)

        params = _basic_params()
        params[PL_HOURS_START] = 22  # Wrap-around filter: 22:00 to 06:00
        params[PL_HOURS_END] = 6
        params = params.reshape(1, -1)
        layout = _default_param_layout()
        metrics = np.zeros((1, NUM_METRICS), dtype=np.float64)

        batch_evaluate(
            high, low, close, spread, pip, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_BASIC, metrics, 1000, 6048.0,
            0.0, 0.0,
        )
        assert metrics[0, M_TRADES] == 1.0  # Hour 2 is within 22-06

        # Signal at hour 10 (should be filtered out by 22-06)
        sig_hour_10 = np.array([10], dtype=np.int64)
        metrics2 = np.zeros((1, NUM_METRICS), dtype=np.float64)

        batch_evaluate(
            high, low, close, spread, pip, 0.0,
            sig_bar, sig_dir, sig_entry, sig_hour_10, sig_day, sig_atr, sig_swing,
            np.zeros_like(sig_atr), np.full_like(sig_bar, -1),
            params, layout, EXEC_BASIC, metrics2, 1000, 6048.0,
            0.0, 0.0,
        )
        assert metrics2[0, M_TRADES] == 0.0  # Hour 10 is outside 22-06
