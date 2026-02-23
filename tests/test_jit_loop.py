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
    PL_DAYS_BITMASK,
    PL_HOURS_END,
    PL_HOURS_START,
    PL_MAX_BARS,
    PL_PARTIAL_ENABLED,
    PL_PARTIAL_PCT,
    PL_PARTIAL_TRIGGER,
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
    spread = np.full(n_bars, 1.0)  # 1 pip spread
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
            high=high, low=low, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
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
            high=high, low=low, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
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
            high=high, low=low, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
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
            high=high, low=low, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
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
            high=high, low=low, spread_arr=spread,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
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
            high=high, low=low, spread_arr=np.zeros_like(spread),
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
        )
        # With 2 pip spread
        spread_2 = np.full_like(spread, 2.0)
        pnl_with_spread, _ = _simulate_trade_basic(
            DIR_BUY, 0, 1.1000,
            sl_price=1.0970, tp_price=1.1060,
            high=high, low=low, spread_arr=spread_2,
            pip_value=pip, slippage_pips=0.0, num_bars=len(high),
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
            params, layout, EXEC_BASIC, metrics, 1000,
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
            params, layout, EXEC_BASIC, metrics, 1000,
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
            params, layout, EXEC_BASIC, metrics, 1000,
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
            params, layout, EXEC_BASIC, metrics, 1000,
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
            params, layout, EXEC_BASIC, metrics, 1000,
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
            params, layout, EXEC_BASIC, metrics, 1000,
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
            params, layout, EXEC_FULL, metrics, 1000,
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
            params, layout, EXEC_FULL, metrics, 1000,
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
            params, layout, EXEC_FULL, metrics, 1000,
        )

        assert metrics[0, M_TRADES] == 1.0
