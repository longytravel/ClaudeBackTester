"""JIT-compiled batch evaluator for backtesting.

Core hot loop: @njit(parallel=True) evaluates N parameter sets in parallel.
Zero allocation inside prange loops — all output arrays pre-allocated.

Architecture:
  - batch_evaluate() is the entry point
  - _filter_signals_for_trial(): time filtering (hours, day bitmask)
  - _compute_sl_tp(): SL/TP from params (3 SL modes × 3 TP modes)
  - _simulate_trade_basic(): bar-by-bar SL/TP check only
  - _simulate_trade_full(): full management (trailing, BE, partial, max bars, stale)
  - _compute_metrics_inline(): JIT metrics matching Python metrics.py
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from backtester.core import numba_setup  # noqa: F401 — TBB bootstrap

# Import constants directly as module-level values for Numba
from backtester.core.dtypes import (
    DIR_BUY,
    DIR_SELL,
    EXEC_BASIC,
    EXEC_FULL,
    EXIT_BREAKEVEN,
    EXIT_MAX_BARS,
    EXIT_NONE,
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
    SL_ATR_BASED,
    SL_FIXED_PIPS,
    SL_SWING,
    TP_ATR_BASED,
    TP_FIXED_PIPS,
    TP_RR_RATIO,
    TRAIL_ATR_CHANDELIER,
    TRAIL_FIXED_PIP,
    TRAIL_OFF,
)


# ---------------------------------------------------------------------------
# Parameter column indices — set by the engine before calling batch_evaluate.
# These are dynamically assigned based on the strategy's ParamSpace.
# The engine passes them as a flat int64 array (param_layout).
# ---------------------------------------------------------------------------
# param_layout indices (fixed order, maps to encoding.py columns):
PL_SL_MODE = 0
PL_SL_FIXED_PIPS = 1
PL_SL_ATR_MULT = 2
PL_TP_MODE = 3
PL_TP_RR_RATIO = 4
PL_TP_ATR_MULT = 5
PL_TP_FIXED_PIPS = 6
PL_HOURS_START = 7
PL_HOURS_END = 8
PL_DAYS_BITMASK = 9
PL_TRAILING_MODE = 10
PL_TRAIL_ACTIVATE = 11
PL_TRAIL_DISTANCE = 12
PL_TRAIL_ATR_MULT = 13
PL_BREAKEVEN_ENABLED = 14
PL_BREAKEVEN_TRIGGER = 15
PL_BREAKEVEN_OFFSET = 16
PL_PARTIAL_ENABLED = 17
PL_PARTIAL_PCT = 18
PL_PARTIAL_TRIGGER = 19
PL_MAX_BARS = 20
PL_STALE_ENABLED = 21
PL_STALE_BARS = 22
PL_STALE_ATR_THRESH = 23
PL_SIGNAL_VARIANT = 24     # Which signal variant to accept (-1 = all)
PL_BUY_FILTER_MAX = 25     # Max filter_value for BUY signals (-1 = no filter)
PL_SELL_FILTER_MIN = 26    # Min filter_value for SELL signals (-1 = no filter)
NUM_PL = 27


@njit(cache=True)
def _filter_signals_for_trial(
    sig_hours: np.ndarray,      # (S,) int64
    sig_days: np.ndarray,       # (S,) int64
    hours_start: int,
    hours_end: int,
    days_bitmask: int,
    mask_out: np.ndarray,       # (S,) bool — written in place
) -> int:
    """Apply time filters. Returns count of passing signals."""
    count = 0
    n = len(sig_hours)
    for i in range(n):
        # Day filter: check if day's bit is set in bitmask
        day_bit = 1 << sig_days[i]
        if (days_bitmask & day_bit) == 0:
            mask_out[i] = False
            continue

        # Hour filter: handle wrap-around (e.g., 22-06)
        h = sig_hours[i]
        if hours_start <= hours_end:
            ok = hours_start <= h <= hours_end
        else:
            ok = h >= hours_start or h <= hours_end

        mask_out[i] = ok
        if ok:
            count += 1
    return count


@njit(cache=True)
def _compute_sl_tp(
    direction: int,
    entry_price: float,
    atr_pips: float,
    pip_value: float,
    sl_mode: int,
    sl_fixed_pips: float,
    sl_atr_mult: float,
    swing_sl_price: float,  # Pre-computed swing SL (NaN if not available)
    tp_mode: int,
    tp_rr_ratio: float,
    tp_atr_mult: float,
    tp_fixed_pips: float,
) -> tuple[float, float, float, float]:
    """Compute SL and TP prices. Returns (sl_price, tp_price, sl_pips, tp_pips)."""
    atr_price = atr_pips * pip_value
    is_buy = direction == DIR_BUY

    # --- Stop Loss ---
    if sl_mode == SL_ATR_BASED:
        sl_distance = atr_price * sl_atr_mult
    elif sl_mode == SL_SWING:
        if not np.isnan(swing_sl_price):
            sl_distance = abs(entry_price - swing_sl_price)
            min_sl = 5.0 * pip_value
            if sl_distance < min_sl:
                sl_distance = min_sl
        else:
            sl_distance = atr_price * 1.5  # Fallback
    else:  # SL_FIXED_PIPS
        sl_distance = sl_fixed_pips * pip_value

    sl_pips = sl_distance / pip_value

    if is_buy:
        sl_price = entry_price - sl_distance
    else:
        sl_price = entry_price + sl_distance

    # --- Take Profit ---
    if tp_mode == TP_ATR_BASED:
        tp_distance = atr_price * tp_atr_mult
    elif tp_mode == TP_FIXED_PIPS:
        tp_distance = tp_fixed_pips * pip_value
    else:  # TP_RR_RATIO
        tp_distance = sl_distance * tp_rr_ratio

    # Enforce TP >= SL
    if tp_distance < sl_distance:
        tp_distance = sl_distance

    tp_pips = tp_distance / pip_value

    if is_buy:
        tp_price = entry_price + tp_distance
    else:
        tp_price = entry_price - tp_distance

    return sl_price, tp_price, sl_pips, tp_pips


@njit(cache=True)
def _simulate_trade_basic(
    direction: int,
    entry_bar: int,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    spread_arr: np.ndarray,
    pip_value: float,
    slippage_pips: float,
    num_bars: int,
    commission_pips: float,
    # Sub-bar arrays (M1 or identity)
    sub_high: np.ndarray,
    sub_low: np.ndarray,
    sub_close: np.ndarray,
    sub_spread: np.ndarray,
    h1_to_sub_start: np.ndarray,
    h1_to_sub_end: np.ndarray,
) -> tuple[float, int]:
    """Simulate a single trade with SL/TP only (basic mode).

    Returns (pnl_pips, exit_reason).
    Conservative tiebreak: if both SL and TP hit on the same sub-bar, SL wins.
    Uses sub-bar (M1) data for SL/TP checks when available.
    """
    is_buy = direction == DIR_BUY
    slippage_price = slippage_pips * pip_value

    # Apply entry slippage + spread (using sub-bar spread at entry)
    # spread_arr values are in price units (ask - bid), NOT in pips
    if is_buy:
        actual_entry = entry_price + slippage_price
        # BUY: pay the spread on entry — use sub-bar spread at first M1 of entry bar
        sub_entry_start = h1_to_sub_start[entry_bar]
        if sub_entry_start < len(sub_spread):
            spread_at_entry = sub_spread[sub_entry_start]
        else:
            spread_at_entry = 0.0
        if np.isnan(spread_at_entry):
            spread_at_entry = 0.0
        actual_entry += spread_at_entry
    else:
        actual_entry = entry_price - slippage_price
        # SELL: no spread adjustment on entry (we sell at bid)

    exit_reason = EXIT_NONE
    exit_bar = num_bars - 1
    pnl = 0.0
    exit_sub_idx = -1

    # Walk forward bar by bar, checking sub-bars within each H1 bar
    for bar in range(entry_bar + 1, num_bars):
        sub_start = h1_to_sub_start[bar]
        sub_end = h1_to_sub_end[bar]

        for sb in range(sub_start, sub_end):
            sb_high = sub_high[sb]
            sb_low = sub_low[sb]

            if is_buy:
                # Check SL first (conservative tiebreak)
                if sb_low <= sl_price:
                    pnl = (sl_price - slippage_price - actual_entry) / pip_value
                    exit_reason = EXIT_SL
                    exit_sub_idx = sb
                    break
                if sb_high >= tp_price:
                    pnl = (tp_price - actual_entry) / pip_value
                    exit_reason = EXIT_TP
                    exit_sub_idx = sb
                    break
            else:
                # SELL: SL is above, TP is below
                if sb_high >= sl_price:
                    pnl = (actual_entry - sl_price - slippage_price) / pip_value
                    exit_reason = EXIT_SL
                    exit_sub_idx = sb
                    break
                if sb_low <= tp_price:
                    pnl = (actual_entry - tp_price) / pip_value
                    exit_reason = EXIT_TP
                    exit_sub_idx = sb
                    break

        if exit_reason != EXIT_NONE:
            exit_bar = bar
            break

    # Trade still open at end of data — close at last bar close (market order)
    if exit_reason == EXIT_NONE:
        close_price = close[exit_bar]
        if is_buy:
            pnl = (close_price - slippage_price - actual_entry) / pip_value
        else:
            pnl = (actual_entry - close_price - slippage_price) / pip_value

    # Apply execution costs
    # SELL: deduct exit spread (use sub-bar spread if we have an exit sub-bar)
    if not is_buy:
        if exit_sub_idx >= 0 and exit_sub_idx < len(sub_spread):
            sell_spread = sub_spread[exit_sub_idx]
        elif exit_bar < len(spread_arr):
            sell_spread = spread_arr[exit_bar]
        else:
            sell_spread = 0.0
        if np.isnan(sell_spread):
            sell_spread = 0.0
        pnl -= sell_spread / pip_value
    # Commission applied to all trades
    pnl -= commission_pips

    return pnl, exit_reason


@njit(cache=True)
def _simulate_trade_full(
    direction: int,
    entry_bar: int,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    atr_pips: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    spread_arr: np.ndarray,
    pip_value: float,
    slippage_pips: float,
    num_bars: int,
    # Management params
    trailing_mode: int,
    trail_activate_pips: float,
    trail_distance_pips: float,
    trail_atr_mult: float,
    breakeven_enabled: int,
    breakeven_trigger_pips: float,
    breakeven_offset_pips: float,
    partial_enabled: int,
    partial_pct: float,
    partial_trigger_pips: float,
    max_bars: int,
    stale_enabled: int,
    stale_bars: int,
    stale_atr_thresh: float,
    commission_pips: float,
    # Sub-bar arrays (M1 or identity)
    sub_high: np.ndarray,
    sub_low: np.ndarray,
    sub_close: np.ndarray,
    sub_spread: np.ndarray,
    h1_to_sub_start: np.ndarray,
    h1_to_sub_end: np.ndarray,
) -> tuple[float, int]:
    """Simulate a single trade with full management features.

    Returns (pnl_pips, exit_reason).
    Uses sub-bar (M1) data for all price-sensitive management checks
    (SL, TP, trailing, breakeven, partial close).
    H1-level checks (max_bars, stale exit) remain at H1 bar resolution.
    """
    is_buy = direction == DIR_BUY
    slippage_price = slippage_pips * pip_value

    # Apply entry costs (using sub-bar spread at first M1 of entry bar)
    # spread_arr values are in price units (ask - bid), NOT in pips
    if is_buy:
        actual_entry = entry_price + slippage_price
        sub_entry_start = h1_to_sub_start[entry_bar]
        if sub_entry_start < len(sub_spread):
            spread_at_entry = sub_spread[sub_entry_start]
        else:
            spread_at_entry = 0.0
        if np.isnan(spread_at_entry):
            spread_at_entry = 0.0
        actual_entry += spread_at_entry
    else:
        actual_entry = entry_price - slippage_price

    current_sl = sl_price
    position_pct = 1.0  # 1.0 = full position, reduced by partial close
    partial_done = False
    be_locked = False
    trailing_active = False
    realized_pnl_pips = 0.0

    # Deferred SL pattern: modifications apply from NEXT sub-bar
    # This prevents same-bar trigger+exit (e.g., BE triggers on high then
    # hits on low within the same M1 bar — we don't know OHLC ordering).
    pending_sl = -1.0           # -1.0 = no pending update
    pending_be_locked = False
    pending_trailing_active = False
    has_pending_update = False

    bars_held = 0
    exit_reason = EXIT_NONE
    exit_bar = num_bars - 1
    exit_sub_idx = -1
    final_pnl = 0.0

    for bar in range(entry_bar + 1, num_bars):
        bar_high = high[bar]
        bar_low = low[bar]
        bar_close = close[bar]
        bars_held += 1

        # --- H1-level checks (max_bars, stale) — market close orders get slippage ---
        if max_bars > 0 and bars_held >= max_bars:
            if is_buy:
                pnl = (bar_close - slippage_price - actual_entry) / pip_value * position_pct
            else:
                pnl = (actual_entry - bar_close - slippage_price) / pip_value * position_pct
            final_pnl = realized_pnl_pips + pnl
            exit_reason = EXIT_MAX_BARS
            exit_bar = bar
            break

        if stale_enabled > 0 and bars_held >= stale_bars:
            lookback_start = max(entry_bar + 1, bar - stale_bars + 1)
            max_range = 0.0
            for b in range(lookback_start, bar + 1):
                r = (high[b] - low[b]) / pip_value
                if r > max_range:
                    max_range = r
            if max_range < stale_atr_thresh * atr_pips:
                if is_buy:
                    pnl = (bar_close - slippage_price - actual_entry) / pip_value * position_pct
                else:
                    pnl = (actual_entry - bar_close - slippage_price) / pip_value * position_pct
                final_pnl = realized_pnl_pips + pnl
                exit_reason = EXIT_STALE
                exit_bar = bar
                break

        # --- Sub-bar trade management ---
        sub_start = h1_to_sub_start[bar]
        sub_end = h1_to_sub_end[bar]

        for sb in range(sub_start, sub_end):
            sb_high = sub_high[sb]
            sb_low = sub_low[sb]
            sb_close = sub_close[sb]

            # Apply any pending SL modification from the PREVIOUS sub-bar
            if has_pending_update:
                if pending_sl > 0.0:
                    current_sl = pending_sl
                be_locked = pending_be_locked
                trailing_active = pending_trailing_active
                pending_sl = -1.0
                has_pending_update = False

            # Current floating PnL on this sub-bar
            if is_buy:
                float_pnl_pips = (sb_high - actual_entry) / pip_value
                worst_pnl_pips = (sb_low - actual_entry) / pip_value
            else:
                float_pnl_pips = (actual_entry - sb_low) / pip_value
                worst_pnl_pips = (actual_entry - sb_high) / pip_value

            # --- Breakeven lock (deferred: sets pending, applied next sub-bar) ---
            if breakeven_enabled > 0 and not be_locked and not pending_be_locked:
                if float_pnl_pips >= breakeven_trigger_pips:
                    be_price = actual_entry + breakeven_offset_pips * pip_value if is_buy \
                        else actual_entry - breakeven_offset_pips * pip_value
                    if is_buy and be_price > current_sl:
                        pending_sl = be_price
                        pending_be_locked = True
                        pending_trailing_active = trailing_active
                        has_pending_update = True
                    elif not is_buy and be_price < current_sl:
                        pending_sl = be_price
                        pending_be_locked = True
                        pending_trailing_active = trailing_active
                        has_pending_update = True

            # --- Trailing stop (deferred: sets pending, applied next sub-bar) ---
            if trailing_mode != TRAIL_OFF:
                if not trailing_active and not pending_trailing_active:
                    if float_pnl_pips >= trail_activate_pips:
                        # Activation + initial trailing SL, both deferred
                        pending_trailing_active = True
                        if trailing_mode == TRAIL_FIXED_PIP:
                            trail_dist = trail_distance_pips * pip_value
                        else:  # TRAIL_ATR_CHANDELIER
                            trail_dist = trail_atr_mult * atr_pips * pip_value
                        if is_buy:
                            new_sl = sb_high - trail_dist
                            effective_sl = pending_sl if has_pending_update and pending_sl > 0 else current_sl
                            if new_sl > effective_sl:
                                pending_sl = new_sl
                        else:
                            new_sl = sb_low + trail_dist
                            effective_sl = pending_sl if has_pending_update and pending_sl > 0 else current_sl
                            if new_sl < effective_sl:
                                pending_sl = new_sl
                        pending_be_locked = be_locked if not has_pending_update else pending_be_locked
                        has_pending_update = True

                if trailing_active:
                    if trailing_mode == TRAIL_FIXED_PIP:
                        trail_dist = trail_distance_pips * pip_value
                    else:  # TRAIL_ATR_CHANDELIER
                        trail_dist = trail_atr_mult * atr_pips * pip_value

                    if is_buy:
                        new_sl = sb_high - trail_dist
                        effective_sl = pending_sl if has_pending_update and pending_sl > 0 else current_sl
                        if new_sl > effective_sl:
                            pending_sl = new_sl
                            pending_be_locked = be_locked if not has_pending_update else pending_be_locked
                            pending_trailing_active = True
                            has_pending_update = True
                    else:
                        new_sl = sb_low + trail_dist
                        effective_sl = pending_sl if has_pending_update and pending_sl > 0 else current_sl
                        if new_sl < effective_sl:
                            pending_sl = new_sl
                            pending_be_locked = be_locked if not has_pending_update else pending_be_locked
                            pending_trailing_active = True
                            has_pending_update = True

            # --- Partial close (immediate — uses close price, no ordering ambiguity) ---
            if partial_enabled > 0 and not partial_done:
                if float_pnl_pips >= partial_trigger_pips:
                    partial_done = True
                    close_pct = partial_pct / 100.0
                    if is_buy:
                        partial_pnl = (sb_close - actual_entry) / pip_value * close_pct
                    else:
                        partial_pnl = (actual_entry - sb_close) / pip_value * close_pct
                    realized_pnl_pips += partial_pnl
                    position_pct -= close_pct

            # --- Check SL (uses current_sl — reflects PREVIOUS bar's state) ---
            if is_buy:
                if sb_low <= current_sl:
                    pnl = (current_sl - slippage_price - actual_entry) / pip_value * position_pct
                    exit_code = EXIT_SL
                    if trailing_active:
                        exit_code = EXIT_TRAILING
                    elif be_locked:
                        exit_code = EXIT_BREAKEVEN
                    final_pnl = realized_pnl_pips + pnl
                    exit_reason = exit_code
                    exit_sub_idx = sb
                    break
                if sb_high >= tp_price:
                    pnl = (tp_price - actual_entry) / pip_value * position_pct
                    final_pnl = realized_pnl_pips + pnl
                    exit_reason = EXIT_TP
                    exit_sub_idx = sb
                    break
            else:
                if sb_high >= current_sl:
                    pnl = (actual_entry - current_sl - slippage_price) / pip_value * position_pct
                    exit_code = EXIT_SL
                    if trailing_active:
                        exit_code = EXIT_TRAILING
                    elif be_locked:
                        exit_code = EXIT_BREAKEVEN
                    final_pnl = realized_pnl_pips + pnl
                    exit_reason = exit_code
                    exit_sub_idx = sb
                    break
                if sb_low <= tp_price:
                    pnl = (actual_entry - tp_price) / pip_value * position_pct
                    final_pnl = realized_pnl_pips + pnl
                    exit_reason = EXIT_TP
                    exit_sub_idx = sb
                    break

        if exit_reason != EXIT_NONE:
            exit_bar = bar
            break

    # End of data — close remaining position (market order, gets slippage)
    if exit_reason == EXIT_NONE:
        close_price = close[exit_bar] if exit_bar < len(close) else (high[exit_bar] + low[exit_bar]) / 2.0
        if is_buy:
            pnl = (close_price - slippage_price - actual_entry) / pip_value * position_pct
        else:
            pnl = (actual_entry - close_price - slippage_price) / pip_value * position_pct
        final_pnl = realized_pnl_pips + pnl

    # Apply execution costs
    # SELL: deduct exit spread (use sub-bar spread if available)
    if not is_buy:
        if exit_sub_idx >= 0 and exit_sub_idx < len(sub_spread):
            sell_spread = sub_spread[exit_sub_idx]
        elif exit_bar < len(spread_arr):
            sell_spread = spread_arr[exit_bar]
        else:
            sell_spread = 0.0
        if np.isnan(sell_spread):
            sell_spread = 0.0
        final_pnl -= sell_spread / pip_value
    # Commission applied to all trades
    final_pnl -= commission_pips

    return final_pnl, exit_reason


@njit(cache=True)
def _compute_metrics_inline(
    pnl_arr: np.ndarray,
    trade_count: int,
    avg_sl_pips: float,
    n_bars: int,
    bars_per_year: float,
    metrics_row: np.ndarray,  # (NUM_METRICS,) — written in place
) -> None:
    """Compute all 10 metrics inline for one trial. Writes to metrics_row."""
    n = trade_count
    metrics_row[M_TRADES] = float(n)

    if n == 0:
        return

    # Win rate
    wins = 0
    for i in range(n):
        if pnl_arr[i] > 0:
            wins += 1
    metrics_row[M_WIN_RATE] = float(wins) / float(n)

    # Profit factor
    gross_profit = 0.0
    gross_loss = 0.0
    total_pnl = 0.0
    for i in range(n):
        p = pnl_arr[i]
        total_pnl += p
        if p > 0:
            gross_profit += p
        elif p < 0:
            gross_loss -= p  # Make positive
    if gross_loss == 0.0:
        pf = 10.0 if gross_profit > 0 else 0.0
    else:
        pf = gross_profit / gross_loss
    metrics_row[M_PROFIT_FACTOR] = pf

    # Mean and standard deviations
    mean = total_pnl / n
    var_sum = 0.0
    down_sq_sum = 0.0
    down_count = 0
    for i in range(n):
        diff = pnl_arr[i] - mean
        var_sum += diff * diff
        if pnl_arr[i] < 0:
            down_sq_sum += pnl_arr[i] * pnl_arr[i]
            down_count += 1

    if n > 1:
        std = np.sqrt(var_sum / (n - 1))
    else:
        std = 0.0

    # Annualization factor: trades per year
    # = n_trades * (bars_per_year / n_bars)
    if n_bars > 0 and bars_per_year > 0:
        ann_factor = float(n) * bars_per_year / float(n_bars)
    else:
        ann_factor = min(float(n), 252.0)

    # Sharpe (annualized)
    if std > 0:
        metrics_row[M_SHARPE] = (mean / std) * np.sqrt(ann_factor)
    else:
        metrics_row[M_SHARPE] = 0.0

    # Sortino (annualized) — root causes of explosion fixed (deferred SL,
    # adverse exit slippage, raised BE floors). No artificial cap needed.
    if down_count > 0:
        downside_std = np.sqrt(down_sq_sum / down_count)
        if downside_std > 0:
            raw_sortino = (mean / downside_std) * np.sqrt(ann_factor)
            metrics_row[M_SORTINO] = raw_sortino
        else:
            metrics_row[M_SORTINO] = 0.0
    else:
        metrics_row[M_SORTINO] = 10.0 if mean > 0 else 0.0

    # Equity curve for MaxDD, R², Ulcer
    # Use pre-allocated workspace in pnl_arr region after trade_count
    # Actually, compute cumulatively:
    equity_peak = 0.0
    max_dd = 0.0
    equity = 0.0
    base_val = 0.0
    sum_sq_dd = 0.0

    # For R²: need cumulative equity, linear regression
    sum_x = 0.0
    sum_y = 0.0
    sum_xy = 0.0
    sum_xx = 0.0

    for i in range(n):
        equity += pnl_arr[i]
        if equity > equity_peak:
            equity_peak = equity
        dd = equity_peak - equity
        if dd > max_dd:
            max_dd = dd

        # Track peak for Ulcer base
        if abs(equity) > base_val:
            base_val = abs(equity)
        if equity_peak > base_val:
            base_val = equity_peak

        # R² accumulators
        x = float(i)
        sum_x += x
        sum_y += equity
        sum_xy += x * equity
        sum_xx += x * x

        # Ulcer: percentage drawdown squared
        if base_val > 0:
            pct_dd = (dd / base_val) * 100.0
        else:
            pct_dd = 0.0
        sum_sq_dd += pct_dd * pct_dd

    # Max DD %
    if base_val <= 0:
        base_val = 1.0
    metrics_row[M_MAX_DD_PCT] = (max_dd / base_val) * 100.0

    # Return %
    if avg_sl_pips > 0:
        metrics_row[M_RETURN_PCT] = (total_pnl / avg_sl_pips) * 100.0
    else:
        metrics_row[M_RETURN_PCT] = 0.0

    # R²
    if n >= 2:
        x_mean = sum_x / n
        y_mean = sum_y / n
        ss_xy = sum_xy - n * x_mean * y_mean
        ss_xx = sum_xx - n * x_mean * x_mean
        if ss_xx > 0:
            slope = ss_xy / ss_xx
            intercept = y_mean - slope * x_mean
            ss_res = 0.0
            ss_tot = 0.0
            eq2 = 0.0
            for i in range(n):
                eq2 += pnl_arr[i]
                y_pred = slope * float(i) + intercept
                ss_res += (eq2 - y_pred) ** 2
                ss_tot += (eq2 - y_mean) ** 2
            if ss_tot > 0:
                rsq = 1.0 - ss_res / ss_tot
                metrics_row[M_R_SQUARED] = max(0.0, rsq)

    # Ulcer Index
    metrics_row[M_ULCER] = np.sqrt(sum_sq_dd / n)

    # Quality Score — losing strategies (Sortino <= 0) score zero
    so = metrics_row[M_SORTINO]
    if so > 0:
        r2 = metrics_row[M_R_SQUARED]
        pf_c = min(pf, 5.0)
        trades_f = np.sqrt(min(float(n), 200.0))
        ret = metrics_row[M_RETURN_PCT]
        # Return% bonus: only positive returns, clamped to [0, 200]
        ret_clamped = min(ret, 200.0)
        if ret_clamped < 0.0:
            ret_clamped = 0.0
        ret_f = 1.0 + ret_clamped / 100.0
        ulc = metrics_row[M_ULCER]
        dd_pct = metrics_row[M_MAX_DD_PCT]

        denom = ulc + dd_pct / 2.0 + 5.0
        if denom > 0:
            metrics_row[M_QUALITY] = (so * r2 * pf_c * trades_f * ret_f) / denom


@njit(parallel=True, cache=True)
def batch_evaluate(
    # Price data (shared across all trials)
    high: np.ndarray,           # (B,) float64
    low: np.ndarray,            # (B,) float64
    close: np.ndarray,          # (B,) float64
    spread: np.ndarray,         # (B,) float64
    pip_value: float,
    slippage_pips: float,
    # Signal data
    sig_bar_index: np.ndarray,  # (S,) int64
    sig_direction: np.ndarray,  # (S,) int64
    sig_entry_price: np.ndarray,  # (S,) float64
    sig_hour: np.ndarray,       # (S,) int64
    sig_day: np.ndarray,        # (S,) int64
    sig_atr_pips: np.ndarray,   # (S,) float64
    sig_swing_sl: np.ndarray,   # (S,) float64 — pre-computed swing SL prices
    sig_filter_value: np.ndarray,  # (S,) float64 — strategy-specific filter value
    sig_variant: np.ndarray,    # (S,) int64 — signal variant index (-1 = no variant)
    # Parameter matrix
    param_matrix: np.ndarray,   # (N, P) float64 — actual values, not indices
    param_layout: np.ndarray,   # (NUM_PL,) int64 — maps PL_* to column index
    # Execution mode
    exec_mode: int,             # EXEC_BASIC or EXEC_FULL
    # Pre-allocated output
    metrics_out: np.ndarray,    # (N, NUM_METRICS) float64 — written in place
    # Working memory: max trades per trial
    max_trades: int,
    # Annualization: estimated bars per year for this timeframe
    bars_per_year: float,
    # Execution costs
    commission_pips: float,
    max_spread_pips: float,
    # Sub-bar arrays for M1 trade simulation (or identity)
    sub_high: np.ndarray,       # (M,) float64 — M1 or identity high
    sub_low: np.ndarray,        # (M,) float64 — M1 or identity low
    sub_close: np.ndarray,      # (M,) float64 — M1 or identity close
    sub_spread: np.ndarray,     # (M,) float64 — M1 or identity spread
    h1_to_sub_start: np.ndarray,  # (B,) int64 — H1 bar → sub-bar start index
    h1_to_sub_end: np.ndarray,    # (B,) int64 — H1 bar → sub-bar end index
) -> None:
    """Evaluate N parameter sets in parallel.

    This is the main hot loop. Each trial filters signals, computes SL/TP,
    simulates trades bar-by-bar, and computes metrics.
    """
    n_trials = param_matrix.shape[0]
    n_signals = len(sig_bar_index)
    n_bars = len(high)

    # Pre-allocate PnL buffers OUTSIDE prange (zero-allocation rule)
    pnl_buffers = np.empty((n_trials, max_trades), dtype=np.float64)

    for trial in prange(n_trials):
        # Get params for this trial
        params = param_matrix[trial]

        # Extract standard params via layout
        sl_mode = int(params[param_layout[PL_SL_MODE]])
        sl_fixed_pips = params[param_layout[PL_SL_FIXED_PIPS]]
        sl_atr_mult = params[param_layout[PL_SL_ATR_MULT]]
        tp_mode = int(params[param_layout[PL_TP_MODE]])
        tp_rr_ratio = params[param_layout[PL_TP_RR_RATIO]]
        tp_atr_mult = params[param_layout[PL_TP_ATR_MULT]]
        tp_fixed_pips_val = params[param_layout[PL_TP_FIXED_PIPS]]
        hours_start = int(params[param_layout[PL_HOURS_START]])
        hours_end = int(params[param_layout[PL_HOURS_END]])
        days_bitmask = int(params[param_layout[PL_DAYS_BITMASK]])

        # Management params (only used in full mode)
        trailing_mode = int(params[param_layout[PL_TRAILING_MODE]])
        trail_activate = params[param_layout[PL_TRAIL_ACTIVATE]]
        trail_distance = params[param_layout[PL_TRAIL_DISTANCE]]
        trail_atr_m = params[param_layout[PL_TRAIL_ATR_MULT]]
        be_enabled = int(params[param_layout[PL_BREAKEVEN_ENABLED]])
        be_trigger = params[param_layout[PL_BREAKEVEN_TRIGGER]]
        be_offset = params[param_layout[PL_BREAKEVEN_OFFSET]]
        partial_en = int(params[param_layout[PL_PARTIAL_ENABLED]])
        partial_pct = params[param_layout[PL_PARTIAL_PCT]]
        partial_trig = params[param_layout[PL_PARTIAL_TRIGGER]]
        max_bars_val = int(params[param_layout[PL_MAX_BARS]])
        stale_en = int(params[param_layout[PL_STALE_ENABLED]])
        stale_bars_val = int(params[param_layout[PL_STALE_BARS]])
        stale_atr = params[param_layout[PL_STALE_ATR_THRESH]]

        # --- Filter signals ---
        trade_count = 0
        total_sl_pips = 0.0
        pnl_buffer = pnl_buffers[trial]

        # Strategy-specific signal filter params (-1 in layout = disabled)
        variant_col = param_layout[PL_SIGNAL_VARIANT]
        if variant_col >= 0:
            trial_variant = int(params[variant_col])
        else:
            trial_variant = -1
        bfm_col = param_layout[PL_BUY_FILTER_MAX]
        if bfm_col >= 0:
            buy_filter_max = params[bfm_col]
        else:
            buy_filter_max = -1.0
        sfm_col = param_layout[PL_SELL_FILTER_MIN]
        if sfm_col >= 0:
            sell_filter_min = params[sfm_col]
        else:
            sell_filter_min = -1.0

        for si in range(n_signals):
            # Signal variant filter (e.g., RSI period matching)
            if trial_variant >= 0 and sig_variant[si] >= 0:
                if sig_variant[si] != trial_variant:
                    continue

            # Strategy-specific value filter (exact match on threshold)
            direction = sig_direction[si]
            if buy_filter_max >= 0.0 and direction == DIR_BUY:
                if sig_filter_value[si] != buy_filter_max:
                    continue
            if sell_filter_min >= 0.0 and direction == DIR_SELL:
                if sig_filter_value[si] != sell_filter_min:
                    continue

            # Time filter
            day_bit = 1 << sig_day[si]
            if (days_bitmask & day_bit) == 0:
                continue

            h = sig_hour[si]
            if hours_start <= hours_end:
                if not (hours_start <= h <= hours_end):
                    continue
            else:
                if not (h >= hours_start or h <= hours_end):
                    continue

            # Compute SL/TP
            bar_idx = sig_bar_index[si]
            entry_p = sig_entry_price[si]
            atr_p = sig_atr_pips[si]
            swing_sl = sig_swing_sl[si]

            # Max spread filter: skip signals where spread exceeds threshold
            if max_spread_pips > 0.0:
                spread_at_signal = spread[bar_idx] / pip_value  # convert to pips
                if np.isnan(spread_at_signal) or spread_at_signal > max_spread_pips:
                    continue

            sl_p, tp_p, sl_pip, tp_pip = _compute_sl_tp(
                direction, entry_p, atr_p, pip_value,
                sl_mode, sl_fixed_pips, sl_atr_mult, swing_sl,
                tp_mode, tp_rr_ratio, tp_atr_mult, tp_fixed_pips_val,
            )

            total_sl_pips += sl_pip

            # Simulate trade
            if exec_mode == EXEC_BASIC:
                pnl, exit_reason = _simulate_trade_basic(
                    direction, bar_idx, entry_p, sl_p, tp_p,
                    high, low, close, spread, pip_value, slippage_pips, n_bars,
                    commission_pips,
                    sub_high, sub_low, sub_close, sub_spread,
                    h1_to_sub_start, h1_to_sub_end,
                )
            else:
                pnl, exit_reason = _simulate_trade_full(
                    direction, bar_idx, entry_p, sl_p, tp_p, atr_p,
                    high, low, close, spread, pip_value, slippage_pips, n_bars,
                    trailing_mode, trail_activate, trail_distance, trail_atr_m,
                    be_enabled, be_trigger, be_offset,
                    partial_en, partial_pct, partial_trig,
                    max_bars_val, stale_en, stale_bars_val, stale_atr,
                    commission_pips,
                    sub_high, sub_low, sub_close, sub_spread,
                    h1_to_sub_start, h1_to_sub_end,
                )

            if trade_count < max_trades:
                pnl_buffer[trade_count] = pnl
                trade_count += 1

        # --- Compute metrics ---
        avg_sl = total_sl_pips / trade_count if trade_count > 0 else 30.0
        _compute_metrics_inline(
            pnl_buffer, trade_count, avg_sl,
            n_bars, bars_per_year, metrics_out[trial],
        )
