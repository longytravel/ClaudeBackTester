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
NUM_PL = 24


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
    spread_arr: np.ndarray,
    pip_value: float,
    slippage_pips: float,
    num_bars: int,
) -> tuple[float, int]:
    """Simulate a single trade with SL/TP only (basic mode).

    Returns (pnl_pips, exit_reason).
    Conservative tiebreak: if both SL and TP hit on the same bar, SL wins.
    """
    is_buy = direction == DIR_BUY
    slippage_price = slippage_pips * pip_value

    # Apply entry slippage + spread
    if is_buy:
        actual_entry = entry_price + slippage_price
        # BUY: pay the spread on entry
        spread_at_entry = spread_arr[entry_bar] if entry_bar < len(spread_arr) else 0.0
        actual_entry += spread_at_entry * pip_value
    else:
        actual_entry = entry_price - slippage_price
        # SELL: no spread adjustment on entry (we sell at bid)

    # Walk forward bar by bar
    for bar in range(entry_bar + 1, num_bars):
        bar_high = high[bar]
        bar_low = low[bar]

        if is_buy:
            # Check SL first (conservative tiebreak)
            if bar_low <= sl_price:
                pnl = (sl_price - actual_entry) / pip_value
                return pnl, EXIT_SL
            if bar_high >= tp_price:
                pnl = (tp_price - actual_entry) / pip_value
                return pnl, EXIT_TP
        else:
            # SELL: SL is above, TP is below
            if bar_high >= sl_price:
                pnl = (actual_entry - sl_price) / pip_value
                return pnl, EXIT_SL
            if bar_low <= tp_price:
                pnl = (actual_entry - tp_price) / pip_value
                return pnl, EXIT_TP

    # Trade still open at end of data — close at last bar close
    # (We use high/low arrays; approximate close as (H+L)/2 of last bar)
    last_bar = num_bars - 1
    close_price = (high[last_bar] + low[last_bar]) / 2.0
    if is_buy:
        pnl = (close_price - actual_entry) / pip_value
    else:
        pnl = (actual_entry - close_price) / pip_value
    return pnl, EXIT_NONE


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
) -> tuple[float, int]:
    """Simulate a single trade with full management features.

    Returns (pnl_pips, exit_reason).
    """
    is_buy = direction == DIR_BUY
    slippage_price = slippage_pips * pip_value

    # Apply entry costs
    if is_buy:
        actual_entry = entry_price + slippage_price
        spread_at_entry = spread_arr[entry_bar] if entry_bar < len(spread_arr) else 0.0
        actual_entry += spread_at_entry * pip_value
    else:
        actual_entry = entry_price - slippage_price

    current_sl = sl_price
    position_pct = 1.0  # 1.0 = full position, reduced by partial close
    partial_done = False
    be_locked = False
    trailing_active = False
    realized_pnl_pips = 0.0

    bars_held = 0

    for bar in range(entry_bar + 1, num_bars):
        bar_high = high[bar]
        bar_low = low[bar]
        bar_close = close[bar]
        bars_held += 1

        # --- Max bars exit ---
        if max_bars > 0 and bars_held >= max_bars:
            if is_buy:
                pnl = (bar_close - actual_entry) / pip_value * position_pct
            else:
                pnl = (actual_entry - bar_close) / pip_value * position_pct
            return realized_pnl_pips + pnl, EXIT_MAX_BARS

        # --- Stale exit: check if price hasn't moved enough ---
        if stale_enabled > 0 and bars_held >= stale_bars:
            # Check if recent range is below threshold
            lookback_start = max(entry_bar + 1, bar - stale_bars + 1)
            max_range = 0.0
            for b in range(lookback_start, bar + 1):
                r = (high[b] - low[b]) / pip_value
                if r > max_range:
                    max_range = r
            if max_range < stale_atr_thresh * atr_pips:
                if is_buy:
                    pnl = (bar_close - actual_entry) / pip_value * position_pct
                else:
                    pnl = (actual_entry - bar_close) / pip_value * position_pct
                return realized_pnl_pips + pnl, EXIT_STALE

        # Current floating PnL in pips
        if is_buy:
            float_pnl_pips = (bar_high - actual_entry) / pip_value  # Best case this bar
            worst_pnl_pips = (bar_low - actual_entry) / pip_value
        else:
            float_pnl_pips = (actual_entry - bar_low) / pip_value
            worst_pnl_pips = (actual_entry - bar_high) / pip_value

        # --- Breakeven lock ---
        if breakeven_enabled > 0 and not be_locked:
            if float_pnl_pips >= breakeven_trigger_pips:
                be_locked = True
                be_price = actual_entry + breakeven_offset_pips * pip_value if is_buy \
                    else actual_entry - breakeven_offset_pips * pip_value
                # Only tighten SL, never widen it
                if is_buy and be_price > current_sl:
                    current_sl = be_price
                elif not is_buy and be_price < current_sl:
                    current_sl = be_price

        # --- Trailing stop ---
        if trailing_mode != TRAIL_OFF:
            if not trailing_active:
                if float_pnl_pips >= trail_activate_pips:
                    trailing_active = True

            if trailing_active:
                if trailing_mode == TRAIL_FIXED_PIP:
                    trail_dist = trail_distance_pips * pip_value
                else:  # TRAIL_ATR_CHANDELIER
                    trail_dist = trail_atr_mult * atr_pips * pip_value

                if is_buy:
                    new_sl = bar_high - trail_dist
                    if new_sl > current_sl:
                        current_sl = new_sl
                else:
                    new_sl = bar_low + trail_dist
                    if new_sl < current_sl:
                        current_sl = new_sl

        # --- Partial close ---
        if partial_enabled > 0 and not partial_done:
            if float_pnl_pips >= partial_trigger_pips:
                partial_done = True
                close_pct = partial_pct / 100.0
                if is_buy:
                    partial_pnl = (bar_close - actual_entry) / pip_value * close_pct
                else:
                    partial_pnl = (actual_entry - bar_close) / pip_value * close_pct
                realized_pnl_pips += partial_pnl
                position_pct -= close_pct

        # --- Check SL (with possible trailing/BE adjustments) ---
        if is_buy:
            if bar_low <= current_sl:
                pnl = (current_sl - actual_entry) / pip_value * position_pct
                exit_code = EXIT_SL
                if trailing_active:
                    exit_code = EXIT_TRAILING
                elif be_locked:
                    exit_code = EXIT_BREAKEVEN
                return realized_pnl_pips + pnl, exit_code
            if bar_high >= tp_price:
                pnl = (tp_price - actual_entry) / pip_value * position_pct
                return realized_pnl_pips + pnl, EXIT_TP
        else:
            if bar_high >= current_sl:
                pnl = (actual_entry - current_sl) / pip_value * position_pct
                exit_code = EXIT_SL
                if trailing_active:
                    exit_code = EXIT_TRAILING
                elif be_locked:
                    exit_code = EXIT_BREAKEVEN
                return realized_pnl_pips + pnl, exit_code
            if bar_low <= tp_price:
                pnl = (actual_entry - tp_price) / pip_value * position_pct
                return realized_pnl_pips + pnl, EXIT_TP

    # End of data — close remaining position
    last_bar = num_bars - 1
    close_price = close[last_bar] if last_bar < len(close) else (high[last_bar] + low[last_bar]) / 2.0
    if is_buy:
        pnl = (close_price - actual_entry) / pip_value * position_pct
    else:
        pnl = (actual_entry - close_price) / pip_value * position_pct
    return realized_pnl_pips + pnl, EXIT_NONE


@njit(cache=True)
def _compute_metrics_inline(
    pnl_arr: np.ndarray,
    trade_count: int,
    avg_sl_pips: float,
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

    # Sharpe (using trade count as annualization, matching Python metrics)
    if std > 0:
        metrics_row[M_SHARPE] = (mean / std) * np.sqrt(float(n))
    else:
        metrics_row[M_SHARPE] = 0.0

    # Sortino
    if down_count > 0:
        downside_std = np.sqrt(down_sq_sum / down_count)
        if downside_std > 0:
            metrics_row[M_SORTINO] = (mean / downside_std) * np.sqrt(float(n))
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

    # Quality Score
    so = metrics_row[M_SORTINO]
    r2 = metrics_row[M_R_SQUARED]
    pf_c = min(pf, 5.0)
    trades_f = np.sqrt(min(float(n), 200.0))
    ret = metrics_row[M_RETURN_PCT]
    ret_f = 1.0 + min(ret, 200.0) / 100.0
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
    # Parameter matrix
    param_matrix: np.ndarray,   # (N, P) float64 — actual values, not indices
    param_layout: np.ndarray,   # (NUM_PL,) int64 — maps PL_* to column index
    # Execution mode
    exec_mode: int,             # EXEC_BASIC or EXEC_FULL
    # Pre-allocated output
    metrics_out: np.ndarray,    # (N, NUM_METRICS) float64 — written in place
    # Working memory: max trades per trial
    max_trades: int,
) -> None:
    """Evaluate N parameter sets in parallel.

    This is the main hot loop. Each trial filters signals, computes SL/TP,
    simulates trades bar-by-bar, and computes metrics.
    """
    n_trials = param_matrix.shape[0]
    n_signals = len(sig_bar_index)
    n_bars = len(high)

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
        # We can't allocate inside prange, so use a fixed-size buffer
        # by iterating signals and counting/tracking which pass
        trade_count = 0
        total_sl_pips = 0.0
        # PnL array — pre-allocated per trial slice
        # We use a local array trick: write to a contiguous region
        pnl_buffer = np.empty(max_trades, dtype=np.float64)

        for si in range(n_signals):
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
            direction = sig_direction[si]
            entry_p = sig_entry_price[si]
            atr_p = sig_atr_pips[si]
            swing_sl = sig_swing_sl[si]

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
                    high, low, spread, pip_value, slippage_pips, n_bars,
                )
            else:
                pnl, exit_reason = _simulate_trade_full(
                    direction, bar_idx, entry_p, sl_p, tp_p, atr_p,
                    high, low, close, spread, pip_value, slippage_pips, n_bars,
                    trailing_mode, trail_activate, trail_distance, trail_atr_m,
                    be_enabled, be_trigger, be_offset,
                    partial_en, partial_pct, partial_trig,
                    max_bars_val, stale_en, stale_bars_val, stale_atr,
                )

            if trade_count < max_trades:
                pnl_buffer[trade_count] = pnl
                trade_count += 1

        # --- Compute metrics ---
        avg_sl = total_sl_pips / trade_count if trade_count > 0 else 30.0
        _compute_metrics_inline(pnl_buffer, trade_count, avg_sl, metrics_out[trial])
