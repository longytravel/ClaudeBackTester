"""Single-trial detailed telemetry output.

Python-based (not JIT) — captures per-trade detail:
MFE, MAE, exit reason, bars held, entry/exit prices.

Used for post-analysis and strategy debugging, not optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backtester.core.dtypes import (
    DIR_BUY,
    DIR_SELL,
    EXEC_FULL,
    EXIT_BREAKEVEN,
    EXIT_MAX_BARS,
    EXIT_NONE,
    EXIT_SL,
    EXIT_STALE,
    EXIT_TP,
    EXIT_TRAILING,
    TRAIL_ATR_CHANDELIER,
    TRAIL_FIXED_PIP,
    TRAIL_OFF,
)
from backtester.core.encoding import encode_params
from backtester.core.engine import BacktestEngine
from backtester.core.metrics import compute_metrics


EXIT_NAMES = {
    EXIT_NONE: "open",
    EXIT_SL: "stop_loss",
    EXIT_TP: "take_profit",
    EXIT_TRAILING: "trailing_stop",
    EXIT_BREAKEVEN: "breakeven",
    EXIT_MAX_BARS: "max_bars",
    EXIT_STALE: "stale_exit",
}


@dataclass
class TradeDetail:
    """Detailed information about a single trade."""
    signal_index: int
    bar_entry: int
    bar_exit: int
    bars_held: int
    direction: str        # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    sl_pips: float
    tp_pips: float
    pnl_pips: float
    exit_reason: str
    mfe_pips: float       # Maximum Favorable Excursion
    mae_pips: float       # Maximum Adverse Excursion


@dataclass
class TelemetryResult:
    """Full telemetry output for a single parameter set."""
    trades: list[TradeDetail] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    n_signals_total: int = 0
    n_signals_filtered: int = 0


def run_telemetry(
    engine: BacktestEngine,
    params: dict[str, Any],
    exec_mode: int = EXEC_FULL,
) -> TelemetryResult:
    """Run a single trial with full per-trade telemetry.

    This is a Python implementation that mirrors the JIT loop but
    captures detailed per-trade information.
    """
    result = TelemetryResult(params=params)

    # Get encoded params
    spec = engine.encoding
    encoded = encode_params(spec, params)

    pip = engine.pip_value
    slippage = engine.slippage_pips

    # Extract param values
    sl_mode = params.get("sl_mode", "fixed_pips")
    sl_fixed = params.get("sl_fixed_pips", 30)
    sl_atr_mult = params.get("sl_atr_mult", 1.5)
    tp_mode = params.get("tp_mode", "rr_ratio")
    tp_rr = params.get("tp_rr_ratio", 2.0)
    tp_atr_mult = params.get("tp_atr_mult", 2.0)
    tp_fixed = params.get("tp_fixed_pips", 60)
    hours_start = params.get("allowed_hours_start", 0)
    hours_end = params.get("allowed_hours_end", 23)
    allowed_days = params.get("allowed_days", [0, 1, 2, 3, 4])
    trailing_mode = params.get("trailing_mode", "off")
    trail_activate = params.get("trail_activate_pips", 0)
    trail_distance = params.get("trail_distance_pips", 10)
    trail_atr_m = params.get("trail_atr_mult", 2.0)
    be_enabled = params.get("breakeven_enabled", False)
    be_trigger = params.get("breakeven_trigger_pips", 20)
    be_offset = params.get("breakeven_offset_pips", 2)
    partial_en = params.get("partial_close_enabled", False)
    partial_pct = params.get("partial_close_pct", 50)
    partial_trig = params.get("partial_close_trigger_pips", 30)
    max_bars = params.get("max_bars", 0)
    stale_en = params.get("stale_exit_enabled", False)
    stale_bars = params.get("stale_exit_bars", 50)
    stale_atr_thresh = params.get("stale_exit_atr_threshold", 0.5)

    # Build day set
    day_set = set(allowed_days) if isinstance(allowed_days, list) else {0, 1, 2, 3, 4}

    # Signal filter params (for strategy-specific filtering like RSI variant/threshold)
    # These may be mapped from strategy-specific params (e.g. rsi_period → signal_variant)
    trial_variant = -1
    buy_filter_max_val = -1.0
    sell_filter_min_val = -1.0
    # Check for direct filter params or mapped strategy params
    if "signal_variant" in params:
        trial_variant = int(params["signal_variant"])
    elif "rsi_period" in params:
        trial_variant = int(params["rsi_period"])
    elif "ema_combo" in params:
        trial_variant = int(params["ema_combo"])
    if "buy_filter_max" in params:
        buy_filter_max_val = float(params["buy_filter_max"])
    elif "rsi_oversold" in params:
        buy_filter_max_val = float(params["rsi_oversold"])
    if "sell_filter_min" in params:
        sell_filter_min_val = float(params["sell_filter_min"])
    elif "rsi_overbought" in params:
        sell_filter_min_val = float(params["rsi_overbought"])

    result.n_signals_total = engine.n_signals
    pnl_list = []

    for si in range(engine.n_signals):
        bar_idx = int(engine.sig_bar_index[si])
        direction = int(engine.sig_direction[si])
        entry_price = float(engine.sig_entry_price[si])
        hour = int(engine.sig_hour[si])
        day = int(engine.sig_day[si])
        atr_pips = float(engine.sig_atr_pips[si])

        # Signal variant filter (e.g., RSI period matching)
        if trial_variant >= 0 and hasattr(engine, 'sig_variant'):
            sig_var = int(engine.sig_variant[si])
            if sig_var >= 0 and sig_var != trial_variant:
                continue

        # Strategy-specific value filter (exact match on threshold)
        if hasattr(engine, 'sig_filter_value'):
            fv = float(engine.sig_filter_value[si])
            if buy_filter_max_val >= 0.0 and direction == DIR_BUY:
                if fv != buy_filter_max_val:
                    continue
            if sell_filter_min_val >= 0.0 and direction == DIR_SELL:
                if fv != sell_filter_min_val:
                    continue

        # Time filter
        if day not in day_set:
            continue
        if hours_start <= hours_end:
            if not (hours_start <= hour <= hours_end):
                continue
        else:
            if not (hour >= hours_start or hour <= hours_end):
                continue

        # Max spread filter
        if engine.max_spread_pips > 0:
            spread_at_signal = engine.spread[bar_idx] / pip if bar_idx < len(engine.spread) else 0.0
            if np.isnan(spread_at_signal) or spread_at_signal > engine.max_spread_pips:
                continue

        result.n_signals_filtered += 1

        # Compute SL/TP
        atr_price = atr_pips * pip
        is_buy = direction == DIR_BUY

        # SL distance
        if sl_mode == "atr_based":
            sl_dist = atr_price * sl_atr_mult
        elif sl_mode == "swing":
            swing_sl = float(engine.sig_swing_sl[si])
            if not np.isnan(swing_sl):
                sl_dist = abs(entry_price - swing_sl)
                sl_dist = max(sl_dist, 5 * pip)
            else:
                sl_dist = atr_price * 1.5
        else:
            sl_dist = sl_fixed * pip

        sl_pips_val = sl_dist / pip

        if is_buy:
            sl_price = entry_price - sl_dist
        else:
            sl_price = entry_price + sl_dist

        # TP distance
        if tp_mode == "atr_based":
            tp_dist = atr_price * tp_atr_mult
        elif tp_mode == "fixed_pips":
            tp_dist = tp_fixed * pip
        else:
            tp_dist = sl_dist * tp_rr

        tp_dist = max(tp_dist, sl_dist)
        tp_pips_val = tp_dist / pip

        if is_buy:
            tp_price = entry_price + tp_dist
        else:
            tp_price = entry_price - tp_dist

        # Simulate trade — use sub-bar arrays from engine
        sub_high = engine.sub_high
        sub_low = engine.sub_low
        sub_close = engine.sub_close
        sub_spread = engine.sub_spread
        h1_to_sub_start = engine.h1_to_sub_start
        h1_to_sub_end = engine.h1_to_sub_end

        actual_entry = entry_price
        if is_buy:
            actual_entry += slippage * pip
            # BUY: pay spread at first sub-bar of entry bar
            sub_entry_start = int(h1_to_sub_start[bar_idx])
            if sub_entry_start < len(sub_spread):
                spread_val = sub_spread[sub_entry_start]
                if not np.isnan(spread_val):
                    actual_entry += spread_val
        else:
            # SELL: slippage works against us (we enter at a worse price)
            actual_entry -= slippage * pip

        current_sl = sl_price
        mfe = 0.0
        mae = 0.0
        exit_reason = EXIT_NONE
        exit_bar = engine.n_bars - 1
        exit_sub_idx = -1
        exit_price = 0.0
        bars_held = 0
        position_pct = 1.0
        realized_pnl_pips = 0.0
        partial_done = False

        trailing_active = False
        be_locked = False

        for bar in range(bar_idx + 1, engine.n_bars):
            bar_high = engine.high[bar]
            bar_low = engine.low[bar]
            bar_close = engine.close[bar]
            bars_held += 1

            # Track MFE/MAE at H1 level (aggregate view)
            if is_buy:
                fav = (bar_high - actual_entry) / pip
                adv = (actual_entry - bar_low) / pip
            else:
                fav = (actual_entry - bar_low) / pip
                adv = (bar_high - actual_entry) / pip
            mfe = max(mfe, fav)
            mae = max(mae, adv)

            # Full mode management checks
            if exec_mode == EXEC_FULL:
                # --- H1-level checks (max_bars, stale) ---
                if max_bars > 0 and bars_held >= max_bars:
                    exit_reason = EXIT_MAX_BARS
                    exit_price = bar_close
                    exit_bar = bar
                    break

                if stale_en and bars_held >= stale_bars:
                    lookback_start = max(bar_idx + 1, bar - stale_bars + 1)
                    max_range = 0.0
                    for b in range(lookback_start, bar + 1):
                        r = (engine.high[b] - engine.low[b]) / pip
                        if r > max_range:
                            max_range = r
                    if max_range < stale_atr_thresh * atr_pips:
                        exit_reason = EXIT_STALE
                        exit_price = bar_close
                        exit_bar = bar
                        break

                # --- Sub-bar trade management ---
                sb_start = int(h1_to_sub_start[bar])
                sb_end = int(h1_to_sub_end[bar])
                for sb in range(sb_start, sb_end):
                    sb_h = sub_high[sb]
                    sb_l = sub_low[sb]
                    sb_c = sub_close[sb]

                    # Floating PnL on this sub-bar
                    if is_buy:
                        float_pnl = (sb_h - actual_entry) / pip
                    else:
                        float_pnl = (actual_entry - sb_l) / pip

                    # Breakeven
                    if be_enabled and not be_locked and float_pnl >= be_trigger:
                        be_locked = True
                        be_price = actual_entry + be_offset * pip if is_buy \
                            else actual_entry - be_offset * pip
                        if is_buy and be_price > current_sl:
                            current_sl = be_price
                        elif not is_buy and be_price < current_sl:
                            current_sl = be_price

                    # Trailing
                    if trailing_mode != "off":
                        if not trailing_active and float_pnl >= trail_activate:
                            trailing_active = True
                        if trailing_active:
                            if trailing_mode == "fixed_pip":
                                t_dist = trail_distance * pip
                            else:
                                t_dist = trail_atr_m * atr_pips * pip
                            if is_buy:
                                new_sl = sb_h - t_dist
                                if new_sl > current_sl:
                                    current_sl = new_sl
                            else:
                                new_sl = sb_l + t_dist
                                if new_sl < current_sl:
                                    current_sl = new_sl

                    # Partial close
                    if partial_en and not partial_done:
                        if float_pnl >= partial_trig:
                            partial_done = True
                            close_pct = partial_pct / 100.0
                            if is_buy:
                                partial_pnl = (sb_c - actual_entry) / pip * close_pct
                            else:
                                partial_pnl = (actual_entry - sb_c) / pip * close_pct
                            realized_pnl_pips += partial_pnl
                            position_pct -= close_pct

                    # Check SL/TP on sub-bar
                    if is_buy:
                        if sb_l <= current_sl:
                            exit_reason = EXIT_SL
                            if trailing_active:
                                exit_reason = EXIT_TRAILING
                            elif be_locked:
                                exit_reason = EXIT_BREAKEVEN
                            exit_price = current_sl
                            exit_sub_idx = sb
                            break
                        if sb_h >= tp_price:
                            exit_reason = EXIT_TP
                            exit_price = tp_price
                            exit_sub_idx = sb
                            break
                    else:
                        if sb_h >= current_sl:
                            exit_reason = EXIT_SL
                            if trailing_active:
                                exit_reason = EXIT_TRAILING
                            elif be_locked:
                                exit_reason = EXIT_BREAKEVEN
                            exit_price = current_sl
                            exit_sub_idx = sb
                            break
                        if sb_l <= tp_price:
                            exit_reason = EXIT_TP
                            exit_price = tp_price
                            exit_sub_idx = sb
                            break

                if exit_reason != EXIT_NONE:
                    exit_bar = bar
                    break
            else:
                # Basic mode — sub-bar SL/TP only
                sb_start = int(h1_to_sub_start[bar])
                sb_end = int(h1_to_sub_end[bar])
                for sb in range(sb_start, sb_end):
                    sb_h = sub_high[sb]
                    sb_l = sub_low[sb]

                    if is_buy:
                        if sb_l <= current_sl:
                            exit_reason = EXIT_SL
                            exit_price = current_sl
                            exit_sub_idx = sb
                            break
                        if sb_h >= tp_price:
                            exit_reason = EXIT_TP
                            exit_price = tp_price
                            exit_sub_idx = sb
                            break
                    else:
                        if sb_h >= current_sl:
                            exit_reason = EXIT_SL
                            exit_price = current_sl
                            exit_sub_idx = sb
                            break
                        if sb_l <= tp_price:
                            exit_reason = EXIT_TP
                            exit_price = tp_price
                            exit_sub_idx = sb
                            break

                if exit_reason != EXIT_NONE:
                    exit_bar = bar
                    break

        # If still open
        if exit_reason == EXIT_NONE:
            exit_price = engine.close[engine.n_bars - 1]

        # Compute PnL (accounting for partial close)
        if is_buy:
            pnl = (exit_price - actual_entry) / pip * position_pct
        else:
            pnl = (actual_entry - exit_price) / pip * position_pct
        pnl += realized_pnl_pips

        # Apply execution costs
        # SELL: deduct exit spread (use sub-bar spread if available)
        if not is_buy:
            if exit_sub_idx >= 0 and exit_sub_idx < len(sub_spread):
                sell_spread = sub_spread[exit_sub_idx]
            elif exit_bar < len(engine.spread):
                sell_spread = engine.spread[exit_bar]
            else:
                sell_spread = 0.0
            if np.isnan(sell_spread):
                sell_spread = 0.0
            pnl -= sell_spread / pip
        # Commission applied to all trades
        pnl -= engine.commission_pips

        pnl_list.append(pnl)

        trade = TradeDetail(
            signal_index=si,
            bar_entry=bar_idx,
            bar_exit=exit_bar,
            bars_held=bars_held,
            direction="BUY" if is_buy else "SELL",
            entry_price=actual_entry,
            exit_price=exit_price,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pips=sl_pips_val,
            tp_pips=tp_pips_val,
            pnl_pips=pnl,
            exit_reason=EXIT_NAMES.get(exit_reason, "unknown"),
            mfe_pips=mfe,
            mae_pips=mae,
        )
        result.trades.append(trade)

    # Compute metrics
    if pnl_list:
        pnl_arr = np.array(pnl_list, dtype=np.float64)
        avg_sl = np.mean([t.sl_pips for t in result.trades])
        # Compute annualized trades per year
        n_trades = len(pnl_list)
        trades_per_year = n_trades * engine.bars_per_year / engine.n_bars
        result.metrics = compute_metrics(pnl_arr, avg_sl, trades_per_year)
    else:
        result.metrics = compute_metrics(np.array([]))

    return result
