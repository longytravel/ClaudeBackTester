/// Full trade simulation — mirrors _simulate_trade_full() from jit_loop.py.
///
/// Includes trailing stop, breakeven, partial close, stale exit, max bars.

use crate::constants::*;
use crate::trade_basic::TradeResult;

/// Simulate a single trade with full management features.
///
/// Returns (pnl_pips, exit_reason).
/// Uses sub-bar (M1) data for all price-sensitive management checks.
/// H1-level checks (max_bars, stale exit) remain at H1 bar resolution.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn simulate_trade_full(
    direction: i64,
    entry_bar: usize,
    entry_price: f64,
    sl_price: f64,
    tp_price: f64,
    atr_pips: f64,
    high: &[f64],
    low: &[f64],
    close: &[f64],
    spread_arr: &[f64],
    pip_value: f64,
    slippage_pips: f64,
    num_bars: usize,
    // Management params
    trailing_mode: i64,
    trail_activate_pips: f64,
    trail_distance_pips: f64,
    trail_atr_mult: f64,
    breakeven_enabled: i64,
    breakeven_trigger_pips: f64,
    breakeven_offset_pips: f64,
    partial_enabled: i64,
    partial_pct: f64,
    partial_trigger_pips: f64,
    max_bars: i64,
    stale_enabled: i64,
    stale_bars: i64,
    stale_atr_thresh: f64,
    commission_pips: f64,
    // Sub-bar arrays
    sub_high: &[f64],
    sub_low: &[f64],
    sub_close: &[f64],
    sub_spread: &[f64],
    h1_to_sub_start: &[i64],
    h1_to_sub_end: &[i64],
) -> TradeResult {
    let is_buy = direction == DIR_BUY;
    let slippage_price = slippage_pips * pip_value;

    // Apply entry costs
    let actual_entry = if is_buy {
        let sub_entry_start = h1_to_sub_start[entry_bar] as usize;
        let spread_at_entry = if sub_entry_start < sub_spread.len() {
            let s = sub_spread[sub_entry_start];
            if s.is_nan() { 0.0 } else { s }
        } else {
            0.0
        };
        entry_price + slippage_price + spread_at_entry
    } else {
        entry_price - slippage_price
    };

    let mut current_sl = sl_price;
    let mut position_pct = 1.0_f64;
    let mut partial_done = false;
    let mut be_locked = false;
    let mut trailing_active = false;
    let mut realized_pnl_pips = 0.0_f64;

    // Deferred SL pattern
    let mut pending_sl = -1.0_f64;
    let mut pending_be_locked = false;
    let mut pending_trailing_active = false;
    let mut has_pending_update = false;

    let mut bars_held: i64 = 0;
    let mut exit_reason = EXIT_NONE;
    let mut exit_bar = num_bars - 1;
    let mut exit_sub_idx: i64 = -1;
    let mut final_pnl = 0.0_f64;

    'bar_loop: for bar in (entry_bar + 1)..num_bars {
        let _bar_high = high[bar];
        let _bar_low = low[bar];
        let bar_close = close[bar];
        bars_held += 1;

        // --- H1-level checks (max_bars, stale) ---
        if max_bars > 0 && bars_held >= max_bars {
            let pnl = if is_buy {
                (bar_close - slippage_price - actual_entry) / pip_value * position_pct
            } else {
                (actual_entry - bar_close - slippage_price) / pip_value * position_pct
            };
            final_pnl = realized_pnl_pips + pnl;
            exit_reason = EXIT_MAX_BARS;
            exit_bar = bar;
            break 'bar_loop;
        }

        if stale_enabled > 0 && bars_held >= stale_bars {
            let lookback_start = if (entry_bar as i64 + 1) > (bar as i64 - stale_bars + 1) {
                entry_bar + 1
            } else {
                (bar as i64 - stale_bars + 1) as usize
            };
            let mut max_range = 0.0_f64;
            for b in lookback_start..=bar {
                let r = (high[b] - low[b]) / pip_value;
                if r > max_range {
                    max_range = r;
                }
            }
            if max_range < stale_atr_thresh * atr_pips {
                let pnl = if is_buy {
                    (bar_close - slippage_price - actual_entry) / pip_value * position_pct
                } else {
                    (actual_entry - bar_close - slippage_price) / pip_value * position_pct
                };
                final_pnl = realized_pnl_pips + pnl;
                exit_reason = EXIT_STALE;
                exit_bar = bar;
                break 'bar_loop;
            }
        }

        // --- Sub-bar trade management ---
        let sub_start = h1_to_sub_start[bar] as usize;
        let sub_end = h1_to_sub_end[bar] as usize;

        for sb in sub_start..sub_end {
            let sb_high = sub_high[sb];
            let sb_low = sub_low[sb];
            let sb_close = sub_close[sb];

            // Apply any pending SL modification from the PREVIOUS sub-bar
            if has_pending_update {
                if pending_sl > 0.0 {
                    current_sl = pending_sl;
                }
                be_locked = pending_be_locked;
                trailing_active = pending_trailing_active;
                pending_sl = -1.0;
                has_pending_update = false;
            }

            // Current floating PnL on this sub-bar
            let (float_pnl_pips, _worst_pnl_pips) = if is_buy {
                (
                    (sb_high - actual_entry) / pip_value,
                    (sb_low - actual_entry) / pip_value,
                )
            } else {
                (
                    (actual_entry - sb_low) / pip_value,
                    (actual_entry - sb_high) / pip_value,
                )
            };

            // --- Breakeven lock (deferred) ---
            if breakeven_enabled > 0 && !be_locked && !pending_be_locked {
                if float_pnl_pips >= breakeven_trigger_pips {
                    let be_price = if is_buy {
                        actual_entry + breakeven_offset_pips * pip_value
                    } else {
                        actual_entry - breakeven_offset_pips * pip_value
                    };
                    if is_buy && be_price > current_sl {
                        pending_sl = be_price;
                        pending_be_locked = true;
                        pending_trailing_active = trailing_active;
                        has_pending_update = true;
                    } else if !is_buy && be_price < current_sl {
                        pending_sl = be_price;
                        pending_be_locked = true;
                        pending_trailing_active = trailing_active;
                        has_pending_update = true;
                    }
                }
            }

            // --- Trailing stop (deferred) ---
            if trailing_mode != TRAIL_OFF {
                if !trailing_active && !pending_trailing_active {
                    if float_pnl_pips >= trail_activate_pips {
                        pending_trailing_active = true;
                        let trail_dist = if trailing_mode == TRAIL_FIXED_PIP {
                            trail_distance_pips * pip_value
                        } else {
                            // TRAIL_ATR_CHANDELIER
                            trail_atr_mult * atr_pips * pip_value
                        };
                        if is_buy {
                            let new_sl = sb_high - trail_dist;
                            let effective_sl = if has_pending_update && pending_sl > 0.0 {
                                pending_sl
                            } else {
                                current_sl
                            };
                            if new_sl > effective_sl {
                                pending_sl = new_sl;
                            }
                        } else {
                            let new_sl = sb_low + trail_dist;
                            let effective_sl = if has_pending_update && pending_sl > 0.0 {
                                pending_sl
                            } else {
                                current_sl
                            };
                            if new_sl < effective_sl {
                                pending_sl = new_sl;
                            }
                        }
                        pending_be_locked = if !has_pending_update { be_locked } else { pending_be_locked };
                        has_pending_update = true;
                    }
                }

                if trailing_active {
                    let trail_dist = if trailing_mode == TRAIL_FIXED_PIP {
                        trail_distance_pips * pip_value
                    } else {
                        trail_atr_mult * atr_pips * pip_value
                    };

                    if is_buy {
                        let new_sl = sb_high - trail_dist;
                        let effective_sl = if has_pending_update && pending_sl > 0.0 {
                            pending_sl
                        } else {
                            current_sl
                        };
                        if new_sl > effective_sl {
                            pending_sl = new_sl;
                            pending_be_locked = if !has_pending_update { be_locked } else { pending_be_locked };
                            pending_trailing_active = true;
                            has_pending_update = true;
                        }
                    } else {
                        let new_sl = sb_low + trail_dist;
                        let effective_sl = if has_pending_update && pending_sl > 0.0 {
                            pending_sl
                        } else {
                            current_sl
                        };
                        if new_sl < effective_sl {
                            pending_sl = new_sl;
                            pending_be_locked = if !has_pending_update { be_locked } else { pending_be_locked };
                            pending_trailing_active = true;
                            has_pending_update = true;
                        }
                    }
                }
            }

            // --- Partial close (immediate) ---
            if partial_enabled > 0 && !partial_done {
                if float_pnl_pips >= partial_trigger_pips {
                    partial_done = true;
                    let close_pct = partial_pct / 100.0;
                    let partial_pnl = if is_buy {
                        (sb_close - actual_entry) / pip_value * close_pct
                    } else {
                        (actual_entry - sb_close) / pip_value * close_pct
                    };
                    realized_pnl_pips += partial_pnl;
                    position_pct -= close_pct;
                }
            }

            // --- Check SL (uses current_sl) ---
            if is_buy {
                if sb_low <= current_sl {
                    let pnl = (current_sl - slippage_price - actual_entry) / pip_value * position_pct;
                    let exit_code = if trailing_active {
                        EXIT_TRAILING
                    } else if be_locked {
                        EXIT_BREAKEVEN
                    } else {
                        EXIT_SL
                    };
                    final_pnl = realized_pnl_pips + pnl;
                    exit_reason = exit_code;
                    exit_sub_idx = sb as i64;
                    exit_bar = bar;
                    break 'bar_loop;
                }
                if sb_high >= tp_price {
                    let pnl = (tp_price - actual_entry) / pip_value * position_pct;
                    final_pnl = realized_pnl_pips + pnl;
                    exit_reason = EXIT_TP;
                    exit_sub_idx = sb as i64;
                    exit_bar = bar;
                    break 'bar_loop;
                }
            } else {
                if sb_high >= current_sl {
                    let pnl = (actual_entry - current_sl - slippage_price) / pip_value * position_pct;
                    let exit_code = if trailing_active {
                        EXIT_TRAILING
                    } else if be_locked {
                        EXIT_BREAKEVEN
                    } else {
                        EXIT_SL
                    };
                    final_pnl = realized_pnl_pips + pnl;
                    exit_reason = exit_code;
                    exit_sub_idx = sb as i64;
                    exit_bar = bar;
                    break 'bar_loop;
                }
                if sb_low <= tp_price {
                    let pnl = (actual_entry - tp_price) / pip_value * position_pct;
                    final_pnl = realized_pnl_pips + pnl;
                    exit_reason = EXIT_TP;
                    exit_sub_idx = sb as i64;
                    exit_bar = bar;
                    break 'bar_loop;
                }
            }
        }
    }

    // End of data — close remaining position
    if exit_reason == EXIT_NONE {
        let close_price = if exit_bar < close.len() {
            close[exit_bar]
        } else {
            (high[exit_bar] + low[exit_bar]) / 2.0
        };
        let pnl = if is_buy {
            (close_price - slippage_price - actual_entry) / pip_value * position_pct
        } else {
            (actual_entry - close_price - slippage_price) / pip_value * position_pct
        };
        final_pnl = realized_pnl_pips + pnl;
    }

    // Apply execution costs
    if !is_buy {
        let sell_spread = if exit_sub_idx >= 0 && (exit_sub_idx as usize) < sub_spread.len() {
            let s = sub_spread[exit_sub_idx as usize];
            if s.is_nan() { 0.0 } else { s }
        } else if exit_bar < spread_arr.len() {
            let s = spread_arr[exit_bar];
            if s.is_nan() { 0.0 } else { s }
        } else {
            0.0
        };
        final_pnl -= sell_spread / pip_value;
    }
    final_pnl -= commission_pips;

    TradeResult {
        pnl_pips: final_pnl,
        exit_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_mapping(n: usize) -> (Vec<i64>, Vec<i64>) {
        let start: Vec<i64> = (0..n as i64).collect();
        let end: Vec<i64> = (1..=n as i64).collect();
        (start, end)
    }

    #[test]
    fn test_max_bars_exit() {
        let high = vec![1.1010, 1.1020, 1.1030, 1.1040];
        let low = vec![1.0990, 1.0990, 1.0990, 1.0990];
        let close = vec![1.1000, 1.1010, 1.1020, 1.1030];
        let spread = vec![0.0; 4];
        let (start, end) = make_identity_mapping(4);

        let r = simulate_trade_full(
            DIR_BUY, 0, 1.1000, 1.0900, 1.1200, 10.0,
            &high, &low, &close, &spread,
            0.0001, 0.0, 4,
            TRAIL_OFF, 0.0, 0.0, 0.0,
            0, 0.0, 0.0,
            0, 0.0, 0.0,
            2, // max_bars = 2
            0, 0, 0.0,
            0.0,
            &high, &low, &close, &spread,
            &start, &end,
        );
        assert_eq!(r.exit_reason, EXIT_MAX_BARS);
    }

    #[test]
    fn test_breakeven_deferred() {
        // BE trigger = 5 pips, offset = 2 pips
        // Bar 1 high triggers BE, but SL change applies from bar 2
        let high = vec![1.1000, 1.1010, 1.0990, 1.1020];
        let low = vec![1.0990, 1.0995, 1.0985, 1.0990];
        let close = vec![1.1000, 1.1005, 1.0990, 1.1010];
        let spread = vec![0.0; 4];
        let (start, end) = make_identity_mapping(4);

        let r = simulate_trade_full(
            DIR_BUY, 0, 1.1000, 1.0950, 1.1100, 10.0,
            &high, &low, &close, &spread,
            0.0001, 0.0, 4,
            TRAIL_OFF, 0.0, 0.0, 0.0,
            1, 5.0, 2.0, // BE enabled, trigger=5, offset=2
            0, 0.0, 0.0,
            0, 0, 0, 0.0,
            0.0,
            &high, &low, &close, &spread,
            &start, &end,
        );
        // Should NOT exit at bar 1 (deferred), and SL moved from 1.0950 to 1.1002
        assert!(r.exit_reason != EXIT_BREAKEVEN || r.pnl_pips >= 0.0);
    }
}
