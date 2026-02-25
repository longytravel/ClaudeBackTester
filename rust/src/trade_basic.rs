/// Basic trade simulation (SL/TP only) — mirrors _simulate_trade_basic() from jit_loop.py.

use crate::constants::*;

/// Result of a trade simulation.
pub struct TradeResult {
    pub pnl_pips: f64,
    pub exit_reason: i64,
}

/// Simulate a single trade with SL/TP only (basic mode).
///
/// Conservative tiebreak: if both SL and TP hit on the same sub-bar, SL wins.
/// Uses sub-bar (M1) data for SL/TP checks when available.
#[inline(always)]
pub fn simulate_trade_basic(
    direction: i64,
    entry_bar: usize,
    entry_price: f64,
    sl_price: f64,
    tp_price: f64,
    _high: &[f64],
    _low: &[f64],
    close: &[f64],
    spread_arr: &[f64],
    pip_value: f64,
    slippage_pips: f64,
    num_bars: usize,
    commission_pips: f64,
    // Sub-bar arrays
    sub_high: &[f64],
    sub_low: &[f64],
    _sub_close: &[f64],
    sub_spread: &[f64],
    h1_to_sub_start: &[i64],
    h1_to_sub_end: &[i64],
) -> TradeResult {
    let is_buy = direction == DIR_BUY;
    let slippage_price = slippage_pips * pip_value;

    // Apply entry slippage + spread
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

    let mut exit_reason = EXIT_NONE;
    let mut exit_bar = num_bars - 1;
    let mut pnl = 0.0_f64;
    let mut exit_sub_idx: i64 = -1;

    // Walk forward bar by bar, checking sub-bars within each bar
    'outer: for bar in (entry_bar + 1)..num_bars {
        let sub_start = h1_to_sub_start[bar] as usize;
        let sub_end = h1_to_sub_end[bar] as usize;

        for sb in sub_start..sub_end {
            let sb_high = sub_high[sb];
            let sb_low = sub_low[sb];

            if is_buy {
                // Check SL first (conservative tiebreak)
                if sb_low <= sl_price {
                    pnl = (sl_price - slippage_price - actual_entry) / pip_value;
                    exit_reason = EXIT_SL;
                    exit_sub_idx = sb as i64;
                    exit_bar = bar;
                    break 'outer;
                }
                if sb_high >= tp_price {
                    pnl = (tp_price - actual_entry) / pip_value;
                    exit_reason = EXIT_TP;
                    exit_sub_idx = sb as i64;
                    exit_bar = bar;
                    break 'outer;
                }
            } else {
                // SELL: SL is above, TP is below
                if sb_high >= sl_price {
                    pnl = (actual_entry - sl_price - slippage_price) / pip_value;
                    exit_reason = EXIT_SL;
                    exit_sub_idx = sb as i64;
                    exit_bar = bar;
                    break 'outer;
                }
                if sb_low <= tp_price {
                    pnl = (actual_entry - tp_price) / pip_value;
                    exit_reason = EXIT_TP;
                    exit_sub_idx = sb as i64;
                    exit_bar = bar;
                    break 'outer;
                }
            }
        }
    }

    // Trade still open at end of data
    if exit_reason == EXIT_NONE {
        let close_price = close[exit_bar];
        pnl = if is_buy {
            (close_price - slippage_price - actual_entry) / pip_value
        } else {
            (actual_entry - close_price - slippage_price) / pip_value
        };
    }

    // Apply execution costs
    // SELL: deduct exit spread
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
        pnl -= sell_spread / pip_value;
    }
    // Commission applied to all trades
    pnl -= commission_pips;

    TradeResult {
        pnl_pips: pnl,
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
    fn test_buy_sl_hit() {
        // Entry at bar 0, price 1.10, SL at 1.098, TP at 1.104
        // Bar 1 low hits SL
        let high = vec![1.1010, 1.1005, 1.1020];
        let low = vec![1.0990, 1.0975, 1.1000];
        let close = vec![1.1000, 1.0980, 1.1010];
        let spread = vec![0.0001, 0.0001, 0.0001];
        let (start, end) = make_identity_mapping(3);

        let r = simulate_trade_basic(
            DIR_BUY, 0, 1.1000, 1.0980, 1.1040,
            &high, &low, &close, &spread,
            0.0001, 0.0, 3, 0.0,
            &high, &low, &close, &spread,
            &start, &end,
        );
        assert_eq!(r.exit_reason, EXIT_SL);
        // pnl = (1.0980 - 0 - (1.1000 + 0 + 0.0001)) / 0.0001 = (1.0980 - 1.1001) / 0.0001 = -21
        assert!((r.pnl_pips - (-21.0)).abs() < 1e-6);
    }

    #[test]
    fn test_sell_tp_hit() {
        // SELL at bar 0, price 1.10, SL at 1.102, TP at 1.096
        let high = vec![1.1010, 1.1005, 1.0950];
        let low = vec![1.0990, 1.0990, 1.0950];
        let close = vec![1.1000, 1.0995, 1.0955];
        let spread = vec![0.0001, 0.0001, 0.0001];
        let (start, end) = make_identity_mapping(3);

        let r = simulate_trade_basic(
            DIR_SELL, 0, 1.1000, 1.1020, 1.0960,
            &high, &low, &close, &spread,
            0.0001, 0.0, 3, 0.0,
            &high, &low, &close, &spread,
            &start, &end,
        );
        assert_eq!(r.exit_reason, EXIT_TP);
    }
}
