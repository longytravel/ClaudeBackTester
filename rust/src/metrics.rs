/// Metric computation — mirrors _compute_metrics_inline() from jit_loop.py.

use crate::constants::*;

/// Compute all 10 metrics inline for one trial. Writes to metrics_row.
#[inline(always)]
pub fn compute_metrics_inline(
    pnl_arr: &[f64],
    trade_count: usize,
    avg_sl_pips: f64,
    n_bars: usize,
    bars_per_year: f64,
    metrics_row: &mut [f64],
) {
    let n = trade_count;
    metrics_row[M_TRADES] = n as f64;

    if n == 0 {
        return;
    }

    // Win rate
    let mut wins = 0usize;
    for i in 0..n {
        if pnl_arr[i] > 0.0 {
            wins += 1;
        }
    }
    metrics_row[M_WIN_RATE] = wins as f64 / n as f64;

    // Profit factor
    let mut gross_profit = 0.0_f64;
    let mut gross_loss = 0.0_f64;
    let mut total_pnl = 0.0_f64;
    for i in 0..n {
        let p = pnl_arr[i];
        total_pnl += p;
        if p > 0.0 {
            gross_profit += p;
        } else if p < 0.0 {
            gross_loss -= p; // Make positive
        }
    }
    let pf = if gross_loss == 0.0 {
        if gross_profit > 0.0 { 10.0 } else { 0.0 }
    } else {
        gross_profit / gross_loss
    };
    metrics_row[M_PROFIT_FACTOR] = pf;

    // Mean and standard deviations
    let mean = total_pnl / n as f64;
    let mut var_sum = 0.0_f64;
    let mut down_sq_sum = 0.0_f64;
    let mut down_count = 0usize;
    for i in 0..n {
        let diff = pnl_arr[i] - mean;
        var_sum += diff * diff;
        if pnl_arr[i] < 0.0 {
            down_sq_sum += pnl_arr[i] * pnl_arr[i];
            down_count += 1;
        }
    }

    let std = if n > 1 {
        (var_sum / (n - 1) as f64).sqrt()
    } else {
        0.0
    };

    // Annualization factor
    let ann_factor = if n_bars > 0 && bars_per_year > 0.0 {
        n as f64 * bars_per_year / n_bars as f64
    } else {
        (n as f64).min(252.0)
    };

    // Sharpe (annualized)
    metrics_row[M_SHARPE] = if std > 0.0 {
        (mean / std) * ann_factor.sqrt()
    } else {
        0.0
    };

    // Sortino (annualized)
    if down_count > 0 {
        let downside_std = (down_sq_sum / down_count as f64).sqrt();
        metrics_row[M_SORTINO] = if downside_std > 0.0 {
            (mean / downside_std) * ann_factor.sqrt()
        } else {
            0.0
        };
    } else {
        metrics_row[M_SORTINO] = if mean > 0.0 { 10.0 } else { 0.0 };
    }

    // Equity curve for MaxDD, R², Ulcer
    let mut equity_peak = 0.0_f64;
    let mut max_dd = 0.0_f64;
    let mut equity = 0.0_f64;
    let mut base_val = 0.0_f64;
    let mut sum_sq_dd = 0.0_f64;

    // For R²
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_xy = 0.0_f64;
    let mut sum_xx = 0.0_f64;

    for i in 0..n {
        equity += pnl_arr[i];
        if equity > equity_peak {
            equity_peak = equity;
        }
        let dd = equity_peak - equity;
        if dd > max_dd {
            max_dd = dd;
        }

        // Track peak for Ulcer base
        if equity.abs() > base_val {
            base_val = equity.abs();
        }
        if equity_peak > base_val {
            base_val = equity_peak;
        }

        // R² accumulators
        let x = i as f64;
        sum_x += x;
        sum_y += equity;
        sum_xy += x * equity;
        sum_xx += x * x;

        // Ulcer: percentage drawdown squared
        let pct_dd = if base_val > 0.0 {
            (dd / base_val) * 100.0
        } else {
            0.0
        };
        sum_sq_dd += pct_dd * pct_dd;
    }

    // Max DD %
    if base_val <= 0.0 {
        base_val = 1.0;
    }
    metrics_row[M_MAX_DD_PCT] = (max_dd / base_val) * 100.0;

    // Return %
    metrics_row[M_RETURN_PCT] = if avg_sl_pips > 0.0 {
        (total_pnl / avg_sl_pips) * 100.0
    } else {
        0.0
    };

    // R²
    if n >= 2 {
        let x_mean = sum_x / n as f64;
        let y_mean = sum_y / n as f64;
        let ss_xy = sum_xy - n as f64 * x_mean * y_mean;
        let ss_xx = sum_xx - n as f64 * x_mean * x_mean;
        if ss_xx > 0.0 {
            let slope = ss_xy / ss_xx;
            let intercept = y_mean - slope * x_mean;
            let mut ss_res = 0.0_f64;
            let mut ss_tot = 0.0_f64;
            let mut eq2 = 0.0_f64;
            for i in 0..n {
                eq2 += pnl_arr[i];
                let y_pred = slope * i as f64 + intercept;
                ss_res += (eq2 - y_pred).powi(2);
                ss_tot += (eq2 - y_mean).powi(2);
            }
            if ss_tot > 0.0 {
                let rsq = 1.0 - ss_res / ss_tot;
                metrics_row[M_R_SQUARED] = rsq.max(0.0);
            }
        }
    }

    // Ulcer Index
    metrics_row[M_ULCER] = (sum_sq_dd / n as f64).sqrt();

    // Quality Score
    let so = metrics_row[M_SORTINO];
    if so > 0.0 {
        let r2 = metrics_row[M_R_SQUARED];
        let pf_c = pf.min(5.0);
        let trades_f = (n as f64).min(200.0).sqrt();
        let ret = metrics_row[M_RETURN_PCT];
        let ret_clamped = ret.min(200.0).max(0.0);
        let ret_f = 1.0 + ret_clamped / 100.0;
        let ulc = metrics_row[M_ULCER];
        let dd_pct = metrics_row[M_MAX_DD_PCT];

        let denom = ulc + dd_pct / 2.0 + 5.0;
        if denom > 0.0 {
            metrics_row[M_QUALITY] = (so * r2 * pf_c * trades_f * ret_f) / denom;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_trades() {
        let pnl = vec![];
        let mut metrics = vec![0.0; NUM_METRICS];
        compute_metrics_inline(&pnl, 0, 30.0, 1000, 6048.0, &mut metrics);
        assert_eq!(metrics[M_TRADES], 0.0);
        assert_eq!(metrics[M_WIN_RATE], 0.0);
    }

    #[test]
    fn test_all_winners() {
        let pnl = vec![10.0, 20.0, 15.0];
        let mut metrics = vec![0.0; NUM_METRICS];
        compute_metrics_inline(&pnl, 3, 20.0, 1000, 6048.0, &mut metrics);
        assert_eq!(metrics[M_TRADES], 3.0);
        assert_eq!(metrics[M_WIN_RATE], 1.0);
        assert_eq!(metrics[M_PROFIT_FACTOR], 10.0); // no losses → capped at 10
        assert!(metrics[M_MAX_DD_PCT] == 0.0);
    }

    #[test]
    fn test_mixed_trades() {
        let pnl = vec![10.0, -5.0, 20.0, -3.0];
        let mut metrics = vec![0.0; NUM_METRICS];
        compute_metrics_inline(&pnl, 4, 15.0, 1000, 6048.0, &mut metrics);
        assert_eq!(metrics[M_TRADES], 4.0);
        assert_eq!(metrics[M_WIN_RATE], 0.5);
        // PF = 30 / 8 = 3.75
        assert!((metrics[M_PROFIT_FACTOR] - 3.75).abs() < 1e-10);
    }
}
