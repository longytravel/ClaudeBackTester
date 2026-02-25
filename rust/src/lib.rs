mod constants;
mod filter;
mod metrics;
mod sl_tp;
mod trade_basic;
mod trade_full;

use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

use constants::*;
use filter::signal_passes_time_filter;
use metrics::compute_metrics_inline;
use sl_tp::compute_sl_tp;
use trade_basic::simulate_trade_basic;
use trade_full::simulate_trade_full;

/// Evaluate N parameter sets in parallel.
///
/// This is the Rust replacement for jit_loop.batch_evaluate().
/// Signature matches the Numba version exactly for drop-in replacement.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn batch_evaluate<'py>(
    py: Python<'py>,
    // Price data
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
    spread: PyReadonlyArray1<'py, f64>,
    pip_value: f64,
    slippage_pips: f64,
    // Signal data
    sig_bar_index: PyReadonlyArray1<'py, i64>,
    sig_direction: PyReadonlyArray1<'py, i64>,
    sig_entry_price: PyReadonlyArray1<'py, f64>,
    sig_hour: PyReadonlyArray1<'py, i64>,
    sig_day: PyReadonlyArray1<'py, i64>,
    sig_atr_pips: PyReadonlyArray1<'py, f64>,
    sig_swing_sl: PyReadonlyArray1<'py, f64>,
    sig_filter_value: PyReadonlyArray1<'py, f64>,
    sig_variant: PyReadonlyArray1<'py, i64>,
    // Parameter matrix
    param_matrix: PyReadonlyArray2<'py, f64>,
    param_layout: PyReadonlyArray1<'py, i64>,
    // Execution mode
    exec_mode: i64,
    // Output (mutable)
    metrics_out: &Bound<'py, PyArray2<f64>>,
    // Working memory
    max_trades: i64,
    bars_per_year: f64,
    // Execution costs
    commission_pips: f64,
    max_spread_pips: f64,
    // Sub-bar arrays
    sub_high: PyReadonlyArray1<'py, f64>,
    sub_low: PyReadonlyArray1<'py, f64>,
    sub_close: PyReadonlyArray1<'py, f64>,
    sub_spread: PyReadonlyArray1<'py, f64>,
    h1_to_sub_start: PyReadonlyArray1<'py, i64>,
    h1_to_sub_end: PyReadonlyArray1<'py, i64>,
    pnl_buffers: &Bound<'py, PyArray2<f64>>,
) -> PyResult<()> {
    // Get raw slices from numpy arrays (zero-copy)
    let high_s = high.as_slice()?;
    let low_s = low.as_slice()?;
    let close_s = close.as_slice()?;
    let spread_s = spread.as_slice()?;

    let sig_bar_index_s = sig_bar_index.as_slice()?;
    let sig_direction_s = sig_direction.as_slice()?;
    let sig_entry_price_s = sig_entry_price.as_slice()?;
    let sig_hour_s = sig_hour.as_slice()?;
    let sig_day_s = sig_day.as_slice()?;
    let sig_atr_pips_s = sig_atr_pips.as_slice()?;
    let sig_swing_sl_s = sig_swing_sl.as_slice()?;
    let sig_filter_value_s = sig_filter_value.as_slice()?;
    let sig_variant_s = sig_variant.as_slice()?;

    let param_matrix_s = param_matrix.as_slice()?;
    let param_layout_s = param_layout.as_slice()?;

    let sub_high_s = sub_high.as_slice()?;
    let sub_low_s = sub_low.as_slice()?;
    let sub_close_s = sub_close.as_slice()?;
    let sub_spread_s = sub_spread.as_slice()?;
    let h1_to_sub_start_s = h1_to_sub_start.as_slice()?;
    let h1_to_sub_end_s = h1_to_sub_end.as_slice()?;

    let n_trials = param_matrix.shape()[0];
    let n_params = param_matrix.shape()[1];
    let n_signals = sig_bar_index_s.len();
    let n_bars = high_s.len();
    let max_trades_usize = max_trades as usize;

    // Get mutable access to output arrays
    // SAFETY: We release the GIL below and use Rayon for parallelism.
    // Each trial writes to non-overlapping slices of metrics_out and pnl_buffers.
    let metrics_out_ptr = unsafe { metrics_out.as_slice_mut()? };
    let pnl_buffers_ptr = unsafe { pnl_buffers.as_slice_mut()? };

    // Release the GIL and run in parallel
    py.allow_threads(|| {
        // Split output into per-trial chunks for non-overlapping writes
        let metrics_chunks: Vec<&mut [f64]> = metrics_out_ptr
            .chunks_mut(NUM_METRICS)
            .collect();
        let pnl_chunks: Vec<&mut [f64]> = pnl_buffers_ptr
            .chunks_mut(max_trades_usize)
            .collect();

        // Zip the mutable slices and iterate in parallel
        metrics_chunks
            .into_iter()
            .zip(pnl_chunks)
            .enumerate()
            .collect::<Vec<_>>()
            .into_par_iter()
            .for_each(|(trial, (metrics_row, pnl_buffer))| {
                // Get params for this trial (row-major indexing)
                let params_offset = trial * n_params;
                let params = &param_matrix_s[params_offset..params_offset + n_params];

                // Extract standard params via layout
                let sl_mode = params[param_layout_s[PL_SL_MODE] as usize] as i64;
                let sl_fixed_pips = params[param_layout_s[PL_SL_FIXED_PIPS] as usize];
                let sl_atr_mult = params[param_layout_s[PL_SL_ATR_MULT] as usize];
                let tp_mode = params[param_layout_s[PL_TP_MODE] as usize] as i64;
                let tp_rr_ratio = params[param_layout_s[PL_TP_RR_RATIO] as usize];
                let tp_atr_mult = params[param_layout_s[PL_TP_ATR_MULT] as usize];
                let tp_fixed_pips_val = params[param_layout_s[PL_TP_FIXED_PIPS] as usize];
                let hours_start = params[param_layout_s[PL_HOURS_START] as usize] as i64;
                let hours_end = params[param_layout_s[PL_HOURS_END] as usize] as i64;
                let days_bitmask = params[param_layout_s[PL_DAYS_BITMASK] as usize] as i64;

                // Management params
                let trailing_mode = params[param_layout_s[PL_TRAILING_MODE] as usize] as i64;
                let trail_activate = params[param_layout_s[PL_TRAIL_ACTIVATE] as usize];
                let trail_distance = params[param_layout_s[PL_TRAIL_DISTANCE] as usize];
                let trail_atr_m = params[param_layout_s[PL_TRAIL_ATR_MULT] as usize];
                let be_enabled = params[param_layout_s[PL_BREAKEVEN_ENABLED] as usize] as i64;
                let be_trigger = params[param_layout_s[PL_BREAKEVEN_TRIGGER] as usize];
                let be_offset = params[param_layout_s[PL_BREAKEVEN_OFFSET] as usize];
                let partial_en = params[param_layout_s[PL_PARTIAL_ENABLED] as usize] as i64;
                let partial_pct = params[param_layout_s[PL_PARTIAL_PCT] as usize];
                let partial_trig = params[param_layout_s[PL_PARTIAL_TRIGGER] as usize];
                let max_bars_val = params[param_layout_s[PL_MAX_BARS] as usize] as i64;
                let stale_en = params[param_layout_s[PL_STALE_ENABLED] as usize] as i64;
                let stale_bars_val = params[param_layout_s[PL_STALE_BARS] as usize] as i64;
                let stale_atr = params[param_layout_s[PL_STALE_ATR_THRESH] as usize];

                // Strategy-specific signal filter params
                let variant_col = param_layout_s[PL_SIGNAL_VARIANT];
                let trial_variant = if variant_col >= 0 {
                    params[variant_col as usize] as i64
                } else {
                    -1
                };
                let bfm_col = param_layout_s[PL_BUY_FILTER_MAX];
                let buy_filter_max = if bfm_col >= 0 {
                    params[bfm_col as usize]
                } else {
                    -1.0
                };
                let sfm_col = param_layout_s[PL_SELL_FILTER_MIN];
                let sell_filter_min = if sfm_col >= 0 {
                    params[sfm_col as usize]
                } else {
                    -1.0
                };

                let mut trade_count = 0usize;
                let mut total_sl_pips = 0.0_f64;

                for si in 0..n_signals {
                    // Signal variant filter
                    if trial_variant >= 0 && sig_variant_s[si] >= 0 {
                        if sig_variant_s[si] != trial_variant {
                            continue;
                        }
                    }

                    // Strategy-specific value filter
                    let direction = sig_direction_s[si];
                    if buy_filter_max >= 0.0 && direction == DIR_BUY {
                        if sig_filter_value_s[si] != buy_filter_max {
                            continue;
                        }
                    }
                    if sell_filter_min >= 0.0 && direction == DIR_SELL {
                        if sig_filter_value_s[si] != sell_filter_min {
                            continue;
                        }
                    }

                    // Time filter
                    if !signal_passes_time_filter(
                        sig_hour_s[si],
                        sig_day_s[si],
                        hours_start,
                        hours_end,
                        days_bitmask,
                    ) {
                        continue;
                    }

                    let bar_idx = sig_bar_index_s[si] as usize;
                    let entry_p = sig_entry_price_s[si];
                    let atr_p = sig_atr_pips_s[si];
                    let swing_sl = sig_swing_sl_s[si];

                    // Max spread filter
                    if max_spread_pips > 0.0 {
                        let spread_at_signal = spread_s[bar_idx] / pip_value;
                        if spread_at_signal.is_nan() || spread_at_signal > max_spread_pips {
                            continue;
                        }
                    }

                    let sl_tp = compute_sl_tp(
                        direction,
                        entry_p,
                        atr_p,
                        pip_value,
                        sl_mode,
                        sl_fixed_pips,
                        sl_atr_mult,
                        swing_sl,
                        tp_mode,
                        tp_rr_ratio,
                        tp_atr_mult,
                        tp_fixed_pips_val,
                    );

                    total_sl_pips += sl_tp.sl_pips;

                    // Simulate trade
                    let result = if exec_mode == EXEC_BASIC {
                        simulate_trade_basic(
                            direction,
                            bar_idx,
                            entry_p,
                            sl_tp.sl_price,
                            sl_tp.tp_price,
                            high_s,
                            low_s,
                            close_s,
                            spread_s,
                            pip_value,
                            slippage_pips,
                            n_bars,
                            commission_pips,
                            sub_high_s,
                            sub_low_s,
                            sub_close_s,
                            sub_spread_s,
                            h1_to_sub_start_s,
                            h1_to_sub_end_s,
                        )
                    } else {
                        simulate_trade_full(
                            direction,
                            bar_idx,
                            entry_p,
                            sl_tp.sl_price,
                            sl_tp.tp_price,
                            atr_p,
                            high_s,
                            low_s,
                            close_s,
                            spread_s,
                            pip_value,
                            slippage_pips,
                            n_bars,
                            trailing_mode,
                            trail_activate,
                            trail_distance,
                            trail_atr_m,
                            be_enabled,
                            be_trigger,
                            be_offset,
                            partial_en,
                            partial_pct,
                            partial_trig,
                            max_bars_val,
                            stale_en,
                            stale_bars_val,
                            stale_atr,
                            commission_pips,
                            sub_high_s,
                            sub_low_s,
                            sub_close_s,
                            sub_spread_s,
                            h1_to_sub_start_s,
                            h1_to_sub_end_s,
                        )
                    };

                    if trade_count < max_trades_usize {
                        pnl_buffer[trade_count] = result.pnl_pips;
                        trade_count += 1;
                    }
                }

                // Compute metrics
                let avg_sl = if trade_count > 0 {
                    total_sl_pips / trade_count as f64
                } else {
                    30.0
                };
                compute_metrics_inline(
                    pnl_buffer,
                    trade_count,
                    avg_sl,
                    n_bars,
                    bars_per_year,
                    metrics_row,
                );
            });
    });

    Ok(())
}

/// Python module
#[pymodule]
fn backtester_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(batch_evaluate, m)?)?;

    // Export constants for Python-side verification
    m.add("NUM_PL", NUM_PL)?;
    m.add("NUM_METRICS", NUM_METRICS)?;
    m.add("EXEC_BASIC", EXEC_BASIC)?;
    m.add("EXEC_FULL", EXEC_FULL)?;

    // Export PL_* constants
    m.add("PL_SL_MODE", PL_SL_MODE)?;
    m.add("PL_SL_FIXED_PIPS", PL_SL_FIXED_PIPS)?;
    m.add("PL_SL_ATR_MULT", PL_SL_ATR_MULT)?;
    m.add("PL_TP_MODE", PL_TP_MODE)?;
    m.add("PL_TP_RR_RATIO", PL_TP_RR_RATIO)?;
    m.add("PL_TP_ATR_MULT", PL_TP_ATR_MULT)?;
    m.add("PL_TP_FIXED_PIPS", PL_TP_FIXED_PIPS)?;
    m.add("PL_HOURS_START", PL_HOURS_START)?;
    m.add("PL_HOURS_END", PL_HOURS_END)?;
    m.add("PL_DAYS_BITMASK", PL_DAYS_BITMASK)?;
    m.add("PL_TRAILING_MODE", PL_TRAILING_MODE)?;
    m.add("PL_TRAIL_ACTIVATE", PL_TRAIL_ACTIVATE)?;
    m.add("PL_TRAIL_DISTANCE", PL_TRAIL_DISTANCE)?;
    m.add("PL_TRAIL_ATR_MULT", PL_TRAIL_ATR_MULT)?;
    m.add("PL_BREAKEVEN_ENABLED", PL_BREAKEVEN_ENABLED)?;
    m.add("PL_BREAKEVEN_TRIGGER", PL_BREAKEVEN_TRIGGER)?;
    m.add("PL_BREAKEVEN_OFFSET", PL_BREAKEVEN_OFFSET)?;
    m.add("PL_PARTIAL_ENABLED", PL_PARTIAL_ENABLED)?;
    m.add("PL_PARTIAL_PCT", PL_PARTIAL_PCT)?;
    m.add("PL_PARTIAL_TRIGGER", PL_PARTIAL_TRIGGER)?;
    m.add("PL_MAX_BARS", PL_MAX_BARS)?;
    m.add("PL_STALE_ENABLED", PL_STALE_ENABLED)?;
    m.add("PL_STALE_BARS", PL_STALE_BARS)?;
    m.add("PL_STALE_ATR_THRESH", PL_STALE_ATR_THRESH)?;
    m.add("PL_SIGNAL_VARIANT", PL_SIGNAL_VARIANT)?;
    m.add("PL_BUY_FILTER_MAX", PL_BUY_FILTER_MAX)?;
    m.add("PL_SELL_FILTER_MIN", PL_SELL_FILTER_MIN)?;

    Ok(())
}
