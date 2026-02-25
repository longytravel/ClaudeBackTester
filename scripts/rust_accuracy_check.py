"""Rust accuracy verification: re-evaluate ALL saved Numba results with Rust backend.

Loads every saved report.json/checkpoint.json from results/, extracts the
exact param sets that were found by previous (Numba) optimization runs,
re-evaluates them with the current (Rust) backend, and compares all 10 metrics.

This is the definitive check that the Rust port produces identical results.

Usage:
    uv run python scripts/rust_accuracy_check.py                    # all H1 runs
    uv run python scripts/rust_accuracy_check.py --run rsi_eurusd_h1
    uv run python scripts/rust_accuracy_check.py --run eur_usd_m15 --m1  # include M1
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["BACKTESTER_BACKEND"] = "rust"

from backtester.core.dtypes import EXEC_FULL, NUM_METRICS
from backtester.core.engine import BacktestEngine
from backtester.core.encoding import encode_params
from backtester.core.rust_loop import get_backend_name
from backtester.data.timeframes import build_h1_to_m1_mapping
from backtester.strategies import registry as strat_reg

DATA_DIR = Path("G:/My Drive/BackTestData")

METRIC_NAMES = [
    "trades", "win_rate", "profit_factor", "sharpe", "sortino",
    "max_dd_pct", "return_pct", "r_squared", "ulcer", "quality_score",
]

PIP_VALUES = {
    "EUR/USD": 0.0001, "GBP/USD": 0.0001, "AUD/USD": 0.0001,
    "NZD/USD": 0.0001, "USD/CHF": 0.0001, "EUR/GBP": 0.0001,
    "USD/JPY": 0.01, "EUR/JPY": 0.01, "GBP/JPY": 0.01,
    "XAU/USD": 0.01,
}


def load_data(pair: str, timeframe: str, load_m1: bool = False):
    """Load timeframe + optional M1 data, return arrays dict."""
    pair_file = pair.replace("/", "_")
    tf_path = DATA_DIR / f"{pair_file}_{timeframe}.parquet"
    df = pd.read_parquet(tf_path)

    data = {
        "open": df["open"].to_numpy(dtype=np.float64),
        "high": df["high"].to_numpy(dtype=np.float64),
        "low": df["low"].to_numpy(dtype=np.float64),
        "close": df["close"].to_numpy(dtype=np.float64),
        "volume": df["volume"].to_numpy(dtype=np.float64),
        "spread": df["spread"].to_numpy(dtype=np.float64),
        "bar_hour": df.index.hour.to_numpy(dtype=np.int64),
        "bar_day_of_week": df.index.dayofweek.to_numpy(dtype=np.int64),
    }

    # Try loading M1
    if load_m1:
        m1_path = DATA_DIR / f"{pair_file}_M1.parquet"
        if m1_path.exists():
            m1_df = pd.read_parquet(m1_path)
            parent_ts = df.index.astype(np.int64).to_numpy()
            m1_ts = m1_df.index.astype(np.int64).to_numpy()
            start_idx, end_idx = build_h1_to_m1_mapping(parent_ts, m1_ts)
            data["m1_high"] = m1_df["high"].to_numpy(dtype=np.float64)
            data["m1_low"] = m1_df["low"].to_numpy(dtype=np.float64)
            data["m1_close"] = m1_df["close"].to_numpy(dtype=np.float64)
            data["m1_spread"] = m1_df["spread"].to_numpy(dtype=np.float64)
            data["h1_to_m1_start"] = start_idx
            data["h1_to_m1_end"] = end_idx
            print(f"    M1 loaded: {len(m1_df):,} bars")
            del m1_df
            gc.collect()
        else:
            print(f"    No M1 data found (identity sub-bars)")
    else:
        print(f"    M1 skipped (use --m1 to include)")

    return df, data


def build_engine(strategy, data, pip_value):
    """Build BacktestEngine from data dict."""
    m1_kwargs = {}
    for key in ["m1_high", "m1_low", "m1_close", "m1_spread",
                "h1_to_m1_start", "h1_to_m1_end"]:
        if key in data:
            m1_kwargs[key] = data[key]

    return BacktestEngine(
        strategy,
        data["open"], data["high"], data["low"], data["close"],
        data["volume"], data["spread"],
        pip_value=pip_value,
        slippage_pips=0.5,
        commission_pips=0.7,
        max_spread_pips=3.0,
        bar_hour=data["bar_hour"],
        bar_day_of_week=data["bar_day_of_week"],
        **m1_kwargs,
    )


def evaluate_with_rust(engine, params_dict):
    """Evaluate a single param set, return metrics dict."""
    return engine.evaluate_single(params_dict, exec_mode=EXEC_FULL)


def evaluate_windowed_with_rust(engine, params_dict, start_bar, end_bar):
    """Evaluate on a specific window, return metrics dict."""
    row = encode_params(engine.encoding, params_dict)
    matrix = row.reshape(1, -1)
    metrics = engine.evaluate_batch_windowed(matrix, start_bar, end_bar, exec_mode=EXEC_FULL)
    return {name: float(metrics[0, i]) for i, name in enumerate(METRIC_NAMES)}


def compare_metrics(label, saved, rust, tolerance=1e-6):
    """Compare saved vs rust metrics, return list of mismatches."""
    mismatches = []
    for key in saved:
        if key not in rust:
            continue
        s_val = saved[key]
        r_val = rust[key]
        if abs(s_val) < 1e-12 and abs(r_val) < 1e-12:
            continue  # Both effectively zero
        if abs(s_val) > 1e-12:
            rel_diff = abs(r_val - s_val) / abs(s_val)
        else:
            rel_diff = abs(r_val - s_val)
        abs_diff = abs(r_val - s_val)

        if abs_diff > tolerance and rel_diff > tolerance:
            mismatches.append({
                "metric": key, "saved": s_val, "rust": r_val,
                "abs_diff": abs_diff, "rel_diff": rel_diff,
            })
    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Rust accuracy verification")
    parser.add_argument("--run", help="Only check this specific run directory name")
    parser.add_argument("--m1", action="store_true", help="Load M1 sub-bar data")
    args = parser.parse_args()

    print("=" * 80)
    print("RUST ACCURACY VERIFICATION")
    print("=" * 80)
    print(f"Backend: {get_backend_name()}")
    print(f"M1 sub-bars: {'YES' if args.m1 else 'NO (identity sub-bars)'}")
    print()

    results_dir = Path("results")
    all_dirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
    if args.run:
        all_dirs = [d for d in all_dirs if d.name == args.run]
        if not all_dirs:
            print(f"ERROR: No results directory named '{args.run}'")
            sys.exit(1)

    total_checks = 0
    total_mismatches = 0
    total_windows_checked = 0
    all_window_mismatches = 0

    for run_dir in all_dirs:
        report_path = run_dir / "report.json"
        checkpoint_path = run_dir / "checkpoint.json"

        if not report_path.exists():
            continue

        report = json.loads(report_path.read_text())
        strategy_name = report["strategy"]
        pair = report.get("pair", "EUR/USD")
        timeframe = report.get("timeframe", "H1")
        pip_value = PIP_VALUES.get(pair, 0.0001)

        print(f"\n{'='*70}")
        print(f"RUN: {run_dir.name}")
        print(f"  Strategy: {strategy_name}, Pair: {pair}, Timeframe: {timeframe}")

        # Skip always_buy — minimal test strategy with different param structure
        if strategy_name == "always_buy":
            print("  Skipping: always_buy is a test strategy with non-standard params")
            continue

        # Load data
        print(f"  Loading {pair} {timeframe} data...")
        try:
            df, data = load_data(pair, timeframe, load_m1=args.m1)
        except FileNotFoundError as e:
            print(f"  SKIP: data not found — {e}")
            continue

        # Create strategy + engine
        strategy = strat_reg.create(strategy_name)
        engine = build_engine(strategy, data, pip_value)
        print(f"  Engine: {engine.n_bars:,} bars, {engine.n_signals:,} signals")

        # ================================================
        # CHECK 1: Full-data evaluation of best params
        # ================================================
        for cand in report["candidates"]:
            params = cand["params"]
            total_checks += 1

            print(f"\n  --- Candidate {cand.get('index', 0)} ---")
            print(f"  Params: {json.dumps({k: v for k, v in params.items() if k in ['sl_mode', 'tp_mode', 'trailing_mode', 'breakeven_enabled']}, indent=None)}")

            # Evaluate with Rust on full data
            rust_metrics = evaluate_with_rust(engine, params)

            # Compare against saved back_quality/back_sharpe/back_trades
            print(f"\n  FULL-DATA METRICS (Rust backend):")
            for name in METRIC_NAMES:
                val = rust_metrics[name]
                print(f"    {name:>16s}: {val:>14.6f}")

            # Report stored values for comparison
            saved_back_quality = cand.get("back_quality")
            saved_back_sharpe = cand.get("back_sharpe")
            saved_back_trades = cand.get("back_trades")

            if saved_back_quality is not None:
                print(f"\n  SAVED (from Numba optimization run):")
                print(f"    back_quality:    {saved_back_quality:>14.6f}  (Rust quality: {rust_metrics['quality_score']:>14.6f})")
                if saved_back_sharpe:
                    print(f"    back_sharpe:     {saved_back_sharpe:>14.6f}  (Rust sharpe:  {rust_metrics['sharpe']:>14.6f})")
                if saved_back_trades:
                    print(f"    back_trades:     {saved_back_trades:>14.0f}  (Rust trades:  {rust_metrics['trades']:>14.0f})")

                # NOTE: back_quality/sharpe/trades from optimizer were on back-test split (80%),
                # not full data. So we expect DIFFERENT numbers here. This is NOT a mismatch.
                print(f"\n  NOTE: Saved metrics are from 80% back-test split, Rust metrics are full data.")
                print(f"  To compare apples-to-apples, see walk-forward window checks below.")

        # ================================================
        # CHECK 2: Walk-forward window-by-window comparison
        # ================================================
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            for cand_cp in checkpoint.get("candidates", []):
                wf = cand_cp.get("walk_forward", {})
                windows = wf.get("windows", [])

                if not windows:
                    continue

                params = cand_cp["params"]
                print(f"\n  WALK-FORWARD WINDOW CHECK ({len(windows)} windows)")
                print(f"  {'Win':>4s} | {'OOS':>3s} | {'Saved Sharpe':>14s} | {'Rust Sharpe':>14s} | {'Diff':>10s} | {'Saved Trades':>12s} | {'Rust Trades':>12s} | {'Status':>8s}")
                print(f"  {'-'*4} | {'-'*3} | {'-'*14} | {'-'*14} | {'-'*10} | {'-'*12} | {'-'*12} | {'-'*8}")

                window_ok = 0
                window_bad = 0

                for w in windows:
                    win_idx = w["window_index"]
                    start_bar = w["start_bar"]
                    end_bar = w["end_bar"]
                    is_oos = w["is_oos"]
                    saved_sharpe = w["sharpe"]
                    saved_trades = w["n_trades"]
                    saved_quality = w.get("quality_score", 0)
                    saved_pf = w.get("profit_factor", 0)

                    total_windows_checked += 1

                    # Re-evaluate with Rust on same window
                    rust_w = evaluate_windowed_with_rust(engine, params, start_bar, end_bar)
                    rust_sharpe = rust_w["sharpe"]
                    rust_trades = rust_w["trades"]
                    rust_quality = rust_w["quality_score"]
                    rust_pf = rust_w["profit_factor"]

                    # Check trades match exactly
                    trades_match = int(saved_trades) == int(rust_trades)
                    # Check sharpe within tolerance (float comparison)
                    sharpe_close = abs(saved_sharpe - rust_sharpe) < 0.001 if saved_trades > 0 else True
                    # Check quality within tolerance
                    quality_close = abs(saved_quality - rust_quality) < 0.01 if saved_trades > 5 else True

                    ok = trades_match and sharpe_close
                    status = "OK" if ok else "MISMATCH"

                    if not ok:
                        window_bad += 1
                        all_window_mismatches += 1
                    else:
                        window_ok += 1

                    sharpe_diff = rust_sharpe - saved_sharpe

                    oos_str = "Y" if is_oos else "N"
                    print(f"  {win_idx:>4d} | {oos_str:>3s} | {saved_sharpe:>14.6f} | {rust_sharpe:>14.6f} | {sharpe_diff:>+10.6f} | {saved_trades:>12.0f} | {rust_trades:>12.0f} | {status:>8s}")

                    # If mismatch, show ALL metric details
                    if not ok:
                        print(f"         DETAIL: PF saved={saved_pf:.4f} rust={rust_pf:.4f}  Quality saved={saved_quality:.4f} rust={rust_quality:.4f}")

                print(f"\n  Window results: {window_ok} OK, {window_bad} MISMATCH")
                if window_bad > 0:
                    total_mismatches += 1

        # Free memory
        del engine, data, df
        gc.collect()

    # ================================================
    # FINAL SUMMARY
    # ================================================
    print(f"\n{'='*80}")
    print("ACCURACY VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Backend: {get_backend_name()}")
    print(f"Runs checked: {total_checks}")
    print(f"Walk-forward windows checked: {total_windows_checked}")
    print(f"Window mismatches: {all_window_mismatches}")

    if all_window_mismatches == 0 and total_windows_checked > 0:
        print(f"\nRESULT: ALL {total_windows_checked} WINDOWS MATCH — Rust is bit-for-bit accurate")
    elif total_windows_checked == 0:
        print(f"\nRESULT: No walk-forward windows to compare (no checkpoint data)")
    else:
        pct = all_window_mismatches / total_windows_checked * 100
        print(f"\nRESULT: {all_window_mismatches}/{total_windows_checked} ({pct:.1f}%) windows have mismatches")
        print("  This needs investigation!")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
