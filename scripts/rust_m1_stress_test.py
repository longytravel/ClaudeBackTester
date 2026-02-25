"""Rust backend M1 stress test.

Loads EUR/USD M15 with M1 sub-bars (~384K M15 bars + 5.7M M1 bars),
runs multiple batches through the Rust backend, monitors memory usage,
and verifies no segfaults.

Usage:
    uv run python scripts/rust_m1_stress_test.py
"""

import gc
import os
import time

import numpy as np
import pandas as pd
import psutil

# Force Rust backend — error if not available
os.environ["BACKTESTER_BACKEND"] = "rust"

from pathlib import Path

from backtester.core.dtypes import EXEC_BASIC, EXEC_FULL
from backtester.core.engine import BacktestEngine
from backtester.core.rust_loop import get_backend_name
from backtester.data.timeframes import build_h1_to_m1_mapping
from backtester.strategies import registry as strat_reg

DATA_DIR = Path("G:/My Drive/BackTestData")
PAIR = "EUR/USD"
TIMEFRAME = "M15"
PIP_VALUE = 0.0001
SLIPPAGE_PIPS = 0.5
COMMISSION_PIPS = 0.7
MAX_SPREAD_PIPS = 3.0

N_BATCHES = 20         # Number of consecutive batch evaluations
TRIALS_PER_BATCH = 500  # Trials per batch
STRATEGY = "ema_crossover"


def get_memory_mb():
    """Get current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def main():
    print("=" * 70)
    print("RUST BACKEND M1 STRESS TEST")
    print("=" * 70)
    print(f"Backend: {get_backend_name()}")
    print(f"Pair: {PAIR}, Timeframe: {TIMEFRAME}")
    print(f"Batches: {N_BATCHES} x {TRIALS_PER_BATCH} trials")
    print()

    # --- Load timeframe data ---
    pair_file = PAIR.replace("/", "_")
    tf_path = DATA_DIR / f"{pair_file}_{TIMEFRAME}.parquet"
    print(f"Loading {TIMEFRAME} data from {tf_path}...")
    df = pd.read_parquet(tf_path)
    print(f"  {TIMEFRAME} bars: {len(df):,}")

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

    # --- Load M1 sub-bar data ---
    m1_path = DATA_DIR / f"{pair_file}_M1.parquet"
    print(f"Loading M1 data from {m1_path}...")
    m1_df = pd.read_parquet(m1_path)
    print(f"  M1 bars: {len(m1_df):,}")

    # Build parent→M1 mapping
    parent_ts = df.index.astype(np.int64).to_numpy()
    m1_ts = m1_df.index.astype(np.int64).to_numpy()
    start_idx, end_idx = build_h1_to_m1_mapping(parent_ts, m1_ts)

    total_mapped = int(np.sum(end_idx - start_idx))
    coverage = total_mapped / len(m1_df) * 100
    print(f"  M1 mapping coverage: {total_mapped:,}/{len(m1_df):,} ({coverage:.1f}%)")

    m1_high = m1_df["high"].to_numpy(dtype=np.float64)
    m1_low = m1_df["low"].to_numpy(dtype=np.float64)
    m1_close = m1_df["close"].to_numpy(dtype=np.float64)
    m1_spread = m1_df["spread"].to_numpy(dtype=np.float64)

    # Free DataFrame memory
    del m1_df
    gc.collect()

    mem_after_load = get_memory_mb()
    print(f"\nMemory after data load: {mem_after_load:.0f} MB")

    # --- Create engine ---
    print(f"\nCreating BacktestEngine (strategy={STRATEGY})...")
    strategy = strat_reg.create(STRATEGY)
    engine = BacktestEngine(
        strategy,
        data["open"], data["high"], data["low"], data["close"],
        data["volume"], data["spread"],
        pip_value=PIP_VALUE,
        slippage_pips=SLIPPAGE_PIPS,
        commission_pips=COMMISSION_PIPS,
        max_spread_pips=MAX_SPREAD_PIPS,
        bar_hour=data["bar_hour"],
        bar_day_of_week=data["bar_day_of_week"],
        m1_high=m1_high,
        m1_low=m1_low,
        m1_close=m1_close,
        m1_spread=m1_spread,
        h1_to_m1_start=start_idx,
        h1_to_m1_end=end_idx,
    )
    print(f"  Signals: {engine.n_signals:,}")
    print(f"  Bars: {engine.n_bars:,}")
    print(f"  Encoding params: {engine.encoding.num_params}")

    mem_after_engine = get_memory_mb()
    print(f"  Memory after engine creation: {mem_after_engine:.0f} MB")

    # --- Generate valid parameter matrices ---
    from backtester.core.encoding import indices_to_values
    from backtester.optimizer.sampler import SobolSampler
    sampler = SobolSampler(engine.encoding, seed=42)

    # --- Run stress test ---
    print(f"\n{'='*70}")
    print(f"Running {N_BATCHES} batches x {TRIALS_PER_BATCH} trials...")
    print(f"{'Batch':>5} | {'Mode':>6} | {'Time':>8} | {'Evals/s':>10} | {'Mem MB':>8} | {'Delta':>8}")
    print(f"{'-'*5:>5} | {'-'*6:>6} | {'-'*8:>8} | {'-'*10:>10} | {'-'*8:>8} | {'-'*8:>8}")

    mem_baseline = get_memory_mb()
    total_evals = 0
    total_time = 0.0

    for batch_idx in range(N_BATCHES):
        # Alternate EXEC_BASIC and EXEC_FULL
        exec_mode = EXEC_FULL if batch_idx % 2 == 0 else EXEC_BASIC
        mode_name = "FULL" if exec_mode == EXEC_FULL else "BASIC"

        # Generate fresh parameter matrix each batch (index→value conversion)
        index_matrix = sampler.sample(TRIALS_PER_BATCH)
        param_matrix = indices_to_values(engine.encoding, index_matrix)

        t0 = time.perf_counter()
        metrics = engine.evaluate_batch(param_matrix, exec_mode=exec_mode)
        elapsed = time.perf_counter() - t0

        evals_per_sec = TRIALS_PER_BATCH / elapsed
        mem_now = get_memory_mb()
        mem_delta = mem_now - mem_baseline

        total_evals += TRIALS_PER_BATCH
        total_time += elapsed

        # Sanity check: metrics should not be all zeros (some trials should produce trades)
        n_with_trades = np.sum(metrics[:, 0] > 0)

        print(f"{batch_idx+1:>5} | {mode_name:>6} | {elapsed:>7.2f}s | {evals_per_sec:>10,.0f} | {mem_now:>7.0f} | {mem_delta:>+7.0f}")

        if n_with_trades == 0:
            print(f"  WARNING: Zero trials produced trades in batch {batch_idx+1}")

        # Force GC between batches to simulate real pipeline behavior
        del metrics, param_matrix
        gc.collect()

    # --- Summary ---
    mem_final = get_memory_mb()
    print(f"\n{'='*70}")
    print("STRESS TEST COMPLETE")
    print(f"{'='*70}")
    print(f"Total evaluations: {total_evals:,}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average throughput: {total_evals / total_time:,.0f} evals/sec")
    print(f"Memory baseline: {mem_baseline:.0f} MB")
    print(f"Memory final: {mem_final:.0f} MB")
    print(f"Memory growth: {mem_final - mem_baseline:+.0f} MB")
    print(f"Backend: {get_backend_name()}")

    # Memory growth > 500MB would indicate a leak
    growth = mem_final - mem_baseline
    if growth > 500:
        print(f"\nWARNING: Memory grew by {growth:.0f} MB — possible leak!")
    else:
        print(f"\nMemory stable (growth < 500 MB threshold)")

    print("\nNo segfaults detected!")


if __name__ == "__main__":
    main()
