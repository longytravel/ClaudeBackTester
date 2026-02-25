"""Benchmark: Rust backend throughput.

Measures Rust evals/sec for EXEC_BASIC and EXEC_FULL on H1 and M15 data,
comparing against documented Numba numbers from CLAUDE.md.

Usage:
    uv run python scripts/rust_vs_numba_benchmark.py
"""

import gc
import os
import time

import numpy as np
import pandas as pd
from pathlib import Path

os.environ["BACKTESTER_BACKEND"] = "rust"

from backtester.core.dtypes import EXEC_BASIC, EXEC_FULL
from backtester.core.engine import BacktestEngine
from backtester.core.encoding import indices_to_values
from backtester.core.rust_loop import get_backend_name
from backtester.data.timeframes import build_h1_to_m1_mapping
from backtester.optimizer.sampler import SobolSampler
from backtester.strategies import registry as strat_reg

DATA_DIR = Path("G:/My Drive/BackTestData")
PAIR = "EUR/USD"
PIP_VALUE = 0.0001

# Documented Numba throughput from CLAUDE.md (EUR/USD H1, 96K bars, identity sub-bars)
NUMBA_BASIC_H1 = 26500  # midpoint of 18K-35K range
NUMBA_FULL_H1 = 7000    # midpoint of 4K-10K range


def main():
    print("=" * 70)
    print("RUST BACKEND BENCHMARK")
    print("=" * 70)
    print(f"Backend: {get_backend_name()}")
    print(f"Pair: {PAIR}")
    print()

    pair_file = PAIR.replace("/", "_")
    results = []

    for timeframe in ["H1", "M15"]:
        print(f"\n--- {timeframe} ---")
        df = pd.read_parquet(DATA_DIR / f"{pair_file}_{timeframe}.parquet")
        data = {k: df[k].to_numpy(dtype=np.float64) for k in ["open", "high", "low", "close", "volume", "spread"]}
        data["bar_hour"] = df.index.hour.to_numpy(dtype=np.int64)
        data["bar_day_of_week"] = df.index.dayofweek.to_numpy(dtype=np.int64)

        strategy = strat_reg.create("ema_crossover")

        # Test without M1 first
        engine = BacktestEngine(
            strategy,
            data["open"], data["high"], data["low"], data["close"],
            data["volume"], data["spread"],
            pip_value=PIP_VALUE, slippage_pips=0.5,
            commission_pips=0.7, max_spread_pips=3.0,
            bar_hour=data["bar_hour"], bar_day_of_week=data["bar_day_of_week"],
        )
        print(f"  Engine: {engine.n_bars:,} bars, {engine.n_signals:,} signals")

        sampler = SobolSampler(engine.encoding, seed=42)
        n_trials = 1000 if timeframe == "H1" else 500

        for exec_mode in [EXEC_BASIC, EXEC_FULL]:
            mode_name = "FULL" if exec_mode == EXEC_FULL else "BASIC"
            idx = sampler.sample(n_trials)
            pm = indices_to_values(engine.encoding, idx)

            # Warmup
            engine.evaluate_batch(pm, exec_mode=exec_mode)

            # Timed runs
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                engine.evaluate_batch(pm, exec_mode=exec_mode)
                times.append(time.perf_counter() - t0)

            mean_t = np.mean(times)
            eps = n_trials / mean_t
            results.append({
                "tf": timeframe, "mode": mode_name, "m1": False,
                "n_bars": engine.n_bars, "n_sigs": engine.n_signals,
                "n_trials": n_trials, "eps": eps, "time": mean_t,
            })
            print(f"  {mode_name:>5s}: {eps:>10,.0f} evals/sec ({mean_t:.3f}s)")

        del engine
        gc.collect()

        # Test with M1 if M15
        if timeframe == "M15":
            m1_path = DATA_DIR / f"{pair_file}_M1.parquet"
            if m1_path.exists():
                m1_df = pd.read_parquet(m1_path)
                parent_ts = df.index.astype(np.int64).to_numpy()
                m1_ts = m1_df.index.astype(np.int64).to_numpy()
                start_idx, end_idx = build_h1_to_m1_mapping(parent_ts, m1_ts)

                engine_m1 = BacktestEngine(
                    strategy,
                    data["open"], data["high"], data["low"], data["close"],
                    data["volume"], data["spread"],
                    pip_value=PIP_VALUE, slippage_pips=0.5,
                    commission_pips=0.7, max_spread_pips=3.0,
                    bar_hour=data["bar_hour"], bar_day_of_week=data["bar_day_of_week"],
                    m1_high=m1_df["high"].to_numpy(dtype=np.float64),
                    m1_low=m1_df["low"].to_numpy(dtype=np.float64),
                    m1_close=m1_df["close"].to_numpy(dtype=np.float64),
                    m1_spread=m1_df["spread"].to_numpy(dtype=np.float64),
                    h1_to_m1_start=start_idx, h1_to_m1_end=end_idx,
                )
                print(f"\n  Engine+M1: {engine_m1.n_bars:,} bars + {len(m1_df):,} M1 bars")

                for exec_mode in [EXEC_BASIC, EXEC_FULL]:
                    mode_name = "FULL" if exec_mode == EXEC_FULL else "BASIC"
                    idx = sampler.sample(n_trials)
                    pm = indices_to_values(engine_m1.encoding, idx)

                    engine_m1.evaluate_batch(pm, exec_mode=exec_mode)  # warmup
                    times = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        engine_m1.evaluate_batch(pm, exec_mode=exec_mode)
                        times.append(time.perf_counter() - t0)

                    mean_t = np.mean(times)
                    eps = n_trials / mean_t
                    results.append({
                        "tf": "M15+M1", "mode": mode_name, "m1": True,
                        "n_bars": engine_m1.n_bars, "n_sigs": engine_m1.n_signals,
                        "n_trials": n_trials, "eps": eps, "time": mean_t,
                    })
                    print(f"  {mode_name:>5s}: {eps:>10,.0f} evals/sec ({mean_t:.3f}s)")

                del engine_m1, m1_df
                gc.collect()

        del df
        gc.collect()

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"{'Config':>12s} | {'Bars':>9s} | {'Sigs':>9s} | {'Rust e/s':>10s} | {'Numba e/s':>10s} | {'vs Numba':>8s}")
    print(f"{'-'*12:>12s} | {'-'*9:>9s} | {'-'*9:>9s} | {'-'*10:>10s} | {'-'*10:>10s} | {'-'*8:>8s}")

    for r in results:
        config = f"{r['tf']} {r['mode']}"
        if r["tf"] == "H1":
            numba_ref = NUMBA_BASIC_H1 if r["mode"] == "BASIC" else NUMBA_FULL_H1
            ratio = r["eps"] / numba_ref
            numba_str = f"{numba_ref:>10,d}"
            ratio_str = f"{ratio:>7.2f}x"
        else:
            numba_str = f"{'N/A':>10s}"
            ratio_str = f"{'—':>8s}"

        print(f"{config:>12s} | {r['n_bars']:>9,d} | {r['n_sigs']:>9,d} | "
              f"{r['eps']:>10,.0f} | {numba_str} | {ratio_str}")

    print(f"{'=' * 80}")
    print(f"\nNumba baseline (CLAUDE.md, EUR/USD H1, ~96K bars):")
    print(f"  BASIC: 18K-35K evals/sec (midpoint: {NUMBA_BASIC_H1:,})")
    print(f"  FULL:  4K-10K evals/sec (midpoint: {NUMBA_FULL_H1:,})")


if __name__ == "__main__":
    main()
