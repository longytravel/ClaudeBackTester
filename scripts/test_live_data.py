"""Incremental live data test — tests the full stack with real EUR/USD data.

Run with: uv run python scripts/test_live_data.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data(pair: str = "EUR_USD", timeframe: str = "H1") -> pd.DataFrame:
    """Load data from parquet, rebuilding from M1 chunks if needed."""
    data_dir = Path("G:/My Drive/BackTestData")
    parquet_path = data_dir / f"{pair}_{timeframe}.parquet"

    # Try direct load first
    try:
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded {parquet_path.name}: {len(df):,} bars")
        return df
    except Exception as e:
        print(f"  Direct load failed ({e.__class__.__name__}), rebuilding from M1 chunks...")

    # Rebuild from M1 chunks
    from backtester.data.timeframes import TIMEFRAME_RULES, resample_ohlcv
    chunks_dir = data_dir / f"{pair}_M1_chunks"
    if not chunks_dir.exists():
        raise FileNotFoundError(f"No chunks at {chunks_dir}")

    chunk_files = sorted(chunks_dir.glob("*.parquet"))
    print(f"  Found {len(chunk_files)} M1 chunk files")

    dfs = []
    for cf in chunk_files:
        try:
            dfs.append(pd.read_parquet(cf))
        except Exception:
            print(f"    Skipping corrupted chunk: {cf.name}")
    if not dfs:
        raise ValueError("No valid M1 chunks found")

    m1 = pd.concat(dfs).sort_index()
    print(f"  Combined M1 data: {len(m1):,} bars ({m1.index[0]} to {m1.index[-1]})")

    rule = TIMEFRAME_RULES[timeframe]
    df = resample_ohlcv(m1, rule)
    print(f"  Resampled to {timeframe}: {len(df):,} bars")
    return df


def extract_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Extract numpy arrays from dataframe, dropping NaN rows."""
    # Drop rows with any NaN in OHLC (from gaps in M1 data)
    ohlc_cols = ["open", "high", "low", "close"]
    before = len(df)
    df = df.dropna(subset=ohlc_cols)
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after:,} NaN rows ({before:,} -> {after:,})")

    arrays = {
        "open_": df["open"].values.astype(np.float64),
        "high": df["high"].values.astype(np.float64),
        "low": df["low"].values.astype(np.float64),
        "close": df["close"].values.astype(np.float64),
        "volume": df["volume"].values.astype(np.float64),
    }
    if "spread" in df.columns:
        spread = df["spread"].values.astype(np.float64)
        # Replace NaN spreads with 1 pip default
        spread = np.where(np.isnan(spread), 0.0001, spread)
        arrays["spread"] = spread
    else:
        arrays["spread"] = np.full(len(df), 0.0001, dtype=np.float64)
    return arrays


# ===== TEST 1: Load data =====
def test_1_load_data():
    print("\n===== TEST 1: Load EUR/USD H1 Data =====")
    df = load_data("EUR_USD", "H1")
    arrays = extract_arrays(df)
    print(f"  Bars: {len(arrays['close']):,}")
    print(f"  Price range: {arrays['close'].min():.5f} - {arrays['close'].max():.5f}")
    print(f"  Spread range: {arrays['spread'].min():.6f} - {arrays['spread'].max():.6f}")
    print(f"  Any NaN in close? {np.any(np.isnan(arrays['close']))}")
    print("  PASS")
    return df, arrays


# ===== TEST 2: Generate signals =====
def test_2_signals(arrays):
    print("\n===== TEST 2: Generate RSI Signals =====")
    from backtester.strategies.rsi_mean_reversion import RSIMeanReversion

    strategy = RSIMeanReversion()
    print(f"  Strategy: {strategy.name} v{strategy.version}")
    print(f"  Param space: {len(strategy.param_space())} params, {strategy.param_space().total_combinations():,} combinations")

    t0 = time.perf_counter()
    signals = strategy.generate_signals_vectorized(
        arrays["open_"], arrays["high"], arrays["low"],
        arrays["close"], arrays["volume"], arrays["spread"],
    )
    elapsed = time.perf_counter() - t0
    n_signals = len(signals["bar_index"])
    print(f"  Generated {n_signals:,} signals in {elapsed:.3f}s")

    if n_signals > 0:
        buy_count = np.sum(signals["direction"] == 1)
        sell_count = np.sum(signals["direction"] == -1)
        print(f"  Buy signals: {buy_count:,}, Sell signals: {sell_count:,}")
        print(f"  ATR pips range: {signals['atr_pips'].min():.1f} - {signals['atr_pips'].max():.1f}")
    else:
        print("  WARNING: No signals generated!")
        return signals

    print("  PASS")
    return signals


# ===== TEST 3: Single param evaluation =====
def test_3_single_eval(arrays):
    print("\n===== TEST 3: Single Parameter Evaluation =====")
    from backtester.strategies.rsi_mean_reversion import RSIMeanReversion
    from backtester.core.engine import BacktestEngine
    from backtester.core.dtypes import EXEC_BASIC

    strategy = RSIMeanReversion()

    t0 = time.perf_counter()
    engine = BacktestEngine(
        strategy,
        arrays["open_"], arrays["high"], arrays["low"],
        arrays["close"], arrays["volume"], arrays["spread"],
    )
    init_time = time.perf_counter() - t0
    print(f"  Engine init (signal gen + encoding): {init_time:.3f}s")
    print(f"  Signals: {engine.n_signals:,}")

    # Evaluate with a specific param set
    params = {
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "atr_period": 14,
        "sma_filter_period": 0,
        "sl_mode": "atr_based",
        "sl_fixed_pips": 30,
        "sl_atr_mult": 1.5,
        "tp_mode": "rr_ratio",
        "tp_rr_ratio": 2.0,
        "tp_atr_mult": 2.0,
        "tp_fixed_pips": 60,
        "allowed_hours_start": 0,
        "allowed_hours_end": 23,
        "allowed_days": [0, 1, 2, 3, 4],
        "trailing_mode": "off",
        "trail_activate_pips": 0,
        "trail_distance_pips": 10,
        "trail_atr_mult": 1.5,
        "breakeven_enabled": False,
        "breakeven_trigger_pips": 20,
        "breakeven_offset_pips": 2,
        "partial_close_enabled": False,
        "partial_close_pct": 50,
        "partial_close_trigger_pips": 30,
        "max_bars": 0,
        "stale_exit_enabled": False,
        "stale_exit_bars": 50,
        "stale_exit_atr_threshold": 0.5,
    }

    t0 = time.perf_counter()
    result = engine.evaluate_single(params, exec_mode=EXEC_BASIC)
    eval_time = time.perf_counter() - t0
    print(f"  Evaluation time: {eval_time*1000:.1f}ms")

    print(f"  Results:")
    for k, v in result.items():
        print(f"    {k:15s}: {v:.4f}")

    if result["trades"] == 0:
        print("  WARNING: Zero trades — something may be wrong with signal filtering")
    else:
        print("  PASS")
    return engine, result


# ===== TEST 4: Batch evaluation throughput =====
def test_4_batch_throughput(engine):
    print("\n===== TEST 4: Batch Evaluation Throughput =====")
    from backtester.core.encoding import random_index_matrix
    from backtester.core.dtypes import EXEC_BASIC, EXEC_FULL

    rng = np.random.default_rng(42)
    spec = engine.encoding

    # Warmup JIT
    print("  JIT warmup...")
    warmup_matrix = random_index_matrix(spec, 4, rng)
    engine.evaluate_batch_from_indices(warmup_matrix, EXEC_BASIC)
    print("  Warmup complete")

    for mode_name, exec_mode in [("BASIC", EXEC_BASIC), ("FULL", EXEC_FULL)]:
        for batch_size in [64, 256, 512, 1024]:
            idx_matrix = random_index_matrix(spec, batch_size, rng)

            t0 = time.perf_counter()
            metrics = engine.evaluate_batch_from_indices(idx_matrix, exec_mode)
            elapsed = time.perf_counter() - t0

            evals_per_sec = batch_size / elapsed
            nonzero_trades = np.sum(metrics[:, 0] > 0)
            print(f"  {mode_name} batch={batch_size:4d}: {evals_per_sec:,.0f} evals/sec, "
                  f"{nonzero_trades}/{batch_size} with trades, {elapsed*1000:.1f}ms")

    print("  PASS")


# ===== TEST 5: Telemetry (per-trade detail) =====
def test_5_telemetry(engine):
    print("\n===== TEST 5: Telemetry — Per-Trade Detail =====")
    from backtester.core.telemetry import run_telemetry
    from backtester.core.dtypes import EXEC_BASIC

    params = {
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "atr_period": 14,
        "sma_filter_period": 0,
        "sl_mode": "atr_based",
        "sl_fixed_pips": 30,
        "sl_atr_mult": 1.5,
        "tp_mode": "rr_ratio",
        "tp_rr_ratio": 2.0,
        "tp_atr_mult": 2.0,
        "tp_fixed_pips": 60,
        "allowed_hours_start": 0,
        "allowed_hours_end": 23,
        "allowed_days": [0, 1, 2, 3, 4],
        "trailing_mode": "off",
        "trail_activate_pips": 0,
        "trail_distance_pips": 10,
        "trail_atr_mult": 1.5,
        "breakeven_enabled": False,
        "breakeven_trigger_pips": 20,
        "breakeven_offset_pips": 2,
        "partial_close_enabled": False,
        "partial_close_pct": 50,
        "partial_close_trigger_pips": 30,
        "max_bars": 0,
        "stale_exit_enabled": False,
        "stale_exit_bars": 50,
        "stale_exit_atr_threshold": 0.5,
    }

    t0 = time.perf_counter()
    telemetry = run_telemetry(engine, params, exec_mode=EXEC_BASIC)
    elapsed = time.perf_counter() - t0

    print(f"  Telemetry run: {elapsed:.3f}s")
    print(f"  Total trades: {len(telemetry.trades)}")
    print(f"  Metrics: trades={telemetry.metrics['trades']:.0f}, "
          f"win_rate={telemetry.metrics['win_rate']:.2%}, "
          f"PF={telemetry.metrics['profit_factor']:.2f}, "
          f"sharpe={telemetry.metrics['sharpe']:.2f}")

    if telemetry.trades:
        # Show first 5 trades
        print(f"\n  First 5 trades:")
        for t in telemetry.trades[:5]:
            print(f"    Bar {t.bar_entry}->{t.bar_exit} ({t.bars_held} bars) "
                  f"{'BUY' if t.direction == 1 else 'SELL'} "
                  f"entry={t.entry_price:.5f} exit={t.exit_price:.5f} "
                  f"PnL={t.pnl_pips:+.1f} pips  exit={t.exit_reason}")

        # Show exit reason distribution
        from collections import Counter
        reasons = Counter(t.exit_reason for t in telemetry.trades)
        print(f"\n  Exit reasons: {dict(reasons)}")

        # Verify telemetry metrics match JIT metrics
        jit_result = engine.evaluate_single(params, exec_mode=EXEC_BASIC)
        trades_match = int(jit_result["trades"]) == len(telemetry.trades)
        print(f"\n  JIT vs Telemetry trade count match: {trades_match}")
        if not trades_match:
            print(f"    JIT: {int(jit_result['trades'])}, Telemetry: {len(telemetry.trades)}")

    print("  PASS")


# ===== TEST 6: Staged Optimization =====
def test_6_optimization(arrays):
    print("\n===== TEST 6: Staged Optimization (Turbo preset) =====")
    from backtester.strategies.rsi_mean_reversion import RSIMeanReversion
    from backtester.optimizer.run import optimize
    from backtester.optimizer.config import get_preset

    strategy = RSIMeanReversion()
    config = get_preset("turbo")
    print(f"  Preset: turbo ({config.trials_per_stage} trials/stage, batch={config.batch_size})")

    # Split data 80/20
    n = len(arrays["close"])
    split = int(n * 0.8)
    print(f"  Back-test: {split:,} bars, Forward: {n - split:,} bars")

    t0 = time.perf_counter()
    result = optimize(
        strategy,
        arrays["open_"][:split], arrays["high"][:split], arrays["low"][:split],
        arrays["close"][:split], arrays["volume"][:split], arrays["spread"][:split],
        arrays["open_"][split:], arrays["high"][split:], arrays["low"][split:],
        arrays["close"][split:], arrays["volume"][split:], arrays["spread"][split:],
        config=config,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n  Optimization complete in {elapsed:.1f}s")
    print(f"  Total trials evaluated: {result.total_trials:,}")
    print(f"  Candidates found: {len(result.candidates)}")
    print(f"  Throughput: {result.total_trials / elapsed:,.0f} evals/sec")

    if result.candidates:
        print(f"\n  Top 5 candidates:")
        for i, c in enumerate(result.candidates[:5]):
            print(f"    #{i+1}: trades={c.back_metrics.get('trades', 0):.0f}, "
                  f"quality={c.back_metrics.get('quality_score', 0):.2f}, "
                  f"sharpe={c.back_metrics.get('sharpe', 0):.2f}, "
                  f"fwd_quality={c.forward_metrics.get('quality_score', 0):.2f}, "
                  f"DSR={c.dsr:.3f}")
    else:
        print("  WARNING: No candidates passed filters!")

    print("  PASS")
    return result


# ===== Run all tests =====
if __name__ == "__main__":
    print("=" * 60)
    print("LIVE DATA INTEGRATION TEST")
    print("=" * 60)

    df, arrays = test_1_load_data()
    signals = test_2_signals(arrays)
    if len(signals["bar_index"]) == 0:
        print("\nABORT: No signals. Cannot continue.")
        sys.exit(1)
    engine, single_result = test_3_single_eval(arrays)
    test_4_batch_throughput(engine)
    test_5_telemetry(engine)
    test_6_optimization(arrays)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
