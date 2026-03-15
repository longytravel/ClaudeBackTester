"""Integration smoke test for a strategy — verifies it works end-to-end with real data.

Loads real Dukascopy data, runs signal generation, feeds into BacktestEngine,
and checks that nothing crashes and results are sane. This catches the class of
bugs where unit tests pass (synthetic data) but the real pipeline blows up.

Usage:
    uv run python scripts/smoke_test_strategy.py --strategy hidden_smash_day
    uv run python scripts/smoke_test_strategy.py --strategy hidden_smash_day --pair EUR/USD --timeframe H1
    uv run python scripts/smoke_test_strategy.py --strategy hidden_smash_day --n-trials 20
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("G:/My Drive/BackTestData")

# pip values per pair
PIP_VALUES = {
    "EUR/USD": 0.0001, "GBP/USD": 0.0001, "AUD/USD": 0.0001, "NZD/USD": 0.0001,
    "USD/JPY": 0.01, "EUR/JPY": 0.01, "GBP/JPY": 0.01, "AUD/JPY": 0.01,
    "XAU/USD": 0.01,
}

BARS_PER_YEAR = {
    "M1": 362880, "M5": 72576, "M15": 24192, "M30": 12096,
    "H1": 6048, "H4": 1512, "D1": 252,
}


def load_data(pair: str, timeframe: str) -> pd.DataFrame:
    """Load parquet data for pair/timeframe."""
    pair_file = pair.replace("/", "_")
    path = DATA_DIR / f"{pair_file}_{timeframe}.parquet"
    if not path.exists():
        print(f"FAIL: Data file not found: {path}")
        sys.exit(1)
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df):,} bars from {path.name}")
    return df


def run_smoke_test(strategy_name: str, pair: str, timeframe: str, n_trials: int) -> bool:
    """Run the full smoke test. Returns True if passed."""
    # Late imports so script fails fast on bad args
    from backtester.core.engine import BacktestEngine
    from backtester.core.encoding import encode_params
    from backtester.strategies.registry import create

    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {strategy_name}")
    print(f"Pair: {pair} | Timeframe: {timeframe} | Trials: {n_trials}")
    print(f"{'='*60}\n")

    # --- Step 1: Load strategy ---
    print("[1/5] Loading strategy...")
    try:
        strategy = create(strategy_name)
        print(f"  OK: {strategy.name} v{strategy.version}")
    except Exception as e:
        print(f"  FAIL: Could not create strategy: {e}")
        return False

    # --- Step 2: Load real data ---
    print(f"\n[2/5] Loading {pair} {timeframe} data...")
    try:
        df = load_data(pair, timeframe)
    except Exception as e:
        print(f"  FAIL: Could not load data: {e}")
        return False

    # Extract arrays
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64) if "volume" in df.columns else np.ones(len(df))
    spread = df["spread"].values.astype(np.float64) if "spread" in df.columns else np.full(len(df), 0.00015)
    pip_value = PIP_VALUES.get(pair, 0.0001)

    # Time arrays — must be int64 for Rust/PyO3 compatibility
    if hasattr(df.index, 'hour'):
        bar_hour = df.index.hour.values.astype(np.int64)
        bar_dow = df.index.dayofweek.values.astype(np.int64)
    else:
        bar_hour = np.zeros(len(df), dtype=np.int64)
        bar_dow = np.zeros(len(df), dtype=np.int64)

    n_bars = len(close)
    nan_close = np.isnan(close).sum()
    nan_spread = np.isnan(spread).sum()
    print(f"  Bars: {n_bars:,} | NaN close: {nan_close} | NaN spread: {nan_spread}")

    if nan_close > 0:
        print(f"  WARNING: {nan_close} NaN close values — may cause issues")

    # --- Step 3: Generate signals ---
    print(f"\n[3/5] Generating signals...")
    space = strategy.param_space()
    # Use mid-range params (needed later for engine, not for signal generation)
    params = {}
    for p in space:
        mid_idx = len(p.values) // 2
        params[p.name] = p.values[mid_idx]

    # Engine calls generate_signals_vectorized with this exact signature:
    #   (open_, high, low, close, volume, spread, pip_value, bar_hour, bar_day_of_week)
    # No atr_arr and no params — strategies compute ATR internally
    t0 = time.perf_counter()
    try:
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread, pip_value,
            bar_hour, bar_dow,
        )
    except Exception as e:
        print(f"  FAIL: generate_signals_vectorized crashed: {e}")
        import traceback
        traceback.print_exc()
        return False
    t_sig = time.perf_counter() - t0

    # Validate signal output — engine expects these exact 8 keys
    required_keys = {"bar_index", "direction", "entry_price", "hour", "day_of_week",
                     "atr_pips", "variant", "filter_value"}
    missing = required_keys - set(result.keys())
    if missing:
        print(f"  FAIL: Missing keys in signal output: {missing}")
        print(f"  Got keys: {set(result.keys())}")
        print(f"  Expected: {required_keys}")
        return False

    n_signals = len(result["bar_index"])
    print(f"  OK: {n_signals:,} signals generated in {t_sig:.3f}s")

    if n_signals == 0:
        # Signal generation doesn't take params — the engine filters signals later.
        # So zero signals here means the strategy logic genuinely found nothing.
        print(f"  FAIL: Zero signals on {n_bars:,} bars of real data.")
        print(f"  This means the strategy's pattern detection found no entries at all.")
        print(f"  Check: indicator warmup, pattern conditions, array indexing.")
        return False

    # Sanity checks on signals
    indices = result["bar_index"]
    directions = result["direction"]
    entry_prices = result["entry_price"]
    atr_pips = result["atr_pips"]

    checks_passed = True

    if indices.min() < 0 or indices.max() >= n_bars:
        print(f"  FAIL: Signal indices out of range: [{indices.min()}, {indices.max()}] vs [0, {n_bars-1}]")
        checks_passed = False

    if not np.all(np.isin(directions, [1, -1])):
        print(f"  FAIL: Invalid directions found: {np.unique(directions)}")
        checks_passed = False

    if np.any(np.isnan(entry_prices)):
        nan_count = np.isnan(entry_prices).sum()
        print(f"  FAIL: {nan_count} NaN entry prices")
        checks_passed = False

    if np.any(np.isnan(atr_pips)):
        nan_count = np.isnan(atr_pips).sum()
        print(f"  FAIL: {nan_count} NaN ATR pips")
        checks_passed = False

    if np.any(atr_pips <= 0):
        bad_count = (atr_pips <= 0).sum()
        print(f"  FAIL: {bad_count} non-positive ATR pips")
        checks_passed = False

    n_buy = (directions == 1).sum()
    n_sell = (directions == -1).sum()
    print(f"  BUY: {n_buy:,} | SELL: {n_sell:,}")

    if n_buy == 0 or n_sell == 0:
        print(f"  WARNING: Only one direction — strategy may be directionally biased")

    if not checks_passed:
        return False

    # --- Step 4: Run through BacktestEngine ---
    print(f"\n[4/5] Running {n_trials} trials through Rust engine...")

    try:
        engine = BacktestEngine(
            strategy, open_, high, low, close, volume, spread,
            pip_value=pip_value,
            slippage_pips=0.0,
            commission_pips=0.0,
            max_spread_pips=0.0,
            bars_per_year=BARS_PER_YEAR.get(timeframe, 6048),
            bar_hour=bar_hour,
            bar_day_of_week=bar_dow,
        )
    except Exception as e:
        print(f"  FAIL: BacktestEngine init crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Build param matrix — vary signal params, keep others at mid-range
    spec = engine.encoding
    rows = []
    signal_params_list = [p for p in space if p.group == "signal"]

    for trial in range(n_trials):
        trial_params = dict(params)
        # Vary signal params randomly across their value range
        rng = np.random.RandomState(42 + trial)
        for p in signal_params_list:
            idx = rng.randint(0, len(p.values))
            trial_params[p.name] = p.values[idx]
        try:
            row = encode_params(spec, trial_params)
            rows.append(row)
        except Exception as e:
            print(f"  FAIL: encode_params failed for trial {trial}: {e}")
            return False

    param_matrix = np.vstack(rows)
    print(f"  Param matrix shape: {param_matrix.shape}")

    from backtester.core.dtypes import EXEC_BASIC
    t0 = time.perf_counter()
    try:
        metrics = engine.evaluate_batch(param_matrix, exec_mode=EXEC_BASIC)
    except Exception as e:
        print(f"  FAIL: evaluate_batch crashed: {e}")
        import traceback
        traceback.print_exc()
        return False
    t_eval = time.perf_counter() - t0

    print(f"  OK: {n_trials} trials completed in {t_eval:.3f}s ({n_trials/t_eval:.0f} evals/sec)")
    print(f"  Metrics shape: {metrics.shape}")

    # --- Step 5: Validate results ---
    print(f"\n[5/5] Validating results...")

    # Check for NaN/Inf in metrics
    nan_count = np.isnan(metrics).sum()
    inf_count = np.isinf(metrics).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values in metrics")
    if inf_count > 0:
        print(f"  WARNING: {inf_count} Inf values in metrics")

    # Metrics columns: trades, win_rate, pf, sharpe, sortino, max_dd, return_pct, r2, ulcer, quality
    trades_col = metrics[:, 0]
    total_trades = trades_col.sum()
    trials_with_trades = (trades_col > 0).sum()

    print(f"  Total trades across all trials: {total_trades:.0f}")
    print(f"  Trials with >0 trades: {trials_with_trades}/{n_trials}")

    if trials_with_trades == 0:
        print(f"  FAIL: Zero trades across ALL {n_trials} trials — strategy is broken")
        return False

    # Show best trial
    quality_col = metrics[:, 9]
    best_idx = np.argmax(quality_col)
    best_metrics = metrics[best_idx]
    print(f"\n  Best trial (#{best_idx}):")
    labels = ["trades", "win_rate", "pf", "sharpe", "sortino", "max_dd%", "return%", "r2", "ulcer", "quality"]
    for label, val in zip(labels, best_metrics):
        print(f"    {label:>10}: {val:.4f}")

    # Final verdict
    print(f"\n{'='*60}")
    print(f"SMOKE TEST PASSED: {strategy_name}")
    print(f"  Signals: {n_signals:,} on {n_bars:,} bars")
    print(f"  Trades: {total_trades:.0f} across {n_trials} trials")
    print(f"  Engine: no crashes, metrics finite")
    print(f"{'='*60}\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Integration smoke test for a strategy")
    parser.add_argument("--strategy", required=True, help="Strategy name (registered)")
    parser.add_argument("--pair", default="EUR/USD", help="Currency pair (default: EUR/USD)")
    parser.add_argument("--timeframe", default="H1", help="Timeframe (default: H1)")
    parser.add_argument("--n-trials", type=int, default=10, help="Number of param trials (default: 10)")
    args = parser.parse_args()

    success = run_smoke_test(args.strategy, args.pair, args.timeframe, args.n_trials)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
