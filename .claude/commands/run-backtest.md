# /run-backtest — Full End-to-End Optimization + Validation Run

Run a complete backtest pipeline: load data, optimize parameters, run validation (walk-forward, stability, Monte Carlo, confidence scoring), and produce a comprehensive report.

## Usage
```
/run-backtest <PAIR> <TIMEFRAME> <PRESET>
```
Examples: `/run-backtest EUR/USD H1 turbo`, `/run-backtest GBP/USD H4 fast`

**Arguments:**
- `PAIR`: Currency pair with slash (EUR/USD, GBP/USD, USD/JPY, XAU/USD)
- `TIMEFRAME`: H1, H4, D, M15, M30 (built from M1 data)
- `PRESET`: turbo (50K/stage), standard (200K), deep (500K), max (1M)

## Step 1: Pre-Flight Data Checks

Before running, check data quality:

```bash
export PATH="/c/Users/ROG/.local/bin:$PATH"
uv run python -c "
import pandas as pd, numpy as np
from pathlib import Path

PAIR = '$ARGUMENTS'.split()[0] if '$ARGUMENTS'.strip() else 'EUR/USD'
TF = '$ARGUMENTS'.split()[1] if len('$ARGUMENTS'.split()) > 1 else 'H1'
pair_file = PAIR.replace('/', '_')
DATA_DIR = Path('G:/My Drive/BackTestData')

# Check M1 exists
m1 = DATA_DIR / f'{pair_file}_M1.parquet'
print(f'M1 exists: {m1.exists()}')
if m1.exists():
    df_m1 = pd.read_parquet(m1)
    print(f'M1 bars: {len(df_m1):,}')

# Check target timeframe
tf_path = DATA_DIR / f'{pair_file}_{TF}.parquet'
if tf_path.exists():
    df = pd.read_parquet(tf_path)
    nan_close = df['close'].isna().sum()
    nan_spread = df['spread'].isna().sum()
    print(f'{TF} bars: {len(df):,}, NaN close: {nan_close}, NaN spread: {nan_spread}')
    print(f'Date range: {df.index[0]} to {df.index[-1]}')
    if nan_close > 0:
        print('WARNING: NaN close bars detected! Will rebuild from M1.')
else:
    print(f'{TF} file not found — will build from M1')
"
```

**If NaN close bars detected OR timeframe file missing**, rebuild:
```bash
uv run python -c "
from backtester.data.timeframes import convert_timeframes
convert_timeframes('PAIR', 'G:/My Drive/BackTestData', ['TIMEFRAME'])
"
```

## Step 2: Run the Pipeline

```bash
export PATH="/c/Users/ROG/.local/bin:$PATH"
uv run python scripts/full_run.py --pair PAIR --timeframe TIMEFRAME --preset PRESET
```

Replace PAIR, TIMEFRAME, PRESET with the user's arguments (e.g. `--pair EUR/USD --timeframe H1 --preset turbo`).

The script will:
1. Load data + check for NaN bars (auto-rebuild if needed)
2. Split 80/20 back/forward
3. Run staged optimization
4. Run validation pipeline (walk-forward, CPCV, stability, Monte Carlo + regime, confidence)
5. Run telemetry for trade statistics
6. Print comprehensive report (with 3b: CPCV, 5b: Regime when enabled)

## Step 3: Interpret Results

After the script completes, read and present the output to the user. Key things to check:

### Red Flags (investigate immediately)
- **DSR = 0.0** on a strategy with positive Sharpe → DSR formula bug (was fixed in 5f1e2a8, but watch for regression)
- **0 trades** in forward test → signal filtering too strict, or data issue
- **NaN in any metric** → data quality issue, check spread/close arrays
- **forward_back_ratio < 0.4** → strategy is overfit to back-test period
- **All walk-forward windows fail** → strategy doesn't generalize, consider different param ranges
- **Stability worst_param with ratio < 0.1** → that param is on a cliff edge, strategy fragile

### Known Gotchas
1. **Spread is in price units** not pips — Dukascopy parquet stores ask-bid directly
2. **Sharpe from JIT is annualized** using sqrt(trades_per_year), not sqrt(total_trades)
3. **PIP_VALUE varies**: EUR/USD=0.0001, JPY pairs=0.01, XAU/USD=0.01
4. **Turbo preset** (1K trials/stage) — intermediate stages often fail to find good params; refinement stage rescues by searching around the best-so-far
5. **Optimizer returns top N diverse candidates** — diversity archive selects multiple, forward/back gate filters overfitters
6. **Walk-forward windows**: at H1, default window=8760 bars (~1.45 years), step=4380
7. **Stability worst_param** is often trailing_mode — discrete params flip causing large quality changes

### Confidence Rating Guide
- **GREEN (70+)**: All gates passed, strong composite score. Safe to paper trade.
- **YELLOW (40-69)**: All gates passed but composite score moderate. Needs investigation.
- **RED (<40 or gate failed)**: At least one hard gate failed. Do NOT trade this.

### Hard Gates (all must pass)
1. forward_back_ratio >= 0.4
2. Walk-forward pass rate >= 60%
3. Walk-forward mean OOS Sharpe >= 0.3
4. CPCV: pct_positive_sharpe >= 60% (when CPCV enabled)
5. CPCV: mean_sharpe >= 0.2 (when CPCV enabled)
6. DSR >= 0.95
7. Permutation p-value <= 0.05

### CPCV Interpretation
- **pct_positive_sharpe**: Fraction of C(N,k) folds with positive OOS Sharpe. Higher = more robust across data splits.
- **95% CI width**: Narrow CI (< 1.0) means consistent performance. Wide CI (> 2.0) means high variance across folds.
- **mean vs median Sharpe**: Large gap suggests outlier folds skewing the mean. Prefer median for robustness.
- **CPCV blending**: When CPCV is available, the walk-forward confidence weight blends 60% WF + 40% CPCV. When disabled, 100% WF (backward compatible).

### Regime Analysis Interpretation
- **4 Regimes**: Trend+Quiet, Trend+Volatile, Range+Quiet, Range+Volatile
- **Classified by**: ADX(14) with hysteresis (25 enter/20 exit) + NATR percentile (75th = high vol)
- **Minimum 30 trades** per regime for stats (below = "Insufficient")
- **Advisory only** — does NOT eliminate candidates or affect confidence score
- **Key insight**: Range+Volatile is dangerous for nearly all strategies
- **Robust strategy**: profitable in 2+ regimes, no regime with MaxDD > 40%
- **Weak strategy**: only profitable in 1 regime, or catastrophic loss in any regime

## Step 4: Present Results to User

Summarize the output in a clean format. Include:
- Verdict (GREEN/YELLOW/RED) prominently
- Key metrics table (Sharpe, trades, win rate, profit factor, max DD)
- Parameter values
- Any warnings or concerns
- Recommendation on next steps

## PIP_VALUE Reference
| Pair | pip_value |
|------|-----------|
| EUR/USD, GBP/USD, AUD/USD | 0.0001 |
| USD/JPY, EUR/JPY, GBP/JPY | 0.01 |
| XAU/USD | 0.01 |

## Bug History (watch for regressions)
- **DSR formula** was computing `1 - cdf(SR)` instead of `cdf(SR)` → fixed in commit 5f1e2a8
- **Timeframe conversion** left NaN weekend bars when M1 had gaps → fixed with `dropna(subset=["close"])`
- **Forward/back ratio** was never computed in optimizer → fixed (now computed in optimize())
- **Sell slippage** was missing from both JIT and telemetry → fixed
- **max_trades_per_trial** default was 5000, caused truncation → increased to 50000
