# /run-backtest — Interactive Optimization + Validation Run

Run a complete backtest pipeline for ANY registered strategy. The workflow is interactive: you ask the user about setup, review parameter ranges together, then run and interpret results.

## Usage
```
/run-backtest [PAIR TIMEFRAME PRESET STRATEGY]
```
All arguments are optional. If any are missing, ask the user interactively using AskUserQuestion.

## Step 0: Discover Strategies and Gather Setup

### 0a. List available strategies
```bash
export PATH="/c/Users/ROG/.local/bin:$PATH"
uv run python -c "
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from backtester.strategies import registry
for s in registry.list_strategies():
    print(f'{s[\"name\"]} ({s[\"stage\"]})')
"
```

### 0b. Ask user for missing arguments
Use AskUserQuestion to gather anything not provided in $ARGUMENTS. Ask up to 4 questions at once:

1. **Strategy** — which strategy to test (from registry list)
2. **Pair** — EUR/USD, GBP/USD, USD/JPY, XAU/USD (check what data exists in `G:/My Drive/BackTestData`)
3. **Timeframe** — H1, H4, D, M15, M30
4. **Preset** — How thorough:
   - turbo (50K/stage, ~1 min) — quick sanity check
   - standard (200K/stage, ~2 min) — normal run
   - deep (500K/stage, ~5 min) — thorough search
   - max (1M/stage, ~10 min) — exhaustive, uses full hardware

### 0c. Review parameter space with user
This is the most important interactive step. Load the strategy's param space and show it grouped by optimization stage:

```bash
export PATH="/c/Users/ROG/.local/bin:$PATH"
uv run python -c "
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from backtester.strategies import registry
strat = registry.create('STRATEGY_NAME')
ps = strat.param_space()
stages = strat.optimization_stages()
from collections import defaultdict
by_group = defaultdict(list)
total_combos = 1
for name in ps.names:
    p = ps.get(name)
    by_group[p.group].append(p)
    total_combos *= len(p.values)
for stage in stages:
    print(f'\n=== {stage.upper()} stage ===')
    for p in by_group.get(stage, []):
        vals = str(p.values)
        print(f'  {p.name:30s} : {len(p.values):3d} values  {vals}')
print(f'\nTotal params: {len(ps.names)}, Total combinations: {total_combos:.2e}')
print(f'Optimization stages: {\" -> \".join(stages)} -> refinement')
"
```

Present this to the user as a clear table and ask:
- **Are these parameter ranges wide enough?** Should any be wider/narrower?
- **Signal params specifically** — these are strategy-specific and control trade frequency. Wider = more signals to find patterns in. Narrower = focused search on known-good territory.
- **Risk params** — SL/TP ranges. Current ranges: SL 10-100 pips, TP 10-200 pips. Appropriate for the pair?
- **Management params** — trailing stops, breakeven, partial close. These are feature toggles (on/off) plus numeric ranges. Usually fine as-is.
- **Time params** — hour and day filters. 24 hours * 24 hours * 4 day combos = 2,304 combos. Often the tightest filter.

If the user wants to change parameter ranges, edit the strategy file directly (the `param_space()` method or the shared `risk_params()`/`management_params()`/`time_params()` in `backtester/strategies/base.py`).

**Important context for the user**: The optimizer uses staged search (signal -> time -> risk -> management -> refinement), so it doesn't need to test all combinations at once. Each stage locks the best params before moving to the next. But wider ranges per stage = more exploration needed = higher preset recommended.

## Step 1: Pre-Flight Data Checks

```bash
export PATH="/c/Users/ROG/.local/bin:$PATH"
uv run python -c "
import pandas as pd, numpy as np
from pathlib import Path

PAIR = 'PAIR_VALUE'
TF = 'TF_VALUE'
pair_file = PAIR.replace('/', '_')
DATA_DIR = Path('G:/My Drive/BackTestData')

m1 = DATA_DIR / f'{pair_file}_M1.parquet'
print(f'M1 exists: {m1.exists()}')
if m1.exists():
    df_m1 = pd.read_parquet(m1)
    print(f'M1 bars: {len(df_m1):,}')

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
    print(f'{TF} file not found -- will build from M1')
"
```

**If NaN close bars detected OR timeframe file missing**, rebuild:
```bash
uv run python -c "
from backtester.data.timeframes import convert_timeframes
convert_timeframes('PAIR_VALUE', 'G:/My Drive/BackTestData', ['TF_VALUE'])
"
```

## Step 2: Run the Pipeline

```bash
export PATH="/c/Users/ROG/.local/bin:$PATH"
uv run python scripts/full_run.py --strategy STRATEGY --pair PAIR --timeframe TIMEFRAME --preset PRESET
```

The script will:
1. Load data + check for NaN bars (auto-rebuild if needed)
2. Split 80/20 back/forward
3. Run staged optimization (signal -> time -> risk -> management -> refinement)
4. Run validation pipeline (walk-forward, CPCV, stability, Monte Carlo + regime, confidence)
5. Run telemetry for trade statistics
6. Print comprehensive report (with 3b: CPCV, 5b: Regime when enabled)

## Step 3: Interpret Results

After the script completes, read and present the output to the user. Key things to check:

### Red Flags (investigate immediately)
- **DSR = 0.0** on a strategy with positive Sharpe -> DSR formula regression
- **0 trades** in forward test -> signal filtering too strict, or data issue
- **NaN in any metric** -> data quality issue, check spread/close arrays
- **forward_back_ratio < 0.4** -> strategy is overfit to back-test period
- **All walk-forward windows fail** -> strategy doesn't generalize
- **Stability worst_param with ratio < 0.1** -> that param is on a cliff edge
- **Time or management stage 0 passing** -> too few base signals, widen signal params or use a less selective strategy

### Known Gotchas
1. **Spread is in price units** not pips (Dukascopy stores ask-bid directly)
2. **Sharpe from JIT is annualized** using sqrt(trades_per_year)
3. **PIP_VALUE varies** by pair (see reference below)
4. **Turbo preset** may miss good params in intermediate stages; refinement rescues
5. **Walk-forward windows**: at H1, window=8760 bars (~1.45 years), step=4380
6. **Stability worst_param** is often trailing_mode (discrete param flips)
7. **Trade frequency matters**: a strategy needs ~10+ trades per WF window to pass. If your strategy only produces 20-50 trades over 15 years, walk-forward will always fail regardless of how good the params are

### Confidence Rating Guide
- **GREEN (70+)**: All gates passed, strong composite score. Safe to paper trade.
- **YELLOW (40-69)**: All gates passed but composite score moderate. Needs investigation.
- **RED (<40 or gate failed)**: At least one hard gate failed. Do NOT trade this.

### Hard Gates (all must pass)
1. forward_back_ratio >= 0.4
2. Walk-forward pass rate >= 60%
3. Walk-forward mean OOS Sharpe >= 0.3
4. CPCV: pct_positive_sharpe >= 60% (when enabled)
5. CPCV: mean_sharpe >= 0.2 (when enabled)
6. DSR >= 0.95
7. Permutation p-value <= 0.05

### CPCV Interpretation
- **pct_positive_sharpe**: Fraction of C(N,k) folds with positive OOS Sharpe
- **95% CI width**: Narrow CI (< 1.0) = consistent. Wide CI (> 2.0) = high variance
- **CPCV blending**: 60% WF + 40% CPCV when available, 100% WF when disabled

### Regime Analysis Interpretation
- **4 Regimes**: Trend+Quiet, Trend+Volatile, Range+Quiet, Range+Volatile
- **Classified by**: ADX(14) with hysteresis (25/20) + NATR percentile (75th = high vol)
- **Minimum 30 trades** per regime for stats (below = "Insufficient")
- **Advisory only** -- does NOT affect confidence score
- **Robust strategy**: profitable in 2+ regimes, no regime with MaxDD > 40%

## Step 4: Present Results to User

Summarize the output in a clean format. Include:
- Verdict (GREEN/YELLOW/RED) prominently
- Key metrics table (Sharpe, trades, win rate, profit factor, max DD)
- Parameter values
- Parameter range review: did the optimizer hit edge values? (best param = min or max of range suggests the range needs widening)
- Any warnings or concerns
- Recommendation on next steps

## PIP_VALUE Reference
| Pair | pip_value |
|------|-----------|
| EUR/USD, GBP/USD, AUD/USD | 0.0001 |
| USD/JPY, EUR/JPY, GBP/JPY | 0.01 |
| XAU/USD | 0.01 |

## Estimated Run Times (i9-14900HX, EUR/USD H1)
| Preset | Trials/stage | Total trials | Approx time |
|--------|-------------|-------------|-------------|
| turbo | 50K | ~350K | ~1 min |
| standard | 200K | ~1.2M | ~2 min |
| deep | 500K | ~3.5M | ~5-6 min |
| max | 1M | ~7M | ~10-12 min |

## Bug History (watch for regressions)
- **DSR formula** was computing `1 - cdf(SR)` instead of `cdf(SR)` (fixed 5f1e2a8)
- **Timeframe conversion** left NaN weekend bars when M1 had gaps (fixed)
- **Forward/back ratio** was never computed in optimizer (fixed)
- **Sell slippage** was missing from JIT and telemetry (fixed)
- **max_trades_per_trial** default was 5000, caused truncation (increased to 50000)
