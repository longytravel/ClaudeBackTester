# /run-backtest — Interactive Optimization + Validation Run

Run a complete backtest pipeline for ANY registered strategy with live dashboard monitoring.

## Usage
```
/run-backtest [STRATEGY PAIR TIMEFRAME PRESET]
```
All arguments are optional. If any are missing, ask the user interactively using AskUserQuestion.

## Environment Setup (ALWAYS run first)

Every bash command in this skill MUST begin with this PATH setup:
```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
```

## Step 0: Gather Setup (FAST — single question if args provided)

### 0a. If ALL 4 args provided, skip to Step 1

### 0b. If any args missing, ask user with AskUserQuestion
Ask ONLY the missing arguments in a single AskUserQuestion call (up to 4 questions):

1. **Strategy** — run this to list: `uv run python -c "import sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8'); from backtester.strategies import registry; [print(s['name']) for s in registry.list_strategies()]"`
2. **Pair** — EUR/USD, GBP/USD, USD/JPY, AUD/USD, etc. (data in `G:/My Drive/BackTestData`)
3. **Timeframe** — H1, H4, M30, M15
4. **Preset** — turbo (~1 min), standard (~2 min, recommended), deep (~5 min), max (~10 min)

### 0c. Parameter space review — SKIP unless user explicitly asks
Show parameter space only if user says "show params", "review ranges", etc. Otherwise go straight to running.

### 0d. Budget & candidate review
Run `compute_stage_budgets()` to check for wasted compute, then ask the user what to do.

```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
uv run python -c "
from backtester.strategies import registry
from backtester.optimizer.config import get_preset
from backtester.optimizer.staged import compute_stage_budgets

strategy_cls = registry.get('STRATEGY')
strategy = strategy_cls()
config = get_preset('PRESET')
budgets = compute_stage_budgets(strategy, config)

print(f'Strategy: STRATEGY | Preset: PRESET | Candidates: {config.top_n_candidates}')
print()
print(f'{\"Stage\":<14} {\"Unique Combos\":>14} {\"Budget\":>10} {\"Coverage\":>10}')
print('-' * 52)
for b in budgets:
    combos = b['unique_combos']
    combo_str = f'{combos:,}' if combos < 1_000_000 else f'{combos/1e6:.0f}M' if combos < 1_000_000_000 else f'{combos/1e9:.0f}B'
    cov = b['coverage']
    cov_str = f'{cov:,.0f}x' if cov >= 1 else f'<1x'
    print(f'{b[\"stage\"]:<14} {combo_str:>14} {b[\"budget\"]:>10,} {cov_str:>10}')
"
```

Show the table to the user and highlight any stages with coverage >100x (heavy waste) or <1x (under-explored).

Then ask via AskUserQuestion with options:
- **"Run as-is (Recommended)"** — use current preset settings unchanged
- **"Save time"** — for over-covered stages (>100x), reduce their budget to `unique_combos * 10` (10x coverage is plenty). Show estimated time savings
- **"Bump candidates"** — increase candidates tested. Current default is 10. User can choose: 25, 50, or top 25% of passing trials. Explain: more candidates = more forward tests + validation runs, adds ~30s per extra candidate

If user picks "Save time": modify the `full_run.py` command below to pass `--trials-per-stage` with the reduced value. Note: this only makes sense if ALL non-refinement stages are over-covered; otherwise keep the default.

If user picks "Bump candidates": add `--top-n-candidates N` to the command. If they choose percentage, add `--top-n-candidates-pct 0.25`.

If user picks "Run as-is": proceed with defaults.

## Step 1: Pre-Flight Data Check + Dashboard Startup

Run data check and start React dashboard **in parallel**:

```bash
# Data check (replace PAIR/TF)
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
uv run python -c "
import pandas as pd; from pathlib import Path
pair_file = 'PAIR'.replace('/', '_'); DATA = Path('G:/My Drive/BackTestData')
tf_path = DATA / f'{pair_file}_TF.parquet'
if tf_path.exists():
    df = pd.read_parquet(tf_path)
    print(f'TF bars: {len(df):,}, NaN close: {df[\"close\"].isna().sum()}, Date range: {df.index[0]} to {df.index[-1]}')
else:
    print(f'TF file not found -- will build from M1')
"
```

```bash
# Start React dashboard (background, non-blocking)
cd /c/Users/ROG/Projects/ClaudeBackTester/dashboard && npm run dev &
```

**If NaN close bars detected OR timeframe file missing**, rebuild first:
```bash
uv run python -c "from backtester.data.timeframes import convert_timeframes; convert_timeframes('PAIR', 'G:/My Drive/BackTestData', ['TF'])"
```

## Step 2: Run the Pipeline with Dashboard

**CRITICAL**: Always use `--dashboard` flag. The dashboard server is embedded in full_run.py (NOT standalone).

```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
uv run python scripts/full_run.py \
  --strategy STRATEGY \
  --pair PAIR \
  --timeframe TIMEFRAME \
  --preset PRESET \
  --dashboard
```

Run this as a **foreground** command with `timeout: 600000` (10 min). Do NOT run in background — the server stays alive after completion for the user to view results, and killing the process kills the dashboard.

After launching, tell the user: **"Dashboard is live at http://localhost:5173 — open it to watch progress in real-time."**

Then use the Playwright browser to take a screenshot of the dashboard and show it to the user while the pipeline runs. Navigate to `http://localhost:5173` and take a full-page screenshot.

The pipeline will:
1. Load data, split 80/20 back/forward
2. Run staged optimization (signal → time → risk → management → refinement)
3. Run validation pipeline (walk-forward, CPCV, stability, Monte Carlo + regime, confidence)
4. Run telemetry for trade statistics
5. Print report + keep dashboard server alive for browsing

After the pipeline prints results (look for `"Total elapsed"` in output), the server stays alive. Read the output and move to Step 3.

## Step 3: Interpret Results

Read the pipeline output. Key things to check:

### Red Flags (investigate immediately)
- **0 trades** in forward test → signal filtering too strict or data issue
- **NaN in any metric** → data quality issue
- **forward_back_ratio < 0.4** → overfit to back-test period
- **All walk-forward windows fail** → strategy doesn't generalize
- **< 50 trades over full period** → statistically meaningless, even if quality score looks good
- **High win rate (>90%) + mostly breakeven exits** → breakeven trap, not a real edge

### Confidence Rating
- **GREEN (70+)**: All gates passed, safe to paper trade
- **YELLOW (40-69)**: Gates passed but moderate score, needs investigation
- **RED (<40 or gate failed)**: Hard gate failed, do NOT trade

### Hard Gates (all must pass)
1. forward_back_ratio >= 0.4
2. Walk-forward pass rate >= 60%
3. Walk-forward mean OOS Sharpe >= 0.3
4. CPCV: pct_positive_sharpe >= 60% (when enabled)
5. CPCV: mean_sharpe >= 0.2 (when enabled)
6. DSR >= 0.95
7. Permutation p-value <= 0.05

## Step 4: Show Dashboard Results

After the pipeline completes, take a screenshot of the dashboard and present results to user:

1. Navigate browser to `http://localhost:5173`
2. Take a full-page screenshot
3. Summarize: verdict (GREEN/YELLOW/RED), key metrics, parameter values, warnings

If the dashboard shows an error or "Connecting", check:
- Is the full_run.py process still alive? (`tasklist | grep python`)
- API responding? (`curl http://localhost:8765/api/run/status`)
- If server died, read `results/{pair}_{tf}/report.json` directly

## Step 5: Cleanup

After user has seen results, kill the pipeline process (it's in a sleep loop keeping the server alive):
```bash
# The user will Ctrl+C the foreground process, or we kill it
taskkill //F //IM python.exe //FI "WINDOWTITLE eq *full_run*" 2>/dev/null
```

Also stop the Vite dev server if no longer needed.

## PIP_VALUE Reference
| Pair | pip_value |
|------|-----------|
| EUR/USD, GBP/USD, AUD/USD, NZD/USD | 0.0001 |
| USD/JPY, EUR/JPY, GBP/JPY, AUD/JPY | 0.01 |
| XAU/USD | 0.01 |

## Estimated Run Times (i9-14900HX)
| Preset | Trials/stage | Approx time |
|--------|-------------|-------------|
| turbo | 50K | ~1 min |
| standard | 200K | ~2-6 min |
| deep | 500K | ~5-10 min |
| max | 1M | ~10-15 min |

## Rust Rebuild (only if Rust source changed)

If `rust/src/*.rs` files were modified, rebuild before running:
```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
.venv/Scripts/maturin.exe develop --release --manifest-path rust/Cargo.toml
```

## Bug History (watch for regressions)
- **Quality score 655 with 40 trades** — Sortino was used raw in quality formula. Fixed: now uses `ln(1+Sortino)` to compress extreme values naturally (Feb 2026)
- **Equity curve unsorted** — trades listed BUY-first then SELL, not chronological. Dashboard chart crashed. Fixed: sort by exit bar + sort defense in EquityCurve.tsx (Feb 2026)
- **Dashboard "disappeared"** — EquityCurve component crash (unsorted data) killed entire React tree. No error boundary. Fixed: sort data in component (Feb 2026)
- **Timezone mismatch** — `pd.Timestamp("1970-01-01")` is tz-naive but data index is tz-aware. Fixed: use `pd.Timestamp("1970-01-01", tz="UTC")` in full_run.py (Feb 2026)
- **Dashboard not connecting** — Must use `--dashboard` flag when running full_run.py. Dashboard server is embedded, NOT standalone. Do NOT start `backtester/dashboard/server.py` separately (Feb 2026)
- **DSR formula** was computing `1 - cdf(SR)` instead of `cdf(SR)` (fixed 5f1e2a8)
- **Timeframe conversion** left NaN weekend bars when M1 had gaps (fixed)
- **Forward/back ratio** was never computed in optimizer (fixed)
- **Sell slippage** was missing from JIT and telemetry (fixed)
- **max_trades_per_trial** default was 5000, caused truncation (increased to 50000)
- **maturin not in PATH** — use `.venv/Scripts/maturin.exe` directly, and always add `$HOME/.cargo/bin` to PATH for rustc
