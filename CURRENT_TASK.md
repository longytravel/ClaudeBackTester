# Current Task

## Last Completed (Mar 15 2026)

### Pipeline Candidate Selection Redesign
Major redesign of how candidates flow from optimizer to validation pipeline, based on research from Codex GPT-5.4, NotebookLM (55 papers), and our own analysis.

**Problem**: MAP-Elites diversity archive (3×4=12 cell grid) collapsed 58K candidates to 4. Forward/back gate killed all 4. Walk-forward was skipped entirely. Pipeline produced 0 survivors.

**Fix**: Replaced with DSR prefilter → signal+risk param deduplication → top N by IS quality → forward-test for reporting only (not selection).

**Result**: Same strategy (Hidden Smash Day, EUR/USD H1) went from 0/4 survivors to 10/10 survivors. Walk-forward now actually runs (4/6 windows pass, mean Sharpe 1.49). Strategy is forward-profitable (+128 pips).

**Files changed**:
- `backtester/optimizer/run.py` — new `_select_pipeline_candidates()` replacing `_add_multi_candidates()`
- `backtester/optimizer/config.py` — added `max_pipeline_candidates`, `dsr_prefilter_threshold`, `dsr_prefilter_fallback`, `max_per_dedup_group`
- `scripts/full_run.py` — removed forward_gate pre-elimination
- `backtester/pipeline/confidence.py` — forward_back_ratio demoted from hard gate to soft score
- `backtester/pipeline/runner.py` — removed forward_gate from stage ordering
- `dashboard/src/components/results/EquityCurve.tsx` — £ currency conversion, 500px height, fitContent(), pair-aware pip value
- `dashboard/src/components/results/PipelineFunnel.tsx` — new funnel labels (DSR, dedup, pipeline candidates)
- `dashboard/src/components/results/RunSummary.tsx` — updated narrative
- `dashboard/src/types/api.ts` — new OptimizerFunnel fields
- All 521 tests pass

**Research briefs saved**: `research/briefs/candidate_selection_*.md`

### Previous Work
- Multi-AI Quality Loop + Bug Fixes (entry price, breakeven, partial close, sell spread)
- Modular Staged Optimizer: NUM_PL 27→64, ManagementModule system
- Live trading engine deployed on IC Markets demo (EMA, MACD, Stochastic)

## Next Steps

### Step 1: Run Hidden Smash Day with Standard Preset
- Turbo only tested 200K trials (50K/stage). Standard does 200K/stage = 4× more exploration.
- Turbo `max_pipeline_candidates=10`. Standard = 20.
- Run: `/run-backtest hidden_smash_day EUR/USD H1 standard`
- Compare candidate diversity and pipeline survival rates vs turbo

### Step 2: Commit Pipeline Redesign
- Commit all changes with descriptive message
- Update PROGRESS.md

### Step 3: Run a Different Strategy
- Pick a strategy with more signal parameters (EMA, MACD, Bollinger) to test dedup diversity
- Hidden Smash Day has very few signal params (just hsd_variant) — not a great diversity test

### Step 4: Monitor Live Traders
- Market opens Monday — check traders generating signals
- Use `/verify-trades` to compare backtest vs live

## Blockers
- None

## Key Context
- Pipeline redesign: DSR prefilter → dedup → top N → pipeline (no forward gate elimination)
- Dashboard: equity curve now in £ (£3K start, 0.01 lot, GBP account)
- Account settings in EquityCurve.tsx constants (STARTING_CAPITAL, LOT_SIZE, ACCOUNT_RATE)
- Presets: turbo (50K/stage, 10 candidates), standard (200K/stage, 20 candidates), deep (500K/stage, 30), max (1M/stage, 50)
