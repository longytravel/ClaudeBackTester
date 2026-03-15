# E2E Pipeline Audit Report

**Date**: 2026-03-15
**Run**: `rsi_mean_reversion` on EUR/USD H1 (turbo preset)
**Objective**: Test the full pipeline end-to-end, from data loading to practice deployment, focusing on system correctness and usability — NOT profitability.

---

## Executive Summary

The pipeline **works end-to-end**. A strategy can go from raw data through optimization, 7-stage validation, report generation, and deployment to an IC Markets demo account. Total time: **54 seconds** (turbo preset).

However, the system has significant **usability and safety gaps** that would make it difficult for a user to understand results, repeat the process confidently, or avoid deploying bad strategies. This report catalogs every issue found and recommends fixes.

### Verdict: Pipeline Mechanically Sound, UX Needs Work

| Area | Status | Grade |
|------|--------|-------|
| Data loading & splitting | Works correctly | A |
| Optimization (staged) | Works, produces candidates | B |
| Forward gate | Works correctly (rejects overfitted) | A |
| Walk-forward validation | Works, runs advisory on eliminated | B+ |
| Stability analysis | Works, all MODERATE | B+ |
| Monte Carlo + regime | Works correctly | B+ |
| Confidence scoring | Works, rates RED correctly | A |
| Report JSON output | Missing critical fields | D |
| Console output clarity | Too noisy, key info buried | C |
| Dashboard | Exists, dual-mode, but not tested end-to-end | C+ |
| Deployment (dry_run) | Works perfectly | A |
| Deployment (practice) | Works, MT5 connects, params load | A |
| Safety gates on deploy | MISSING — RED candidates deploy silently | F |

---

## Stage-by-Stage Analysis

### Stage 0: Data Loading

**What happened**: Loaded 96,262 H1 bars (2007-2026), 5.7M M1 sub-bars. 80/20 split at 2023-01-18. Zero NaN close/spread.

**Issues**:
- None. Data loading is solid.

**Recommendation**: None needed.

---

### Stage 1-2: Optimization

**What happened**: 225,920 trials across 5 stages in 40.7 seconds (5,557 evals/sec). 12 candidates selected from 36,134 passing refinement trials.

**Stage breakdown**:
| Stage | Combos | Budget | Valid | Best Quality |
|-------|--------|--------|-------|-------------|
| Signal | 288 | 2,880 | 6 (0.2%) | 1.28 |
| Time | 2,304 | 23,040 | 2,013 | 2.59 |
| Risk | 1.34M | 50,000 | 27,472 | 2.85 |
| Management | 189M | 50,000 | 38,371 | 2.85 |
| Refinement | 425T | 100,000 | 36,134 | 4.05 |

**Issues found**:

1. **Signal stage: only 6/288 valid combos (2.1%)** — The RSI strategy has very few parameter combinations that produce enough trades with positive quality. This is concerning — either the strategy is genuinely weak or the signal generation is too restrictive.

2. **Management stage adds zero value**: Quality stayed at 2.85 from risk→management. Trailing stops, breakeven, partial close all found to be unhelpful. This is expected for a mean reversion strategy but the user has no visibility into WHY management didn't help.

3. **Budget capping silently**: Signal and time stages are capped to 10x combos ("budget capped 50,000 -> 2,880"). This is good for efficiency but not communicated to the user.

4. **Refinement quality jump**: 2.85 → 4.05. The refinement stage nearly doubled quality. This is suspicious — refinement should fine-tune, not transform. Suggests earlier stages locked suboptimal params that refinement partially rescued.

**Recommendations**:
- Add a "stage summary" to the report showing per-stage quality progression
- Flag when management stage adds zero improvement (suggests management features unnecessary)
- Investigate why refinement shows such a large quality jump

---

### Forward Gate

**What happened**: ALL 12 candidates failed. Best forward/back ratio: 0.021 (threshold: 0.4).

| Candidate | Back Quality | Forward Quality | Ratio | Verdict |
|-----------|-------------|----------------|-------|---------|
| 0 | 4.05 | 0.05 | 0.014 | FAIL |
| 1 | 3.28 | 0.07 | 0.021 | FAIL |
| 2 | 3.71 | 0.05 | 0.013 | FAIL |
| 3 | 3.89 | 0.03 | 0.008 | FAIL |
| 4-5 | 2.7-3.3 | 0.03-0.04 | 0.01 | FAIL |
| 6-11 | 1.8-3.3 | 0.00-0.01 | 0.000 | FAIL |

**Analysis**: The forward quality is near zero for all candidates. This means the RSI mean reversion strategy on EUR/USD H1 is **completely overfitted** — it finds patterns in historical data that don't persist into the 2023-2026 forward period. This is the system working correctly.

**Issues**:
- The console output says "ELIMINATED candidate 0 at forward_gate" but doesn't explain what the gate IS or what the user should do about it. A non-technical user seeing this has no idea what's wrong.
- All 12 candidates had the same problem but each is reported individually — a summary "0/12 passed forward gate" would be clearer.

**Recommendations**:
- Add human-readable explanation: "Forward quality (0.05) is only 1.4% of back-test quality (4.05). Minimum required: 40%. Strategy is overfitted to historical data."
- Add batch summary before per-candidate details
- Suggest next steps: "Try different preset, different pair, or different strategy"

---

### Stage 3: Walk-Forward (Advisory)

**What happened**: All candidates were already eliminated at forward gate. Walk-forward ran in advisory mode. All failed both gates (pass_rate and mean_sharpe).

**Issues**:
- Walk-forward results are in the checkpoint but NOT clearly shown in report.json (wf_pass_rate and wf_mean_sharpe are undefined in report for candidate 0)
- Running WF on already-eliminated candidates is good for diagnostics but adds ~3s of compute time

**Recommendations**:
- Ensure wf_pass_rate and wf_mean_sharpe appear in report.json even for eliminated candidates
- Label advisory-mode results clearly in the output

---

### Stage 4: Stability (Advisory)

**What happened**: All candidates rated MODERATE (mean_ratio 0.74-0.84). Worst parameters: rsi_period, rsi_oversold, rsi_overbought.

**Issues**:
- All `min_ratio=0.000` means at least one perturbation step produced quality=0 for every candidate. This is fragile despite MODERATE overall rating.
- The MODERATE rating feels too lenient when min_ratio is 0.0

**Recommendations**:
- Consider downgrading to FRAGILE when min_ratio < 0.1 regardless of mean
- Add specific perturbation details to report (which param at which step caused quality=0?)

---

### Stage 5: Monte Carlo + Regime

**What happened**: DSR: 0.456 (need >= 0.95). Permutation p-value: 0.000 (passes). Regime analysis ran.

**Issues**:
- The permutation_p=0 is suspicious — this means 0 out of 1000 random shuffles beat the real strategy. For a strategy with near-zero forward quality, this seems contradictory. The MC test is run on FULL data (back+forward) while forward gate is forward-only, which could explain the discrepancy.
- Regime results are in the report but hard to interpret without context

**Recommendations**:
- Clarify which data MC tests are run on (full? back only? forward only?)
- Add interpretation guide for regime stats

---

### Stage 6: Confidence Scoring

**What happened**:
| Component | Score | Weight |
|-----------|-------|--------|
| Walk-forward | 0.0 | 0.30 |
| Monte Carlo | 56.5 | 0.25 |
| Forward/Back | 0.0 | 0.15 |
| Stability | 46.0 | 0.10 |
| DSR | 0.0 | 0.10 |
| Backtest Quality | 44.8 | 0.10 |
| **Composite** | **23.2** | |
| **Rating** | **RED** | |

**Issues**:
- Score breakdown is shown per-candidate in console but NOT in report.json
- The composite score calculation is opaque — where does Monte Carlo=56.5 come from when DSR=0.456 and the gate failed?

**Recommendations**:
- Include component scores in report.json (currently only composite_score and rating)
- Add explanation of how each component score is computed

---

### Stage 7: Report Generation

**What happened**: report.json and checkpoint.json saved to results/e2e_audit_rsi_h1/

**CRITICAL issues with report.json**:

| Missing Field | Why It Matters |
|---------------|---------------|
| Run date/timestamp | When was this run? Can't compare runs over time |
| Data summary (bars, date range) | How much data was used? |
| Total trials count (top-level) | How thorough was the optimization? |
| Timing info (elapsed, evals/sec) | How long did it take? |
| Preset used | What settings were used? Can the run be reproduced? |
| Per-stage quality progression | How did optimization progress? |
| Recommendations / next steps | What should the user do now? |
| Confidence component scores | Only composite shown, not breakdown |
| WF pass_rate / mean_sharpe | Missing from report for some candidates |

**What IS in the report** (good):
- Strategy name and version
- All 12 candidates with params, back/forward quality, elimination info
- Trade stats (530 trades, 52.3% win rate, PF 1.6)
- Exit breakdown (TP/SL/max_bars counts)
- Direction stats (BUY/SELL split)
- Equity curve (531 data points)
- Regime distribution and per-regime stats
- Gates passed/failed per candidate
- Optimizer funnel (trials → refinement → diverse → forward → pipeline)

---

## Deployment Audit

### Dry-Run Deployment
- **Status**: WORKS
- Params loaded from checkpoint correctly
- No MT5 needed
- Waits for next bar close, then would simulate a cycle

### Practice Deployment (IC Markets Demo)
- **Status**: WORKS
- MT5 connected: Raw Trading Ltd, ICMarketsSC-Demo, login 52754648
- Balance: $2,975.31
- Params loaded correctly from checkpoint
- Waiting for next H1 bar close

### CRITICAL SAFETY ISSUE: No Rating Gate on Deployment

`live_trade.py` **does not check** the candidate's rating or elimination status before deploying. A RED-rated, eliminated candidate deploys silently to a real broker account.

- `start_all.py` HAS rating checks (with `--testing` flag to bypass)
- `live_trade.py` has ZERO validation — loads params, starts trading
- No warning printed about RED rating
- No warning about candidate being eliminated
- No confirmation prompt for practice/live modes

**This is the #1 fix needed.** A user could accidentally deploy a demonstrably bad strategy to a live funded account.

---

## Console Output Usability

### Issues:
1. **Noisy debug logs**: 9 "strategy_registered" debug lines print on EVERY command (full_run, live_trade, etc.). These are useless to the user.
2. **Key info buried**: The ELIMINATED lines and VERDICT are the most important output but they're buried after pages of optimizer progress logs.
3. **No clear "here's what you should do next"**: The VERDICT says RED but doesn't tell the user what to try differently.
4. **Startup banner is good**: The "Starting live trader" block with strategy/pair/mode is clear and useful.

### Recommendations:
- Suppress strategy_registered debug logs (or gate behind --verbose)
- Add a clear "NEXT STEPS" section after the verdict
- Move verdict/summary to the TOP of the final output (before per-candidate details)

---

## Dashboard Audit

### Architecture (Good):
- Dual-mode: live WebSocket during runs + REST API for historical results
- `/api/runs` endpoint scans results/ directory for past runs
- `/api/runs/{run_id}/report` serves individual reports
- Zustand store manages state transitions (idle → connecting → running → complete)

### Components:
- CandidateTable.tsx — shows candidate grid
- EquityCurve.tsx — lightweight-charts equity display
- PipelineFunnel.tsx — stage progression visualization
- RunSummary.tsx — overview of run
- RunHistory.tsx — past runs (stub)

### Issues Found:
1. **RunHistory.tsx is a stub** — can't browse past runs from the UI yet
2. **Dashboard requires server running** — no static report viewer
3. **Not tested with current run** — would need to re-run with `--dashboard` flag
4. **No way to compare runs** — can't put two results side by side

### Recommendations:
- Build out RunHistory page to browse results/ directory
- Add "open report" button that loads any report.json
- Add run comparison view (side-by-side candidates)

---

## Summary of All Issues (Prioritized)

### CRITICAL (must fix)
| # | Issue | Location |
|---|-------|----------|
| 1 | No rating/elimination check before deployment | `scripts/live_trade.py` |
| 2 | Report.json missing run metadata (date, timing, preset, data summary) | `backtester/pipeline/runner.py` report stage |
| 3 | Confidence component scores not in report.json | `backtester/pipeline/runner.py` report stage |

### HIGH (should fix)
| # | Issue | Location |
|---|-------|----------|
| 4 | WF pass_rate/mean_sharpe missing from report for some candidates | `backtester/pipeline/runner.py` |
| 5 | No "next steps" recommendation in console output | `scripts/full_run.py` |
| 6 | Noisy strategy_registered debug logs on every command | `backtester/strategies/__init__.py` |

### MEDIUM (improve UX)
| # | Issue | Location |
|---|-------|----------|
| 7 | Forward gate explanation too terse for non-technical user | `scripts/full_run.py` |
| 8 | Stability min_ratio=0.0 should maybe downgrade to FRAGILE | `backtester/pipeline/stability.py` |
| 9 | Monte Carlo data scope unclear (full vs forward) | `backtester/pipeline/runner.py` |
| 10 | RunHistory.tsx is a stub | `dashboard/src/pages/RunHistory.tsx` |
| 11 | Management stage zero-improvement not flagged | `scripts/full_run.py` |
| 12 | No run comparison in dashboard | Dashboard |

### LOW (nice to have)
| # | Issue | Location |
|---|-------|----------|
| 13 | Stage quality progression not in report | Report generation |
| 14 | Perturbation detail (which step caused quality=0) not recorded | Stability |
| 15 | Regime stats hard to interpret without guide | Report/dashboard |

---

## Reproduction Steps

To repeat this exact E2E test:
```bash
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester

# 1. Run pipeline (54 seconds)
uv run python scripts/full_run.py \
  --strategy rsi_mean_reversion \
  --pair "EUR/USD" \
  --timeframe H1 \
  --preset turbo \
  --output results/e2e_audit_rsi_h1

# 2. Check results
cat results/e2e_audit_rsi_h1/report.json | python -m json.tool

# 3. Deploy to practice (requires MT5 running)
uv run python scripts/live_trade.py \
  --strategy rsi_mean_reversion \
  --pair EURUSD \
  --timeframe H1 \
  --pipeline results/e2e_audit_rsi_h1/checkpoint.json \
  --candidate 0 \
  --mode practice

# 4. Deploy dry-run (no MT5 needed)
# Same as above but --mode dry_run
```

---

## Next Steps

1. **Fix critical issues #1-3** (this session)
2. **Fix high issues #4-6** (this session)
3. **Re-run pipeline** to verify fixes
4. **Test dashboard** with `--dashboard` flag
5. **Try different strategy/pair** to get a GREEN/YELLOW candidate through the full flow
