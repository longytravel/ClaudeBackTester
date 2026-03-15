# E2E Pipeline Audit Report

**Date**: 2026-03-15
**Objective**: Test the full pipeline end-to-end, from data loading to VPS deployment, focusing on system correctness — NOT profitability.

---

## Executive Summary

The pipeline works end-to-end. Strategies go from raw data through optimization, 7-stage validation, report generation, and deployment to IC Markets demo account on VPS. Total backtest time per strategy: **~1-5 minutes** (turbo preset, EUR/USD H1).

**Three critical bugs were found and fixed during this audit.** The most significant was that `.env` credentials were never loaded — traders connected to whichever MT5 account they found first instead of the configured one. This explains why previous VPS deployments appeared to produce no trades when checked from the correct account.

## Scorecard

| Area | Status | Grade |
|------|--------|-------|
| Data loading & splitting | Works correctly | A |
| Optimization (staged) | Works, produces candidates | B |
| Validation pipeline (WF/MC/stability/confidence) | All stages work correctly | B+ |
| Report JSON output | Fixed — now includes metadata + confidence breakdown | B+ |
| Deployment (dry_run) | Works perfectly | A |
| Deployment (practice/VPS) | Fixed — now connects to correct account | A |
| DEPLOY.bat one-click | Fixed — stops old traders, cleans state, deploys | A |
| Console output clarity | Too noisy, key info buried | C |
| Dashboard | Exists but not fully tested | C+ |

---

## Bugs Found and Fixed

### CRITICAL: .env credentials never loaded (FIXED)
- **File**: `scripts/live_trade.py`, `scripts/start_all.py`
- **Bug**: Neither script called `load_dotenv()`, so `MT5_LOGIN` from `.env` was never read. `expected_login` was always 0, so the account verification check was skipped. Traders connected to whichever MT5 terminal was found first.
- **Impact**: VPS had two MT5 terminals (accounts 648 and 831). Traders always connected to 831 while user monitored 648. All previous VPS deployments were trading on the wrong account.
- **Fix**: Added `load_dotenv()` to both scripts, added `python-dotenv` to project dependencies.
- **Commit**: `a47ab65`

### HIGH: No rating/elimination warning on deployment (FIXED)
- **File**: `scripts/live_trade.py`
- **Bug**: `live_trade.py` deployed any candidate without checking its pipeline rating or elimination status. A user could accidentally deploy a candidate that failed all validation gates.
- **Fix**: Added pre-deployment check that reads checkpoint, shows elimination reason and failed gates. Blocks LIVE mode for eliminated candidates. Warns on practice mode.
- **Commit**: `9f9b01d`

### HIGH: Report.json missing critical metadata (FIXED)
- **File**: `backtester/pipeline/runner.py`
- **Bug**: Report had no run date, no data summary, no timing info, no preset used, no confidence component breakdown. User couldn't tell when a run happened or reproduce it.
- **Fix**: Added `run_date`, `verdict` (quick summary), `data_summary` (bar counts), `run_config` (preset, trials, timing, costs), and `confidence_breakdown` (all 6 component scores per candidate).
- **Commit**: `9f9b01d`

### MEDIUM: DEPLOY.bat didn't clean old state (FIXED)
- **File**: `DEPLOY.bat`
- **Bug**: Old trader state directories persisted across deploys. Stale heartbeats and state files confused STATUS.bat.
- **Fix**: DEPLOY.bat now runs `stop_all.py`, cleans `state/` directory, and uses `--testing` flag.
- **Commit**: `cda0c0d`

---

## What Was Tested

### Pipeline Run: 3 strategies on EUR/USD H1 (turbo preset)

| Strategy | Candidates | Trades/candidate | Backtest time |
|----------|-----------|-----------------|--------------|
| stochastic_crossover | 8 | 383-619 | ~18 min (incl. pipeline) |
| macd_crossover | 8 | 403-427 | ~5 min |
| ema_crossover | 2 | 688 | ~2 min |

### Deployment Verified
- **Dry-run**: Params loaded from checkpoint, trader started, waited for bar ✓
- **Practice (local)**: MT5 connected to account 648, params loaded ✓
- **Practice (VPS)**: DEPLOY.bat pulled code, started 3 traders on account 648 ✓

### Full Flow Proven
```
Data (Dukascopy parquet) → Optimization (Sobol+EDA, Rust engine)
→ Validation (7 stages) → Report (JSON) → Deploy (VPS, MT5 practice)
```

---

## Remaining Issues (Not Fixed Yet)

### HIGH
| # | Issue | Location |
|---|-------|----------|
| 1 | Noisy `strategy_registered` debug logs on every command | `backtester/strategies/__init__.py` |
| 2 | No "next steps" recommendation in console after pipeline run | `scripts/full_run.py` |

### MEDIUM
| # | Issue | Location |
|---|-------|----------|
| 3 | Forward gate explanation too terse for non-technical user | `scripts/full_run.py` |
| 4 | Stability min_ratio=0.0 should maybe downgrade to FRAGILE | `backtester/pipeline/stability.py` |
| 5 | RunHistory.tsx is a stub — can't browse past runs | `dashboard/src/pages/RunHistory.tsx` |
| 6 | No run comparison view in dashboard | Dashboard |
| 7 | WF pass_rate/mean_sharpe missing from report for eliminated candidates | `backtester/pipeline/runner.py` |

### LOW
| # | Issue | Location |
|---|-------|----------|
| 8 | Stage quality progression not recorded in report | Report generation |
| 9 | Perturbation detail (which step caused quality=0) not in report | Stability |
| 10 | Regime stats hard to interpret without context | Report/dashboard |

---

## Deployment Instructions

### VPS (one-click)
Double-click `DEPLOY.bat` on VPS. It handles everything: stop old traders → git pull → clean state → install deps → launch strategies.

### Local testing
```bash
# Run a backtest
uv run python scripts/full_run.py --strategy stochastic_crossover --pair "EUR/USD" --timeframe H1 --preset turbo

# Deploy to practice (local MT5)
uv run python scripts/live_trade.py --strategy stochastic_crossover --pair EURUSD --timeframe H1 --pipeline results/stochastic_eur_usd_h1/checkpoint.json --mode practice

# Check VPS status
# Double-click STATUS.bat on VPS
```

---

## What's Next

1. **Wait for market open** (Sunday evening) — traders will start generating signals
2. **Compare backtest vs live trades** — check MT5 history against backtest predictions
3. **Build backtest-to-live comparison tool** — automated matching of live trades to backtest expectations
4. **Address remaining issues** from the list above
5. **Test dashboard** with `--dashboard` flag during a live run
