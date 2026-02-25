# Automated Forex Trading System

## Project Overview
Fully automated forex trading system that discovers, validates, deploys, and monitors trading strategies. See `PRD.md` for complete requirements.

## Tech Stack
- **Language**: Python 3.12
- **Hot Loop**: Rust (PyO3 + Rayon) via `backtester_core` native extension — replaces Numba JIT
- **Fallback**: Numba @njit(parallel=True) + prange (auto-fallback if Rust not built)
- **Threading Backend**: TBB for Numba fallback — MUST be installed on Windows
- **Data**: numpy arrays, Parquet storage, Dukascopy historical data
- **Parallelism**: Rayon (Rust) for hot loop, Numba prange as fallback
- **Broker**: IC Markets Global (MT5) — offshore branch (Seychelles FSA), 1:500 leverage
- **Execution**: MetaTrader 5 Python library (`MetaTrader5` package)
- **Historical Data**: Dukascopy (free tick/M1 data, 15+ years) via `dukascopy-python`
- **Data Location**: `G:\My Drive\BackTestData` (Google Drive, NOT in repo)
- **Dashboard**: React + Tailwind CSS
- **Notifications**: Telegram Bot
- **Package Manager**: uv
- **OS**: Windows 11 (dev), Linux VPS (production)

## Broker & Data Setup
- **Broker**: IC Markets Global (Raw Trading Ltd, Seychelles FSA / Bahamas SCB)
- **Leverage**: 1:500 (UK resident, offshore branch)
- **Platform**: MetaTrader 5 — demo account creds in `.env` (gitignored)
- **Historical data**: Dukascopy M1 data (15+ years), stored as Parquet on `G:\My Drive\BackTestData`
- **Naming**: `{PAIR}_{TIMEFRAME}.parquet` + yearly chunks in `{PAIR}_{TIMEFRAME}_chunks/`
- **Target instruments**: EURUSD, GBPUSD, USDJPY, XAUUSD (expanding to 20+ pairs)
- **Deployment**: Dev/test locally on Windows 11 → deploy to VPS for 24/7 live trading
- **MT5 config**: Set "Max Bars in Chart" to 10,000,000+ (Tools > Options > Charts)
- **MT5 server timezone**: UTC+2 (all MT5 timestamps must subtract 2h to get UTC)
- **Data source parity**: Dukascopy vs IC Markets M1 prices differ by ~0.1-0.3 pip median, 99%+ within 1 pip, correlation 0.9999+. Backtest-live verification threshold of 0.5 pip (REQ-V07) is achievable.

## Architecture Principles
- All backtest-critical code uses Rust (PyO3 + Rayon) for the hot loop, with Numba as fallback
- Primary: Rust `batch_evaluate()` via `backtester_core` extension — zero-copy numpy arrays, Rayon parallel iter, GIL released
- Fallback: Numba @njit(parallel=True) + prange (auto-detected if Rust not built)
- Rust uses system allocator with deterministic deallocation — no NRT memory pooling, no segfaults with M1 data
- CRITICAL: Pre-allocate ALL output arrays (metrics_out, pnl_buffers) in Python before passing to Rust/Numba
- Precompute-Once, Filter-Many: indicators computed once, parameter filtering is cheap per-trial
- **Optimizer Design**: Sobol exploration + EDA exploitation + MAP-Elites diversity archive. Staged optimization is the biggest lever (reduces search space exponentially per stage). Optuna/TPE is NOT used in the hot loop (suggestion overhead dominates at sub-ms eval times per research). Optuna kept as dependency for future expensive-objective use (walk-forward, Monte Carlo)
- **Batch-first principle**: optimizer generates N param sets → engine evaluates all N via prange → optimizer updates once per batch. Never single-trial evaluation in the hot loop
- **DSR overfitting gate**: Deflated Sharpe Ratio for multiple-testing correction. Forward/back quality ratio >= 0.4 as promotion gate
- Broker integration is an abstraction layer — swappable without touching strategy/pipeline code (currently IC Markets via MT5)
- Backtest and live trader use identical signal generation and trade management logic (position_manager.py mirrors backtest check order)
- State files will use atomic writes (write temp, then rename) — currently only in data downloader
- Long-running processes checkpoint to disk for crash recovery (pipeline checkpoints, live trader state.json)
- Rust hot loop implemented (PyO3 + Rayon) — Numba kept as fallback

## Staged Optimization
- Stage order is **strategy-defined** via `optimization_stages()`, NOT hard-coded
- Default order: signal → time → risk → management → refinement
- Custom strategies override with their own group order
- Engine switches from Basic mode (SL/TP only) to Full mode at management stage
- Each stage locks best params before advancing to next stage
- Refinement stage: all params active with narrow range around locked best values

## Extensibility Patterns
- **New exit mechanism**: pre-compute signal attrs + add exit code in `dtypes.py` + add if-block in `_simulate_trade_full()` + add `ParamDef` to `management_params()`
- **New SL/TP mode**: add constant in `dtypes.py` + add elif in `_compute_sl_tp()` + add value to `risk_params()`
- **New indicator**: add numpy function to `indicators.py` + use in strategy's `generate_signals()`
- **New fitness function**: add numeric code to `dtypes.py` + elif in metrics section
- **New param group**: add `group="my_group"` to `ParamDef` + add group to strategy's `optimization_stages()` list
- **New strategy**: subclass `Strategy`, implement `generate_signals/filter_signals/calc_sl_tp`, register with `@register`
- Principle: the framework adapts to strategy needs, not the other way around

## Key Interfaces
- `BacktestEngine.evaluate_batch(param_matrix, exec_mode) -> metrics_matrix` — (N,P) → (N,10), exec_mode=EXEC_BASIC or EXEC_FULL
- `BacktestEngine.evaluate_single(params_dict) -> metrics_dict` — convenience wrapper
- `StagedOptimizer` reads strategy's `optimization_stages()` dynamically
- `EncodingSpec` bridges Python ParamSpace ↔ JIT float64 arrays (categoricals as indices, booleans as 0/1, lists as bitmasks)

## Rust Backend Build (backtester_core)
The hot loop is a Rust native extension via PyO3/maturin. Build once:
```bash
cd rust && bash build.sh          # Windows (Git Bash)
cd rust && maturin develop --release   # Linux/macOS
```
Prerequisites: Rust toolchain (`rustup`), MSVC Build Tools 2022 (Windows), `uv pip install maturin`

Backend control: `BACKTESTER_BACKEND=rust|numba|auto` (default: auto — Rust if built, else Numba)

## Windows-Specific Requirements
- Install TBB: `pip install tbb` (or via conda) — for Numba fallback
- Set environment variable: NUMBA_THREADING_LAYER=tbb — for Numba fallback
- Verify threading layer: `numba -s` (must show TBB, not workqueue)
- Without TBB, Numba silently falls back to SINGLE-THREADED on Windows

## Hardware Target (i9-14900HX)
- 24 physical cores (8P + 16E), 32 logical threads, 64 GB RAM
- Realistic parallel scaling: 16-20x on 24 cores (memory bandwidth limited)
- Measured throughput: 18K-35K evals/sec BASIC, 4K-10K evals/sec FULL (EUR/USD H1, 96K bars)

## Project Structure
```
rust/                              # Rust native extension (PyO3 + Rayon)
  src/                             #   lib.rs, constants.rs, filter.rs, sl_tp.rs, trade_basic.rs, trade_full.rs, metrics.rs
  Cargo.toml                       #   pyo3 + numpy + rayon deps
  pyproject.toml                   #   maturin build config
  build.sh                         #   Windows build script (sets MSVC env)
backtester/
  core/           # Backtest engine, metrics, trade simulation (Rust + Numba fallback), telemetry
  data/           # Data download, caching, validation, timeframe conversion
  strategies/     # Strategy framework, base classes, indicator library
  optimizer/      # Parameter optimization, staged search
  pipeline/       # 7-stage validation pipeline (walk-forward, CPCV, MC, stability, confidence, regime)
  live/           # Live trading engine, position management, state persistence
  risk/           # Risk management, position sizing, circuit breakers
  broker/         # Broker abstraction layer (IC Markets / MT5 orders + data)
  verification/   # Trade verification, signal replay (not yet built)
  reporting/      # HTML report generation, leaderboard (not yet built)
  research/       # Strategy research factory, source tracker (not yet built)
  notifications/  # Telegram bot integration (not yet built)
  config/         # Configuration management, defaults
  cli/            # CLI entry points
scripts/          # Operational scripts (download, deploy, live trading, benchmarks)
Research/         # Strategy research papers and documents
tests/            # Unit and integration tests (533 tests)
DEPLOY.bat        # One-click VPS deployment (git pull + start traders)
STATUS.bat        # Check running trader status
STOP.bat          # Stop all running traders
```

## Conventions
- Use `uv` for dependency management
- Type hints on all public functions
- Docstrings only where logic is non-obvious
- Tests for all core logic (backtest engine, metrics, indicators)
- Structured logging with timestamps
- Config hierarchy: defaults -> config file -> CLI args -> env vars

## Key Files
- `PRD.md` — Full product requirements document
- `PROGRESS.md` — Requirement-level progress tracking
- `CURRENT_TASK.md` — What to do next (exact steps, blockers, last completed)
- `pyproject.toml` — Python project config and dependencies
- `.env` — Credentials (gitignored, NEVER committed)

## Session Continuity Protocol
On every new session or after /clear:
1. Read `CURRENT_TASK.md` — pick up exactly where we left off
2. Read `PROGRESS.md` — see overall phase status
3. Check `git log --oneline -10` — see recent work
4. Resume building — no re-explaining unless asked

Before context gets long or session ends:
1. Commit all working code to git with descriptive message
2. Update `PROGRESS.md` with newly completed requirement IDs
3. Update `CURRENT_TASK.md` with exact next steps and any blockers
4. Run `uv run pytest tests/ -v` to confirm everything passes

## Critical Assessment Rule — NEVER WALK PAST ANYTHING
- Before implementing ANY section of the PRD, critically assess it first. The PRD was written early and our thinking has evolved. Do NOT blindly accept requirements, numbers, or formulas
- Question every threshold, magic number, and scoring formula. If a number looks arbitrary or wrong, flag it to the user before building it
- If output looks suspicious (like a quality score of 60 on good data), investigate WHY before moving on. Assume the code is wrong before assuming the data is wrong
- If data is missing, incomplete, or unexpected, say so loudly — don't silently accept partial results
- When a validation, metric, or gate produces surprising results, dig into the breakdown. Show the raw numbers
- Treat every PRD requirement as a starting point, not gospel. Challenge anything that doesn't make sense in context
- This applies to everything: scoring formulas, thresholds, architectural decisions, data assumptions, test results

## System Correctness First
- NEVER optimize for positive backtest results. Focus on system correctness and long-term robustness
- Negative results are fine — they represent truth. A system that correctly reports "no edge" is better than one that fabricates green lights
- When investigating poor results, fix the SYSTEM not the RESULTS
- Always question whether backtesting behavior matches live trading reality

## Phase 5b: Validation & Optimizer Enhancements — COMPLETE (except OPT-2, OPT-3)

### Completed
- **VP-1: CPCV** — Combinatorial Purged Cross-Validation. 45 folds, integrated into confidence scoring. `pipeline/cpcv.py`
- **VP-2: Multi-Candidate Pipeline** — Optimizer returns top N, pipeline validates all. `optimizer/run.py` + `pipeline/runner.py`
- **VP-3: Regime-Aware Validation** — ADX+NATR 4-quadrant classification, per-regime stats. `pipeline/regime.py`
- **OPT-1: CE Exploitation Upgrade** — Adaptive LR + entropy monitoring. Pairwise dependencies descoped. `optimizer/sampler.py`
- **Causality Contract** — SignalCausality enum, pipeline/engine guards, verification tests
- **Rust Backend** — PyO3 + Rayon replaces Numba, 2.7x faster EXEC_FULL, M1 stable, subprocess isolation removed
- **Telemetry Accuracy** — 5 deferred-SL bugs fixed, 11 parity tests prevent divergence

### Remaining
- **OPT-2: GT-Score Objective** — A/B test vs Quality Score on FX data (MEDIUM)
- **OPT-3: Batch Size Auto-Tuning** — Benchmark and auto-select optimal batch size (LOW)

## Phase 6: Live Trading — BUILT & DEPLOYED

Live trading engine, risk management, broker integration, and deployment scripts are complete.
Practice deployment running on IC Markets demo account.

### Key Files
- `backtester/live/trader.py` — Candle-close event loop, order placement, broker sync
- `backtester/live/position_manager.py` — Mirrors backtest management (trailing, BE, partial, stale, max bars)
- `backtester/risk/manager.py` — Pre-trade checks, position sizing, circuit breaker
- `backtester/broker/mt5_orders.py` — MT5 order execution (market, modify, close, partial)
- `scripts/live_trade.py` — CLI entry point (dry_run/practice/live)
- `scripts/start_all.py` — Auto-discover strategies, launch as detached processes
- `DEPLOY.bat` — One-click VPS deploy (git pull, deps, start)

### NOT building now
- NSGA-II/GA rewrite, island model, White's RC/Hansen SPA, full HMM, multi-fidelity, factor attribution, HTML reports — all deferred to later phases

## Development Workflow
- Work in small, testable increments (one FR sub-section at a time)
- Run tests after each increment
- Commit after each logical chunk of work
- Always leave the codebase in a passing-tests state
