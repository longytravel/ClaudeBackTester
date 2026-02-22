# Automated Forex Trading System

## Project Overview
Fully automated forex trading system that discovers, validates, deploys, and monitors trading strategies. See `PRD.md` for complete requirements.

## Tech Stack
- **Language**: Python 3.12
- **JIT Compilation**: Numba @njit(parallel=True) + prange for backtest hot loops
- **Threading Backend**: TBB (Intel Threading Building Blocks) — MUST be installed on Windows
- **Data**: numpy arrays, Parquet storage, Dukascopy historical data
- **Parallelism**: Numba prange (NOT ThreadPoolExecutor, NOT multiprocessing)
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

## Architecture Principles
- All backtest-critical code uses numpy arrays and Numba @njit(parallel=True) — no pandas/Python objects in hot paths
- Parallelism via Numba prange: outer loop across trials is parallel, inner bar-by-bar loop is sequential per trial. Numba manages the thread pool internally via TBB
- CRITICAL: Zero allocation inside prange loops. No np.empty(), no .append(), no NumPy function calls in the hot path. Pre-allocate ALL output arrays before the parallel region. Violating this causes worse-than-single-threaded performance (Numba issue #8686)
- JIT compiles ONCE, all threads share compiled code natively
- Precompute-Once, Filter-Many: indicators computed once, parameter filtering is cheap per-trial
- Staged optimization uses random search (zero overhead, <=12 params per stage)
- Full-space uses BATCHED Bayesian/Optuna TPE (13+ params) — ask() batch of trials, evaluate via prange, tell() results back
- Auto-selects search strategy: <=12 params -> random, 13+ params -> batched TPE. CLI override: --search random|tpe|cma-es
- Broker integration is an abstraction layer — swappable without touching strategy/pipeline code (currently IC Markets via MT5)
- Backtest and live trader use identical signal generation and trade management logic
- All state files use atomic writes (write temp, then rename)
- All long-running processes checkpoint to disk for crash recovery
- Upgrade path: Rust (PyO3) + Rayon for hot loop if more speed needed later

## Windows-Specific Requirements
- Install TBB: `pip install tbb` (or via conda)
- Set environment variable: NUMBA_THREADING_LAYER=tbb
- Verify threading layer: `numba -s` (must show TBB, not workqueue)
- Without TBB, Numba silently falls back to SINGLE-THREADED on Windows

## Hardware Target (i9-14900HX)
- 24 physical cores (8P + 16E), 32 logical threads, 64 GB RAM
- Realistic parallel scaling: 16-20x on 24 cores (memory bandwidth limited)
- Estimated throughput: 600-800 evals/sec (random), 400-600 evals/sec (batched TPE)

## Project Structure
```
backtester/
  core/           # Backtest engine, metrics, trade simulation (Numba JIT)
  data/           # Data download, caching, validation, timeframe conversion
  strategies/     # Strategy framework, base classes, indicator library
  optimizer/      # Parameter optimization, staged search
  pipeline/       # 7-stage validation pipeline orchestration
  live/           # Live trading engine, position management, broker sync
  risk/           # Risk management, position sizing, circuit breakers
  verification/   # Trade verification, signal replay, P&L reconciliation
  reporting/      # HTML report generation, leaderboard
  research/       # Strategy research factory, source tracker
  broker/         # Broker abstraction layer (IC Markets / MT5 implementation)
  notifications/  # Telegram bot integration
  config/         # Configuration management, defaults
  cli/            # CLI entry points
dashboard/        # React + Tailwind frontend (separate)
output/           # Pipeline run outputs — gitignored
state/            # Live trading state files — gitignored
logs/             # Log files — gitignored
tests/            # Unit and integration tests
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

## Development Workflow
- Work in small, testable increments (one FR sub-section at a time)
- Run tests after each increment
- Commit after each logical chunk of work
- Always leave the codebase in a passing-tests state
