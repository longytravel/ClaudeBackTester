# Automated Forex Trading System

## Project Overview
Fully automated forex trading system that discovers, validates, deploys, and monitors trading strategies. See `PRD.md` for complete requirements.

## Tech Stack
- **Language**: Python 3.12
- **JIT Compilation**: Numba @njit(parallel=True) + prange for backtest hot loops
- **Threading Backend**: TBB (Intel Threading Building Blocks) — MUST be installed on Windows
- **Data**: numpy arrays, Parquet storage
- **Parallelism**: Numba prange (NOT ThreadPoolExecutor, NOT multiprocessing)
- **Broker**: OANDA (REST API via oandapyV20 or tpqoa)
- **Dashboard**: React + Tailwind CSS
- **Notifications**: Telegram Bot
- **Package Manager**: uv
- **OS**: Windows 11

## Architecture Principles
- All backtest-critical code uses numpy arrays and Numba @njit(parallel=True) — no pandas/Python objects in hot paths
- Parallelism via Numba prange: outer loop across trials is parallel, inner bar-by-bar loop is sequential per trial. Numba manages the thread pool internally via TBB
- CRITICAL: Zero allocation inside prange loops. No np.empty(), no .append(), no NumPy function calls in the hot path. Pre-allocate ALL output arrays before the parallel region. Violating this causes worse-than-single-threaded performance (Numba issue #8686)
- JIT compiles ONCE, all threads share compiled code natively
- Precompute-Once, Filter-Many: indicators computed once, parameter filtering is cheap per-trial
- Staged optimization uses random search (zero overhead, <=12 params per stage)
- Full-space uses BATCHED Bayesian/Optuna TPE (13+ params) — ask() batch of trials, evaluate via prange, tell() results back
- Auto-selects search strategy: <=12 params -> random, 13+ params -> batched TPE. CLI override: --search random|tpe|cma-es
- Broker integration is an abstraction layer — swappable without touching strategy/pipeline code
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
  broker/         # Broker abstraction layer (OANDA implementation)
  notifications/  # Telegram bot integration
  config/         # Configuration management, defaults
  cli/            # CLI entry points
dashboard/        # React + Tailwind frontend (separate)
data_cache/       # Downloaded price data (Parquet files) — gitignored
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
- `pyproject.toml` — Python project config and dependencies

## Development Workflow
- Work in small, testable increments (one FR sub-section at a time)
- Run tests after each increment
- Update PROGRESS.md after completing requirements
