# Progress Tracker

## Legend
- [ ] Not started
- [~] In progress
- [x] Complete
- [!] Blocked / needs decision

---

## Phase 0: Project Setup
- [x] PRD finalized
- [x] Tech stack decisions (Python, IC Markets MT5, React+Tailwind, Telegram, uv)
- [x] CLAUDE.md created
- [x] PROGRESS.md created
- [x] Initialize git repo (+ GitHub remote: longytravel/ClaudeBackTester)
- [x] Initialize uv project (pyproject.toml, dependencies)
- [x] Create project directory structure
- [x] Set up .gitignore
- [x] Set up basic test framework (pytest — 4 smoke tests passing)
- [x] Verify Numba + TBB parallel works (11.5x speedup, ~9M evals/sec on synthetic data)
- [ ] Verify IC Markets MT5 demo account connectivity

---

## Phase 1: Historical Data Management (FR-1)

### Data Acquisition
- [x] REQ-D01: Download OHLCV data from Dukascopy (bid + ask, compute spread)
- [x] REQ-D02: Support 15+ years M1 history (2005-2026)
- [x] REQ-D03: Handle API rate limits (1s delay between bid/ask, 7 retries)
- [x] REQ-D04: Retry logic (max_retries=7 via dukascopy_python)
- [x] REQ-D05: Download progress display (structured logging per year/pair)
- [x] REQ-D06: Incremental updates (re-download current year only)

### Caching & Storage
- [x] REQ-D07: Parquet columnar storage (snappy compression)
- [x] REQ-D08: Canonical file naming ({PAIR}_{TIMEFRAME}.parquet)
- [x] REQ-D09: Crash-recovery (yearly chunks, resume skips completed years)
- [x] REQ-D10: Yearly chunk splitting for large downloads

### Timeframe Conversion
- [x] REQ-D11: M1 as single source of truth
- [x] REQ-D12: Correct OHLCV aggregation (first/max/min/last/sum + median spread)
- [x] REQ-D13: True intra-period highs/lows from M1
- [x] REQ-D14: N/A — Dukascopy always has M1

### Data Validation
- [x] REQ-D15: Gap detection (3x expected interval, weekend/holiday aware)
- [x] REQ-D16: Zero/NaN detection
- [x] REQ-D17: Anomaly detection (timeframe-aware thresholds, OHLC violations)
- [x] REQ-D18: Quality score (0-100, with yearly coverage)
- [x] REQ-D19: Reject low-quality datasets (configurable min_score)

### Data Splitting
- [x] REQ-D20: 80/20 back/forward split (chronological)
- [x] REQ-D21: Holdout mode (last N months)
- [x] REQ-D22: Both portions available downstream (dict with back/forward keys)

### Data Freshness
- [x] REQ-D23: Stale data check before pipeline (is_stale / ensure_fresh)
- [ ] REQ-D24: Recent candle fetch for live trading context

### CLI Tooling
- [x] REQ-D25: CLI commands (download, update, status, build-timeframes, validate)
- [~] REQ-D26: Background downloads (script running, no PID file yet)

---

## Phase 2: Strategy Framework (FR-2)

### Strategy Interface
- [ ] REQ-S01: Name and version identifiers
- [ ] REQ-S02: Parameter space definition
- [ ] REQ-S03: Signal generation method
- [ ] REQ-S04: Signal filtering method
- [ ] REQ-S05: SL/TP calculation method

### Performance Architecture
- [ ] REQ-S06: Precompute-once, filter-many separation
- [ ] REQ-S07: 300+ evals/sec throughput
- [ ] REQ-S08: Vectorized fast path (numpy boolean masks)

### Signal Attributes
- [ ] REQ-S09: Mandatory atr_pips attribute
- [ ] REQ-S10: Arbitrary additional attributes

### Parameter Organization
- [ ] REQ-S11: Named parameter groups
- [ ] REQ-S12: Reusable standard groups (Risk, Management, Time)
- [ ] REQ-S13: Management params default to OFF

### SL/TP Modes
- [ ] REQ-S14: Fixed, ATR-based, swing-based SL
- [ ] REQ-S15: RR ratio, ATR-based, fixed TP
- [ ] REQ-S16: ATR auto-scaling with volatility
- [ ] REQ-S17: Minimum TP >= SL constraint

### Trade Management
- [ ] REQ-S18: Trailing stop (fixed-pip + chandelier/ATR)
- [ ] REQ-S19: Breakeven lock
- [ ] REQ-S20: Partial close
- [ ] REQ-S21: Max bars exit
- [ ] REQ-S22: Stale exit
- [ ] REQ-S23: Identical management logic in backtest and live

### Indicator Library
- [ ] REQ-S24: Full indicator set (RSI, ATR, SMA, EMA, BB, Stoch, MACD, ADX, Donchian, Supertrend, Keltner, Williams %R, CCI, Swing Hi/Lo)
- [ ] REQ-S25: All indicators numpy-based

### Registration & Lifecycle
- [ ] REQ-S26: Central strategy registry
- [ ] REQ-S27: Strategy lifecycle tracking

---

## Phase 3: Backtesting Engine (FR-3)

### Execution Modes
- [ ] REQ-B01: Basic mode (SL/TP only, 700+ evals/sec)
- [ ] REQ-B02: Full mode (all management, 400+ evals/sec)
- [ ] REQ-B03: Grid mode (multiple concurrent positions)
- [ ] REQ-B04: Telemetry mode (per-trade details)

### Performance
- [ ] REQ-B05: JIT-compiled core loop (sub-ms per eval)
- [ ] REQ-B06: Parallel eval across all cores (300-500 evals/sec)
- [ ] REQ-B07: Shared memory for price arrays
- [ ] REQ-B08: Shared pre-computed signals
- [ ] REQ-B09: Batched trial submission (32 per IPC call)
- [ ] REQ-B10: Persistent reusable worker pool

### Metrics
- [ ] All 10 metrics: Trades, Win Rate, PF, Sharpe, Sortino, MaxDD%, Return%, R-squared, Ulcer Index, Quality Score

### Simulation Fidelity
- [ ] REQ-B11: Bar-level SL/TP checking (intra-bar extremes)
- [ ] REQ-B12: Conservative same-bar SL/TP tiebreak
- [ ] REQ-B13: Configurable spread and slippage
- [ ] REQ-B14: Management uses intra-bar highs/lows
- [ ] REQ-B15: Fixed lot sizing only

---

## Phase 4: Parameter Optimization (FR-4)

### Optimization Modes
- [ ] REQ-O01: Staged optimization (signal → filters → risk → management → refinement)
- [ ] REQ-O02: Full-space optimization
- [ ] REQ-O03: Configurable trial counts per stage

### Search Strategy
- [ ] REQ-O04: Zero-overhead search (random for staged, Bayesian optional)
- [ ] REQ-O05: 300-500 trials/sec throughput
- [ ] REQ-O06: Hard-cap workers (12 on Windows)

### Hard Pre-Filters
- [ ] REQ-O07: Max drawdown rejection (30%)
- [ ] REQ-O08: R-squared floor (0.5)
- [ ] REQ-O09: Minimum trades (20)
- [ ] REQ-O10: Skip invalid combinations

### Ranking & Candidates
- [ ] REQ-O11: Forward-test all valid results
- [ ] REQ-O12: Independent back/forward ranking
- [ ] REQ-O13: Combined rank with forward weight
- [ ] REQ-O14: Top N selection (default 50)
- [ ] REQ-O15: Diversity selection
- [ ] REQ-O16: Forward/back quality ratio gate

### Speed Presets & Memory
- [ ] REQ-O17: Turbo/Fast/Default presets
- [ ] REQ-O18-O21: Memory management (shared memory, compact signals, cleanup, pre-check)

---

## Phase 5: Validation Pipeline (FR-5)

### Pipeline Architecture
- [ ] REQ-P01-P06: 7-stage pipeline, timestamped dirs, checkpointing, resume, stop-after, retries

### Stage 3: Walk-Forward
- [ ] REQ-P13-P21: Rolling windows, IS/OOS marking, per-window metrics, aggregate stats, gates

### Stage 4: Stability
- [ ] REQ-P22-P25: Parameter perturbation, stability rating

### Stage 5: Monte Carlo
- [ ] REQ-P26-P32: Shuffle, bootstrap, permutation, trade-skip, PSR, parallelized

### Stage 6: Confidence Scoring
- [ ] REQ-P33-P37: 6-component scoring, DOF penalty, Ulcer caps, rating thresholds

### Stage 7: Reporting
- [ ] REQ-P38-P40: HTML report, JSON data, leaderboard sync

### Pipeline CLI
- [ ] REQ-P41-P43: Full CLI args, config override, progress logging

---

## Phase 6: Live Trading (FR-6) & Risk Management (FR-7)
- [ ] REQ-L01-L41: Trading loop, orders, position management, broker sync, state, multi-instance
- [ ] REQ-R01-R13: Pre-trade checks, position sizing, circuit breaker, status

---

## Phase 7: Verification (FR-8), Dashboard (FR-9), Reporting (FR-11)
- [ ] REQ-V01-V19: Signal replay, matching, P&L reconciliation
- [ ] REQ-M01-M35: Dashboard API, frontend, charts, alerts, service control
- [ ] REQ-A01-A11: HTML reports, leaderboard, scan reports

---

## Phase 8: Research Factory (FR-10) & Deployment (FR-12)
- [ ] REQ-F01-F16: Source tracker, triage, strategy build workflow
- [ ] REQ-X01-X21: Service management, overnight automation, parameter export, health monitoring
