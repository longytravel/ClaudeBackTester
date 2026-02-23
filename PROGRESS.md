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
- [x] Verify IC Markets MT5 demo account connectivity (broker/mt5.py, UTC conversion verified)

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
- [x] REQ-D24: Recent candle fetch for live trading context (MT5 broker module)

### CLI Tooling
- [x] REQ-D25: CLI commands (download, update, status, build-timeframes, validate)
- [~] REQ-D26: Background downloads (script running, no PID file yet)

---

## Phase 2: Strategy Framework (FR-2)

### Strategy Interface
- [x] REQ-S01: Name and version identifiers (Strategy.name, Strategy.version)
- [x] REQ-S02: Parameter space definition (ParamSpace with ParamDef)
- [x] REQ-S03: Signal generation method (generate_signals + vectorized variant)
- [x] REQ-S04: Signal filtering method (filter_signals + vectorized variant)
- [x] REQ-S05: SL/TP calculation method (sl_tp.calc_sl_tp)

### Performance Architecture
- [x] REQ-S06: Precompute-once, filter-many separation (generate once, filter per trial)
- [~] REQ-S07: 300+ evals/sec throughput (engine built, needs real-data benchmark)
- [x] REQ-S08: Vectorized fast path (generate_signals_vectorized, filter_signals_vectorized)

### Signal Attributes
- [x] REQ-S09: Mandatory atr_pips attribute (Signal.atr_pips)
- [x] REQ-S10: Arbitrary additional attributes (Signal.attrs dict)

### Parameter Organization
- [x] REQ-S11: Named parameter groups (ParamDef.group, ParamSpace.groups)
- [x] REQ-S12: Reusable standard groups (risk_params, management_params, time_params)
- [x] REQ-S13: Management params default to OFF (trailing_mode="off", breakeven=False, etc.)

### SL/TP Modes
- [x] REQ-S14: Fixed, ATR-based, swing-based SL (sl_tp.py)
- [x] REQ-S15: RR ratio, ATR-based, fixed TP (sl_tp.py)
- [x] REQ-S16: ATR auto-scaling with volatility (atr_pips * multiplier)
- [x] REQ-S17: Minimum TP >= SL constraint (enforced in calc_sl_tp)

### Trade Management
- [x] REQ-S18: Trailing stop (params + execution in `_simulate_trade_full()`)
- [x] REQ-S19: Breakeven lock (params + execution in `_simulate_trade_full()`)
- [x] REQ-S20: Partial close (params + execution in `_simulate_trade_full()`)
- [x] REQ-S21: Max bars exit (params + execution in `_simulate_trade_full()`)
- [x] REQ-S22: Stale exit (params + execution in `_simulate_trade_full()`)
- [~] REQ-S23: Identical management logic in backtest and live (backtest done, live pending)

### Indicator Library
- [x] REQ-S24: Full indicator set (RSI, ATR, SMA, EMA, BB, Stoch, MACD, ADX, Donchian, Supertrend, Keltner, Williams %R, CCI, Swing Hi/Lo)
- [x] REQ-S25: All indicators numpy-based (indicators.py)

### Registration & Lifecycle
- [x] REQ-S26: Central strategy registry (registry.py with aliases)
- [x] REQ-S27: Strategy lifecycle tracking (StrategyStage enum, set_stage/get_stage)

---

## Phase 3: Backtesting Engine (FR-3)

### Execution Modes
- [x] REQ-B01: Basic mode (SL/TP only) — `jit_loop.py _simulate_trade_basic()`
- [x] REQ-B02: Full mode (all management) — `jit_loop.py _simulate_trade_full()`
- [ ] REQ-B03: Grid mode (multiple concurrent positions) — DEFERRED (different simulation model)
- [x] REQ-B04: Telemetry mode (per-trade details) — `core/telemetry.py`

### Performance
- [x] REQ-B05: JIT-compiled core loop (sub-ms per eval) — `@njit(parallel=True)`
- [x] REQ-B06: Parallel eval across all cores — `prange(N)` over trials via TBB
- [x] REQ-B07: Shared memory for price arrays — contiguous numpy arrays shared across threads
- [x] REQ-B08: Shared pre-computed signals — generated once, shared read-only
- [x] REQ-B09: Batched trial submission — `batch_evaluate()` processes N trials per call
- [x] REQ-B10: Persistent reusable worker pool — TBB thread pool persists across calls

### Metrics
- [x] All 10 metrics: Trades, Win Rate, PF, Sharpe, Sortino, MaxDD%, Return%, R², Ulcer Index, Quality Score — `core/metrics.py` (Python) + `_compute_metrics_inline()` (JIT)

### Simulation Fidelity
- [x] REQ-B11: Bar-level SL/TP checking (intra-bar extremes)
- [x] REQ-B12: Conservative same-bar SL/TP tiebreak (SL wins)
- [x] REQ-B13: Configurable spread and slippage
- [x] REQ-B14: Management uses intra-bar highs/lows
- [x] REQ-B15: Fixed lot sizing only

### Execution Cost Modeling
- [x] SELL exit spread: SELL trades deduct exit bar's spread from PnL (BUY already pays at entry)
- [x] Round-trip commission: configurable `commission_pips` (default 0.7, IC Markets Raw) deducted from all trades
- [x] Max spread filter: `max_spread_pips` (default 3.0) rejects signals on high-spread bars
- [x] JIT + Telemetry parity: both paths apply identical cost deductions
- [x] 233 tests passing (11 new execution cost tests)

---

## Phase 4: Parameter Optimization (FR-4)

### Optimization Modes
- [x] REQ-O01: Staged optimization (signal → time → risk → management → refinement) — `optimizer/staged.py`
- [x] REQ-O02: Full-space optimization (refinement stage uses all params)
- [x] REQ-O03: Configurable trial counts per stage — `OptimizationConfig`

### Search Strategy
- [x] REQ-O04: Sobol/LHS exploration + Cross-Entropy/EDA exploitation (replaces TPE per research) — `optimizer/sampler.py`
- [~] REQ-O05: Throughput (needs real-data benchmark, Increment 13)
- [x] REQ-O06: Uses all available threads via TBB (no artificial cap)

### Hard Pre-Filters
- [x] REQ-O07: Max drawdown rejection (30%) — `prefilter.py postfilter_results()`
- [x] REQ-O08: R-squared floor (0.5)
- [x] REQ-O09: Minimum trades (20)
- [x] REQ-O10: Skip invalid combinations — `prefilter.py prefilter_invalid_combos()`

### Ranking & Candidates
- [x] REQ-O11: Forward-test evaluation — `optimizer/run.py`
- [x] REQ-O12: Independent back/forward ranking — `ranking.py rank_by_quality()`
- [x] REQ-O13: Combined rank with forward weight — `ranking.py combined_rank()`
- [x] REQ-O14: Top N selection (default 50) — `ranking.py select_top_n()`
- [x] REQ-O15: Diversity selection — `archive.py MAP-Elites DiversityArchive`
- [x] REQ-O16: Forward/back quality ratio gate (0.4) — `ranking.py forward_back_gate()`

### Speed Presets & Memory
- [x] REQ-O17: Turbo/Fast/Default presets — `optimizer/config.py`
- [x] REQ-O18-O21: Shared memory via numpy (no IPC needed), memory estimation in `run.py`

---

## Code Review (Phases 1-4)
- [x] Full code review of all 4 phases (6 parallel review agents)
- [x] 7 critical bugs found and fixed (commit 8adae74)
- [x] JIT pnl_buffers allocation moved outside prange (zero-allocation rule fix)
- [x] RSI variant value bug (period_idx → actual period value)
- [x] Filter PL layout defaults fixed (column 0 → -1 sentinel)
- [x] Telemetry signal filtering, stale exit, partial close added
- [x] Duplicate ParamDefs removed from RSI strategy

## Code Review Round 2 (External reviewer, Feb 2026)
- [x] RSI threshold crossover: generate signals at EACH threshold crossing (20/25/30/35, 65/70/75/80)
- [x] JIT filter changed from range comparison to exact match on filter_value
- [x] Telemetry filter changed to match JIT (exact match)
- [x] Dead params removed (atr_period, sma_filter_period) from RSI strategy
- [x] Staged optimizer: always use refinement result as overall best
- [x] Staged optimizer: guard for zero-passing stages (don't lock -inf params)
- [x] NaN spread sanitization in JIT basic + full modes (entry & exit)
- [x] NaN spread handling in max_spread_filter (reject NaN as over-threshold)
- [x] NaN spread sanitization in telemetry (entry & exit)
- [x] Time array warning when bar_hour/bar_day_of_week not provided
- [x] Weekly timeframe anchor: "1W" → "W-MON" for FX weekly bars
- [x] 253 tests passing (15 new regression tests), 362 after Phase 5

---

## Phase 5: Validation Pipeline (FR-5)

### Pipeline Architecture
- [x] REQ-P01-P06: 7-stage pipeline, JSON checkpointing, resume, stop-after (runner.py, checkpoint.py)

### Stage 3: Walk-Forward
- [x] REQ-P13-P21: Rolling/anchored windows, embargo gap, IS/OOS labeling, per-window metrics, aggregate stats, 60%+0.3 gates, WFE

### Stage 4: Stability
- [x] REQ-P22-P25: +-3 step perturbation on forward data, ROBUST/MODERATE/FRAGILE/OVERFIT rating (advisory)

### Stage 5: Monte Carlo
- [x] REQ-P26-P32: Block bootstrap, sign-flip permutation, trade-skip (5%+10%), execution stress, DSR hard gate (>=0.95)

### Stage 6: Confidence Scoring
- [x] REQ-P33-P37: Sequential gates + 6-component weighted composite (0-100), RED/YELLOW/GREEN rating. DSR replaces DOF penalty per research

### Stage 7: Reporting
- [x] REQ-P38 (partial): JSON report output (MVP). HTML report deferred to Phase 5b
- [ ] REQ-P39-P40: HTML report, leaderboard sync (Phase 5b)

### Pipeline CLI
- [ ] REQ-P41-P43: Full CLI args, config override, progress logging (deferred)

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
