# Product Requirements Document: Automated Forex Trading System

**Version:** 1.0
**Date:** February 2026
**Audience:** Solo Developer
**Status:** Draft

---

## Table of Contents

1. [Vision & Objectives](#1-vision--objectives)
2. [System Overview](#2-system-overview)
3. [FR-1: Historical Data Management](#fr-1-historical-data-management)
4. [FR-2: Strategy Framework](#fr-2-strategy-framework)
5. [FR-3: Backtesting Engine](#fr-3-backtesting-engine)
6. [FR-4: Parameter Optimization](#fr-4-parameter-optimization)
7. [FR-5: Validation Pipeline](#fr-5-validation-pipeline)
8. [FR-6: Live Trade Execution](#fr-6-live-trade-execution)
9. [FR-7: Risk Management](#fr-7-risk-management)
10. [FR-8: Trade Verification](#fr-8-trade-verification)
11. [FR-9: Monitoring Dashboard](#fr-9-monitoring-dashboard)
12. [FR-10: Strategy Research Factory](#fr-10-strategy-research-factory)
13. [FR-11: Reporting & Analytics](#fr-11-reporting--analytics)
14. [FR-12: Deployment & Operations](#fr-12-deployment--operations)
15. [NFR: Non-Functional Requirements](#nfr-non-functional-requirements)
16. [Appendix A: Quality Score Specification](#appendix-a-quality-score-specification)
17. [Appendix B: Confidence Score Specification](#appendix-b-confidence-score-specification)
18. [Appendix C: Metrics Glossary](#appendix-c-metrics-glossary)

---

## 1. Vision & Objectives

### 1.1 North Star

Build a **fully automated forex trading system** that autonomously discovers, validates, deploys, and monitors trading strategies with minimal human intervention. The system must take a strategy idea from concept through statistical validation to live execution, continuously monitoring for performance degradation.

### 1.2 Core Objectives

1. **Discover**: Systematically source and build trading strategies from external research, producing machine-testable implementations
2. **Optimize**: Search large parameter spaces to find profitable configurations, utilizing all available CPU cores, threads, and memory on a high-performance laptop
3. **Validate**: Subject every candidate to rigorous multi-stage statistical validation to prevent overfitting and ensure real-world robustness
4. **Deploy**: Translate validated strategies into live market execution with exact parity between backtest and live signal generation
5. **Monitor**: Provide real-time portfolio visibility, alerting, and performance-vs-expectation tracking across all deployed strategies
6. **Verify**: Continuously compare live trade outcomes against backtester predictions to detect execution drift or signal errors

### 1.3 Design Principles

- **Outcomes over implementation**: This document specifies WHAT the system must achieve, not HOW. The developer chooses languages, frameworks, brokers, and architectures
- **Backtest-live parity**: The backtesting engine and live trader MUST produce identical signals given identical data. Any divergence is a bug
- **Statistical rigor**: No strategy reaches live trading without passing walk-forward, Monte Carlo, stability, and confidence scoring gates
- **Maximum hardware utilization**: Optimization and backtesting must saturate all available CPU cores, leverage shared memory to avoid data duplication, and minimize IPC overhead
- **Crash resilience**: Every long-running process must persist state to disk and resume from where it left off after a crash or restart

### 1.4 Scope

- **Asset class**: Forex currency pairs only (e.g., GBP/USD, EUR/USD, USD/JPY). 20+ major and cross pairs
- **Timeframes**: M1, M5, M15, M30, H1, H4, Daily
- **Broker**: IC Markets Global (MT5) for execution. Dukascopy for historical data. The broker integration layer must be swappable without affecting strategy logic, backtesting, or the pipeline
- **Deployment**: Develop and test locally (Windows 11), deploy to Linux VPS for 24/7 live trading
- **Target operator**: A single developer who builds, runs, and monitors the entire system

---

## 2. System Overview

The system comprises 12 functional domains organized into three phases:

### Research & Development Phase
- **Historical Data Management** (FR-1): Acquire, cache, validate, and serve price data
- **Strategy Framework** (FR-2): Define, implement, and register trading strategies
- **Backtesting Engine** (FR-3): Simulate strategy execution on historical data
- **Parameter Optimization** (FR-4): Search parameter spaces to maximize a quality metric
- **Validation Pipeline** (FR-5): Multi-stage statistical validation (7 stages)

### Live Execution Phase
- **Live Trade Execution** (FR-6): Real-time signal detection, order placement, and position management
- **Risk Management** (FR-7): Pre-trade compliance, circuit breakers, position sizing

### Operations Phase
- **Trade Verification** (FR-8): Compare live trades against backtester expectations
- **Monitoring Dashboard** (FR-9): Real-time portfolio monitoring, alerts, and analytics
- **Strategy Research Factory** (FR-10): Systematic sourcing and building of new strategies
- **Reporting & Analytics** (FR-11): Pipeline run reports, cross-run leaderboards
- **Deployment & Operations** (FR-12): Service management, automation, remote access

---

## FR-1: Historical Data Management

### 1.1 Outcomes

The data system must provide accurate, complete, and efficiently accessible historical price data for all supported currency pairs and timeframes.

### 1.2 Data Acquisition

| Requirement | Description |
|---|---|
| **REQ-D01** | Download historical OHLCV (Open, High, Low, Close, Volume) candlestick data from the broker API for any supported currency pair and timeframe |
| **REQ-D02** | Support downloading up to 8+ years of minute-level (M1) history per pair |
| **REQ-D03** | Handle broker API rate limits gracefully (chunk requests, respect throttling) |
| **REQ-D04** | Implement retry logic with exponential backoff for failed API requests (minimum 5 retries per chunk) |
| **REQ-D05** | Display download progress (candles downloaded, rate, estimated completion) |
| **REQ-D06** | Support incremental updates: only download new data since the last cached timestamp, not the full history |

### 1.3 Caching & Storage

| Requirement | Description |
|---|---|
| **REQ-D07** | Store downloaded data in a compressed columnar format suitable for fast time-series access (millions of rows) |
| **REQ-D08** | Use a canonical file naming convention: `{PAIR}_{TIMEFRAME}.{ext}` |
| **REQ-D09** | Implement crash-recovery checkpointing: save partial downloads to disk at regular intervals (e.g., every 100K candles) so that a crash does not lose hours of download progress |
| **REQ-D10** | For large downloads (M1 data spanning years), split into yearly chunks to avoid memory exhaustion or serialization failures |

### 1.4 Timeframe Conversion

| Requirement | Description |
|---|---|
| **REQ-D11** | Designate M1 data as the single source of truth. All higher timeframes (M5, M15, M30, H1, H4, D, W) must be buildable on-demand from cached M1 data |
| **REQ-D12** | Conversion must use correct OHLCV aggregation: first open, max high, min low, last close, sum volume |
| **REQ-D13** | Building from M1 must capture true intra-period highs and lows (more accurate than downloading pre-aggregated higher timeframes directly from the broker) |
| **REQ-D14** | Fall back to direct timeframe download if M1 data is unavailable, with a logged warning recommending M1 download |

### 1.5 Data Validation

| Requirement | Description |
|---|---|
| **REQ-D15** | Detect timestamp gaps exceeding 3x the expected candle interval (e.g., >180 minutes for H1). Distinguish between expected gaps (weekends, holidays) and unexpected gaps |
| **REQ-D16** | Detect zero or NaN values in OHLC fields. Report count and percentage |
| **REQ-D17** | Detect anomalies: candles with range > 10x typical median range, zero-range candles (high == low), and OHLC violations (high < low) |
| **REQ-D18** | Compute a data quality score (0-100) based on the severity and count of issues found. Deduct heavily for critical issues (large gaps, zero values) and lightly for minor warnings (weekend gaps, anomalies) |
| **REQ-D19** | Reject datasets below a minimum quality threshold (e.g., score < 50) or with fewer than a configurable minimum number of candles (e.g., 5,000) |

### 1.6 Data Splitting

| Requirement | Description |
|---|---|
| **REQ-D20** | Split validated data into a back-test portion and a forward-test portion. Default split: 80% back / 20% forward |
| **REQ-D21** | Support an alternative holdout mode: reserve the last N months as a pure out-of-sample holdout (configurable, e.g., 3 months) |
| **REQ-D22** | Both portions must be available downstream for optimization, walk-forward, and reporting stages |

### 1.7 Data Freshness

| Requirement | Description |
|---|---|
| **REQ-D23** | Before pipeline execution, check if cached data is stale (e.g., last timestamp > 2 hours old). If stale, perform an incremental update |
| **REQ-D24** | For live trading, fetch the most recent N candles on startup (e.g., 500-1000) to provide indicator lookback context, then stream or poll for new candles as they close |

### 1.8 CLI Tooling

| Requirement | Description |
|---|---|
| **REQ-D25** | Provide a CLI tool for data operations: `download`, `update`, `status` (show all cached files with date ranges and sizes), and `rebuild` (rebuild higher timeframes from M1) |
| **REQ-D26** | Support detached/background downloads for long-running M1 downloads (30-60+ minutes), with a log file and PID file for monitoring from a separate terminal |

---

## FR-2: Strategy Framework

### 2.1 Outcomes

The strategy framework must provide a standardized interface for defining, implementing, and registering trading strategies. Strategies must be decoupled from the broker, backtester, and optimizer so that they can be developed, tested, and deployed independently.

### 2.2 Strategy Interface Contract

| Requirement | Description |
|---|---|
| **REQ-S01** | Every strategy must expose a `name` and `version` identifier |
| **REQ-S02** | Every strategy must define its **parameter space**: a dictionary mapping parameter names to lists of allowed values. This space is what the optimizer searches |
| **REQ-S03** | Every strategy must implement a **signal generation** method that, given a DataFrame of OHLCV data, returns a list of entry signals. Each signal must include: bar index, direction (buy/sell), entry price, hour of day, day of week, and a dictionary of computed attributes (e.g., ATR, indicator values) |
| **REQ-S04** | Every strategy must implement a **signal filtering** method that, given pre-computed signals and a parameter set, returns only the signals that pass the parameter-based filters. This method must be computationally cheap (no indicator recalculation) |
| **REQ-S05** | Every strategy must implement a **stop-loss/take-profit calculation** method that, given a signal and parameter set, returns exact SL and TP price levels |

### 2.3 Performance Architecture: Precompute-Once, Filter-Many

| Requirement | Description |
|---|---|
| **REQ-S06** | Signal generation (scanning bars, computing indicators) must be separated from signal filtering (applying parameter-based thresholds). The expensive step (generation) runs ONCE per dataset. The cheap step (filtering) runs ONCE PER TRIAL during optimization |
| **REQ-S07** | This separation must enable throughput of 300+ parameter evaluations per second during optimization |
| **REQ-S08** | Strategies should support a **vectorized fast path** using array operations (e.g., numpy boolean masks) for filtering, achieving 10-50x speedup over Python-loop filtering |

### 2.4 Signal Attributes

| Requirement | Description |
|---|---|
| **REQ-S09** | Every signal must carry an `atr_pips` attribute (Average True Range in pips) for ATR-based risk management calculations. This is a mandatory field |
| **REQ-S10** | Strategies may attach arbitrary additional attributes to signals (e.g., RSI value, trend direction, divergence magnitude) for use in filtering without recalculation |

### 2.5 Parameter Organization

| Requirement | Description |
|---|---|
| **REQ-S11** | Parameters must be organizable into named **groups** for staged optimization. Typical groups: Signal (entry logic), Filters (confirmation), Risk (SL/TP), Management (trailing/breakeven/partial), Time (session/day filters) |
| **REQ-S12** | Provide **reusable standard parameter groups** for Risk, Management, and Time that can be shared across all strategies. Only Signal and Filters are strategy-specific |
| **REQ-S13** | All trade management parameters (trailing, breakeven, partial close, max bars) must default to OFF. Signal-stage testing must evaluate pure entry quality before management optimization |

### 2.6 Stop-Loss & Take-Profit Modes

| Requirement | Description |
|---|---|
| **REQ-S14** | Support at least three SL modes: fixed pips, ATR-based (percentage of ATR), and swing-based (recent swing high/low) |
| **REQ-S15** | Support at least three TP modes: risk-reward ratio (multiple of SL distance), ATR-based (multiple of ATR), and fixed pips |
| **REQ-S16** | ATR-based modes must auto-scale with volatility and timeframe (e.g., 50 pips means different things on M5 vs D1, but 1.5x ATR is always proportional) |
| **REQ-S17** | Enforce a minimum constraint: TP distance >= SL distance (at least 1:1 reward-to-risk) |

### 2.7 Trade Management Features

| Requirement | Description |
|---|---|
| **REQ-S18** | **Trailing Stop**: Two modes — (a) fixed-pip trailing that activates after N pips profit and trails by M pips; (b) chandelier/ATR-based trailing that is always active and trails at N x ATR from the highest price since entry |
| **REQ-S19** | **Breakeven Lock**: Move SL to entry price (plus optional small offset) once profit reaches a configurable threshold. Must trigger at most once per trade |
| **REQ-S20** | **Partial Close**: Close a configurable percentage (e.g., 30-70%) of the position when profit reaches a first target. Must trigger at most once per trade |
| **REQ-S21** | **Max Bars Exit**: Force-close the position after N bars if still open (prevents zombie trades) |
| **REQ-S22** | **Stale Exit**: Force-close if price has moved less than a threshold (e.g., 0.5x ATR) after N bars (kills stagnant trades) |
| **REQ-S23** | Trade management in live trading must execute in the exact same order and with the same logic as in the backtester. Any divergence is a bug |

### 2.8 Indicator Library

| Requirement | Description |
|---|---|
| **REQ-S24** | Provide at minimum these indicators, implemented for performance (no external library dependency in hot path): RSI, ATR, SMA, EMA, Bollinger Bands, Stochastic (%K/%D), MACD (line/signal/histogram), ADX (+DI/-DI), Donchian Channel, Supertrend, Keltner Channel, Williams %R, CCI, Swing High/Low detection |
| **REQ-S25** | All indicators must operate on numpy arrays and return numpy arrays for vectorized downstream processing |

### 2.9 Strategy Registration & Lifecycle

| Requirement | Description |
|---|---|
| **REQ-S26** | A central registry must map strategy names (including aliases) to their implementations. The optimizer and live trader look up strategies by name from this registry |
| **REQ-S27** | Track strategy lifecycle stages: `built` -> `validated` -> `pipeline_run` -> `refined` -> `live` or `archived` |

---

## FR-3: Backtesting Engine

### 3.1 Outcomes

The backtesting engine must simulate strategy execution on historical data with high fidelity and extreme speed, producing trade-level results and aggregate performance metrics.

### 3.2 Execution Modes

| Requirement | Description |
|---|---|
| **REQ-B01** | **Basic mode**: SL/TP exit only, no trade management. Fastest execution (~700+ evaluations/sec). Used for initial signal-quality screening |
| **REQ-B02** | **Full mode**: Complete trade management support (trailing stops, breakeven, partial closes, max bars exit, stale exit). Used for full parameter optimization (~400+ evaluations/sec) |
| **REQ-B03** | **Grid mode**: Support multiple concurrent positions with group identifiers. Each signal carries a group ID; the engine tracks positions per group. Used for grid/martingale-style strategies |
| **REQ-B04** | **Telemetry mode**: Return detailed per-trade telemetry — entry/exit bars, prices, direction, P&L in pips and currency, bars held, maximum favorable excursion (MFE), maximum adverse excursion (MAE), exit reason. Used for post-analysis and debugging, not optimization |

### 3.3 Performance Requirements

| Requirement | Description |
|---|---|
| **REQ-B05** | The core backtest loop must be JIT-compiled or otherwise optimized to achieve sub-millisecond execution per parameter evaluation on a single core |
| **REQ-B06** | Must support parallel evaluation across all available CPU cores. Target: 300-500 parameter evaluations per second with 8-12 workers on a modern laptop |
| **REQ-B07** | Price data arrays (highs, lows, closes) must be stored in shared memory accessible by all worker processes without copying. Zero-copy read-only views for workers |
| **REQ-B08** | Pre-computed signal arrays must also be shareable across workers to avoid redundant computation |
| **REQ-B09** | Worker processes must use chunked/batched trial submission (e.g., 32 trials per IPC call) to amortize inter-process communication overhead (~5ms per roundtrip) |
| **REQ-B10** | Worker pool must be persistent and reusable across optimization stages (create once, reuse across all stages) to avoid repeated pool creation overhead |

### 3.4 Metrics Computation

The backtester must compute and return the following metrics for each parameter evaluation:

| Metric | Description |
|---|---|
| **Trades** | Total number of closed trades |
| **Win Rate** | Percentage of trades with positive P&L |
| **Profit Factor** | Gross profit / gross loss |
| **Sharpe Ratio** | Annualized risk-adjusted return (mean / std of returns) |
| **Sortino Ratio** | Annualized downside-risk-adjusted return (mean / downside std). Preferred over Sharpe for asymmetric strategies |
| **Max Drawdown %** | Largest peak-to-trough decline as a percentage |
| **Total Return %** | Cumulative return over the test period |
| **R-squared** | Goodness of fit of equity curve to a linear regression. Measures smoothness of growth (0 = random walk, 1 = perfectly linear) |
| **Ulcer Index** | Root-mean-square of per-bar drawdown percentages. Captures chronic underwater pain, not just peak drawdown |
| **Quality Score** | Composite metric combining all of the above (see [Appendix A](#appendix-a-quality-score-specification)) |

### 3.5 Simulation Fidelity

| Requirement | Description |
|---|---|
| **REQ-B11** | Use bar-level simulation: on each bar, check SL/TP against the bar's high and low (intra-bar extremes), not just the close price |
| **REQ-B12** | When both SL and TP could trigger on the same bar, apply a conservative worst-case assumption (SL hit first for longs if low <= SL, TP hit first for shorts if high >= TP — or use a configurable tiebreak rule) |
| **REQ-B13** | Apply configurable spread and slippage costs to entry and exit prices |
| **REQ-B14** | Trade management (trailing, breakeven) must use intra-bar highs/lows for activation checks, matching live trader behavior |
| **REQ-B15** | Fixed lot sizing only (no variable position sizing, no compounding). All trades use the same notional size |

### 3.6 Known Limitations (Documented Gaps)

The following capabilities are explicitly out of scope for the initial version. They must be tracked as known gaps:

| Gap | Priority | Description |
|---|---|---|
| Variable lot sizing | Medium | Martingale, Kelly criterion, anti-martingale position sizing |
| Hedging | Low | Opposite positions on the same pair simultaneously |
| Scale-in / pyramiding | Medium | Adding to a winning position at multiple levels |
| Multi-pair correlation | Low | Cross-pair signal generation or portfolio-level optimization |
| Tick-level execution | Low | Sub-minute (M1) granularity. M1 is the finest available |

---

## FR-4: Parameter Optimization

### 4.1 Outcomes

The optimizer must search a strategy's parameter space to find configurations that maximize a quality metric, producing ranked candidates for downstream validation. It must fully utilize all available CPU cores and memory.

### 4.2 Optimization Modes

| Requirement | Description |
|---|---|
| **REQ-O01** | **Staged optimization**: Divide parameters into groups (signal, filters, risk, management, time). Optimize each group sequentially, locking the best values from prior groups. Follow with a final refinement stage that searches tight ranges (e.g., +/-5 neighbors) around all locked values. This prevents the greedy-locking problem where early bad decisions propagate |
| **REQ-O02** | **Full-space optimization**: Search all parameters simultaneously in a single stage. Useful for small parameter spaces (<15 params) where staging is unnecessary |
| **REQ-O03** | Support configurable trial counts per stage (e.g., 5,000 per stage, 10,000 for final) |

### 4.3 Search Strategy

| Requirement | Description |
|---|---|
| **REQ-O04** | Use a search algorithm with zero per-trial overhead that keeps all workers 100% busy. Random search satisfies this requirement for staged optimization (3-11 params per stage). Bayesian/adaptive search may be used for full-space optimization where the search space is larger |
| **REQ-O05** | Achieve throughput of 300-500 trials/second with 8-12 parallel workers |
| **REQ-O06** | Hard-cap parallel workers at a safe maximum for the platform (e.g., 12 on Windows with JIT-compiled code) to prevent crashes from excessive concurrent JIT compilation |

### 4.4 Hard Pre-Filters

Applied instantly during optimization to reject garbage configurations before full scoring:

| Requirement | Description |
|---|---|
| **REQ-O07** | Reject any parameter set where Max Drawdown exceeds a hard limit (configurable, default 30%) |
| **REQ-O08** | Reject any parameter set where R-squared is below a hard floor (configurable, default 0.5) |
| **REQ-O09** | Reject any parameter set producing fewer than a minimum number of trades (configurable, default 20) |
| **REQ-O10** | Skip invalid parameter combinations without backtesting (e.g., trailing step > trailing start, breakeven offset > breakeven trigger) |

### 4.5 Ranking & Candidate Selection

| Requirement | Description |
|---|---|
| **REQ-O11** | After optimization, forward-test ALL valid results on the forward-test data portion |
| **REQ-O12** | Rank candidates by back-test Quality Score and by forward-test Quality Score independently |
| **REQ-O13** | Compute a combined rank: `back_rank + forward_rank * forward_weight`. The forward weight should be configurable (default 2.0) to heavily favor forward-test consistency |
| **REQ-O14** | Select the top N candidates (configurable, default 50) for downstream validation |
| **REQ-O15** | Apply diversity selection: prefer candidates with distinct parameter signatures. Limit to 1 candidate per unique parameter combination in the first pass, then fill remaining slots with duplicates if needed |
| **REQ-O16** | Optionally reject candidates where forward Quality Score < back Quality Score * minimum ratio (configurable, default 0.4). This guards against severe overfitting |

### 4.6 Speed Presets

| Requirement | Description |
|---|---|
| **REQ-O17** | Provide named speed presets for common use cases: **Turbo** (~20 sec total pipeline: minimal trials, small candidate list), **Fast** (~30-40 sec: moderate trials), **Default** (~1-3 min: full statistical power) |

### 4.7 Memory Management

| Requirement | Description |
|---|---|
| **REQ-O18** | Price arrays must be placed in shared memory (one copy for all workers). Per-worker memory overhead should be ~150-200MB, not scaling with data size |
| **REQ-O19** | Pre-computed signals should be serialized compactly and shared across workers |
| **REQ-O20** | Release optimization-specific data structures after each stage completes to prevent memory accumulation across stages |
| **REQ-O21** | Compute memory requirements before optimization begins: `workers * 180MB + base 200MB`. Warn if available memory is insufficient |

---

## FR-5: Validation Pipeline

### 5.1 Outcomes

The validation pipeline subjects optimizer candidates to a 7-stage gauntlet of statistical tests. Its purpose is to prevent overfitting and ensure only genuinely robust strategies reach live trading. Each stage produces specific metrics, and the final output is a confidence score (0-100) with a rating (RED/YELLOW/GREEN).

### 5.2 Pipeline Architecture

| Requirement | Description |
|---|---|
| **REQ-P01** | The pipeline must execute 7 stages in strict order: (1) Data, (2) Optimization, (3) Walk-Forward, (4) Stability, (5) Monte Carlo, (6) Confidence, (7) Report |
| **REQ-P02** | Each pipeline run must write to a unique, timestamped output directory (e.g., `{PAIR}_{TIMEFRAME}_{YYYYMMDD}_{HHMMSS}/`) |
| **REQ-P03** | Pipeline state must be serialized to disk after every stage completion. A crashed or interrupted pipeline must be resumable from any completed stage |
| **REQ-P04** | Support a `--resume-from {stage}` CLI argument that reloads state and data, then resumes from the specified stage |
| **REQ-P05** | Support a `--stop-after {stage}` CLI argument for partial runs (e.g., stop after optimization without running walk-forward) |
| **REQ-P06** | Support a `--max-retries N` supervisor mode that reruns the pipeline in a fresh process on transient failures (e.g., JIT compilation segfaults). Non-transient errors must fail immediately |

### 5.3 Stage 1: Data Download & Validation

(See [FR-1](#fr-1-historical-data-management) for full data requirements)

| Requirement | Description |
|---|---|
| **REQ-P07** | Download and validate data per FR-1 requirements |
| **REQ-P08** | Split data into back-test and forward-test portions per REQ-D20/D21 |
| **REQ-P09** | **Gate**: Fail the pipeline if data quality score < 50 or candle count < minimum threshold |

### 5.4 Stage 2: Optimization

(See [FR-4](#fr-4-parameter-optimization) for full optimization requirements)

| Requirement | Description |
|---|---|
| **REQ-P10** | Run optimization per FR-4 requirements, producing ranked candidates |
| **REQ-P11** | Log per-stage progress: trial count, best quality score, best Sharpe, best R-squared |
| **REQ-P12** | **No hard gate**: Stage always produces candidates (may be 0 if no valid results found, which results in a RED report) |

### 5.5 Stage 3: Walk-Forward Validation

| Requirement | Description |
|---|---|
| **REQ-P13** | Generate rolling train/test windows across the full dataset. Configurable: train months (default 18), test months (default 6), roll step (default 6 months, non-overlapping per Pardo recommendation) |
| **REQ-P14** | Require a minimum number of windows (configurable, default 6) for statistical significance. If insufficient windows, fail the stage with a clear error message |
| **REQ-P15** | Mark each window as In-Sample (IS) or Out-of-Sample (OOS) based on whether the test period overlaps with the optimization back-test period. OOS windows are the gold standard for robustness |
| **REQ-P16** | For each candidate, test parameters on each window. Optimization: precompute signals ONCE on the full dataset, then filter by window date range (avoid N candidates x W windows recomputes) |
| **REQ-P17** | Compute per-window metrics: quality score, Sharpe, trades, win rate, profit factor, max drawdown |
| **REQ-P18** | Compute aggregate walk-forward statistics per candidate: pass rate (windows with quality score > 0), mean Sharpe, consistency (mean Sharpe / std Sharpe), quality score coefficient of variation, geometric mean of quality scores, worst single window |
| **REQ-P19** | When OOS windows exist, compute separate OOS-specific metrics (OOS pass rate, OOS mean Sharpe, OOS geometric mean QS) and prefer those for downstream scoring |
| **REQ-P20** | **Gate**: Candidates must achieve pass rate >= configurable minimum (default 50%) and mean Sharpe >= configurable minimum (default 0.5) to advance. Failed candidates are logged but not passed downstream |
| **REQ-P21** | Require a minimum number of trades per window (configurable, default 30) for the Central Limit Theorem to apply |

### 5.6 Stage 4: Parameter Stability Analysis

| Requirement | Description |
|---|---|
| **REQ-P22** | For each surviving candidate, perturb each numeric parameter by +/- one step (or +/- 10% for continuous parameters). Boolean and categorical parameters are skipped |
| **REQ-P23** | Backtest each perturbed parameter set and compute the quality score ratio: `avg(neighbor_scores) / baseline_score` |
| **REQ-P24** | Flag fragile parameters (ratio < 0.7) and compute an overall stability rating: Robust (mean > 0.8, min > 0.5), Moderate (mean > 0.6), Fragile (mean > 0.4), Overfit (mean <= 0.4) |
| **REQ-P25** | **Advisory only**: Stability is NOT a gate. All candidates pass through to Monte Carlo. Stability scores are incorporated as a 15% weight in the final confidence score |

### 5.7 Stage 5: Monte Carlo Simulation

| Requirement | Description |
|---|---|
| **REQ-P26** | **Shuffle Monte Carlo** (default 2000 iterations): Randomly reorder trades, simulate equity curves, and produce distributions of max drawdown and final return. Answers: "What if trades happened in a different order?" |
| **REQ-P27** | **Bootstrap Resampling** (default 2000 iterations): Sample trades with replacement, compute 95% confidence intervals for Sharpe, win rate, profit factor, total return, and max drawdown |
| **REQ-P28** | **Permutation Test** (default 1000 iterations): Randomly flip the sign of each trade's P&L (50% chance). Compute the distribution of Sharpe ratios under the null hypothesis that entries are random. Report p-value (fraction of null Sharpes exceeding observed Sharpe) |
| **REQ-P29** | **Trade-Skip Stress Test** (default 1000 iterations): Randomly drop 10% of trades each iteration. Measure return degradation and probability of remaining profitable. Simulates missed fills and execution gaps |
| **REQ-P30** | **Probabilistic Sharpe Ratio (PSR)**: Compute the analytical probability that the true Sharpe ratio exceeds a benchmark (e.g., 0), accounting for sample size, skewness, and kurtosis of the return distribution (per Bailey & Lopez de Prado, 2012) |
| **REQ-P31** | Monte Carlo computations must be parallelized across all CPU cores. Target: 1000+ iterations in <100ms |
| **REQ-P32** | **No hard gate**: Monte Carlo always produces results. They are incorporated into the confidence score |

### 5.8 Stage 6: Confidence Scoring

(See [Appendix B](#appendix-b-confidence-score-specification) for the full scoring formula)

| Requirement | Description |
|---|---|
| **REQ-P33** | Compute a composite confidence score (0-100) from 6 weighted components: Walk-Forward (30%), Monte Carlo (25%), Stability (15%), Forward/Back Ratio (10%), Backtest Quality (10%), Quality Score (10%) |
| **REQ-P34** | Apply a Degrees-of-Freedom (DOF) penalty (per Bailey & Lopez de Prado, 2014): degrade the score based on the expected Sharpe inflation from testing N parameter combinations. Penalty formula: `sqrt(2 * ln(N_trials_tested))` divided by observed Sharpe. Maximum penalty: 20 points |
| **REQ-P35** | Apply Ulcer Index caps: if Ulcer > 10, cap the candidate's score at YELLOW (70). If Ulcer > 20, cap at RED (40). This penalizes strategies with chronic, prolonged drawdowns |
| **REQ-P36** | Assign a rating based on score thresholds: RED (0-39: do not trade), YELLOW (40-69: caution), GREEN (70-100: deploy with appropriate sizing) |
| **REQ-P37** | Sort all candidates by confidence score. The top candidate becomes the pipeline's recommended best candidate |

### 5.9 Stage 7: Report Generation

(See [FR-11](#fr-11-reporting--analytics) for full reporting requirements)

| Requirement | Description |
|---|---|
| **REQ-P38** | Generate a self-contained interactive HTML report with charts, tables, and per-candidate breakdowns |
| **REQ-P39** | Generate a structured data file (JSON) with all report data for programmatic consumption |
| **REQ-P40** | Optionally sync results to a cross-run leaderboard and remote dashboard |

### 5.10 Pipeline CLI

| Requirement | Description |
|---|---|
| **REQ-P41** | CLI must accept: pair, timeframe, strategy name, years of data, trial counts, test months, top-N candidates, Monte Carlo iterations, worker count, speed preset, resume/stop flags, retry count, verbosity |
| **REQ-P42** | All CLI arguments must override configuration file defaults |
| **REQ-P43** | Log pipeline progress with stage timing, candidate counts, and key metrics after each stage |

---

## FR-6: Live Trade Execution

### 6.1 Outcomes

The live trading engine must execute strategy signals in real-time on the broker, managing open positions with the exact same logic used in backtesting, and persisting all state for crash recovery.

### 6.2 Core Trading Loop

| Requirement | Description |
|---|---|
| **REQ-L01** | Run a continuous loop: wait for candle close -> fetch latest candles -> generate signals -> execute trades -> manage positions -> sync with broker -> update state -> repeat |
| **REQ-L02** | Calculate exact next candle close time for any timeframe (M1 through Daily) and sleep until that time |
| **REQ-L03** | Fetch the last N candles (configurable, e.g., 200) on each iteration to provide indicator lookback |
| **REQ-L04** | Generate signals on the LATEST bar only (not historical bars). Skip if the signal timestamp was already processed (deduplication) |
| **REQ-L05** | Apply all risk management checks (see FR-7) before placing any order |

### 6.3 Order Execution

| Requirement | Description |
|---|---|
| **REQ-L06** | Place market orders with SL and TP attached at entry time |
| **REQ-L07** | Tag each order with the strategy instance identifier (persisted on the trade at the broker) for multi-strategy attribution |
| **REQ-L08** | Calculate position size based on: account balance, risk percentage per trade, and stop-loss distance in pips. Return integer units >= 1 |
| **REQ-L09** | Support modifying SL/TP on open trades post-entry (required for trailing stops, breakeven) |
| **REQ-L10** | Support partial position closes (required for partial take-profit) |

### 6.4 Position Management

| Requirement | Description |
|---|---|
| **REQ-L11** | Implement all trade management features from FR-2 (trailing stop, breakeven, partial close, max bars exit, stale exit) with IDENTICAL logic and execution order as the backtester |
| **REQ-L12** | Track per-trade management state: trailing high/low, breakeven triggered flag, partial close flag, bars in trade counter |
| **REQ-L13** | Persist management state to disk after every management cycle for crash recovery |
| **REQ-L14** | Log every management action to an append-only audit trail (timestamp, trade ID, action type, old SL, new SL) |

### 6.5 Broker Synchronization

| Requirement | Description |
|---|---|
| **REQ-L15** | On each cycle, compare locally tracked open positions against the broker's actual open trades |
| **REQ-L16** | Detect externally closed positions (SL/TP hit, margin call, manual close). Retrieve actual exit price, exit reason, and realized P&L from the broker |
| **REQ-L17** | Only remove a local position record after successfully retrieving real exit data. If retrieval fails, retry on the next sync cycle |
| **REQ-L18** | Update unrealized P&L on all open positions from live broker data |

### 6.6 State Persistence

| Requirement | Description |
|---|---|
| **REQ-L19** | Persist to disk after every cycle: open positions, daily statistics, and trade history |
| **REQ-L20** | Use atomic writes (write to temp file, then rename) to prevent corruption from mid-write crashes |
| **REQ-L21** | On startup, load persisted state and resume from where the previous session left off |
| **REQ-L22** | Maintain a rolling trade history (e.g., last 500 trades) in memory, archived to disk |

### 6.7 Daily Statistics

| Requirement | Description |
|---|---|
| **REQ-L23** | Track per-day: starting balance, current balance, trades opened/closed, wins, losses, gross profit, gross loss, peak balance, max drawdown |
| **REQ-L24** | Automatically roll over daily stats at midnight: archive the previous day's stats to a daily history file, then reset for the new day |

### 6.8 Multi-Instance Support

| Requirement | Description |
|---|---|
| **REQ-L25** | Support running multiple strategy instances concurrently (e.g., RSI on GBP/USD M15, EMA on EUR/USD H1), each with its own instance ID, state directory, and log files |
| **REQ-L26** | Implement a pair occupancy guard: if one instance already has an open position on a pair, other instances must not open a new position on the same pair |

### 6.9 Practice vs Live Mode

| Requirement | Description |
|---|---|
| **REQ-L27** | Support a practice/paper trading mode that uses the broker's demo account (real market data, simulated execution) |
| **REQ-L28** | Support a dry-run mode that logs signals without placing any orders (for testing) |
| **REQ-L29** | Switching to live (real money) mode must require an explicit confirmation step (e.g., typing "LIVE") |

### 6.10 Heartbeat & Health

| Requirement | Description |
|---|---|
| **REQ-L30** | Write a heartbeat file on every cycle: timestamp, status, open position count, last candle processed, error count, last error message |
| **REQ-L31** | The heartbeat must be readable by the monitoring system (FR-9) for stale-instance detection |

### 6.11 Notifications

| Requirement | Description |
|---|---|
| **REQ-L32** | Send push notifications for: trade opened, trade closed, daily summary, risk warnings, circuit breaker triggered, startup, shutdown |
| **REQ-L33** | Notifications must include: instance ID prefix, trade details (pair, direction, price, SL, TP, P&L), and severity level |
| **REQ-L34** | If no notification channel is configured, the system must continue operating without errors (graceful degradation) |

### 6.12 Error Resilience

| Requirement | Description |
|---|---|
| **REQ-L35** | Wrap every iteration of the main loop in error handling. Log errors but continue running |
| **REQ-L36** | Apply a backoff delay (e.g., 10 seconds) after any error before the next iteration |
| **REQ-L37** | Track cumulative error count for health monitoring |
| **REQ-L38** | Handle graceful shutdown on process termination signals: save state, print daily summary, then exit |

### 6.13 Pipeline Integration

| Requirement | Description |
|---|---|
| **REQ-L39** | Accept optimized parameters from a pipeline run directory or exported JSON file |
| **REQ-L40** | Wrap the optimizer's strategy representation to work with the live trading interface. The same strategy code must produce the same signals in both optimization and live contexts |
| **REQ-L41** | Support loading any strategy from the registry by name and applying the corresponding parameter file |

---

## FR-7: Risk Management

### 7.1 Outcomes

The risk management system must enforce pre-trade compliance checks, position sizing rules, and circuit breakers to protect capital. All checks run BEFORE every trade.

### 7.2 Pre-Trade Checks

| Requirement | Description |
|---|---|
| **REQ-R01** | **Daily trade limit**: Reject new trades if trades opened today >= configurable maximum (default 5) |
| **REQ-R02** | **Daily loss limit**: Reject new trades if daily P&L loss >= configurable maximum percentage (default 3% of balance) |
| **REQ-R03** | **Drawdown circuit breaker**: Reject ALL trades if drawdown from peak balance >= configurable maximum (default 25%). This is a hard stop requiring manual reset |
| **REQ-R04** | **Drawdown warning**: Log a warning when drawdown reaches a softer threshold (default 15%) but still allow trades |
| **REQ-R05** | **Open position limit**: Reject new trades if open positions >= configurable maximum (default 3) |
| **REQ-R06** | **Instrument occupancy**: Reject new trades if the instrument already has an open position (single position per pair rule) |
| **REQ-R07** | **Spread check**: Reject new trades if the current bid-ask spread exceeds a configurable maximum (default 3.0 pips) |

### 7.3 Position Sizing

| Requirement | Description |
|---|---|
| **REQ-R08** | Calculate position size: `risk_amount = balance * risk_pct / 100`, `units = risk_amount / (sl_pips * pip_value)`. Return integer units >= 1 |
| **REQ-R09** | Risk percentage must be configurable per instance (default 1.0%) |

### 7.4 Circuit Breaker

| Requirement | Description |
|---|---|
| **REQ-R10** | When the circuit breaker trips (drawdown >= max), ALL trading must halt immediately |
| **REQ-R11** | A manual reset function must be provided to re-enable trading after the operator has assessed the situation |
| **REQ-R12** | Circuit breaker state must persist across restarts |

### 7.5 Status Reporting

| Requirement | Description |
|---|---|
| **REQ-R13** | Expose a status summary: circuit breaker state, trading paused flag, daily trade count, open position count, daily P&L percentage, and all configured limits |

---

## FR-8: Trade Verification

### 8.1 Outcomes

The trade verification system must compare live trades executed on the broker against the backtester's signal generation to confirm parity. Any divergence indicates a bug in signal generation, parameter application, or trade management.

### 8.2 Input: Broker Trade Export

| Requirement | Description |
|---|---|
| **REQ-V01** | Parse the broker's transaction export (e.g., CSV) to extract trade records: entry time, entry price, direction, units, SL, TP, exit time, exit price, exit reason, realized P&L, spread cost |
| **REQ-V02** | Link each trade to its originating strategy instance by extracting the strategy tag from the broker's order metadata |
| **REQ-V03** | Handle timezone conversions: broker export timestamps must be converted to UTC for comparison |

### 8.3 Signal Replay

| Requirement | Description |
|---|---|
| **REQ-V04** | For each live trade, fetch the exact candle data the live trader would have seen at entry time (e.g., 200 candles ending at the candle boundary before entry) |
| **REQ-V05** | Replay the strategy's signal generation with the live parameters on that data window |
| **REQ-V06** | Find the matching signal on the last bar. Compare direction, entry price, SL price, TP price |

### 8.4 Matching & Verdicts

| Requirement | Description |
|---|---|
| **REQ-V07** | **MATCH**: SL and TP both within 0.5 pips of the backtester's expected values |
| **REQ-V08** | **CLOSE**: Entry price differs by 5-15 pips (normal spread/slippage) but SL/TP match |
| **REQ-V09** | **MISMATCH**: SL or TP differs by > 0.5 pips, OR no matching signal found on replay. This indicates a bug |
| **REQ-V10** | Report: match rate (X/Y trades matched), list of mismatches with details |

### 8.5 Statistical Analysis

| Requirement | Description |
|---|---|
| **REQ-V11** | **Win rate z-test**: Compare live win rate against expected (from backtest). Flag if statistically significantly worse (z < -1.96 at 95% CI) |
| **REQ-V12** | **Streak analysis**: Report current winning/losing streak, probability of the streak occurring by chance, maximum observed loss streak, and last 20 trades as W/L sequence |

### 8.6 P&L Reconciliation

| Requirement | Description |
|---|---|
| **REQ-V13** | For each matched trade, compute expected P&L from the backtester model and compare to actual realized P&L. Report the difference in dollars and pips |
| **REQ-V14** | Aggregate: total P&L, win/loss count, payoff ratio, profit factor, trades per month |
| **REQ-V15** | Track entry slippage in pips and cost. Flag trades with slippage > 5 pips for investigation |

### 8.7 Management Validation

| Requirement | Description |
|---|---|
| **REQ-V16** | Compare live SL modifications (trailing, breakeven) against backtester expected modifications. Report match rate for each management action type |
| **REQ-V17** | Detect missing or extra management actions |

### 8.8 Reporting

| Requirement | Description |
|---|---|
| **REQ-V18** | Output a summary table: each trade with live values, expected values, pip differences, and verdict (MATCH/CLOSE/MISMATCH) |
| **REQ-V19** | Flag: unmatched trades, rejected orders, insufficient margin events |

---

## FR-9: Monitoring Dashboard

### 9.1 Outcomes

The monitoring dashboard must provide real-time, at-a-glance visibility into the entire portfolio: all deployed strategy instances, their P&L, health status, trade history, and performance vs expectations. It must be accessible remotely via a web browser.

### 9.2 Architecture

| Requirement | Description |
|---|---|
| **REQ-M01** | A backend API server that reads state from all strategy instance directories and serves aggregated data via REST endpoints |
| **REQ-M02** | A frontend web application (static, deployable to a CDN) that consumes the API and renders the dashboard. Must work in modern browsers without server-side rendering |
| **REQ-M03** | The frontend must proxy API calls through a CDN/hosting service to avoid exposing the backend server's IP directly |
| **REQ-M04** | Service control endpoints (start/stop/restart strategy instances) must require authentication (e.g., HTTP Basic auth) |
| **REQ-M05** | Read-only endpoints (status, trades, performance) may be public or optionally authenticated |

### 9.3 Summary Panel (Always Visible)

| Requirement | Description |
|---|---|
| **REQ-M06** | Display top-level summary cards: Daily P&L (color-coded green/red), Open Positions, Trades Today, Running Instances (X/Y format with warning if not all running), Error Count (clickable to expand error panel) |
| **REQ-M07** | Each summary card must include a sparkline chart showing the last 7 days of trend data |

### 9.4 Charts & Visualizations

| Requirement | Description |
|---|---|
| **REQ-M08** | **Portfolio equity curve**: Time-series chart of cumulative P&L with range filters (1W, 1M, 3M, ALL) |
| **REQ-M09** | **Daily P&L bar chart**: Daily profit/loss bars showing daily volatility |
| **REQ-M10** | **Strategy ranking chart**: Bar chart ranking all strategies by total P&L, highlighting top and bottom performers |
| **REQ-M11** | **Currency pair heatmap**: Grid of currency pair P&L (today), color-coded green/red/neutral |

### 9.5 Strategy Table (Filterable, Sortable)

| Requirement | Description |
|---|---|
| **REQ-M12** | Sortable columns: Status (with heartbeat indicator), Strategy/Pair (with timeframe badge), Heartbeat age (human-readable: "5m ago"), Open Positions, P&L (color-coded), Wins/Losses, Action buttons |
| **REQ-M13** | Filtering: text search (by name, pair, ID), strategy-type filter buttons, grouping toggle (group by strategy type with collapsible headers) |
| **REQ-M14** | Compact mode toggle for dense view |
| **REQ-M15** | Row styling: left-border accent color (green/red/gray), status dots (green=running, red=stopped, yellow=stale, gray=unknown), dimmed rows for disabled instances |

### 9.6 Heartbeat Monitoring

| Requirement | Description |
|---|---|
| **REQ-M16** | Detect stale instances by comparing heartbeat age against timeframe-aware thresholds (e.g., M15 instance stale after 3 minutes, H1 after 90 minutes, H4 after 5 hours) |
| **REQ-M17** | Display status indicators: Running (green pulse), Stale (yellow), Stopped (red), Disabled (gray) |

### 9.7 Performance vs Backtest Comparison

| Requirement | Description |
|---|---|
| **REQ-M18** | For each instance, show a collapsible comparison panel: live vs expected metrics for win rate, profit factor, trades/month, max drawdown, total trades |
| **REQ-M19** | Color-code each metric: green (within 10% of expected), yellow (10-25% deviation), red (>25% deviation), gray (no baseline data) |
| **REQ-M20** | Include a bar chart comparing live vs backtest metrics visually |

### 9.8 Trade History

| Requirement | Description |
|---|---|
| **REQ-M21** | Collapsible trade history panel per instance showing the last 50 trades: entry/exit times, prices, P&L, exit reason |
| **REQ-M22** | CSV export button per instance: download all trade data as a CSV file |

### 9.9 Alerts System

| Requirement | Description |
|---|---|
| **REQ-M23** | **Best/Worst Today**: Top 3 and bottom 3 strategies by daily P&L |
| **REQ-M24** | **Losing streak alert**: Triggered when an instance has 3+ consecutive losing trades |
| **REQ-M25** | **Drawdown anomaly alert**: Triggered when live drawdown exceeds 1.5x expected drawdown |
| **REQ-M26** | **Trade frequency alert**: Triggered when actual trade frequency < 40% or > 250% of expected frequency |
| **REQ-M27** | **Error panel**: Auto-appears when errors > 0. Shows per-instance error count, service status, and last error message. Collapsible |

### 9.10 Service Control

| Requirement | Description |
|---|---|
| **REQ-M28** | Buttons per instance: Start, Restart, Stop (authenticated) |
| **REQ-M29** | Visual feedback on action success/failure |

### 9.11 Auto-Refresh

| Requirement | Description |
|---|---|
| **REQ-M30** | Auto-refresh at a configurable interval (default 15 seconds) |
| **REQ-M31** | Manual refresh button for immediate update |
| **REQ-M32** | Visual indicator showing last update time and network activity |

### 9.12 Responsive Design

| Requirement | Description |
|---|---|
| **REQ-M33** | Desktop (full layout), tablet (single-column charts), mobile (hide non-essential columns, compact card grid) |
| **REQ-M34** | Dark mode color scheme |

### 9.13 Pipeline Scan Tracker

| Requirement | Description |
|---|---|
| **REQ-M35** | If an optimization pipeline is running, show its progress: current phase, progress percentage, GREEN count so far, candidate list |

---

## FR-10: Strategy Research Factory

### 10.1 Outcomes

The research factory provides a systematic workflow for discovering, evaluating, and building new trading strategies from external sources (articles, papers, forums). It tracks all sources, their triage status, and any strategies built from them.

### 10.2 Source Tracking

| Requirement | Description |
|---|---|
| **REQ-F01** | Maintain a tracker database (e.g., JSON file) of all external sources (articles, papers) with: URL, title, status (unreviewed/viable/not_a_strategy/not_compatible/built), review date, linked strategy ID, and skip reason |
| **REQ-F02** | Track 80+ sources initially, with the ability to add more over time |
| **REQ-F03** | Record backtester gaps/limitations (variable lot sizing, hedging, etc.) with priority and list of blocked sources that require each capability |

### 10.3 Triage Workflow

| Requirement | Description |
|---|---|
| **REQ-F04** | For each unreviewed source: fetch the article, extract entry/exit rules, indicators, and parameters |
| **REQ-F05** | Classify the source: (a) `not_a_strategy` — UI tutorial, helper article, no trade logic; (b) `not_compatible` — requires unsupported backtester capability (e.g., variable lot sizing); (c) `viable` — has clear entry/exit logic that can be implemented and tested |
| **REQ-F06** | When a source requires an unsupported capability, DO NOT silently skip it. Flag it to the user and add the source ID to the relevant backtester gap's blocked list |
| **REQ-F07** | Check for duplicates: compare the core concept against existing strategies before building |

### 10.4 Strategy Build Workflow

| Requirement | Description |
|---|---|
| **REQ-F08** | Write a research note documenting: source, concept, entry/exit logic, mapping to the strategy framework, and any adaptations made |
| **REQ-F09** | Implement the strategy using the FastStrategy interface (REQ-S01 through REQ-S08), including vectorized fast path |
| **REQ-F10** | Use standard parameter groups (risk, management, time) from the helper library. Only signal and filter groups are strategy-specific |
| **REQ-F11** | Set all management parameters to OFF by default (pure signal quality testing first) |
| **REQ-F12** | Register the strategy in the central registry with multiple name aliases |
| **REQ-F13** | Validate the implementation: confirm parameter groups load, total search space is reasonable, and precompute runs without error |

### 10.5 Tracker Updates

| Requirement | Description |
|---|---|
| **REQ-F14** | After building, update the source status to `built` with the strategy ID and build date |
| **REQ-F15** | Add a strategy entry to the tracker with: ID, class name, status (`built`), parameter count, group structure, and build date |
| **REQ-F16** | Report the strategy as ready for pipeline testing with the command to run it |

---

## FR-11: Reporting & Analytics

### 11.1 Outcomes

The reporting system generates comprehensive, interactive reports for each pipeline run and maintains a cross-run leaderboard for comparing results over time.

### 11.2 Per-Run Report

| Requirement | Description |
|---|---|
| **REQ-A01** | Generate a self-contained HTML report (all assets embedded, no external dependencies) that can be opened in any browser |
| **REQ-A02** | Report must include interactive charts: equity curve (back + forward), daily P&L, drawdown curve, Monte Carlo distributions (return and max DD histograms), stability heatmap |
| **REQ-A03** | Report must include tables: top candidates ranked by confidence score, per-candidate metrics breakdown (back, forward, walk-forward, stability, Monte Carlo, confidence components) |
| **REQ-A04** | Report must include the final verdict: confidence score, rating (RED/YELLOW/GREEN), and recommended action |
| **REQ-A05** | Report must include trade-level details for the best candidate: entry/exit bars, direction, P&L per trade, exit reason |
| **REQ-A06** | Generate a structured data file (JSON) alongside the HTML for programmatic consumption |

### 11.3 Cross-Run Leaderboard

| Requirement | Description |
|---|---|
| **REQ-A07** | Maintain a leaderboard file aggregating results from all pipeline runs: pair, timeframe, strategy, confidence score, rating, key metrics, run timestamp |
| **REQ-A08** | The leaderboard must be sortable and filterable by score, rating, pair, strategy, and timeframe |
| **REQ-A09** | Optionally auto-sync the leaderboard to the remote dashboard after each pipeline run |

### 11.4 Pipeline Scan Report

| Requirement | Description |
|---|---|
| **REQ-A10** | Support multi-pair batch scanning: run the pipeline across multiple pairs sequentially or in parallel, logging confidence scores and ratings per pair |
| **REQ-A11** | Generate a scan summary: pair, score, rating, best candidate params, scan duration |

---

## FR-12: Deployment & Operations

### 12.1 Outcomes

The deployment system must enable 24/7 unattended operation of all trading instances and the dashboard, with auto-restart on crashes, centralized configuration, and remote update capability.

### 12.2 Strategy Configuration

| Requirement | Description |
|---|---|
| **REQ-X01** | A master configuration file defining all deployed strategies: instance ID, strategy name, pair, timeframe, risk percentage, enabled/disabled flag, source pipeline run ID, description (including confidence score and rating) |
| **REQ-X02** | Adding, removing, or modifying strategies must be achievable by editing the master config and running an update command |

### 12.3 Service Management

| Requirement | Description |
|---|---|
| **REQ-X03** | Each strategy instance must run as a managed service that auto-restarts on crash (with configurable restart delay, e.g., 30 seconds) |
| **REQ-X04** | The dashboard must also run as a managed service with auto-restart |
| **REQ-X05** | Log rotation must be configured to prevent disk space exhaustion (e.g., 10MB per log file, 1 backup) |
| **REQ-X06** | Each instance must have its own state directory, log directory, and configuration |

### 12.4 Setup & Updates

| Requirement | Description |
|---|---|
| **REQ-X07** | A first-time setup script that: verifies runtime version, installs dependencies, configures credentials (interactive prompts), tests broker connection, registers all services, and starts them |
| **REQ-X08** | An update script that: pulls latest code, reinstalls services with updated configuration, and restarts all instances |
| **REQ-X09** | The update process must stop old services gracefully before reinstalling |

### 12.5 Overnight Automation

| Requirement | Description |
|---|---|
| **REQ-X10** | An overnight batch script that: downloads/updates M1 data for all 25+ pairs, then runs the optimization pipeline for each pair with configurable speed preset |
| **REQ-X11** | Progress tracking: save download and pipeline status per pair to a resumable progress file. If the overnight run crashes, it must resume from the last incomplete pair |
| **REQ-X12** | Per-pair timeout (e.g., 1 hour) to prevent a single stuck pair from blocking the entire batch |

### 12.6 Parameter Export

| Requirement | Description |
|---|---|
| **REQ-X13** | An export tool that extracts the best candidate's parameters from a pipeline run directory and writes them to a JSON configuration file suitable for the live trader |
| **REQ-X14** | The exported file must include: strategy name, pair, timeframe, all optimized parameters, the pipeline run ID, and the confidence score/rating |

### 12.7 Health Monitoring

| Requirement | Description |
|---|---|
| **REQ-X15** | A monitoring script that scans all instance heartbeat files and alerts (via push notification) if: an instance's heartbeat is stale (configurable threshold, default 5 minutes), or an instance's error count exceeds a threshold (default 3) |
| **REQ-X16** | Support one-time check mode and continuous polling mode |
| **REQ-X17** | Support a summary mode that prints all instance statuses |

### 12.8 Multi-Pair Validation

| Requirement | Description |
|---|---|
| **REQ-X18** | A tool to test a strategy across multiple pairs in parallel: validates the strategy works beyond a single pair (guards against pair-specific overfitting) |
| **REQ-X19** | Report results per pair: score, rating, trade count, key metrics |

### 12.9 Containerization (Optional)

| Requirement | Description |
|---|---|
| **REQ-X20** | Optionally support containerized deployment: auto-generate container definitions from the master strategy configuration |
| **REQ-X21** | Each strategy instance maps to one container with its own state volume |

---

## NFR: Non-Functional Requirements

### Performance

| Requirement | Description |
|---|---|
| **NFR-P01** | **CPU utilization**: Optimization and backtesting must saturate all available CPU cores. On a 16-core laptop, at least 12 cores must be actively computing during optimization |
| **NFR-P02** | **Memory efficiency**: Shared memory for price data arrays. Per-worker overhead must not exceed ~200MB. Total memory for 12 workers must not exceed ~4GB (excluding base process) |
| **NFR-P03** | **Optimization throughput**: Minimum 300 parameter evaluations per second with 8+ workers |
| **NFR-P04** | **Backtest latency**: Sub-millisecond per parameter evaluation on a single core |
| **NFR-P05** | **Monte Carlo speed**: 1000+ iterations in < 100ms using parallel execution |
| **NFR-P06** | **Pipeline total time**: Full 7-stage pipeline must complete in < 3 minutes with default settings; < 25 seconds in turbo mode |
| **NFR-P07** | **IPC overhead**: Batch trial submissions to amortize per-roundtrip overhead. Target < 5% overhead from inter-process communication |
| **NFR-P08** | **Worker startup**: Worker pool created once and reused across all optimization stages. No per-stage pool creation |

### Reliability

| Requirement | Description |
|---|---|
| **NFR-R01** | **Crash recovery**: All long-running processes (pipeline, live trader, downloads) must persist state to disk and resume from the last checkpoint |
| **NFR-R02** | **Atomic writes**: All state file writes must use write-to-temp-then-rename to prevent corruption |
| **NFR-R03** | **Error isolation**: A single strategy instance failure must not affect other instances |
| **NFR-R04** | **Auto-restart**: Managed services must auto-restart after crashes with configurable delay |
| **NFR-R05** | **Graceful shutdown**: All processes must handle termination signals, save state, and exit cleanly |
| **NFR-R06** | **Retry on transient failure**: Pipeline supervisor must retry in a fresh process on JIT segfaults or access violations. Max configurable retries |

### Security

| Requirement | Description |
|---|---|
| **NFR-S01** | **API key storage**: Broker API keys stored in environment files excluded from version control |
| **NFR-S02** | **Key redaction**: API keys must be redacted in all log output (show only last 8 characters) |
| **NFR-S03** | **Dashboard auth**: Service control endpoints (start/stop/restart) must require authentication |
| **NFR-S04** | **Live trading confirmation**: Switching from paper to live trading must require explicit confirmation |

### Observability

| Requirement | Description |
|---|---|
| **NFR-O01** | **Structured logging**: All components must use structured, timestamped logging with configurable verbosity |
| **NFR-O02** | **Per-instance logs**: Each trading instance writes to its own log file |
| **NFR-O03** | **Pipeline progress**: Each pipeline stage must log start, progress, completion time, and key results |
| **NFR-O04** | **Heartbeat monitoring**: All live trading instances write heartbeat files readable by the monitoring system |

### Maintainability

| Requirement | Description |
|---|---|
| **NFR-M01** | **Modular architecture**: Broker integration, strategy logic, backtesting, optimization, and live trading must be separate modules. Swapping the broker must not require changes to strategy or pipeline code |
| **NFR-M02** | **Strategy isolation**: Adding a new strategy must require only: (a) implementing the strategy interface, (b) registering in the central registry. No changes to the optimizer, pipeline, or live trader |
| **NFR-M03** | **Configuration hierarchy**: Configuration must flow from defaults -> config file -> CLI arguments, with each level overriding the previous |

---

## Appendix A: Quality Score Specification

### Formula

```
Quality Score = (Sortino * R_squared * min(PF, 5) * sqrt(min(Trades, 200)) * (1 + min(Return_pct, 200) / 100))
                / (Ulcer + MaxDD_pct / 2 + 5)
```

### Component Rationale

| Component | Role | Why This Metric |
|---|---|---|
| **Sortino** | Risk-adjusted return (downside only) | Doesn't penalize upside volatility. Better for asymmetric strategies (trend following, big risk:reward) than Sharpe |
| **R-squared** | Equity curve smoothness | Two strategies with identical Sharpe can have wildly different equity curves. R-squared distinguishes smooth linear growth from erratic swings with recovery |
| **Profit Factor** (capped at 5) | Trade efficiency | Gross profit / gross loss. Capped to prevent single-outlier-trade dominance |
| **sqrt(Trades)** (capped at sqrt(200)) | Statistical confidence | More trades = more confident the result isn't luck. Square root prevents high-frequency from dominating. Cap at 200 equalizes M15 (478 trades) and M5 (2000 trades) |
| **Return multiplier** (1x to 3x) | Absolute profitability | Rewards strategies that actually make money, not just have good ratios. Capped at 200% to prevent compound-sizing inflation on fast timeframes |
| **Ulcer Index** | Chronic drawdown pain | RMS of per-bar drawdown percentages. Captures time spent underwater. A strategy with repeated 5% drawdowns scores worse than one with a single 10% drawdown that recovers quickly |
| **MaxDD/2** | Peak drawdown severity | Worst single decline. Halved to balance with Ulcer Index |
| **Base of 5** | Denominator stability | Prevents division-by-zero when both Ulcer and MaxDD are very small |

### Zero Conditions

Returns 0 when ANY of these are true:
- Sortino <= 0
- Profit Factor <= 0
- R-squared <= 0
- No trades

### Hard Pre-Filters (applied before scoring)

- Max Drawdown > 30% -> instant reject
- R-squared < 0.5 -> instant reject

---

## Appendix B: Confidence Score Specification

### Architecture: 6 Components + DOF Penalty

```
Total Score = sum(component_i * weight_i) - DOF_penalty

Range: 0-100
```

### Component Weights

| Component | Weight | Input Data | Why This Weight |
|---|---|---|---|
| Walk-Forward Consistency | 30% | OOS pass rate, geometric mean QS, min QS, QS CV, window count | Strongest out-of-sample time-series evidence. The gold standard for robustness |
| Monte Carlo Risk | 25% | PSR, bootstrap CI, permutation p-value, trade-skip resilience, DD distribution, CI width | Statistical significance and stress testing. Multiple independent tests of whether the result is real |
| Parameter Stability | 15% | Mean stability ratio, worst single parameter, count of unstable params | Fragile parameters suggest curve-fitting. Adjacent parameter values should produce similar results |
| Forward/Back Ratio | 10% | Forward QS / Back QS | Direct overfitting measure. Ratio near 1.0 means the strategy works beyond training data |
| Backtest Quality | 10% | Profit factor, Sortino, R-squared, trade count, blended drawdown | Absolute quality of the backtest results (not relative). A baseline quality check |
| Quality Score | 10% | Forward-test QS (or WF average fallback) | Direct use of the universal quality metric as a sanity check on absolute performance |

### Walk-Forward Component (30% of total) — Sub-Scores

| Sub-Score | Weight | Formula |
|---|---|---|
| OOS Pass Rate | 30% | `pass_rate * 100` |
| Geometric Mean QS | 25% | `geo_mean / 3.0 * 100` (0->0, 3.0+->100) |
| Min Window QS | 20% | `min_qs / 2.0 * 100` (can't hide weak windows) |
| QS Coefficient of Variation | 15% | `(2.0 - CV) / 2.0 * 100` (CV 0->100, CV 2.0->0) |
| Window Count Bonus | 10% | `n_windows / 10 * 100` (more evidence -> higher score) |

### Monte Carlo Component (25% of total) — Sub-Scores

| Sub-Score | Weight | Formula |
|---|---|---|
| Bootstrap Sharpe CI Lower | 25% | `CI_lower / 2.0 * 100` |
| PSR (>0) | 20% | `PSR * 100` |
| Permutation p-value | 20% | p<0.01->100, p<0.05->75, p<0.10->50, p>0.20->0 |
| Trade-Skip Resilience | 15% | `(1 - relative_degradation) * 100` |
| DD Distribution 95th pctile | 10% | `(30 - pct_95_dd) / 30 * 100` |
| Bootstrap CI Width | 10% | `(4.0 - width) / 4.0 * 100` |

### DOF Penalty (Bailey & Lopez de Prado, 2014)

```
expected_inflation = sqrt(2 * ln(N_trials_tested))
ratio = expected_inflation / observed_sharpe

ratio <= 0.3: penalty = 0
ratio 0.3-1.0: penalty = (ratio - 0.3) / 0.7 * 15
ratio > 1.0: penalty = min(20, 15 + (ratio - 1.0) * 10)

Maximum penalty: 20 points
```

### Ulcer Index Caps

| Ulcer Index | Score Cap | Rating Cap |
|---|---|---|
| > 10.0 | 70 | YELLOW maximum |
| > 20.0 | 40 | RED maximum |

### Rating Thresholds

| Score Range | Rating | Recommended Action |
|---|---|---|
| 0-39 | RED | Do not trade. Re-examine strategy fundamentals |
| 40-69 | YELLOW | Proceed with extreme caution. Re-optimize or paper trade extensively |
| 70-84 | GREEN | Paper trade first, then small live position |
| 85-100 | GREEN | Ready for live trading with appropriate position sizing |

---

## Appendix C: Metrics Glossary

| Metric | Definition |
|---|---|
| **ATR (Average True Range)** | Average of true ranges (max of: high-low, abs(high-prev_close), abs(low-prev_close)) over N periods. Measures volatility |
| **Confidence Score** | Composite 0-100 score combining walk-forward, Monte Carlo, stability, forward/back ratio, backtest quality, and quality score. The pipeline's final verdict |
| **DOF Penalty** | Degrees-of-freedom adjustment that reduces the confidence score based on how many parameter combinations were tested. Prevents data-mining bias |
| **Drawdown** | Decline from peak equity to subsequent trough, expressed as a percentage |
| **Equity Curve** | Cumulative P&L plotted over time |
| **In-Sample (IS)** | Data or time period used during optimization. Results on IS data may be overfit |
| **Max Drawdown (MaxDD)** | Largest peak-to-trough percentage decline during the test period |
| **Monte Carlo Simulation** | Statistical technique using random resampling to estimate probability distributions of outcomes |
| **Out-of-Sample (OOS)** | Data or time period NOT used during optimization. Performance on OOS data is the gold standard for robustness |
| **Pip** | Smallest price increment for a currency pair. Typically 0.0001 for most pairs, 0.01 for JPY pairs |
| **Probabilistic Sharpe Ratio (PSR)** | Probability that the true Sharpe ratio exceeds a benchmark, accounting for sample size, skewness, and kurtosis |
| **Profit Factor (PF)** | Gross profit divided by gross loss. PF > 1 means the strategy is profitable |
| **Quality Score** | Composite metric: Sortino * R-squared * PF * sqrt(Trades) * Return / (Ulcer + DD/2 + 5). The universal optimization objective |
| **R-squared** | Coefficient of determination of the equity curve's linear regression. Measures how closely the equity curve follows a straight line (0=random, 1=perfectly linear) |
| **Sharpe Ratio** | (Annualized return - risk-free rate) / annualized standard deviation. Measures risk-adjusted return |
| **Sortino Ratio** | Like Sharpe but only penalizes downside volatility. Better for asymmetric return distributions |
| **Ulcer Index** | Root-mean-square of all drawdown percentages across all bars. Measures chronic underwater pain, not just peak depth |
| **Walk-Forward Analysis** | Rolling window validation where parameters are tested on multiple consecutive out-of-sample periods to verify robustness across market regimes |
| **Walk-Forward Efficiency (WFE)** | Forward QS / Back QS. Measures how well backtest results hold up out-of-sample. < 0.4 = strong overfitting, 0.8-1.0 = excellent robustness |

---

*End of PRD*
