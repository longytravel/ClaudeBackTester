# Current Task

## Status: Phase 3+4 Complete + Live Data Validated — Ready for Phase 5

## What's Built
- **Phase 1**: Data pipeline (Dukascopy downloader, timeframes, validation, splitting, MT5 broker)
- **Phase 2**: Strategy framework (14 indicators, base class, param space, SL/TP calc, registry)
- **Phase 3**: Backtest engine (JIT batch evaluator, metrics, encoding, telemetry, orchestrator)
- **Phase 4**: Parameter optimizer (Sobol/EDA samplers, staged optimization, ranking, diversity archive)
- **Live Data Validation**: RSI mean reversion strategy tested on 96K bars EUR/USD H1 (2007-2026)
- **219 tests passing**

## Live Data Test Results (EUR/USD H1)
- 16,579 signals, all 10 JIT vs Telemetry metrics match exactly
- Throughput: 18K-35K evals/sec BASIC, 4K-10K evals/sec FULL (exceeds PRD targets)
- Staged optimization: 6,000 trials in 1.0s, found candidate with sharpe=8.05
- 5 bugs found and fixed through live data testing (spread, annualization, sell slippage, close price, MaxDD base)

## Bugs Fixed This Session
1. Spread double-scaling: spread already in price units, removed pip_value multiplication
2. Sharpe/Sortino annualization: sqrt(total_trades) → sqrt(trades_per_year)
3. JIT basic end-of-data: (H+L)/2 → actual close price
4. Telemetry sell slippage: was missing for SELL trades
5. MaxDD%/Ulcer mismatch: aligned Python metrics to match JIT base calculation

## Deferred / Post-MVP
- Increment 11: CLI integration (`bt optimize` command) — not blocking Phase 5
- Increment 12: Cross-Entropy exploitation upgrade
- Increment 13: Throughput benchmarking, batch size tuning
- REQ-B03: Grid mode (multiple concurrent positions) — fundamentally different simulation model
- REQ-O05: Real-data throughput benchmark

## Next Steps: Phase 5 — Validation Pipeline (FR-5)
1. **Pipeline Architecture** (REQ-P01-P06): 7-stage pipeline, timestamped dirs, checkpointing
2. **Walk-Forward** (REQ-P13-P21): Rolling windows, IS/OOS, per-window metrics
3. **Stability** (REQ-P22-P25): Parameter perturbation testing
4. **Monte Carlo** (REQ-P26-P32): Shuffle, bootstrap, permutation tests
5. **Confidence Scoring** (REQ-P33-P37): 6-component scoring, DOF penalty
6. **Reporting** (REQ-P38-P40): HTML report, JSON data
7. **Pipeline CLI** (REQ-P41-P43): Full CLI args, config override

## Blockers
- None
