# Current Task

## Status: Phase 1-4 Code Review Complete — Ready for Phase 5

## What's Built
- **Phase 1**: Data pipeline (Dukascopy downloader, timeframes, validation, splitting, MT5 broker)
- **Phase 2**: Strategy framework (14 indicators, base class, param space, SL/TP calc, registry)
- **Phase 3**: Backtest engine (JIT batch evaluator, metrics, encoding, telemetry, orchestrator)
- **Phase 4**: Parameter optimizer (Sobol/EDA samplers, staged optimization, ranking, diversity archive)
- **Execution Cost Modeling**: SELL exit spread, round-trip commission, max spread filter
- **Live Data Validation**: RSI mean reversion strategy tested on 96K bars EUR/USD H1 (2007-2026)
- **Full Code Review**: 7 critical bugs fixed across 10 files (commit 8adae74)
- **233 tests passing**

## Code Review Summary (Feb 2026)
Full review of all 4 phases found 7 critical bugs, all fixed:

### Critical Bugs Fixed (commit 8adae74)
1. **JIT pnl_buffers allocation inside prange** — violated zero-allocation rule, caused worse-than-single-threaded perf. Moved pre-allocation outside parallel region.
2. **RSI variant value bug** — `variants.append(period_idx)` stored 0-3 index instead of actual RSI period (7/9/14/21). Variant matching always failed.
3. **Filter PL layout defaults to column 0** — when strategies don't define filter params, layout pointed to column 0 (unrelated param), causing SELL signals to be incorrectly filtered. Fixed: filter PL slots default to -1, JIT checks `>= 0` before reading.
4. **Telemetry missing signal filtering** — telemetry didn't filter by variant/threshold, causing JIT vs telemetry mismatch for strategies with signal filtering.
5. **Telemetry missing stale exit** — stale exit logic was absent from the Python mirror.
6. **Telemetry missing partial close** — partial close logic was absent from the Python mirror.
7. **Duplicate ParamDefs in RSI strategy** — `signal_variant`, `buy_filter_max`, `sell_filter_min` duplicated the engine-mapped `rsi_period`, `rsi_oversold`, `rsi_overbought`. Removed duplicates.

### Medium-Severity Issues (noted, not blocking Phase 5)
- **Phase 1 data**: Weekly resampling anchor day (`"1W"` should be `"W-FRI"`), NaN spread handling, limited holiday detection
- **Phase 4 optimizer `run.py`**: Ranking, diversity, and gating code is called but integration is minimal. Refinement stage doesn't seed EDA with best params. DSR computed but not used as a hard gate.
- **Phase 4 archive**: Grid is coarse (12 cells max) — fine for MVP, may need more dimensions later

## Deferred / Post-MVP
- Increment 11: CLI integration (`bt optimize` command) — not blocking Phase 5
- Increment 12: Cross-Entropy exploitation upgrade
- Increment 13: Throughput benchmarking, batch size tuning
- REQ-B03: Grid mode (multiple concurrent positions) — fundamentally different simulation model
- REQ-O05: Real-data throughput benchmark
- Variable slippage by volatility (currently fixed 0.5 pips)
- Swap/financing costs (negligible for <24h hold time)

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
