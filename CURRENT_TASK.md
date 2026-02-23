# Current Task

## Status: Code Review Round 2 Complete — Ready for Phase 5

## What's Built
- **Phase 1**: Data pipeline (Dukascopy downloader, timeframes, validation, splitting, MT5 broker)
- **Phase 2**: Strategy framework (14 indicators, base class, param space, SL/TP calc, registry)
- **Phase 3**: Backtest engine (JIT batch evaluator, metrics, encoding, telemetry, orchestrator)
- **Phase 4**: Parameter optimizer (Sobol/EDA samplers, staged optimization, ranking, diversity archive)
- **Execution Cost Modeling**: SELL exit spread, round-trip commission, max spread filter
- **Live Data Validation**: RSI mean reversion strategy tested on 96K bars EUR/USD H1 (2007-2026)
- **Code Review Round 1**: 7 critical bugs fixed across 10 files (commit 8adae74)
- **Code Review Round 2**: 9 issues fixed from external reviewer feedback
- **253 tests passing**

## Code Review Round 2 Fixes (Feb 2026)
External developer review found additional issues, all fixed:

1. **RSI threshold crossover (CRITICAL)** — Strategy only generated signals at 35/65, making rsi_oversold and rsi_overbought params dead. Fixed: now generates at EACH threshold (20/25/30/35, 65/70/75/80) with filter_value = threshold value.
2. **JIT filter range→exact match (CRITICAL)** — Filter used `>` / `<` (range), but with threshold-as-filter-value design needs `!=` (exact match). Fixed in both JIT and telemetry.
3. **Dead params removed** — `atr_period` and `sma_filter_period` had no effect. Removed from RSI strategy.
4. **Staged optimizer best-stage** — `max(stages, key=quality)` could pick an early stage with partial params. Fixed: always use refinement result.
5. **Zero-passing stage guard** — If a stage finds no passing candidates (quality=-inf), don't lock zeros as "best". Fixed: skip locking, warn, let refinement explore.
6. **NaN spread sanitization** — NaN spreads from Dukascopy bid-only downloads could poison PnL. Added NaN→0 guards in JIT (basic + full), telemetry, and max_spread_filter.
7. **Time array warning** — Default-zero bar_hour/bar_day_of_week silently makes time filters ineffective. Added warning log.
8. **Weekly timeframe anchor** — `"1W"` defaults to Sunday. Changed to `"W-MON"` for FX weekly bars.

## Deferred / Post-MVP
- Increment 11: CLI integration (`bt optimize` command) — not blocking Phase 5
- Increment 12: Cross-Entropy exploitation upgrade
- Increment 13: Throughput benchmarking, batch size tuning
- REQ-B03: Grid mode (multiple concurrent positions) — fundamentally different simulation model
- REQ-O05: Real-data throughput benchmark
- Variable slippage by volatility (currently fixed 0.5 pips)
- Swap/financing costs (negligible for <24h hold time)
- Phase 4 optimizer `run.py`: ranking/diversity/gating integration is minimal, refinement EDA not seeded
- Phase 4 archive: coarse grid (12 cells max) — fine for MVP

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
