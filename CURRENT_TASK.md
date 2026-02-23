# Current Task

## Status: Phase 5 — Validation Pipeline Complete (MVP)

## What's Built
- **Phase 1**: Data pipeline (Dukascopy downloader, timeframes, validation, splitting, MT5 broker)
- **Phase 2**: Strategy framework (14 indicators, base class, param space, SL/TP calc, registry)
- **Phase 3**: Backtest engine (JIT batch evaluator, metrics, encoding, telemetry, orchestrator)
- **Phase 4**: Parameter optimizer (Sobol/EDA samplers, staged optimization, ranking, diversity archive)
- **Execution Cost Modeling**: SELL exit spread, round-trip commission, max spread filter
- **Phase 5**: Validation pipeline (walk-forward, stability, Monte Carlo, confidence scoring, checkpoint/resume, JSON reports)
- **362 tests passing**

## Phase 5 Summary (Feb 2026)
Research-informed design with improvements over the original PRD:

1. **Types + Config** (`pipeline/types.py`, `pipeline/config.py`) — All dataclasses, configurable thresholds
2. **Walk-Forward** (`pipeline/walk_forward.py`) — Rolling/anchored windows with embargo gap, IS/OOS labeling, per-window engine instantiation, 60% pass rate + 0.3 Sharpe gates, WFE metric
3. **Monte Carlo** (`pipeline/monte_carlo.py`) — Block bootstrap (preserves autocorrelation), sign-flip permutation test, trade-skip resilience (5%/10%), execution stress (+50% slippage, +30% commission), DSR hard gate (>= 0.95)
4. **Stability** (`pipeline/stability.py`) — +-3 step perturbation on forward data, ROBUST/MODERATE/FRAGILE/OVERFIT rating (advisory only)
5. **Confidence** (`pipeline/confidence.py`) — Sequential hard gates then 6-component weighted composite (0-100), RED/YELLOW/GREEN rating
6. **Checkpoint + Runner** (`pipeline/checkpoint.py`, `pipeline/runner.py`) — JSON checkpoint/resume, stage orchestration, JSON report output

Key PRD deviations:
- **DSR as hard gate** (not just scoring component) — research recommendation
- **Block bootstrap** instead of naive shuffle — preserves autocorrelation
- **+-3 steps** perturbation (not +-1) on forward data — catches fragile optima
- **Removed DOF penalty** — redundant with DSR
- **JSON report only** for MVP — HTML deferred to Phase 5b

## Deferred / Post-MVP
- Phase 5b: HTML report generation (REQ-P38-P40)
- Phase 5b: CPCV (Combinatorial Purged Cross-Validation)
- Pipeline CLI integration (`bt validate` command)
- Increment 11: CLI integration (`bt optimize` command)
- Increment 12: Cross-Entropy exploitation upgrade
- Increment 13: Throughput benchmarking, batch size tuning
- REQ-B03: Grid mode (multiple concurrent positions)
- Regime-aware validation (HMM, ADX/ATR)
- GT-Score as optimizer objective

## Next Steps: Phase 6 — Live Trading (FR-6)
1. **Live Trading Engine** (REQ-L01-L15): Trading loop, order execution, position management
2. **Risk Management** (REQ-R01-R13): Pre-trade checks, position sizing, circuit breaker
3. **Broker Integration** (REQ-L16-L30): MT5 order management, state sync

## Blockers
- None
