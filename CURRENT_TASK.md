# Current Task

## Status: System Accurate & Deployed to Practice — Ready for Live Verification

All phases through Phase 6 are built. Rust backend verified accurate. Telemetry matches batch evaluator. Live trading infrastructure deployed to practice account. Next: verify live trades match backtest predictions.

## What's Built

### Phases 1-4: Core System
- **Phase 1**: Data pipeline (Dukascopy downloader, timeframes, validation, splitting, MT5 broker)
- **Phase 2**: Strategy framework (14 indicators, base class, param space, SL/TP calc, registry)
- **Phase 3**: Backtest engine (Rust batch evaluator, metrics, encoding, telemetry, orchestrator)
- **Phase 4**: Parameter optimizer (Sobol/EDA samplers, staged optimization, ranking, diversity archive)
- **Execution Cost Modeling**: SELL exit spread, round-trip commission, max spread filter

### Phase 5: Validation Pipeline
- Walk-forward, stability, Monte Carlo, confidence scoring, checkpoint/resume, JSON reports
- **VP-1**: CPCV — C(N,k) purged cross-validation, 45 folds
- **VP-2**: Multi-candidate pipeline — optimizer returns top N, pipeline validates all
- **VP-3**: Regime-aware validation — ADX+NATR 4-quadrant, per-regime stats
- **OPT-1**: Adaptive LR + entropy diagnostics for EDA sampler
- **Causality Contract**: SignalCausality enum, guards, verification tests

### Rust Backend
- PyO3 + Rayon native extension replaces Numba JIT hot loop
- 2.7x faster than Numba (EXEC_FULL), handles M1 without segfaults
- Bit-for-bit parity verified against Numba
- Telemetry matches batch evaluator at 0.0000 diff (11 parity tests)
- Subprocess isolation removed (no longer needed)

### Phase 6: Live Trading (BUILT)
- **Live trader** (`backtester/live/trader.py`, 568 lines): Candle-close event loop, MT5 integration, signal generation, order placement, position management, broker sync, state persistence, heartbeat, audit trail
- **Position manager** (`backtester/live/position_manager.py`, 220 lines): Mirrors backtest check order — max bars, stale, breakeven, trailing, partial close
- **Risk manager** (`backtester/risk/manager.py`, 187 lines): Pre-trade checks (circuit breaker, daily limits, drawdown, max positions, spread filter), risk-based position sizing
- **Broker integration** (`backtester/broker/mt5.py` + `mt5_orders.py`, 556 lines): MT5 connect, candle fetch, market orders, modify SL/TP, partial close, position queries
- **Deployment** (`DEPLOY.bat`, `start_all.py`, `stop_all.py`, `status_all.py`): One-click VPS deploy, safe restart (doesn't interrupt running traders), process monitoring
- **Types & state** (`types.py`, `config.py`, `state.py`): LivePosition, DailyStats, TraderState, TradingMode, atomic state persistence
- **Tests**: 3 test files (test_live_state.py, test_position_manager.py, test_risk_manager.py)
- **533 tests passing**

### Deployed Strategies (Practice Mode)
- ema_crossover EUR/USD M15
- ema_crossover EUR/USD H1

## Completed Milestones
- [x] Phases 1-4: Core system (data, strategies, engine, optimizer)
- [x] Phase 5: Validation pipeline (walk-forward, CPCV, Monte Carlo, stability, confidence, regime)
- [x] Phase 5b: Enhancements (VP-1, VP-2, VP-3, OPT-1, causality contract)
- [x] Rust backend: PyO3 + Rayon, M1 stable, 2.7x faster, subprocess isolation removed
- [x] Accuracy verification: Rust == Numba, telemetry == batch evaluator (0.0000 diff)
- [x] Phase 6: Live trading engine, risk management, broker integration, deployment scripts
- [x] Practice deployment: ema_crossover running on demo account

## Old Saved Results — INVALID
Results in `results/` were generated before:
- Cost consistency fix (commit 7859907)
- Deferred SL fix (commit 1c616ea)
- Adverse exit slippage fix (commit 5c4e503)

These need to be re-run with the current (corrected) engine.

## Next Steps
1. **Verify live trades match backtest** — compare practice trade results against telemetry predictions
2. **Re-run strategies** with corrected Rust backend to generate valid pipeline results
3. **OPT-2: GT-Score Objective** — A/B test vs Quality Score
4. **OPT-3: Batch Size Auto-Tuning** — Benchmark and auto-select

## Blockers
- None
