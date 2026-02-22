# Current Task

## Status: Phase 3 — Backtesting Engine (FR-3)

## What's Built
- **Phase 1**: Data pipeline (Dukascopy downloader, timeframes, validation, splitting, MT5 broker)
- **Phase 2**: Strategy framework (14 indicators, base class, param space, SL/TP calc, registry)
- **63 tests passing**

## Next Steps: Phase 3 Backtest Engine
1. **Core backtest loop** (`backtester/core/engine.py`)
   - Bar-by-bar simulation with SL/TP checking on intra-bar highs/lows
   - Trade management: trailing stop, breakeven, partial close, max bars, stale exit
   - Conservative same-bar SL/TP tiebreak (SL wins)
   - Configurable spread and slippage
2. **Metrics computation** (`backtester/core/metrics.py`)
   - Win rate, profit factor, Sharpe, Sortino, max drawdown, return%, R-squared, Ulcer Index
3. **Numba JIT hot loop** (`backtester/core/jit_loop.py`)
   - @njit(parallel=True) with prange over trials
   - Pre-allocated output arrays (zero allocation in prange)
   - Shared memory for price arrays
4. **Execution modes**
   - Basic (SL/TP only, 700+ evals/sec target)
   - Full (all management, 400+ evals/sec)

## Last Completed
- Phase 2 strategy framework committed with 38 tests
- All 14 PRD-required indicators implemented and tested
- SL/TP calculator with all 3 modes each + TP>=SL constraint

## Blockers
- None
