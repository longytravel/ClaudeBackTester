# Current Task

## Status: Phase 1 — Historical Data Management (FR-1) — NOT STARTED

## Next Steps (in order)
1. Build OANDA broker abstraction layer (`backtester/broker/`)
   - Base abstract class for broker API
   - OANDA implementation (connect, fetch candles, account info)
   - Test with practice account
2. Build data download module (`backtester/data/`)
   - REQ-D01 through REQ-D06: download OHLCV, rate limits, retries, progress, incremental
3. Build Parquet caching layer
   - REQ-D07 through REQ-D10: storage, naming, checkpointing, chunking
4. Timeframe conversion
   - REQ-D11 through REQ-D14: M1 source of truth, aggregation, fallback
5. Data validation
   - REQ-D15 through REQ-D19: gaps, zeros, anomalies, quality score, rejection
6. Data splitting
   - REQ-D20 through REQ-D22: back/forward split, holdout mode
7. CLI tooling
   - REQ-D25 through REQ-D26: download, update, status, rebuild commands

## Last Completed
- Phase 0: Project scaffold, Numba+TBB parallel verified (11.5x speedup)
- Initial commit: a373c36

## Blockers
- Need OANDA API key from user to test broker connectivity
