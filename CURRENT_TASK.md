# Current Task

## Status: Phase 1 COMPLETE — Starting Phase 2 (Strategy Framework)

## What's Built
- **Data Pipeline** (Phase 1): Dukascopy downloader (bid+ask spread), timeframe conversion, validation, splitting, CLI
- **MT5 Broker Module** (`backtester/broker/mt5.py`): IC Markets connectivity, UTC timezone conversion, candle fetching
- **Data Verification**: Dukascopy vs IC Markets prices match within 0.1-0.3 pip (99%+ within 1 pip)
- **25 tests passing** (smoke + data pipeline + spread + splitting)

## Data Columns
All M1 data: `open, high, low, close, volume, spread`
- Prices are bid-side, spread = avg(ask-bid) per candle
- Higher timeframes use median spread

## Data Download Status
- Background download running for 25 pairs with `--force` (bid+ask spread)
- Completed: USD/JPY, USD/CAD
- Failed (need retry): EUR/USD, GBP/USD, AUD/USD, NZD/USD
- Run `uv run python scripts/download_retry.py` after batch completes

## Next Steps (in order)
1. **Start Phase 2: Strategy Framework** (FR-2)
   - Strategy base class with parameter space definition
   - Signal generation interface
   - Indicator library (numpy-based: RSI, ATR, SMA, EMA, BB, etc.)
   - SL/TP calculation modes
   - Trade management (trailing stop, breakeven, partial close)
2. **Phase 3: Backtest Engine** (FR-3)
   - Numba JIT core loop
   - Parallel eval across cores
   - Metrics computation

## Last Completed
- Phase 0: Scaffold, Numba+TBB parallel verified
- Phase 1: Full data pipeline + MT5 broker integration
- MT5 timezone handling: EET → UTC at ingestion boundary

## Blockers
- None — ready for Phase 2
