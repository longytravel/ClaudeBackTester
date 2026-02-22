# Current Task

## Status: Phase 1 Data Pipeline — COMPLETE (pending full re-download with spread)

## What's Built
- **Downloader** (`backtester/data/downloader.py`): Downloads bid + ask M1 data from Dukascopy, computes per-candle spread, stores as yearly Parquet chunks, consolidates
- **Timeframe conversion** (`backtester/data/timeframes.py`): M1 → M5, M15, M30, H1, H4, D, W with correct OHLCV aggregation + median spread
- **Validation** (`backtester/data/validation.py`): Gap detection (weekend/holiday-aware), anomaly checks, yearly coverage, quality scoring
- **CLI commands**: `bt data download`, `bt data update`, `bt data status`, `bt data build-timeframes`, `bt data validate`
- **20 tests passing** (smoke + data pipeline + spread computation)

## Data Columns
All M1 data now includes: `open, high, low, close, volume, spread`
- Prices are bid-side
- Spread = average of (ask_open - bid_open) and (ask_close - bid_close) per candle
- Higher timeframes use median spread for the period

## Data Download Status
- Previous downloads (bid-only) exist for some pairs on G: drive
- All pairs need re-downloading with `--force` to get bid+ask spread data
- 25 pairs total, ~2005-2026 each

## Next Steps (in order)
1. **Re-download all pairs with spread data** (`bt data download --force`)
   - Start with majors: EUR/USD, GBP/USD, USD/JPY, AUD/USD
   - This takes time (~10-15 min per pair) — run overnight or in batches
2. **Build MT5 broker abstraction** (`backtester/broker/`)
   - Connect to IC Markets demo account
   - Fetch live candles, place orders, manage positions
3. **Continue with Phase 2** (Strategy Framework) per PRD

## Last Completed
- Phase 0: Scaffold, Numba+TBB parallel verified (11.5x speedup)
- Phase 1: Full data pipeline with bid+ask spread support
- Environment: uv + Python 3.12 + all deps working

## Blockers
- MetaTrader5 Python package only works on Windows (VPS needs consideration)
