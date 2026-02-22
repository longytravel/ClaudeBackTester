# Current Task

## Status: Infrastructure Reset — Updating to IC Markets / Dukascopy / MT5

## What Changed (from previous sessions, now documented)
- **Broker**: OANDA → IC Markets Global (Seychelles, MT5, 1:500 leverage)
- **Historical Data**: OANDA API → Dukascopy (free 15+ year tick/M1 data)
- **Execution**: REST API → MetaTrader 5 Python library
- **Deployment**: Local only → Local dev + VPS production
- **Data Storage**: `G:\My Drive\BackTestData` (Google Drive)
- Full details in `INFRASTRUCTURE.md`

## Data Already Downloaded
- EUR_USD: M1 (2005-2026) + all higher TFs — COMPLETE
- GBP_USD: M1 (2005-2026) + all higher TFs — COMPLETE
- AUD_USD: M1 chunks started (19 years) — IN PROGRESS
- Still need: USD_JPY, XAU_USD, and remaining majors/crosses

## Next Steps (in order)
1. Get Python/uv environment working (verify deps install)
2. Rebuild Dukascopy data download pipeline (`backtester/data/`)
   - Downloader that fetches M1 data from Dukascopy
   - Yearly chunk storage with crash recovery
   - Timeframe conversion (M1 → M5, M15, M30, H1, H4, D, W)
   - Data validation (gaps, quality score)
   - Resume capability for partially downloaded pairs
3. Build MT5 broker abstraction (`backtester/broker/`)
   - Connect to IC Markets demo account
   - Fetch live candles, place orders, manage positions
4. Continue with Phase 2 (Strategy Framework) onward per PRD

## Last Completed
- Phase 0: Project scaffold, Numba+TBB parallel verified (11.5x speedup)
- Infrastructure documented: IC Markets, Dukascopy, MT5, VPS deployment
- Existing data preserved on G: drive

## Blockers
- Need to verify uv/Python environment is set up correctly on this machine
- MetaTrader5 Python package only works on Windows (VPS deployment needs consideration)
