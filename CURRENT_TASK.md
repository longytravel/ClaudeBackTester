# Current Task

## Last Completed (Mar 15 2026)

### Multi-AI Quality Loop + Bug Fixes
- Built `/new-strategy` skill — 7-stage repeatable workflow (scrape → extract → build → review → test → smoke → backtest)
- Built `/review` skill — multi-AI code review (Codex GPT-5.4 + Gemini 3 + Claude Opus 4.6)
- Built `scripts/smoke_test_strategy.py` — integration test on real data + Rust engine (~1 sec)
- Built `scripts/scrape_mql5_article.py` — MQL5 article scraper
- Gemini CLI installed (0.33.1) and authenticated via Google Account (Pro plan)
- Ran first multi-AI review on all 8 strategies + 5 management modules
- Fixed 4 confirmed bugs (all verified by both Codex and Gemini):
  1. Entry price look-ahead bias in all 8 strategies (close → open[bar_idx+1] with np.where fallback for live)
  2. Breakeven offset > trigger in modules.py (capped offset values)
  3. Partial close missing slippage in Rust + telemetry
  4. Sell spread not proportional after partial close
- Redeployed to VPS — 3 traders running (EMA, MACD, Stochastic on EURUSD H1)

### Previous Work
- Modular Staged Optimizer: NUM_PL 27→64, ManagementModule system, sub-grouped stages
- Strategy Research Pipeline: 2,731 articles, 200 triaged (62 STRATEGY)
- First strategy: Hidden Smash Day (article 21391, Larry Williams)
- Live trading engine built and deployed on IC Markets demo

## Next Steps

### Step 1: Run `/new-strategy` on a Fresh Article
- Pick a STRATEGY article from the catalogue (62 available)
- Run the full 7-stage pipeline end-to-end
- This is the real test of the quality loop

### Step 2: Investigate Remaining Review Findings
- ATR units question (Gemini flagged atr_pips as wrong unit — verify what Rust expects)
- HSD variant wiring (Codex flagged — verify encoding path)
- StaleExit naming/logic mismatch (cosmetic but should document)

### Step 3: Monitor Live Traders
- Wait for market open, check traders are generating signals
- Use `/verify-trades` to compare backtest vs live once we have data

### Step 4: Slim CLAUDE.md to <200 lines
- Move architecture/history to docs/ and memory

## Blockers
- Gemini 3.1 Pro Preview has capacity issues (429 errors) — use auto-routing which falls back to Gemini 3 Pro
- Market closed until Monday — can't verify live traders until then

## Key Context
- Deployed strategies: EMA, MACD, Stochastic on EURUSD H1 (VPS, IC Markets demo, --testing mode)
- Skills: `/new-strategy`, `/review`, `/run-backtest`, `/deploy`, `/verify-trades`
- Codex: authenticated via ChatGPT subscription, model gpt-5.4
- Gemini: authenticated via Google Account (Pro plan), auto-routing (Gemini 3 Pro / 3.1 when available)
- Rust build: `cd rust && bash build.sh` (Windows) or `maturin develop --release`
