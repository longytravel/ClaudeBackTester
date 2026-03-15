# Current Task

## Last Completed (Mar 15 2026)

### Modular Staged Optimizer (Phases 1-4)
- Expanded Rust NUM_PL from 27 → 64 (10 signal slots PL_SIGNAL_P0-P9 + 27 reserved)
- Created ManagementModule system: 5 built-in modules in `strategies/modules.py`
- Each module has its own optimization sub-group (exit_trailing, exit_protection, exit_time)
- Added generic signal filter loop in Rust for expanded signal params
- Updated live trader with `signal_pl_mapping()` support
- 520 tests pass, zero behavior change for existing strategies

### Strategy Research Pipeline
- Built crawler: `scripts/crawl_mql5_articles.py` — 2,731 articles catalogued
- Built Haiku triage: `scripts/triage_articles.py` — 200 articles classified
- Results: 62 STRATEGY, 11 MODULE, 24 INDICATOR, 31 SYSTEM, 72 SKIP

### First Article Strategy
- Built Hidden Smash Day strategy from MQL5 article 21391 (Larry Williams)
- 36 signal variants, fully vectorized, uses modular management stages
- NOT YET TESTED through optimizer/pipeline

### Tooling
- Installed claude-mem plugin (automatic session memory)
- Installed claude-mermaid MCP (architecture diagrams)
- Created docs/architecture.md with full system overview
- Created docs/tooling-setup.md with plugin install instructions

## Next Steps

1. **Run Hidden Smash Day through optimizer + pipeline** on real EUR/USD H1 data
   - Command: `/run-backtest` or `uv run python scripts/full_run.py --strategy hidden_smash_day --pair EURUSD --timeframe H1`
   - This is the first test of a research-pipeline strategy

2. **Codex review** the Hidden Smash Day strategy code
   - `/codex` to review `backtester/strategies/hidden_smash_day.py`
   - Check signal logic matches article rules

3. **Slim CLAUDE.md to <200 lines**
   - Architecture moved to docs/architecture.md
   - History/learnings in .claude/memory/
   - Keep only rules, conventions, gotchas

4. **Pick next strategy** from the 62 classified
   - Good candidates: Adaptive Channel Breakout (21443), Liquidity + Trend Filter (21133), NRTR (21096)

5. **Classify remaining 2,531 articles** (optional, ~$3 cost)

## Blockers
- None

## Key Context
- ANTHROPIC_API_KEY is in `.env` (for Haiku triage script)
- Rust build must target MAIN venv, not rust/.venv:
  `cd rust && VIRTUAL_ENV=../.venv PATH="../.venv/Scripts:$PATH" ../.venv/Scripts/python.exe -m maturin develop --release`
- Management params now use sub-groups (exit_trailing, exit_protection, exit_time) not "management"
- New strategies can use `signal_pl_mapping()` + `sig_filter_N` arrays for expanded signal params
