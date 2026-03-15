# Current Task

## Last Completed (Mar 15 2026)

### Modular Staged Optimizer (Phases 1-4)
- Expanded Rust NUM_PL from 27 → 64 (10 signal slots PL_SIGNAL_P0-P9 + 27 reserved)
- Created ManagementModule system: 5 built-in modules in `strategies/modules.py`
- Sub-grouped optimization stages (exit_trailing, exit_protection, exit_time)
- Generic signal filter loop in Rust + live trader wiring
- 520 tests pass, zero behavior change

### Strategy Research Pipeline
- Crawler: 2,731 MQL5 articles in `Research/mql5_catalogue.json`
- Haiku triage: 200 classified (62 STRATEGY, 11 MODULE, 24 INDICATOR, 31 SYSTEM, 72 SKIP)
- First strategy built: Hidden Smash Day (article 21391, Larry Williams)

### Tooling Installed
- claude-mem (auto session memory), claude-mermaid MCP (diagrams)
- docs/architecture.md (system overview), docs/tooling-setup.md

## Next Steps — Quality Pipeline First

### Step 1: Codex Review of Hidden Smash Day
- Run `/codex` to review `backtester/strategies/hidden_smash_day.py`
- Codex checks: does signal logic match article rules? Vectorization correct? Edge cases?
- Review against the article source (MQL5 article 21391)
- Fix any issues found

### Step 2: Codex Review of Management Modules
- Run `/codex` to review `backtester/strategies/modules.py` + `rust/src/trade_full.rs`
- These modules affect EVERY strategy — must be correct
- Check: param ranges sensible? PL slot mappings correct? Rust logic matches Python config?

### Step 3: Define the Strategy Creation Flow
- Document the exact end-to-end process for building a new strategy
- This becomes a repeatable workflow / skill
- Flow: article → scrape → extract rules → build strategy → Codex review → fix → write tests → run backtest → analyse results

### Step 4: Run Hidden Smash Day Backtest
- Only AFTER Codex review passes
- `uv run python scripts/full_run.py --strategy hidden_smash_day --pair EURUSD --timeframe H1`

### Step 5: Slim CLAUDE.md to <200 lines
- Move architecture/history to docs/ and memory (partially done)

## Strategy Creation Flow (DRAFT — to be finalized in Step 3)

```
1. PICK article from Research/mql5_catalogue.json (category=STRATEGY)
2. SCRAPE full article → Research/articles/{id}.md
3. EXTRACT trading rules (entry, exit, filters, params, indicators needed)
4. BUILD Python strategy class following base.py pattern
5. CODEX REVIEW — /codex reviews strategy code against article rules
6. FIX issues from review
7. WRITE unit tests for signal generation
8. RUN backtest through optimizer + pipeline
9. ANALYSE results — does it pass? What worked/didn't?
10. ITERATE or NEXT article
```

## Blockers
- None

## Key Context
- ANTHROPIC_API_KEY in `.env` (for Haiku triage)
- Rust build: `cd rust && VIRTUAL_ENV=../.venv PATH="../.venv/Scripts:$PATH" ../.venv/Scripts/python.exe -m maturin develop --release`
- Management params use sub-groups (exit_trailing, exit_protection, exit_time) not "management"
- Codex setup guide: `docs/codex-setup-guide.md`
