# Current Task

## Last Completed (Mar 15 2026)

### Quality Loop Skill — `/new-strategy` (JUST BUILT)
- Created `.claude/commands/new-strategy.md` — 7-stage repeatable workflow
- Created `scripts/scrape_mql5_article.py` — MQL5 article scraper
- Created `Research/articles/`, `Research/strategies/`, `Research/reviews/` directories
- Flow: select article → scrape → extract rules → build strategy → Codex review → unit tests → backtest → document results
- Two quality gates: Codex review (zero HIGH findings) + backtest (>= 30 trades)
- Smart resume: skips stages whose artifacts already exist

### Previous Work
- Modular Staged Optimizer: NUM_PL 27→64, ManagementModule system, sub-grouped stages
- Strategy Research Pipeline: 2,731 articles, 200 triaged (62 STRATEGY)
- First strategy: Hidden Smash Day (article 21391, Larry Williams)
- Tooling: claude-mem, claude-mermaid, Codex CLI

## Next Steps

### Step 1: Test the Quality Loop on Hidden Smash Day
- Run `/new-strategy 21391` — but HSD already has strategy code, so it should skip to Stage 4 (Codex review)
- Verify: Does Codex find real issues? Does the fix cycle work?
- Verify: Do unit tests pass? Does the backtest run?

### Step 2: Run Quality Loop on a NEW Article
- Pick a fresh STRATEGY article from the catalogue
- Run `/new-strategy` end-to-end — full 7-stage pipeline
- This is the real test of repeatability

### Step 3: Codex Review of Management Modules
- Run `/codex` to review `backtester/strategies/modules.py` + `rust/src/trade_full.rs`
- These modules affect EVERY strategy — must be correct

### Step 4: Slim CLAUDE.md to <200 lines
- Move architecture/history to docs/ and memory (partially done)

## Blockers
- None

## Key Context
- ANTHROPIC_API_KEY in `.env` (for Haiku triage)
- Codex CLI authenticated via ChatGPT subscription
- Rust build: `cd rust && VIRTUAL_ENV=../.venv PATH="../.venv/Scripts:$PATH" ../.venv/Scripts/python.exe -m maturin develop --release`
- Management params use sub-groups (exit_trailing, exit_protection, exit_time) not "management"
- Strategy quality loop skill: `/new-strategy [ARTICLE_ID]`
