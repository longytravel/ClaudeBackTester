# /review — Multi-AI Code Review (Codex + Gemini + Claude)

Run a thorough code review using three AI reviewers: Codex (GPT-5.4), Gemini (3.1), and Claude.
Codex and Gemini are read-only auditors. Claude reads both reports, makes the final call, and fixes issues.

## Usage
```
/review [FILE_OR_MODULE] [FOCUS]
```

Examples:
```
/review backtester/strategies/hidden_smash_day.py
/review backtester/core/engine.py "check Rust FFI boundary safety"
/review backtester/strategies/modules.py "verify PL slot mappings match Rust constants"
/review all                          # Review ALL key modules (batch mode)
```

## Environment Setup (ALWAYS run first)
```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
```

---

## Step 1: Determine Scope

If FILE_OR_MODULE is `all`, use this batch list of critical files:

| Module | Files | Focus |
|--------|-------|-------|
| **Engine** | `backtester/core/engine.py` | Signal unpacking, Rust bridge, array dtypes |
| **Rust Bridge** | `backtester/core/rust_loop.py` | PyO3 array passing, dtype conversions |
| **Metrics** | `backtester/core/metrics.py` | Quality score formula, Sharpe annualization |
| **Encoding** | `backtester/core/encoding.py` | Param encoding/decoding, PL slot mapping |
| **Management Modules** | `backtester/strategies/modules.py` | PL slot mappings, param ranges, Rust logic match |
| **Rust Trade Full** | `rust/src/trade_full.rs` | Management module logic matches Python config |
| **Rust Trade Basic** | `rust/src/trade_basic.rs` | SL/TP hit detection, spread handling |
| **Rust Constants** | `rust/src/constants.rs` | PL_* slot numbers match Python |
| **Rust Metrics** | `rust/src/metrics.rs` | Quality score matches Python formula |
| **Base Strategy** | `backtester/strategies/base.py` | ParamSpace API, signal contract |
| **SL/TP** | `backtester/strategies/sl_tp.py` | SL/TP calculation correctness |
| **Indicators** | `backtester/strategies/indicators.py` | Numpy correctness, NaN handling |
| **Optimizer** | `backtester/optimizer/staged.py` | Stage locking, param narrowing |
| **Sampler** | `backtester/optimizer/sampler.py` | Sobol + EDA, entropy monitoring |
| **Walk-Forward** | `backtester/pipeline/walk_forward.py` | Window splits, embargo, cost settings |
| **Confidence** | `backtester/pipeline/confidence.py` | Hard gates, DSR formula, scoring |
| **Live Trader** | `backtester/live/trader.py` | Signal filtering, order placement |
| **Risk Manager** | `backtester/risk/manager.py` | Position sizing, circuit breakers |

For single file reviews, just use the specified file.

---

## Step 2: Run Codex Review (read-only)

```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
codex exec --skip-git-repo-check \
  -m gpt-5.4 \
  --config model_reasoning_effort="high" \
  --sandbox read-only \
  "You are a senior code reviewer for an automated forex trading system.

Review FILE_PATH. Focus: FOCUS_AREA

Check for:
1. CORRECTNESS: Logic bugs, off-by-one errors, wrong formulas, race conditions
2. DATA SAFETY: NaN propagation, division by zero, array bounds, dtype mismatches
3. CONSISTENCY: Does this file's logic match what other files expect? (e.g., do PL slot numbers match between Python and Rust?)
4. PERFORMANCE: Unnecessary copies, Python loops that should be numpy, memory leaks
5. SECURITY: Command injection, path traversal, unsafe deserialization
6. REALISM: For trading logic — does this match how real markets work? (spread handling, slippage, fill assumptions)

For each finding, report:
- Severity: HIGH / MEDIUM / LOW
- File:line reference
- What's wrong
- Suggested fix

End with a PASS/FAIL verdict (FAIL = any HIGH findings)." \
  2>/dev/null
```

Save output to `Research/reviews/codex_{module_name}_{date}.md`.

---

## Step 3: Run Gemini Review (read-only)

```bash
cd /c/Users/ROG/Projects/ClaudeBackTester
gemini -p "You are a senior code reviewer for an automated forex trading system built in Python + Rust (PyO3).

Review the file at FILE_PATH. Focus: FOCUS_AREA

Check for:
1. CORRECTNESS: Logic bugs, off-by-one errors, wrong formulas, race conditions
2. DATA SAFETY: NaN propagation, division by zero, array bounds, dtype mismatches (int32 vs int64 matters for PyO3)
3. CONSISTENCY: Does this file's logic match what other files expect?
4. PERFORMANCE: Unnecessary copies, Python loops that should be numpy, memory leaks
5. TRADING REALISM: Spread in price units not pips, commission handling, slippage assumptions

For each finding, report:
- Severity: HIGH / MEDIUM / LOW
- File:line reference
- What's wrong
- Suggested fix

End with a PASS/FAIL verdict (FAIL = any HIGH findings)."
```

Save output to `Research/reviews/gemini_{module_name}_{date}.md`.

---

## Step 4: Claude's Synthesis (THE DECISION)

Claude reads BOTH review reports and the source code, then produces a **final review** that:

1. **Merges findings** — deduplicate where Codex and Gemini found the same issue
2. **Resolves disagreements** — if one says HIGH and the other says LOW, Claude decides
3. **Adds own findings** — Claude may spot issues both reviewers missed
4. **Removes false positives** — if a finding is wrong (the code is actually correct), explain why
5. **Produces action items** — a prioritized list of fixes

Write the synthesis to `Research/reviews/final_{module_name}_{date}.md`:

```markdown
# Final Review: {module_name}
**Date**: {today}
**Reviewers**: Codex (GPT-5.4), Gemini (3.1), Claude (Opus 4.6)
**File**: {file_path}
**Focus**: {focus_area}

## Verdict: PASS / FAIL

## Findings Summary
| # | Severity | Finding | Codex | Gemini | Claude | Action |
|---|----------|---------|-------|--------|--------|--------|
| 1 | HIGH | Description | Found | Missed | Agree | Fix needed |
| 2 | MEDIUM | Description | Found | Found | Disagree — false positive | Skip |
| 3 | LOW | Description | Missed | Found | Agree | Optional |

## Detailed Findings
### Finding 1: {title}
**Severity**: HIGH
**Location**: `file.py:123`
**Codex said**: ...
**Gemini said**: ...
**Claude's verdict**: ...
**Fix**: ...

## Action Items (prioritized)
1. [ ] Fix HIGH finding #1 — ...
2. [ ] Fix HIGH finding #2 — ...
3. [ ] Consider MEDIUM finding #3 — ...
```

---

## Step 5: Fix Issues

If the final review has HIGH findings:
1. Fix each one in the source code
2. Run the relevant tests to verify the fix doesn't break anything
3. If the fix touches Rust code, rebuild: `.venv/Scripts/maturin.exe develop --release --manifest-path rust/Cargo.toml`
4. Re-run the smoke test if a strategy was affected: `uv run python scripts/smoke_test_strategy.py --strategy {name}`

---

## Step 6: Commit

After all fixes are applied and tests pass:
```bash
git add -A
git commit -m "Fix issues from multi-AI review of {module_name}"
```

---

## Batch Mode (`/review all`)

When reviewing all modules:
1. Run Codex and Gemini reviews **in parallel** where possible (they're independent)
2. Process one module at a time through the full pipeline (review → synthesize → fix → test)
3. Start with the highest-risk modules: engine.py, rust trade files, modules.py
4. Track progress — show the user a table of modules with PASS/FAIL status
5. Commit after each module is fixed (not all at once)

Estimated time for full batch review: ~30-45 minutes (18 modules × ~2 min each)

---

## Tips
- Codex is better at: API contract checking, pattern consistency, code style
- Gemini is better at: math/formula verification, dtype issues, performance analysis
- Claude is better at: system-level thinking, cross-file consistency, trading domain knowledge
- When they all agree on a HIGH finding — fix it immediately, it's real
- When only one reviewer flags something — Claude should investigate carefully before acting
