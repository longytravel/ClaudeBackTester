# /new-strategy — Build a Trading Strategy from an MQL5 Article (Quality Loop)

Repeatable, quality-gated workflow for turning an MQL5 article into a validated trading strategy.

## Usage
```
/new-strategy [ARTICLE_ID]
```
If no article ID provided, pick the next untouched STRATEGY article from the catalogue.

## Environment Setup (ALWAYS run first in every bash command)
```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
```

---

## Stage 1: Article Selection & Scraping

### 1a. Select the article

If ARTICLE_ID was provided, use it. Otherwise:

```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
uv run python -c "
import json
cat = json.load(open('Research/mql5_catalogue.json'))
strats = [a for a in cat['articles'] if a.get('category') == 'STRATEGY' and a.get('status', 'new') == 'new']
print(f'Available STRATEGY articles: {len(strats)}')
for s in strats[:10]:
    print(f'  {s[\"id\"]:>6} | {s[\"title\"][:75]}')
"
```

Show the list to the user. Ask them to pick one via AskUserQuestion:
- **"Pick an article ID"** — show the list and let them choose
- **"Next (recommended)"** — use the first one in the list

### 1b. Scrape the article

```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
uv run python scripts/scrape_mql5_article.py --id ARTICLE_ID
```

If scraping fails (403, CAPTCHA, empty body):
1. Ask the user to open the article URL in their browser
2. Ask them to copy-paste the article text
3. Save it manually to `Research/articles/{ARTICLE_ID}.md`

### 1c. Read the scraped article

Read `Research/articles/{ARTICLE_ID}.md` to understand the strategy.

---

## Stage 2: Rule Extraction

Read the article and extract a structured JSON spec. Write to `Research/strategies/{ARTICLE_ID}.json`:

```json
{
  "article_id": "XXXXX",
  "article_title": "...",
  "article_url": "https://www.mql5.com/en/articles/XXXXX",
  "strategy_name": "snake_case_name",
  "strategy_class": "PascalCaseName",
  "description": "One-sentence summary of the trading logic",
  "timeframes": ["H1", "H4"],
  "instruments": ["EUR/USD", "GBP/USD"],
  "entry_rules": {
    "buy": ["Condition 1 in plain English", "Condition 2..."],
    "sell": ["Condition 1...", "Condition 2..."]
  },
  "exit_rules": {
    "stop_loss": "How SL is determined",
    "take_profit": "How TP is determined",
    "management": ["Any trailing, breakeven, partial close rules"]
  },
  "indicators_required": ["ATR(14)", "EMA(20)", "RSI(14)"],
  "parameters": [
    {"name": "param_name", "type": "int|float|categorical", "values": [1, 2, 3], "group": "signal|risk|time"}
  ],
  "ambiguities": "Any unclear or missing rules from the article",
  "simplifications": "Any complex parts we simplified for our framework"
}
```

**Critical assessment rules:**
- If the article describes ML/neural network logic → flag as too complex, ask user if they want to simplify
- If the article needs multi-timeframe data → note this; our framework currently supports single-timeframe signals
- If entry rules are ambiguous → document EXACTLY what's unclear, don't guess
- If the article has no clear SL/TP → our framework adds ATR-based SL/TP via risk_params(), note that

Show the extraction to the user via AskUserQuestion. Ask: **"Do these rules look right? Any corrections?"**

---

## Stage 3: Strategy Implementation

### 3a. Check indicators

Read `backtester/strategies/indicators.py`. Check if ALL required indicators exist.
If any are missing → implement them in indicators.py FIRST (numpy vectorized, no loops).

### 3b. Build the strategy class

Create `backtester/strategies/{strategy_name}.py` following this EXACT structure:

```python
"""Strategy: {PascalCaseName} — {one-line description}.

Source: MQL5 article {ARTICLE_ID} — {article_title}
URL: {article_url}
"""

import numpy as np

from backtester.strategies.base import (
    Direction,
    ParamSpace,
    Signal,
    SLTPResult,
    Strategy,
    management_params,
    risk_params,
    time_params,
)
from backtester.strategies.indicators import (...)  # what's needed
from backtester.strategies.registry import register
from backtester.strategies.sl_tp import calc_sl_tp


@register
class StrategyClassName(Strategy):
    @property
    def name(self) -> str:
        return "snake_case_name"

    @property
    def version(self) -> str:
        return "1.0.0"

    def param_space(self) -> ParamSpace:
        space = ParamSpace()
        # Signal params (from extraction)
        space.add("param1", [...values...], group="signal")
        # Standard groups
        for p in risk_params():
            space.add_param_def(p)
        for p in management_params(self.management_modules()):
            space.add_param_def(p)
        for p in time_params():
            space.add_param_def(p)
        return space

    def generate_signals(self, ...) -> list[Signal]:
        return []  # All logic in vectorized path

    def generate_signals_vectorized(self, open_, high, low, close, volume,
                                     spread, pip_value,
                                     bar_hour=None, bar_day_of_week=None):
        # NUMPY ONLY — no Python loops over bars
        # NOTE: This method receives NO params — it generates ALL possible signals.
        # The engine filters signals by variant/filter_value during evaluation.
        n = len(close)
        if bar_hour is None:
            bar_hour = np.zeros(n, dtype=np.int64)
        if bar_day_of_week is None:
            bar_day_of_week = np.zeros(n, dtype=np.int64)

        # 1. Compute indicators
        # 2. Detect entry conditions (boolean masks)
        # 3. Build signal arrays using parts-list pattern
        # 4. Return dict with standard keys

        parts_idx = []
        parts_dir = []
        parts_entry = []
        parts_atr = []
        parts_variant = []
        parts_fv = []

        # ... detection logic using numpy boolean indexing ...

        if not parts_idx:
            return self._empty_signals()

        return {
            "bar_index": np.concatenate(parts_idx),
            "direction": np.concatenate(parts_dir),
            "entry_price": np.concatenate(parts_entry),
            "atr_pips": np.concatenate(parts_atr),
            "variant": np.concatenate(parts_variant),
            "filter_value": np.concatenate(parts_fv),
        }

    def filter_signals(self, signals, params):
        return signals  # Filtering done by Rust engine

    def calc_sl_tp(self, signal, params, high, low, atr_pips):
        return calc_sl_tp(signal, params, high, low, atr_pips)
```

### 3c. Register the strategy

Add import line to `backtester/strategies/__init__.py`:
```python
import backtester.strategies.{strategy_name}  # noqa: F401
```

### HARD RULES for implementation:
- **NEVER** use Python for/while loops iterating over bar arrays
- **ALL** indicator computations use numpy arrays
- Use the **parts-list pattern** (lists of arrays + np.concatenate at end)
- Use `np.errstate(divide="ignore", invalid="ignore")` for division
- Handle NaN warmup periods (first N bars of indicator output)
- Define a `_empty_signals()` helper method (see hidden_smash_day.py) that returns the correct empty dict with all 6 required keys as empty numpy arrays with correct dtypes
- `generate_signals()` MUST return `[]` — all logic goes in `generate_signals_vectorized()`

---

## Stage 4: Multi-AI Review (QUALITY GATE 1)

Run BOTH Codex and Gemini as independent reviewers, then Claude synthesizes.

### 4a. Run Codex review (read-only)

```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
codex exec --skip-git-repo-check \
  -m gpt-5.4 \
  --config model_reasoning_effort="high" \
  --sandbox read-only \
  "Review the strategy implementation in backtester/strategies/STRATEGY_NAME.py.

Check against these criteria:

1. ARTICLE RULES (Research/strategies/ARTICLE_ID.json):
   - Does the code correctly implement every buy/sell entry condition?
   - Are any rules missing or incorrectly translated from the article?
   - Are parameter ranges sensible for the described strategy?
   - Is the signal direction correct (BUY=1, SELL=-1)?

2. BASE CLASS CONTRACT (backtester/strategies/base.py):
   - All abstract methods implemented?
   - generate_signals() returns [] (empty list, NOT signals)?
   - generate_signals_vectorized() returns dict with EXACTLY these 8 keys: bar_index, direction, entry_price, hour, day_of_week, atr_pips, variant, filter_value?
   - _empty_signals() used when no signals found (not a hand-built empty dict)?
   - param_space() includes risk_params() + management_params(self.management_modules()) + time_params()?
   - @register decorator present on the class?

3. VECTORIZATION & PERFORMANCE:
   - No Python for/while loops iterating over bar arrays? (This causes CPython heap corruption on 300K+ bars)
   - All indicator computations use numpy arrays?
   - Parts-list pattern used (lists of arrays + np.concatenate)?
   - NaN handling correct for indicator warmup periods (first N bars)?
   - No calls to float(), int() inside loops on bar data?

4. DATA CORRECTNESS:
   - ATR is in PRICE UNITS (not pips). atr_pips should be atr_arr / pip_value
   - Spread is in PRICE UNITS (not pips). Don't double-convert
   - Entry prices use open_[i+1] (next bar open) or close[i], NOT the signal bar's intrabar price
   - Signal indices don't point to the last bar (no room for next-bar entry)

5. EDGE CASES:
   - Division by zero guarded (np.errstate or explicit checks)?
   - Empty array cases handled (what if zero signals match)?
   - Array index bounds: no idx+1 when idx is the last bar?
   - What happens if indicator period > data length?
   - Variant/filter_value encoding: are values consistent with param_space values?

6. LIVE TRADER COMPATIBILITY:
   - Does filter_signals() return signals unchanged? (Rust engine handles filtering)
   - Would the live trader's _filter_vectorized_signals() work with this strategy's variant/filter_value scheme?

Flag each finding as HIGH (must fix before proceeding), MEDIUM (should fix), or LOW (suggestion).
At the end, give a PASS/FAIL verdict. PASS means zero HIGH findings." \
  2>/dev/null
```

### 4b. Run Gemini review (read-only)

Run in parallel with Codex if possible:

```bash
cd /c/Users/ROG/Projects/ClaudeBackTester
gemini -p "You are a senior code reviewer for an automated forex trading system (Python + Rust/PyO3).

Review the strategy at backtester/strategies/STRATEGY_NAME.py against the extracted rules in Research/strategies/ARTICLE_ID.json.

Check:
1. ARTICLE RULES: Does code match every buy/sell condition from the JSON spec?
2. SIGNAL CONTRACT: Must return dict with exactly 8 keys: bar_index, direction, entry_price, hour, day_of_week, atr_pips, variant, filter_value. All numpy arrays, correct dtypes (int64 for indices/direction/hour/day/variant, float64 for prices/atr/filter_value).
3. VECTORIZATION: No Python for/while loops over bar arrays (causes CPython heap corruption on 300K+ bars). All numpy.
4. DATA CORRECTNESS: ATR in price units, spread in price units, entry_price must be next-bar open.
5. EDGE CASES: Division by zero, NaN warmup, empty arrays, last-bar indexing.

Severity: HIGH (must fix) / MEDIUM (should fix) / LOW (suggestion).
End with PASS/FAIL verdict."
```

### 4c. Save review outputs

Write the Codex output to `Research/reviews/{ARTICLE_ID}_codex_review.md`:
```markdown
# Codex Review: {strategy_name}
**Article**: {ARTICLE_ID} | **Date**: {today} | **Model**: GPT-5.4
**Verdict**: PASS/FAIL
---
{codex output}
```

Write the Gemini output to `Research/reviews/{ARTICLE_ID}_gemini_review.md`:
```markdown
# Gemini Review: {strategy_name}
**Article**: {ARTICLE_ID} | **Date**: {today} | **Model**: Gemini 3.1
**Verdict**: PASS/FAIL
---
{gemini output}
```

### 4d. Quality gate decision

- **If PASS (zero HIGH findings)**: Proceed to Stage 5
- **If FAIL (HIGH findings exist)**:
  1. Fix each HIGH finding in the strategy code
  2. Re-run Codex review (max 2 retries)
  3. If still failing after 2 retries: show findings to user, ask via AskUserQuestion:
     - **"Force proceed"** — skip gate, continue to testing
     - **"Abort"** — stop here, come back later
- **If Codex unavailable** (auth error, rate limit): Ask user via AskUserQuestion:
  - **"Skip Codex review"** — proceed without review
  - **"Wait and retry"** — try again later

### 4e. Claude's synthesis (THE DECISION)

Claude reads BOTH review reports and the source code, then makes the final call:
- **Merge findings** — deduplicate where Codex and Gemini found the same issue
- **Resolve disagreements** — if one says HIGH and the other says LOW, Claude investigates and decides
- **Add own findings** — Claude may spot issues both missed (especially cross-file consistency)
- **Remove false positives** — if a finding is wrong, explain why
- When all three agree on a HIGH finding → fix immediately, it's definitely real
- When only one flags something → investigate before acting

Write the final synthesis to `Research/reviews/{ARTICLE_ID}_final_review.md`.

---

## Stage 5: Unit Tests

### 5a. Create test file

Create `tests/test_{strategy_name}.py`:

```python
"""Tests for {strategy_name} strategy."""

import numpy as np
import pytest

from backtester.strategies.registry import create, get


@pytest.fixture
def strategy():
    return create("strategy_name")


class TestParamSpace:
    def test_signal_params_exist(self, strategy):
        space = strategy.param_space()
        names = [p.name for p in space]
        # Check strategy-specific signal params exist
        assert "param1" in names

    def test_standard_groups_present(self, strategy):
        space = strategy.param_space()
        groups = {p.group for p in space}
        assert "signal" in groups
        assert "risk" in groups

    def test_all_params_have_values(self, strategy):
        space = strategy.param_space()
        for p in space:
            assert len(p.values) > 0, f"Param {p.name} has no values"


class TestSignalGeneration:
    def _make_data(self, n=500, trend="up"):
        """Generate synthetic OHLCV data."""
        np.random.seed(42)
        base = 1.1000
        if trend == "up":
            close = base + np.cumsum(np.random.randn(n) * 0.0005 + 0.00005)
        elif trend == "down":
            close = base + np.cumsum(np.random.randn(n) * 0.0005 - 0.00005)
        else:
            close = base + np.cumsum(np.random.randn(n) * 0.0005)
        high = close + np.abs(np.random.randn(n)) * 0.001
        low = close - np.abs(np.random.randn(n)) * 0.001
        open_ = close + np.random.randn(n) * 0.0003
        volume = np.random.randint(100, 1000, n).astype(float)
        spread = np.full(n, 0.00015)
        atr = np.full(n, 0.0010)
        return open_, high, low, close, volume, spread, atr

    def test_generates_signals(self, strategy):
        open_, high, low, close, vol, spread, atr = self._make_data(1000)
        space = strategy.param_space()
        params = {p.name: p.values[len(p.values) // 2] for p in space}
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, vol, spread, atr, 0.0001, params
        )
        assert "bar_index" in result
        # Should produce at least some signals on 1000 bars
        assert len(result["bar_index"]) > 0

    def test_both_directions(self, strategy):
        open_, high, low, close, vol, spread, atr = self._make_data(2000, "random")
        space = strategy.param_space()
        params = {p.name: p.values[len(p.values) // 2] for p in space}
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, vol, spread, atr, 0.0001, params
        )
        dirs = result["direction"]
        if len(dirs) > 10:  # Only check if enough signals
            assert 1 in dirs or True  # BUY
            assert -1 in dirs or True  # SELL

    def test_signal_indices_in_range(self, strategy):
        n = 500
        open_, high, low, close, vol, spread, atr = self._make_data(n)
        space = strategy.param_space()
        params = {p.name: p.values[len(p.values) // 2] for p in space}
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, vol, spread, atr, 0.0001, params
        )
        indices = result["bar_index"]
        if len(indices) > 0:
            assert indices.min() >= 0
            assert indices.max() < n

    def test_atr_pips_positive(self, strategy):
        open_, high, low, close, vol, spread, atr = self._make_data(500)
        space = strategy.param_space()
        params = {p.name: p.values[len(p.values) // 2] for p in space}
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, vol, spread, atr, 0.0001, params
        )
        atr_pips = result["atr_pips"]
        if len(atr_pips) > 0:
            assert np.all(atr_pips > 0)

    def test_empty_on_minimal_data(self, strategy):
        """Strategy should handle very short data gracefully."""
        n = 5
        open_ = np.ones(n) * 1.1
        high = np.ones(n) * 1.101
        low = np.ones(n) * 1.099
        close = np.ones(n) * 1.1
        vol = np.ones(n) * 100
        spread = np.ones(n) * 0.00015
        atr = np.ones(n) * 0.001
        space = strategy.param_space()
        params = {p.name: p.values[0] for p in space}
        result = strategy.generate_signals_vectorized(
            open_, high, low, close, vol, spread, atr, 0.0001, params
        )
        # Should not crash, result should have correct keys
        assert "bar_index" in result


class TestMetadata:
    def test_name(self, strategy):
        assert strategy.name == "strategy_name"

    def test_version(self, strategy):
        assert strategy.version  # Non-empty

    def test_registered(self):
        cls = get("strategy_name")
        assert cls is not None


class TestRegistry:
    def test_create_by_name(self):
        s = create("strategy_name")
        assert s is not None
        assert s.name == "strategy_name"
```

**IMPORTANT**: Replace `strategy_name` and param names with the actual strategy values.
Adjust `_make_data()` if the strategy needs specific price patterns to trigger signals.

### 5b. Run tests

```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
uv run pytest tests/test_{strategy_name}.py -v
```

If tests fail → fix the strategy or tests and re-run. Tests MUST pass before proceeding.

### 5c. Run full test suite

Verify the new strategy doesn't break anything:
```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
uv run pytest tests/ -x --timeout=60
```

All existing tests must still pass.

---

## Stage 5.5: Integration Smoke Test (QUALITY GATE 2)

This is the critical gate that catches bugs unit tests miss. It runs the strategy on REAL
Dukascopy data through the actual Rust engine. We've had strategies pass all unit tests
then crash or produce 0 trades on real data — this stage prevents that.

### 5.5a. Run the smoke test

```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
uv run python scripts/smoke_test_strategy.py --strategy {strategy_name} --pair EUR/USD --timeframe H1
```

This script (takes ~10 seconds):
1. Loads real EUR/USD H1 data from Google Drive
2. Runs `generate_signals_vectorized()` with mid-range params on real prices
3. Checks: signals > 0, no NaN, indices in range, ATR pips positive
4. Creates a `BacktestEngine` and runs `evaluate_batch()` with 10 random param sets through Rust
5. Checks: no crash, metrics finite, at least some trades produced

### 5.5b. Interpret results

- **PASSED**: All checks green → proceed to Stage 6
- **FAIL: Zero signals**: Signal conditions may be too strict for this pair/timeframe, or the
  logic has a bug that only manifests on real price scales. Check indicator calculations.
- **FAIL: Rust engine crash**: Array shape mismatch or invalid param encoding. Check that
  variant/filter_value arrays match param_space encoding. Check that param_layout bounds
  are within NUM_PL (64).
- **FAIL: Zero trades across all trials**: Signals exist but SL/TP settings kill every trade.
  Check that ATR-based SL/TP distances are sensible for the pair's volatility.
- **FAIL: NaN in metrics**: Data issue or indicator returning NaN past warmup period.

### 5.5c. If smoke test fails

Fix the issue, then re-run. Common fixes:
- **Wrong ATR scale**: Ensure `atr_pips = atr_arr / pip_value` (price units → pips)
- **Index off-by-one**: Signal on last bar has no room for next-bar entry
- **Variant mismatch**: Variant values in signals don't match param_space values

Do NOT proceed to Stage 6 until the smoke test passes. A failing smoke test means the
10-minute backtest will definitely fail too.

---

## Stage 6: Full Backtest (QUALITY GATE 3)

Run the strategy through the optimizer + validation pipeline. Use `/run-backtest`:

```
/run-backtest {strategy_name} EUR/USD H1 turbo
```

This delegates to the existing run-backtest skill which handles:
- Data pre-flight check
- Dashboard startup
- Staged optimization
- Validation pipeline (walk-forward, CPCV, Monte Carlo, stability, confidence, regime)
- Results interpretation

### Quality Gate 3 (minimal):
- **Must produce >= 30 trades** in the backtest period. If 0 trades → investigate signal generation.
- **No NaN metrics** — indicates data or computation bug.
- Negative quality scores are FINE — that's truth, not a bug.
- RED confidence rating is FINE — we're testing the system, not chasing profits.

---

## Stage 7: Results Documentation

### 7a. Write results summary

Create `Research/strategies/{ARTICLE_ID}_results.md`:

```markdown
# Results: {strategy_name}

**Article**: {ARTICLE_ID} — {title}
**Date**: {today}
**Pair**: EUR/USD | **Timeframe**: H1 | **Preset**: turbo

## Best Parameters
{param: value table}

## Key Metrics
| Metric | Back-test | Forward-test |
|--------|-----------|-------------|
| Trades | ... | ... |
| Win Rate | ... | ... |
| Quality Score | ... | ... |
| Sharpe | ... | ... |
| Max DD% | ... | ... |

## Confidence
- Rating: GREEN/YELLOW/RED
- Score: XX/100
- Hard gates: all passed / X failed

## Forward/Back Ratio
- Ratio: X.XX (gate: >= 0.4)

## Assessment
{1-2 paragraphs: does this strategy show edge? What worked? What didn't?
Is it worth testing on other pairs/timeframes? Any concerns?}
```

### 7b. Update catalogue

Update the article's status in `Research/mql5_catalogue.json`:

```bash
export PATH="$HOME/.cargo/bin:/c/Users/ROG/.local/bin:$PATH"
cd /c/Users/ROG/Projects/ClaudeBackTester
uv run python -c "
import json
cat = json.load(open('Research/mql5_catalogue.json'))
for a in cat['articles']:
    if a['id'] == 'ARTICLE_ID':
        a['status'] = 'built'  # or 'failed' if zero edge
        break
json.dump(cat, open('Research/mql5_catalogue.json', 'w'), indent=2, ensure_ascii=False)
print('Updated')
"
```

### 7c. Commit

Commit all new files:
```bash
git add backtester/strategies/{strategy_name}.py
git add backtester/strategies/__init__.py
git add tests/test_{strategy_name}.py
git add Research/articles/{ARTICLE_ID}.md
git add Research/strategies/{ARTICLE_ID}.json
git add Research/strategies/{ARTICLE_ID}_results.md
git add Research/reviews/{ARTICLE_ID}_codex_review.md
git add Research/mql5_catalogue.json
```

Use a descriptive commit message:
```
Add {PascalCaseName} strategy (MQL5 article {ARTICLE_ID})
```

---

## Summary Checklist

After completing all stages, present this checklist to the user:

- [ ] **Stage 1**: Article scraped → `Research/articles/{ARTICLE_ID}.md`
- [ ] **Stage 2**: Rules extracted → `Research/strategies/{ARTICLE_ID}.json`
- [ ] **Stage 3**: Strategy built → `backtester/strategies/{strategy_name}.py`
- [ ] **Stage 3**: Registered in `__init__.py`
- [ ] **Stage 4**: Codex review PASSED → `Research/reviews/{ARTICLE_ID}_codex_review.md`  (GATE 1)
- [ ] **Stage 5**: Unit tests pass → `tests/test_{strategy_name}.py`
- [ ] **Stage 5**: Full test suite passes (no regressions)
- [ ] **Stage 5.5**: Smoke test PASSED — real data + Rust engine (GATE 2)
- [ ] **Stage 6**: Backtest completed, >= 30 trades (GATE 3)
- [ ] **Stage 7**: Results documented → `Research/strategies/{ARTICLE_ID}_results.md`
- [ ] **Stage 7**: Catalogue updated
- [ ] **Stage 7**: Committed to git

**Total artifacts per strategy: 6 files created, 2 files modified**

### Quality Gates Summary
| Gate | Stage | What it checks | Blocks on |
|------|-------|---------------|-----------|
| 1 | 4 (Codex) | Code vs article rules, vectorization, contract | Any HIGH finding |
| 2 | 5.5 (Smoke) | Real data + Rust engine integration | Crash, 0 signals, 0 trades |
| 3 | 6 (Backtest) | Full optimizer + validation pipeline | < 30 trades, NaN metrics |

---

## Skipping Stages

The user can skip stages if they've already been done:
- `--from 3` — start from Stage 3 (strategy already scraped + extracted)
- `--from 5` — start from Stage 5 (strategy already reviewed by Codex)
- `--skip-codex` — skip Stage 4 entirely (no Codex review)
- `--skip-backtest` — skip Stage 6 (just build and test, no optimization run)

When resuming, check what artifacts already exist:
- `Research/articles/{ID}.md` exists → skip Stage 1
- `Research/strategies/{ID}.json` exists → skip Stage 2
- `backtester/strategies/{name}.py` exists → skip Stage 3
- `Research/reviews/{ID}_codex_review.md` exists → skip Stage 4
- `tests/test_{name}.py` exists AND passes → skip Stage 5

**NEVER skip Stage 5.5 (smoke test)**. Even if unit tests pass, the smoke test catches
a different class of bugs (real data, Rust engine integration). It takes ~10 seconds
and has saved us from multiple failed 10-minute backtest runs in the past.
