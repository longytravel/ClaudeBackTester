# Current Task

## Last Completed (Mar 16 2026)

### Exploitation Sampler Comparison Build
Building 4 configurable exploitation methods for head-to-head comparison.

**Background — CMA-ES bugs discovered:**
1. `update()` was using key-matching (rounded x_tell) instead of exact row-index pairing — CMAwM's internal discretization != simple rounding, so tell() received garbage fitness for many entries
2. Dropped generations when prefilter removed all rows (pending_tell cleared without tell())
3. Neighborhood bounds mismatch between sample() and update()

**Result**: CMA-ES was NEVER learning properly. With pop_size=2048, it was effectively expensive random search (49 gens, broken feedback). All 3 bugs now fixed (Codex verified).

**Testing showed:**
- Original (broken tell): YELLOW 54.1, 2 survivors, 328 evals/sec
- pop_size=50 (overfitting): RED 33.4, 0 survivors, 328 evals/sec
- Fixed tell + pop=2048: RED 44.8, 0 survivors, 456 evals/sec (+39% speed)

**Decision**: Build all exploitation methods, test head-to-head to find best approach.

### 4 Exploitation Methods (all built, awaiting review)
1. **Sobol-only** — `--exploiter sobol` — pure quasi-random baseline
2. **EDA** — `--exploiter eda` — natively discrete, built-in regularization
3. **CMA-ES** — `--exploiter cmaes` — fixed tell(), current default
4. **GA** — `--exploiter ga` — NEW: tournament selection, uniform crossover, per-param mutation

**Files changed:**
- `backtester/optimizer/sampler.py` — new GASampler class + CMA-ES tell() fix
- `backtester/optimizer/staged.py` — GA/Sobol branches in _create_exploiter() + empty batch fix
- `backtester/optimizer/config.py` — GA config fields (pop=200, mutation=0.08)
- `scripts/full_run.py` — `--exploiter` CLI flag
- `tests/test_optimizer.py` — 10 new GASampler tests
- `.claude/commands/run-backtest.md` — updated for new system
- `.claude/skills/gemini/SKILL.md` — new Gemini 3.1 CLI skill
- 579 tests pass, 0 regressions

### Previous Work
- CMA-ES Optimizer Upgrade + Stage Merging (Mar 15)
- Pipeline Candidate Selection Redesign (DSR prefilter + dedup)
- Modular Staged Optimizer: NUM_PL 27→64, ManagementModule system
- Live trading engine deployed on IC Markets demo

## Next Steps

### Step 1: Codex + Gemini Review (IN PROGRESS)
- Both reviewing GASampler implementation for bugs
- Fix any issues found

### Step 2: E2E Comparison Test
- Run hidden_smash_day EUR/USD H1 turbo × 4 methods:
  ```
  --exploiter sobol   (baseline)
  --exploiter eda     (discrete native)
  --exploiter cmaes   (fixed, current default)
  --exploiter ga      (new)
  ```
- Compare: IS quality, OOS quality, evals/sec, survivors, composite score

### Step 3: Set Default Exploiter
- Based on comparison results, set the best method as default
- Update presets in config.py

### Step 4: Dashboard CMA-ES Metrics
- SamplerInfo shows wrong metrics for CMA-ES/GA
- Add generation count, sigma tracking, IPOP restart markers
- Update funnel for new candidate selection flow

### Step 5: Report Serialization Bug
- report.json shows 0 metrics for all candidates (console output correct)
- Need to investigate and fix

## Blockers
- None

## Key Context
- 4 exploitation methods now available via `--exploiter` CLI flag
- CMA-ES tell() bugs fixed (row-index pairing, empty batch handling)
- GA: tournament(3) + uniform crossover + per-param mutation(0.08) + elitism(20%)
- Default stages: signal → time → core_trade_profile → exit_protection → exit_time → refinement
- Cyclic passes: turbo=0, standard=1, deep=2
- Research: Gemini recommends GA or EDA over CMA-ES for discrete financial params
