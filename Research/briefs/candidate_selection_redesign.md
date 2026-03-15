# Research Brief: Candidate Selection Pipeline Redesign

## Context

We have an automated forex backtesting system that discovers trading strategies via parameter optimization, then validates them through a multi-stage pipeline (walk-forward analysis, Monte Carlo simulation, stability testing, confidence scoring). The system is built in Python + Rust (PyO3/Rayon) and runs on a 24-core i9 with 64GB RAM.

### Current Architecture

The optimization flow is:

1. **Staged Optimization** — Sobol exploration + EDA exploitation across parameter stages (signal → time → risk → management → refinement). Evaluates 200K+ parameter combinations on the backtest period (80% of data) using a Rust engine at 4K-10K evals/sec.

2. **Candidate Selection** (THE PROBLEM) — From the refinement stage, all parameter sets with quality > 0 are "passing trials" (typically 30K-60K). A MAP-Elites diversity archive selects "top N diverse" candidates. N is configured as 25 but the archive's grid is only 3×4 = 12 cells max, and in practice produces 4-12 candidates.

3. **Forward Testing** — Selected candidates are evaluated on held-out forward data (20% of data). A forward/back ratio gate (≥0.4) filters for generalization.

4. **Validation Pipeline** — Surviving candidates go through: walk-forward analysis (rolling OOS windows), CPCV (combinatorial purged cross-validation), Monte Carlo (bootstrap, permutation, trade-skip, execution stress), stability analysis (parameter perturbation), and confidence scoring (composite 0-100 with hard gates).

### The Problem We Observed

Running Hidden Smash Day strategy on EUR/USD H1:

- **58,494 parameter sets** passed the refinement quality filter
- The diversity archive selected only **4 candidates** (from a requested 25)
- All 4 failed the forward/back ratio gate (0.0 ratio, needed ≥0.4)
- Walk-forward was **skipped entirely** because all candidates were pre-eliminated
- Pipeline produced zero survivors

The core issue: we're reducing 58,494 candidates to 4 using a coarse metric-space grid BEFORE the cheap forward test. We may be discarding the parameter sets that actually generalize.

### How the MAP-Elites Archive Works (Current)

The diversity archive (`backtester/optimizer/archive.py`) creates a 2D grid:
- **Axis 1**: Trade frequency — 3 buckets based on 33rd/67th percentile of trade counts
- **Axis 2**: Quality tier — 4 buckets based on 25th/50th/75th percentile of quality scores
- Each cell holds exactly **one entry** (the best by quality score)
- Maximum possible candidates: 3 × 4 = 12
- Requested 25, got 4 (most candidates cluster in the same cells)

The archive diversifies on **metrics** (outputs) not **parameters** (inputs). Two genuinely different strategies with similar trade counts collapse to one cell. Two identical strategies with slightly different quality scores go to different cells.

### Relevant System Capabilities

- **Rust engine throughput**: 4K-10K evals/sec in FULL execution mode (with M1 sub-bar data)
- **Forward-testing cost**: One `evaluate_batch()` call. 58K candidates ≈ 6-15 seconds.
- **Pipeline validation cost per candidate**: ~5-10 seconds (walk-forward + Monte Carlo + stability + confidence)
- **Pipeline validation for 20 candidates**: ~2-3 minutes total
- **DSR (Deflated Sharpe Ratio)**: Already built in. Adjusts significance threshold based on number of trials tested. Handles multiple-testing correction.

### What We're Proposing

Replace the diversity archive candidate selection with:

1. **Forward-test ALL passing candidates** (~58K) in one batch call (~15 seconds)
2. **Apply forward/back ratio gate** (≥0.4) to all of them
3. **Rank survivors** by combined_rank (existing metric that weights forward performance 1.5×)
4. **Take top N** (configurable, default 20) for pipeline validation
5. **Optional parameter deduplication** — if many survivors have near-identical parameter vectors (producing identical trades), collapse to the one with best forward performance

The MAP-Elites archive would remain for optimizer exploration (maintaining search diversity during Sobol/EDA stages) but would no longer be the candidate selection mechanism for the pipeline.

## Research Questions

We need a researcher to validate or challenge this proposal. Specific questions:

### 1. Is forward-testing all 58K candidates statistically sound?

- Does testing 58K candidates on the forward period inflate false discovery rate beyond what DSR corrects for?
- DSR is computed per-candidate using `total_trials` (the optimization trial count), not the number of forward-tested candidates. Is this correct, or should DSR account for the number of candidates tested forward?
- Are there better multiple-testing corrections for this specific setup (large candidate pool, single forward period)?
- Reference: López de Prado, "Advances in Financial Machine Learning" (2018), Chapter 6 on false discovery rate in backtesting.

### 2. What do professional quantitative systems do at this stage?

- How do firms like Two Sigma, DE Shaw, or AQR handle the transition from optimization results to out-of-sample validation?
- Is there an industry-standard approach for candidate selection post-optimization?
- What does the academic literature recommend for selecting which parameter sets to validate further?
- Key concern: are we reinventing a solved problem?

### 3. Should diversity selection happen on parameters or metrics?

- Our current approach diversifies on metrics (trade count, quality). The proposal switches to pure ranking.
- Is there value in ensuring parameter-space diversity in the validated set? (i.e., testing genuinely different strategies, not just the top 20 by rank which might all be minor variations)
- If parameter diversity matters, what's the right distance metric for parameter vectors that mix continuous (ATR multiplier), categorical (SL mode), and boolean (breakeven enabled) dimensions?

### 4. Is the forward/back ratio gate the right primary filter?

- We use `forward_quality / back_quality ≥ 0.4` as the main generalization gate
- Is 0.4 the right threshold? Too strict (misses slow-decaying strategies)? Too loose?
- Should we use a different metric for generalization detection? (e.g., forward Sharpe > 0, or forward Sharpe > X% of back Sharpe, or statistical test comparing forward vs back distributions)
- Is a single forward period sufficient, or does it just test "did this strategy work in 2023-2026" rather than "does this strategy generalize"?

### 5. Walk-forward vs forward/back: redundancy or complementary?

- Currently, candidates eliminated by the forward/back gate skip walk-forward entirely
- Walk-forward runs rolling OOS windows across the entire backtest period, which is a more rigorous test of robustness than a single forward holdout
- Should walk-forward run on ALL candidates regardless of forward gate? It tests a fundamentally different thing (consistency across time windows vs generalization to unseen data)
- Or is this wasteful — if a strategy fails a simple forward test, will it pass rolling walk-forward?

### 6. How many candidates should enter pipeline validation?

- The pipeline costs ~5-10 seconds per candidate. 20 candidates = 2-3 minutes. 100 candidates = 8-15 minutes.
- Is there a diminishing returns curve? (i.e., does testing candidate #50 ever find something that candidates #1-20 missed?)
- Should N be adaptive based on how many pass the forward gate? (If 500 pass, test 50. If 5 pass, test all 5.)
- What's the cost-benefit tradeoff?

### 7. Parameter deduplication: necessary or over-engineering?

- Many of the 58K candidates may differ only in non-signal parameters (e.g., max_bars, stale_exit_bars) that barely affect trade outcomes
- Should we deduplicate by signal parameters only (since those determine which trades are taken)?
- Or does the combination of signal + management parameters matter enough to keep them distinct?
- What distance threshold makes two parameter vectors "the same strategy"?

## Files to Review

For the researcher to understand the current implementation:

- `backtester/optimizer/archive.py` — MAP-Elites diversity archive (the thing we're replacing)
- `backtester/optimizer/run.py` lines 141-274 — `_add_multi_candidates()` function (current candidate selection flow)
- `scripts/full_run.py` lines 320-346 — Pre-elimination of candidates before pipeline
- `backtester/pipeline/runner.py` — Pipeline validation stages and candidate elimination logic
- `backtester/optimizer/config.py` — Configuration: `top_n_candidates=25`, `min_forward_back_ratio=0.4`
- `backtester/core/metrics.py` — DSR implementation
- `backtester/optimizer/sampler.py` — Sobol + EDA sampler (where MAP-Elites is used during search)

## Constraints

- The Rust engine is the sole backend for all evaluations. Python-side changes only.
- Forward-testing is a single `engine.evaluate_batch(param_matrix, EXEC_FULL)` call — trivially parallelized.
- The system runs on Windows 11 (dev) and Linux VPS (production).
- We prioritize system correctness over positive results. Negative results (no edge found) are fine — they're truth.
- Memory budget: 64GB RAM. Batch evaluating 58K candidates at once may need chunking if PnL buffers exceed memory (58K × 25K max_trades × 8 bytes = ~11.6GB per buffer). Current batch_size is 2048-4096 for this reason.

## Expected Output

A recommendation with:
1. Whether our proposed approach is sound or needs modification
2. What the academic/industry consensus is on candidate selection post-optimization
3. Specific thresholds or methods we should use
4. Any risks or failure modes we haven't considered
5. Whether the memory constraint changes the "test all" approach (may need batched forward-testing)
