---
title: "Candidate Selection Pipeline Redesign — Research Report"
notebook_id: "23a238b9-7caf-4caa-904a-c8591f24f49d"
source_count: 55+
date: 2026-03-15
tags: [quant, backtesting, multiple-testing, DSR, MAP-Elites, candidate-selection, walk-forward]
---

# Candidate Selection Pipeline Redesign — Research Report

## Executive Summary

**Verdict: The proposed redesign is partially sound but contains a critical statistical flaw that must be corrected before implementation.**

The proposal to forward-test all 58K candidates is computationally cheap (~15 seconds) but **statistically dangerous** if used as a selection filter. Testing 58K candidates on a single forward holdout and selecting the top 20 by rank converts the forward period into a secondary in-sample optimization set, invalidating it as an out-of-sample test (Bailey et al., 2014). However, the underlying insight — that the MAP-Elites archive is too coarse and discards viable candidates — is correct and fixable.

### Recommended Architecture (Revised)

```
58K passing trials
    │
    ├─ Step 1: Return-stream correlation clustering → N_eff independent strategies
    ├─ Step 2: DSR filter (DSR > 0.95) on IS data → statistically significant candidates
    ├─ Step 3: CPCV/CSCV → PBO < 0.05 gate
    ├─ Step 4: Walk-Forward Analysis → WFE > 0.50 gate
    ├─ Step 5: Final forward holdout (pristine, untouched) → sanity check only
    └─ Survivors enter live paper trading
```

---

## 1. Is Forward-Testing All 58K Candidates Statistically Sound?

**Answer: No — not as a selection filter. Yes — as a batch computation for DSR input.**

### The Core Problem
Bailey, Borwein, López de Prado, and Zhu (2014) explicitly warn: *"Holdout assesses the generality of a model as if a single trial had taken place, again ignoring the rise in false positives as more trials occur. If we apply the holdout method enough times... false positives are no longer unlikely: They are expected."*

Testing 58K candidates on the same forward period and selecting the top 20 by rank is **curve-fitting to the noise of that specific forward window**. The forward period remains truly out-of-sample only for the *first* strategy tested against it.

### DSR Calibration
- **Must use total trials from the entire optimization phase** (200K+), not just the 58K "passing" candidates. Using 58K constitutes the "file drawer problem" (selection bias).
- **Must compute N_eff** (effective independent trials), not raw N. Many parameter sets produce highly correlated returns. Formula: `N_eff = ρ + (1 - ρ)M` where ρ is average correlation.
- **DSR is FWER control** (probability that the *single best* strategy is a false positive). For selecting a *portfolio* of top N, you need hybrid FWER-FDR correction (López de Prado, Lipton, Zoonekynd, 2025).

### Recommended Multiple-Testing Corrections
1. **BHY (Benjamini-Hochberg-Yekutieli) FDR control** — valid under arbitrary dependence. Harvey et al. recommend minimum t-statistic hurdle of **3.0–3.18** for any new discovery.
2. **Hybrid FWER-FDR** — for the specific case of pre-selected "best ideas" from K discarded trials.
3. **CSCV** — replace single forward test with combinatorial cross-validation. PBO > 0.05 → reject.

---

## 2. Industry Practices for Candidate Selection Post-Optimization

**Answer: Yes, this is a solved problem. The literature provides a clear pipeline.**

The consensus framework (from López de Prado, Bailey, Harvey, and practitioners at AQR/Winton):

| Stage | What It Does | Primary Metric |
|-------|-------------|----------------|
| 1. Hypothesis & Data Prep | Define testable hypothesis, purge data for leakage | — |
| 2. QD Search (MAP-Elites) | Fill archive with diverse, high-performing strategies | Archive coverage |
| 3. Parameter Deduplication | Cluster correlated return streams → N_eff | Silhouette score |
| 4. Statistical Deflation | DSR filter on IS performance | DSR > 0.95 |
| 5. CPCV/CSCV | Combinatorial cross-validation | PBO < 0.05 |
| 6. Walk-Forward Analysis | Rolling OOS validation | WFE > 50% |
| 7. Forward Holdout | Pristine final sanity check (ONE use only) | Pass/fail |
| 8. Paper Trading | Live market conditions without capital | Signal fidelity |

**Key insight**: DSR must be applied to *training* performance to filter candidates *before* they touch OOS data. The forward holdout is a one-shot sanity check at the end, not a selection mechanism.

---

## 3. Parameter-Space vs. Metric-Space Diversity

**Answer: Diversity must be defined in behavioral (output) space, not parameter (input) space. But your current behavior characterization is broken.**

### What MAP-Elites Literature Says
- QD algorithms work in **behavioral space**, not genotypic/parameter space (Chatzilygeroudis et al.).
- *"Genotypes and phenotypic structure are poor predictors of how solutions will actually behave in the environment"* — Gomez (2009).
- Pugh et al. (2016): *"Diversity must be defined by a behavior characterization that is typically unaligned (orthogonal) with the notion of quality."*

### Your Archive Is Broken
Your current 2D grid uses:
- Axis 1: Trade frequency (3 buckets) — **acceptable behavior descriptor**
- Axis 2: Quality tier (4 buckets) — **VIOLATES MAP-Elites fundamentals**

Using quality as a diversity axis conflates the fitness function with the behavioral descriptor, causing archive collapse. Two strategies with identical behavior but different quality scores go to different cells, while genuinely different strategies with similar quality collapse to one cell.

### Fix
Replace archive axes with **orthogonal market-derived behaviors**:
- Average holding period
- Long/short bias
- Sensitivity to realized volatility
- Correlation to benchmark

Consider **CVT-MAP-Elites** (Centroidal Voronoi Tessellations) to scale beyond a fixed grid.

### Distance Metrics for Mixed Parameters
**Don't compute distance in parameter space.** Instead:
- Compute correlation matrix of return streams
- Cluster using Silhouette-optimized clustering (López de Prado, 2019)
- Or use Normalized Compression Distance (NCD) on behavioral trajectories (Gomez, 2009)

---

## 4. Forward/Back Ratio Gate Assessment

**Answer: 0.4 is dangerously loose. The single-period forward test is fundamentally inadequate.**

### WFE Thresholds (Literature Consensus)

| WFE Range | Interpretation |
|-----------|---------------|
| > 80% | Exceptional robustness — stable market phenomenon |
| 50–70% | Moderate robustness — typical for viable strategies |
| < 40% | **Significant overfitting** — edge is likely a fluke |
| Negative | Pathological curve-fitting |

**Your 0.4 (40%) threshold is at the exact boundary of "significant overfitting."** The minimum acceptable threshold should be **0.50**.

### Superior Alternatives
1. **Probabilistic Sharpe Ratio (PSR > 0.95)** — corrects for non-normality, sample length
2. **Deflated Sharpe Ratio (DSR > 0.95)** — corrects for PSR + multiple testing
3. **CSCV → PBO < 0.05** — the gold standard for generalization testing

### Critical Flaw
A single forward period merely answers *"Did this strategy work in this specific window?"* not *"Does this strategy generalize?"* Different holdout periods will lead to **opposite conclusions** (Van Belle and Kerr, 2012). You need combinatorial cross-validation to get a *distribution* of out-of-sample outcomes.

---

## 5. Walk-Forward vs. Forward Holdout: Complementary, Not Redundant

**Answer: They test fundamentally different things. Both are required. Neither should gate the other.**

| Method | Tests For | Primary Metric |
|--------|----------|----------------|
| Forward Holdout | Single-window generalization | OOS Sharpe |
| Walk-Forward | Parameter stability across regime changes | WFE |
| CSCV/CPCV | Selection bias / overfitting probability | PBO |

### Key Finding
**Failing a static forward test does NOT predict failure on rolling walk-forward.** A strategy with fixed parameters that degraded during the specific forward regime might adapt excellently under WFA's rolling re-optimization. Gating on forward holdout prematurely kills dynamically robust strategies.

### Recommended Ordering
1. DSR filter (IS data, cheap)
2. Walk-Forward Analysis (rolling OOS, moderate cost)
3. CPCV (combinatorial, most rigorous)
4. Forward holdout (pristine, one-shot final check)

Walk-forward should **not** be skipped based on forward-gate results.

---

## 6. How Many Candidates Should Enter Pipeline Validation?

**Answer: 10–20 maximum, selected via statistical deflation and deduplication, not ranking.**

### The Diminishing Returns Curve Is **Negative**
If candidates #1–20 fail validation but candidate #50 passes, #50 has almost certainly found a **statistical fluke**, not a missed edge. As you test more candidates, the expected maximum Sharpe ratio grows purely from random chance, even if the true expected SR is zero (Bailey & López de Prado, "The False Strategy Theorem", 2021).

### Optimal Stopping Rule
López de Prado recommends the **1/e-law (secretary problem)**:
1. From theoretically justifiable configurations, sample ~37% at random and measure performance
2. Then draw additional configurations one-by-one until you find one that beats all previous
3. Stop. Every additional trial irremediably increases false positive probability.

### Practical Recommendation
- **Deduplicate to N_eff** first (likely 10–50 independent strategies from 58K correlated trials)
- **DSR filter** to statistically significant survivors
- Pass **10–20 max** to expensive validation (WFA + CPCV)
- N should NOT be adaptive to forward-gate pass rate — that's data snooping

---

## 7. Parameter Deduplication: Essential, Not Over-Engineering

**Answer: Critical. But deduplicate on return-stream correlation, not parameter distance.**

### Why It Matters
- Many of 58K candidates differ only in non-signal parameters (max_bars, stale_exit_bars) but produce identical trades
- Raw trial count (58K) artificially inflates DSR penalty → excessive Type II errors (rejecting genuine alphas)
- OR if used without DSR, inflates false discovery rate → Type I errors

### Three Methods for N_eff Calculation

**Method 1: Average Correlation Approximation** (simplest)
```
N_eff = ρ + (1 - ρ) × M
```
Where ρ = average pairwise correlation of return streams across M configurations.

**Method 2: Silhouette Clustering** (recommended, López de Prado 2019)
1. Compute correlation matrix of return time series from all trials
2. Cluster the correlation matrix
3. Find optimal K that maximizes t-value of mean Silhouette score
4. Compute one return series per cluster (minimum variance weighted average)
5. K = N_eff

**Method 3: Spectral Analysis** (most rigorous, López de Prado 2018/2020)
1. Fit Marchenko-Pastur distribution to eigenvalues of correlation matrix
2. Count eigenvalues exceeding the distribution's upper bound
3. Iterate until no more non-trivial eigenvalues remain
4. N_eff = number of removed non-trivial eigenvalues

### Recommendation
- **Do NOT use parameter-space distance** (Gower, Euclidean on mixed types)
- **Cluster on return-stream correlations** — two strategies are "the same" if they produce highly correlated returns, regardless of how different their parameters look
- Feed the cluster centroids (best-performing member per cluster) into the validation pipeline

---

## Revised Pipeline Architecture

Based on the research, here is the recommended pipeline:

### Phase 1: Discovery (Current — Keep)
- Staged optimization (Sobol + EDA) → 200K+ evaluations
- MAP-Elites archive for **search diversity** during optimization (keep this)

### Phase 2: Candidate Consolidation (New)
1. Take all 58K passing trials
2. Compute return-stream correlation matrix (batch `evaluate_batch` for PnL curves)
3. Cluster → find N_eff independent strategies (likely 20–100)
4. Select best-performing member per cluster → N_eff candidates

### Phase 3: Statistical Filtering (New)
1. Compute DSR for each of N_eff candidates using:
   - N = total optimization trials (200K+), adjusted by N_eff
   - V = variance of Sharpe ratios across all trials
   - Return moments (γ₃, γ₄) per candidate
2. Gate: DSR > 0.95
3. Compute BHY-adjusted p-values; require t-stat > 3.0

### Phase 4: Robustness Validation (Revised)
1. **CPCV** on DSR survivors → PBO < 0.05
2. **Walk-Forward Analysis** → WFE > 0.50
3. **Monte Carlo** (existing — keep)
4. **Stability Analysis** (existing — keep)

### Phase 5: Final Sanity Check (Revised)
1. Forward holdout (pristine, untouched data) — **one-shot pass/fail only**
2. Do NOT rank or select based on forward performance
3. **Confidence scoring** (existing — keep)

### Memory Considerations
- 58K × correlation matrix is O(n²) = ~3.4B entries → impractical
- **Solution**: Sample or batch. Cluster signal parameters first (cheap), then compute correlations only within the "unique signal parameter" groups
- Or: compute correlations in chunks of 2048–4096 (existing batch_size) and use approximate clustering

---

## Key References

1. Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality." *Journal of Portfolio Management*.
2. Bailey, D.H., Borwein, J.M., López de Prado, M., & Zhu, Q.J. (2014/2017). "The Probability of Backtest Overfitting." *Journal of Computational Finance*.
3. Harvey, C.R., Liu, Y., & Zhu, H. (2016). "...and the Cross-Section of Expected Returns." *Review of Financial Studies*.
4. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
5. López de Prado, M., Lipton, A., & Zoonekynd, P. (2025). "Deflating the Sharpe Ratio" (hybrid FWER-FDR framework).
6. Pugh, J., Soros, L., & Stanley, K. (2016). "An Extended Study of Quality Diversity Algorithms."
7. Mouret, J.-B. & Doncieux, S. (2011). "Encouraging Behavioral Diversity in Evolutionary Robotics."
8. Siper, M. et al. (2026). "Continuous Program Search" (block-factorized embeddings for strategy DSLs).

---

## Notebook

- **NotebookLM**: [Candidate Selection Pipeline Redesign](https://notebooklm.google.com/notebook/23a238b9-7caf-4caa-904a-c8591f24f49d)
- **Sources**: 55+ academic papers, blog posts, and documentation
- **Deep research**: Web search covering DSR, multiple testing, MAP-Elites, walk-forward, CSCV/CPCV
