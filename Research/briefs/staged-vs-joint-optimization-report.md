# Staged vs Joint Parameter Optimization: Research Report

## Executive Summary & Verdict

**Your staged optimization approach is a known anti-pattern when applied to strongly coupled parameters.** It is formally equivalent to Block Coordinate Descent (BCD), which has rigorous mathematical convergence guarantees only when the non-smooth part of the objective function is block-separable. Trading strategy fitness landscapes are non-convex, non-smooth, and heavily coupled — violating every condition required for BCD convergence. Your current architecture is **mathematically guaranteed to be at high risk of stagnating at suboptimal, non-stationary points**.

However, staging is not entirely wrong. The coverage argument is real: joint optimization of 20-50 parameters with 50K-200K trials produces dangerously sparse coverage. **The answer is not "stage everything" or "joint everything" — it's "stage correctly."** This means grouping coupled parameters together, using cyclic passes instead of single-pass, and replacing your narrow refinement with a mechanism that can escape local basins.

### Verdict Summary

| Question | Answer |
|----------|--------|
| Is staged optimization fundamentally sound? | **Conditionally** — only if parameter blocks are genuinely separable |
| Is your current staging splitting coupled params? | **Yes** — SL/TP split, Signal/Risk split are both dangerous |
| Does the ±3-step refinement recover missed interactions? | **No** — it is mathematically incapable of escaping wrong basins |
| Should you use cyclic passes? | **Yes** — single-pass has zero convergence guarantees for coupled params |
| Which params must stay together? | **Signal + SL + TP + Trailing** (core trade profile) |
| Which params can be safely split? | **Time filters** (most independent), individual management modules |

---

## 1. Block Coordinate Descent: Convergence Theory

### When BCD Converges to a Global Optimum

Your staged approach (Signal → Time → Risk → Management → Refinement) is a literal implementation of the Gauss-Seidel Block Coordinate Descent method. Tseng (2001) proves convergence under these conditions:

1. **f must be convex and continuously differentiable**
2. **The non-smooth components r_i must be convex and strictly block-separable** (each r_i depends only on block x_i)
3. **F must be continuous on a compact level set and attain its minimum**

If these hold, any limit point of cyclic BCD is guaranteed to be a global minimizer.

For **non-convex** problems (like trading), guarantees weaken to:
- Any limit point satisfies **Nash equilibrium conditions** (generalized stationarity) — but only if the non-smooth part remains block-separable
- Global convergence of the full sequence requires the **Kurdyka-Lojasiewicz (KL) inequality**

### When BCD Gets Stuck: The Mathematical Failure

When parameters interact strongly (non-separable, non-smooth coupling), BCD stagnates at **non-stationary points** — points that are not even local optima, but rather "corners" in the landscape where no single-coordinate move can improve the objective despite the combined move being beneficial.

**Classic counterexample (Warga 1963):** For f(x,y) = |x-y| - min(x,y), BCD gets permanently stuck at non-stationary points. Geometrically, when level curves form corners not aligned with coordinate axes, BCD cannot make progress.

**The subdifferential failure:** For non-separable, non-smooth functions, componentwise subgradients p_x ∈ ∂_x f(x,y) and p_y ∈ ∂_y f(x,y) do NOT guarantee that [p_x; p_y] ∈ ∂f(x,y). BCD halts because it perceives axis-aligned minima even when the full subdifferential doesn't contain zero.

### Direct Implication for Your System

Your trading objective (Sharpe Ratio, Net Profit, etc.) is:
- **Non-convex**: discrete trade logic, binary thresholding
- **Non-smooth**: stop-loss triggers create discontinuities
- **Non-separable**: Signal↔Risk, SL↔TP are mathematically coupled

This violates every condition for BCD convergence. Locking Signal parameters before exploring Risk limits the solver to axis-aligned search, making it impossible to navigate diagonal "ridges" where specific signals only become profitable with specific risk configurations.

**Gauss-Seidel vs Jacobi:** The sequential (Gauss-Seidel) update is critical for what convergence BCD does offer. "All-at-once" (Jacobi) schemes can oscillate or diverge. Your system correctly uses Gauss-Seidel ordering, but this only helps for separable problems.

---

## 2. Does the Refinement Stage Recover Missed Interactions?

### Answer: No — It Is a False Safety Net

The refinement stage (±3 steps around locked values) acts as a **localized trust-region search**. It is purely an exploitation mechanism. Unless the bounding box of this narrow radius physically intersects the basin of attraction of the global optimum, the local refinement is mathematically incapable of bridging the gap.

### Basin of Attraction Analysis

When you optimize Signal while Risk is fixed at arbitrary defaults, the solver descends into a local basin specific to those default Risk parameters. By locking Signal and moving to Risk, you restrict search to a coordinate hyperplane. If the global optimum lies in a different basin entirely, the staged solution locks into the wrong region.

**Powell (1973) and Warga (1963)** demonstrate that BCD can stagnate at non-stationary points on coupled, non-smooth functions. A point where BCD halts is not even a local optimum — it's a "corner" where no coordinate-aligned move improves the objective. A narrow grid search around this point merely settles at the bottom of the wrong basin.

### How Wide Must Refinement Be?

There is **no universal step-size** that guarantees recovery. The required width depends on:
- Lipschitz constants of your objective function
- Distance between local basins in parameter space

To guarantee recovery, the refinement would need to span the distance between the suboptimal local minimum and the global minimum — at which point you re-encounter the curse of dimensionality, defeating the purpose of staging.

### The Correct Alternative: Cyclic Descent

**Cyclic coordinate descent (multiple passes) is unequivocally theoretically superior to single-pass staged optimization.**

Standard cyclic BCD iterates through coordinate blocks **until convergence** — defined as "no improvement after one cycle of line search along coordinate directions implies a stationary point is reached." Your system executes a single pass. Optimization theory provides **zero convergence guarantees for a single pass** unless parameters are perfectly separable.

**Recommendation:** Loop stages: Signal → Time → Risk → Management → Signal → Risk → ... until parameter values stabilize between cycles (2-3 passes typically sufficient). If parameters stop shifting, you've found a stationary point.

---

## 3. Curse of Dimensionality vs Curse of Decomposition

### The Formal Trade-Off

| Approach | Coverage | Interaction Capture | Budget Efficiency |
|----------|----------|-------------------|-------------------|
| **Full joint search** | O(ζ^-D) evaluations for ζ accuracy — exponentially sparse | Captures all interactions | Extremely poor for D > 15 |
| **Sequential subspaces** | Dense coverage per block | Misses cross-block interactions | High per-block, but accumulates errors |

The query complexity for full joint search scales **exponentially** with dimensions. With D=30 parameters and 200K trials, your per-point coverage is vanishingly small. Staging provides near-complete coverage per subspace but introduces the "curse of decomposition" — accumulated errors from missed interactions.

### When Decomposition Wins

Decomposition is superior when:
1. **The function has low effective dimensionality** — most variance is explained by a small subset of parameters
2. **Interactions are weak or absent** — the function is approximately additive: f(x) ≈ Σ f_i(x_i)
3. **The Sobol first-order indices sum close to 1** — meaning main effects dominate and interaction effects are small

### When Joint Search Wins

Joint search is necessary when:
1. **Strong pairwise interactions exist** (S_Ti >> S_i for key parameters)
2. **The function landscape has diagonal ridges** — optima require simultaneous parameter changes
3. **The effective dimensionality is high** — many parameters jointly determine performance

### The Resolution: Structured Decomposition

Modern approaches don't choose between "stage everything" and "joint everything." Instead:

1. **Additive BO models** decompose f(x) = f(Signal,Risk) + f(Risk,Time) — blocks can **overlap**, unlike your strict sequential staging
2. **Random Tree Decompositions (RDUCB)** randomly group parameters into small tree structures, proving that random groupings balance information gain and functional mismatch better than fixed sequential stages
3. **Sequential Random Embeddings (SRE)** project the full space through random matrices, naturally testing linear combinations across parameter groups

### For Your 20-50 Dimensional Problem

At your dimensionality (20-50 params), the optimal strategy is:
- **Group coupled parameters** into blocks of 5-8 (manageable for 200K trials)
- **Use SRE or random subspace sampling** for the refinement phase instead of narrow grids
- **Run 2-3 cyclic passes** instead of single-pass

---

## 4. Which Trading Parameters Actually Interact?

### SL ↔ TP: Strong Interaction (DO NOT SPLIT)

SL and TP exhibit **strong non-linear interactions** beyond the obvious risk-reward ratio:

- **Empirical evidence (MDPI study):** Stop-loss orders accelerate adverse price movements while take-profit orders oppose trends. Their optimal distances are deeply coupled with asset-specific volatility.
- **Leung & Li (2015):** Stop-loss must be calculated dynamically at every tick relative to price, making it inseparable from the take-profit boundary.
- **The return distribution truncation effect:** SL and TP collectively truncate the tails of the return distribution. Optimizing SL without a paired TP assumes an incomplete trade trajectory.
- **Klement (2013):** Optimal SL distance is highly sensitive to whether TP is fixed, variable, or combined with a reentry condition.

**Verdict:** SL and TP must be in the same optimization block. Splitting them is a known anti-pattern.

### Signal ↔ Risk: Strong Interaction (SHOULD BE JOINT)

**MACD/ATR study evidence:** When MACD entry signals were jointly paired with a sliding, variable ATR stop-loss zone, profitability spiked in specific parameter neighborhoods (N=12, x=6, y=2). This peak was only discoverable through joint optimization — isolated signal optimization produced different (worse) optima.

**Key finding:** Optimizing the MACD entry with a faster TP signal actually underperformed the simple MACD system. But when jointly paired with specific ATR trailing stop geometry, performance improved dramatically. The optimal entry signal parameters are deeply dependent on exit rule geometry.

**Your concern is validated:** Locking signal params in Stage 1 biases them toward whatever default risk parameters are active during that stage. When Stage 2 subsequently optimizes risk, it's bounded by the suboptimal signal.

### Trailing Stop ↔ SL/TP: Strong Interaction (MUST GROUP)

The MACD/ATR study compared static vs sliding (trailing) vs sliding+variable ATR barriers:
- **Static ATR SL:** Small multipliers diminished profits; performance only stabilized at large multipliers that practically never triggered
- **Sliding+variable ATR:** Global optimum shifted drastically to a tight neighborhood (N=12, x=6) — a completely different region than static optimization found
- **Mechanism:** A trailing stop fundamentally changes the required breathing room for a trade. A static SL/TP optimization stage cannot find the trailing-optimal parameters because trailing changes the entire trade survival geometry.

**Verdict:** If trailing is inactive during SL/TP optimization, the optimizer favors wider static stops, structurally handicapping the subsequent trailing stage. These must be grouped.

### Breakeven ↔ SL: Moderate-Strong Interaction

Breakeven effectively replaces the original SL. The "best SL" depends on whether BE is active. If BE triggers early, a wider initial SL is optimal (it provides breathing room knowing BE will protect). If BE is off, a tighter SL is needed for risk control.

**Verdict:** BE should be in the same block as SL/TP/trailing.

### Time Filters ↔ Everything Else: Weak Interaction (SAFE TO SPLIT)

**Empirical evidence:** Closing positions over weekends vs holding them produced a consistent, scaling impact across all tested assets — reduced drawdowns but also reduced profits. This was largely independent of specific SL/TP settings.

Time filters act as **global regime modifiers** — they mask out entire trading regimes rather than altering per-trade execution geometry. Their interaction strength (S_ij) with specific SL/TP thresholds is lower than other parameter groups.

**Verdict:** Time filters are the safest group to optimize separately. They can remain in their own stage.

### Management Modules: Mostly Independent (SAFE TO SPLIT from each other)

Trailing stop, partial close, max bars, and stale exit are largely orthogonal to each other (they address different aspects of position lifecycle). Sub-staging them individually is reasonable.

**However:** Trailing and breakeven MUST be in the same block as SL/TP because they override base risk geometry.

---

## 5. Alternatives to Pure Sequential Staging

### Alternative 1: Group Interacting Parameters (RECOMMENDED)

Expand block sizes to jointly optimize coupled variables (SL + TP + Trailing + BE in one "Trade Management" block).

| Aspect | Assessment |
|--------|-----------|
| **Computational cost** | Higher per-block (multiplied search space), but manageable if blocks stay at 5-8 params |
| **Convergence** | Satisfies block-separability requirement → prevents stagnation on non-smooth ridges |
| **Feasibility** | HIGH — the most practical improvement |

### Alternative 2: Sobol Sensitivity Indices for Data-Driven Grouping

Run variance-based sensitivity analysis before defining stages. Calculate S_i (main effects) and S_Ti (total effects including interactions). If S_Ti >> S_i, that parameter interacts strongly and cannot be isolated.

| Aspect | Assessment |
|--------|-----------|
| **Computational cost** | VERY HIGH — N(d+2) evaluations needed, typically 85K+ for 15 params with N=5000 |
| **Value** | Mathematically rigorous grouping decisions |
| **Feasibility** | LOW to MODERATE as automated step; MODERATE as one-time diagnostic |
| **Limitation** | Assumes independent input distributions — loses interpretability for constrained parameters |

**Recommendation:** Use as a one-time diagnostic to validate your grouping assumptions, not as a per-strategy automated step.

### Alternative 3: Cyclic Coordinate Descent (RECOMMENDED)

Loop stages: Signal → Risk → Management → Signal → Risk → ... until parameters stabilize.

| Aspect | Assessment |
|--------|-----------|
| **Computational cost** | Linear with number of cycles (2-3x budget for 2-3 passes) |
| **Convergence** | Proven for smooth functions with block-separable non-smooth parts |
| **Feasibility** | VERY HIGH — minimal code changes |
| **Limitation** | On strongly coupled, non-convex functions, may cycle endlessly or stagnate at same suboptimal point found in first pass |

**Key insight:** Cyclic descent is necessary but may not be sufficient for strongly coupled params. It should be combined with block merging (Alternative 1).

### Alternative 4: Hybrid Staged + Joint Space Sampling (RECOMMENDED)

Allocate 20% of trial budget to sample the full joint parameter space using Genetic Algorithms or CMA-ES, while 80% goes to staged optimization.

| Aspect | Assessment |
|--------|-----------|
| **Computational cost** | Controlled — just reallocate existing budget |
| **Convergence** | GA mutations can jump across non-smooth ridges, escaping BCD local optima |
| **Feasibility** | HIGH — practical compromise |
| **Value** | Removes the dangerous assumption that the global optimum is adjacent to the staged local optimum |

### Alternative 5: Sequential Random Embeddings (SRE) (ADVANCED, HIGH POTENTIAL)

SRE projects the high-dimensional space through random matrices, optimizes in low-dimensional subspaces, and iteratively reduces the residue. Naturally tests linear combinations across parameter groups.

| Aspect | Assessment |
|--------|-----------|
| **Computational cost** | Highly efficient — optimization runs in d << D dimensions, making 50K-200K budget immensely powerful |
| **Convergence** | Proven to strictly reduce the solution gap at each step for functions with low optimal ε-effective dimension |
| **Feasibility** | MODERATE — requires implementing random projection infrastructure |
| **Key advantage** | If Signal Param A and Risk Param B interact, random embedding naturally projects them into shared subspace — no manual grouping required |

---

## 6. Commercial Platform & Academic Practice

### Commercial Platforms

| Platform | Approach | Notes |
|----------|----------|-------|
| **MetaTrader 5** | **Joint optimization** with Genetic Algorithm | Optimizes all selected parameters simultaneously. GA "skips many combinations" — biased for large search spaces. Does NOT support sequential locking. |
| **StrategyQuant** | **Joint optimization** within Walk-Forward Matrix | Optimizes parameters jointly (fast/slow MAs alongside money management rules) |
| **QuantConnect** | **Joint optimization** during WFO training steps | Evaluates joint parameter space simultaneously during in-sample training |
| **NinjaTrader** | **Joint optimization** via Strategy Analyzer | Iterates through ranges of all selected input values together |

**Key finding:** No major commercial platform uses your staged/sequential locking approach. They all default to joint optimization, using GAs or grid search to handle dimensionality.

### Academic Literature

- **Walk-Forward Optimization (WFO)** frameworks handle multiple parameters by optimizing them jointly in each in-sample window, then testing out-of-sample. The decomposition is temporal (rolling windows), not parametric (staged blocks).
- **Additive Bayesian Optimization** models f(x) as a sum of overlapping low-dimensional functions — blocks CAN overlap, unlike your strict sequential staging.
- **Random Decomposition UCB (RDUCB, Ziomek 2023)** proves that random tree-based groupings balance information gain and interaction mismatch better than fixed sequential stages. Data-driven decomposition learning is "easily misled by local modes that do not hold globally."

### Documented Failures of Staged Optimization

The RDUCB paper demonstrates that the state-of-the-art Tree algorithm (data-driven decomposition) gets stuck in local modes on benchmark functions, while random decompositions eventually find the global optimum. This directly parallels your concern: learned or fixed decompositions that seem good locally can be catastrophically wrong globally.

---

## 7. Revised Pipeline Architecture

### Recommended Stage Structure

```
CURRENT (8+ stages, single pass):
  Signal → Time → Risk(SL) → Risk(TP) → Trailing → BE → PartialClose → MaxBars → StaleExit → Refinement(±3)

PROPOSED (4 stages, 2-3 cyclic passes):
  Pass 1:
    Stage 1: Signal Generation [all signal params]
    Stage 2: Core Trade Profile [SL + TP + Trailing + BE] ← MERGED
    Stage 3: Auxiliary Management [PartialClose + MaxBars + StaleExit]
    Stage 4: Time Filters [session times, day filters]
  Pass 2:
    Stage 1: Signal (with Stage 2-4 values from Pass 1)
    Stage 2: Core Trade Profile (with updated Signal from Pass 2)
    Stage 3: Auxiliary Management
    → Check: did params shift >threshold? If yes, Pass 3. If no, converged.
  Final:
    SRE/Hybrid refinement (random subspace sampling across ALL params, 20% of budget)
    OR: GA-based joint exploration (full parameter space, 20% of budget)
```

### Grouping Rationale

| Stage | Parameters | Interaction Strength | Separability |
|-------|-----------|---------------------|-------------|
| **Signal** | Entry indicators, thresholds, lookbacks | HIGH with Core Trade Profile | Moderate — can be staged if cycled |
| **Core Trade Profile** | SL, TP, Trailing Stop, Breakeven | VERY HIGH internal coupling | MUST be joint — splitting is anti-pattern |
| **Auxiliary Management** | Partial Close, Max Bars, Stale Exit | LOW internal coupling, LOW with Signal | Safe to stage separately |
| **Time Filters** | Session times, day-of-week, holidays | LOW with everything | Safest to split — most independent |

### Key Thresholds

| Parameter | Current | Recommended | Rationale |
|-----------|---------|-------------|-----------|
| **Refinement width** | ±3 steps | ±10 steps or SRE | ±3 is a false safety net; too narrow to escape wrong basins |
| **Number of passes** | 1 (single pass) | 2-3 (cyclic) | Zero convergence guarantee for single pass on coupled params |
| **Stages** | 8+ | 4 | Merge coupled params; keep independent ones separate |
| **Joint exploration budget** | 0% | 15-20% of total | Escape local optima via GA/SRE mutations |

### Convergence Detection

After each cyclic pass, measure parameter stability:
```
convergence_metric = max(|params_pass_n - params_pass_n-1|) / param_range
if convergence_metric < 0.05:  # Parameters shifted less than 5% of range
    STOP — stationary point reached
else:
    RUN ANOTHER PASS
```

---

## 8. Sobol Index Diagnostic Protocol

Before committing to your staging architecture, run this one-time diagnostic:

### Step 1: Generate Sobol Sensitivity Indices

```
N = 5000 samples (quasi-Monte Carlo, Sobol sequence)
d = number of parameters
Total evaluations = N × (d + 2) ≈ 85,000 for d=15
```

### Step 2: Compute Indices

For each parameter i:
- **S_i** (first-order): variance explained by parameter alone
- **S_Ti** (total-order): variance explained including all interactions
- **Interaction ratio**: (S_Ti - S_i) / S_Ti

### Step 3: Decision Rule

| Interaction Ratio | Action |
|-------------------|--------|
| < 0.1 | Parameter is effectively independent → safe to isolate in own stage |
| 0.1 - 0.3 | Weak interactions → can stage separately if cyclic passes are used |
| 0.3 - 0.5 | Moderate interactions → group with interacting partners |
| > 0.5 | Strong interactions → MUST be optimized jointly |

### Step 4: Validate Groupings

If Σ S_i ≈ 1.0, the function is approximately additive and staging is safe.
If Σ S_i << 1.0, interactions dominate and you need larger joint blocks.

---

## 9. Key References

### Optimization Theory
- **Tseng (2001)** — "Convergence of a Block Coordinate Descent Method for Nondifferentiable Minimization" — foundational BCD convergence theory
- **Powell (1973)** — counterexample showing cyclic BCD failure on non-convex functions
- **Warga (1963)** — counterexample showing BCD stagnation on non-smooth, non-separable functions
- **Xu & Yin** — "The block-coordinate descent method for nonconvex optimization" — KL inequality convergence, Nash equilibrium conditions
- **Shi et al. (2016)** — "A Primer on Coordinate Descent Algorithms" — comprehensive survey of BCD variants, failure modes, and convergence results
- **Beck & Tetruashvili (2013)** — convergence rates for cyclic CD: O(1/ε) under Lipschitz gradient, O(log(1/ε)) under strong convexity

### High-Dimensional Optimization
- **Qian, Hu & Yu (2016)** — "Derivative-Free Optimization of High-Dimensional Non-Convex Functions by Sequential Random Embeddings" — SRE theory and ε-effective dimension
- **Ziomek & Bou-Ammar (2023)** — "Are Random Decompositions all we need in High Dimensional Bayesian Optimisation?" — RDUCB algorithm, random tree decompositions
- **Eriksson et al. (2019)** — TuRBO: trust region Bayesian optimization for high dimensions

### Sensitivity Analysis
- **Sobol' (2001)** — Global sensitivity indices for nonlinear mathematical models
- **Saltelli et al.** — Variance-based sensitivity analysis estimators (Jansen, Saltelli methods)
- **OpenTURNS documentation** — Comprehensive treatment of Sobol' decomposition and interaction indices

### Trading Strategy Optimization
- **Thomaidis et al. (2018)** — "Take Profit and Stop Loss Trading Strategies Comparison in Combination with an MACD Trading System" (MDPI) — empirical evidence on SL/TP/trailing interaction effects
- **Leung & Li (2015)** — dynamic stop-loss computation requirements
- **Austin et al. (2004)** — special consideration for SL/TP function interactions in adoptive trading systems
- **QuantConnect, MetaTrader 5, StrategyQuant** — Walk-forward optimization documentation

---

## NotebookLM Research Workspace

**Notebook:** Staged vs Joint Parameter Optimization for Trading Systems
**Notebook ID:** a0235747-949e-4cae-b003-f5c97bae3ec5
**URL:** https://notebooklm.google.com/notebook/a0235747-949e-4cae-b003-f5c97bae3ec5
**Sources:** 19 (1 research brief + 18 academic/documentation URLs)
**Queries executed:** 6 deep analytical queries with citation tracking
