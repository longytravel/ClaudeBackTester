# Optimizer Engine Upgrade: Research Report

## Executive Summary & Verdict

**Your Sobol+CE engine needs an upgrade, but not a full replacement.** Cross-Entropy's non-incremental covariance estimation causes premature convergence in 10-15 dimensional spaces, especially on multimodal landscapes. CMA-ES is the natural successor — it maintains your batch-parallel architecture, handles non-separable parameter interactions via incremental covariance learning, and outperforms CE on exactly the class of problems your trading optimizer faces.

**The minimum viable upgrade is: keep Sobol for exploration, replace CE with CMA-ES (CatCMAwM variant) for exploitation.** This requires minimal architectural changes — both use population-based batch evaluation with ask-tell interfaces.

### Verdict Table

| Question | Answer |
|----------|--------|
| Should we upgrade from Sobol+CE? | **Yes** — CE breaks down at 10-15 dims on multimodal landscapes |
| Best replacement for CE? | **CMA-ES (CatCMAwM)** — handles mixed types, learns interactions, batch-native |
| Can we keep batch-parallel evaluation? | **Yes** — CMA-ES is population-based, perfect fit for Rust hot loop |
| Should we use Optuna? | **No** — TPE is sequential, batch quality degrades, suggestion overhead kills throughput |
| Should we use TuRBO/BO? | **No** — GP overhead is 2000x slower than your eval time |
| Best framework? | **`cmaes` library** (CyberAgentAILab) for CMA-ES; **pymoo** if adding multi-objective |
| Migration risk? | **Low** — ask-tell API mirrors your current batch architecture |

---

## 1. Cross-Entropy Method: Scaling Breakdown

### Why CE Fails at 10-15 Dimensions

CE and CMA-ES share the idea of sampling from a parameterized Gaussian and updating based on elite performers. Their critical difference:

| Mechanism | Cross-Entropy | CMA-ES |
|-----------|--------------|--------|
| **Covariance estimation** | Non-incremental (from scratch each iteration) | Incremental (rank-mu + rank-one updates) |
| **What is modeled** | Likelihood of successful *points* | Likelihood of successful *search steps* |
| **Step-size control** | None | Cumulative Step-size Adaptation (CSA) |
| **Historical memory** | Discarded each iteration | Evolution path (cumulation) |
| **Variance behavior** | Invariably smaller → premature convergence | Maintained via CSA → prevents collapse |

**The core failure:** CE rebuilds its distribution from scratch each iteration using only the current elite samples. To reliably estimate a full n×n covariance matrix without singularity, CE needs elite samples scaling **quadratically** with dimension. At n=15, this means CE needs hundreds of elites per iteration — destroying sample efficiency.

**CMA-ES's advantage:** Because CMA-ES updates incrementally, it decouples covariance reliability from population size. CMA-ES operates with vastly smaller populations than CE, achieving reliable estimates even with λ = 4 + 3·ln(n) ≈ 12 for n=15.

### Empirical Evidence

- On the 20D Michalewicz function, CE demonstrated **premature convergence** for small sample sizes, while CMA-ES avoided local minima
- A "pure CE" (without step-size control) converges prematurely **even on a linear function** — CMA-ES does not
- In a survey of 31+ black-box optimization algorithms, CMA-ES outranked all others, performing "especially strongly on difficult functions or larger-dimensional search spaces"

### Verdict on Sobol+CE for 10-15 Params

**Sobol remains excellent for exploration** — its low-discrepancy sequences provide uniform coverage regardless of dimension. **CE must be replaced for exploitation** — its Gaussian assumption breaks down on multimodal trading landscapes at 10+ dimensions, and its lack of step-size control causes variance collapse before reaching the optimum.

---

## 2. CMA-ES: The Natural Successor

### Why CMA-ES Fits Your Problem

CMA-ES is designed specifically for the exact problem class your trading optimizer faces:

| Property | Your Problem | CMA-ES Capability |
|----------|-------------|-------------------|
| Non-convex | Trading fitness (Sharpe, profit) is highly non-convex | Designed for non-convex optimization |
| Non-separable | SL↔TP↔Trailing are coupled | Learns covariance = inverse Hessian approximation |
| Ill-conditioned | Parameters have vastly different scales | Rotation-invariant, scale-invariant |
| Multimodal | Multiple viable strategy configurations | Population-based global search |
| Noisy | Backtest results have sampling noise | Robust to noise via population averaging |
| 10-50 dimensions | Your merged stages have 10-15 params | "Sweet spot" for CMA-ES (3-100 dims) |

### Three Mechanisms Driving Superiority

1. **Covariance Adaptation:** On convex-quadratic functions, C adapts to approximate H⁻¹ (inverse Hessian). On your non-separable SL/TP landscape, this means CMA-ES automatically learns that "when SL goes wider, TP should go tighter" — the exact interaction your staged optimizer misses.

2. **Evolution Paths (Cumulation):** Records time evolution of the distribution mean across generations. If consecutive steps move in a similar direction, the path lengthens, accelerating variance increase in favorable directions. CE has no equivalent mechanism.

3. **Cumulative Step-Size Control (CSA):** Aims to make consecutive movements orthogonal in expectation. Prevents premature convergence (CE's fatal flaw) while maximizing convergence rate.

### Performance Metrics for n=15

- **Convergence rate:** Log-linear (exponential) — distance to optimum decreases by constant factor exp(-c) per iteration, with c ≈ 0.1·λ/n
- **Default population:** λ = 4 + 3·ln(15) ≈ 12 candidates per generation
- **Scaled population:** For global search, scale up to λ = 10n = 150
- **Computational overhead:** O(n²) per candidate (eigendecomposition postponed Ω(n) iterations)
- **Budget requirement:** On rugged functions, CMA-ES dominates when budget exceeds 100n = 1,500 evaluations. Your 50K-200K budget is **massive overkill** for CMA-ES — enabling extremely thorough search

### Mixed Parameter Handling: CatCMAwM

Your parameters include continuous (SL value), discrete (SL mode: fixed/ATR/percent), boolean (trailing on/off), and categorical types. Standard CMA-ES handles only continuous variables. Modern variants solve this:

| Variant | Handles | Mechanism |
|---------|---------|-----------|
| **CMAwM** (CMA-ES with Margin) | Continuous + integer + binary | Lower bound on marginal probability prevents collapse to single discrete value |
| **CatCMA** | Continuous + categorical | Joint probability distribution of multivariate Gaussian + categorical |
| **CatCMAwM** | All types simultaneously | Unified framework, outperforms CMAwM in mixed-integer benchmarks |

**CatCMAwM is the recommended variant** — it handles your exact parameter mix (continuous SL values + discrete modes + boolean toggles) in a unified framework.

The `cmaes` Python library (CyberAgentAILab) implements all three variants with ask-tell interface:
```python
from cmaes import CMAwM  # or CatCMAwM

optimizer = CMAwM(mean=np.zeros(dim), sigma=2.0, bounds=bounds, steps=steps)
for generation in range(max_gens):
    solutions = []
    for _ in range(optimizer.population_size):
        x_eval, x_tell = optimizer.ask()  # x_eval has discrete values, x_tell is continuous
        value = rust_evaluate(x_eval)       # Send to Rust hot loop
        solutions.append((x_tell, value))   # Tell with continuous representation
    optimizer.tell(solutions)
```

---

## 3. Batch-Parallel Evaluation: Algorithm Comparison

### Your Constraint

Your Rust hot loop evaluates 18K-35K parameter sets per second (BASIC mode). Any optimizer requiring sequential trial-by-trial suggestions destroys this advantage. The key metric is **suggestion overhead per batch** relative to evaluation time.

### Head-to-Head Comparison

| Optimizer | Batch Native? | Suggestion Overhead | Batch Quality | Verdict |
|-----------|:------------:|---------------------|---------------|---------|
| **CMA-ES** | **Yes** | O(n²) per generation (~microseconds for n=15) | Full quality — population sampling is independent | **PERFECT FIT** |
| **GA/NSGA-II** | **Yes** | Negligible arithmetic per generation | Full quality — population-based | **GOOD FIT** |
| **Nevergrad** | **Yes** | Lightweight, wraps CMA/DE/PSO | Full quality — `batch_mode=True` supported | **GOOD FIT** |
| **Sobol+CE** (current) | **Yes** | Negligible | Full quality but CE breaks at high dims | **CURRENT** |
| **TPE/Optuna** | **Partial** | `suggest_...` called in Python loop — slow | **Degrades** with batch size (constant liar hack) | **REJECT** |
| **TuRBO/BoTorch** | **Partial** | GP fitting: O(N³) — takes **minutes** for N>1000 | Good via Thompson Sampling | **REJECT** |
| **SMAC/BOHB** | **Partial** | Random Forest fitting + acquisition optimization | Moderate — designed for expensive evals | **REJECT** |

### The Break-Even Analysis

Your evaluation cost: ~30-55 microseconds per trial (at 18K-35K/sec).

For a smarter optimizer to justify its overhead, it must require proportionally fewer samples:

| Optimizer | Overhead per suggestion | Overhead/Eval ratio | Samples needed to break even |
|-----------|----------------------|--------------------:|------------------------------|
| CMA-ES | ~1 μs | ~0.03x | 1x (always wins) |
| NSGA-II | ~5 μs | ~0.15x | 1x (always wins) |
| TPE | ~1-10 ms | ~200x | Need 200x fewer samples (impossible) |
| TuRBO | ~60 ms per point | ~2000x | Need 2000x fewer samples (impossible) |

**Conclusion:** Only population-based methods (CMA-ES, GA, Nevergrad) operate within the break-even threshold. TPE and BO are architecturally incompatible with your system.

---

## 4. Conditional Parameters: The Architecture Challenge

### The Problem

Your merged Core Trade Profile has conditional dependencies:
- `trailing_distance` only matters if `trailing_mode != OFF`
- `BE_offset` only matters if `breakeven_enabled = True`
- `TP_value` interpretation depends on `TP_mode` (fixed/ATR/percent)

When inactive conditional parameters are included in the covariance matrix, they inject noise into the adaptation, degrading convergence.

### How Each Optimizer Handles This

| Optimizer | Conditional Support | Mechanism | Quality |
|-----------|:------------------:|-----------|---------|
| **TPE/Optuna** | **Native** | Tree-structured KDE — branches only active when condition met | **Best** — but fails batch constraint |
| **SMAC** | **Native** | Random Forest partitions hierarchical spaces | **Good** — but too slow for your eval speed |
| **GA** | **Good** | Hierarchical chromosome encoding | **Good** — batch-compatible |
| **CMA-ES** | **Poor natively** | Requires encoding tricks or masking | **Workable** with architecture below |
| **Nevergrad** | **Moderate** | Instrumentation API supports conditional | **Workable** |

### Practical Solution for CMA-ES + Conditional Params

Since CMA-ES is the best batch-parallel optimizer but handles conditionals poorly, use a **two-layer architecture**:

```
Layer 1: Categorical/Boolean optimizer (GA or exhaustive)
  - Explores all mode combinations: {trailing_mode, BE_enabled, SL_mode, TP_mode}
  - Small discrete space: e.g., 3 × 2 × 3 × 3 = 54 combinations

Layer 2: CMA-ES (CatCMAwM) per active configuration
  - For each mode combination, optimize only the ACTIVE continuous params
  - trailing_mode=OFF → exclude trailing params from CMA-ES
  - This keeps CMA-ES in its "sweet spot" of purely continuous optimization
```

This approach:
- Eliminates conditional noise from the covariance matrix
- Reduces CMA-ES dimensionality per configuration (from 15 to ~8-10)
- Budget allocation: 200K trials / 54 configs ≈ 3,700 trials per config (still 250x CMA-ES minimum)
- Can prune non-viable configs early (e.g., if trailing=OFF consistently underperforms)

---

## 5. Framework Selection

### Detailed Comparison

| Framework | CMA-ES | Mixed Types | Conditional | Batch | Multi-Obj | Maturity | Verdict |
|-----------|:------:|:-----------:|:-----------:|:-----:|:---------:|:--------:|---------|
| **`cmaes`** | CatCMAwM | **Best** | Manual | **Yes** (ask-tell) | Via COMO-CatCMAwM | Production | **Primary choice** |
| **Nevergrad** | Internal CMA | Good | Instrumentation API | **Yes** (`batch_mode`) | DE-based | Production | **Good alternative** |
| **pymoo** | pycma wrapper | SBX operators | Custom operators | **Yes** (population) | **NSGA-II/III** | Production | **For multi-objective** |
| **Optuna** | Via `cmaes` | TPE native | **Best** (TPE) | **Poor** | Pareto sampler | Production | **Reject** (batch issue) |
| **DEAP** | Not included | Custom encoding | Custom | **Yes** (population) | NSGA-II | Mature but old | Low priority |
| **pycma** | Hansen's reference | Basic int handling | No | **Yes** (ask-tell) | MO-CMA-ES | Reference impl | For pure continuous |

### Recommendation

**Primary: `cmaes` library (CyberAgentAILab/cmaes)**
- Implements CatCMAwM — the state-of-the-art for mixed-variable CMA-ES
- Clean ask-tell interface maps directly to your batch architecture
- Warm Starting CMA-ES (WS-CMA) for Sobol→CMA-ES handoff
- Used internally by Optuna's CMA-ES sampler
- Actively maintained, recent paper (2024)

**Secondary: pymoo (for future multi-objective)**
- When ready to optimize Sharpe + drawdown + trade count simultaneously
- NSGA-II/III with custom operators for mixed types
- Population-based, batch-native
- Can wrap `cmaes` for the CMA-ES component

**Reject: Optuna** — Despite excellent conditional parameter handling, TPE's sequential nature and batch quality degradation make it architecturally incompatible with your 18K-35K evals/sec Rust evaluator. The suggestion overhead alone would throttle throughput by 200x.

**Reject: TuRBO/BoTorch** — GP fitting overhead of minutes per 10K evaluations vs your 0.28 seconds. Designed for expensive evaluations (minutes per eval), not your microsecond evaluations.

---

## 6. Recommended Architecture

### Phase 1: Minimum Viable Upgrade (Replace CE with CMA-ES)

```
CURRENT:
  Sobol exploration (N trials) → CE exploitation (remaining budget)

UPGRADED:
  Sobol exploration (50% budget) → CMA-ES/CatCMAwM exploitation (50% budget)
```

**Implementation:**
```python
# Phase 1: Sobol exploration (unchanged)
sobol_results = rust_evaluate_batch(sobol_generate(budget * 0.5))
top_k = select_elite(sobol_results, k=100)

# Phase 2: CMA-ES exploitation (replaces CE)
from cmaes import CMAwM  # or CatCMAwM for categoricals

# Initialize CMA-ES from Sobol elites (Warm Start)
initial_mean = np.mean(top_k, axis=0)
initial_sigma = np.std(top_k)

optimizer = CMAwM(
    mean=initial_mean,
    sigma=initial_sigma,
    bounds=param_bounds,
    steps=discretization_steps,  # 0 for continuous, 1 for integer, etc.
    population_size=max(15, 4 + int(3 * np.log(n_params)))
)

remaining_budget = budget * 0.5
evals_done = 0
while evals_done < remaining_budget:
    # Ask: generate batch
    batch = []
    for _ in range(optimizer.population_size):
        x_eval, x_tell = optimizer.ask()
        batch.append((x_eval, x_tell))

    # Evaluate: send to Rust hot loop
    x_evals = [b[0] for b in batch]
    results = rust_evaluate_batch(x_evals)

    # Tell: update CMA-ES
    solutions = [(batch[i][1], results[i]) for i in range(len(batch))]
    optimizer.tell(solutions)
    evals_done += len(batch)
```

**Effort:** ~1-2 days of implementation. Minimal changes to existing pipeline.

### Phase 2: Two-Layer Architecture for Conditional Params

```
ARCHITECTURE:
  Layer 1: Mode Enumeration
    - Enumerate all {trailing_mode, BE_mode, SL_mode, TP_mode} combinations
    - Filter to viable combinations (e.g., 20-50 configs)

  Layer 2: Per-Config CMA-ES
    - For each mode config, extract ACTIVE continuous params only
    - Run Sobol(50%) → CatCMAwM(50%) on reduced param set
    - Early termination for clearly non-viable configs

  Final: Compare best result across all configs
```

**Effort:** ~1 week. Requires refactoring parameter space definition.

### Phase 3: Multi-Objective (Future)

```
ARCHITECTURE:
  Replace single-objective CMA-ES with NSGA-II (pymoo)
  Objectives: Sharpe ratio, max drawdown, trade count
  Population-based: batch-native for Rust evaluator
  Output: Pareto front of non-dominated strategies
```

**Effort:** ~2 weeks. Requires defining multi-objective fitness function.

### Phase 4: Cyclic Multi-Pass (from prior research)

```
ARCHITECTURE:
  Pass 1: Signal Stage (Sobol→CMA-ES) → Core Trade Profile (Sobol→CMA-ES) → ...
  Pass 2: Signal (with Pass 1 results) → Core Trade Profile (updated) → ...
  Convergence check: stop when params shift < 5% between passes
```

**Effort:** ~1 day on top of Phase 1. Wraps existing stages in a loop.

---

## 7. Budget Allocation Strategy

### Current vs Recommended

| Phase | Current (Sobol+CE) | Recommended (Sobol+CMA-ES) |
|-------|-------------------|---------------------------|
| **Exploration** | Sobol (varies) | Sobol 50% of budget |
| **Exploitation** | CE (varies) | CMA-ES/CatCMAwM 40% of budget |
| **Global escape** | ±3 step refinement | Random restart CMA-ES 10% of budget |

### CMA-ES Population Sizing for Your Budget

| Stage Params | Default λ | Recommended λ | Generations (200K budget) |
|:------------:|:---------:|:-------------:|:-------------------------:|
| 10 | 11 | 50-100 | 1,000-2,000 |
| 15 | 12 | 75-150 | 670-1,330 |
| 20 | 13 | 100-200 | 500-1,000 |

With λ=100 and 200K budget: 1,000 generations of CMA-ES. This is **extremely thorough** — CMA-ES typically converges on 15D problems in 100-300 generations.

### IPOP-CMA-ES: Automatic Restart Strategy

Instead of a fixed population, use **IPOP-CMA-ES** (Increasing Population CMA-ES):
1. Start with small λ (fast local search)
2. When CMA-ES converges, **restart with 2x population** (broader global search)
3. Repeat until budget exhausted
4. Best result across all restarts wins

This eliminates the need for separate "exploration" and "exploitation" phases — CMA-ES handles both via the restart mechanism.

---

## 8. What NOT to Use (and Why)

### Optuna/TPE

- TPE is sequential — each suggestion depends on all prior trials
- Batch mode uses "constant liar" hack — quality degrades with batch size
- `suggest_...` called in Python loop — suggestion latency >> eval time
- **Your evals cost 30μs; TPE suggestions cost 1-10ms = 200x overhead**

### TuRBO/BoTorch/Gaussian Process BO

- GP fitting: O(N³) complexity — at N=10,000 evals, takes **10 minutes** of overhead
- Your Rust loop does 10,000 evals in **0.28 seconds**
- Designed for expensive evaluations (minutes per eval), not cheap evaluations
- **Overhead ratio: 2000x — totally incompatible with your architecture**

### SMAC/BOHB

- Random Forest surrogates are excellent for conditional parameters
- But designed for expensive evaluations (model training taking hours)
- Acquisition function optimization overhead exceeds your eval cost
- BOHB's multi-fidelity is irrelevant — your evals are already fast

---

## 9. Key References

### CMA-ES Theory & Implementation
- **Hansen & Ostermeier (2001)** — "Completely Derandomized Self-Adaptation in Evolution Strategies" — foundational CMA-ES paper
- **Hansen (2006)** — "The CMA Evolution Strategy: A Comparing Review" — critical comparison with EDAs/CE, reveals incremental vs non-incremental estimation differences
- **Nomura et al. (2024)** — "cmaes: A Simple yet Practical Python Library for CMA-ES" — CyberAgentAILab library documentation

### Mixed-Variable CMA-ES
- **Hamano et al. (2022)** — CMA-ES with Margin (CMAwM) — margin-corrected discrete variable handling
- **Hamano et al. (2024)** — CatCMA — categorical distribution extension
- **Hamano et al. (2025)** — CatCMAwM — unified continuous + integer + categorical framework

### Competitor Methods
- **Falkner et al. (2018)** — "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" — BOHB/SMAC analysis
- **Eriksson et al. (2019)** — "Scalable Global Optimization via Local Bayesian Optimization" — TuRBO paper with timing benchmarks
- **Optuna GitHub Issue #2660** — Batch optimization limitations documented

### Trading Strategy Optimization
- **Thomaidis et al. (2018)** — MACD/ATR SL/TP interaction study (MDPI) — empirical parameter coupling evidence
- **Nabi et al. (2021)** — "Optimal Technical Indicator-based Trading Strategies Using NSGA-II" — multi-objective trading optimization

### Nevergrad
- **Meta AI** — Nevergrad gradient-free optimization platform documentation
- CMA, Differential Evolution, PSO implementations with batch support

---

## NotebookLM Research Workspace

**Notebook:** Optimizer Engine Upgrade: CMA-ES, TPE, BO for Batch-Parallel Trading Optimization
**Notebook ID:** be2a6709-d576-4350-8747-206f65fe6df3
**URL:** https://notebooklm.google.com/notebook/be2a6709-d576-4350-8747-206f65fe6df3
**Sources:** 21 (2 research briefs + 19 academic/documentation URLs)
**Queries executed:** 4 deep analytical queries with citation tracking
