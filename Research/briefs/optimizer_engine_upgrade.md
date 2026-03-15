# Research Brief: Optimizer Engine Upgrade for Larger Joint Parameter Groups

## Background
We built a staged optimizer using Sobol + Cross-Entropy (CE) sampling. It was designed around small parameter groups optimized sequentially. Research has now shown this sequential locking approach misses critical parameter interactions.

We're moving to fewer, larger stages:
- **Core Trade Profile**: SL + TP + Trailing Stop + Breakeven (merged from 4+ separate stages)
- **Signal**: all entry signal params
- **Auxiliary Management**: Partial Close, Max Bars, Stale Exit
- **Time Filters**: session/day filters

The merged "Core Trade Profile" stage will have significantly more parameters and combinations than any single stage had before. As we add more management modules and strategy complexity over time, these groups will keep growing.

## Our Current Optimizer
- **Sobol quasi-random exploration** (low-discrepancy sequences for uniform coverage)
- **Cross-Entropy (CE) exploitation** (fit Gaussian to top performers, sample from it, adaptive learning rate, entropy monitoring)
- **Budget**: 50K-200K trials per stage depending on preset
- **Evaluation speed**: 18K-35K evals/sec (BASIC), 4K-10K evals/sec (FULL) via Rust hot loop
- **Batch-first**: optimizer generates N param sets → Rust evaluates all N in parallel → optimizer updates once per batch
- CE works well for small groups (5-10 params) but we haven't tested it on larger joint spaces (15-20+ params)

## The Problem
With merged stages, the parameter space per stage is much larger:
- Old risk stage: ~1.3M combos (SL mode + value + TP mode + value + RR ratio = ~5 params)
- New Core Trade Profile: SL + TP + trailing trigger/distance/step + BE trigger/offset = potentially 10-15 params, millions to billions of combos
- At 200K trials in a 15-dimensional space, Sobol+CE coverage may be too sparse
- CE assumes unimodal Gaussian — may miss multiple good regions in a larger space

## Key Research Questions

### 1. Sobol+CE Scaling
- How does Cross-Entropy method scale with dimensionality? At what point does it break down?
- Is Sobol+CE sufficient for 10-15 parameter joint optimization, or do we need a fundamentally different approach?
- Does CE's Gaussian assumption become problematic in higher dimensions (multimodal landscapes)?

### 2. Optuna / TPE (Tree-structured Parzen Estimator)
- Optuna uses TPE which builds separate kernel density estimates for good/bad trials
- TPE handles conditional parameters naturally (e.g., trailing params only matter if trailing is enabled)
- But TPE is sequential by nature — each trial informs the next. Our Rust hot loop is batch-parallel (evaluate thousands simultaneously)
- **Critical question**: Can Optuna/TPE work in batch mode? What's the overhead per suggestion vs our sub-millisecond eval time?
- Is the smarter sampling worth the loss of batch parallelism?
- Optuna has a CMA-ES sampler — how does that compare?

### 3. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
- Designed specifically for non-convex, non-separable optimization
- Learns the covariance structure (parameter interactions) automatically
- Works well in 5-50 dimensions
- Can it work in batch mode? (generate N candidates, evaluate in parallel, update)
- How does it compare to CE in terms of convergence speed and quality?
- Does it handle mixed parameter types (continuous, discrete, categorical)?

### 4. Genetic Algorithms / Evolutionary Strategies
- The research report recommended 15-20% of budget for GA-based joint exploration
- GAs naturally handle large mixed-type parameter spaces
- NSGA-II for multi-objective (Sharpe + drawdown + trade count)?
- How do modern GAs compare to CMA-ES and TPE for our problem size?

### 5. Bayesian Optimization
- Classical BO (Gaussian Process surrogate) works well up to ~20 dimensions
- TuRBO (trust region BO) extends this to higher dimensions
- RDUCB (random decomposition) was recommended in prior research
- Can any of these work with our batch evaluation paradigm?
- Are there BO methods designed for batch parallel evaluation?

### 6. Hybrid Approaches
- Can we keep Sobol+CE for the exploration phase and switch to a smarter method for exploitation?
- Is there a way to use our fast Rust evaluator for broad coverage AND a smarter optimizer for refinement?
- What about: Sobol exploration (50% budget) → CMA-ES or TPE refinement (50% budget)?
- Can we run a cheap surrogate model to pre-screen candidates before expensive Rust evaluation?

### 7. The Batch Parallelism Constraint
- Our key advantage is the Rust hot loop: 4K-35K evals/sec, batch evaluation
- Any optimizer that requires sequential trial-by-trial suggestions loses this advantage
- **What optimizers support batch/population-based evaluation?**
  - CMA-ES: yes (population-based)
  - GA/ES: yes (population-based)
  - TPE/Optuna: partially (ask for N suggestions, but quality degrades vs sequential)
  - Bayesian Optimization: batch acquisition functions exist but are expensive to compute
- What's the break-even point? At what suggestion overhead does batch-parallel Sobol+CE beat smarter-but-slower methods?

### 8. Conditional Parameters
- Many of our params are conditional: trailing_distance only matters if trailing_mode != OFF
- BE_offset only matters if breakeven is enabled
- In the merged Core Trade Profile, there are many such conditional relationships
- How do different optimizers handle conditional/hierarchical parameter spaces?
- TPE handles this natively. CMA-ES does not. GAs can with encoding tricks.

### 9. Practical Recommendations from Literature
- What do quantitative finance firms actually use for strategy parameter optimization?
- What does the AutoML community recommend for mixed-type, conditional, batch-parallel optimization in 10-30 dimensions?
- Are there any purpose-built tools for this exact problem profile?

## Our Constraints
- Must maintain batch-parallel evaluation (Rust hot loop is our moat)
- Budget is 50K-200K trials per stage (wall-clock time matters — we run on i9-14900HX, 24 cores)
- Parameters are mixed type: continuous (SL value), discrete (SL mode: fixed/ATR/percent), boolean (trailing on/off), categorical
- Some params are conditional on others
- We need this to work reliably without extensive per-strategy tuning of the optimizer itself
- Python implementation is fine for the optimizer — only the evaluation loop needs to be fast

## Desired Output
- Should we upgrade from Sobol+CE, and if so, to what?
- Can the upgrade maintain our batch-parallel evaluation advantage?
- Is there a recommended migration path? (e.g., keep Sobol+CE for now, add CMA-ES as option, migrate over time)
- What's the minimum viable change to handle the merged Core Trade Profile stage effectively?
- Is Optuna the right framework, or should we look at something else (DEAP, Nevergrad, custom CMA-ES)?
