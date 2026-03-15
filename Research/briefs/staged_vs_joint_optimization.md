# Research Brief: Staged vs Joint Parameter Optimization

## Question
We break our trading strategy optimization into many small sequential stages. Each stage optimizes a subset of parameters, locks the best values, then moves on. A final "refinement" stage unlocks everything with narrow ranges to catch interactions. **Is this approach sound, or does breaking things into small groups cause us to miss critical parameter interactions?**

## Our Current System
- **Stage order**: signal → time → risk → management modules (each module is its own sub-stage: trailing stop, breakeven, partial close, max bars, stale exit) → refinement
- That's potentially 8+ sequential stages, each locking params before the next runs
- Each stage uses Sobol + Cross-Entropy sampling (intelligent, not random grid)
- **Refinement stage** (final): unlocks ALL parameters with narrow ranges (±3 steps) around locked best values — intended to capture cross-group interactions
- Budget per stage: 50K trials (turbo) to 200K (standard)

## The Interaction Concern
Every time we split a group, we assume the parameters within one stage don't interact strongly with parameters in other stages. Examples of potential interactions we might be missing:

- **Signal ↔ Risk**: A momentum signal that catches big moves might need wide SL + tight TP. A mean-reversion signal might need tight SL + wide TP. Locking signal params before seeing risk params could lock a signal that only works with specific risk settings.
- **SL ↔ TP** (proposed split): SL and TP are mathematically linked via risk-reward ratio. Optimizing SL without knowing TP means optimizing half the trade profile.
- **Time filters ↔ Signal**: A signal might only work in Asian session but time filters are locked before risk/management are explored.
- **Trailing stop ↔ TP**: If trailing is aggressive, you don't need a TP. If trailing is off, TP matters a lot. These are in different stages.
- **Breakeven ↔ SL**: BE effectively replaces the original SL, so the "best SL" depends on whether BE is active.

## The Coverage Argument FOR Staging
- Joint optimization of all params would be a massive space (millions+ of combinations)
- With a fixed trial budget, more dimensions = sparser coverage = more likely to miss the optimum
- Staging gives near-complete coverage of each subspace
- Example: risk stage has 1.3M combos, only 3.7% explored with turbo. If we add signal params too, coverage drops to <0.1%

## Key Research Questions

1. **Block coordinate descent in optimization theory**: Our approach is essentially block coordinate descent. When does this converge to the global optimum? When does it get stuck in bad local optima due to missed interactions? What conditions must hold (separability, weak coupling)?

2. **Does the refinement stage actually recover missed interactions?** It only searches ±3 steps around locked values. If the global optimum is far from the staged solution (because staging locked a bad region), refinement can't reach it. Is narrow-range joint refinement sufficient in practice?

3. **The curse of dimensionality vs the curse of decomposition**: Is it better to search sparsely in the full joint space, or densely in sequential subspaces? Is there a mathematical framework or empirical evidence for this trade-off?

4. **Which trading strategy parameters actually interact?** Is there empirical evidence on interaction strength between:
   - Entry signal params vs exit/risk params
   - SL vs TP (beyond the obvious RR link)
   - Management features (trailing, BE) vs base risk (SL/TP)
   - Time filters vs everything else

5. **Alternatives to pure sequential staging**:
   - Grouping interacting params together (e.g., keep SL+TP+trailing in one "trade management" stage)
   - Running interaction detection first (e.g., ANOVA, Sobol sensitivity indices) to decide what to group
   - Iterating: stage1 → stage2 → ... → refinement → back to stage1 with updated context (cyclic coordinate descent)
   - Hybrid: mostly staged but with a % of trials sampling the full joint space

6. **Practical experience**: Do commercial trading platforms or academic papers on strategy optimization use staged/decomposed approaches? What groupings do they use?

## What We Already Know
- Management modules (trailing, BE, partial close) seem fairly independent of each other — sub-staging them works well
- Signal params tend to be the most critical (they determine IF you trade)
- Risk params determine HOW MUCH you make/lose per trade
- Time filters are likely the most independent group
- Our refinement stage is the safety net, but we haven't validated how effective it actually is at recovering missed interactions

## Desired Output
- Is our overall staged approach fundamentally sound, or is it a known anti-pattern?
- Which parameter groups should definitely stay together vs can be safely split?
- Is our refinement stage (±3 steps, all params) a sufficient safety net? Should it be wider?
- Should we consider cyclic refinement (multiple passes) instead of a single final pass?
- Any practical recommendations for our specific system
