# Codex Research Response: Candidate Selection Pipeline Redesign

**Model**: GPT-5.4 | **Date**: 2026-03-15 | **Mode**: read-only sandbox

---

## Recommendation

Replace the archive as the pipeline selector, but not with a pure "forward-test 58K, take top 20, done" rule.

The right redesign is:

1. Batch forward-test all refinement passers.
2. Use forward as a cheap screening signal, not as final proof.
3. Deduplicate near-identical strategies after forward testing.
4. Send a wider post-dedup set into walk-forward/CPCV.
5. Keep an untouched final acceptance step outside this loop, or treat live paper trading as the final holdout.

That is better than the current flow, where a 2D archive with a practical ceiling of 12 cells selects candidates before forward testing in `archive.py` and `run.py`. It also fixes the current failure mode where candidates pre-eliminated on the forward gate never reach walk-forward in `full_run.py` and `runner.py`.

## Answers

### 1. Is forward-testing all 58K statistically sound?

Forward-testing all 58K is operationally sound and statistically acceptable as a screening step. It is not statistically clean as final confirmation, because once you rank or filter on the forward set, that set is no longer a pristine OOS test.

### 2. Is our DSR handling sufficient?

Your current DSR handling is not enough to justify "test all then select on forward." The implementation in `ranking.py` is a simplified proxy, not the full Bailey/López de Prado treatment of selection bias across a family of trials. If forward results affect selection, per-candidate backtest DSR does not fully correct that second-stage snooping.

### 3. What corrections should we use?

If you want a formal family-level correction for the forward-selection step, White's Reality Check or Hansen's SPA are better fits than per-candidate DSR. They test whether the best model in a searched family beats the null after data snooping. BH/FDR-style procedures are possible, but dependence between nearby parameter sets makes them less clean here.

### 4. Is the forward/back ratio gate (≥0.4) the right filter?

The current `forward_back_ratio >= 0.4` on raw `quality_score` is the weakest part of the design. In `metrics.py`, `quality_score()` includes non-horizon-normalized components like `trades_factor` and `return_pct`. Comparing raw quality on an 80% back period versus a 20% forward period is structurally biased downward. A fixed 0.4 ratio is therefore not a statistically stable generalization test.

**Recommended replacement**: Use absolute forward floors plus a softer relative degradation term:
- `forward_sharpe > 0`
- `forward_quality > 0`
- minimum forward trades consistent with forward window length
- optional ratio only on a length-normalized metric, or as a ranking penalty rather than a hard kill switch

### 5. Walk-forward vs forward/back: redundancy or complementary?

Walk-forward/CPCV and single forward holdout are complementary. The holdout asks "did it survive one unseen regime?" Walk-forward/CPCV ask "is performance consistent across time splits?" Do not run walk-forward on all 58K, but do not let one holdout gate suppress it entirely either.

### 6. Should diversity selection happen on parameters or metrics?

Use diversity after forward testing, not before. Metric-grid MAP-Elites is the wrong tool for pipeline intake. If you want diversity, dedup on behavior first:
- **Best option**: trade fingerprint / signal fingerprint
- **Second best**: signal-parameter subset only
- **Fallback for mixed types**: Gower distance or normalized L1 + Hamming

### 7. How many candidates should enter pipeline validation?

Pure top-20 ranking is too narrow once you remove the archive bottleneck. A better rule is `min(50, max(20, ceil(sqrt(n_dedup_passers))))`. If fewer than 20 survive, test all of them. Given your runtime, 30-50 post-dedup candidates is a reasonable default.

### 8. Rescue lane

Keep a small rescue lane: 5-10 cluster representatives with weak forward scores but strong backtest robustness. That prevents one regime-specific forward period from zeroing the entire walk-forward stage.

### 9. Memory constraints

Memory does not block "test all." It just means "test all in chunks," not one monolithic call. Your existing `batch_size` and `max_trades_per_trial` settings in `config.py` already imply the correct approach.

## Concrete Design

Use this funnel:

```
Refinement passers (~58K)
  → Batch forward-test all (chunked by batch_size)
  → Apply absolute forward floors (Sharpe > 0, quality > 0, min trades)
  → Dedup by behavior/signal parameters
  → Rank by forward-first, back-second (combined_rank)
  → Send 30-50 into walk-forward + CPCV
  → Monte Carlo / stability / confidence after that
  → Final untouched acceptance set or paper-trade before production
```

If you keep `combined_rank`, make forward the primary term. Do not keep the current raw-quality ratio as the main hard gate.

## Industry / Literature

Public evidence points to layered validation, hidden OOS, selection-bias correction, and correlation-aware approval. No public documentation from Two Sigma or D. E. Shaw exposes their exact post-optimization candidate-selection algorithm; public sources are high-level.

### References

- Bailey & López de Prado, "The Deflated Sharpe Ratio": https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- Bailey et al., "The Probability of Backtest Overfitting": https://www.risk.net/journal-of-risk/2461206/the-probability-of-backtest-overfitting
- White, "A Reality Check for Data Snooping": https://www.econometricsociety.org/publications/econometrica/2000/09/01/reality-check-data-snooping
- Hansen, "A Test for Superior Predictive Ability": https://www.tandfonline.com/doi/abs/10.1198/073500105000000063
- Harvey, Liu, Zhu, "… and the Cross-Section of Expected Returns": https://academic.oup.com/rfs/article/29/1/5/1843824
- Harvey & Liu, "False (and Missed) Discoveries in Financial Economics": https://people.duke.edu/~charvey/Research/Published_Papers/P165_False_and_missed.pdf
- Bailey & Prado, "The Sharpe Ratio Efficient Frontier / Strategy Approval Decision": https://www.davidhbailey.com/dhbpapers/stratapproval.pdf
- AQR on broad OOS evidence: https://www.aqr.com/Insights/Research/Journal-Article/How-Do-Factor-Premia-Vary-Over-Time-A-Century-of-Evidence
- Two Sigma on scientific process / negative results: https://www.twosigma.com/articles/why-failure-is-an-option-in-science/
- D. E. Shaw public description of research process: https://www.campus.deshaw.com/who-we-are
