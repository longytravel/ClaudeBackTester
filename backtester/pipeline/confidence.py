"""Confidence scoring — sequential gates + weighted composite score.

Applies hard gates first (eliminate candidates), then computes a
weighted composite score (0-100) for surviving candidates.
"""

from __future__ import annotations

import math
from typing import Any

from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.types import (
    CandidateResult,
    ConfidenceResult,
    MonteCarloResult,
    Rating,
    StabilityResult,
    WalkForwardResult,
)


# ---------------------------------------------------------------------------
# Hard Gates
# ---------------------------------------------------------------------------

def apply_gates(candidate: CandidateResult, config: PipelineConfig) -> dict[str, bool]:
    """Apply sequential hard gates. Returns dict of gate_name -> passed."""
    gates: dict[str, bool] = {}

    # Gate 1: Forward/back quality ratio (from Stage 2)
    gates["forward_back_ratio"] = candidate.forward_back_ratio >= 0.4

    # Gate 2: Walk-forward pass rate (from Stage 3)
    wf = candidate.walk_forward
    if wf is not None:
        gates["wf_pass_rate"] = wf.pass_rate >= config.wf_pass_rate_gate
        gates["wf_mean_sharpe"] = wf.mean_sharpe >= config.wf_mean_sharpe_gate
    else:
        gates["wf_pass_rate"] = False
        gates["wf_mean_sharpe"] = False

    # Gate 3: DSR + permutation (from Stage 5)
    mc = candidate.monte_carlo
    if mc is not None:
        gates["dsr"] = mc.dsr >= config.mc_dsr_gate
        gates["permutation_p"] = mc.permutation_p_value <= config.mc_permutation_p_gate
    else:
        gates["dsr"] = False
        gates["permutation_p"] = False

    return gates


# ---------------------------------------------------------------------------
# Sub-Scorers (each returns 0-100)
# ---------------------------------------------------------------------------

def score_walk_forward(wf: WalkForwardResult | None) -> float:
    """Score walk-forward consistency (0-100).

    Components:
    - OOS pass rate (40% weight)
    - Geo mean quality (20% weight, scaled to 0-100)
    - Min quality (15% weight, scaled)
    - Low CV (15% weight, inverse — lower CV = better)
    - Window count bonus (10% weight)
    """
    if wf is None or wf.n_oos_windows == 0:
        return 0.0

    # Pass rate: 0-100 directly
    pass_score = wf.pass_rate * 100.0

    # Geo mean quality: cap at 50 → 100
    geo_score = min(wf.geo_mean_quality / 50.0, 1.0) * 100.0

    # Min quality: cap at 20 → 100
    min_score = min(max(wf.min_quality, 0.0) / 20.0, 1.0) * 100.0

    # CV: lower is better. CV=0 → 100, CV>=1 → 0
    cv_score = max(0.0, (1.0 - wf.quality_cv)) * 100.0

    # Window count: 4+ windows → 100
    window_score = min(wf.n_oos_windows / 4.0, 1.0) * 100.0

    return (
        0.40 * pass_score
        + 0.20 * geo_score
        + 0.15 * min_score
        + 0.15 * cv_score
        + 0.10 * window_score
    )


def score_monte_carlo(mc: MonteCarloResult | None) -> float:
    """Score Monte Carlo robustness (0-100).

    Components:
    - Bootstrap Sharpe CI (30% — narrow CI above 0)
    - Permutation p-value (30% — lower is better)
    - Trade-skip resilience (20% — skip quality / original quality)
    - Stress test resilience (20%)
    """
    if mc is None:
        return 0.0

    # Bootstrap CI: score based on lower bound being positive
    if mc.bootstrap_sharpe_ci_low > 0:
        # CI lower bound > 0: score based on how far above 0
        ci_score = min(mc.bootstrap_sharpe_ci_low / 1.0, 1.0) * 100.0
    else:
        # CI includes 0: partial score based on how close to positive
        ci_range = mc.bootstrap_sharpe_ci_high - mc.bootstrap_sharpe_ci_low
        if ci_range > 0:
            ci_score = max(0, mc.bootstrap_sharpe_ci_high / ci_range) * 50.0
        else:
            ci_score = 0.0

    # Permutation: p < 0.01 → 100, p=0.05 → 80, p=0.5 → 0
    if mc.permutation_p_value <= 0.001:
        perm_score = 100.0
    elif mc.permutation_p_value <= 0.05:
        # Linear scale from 80-100 for p in [0.001, 0.05]
        perm_score = 80.0 + 20.0 * (0.05 - mc.permutation_p_value) / 0.049
    elif mc.permutation_p_value <= 0.5:
        # Linear scale from 0-80 for p in [0.05, 0.5]
        perm_score = 80.0 * (0.5 - mc.permutation_p_value) / 0.45
    else:
        perm_score = 0.0

    # Trade skip: average resilience across skip levels
    if mc.skip_results:
        skip_ratios = list(mc.skip_results.values())
        # These are quality scores; compare to a reasonable baseline
        skip_score = min(sum(skip_ratios) / len(skip_ratios) / 30.0, 1.0) * 100.0
    else:
        skip_score = 50.0  # Neutral if not tested

    # Stress: quality ratio >= 0.8 → 100, 0 → 0
    stress_score = min(max(mc.stress_quality_ratio, 0.0) / 0.8, 1.0) * 100.0

    return (
        0.30 * ci_score
        + 0.30 * perm_score
        + 0.20 * skip_score
        + 0.20 * stress_score
    )


def score_forward_back(ratio: float) -> float:
    """Score forward/back quality ratio (0-100).

    ratio >= 1.0 → 100 (forward as good or better)
    ratio = 0.4 → 0 (at the gate threshold)
    """
    if ratio <= 0.4:
        return 0.0
    if ratio >= 1.0:
        return 100.0
    # Linear scale from 0.4 → 0 to 1.0 → 100
    return (ratio - 0.4) / 0.6 * 100.0


def score_stability(stab: StabilityResult | None) -> float:
    """Score parameter stability (0-100).

    Based on mean ratio and min ratio.
    """
    if stab is None:
        return 0.0

    # Mean ratio: 1.0 → 100, 0.4 → 0
    mean_score = max(0.0, min((stab.mean_ratio - 0.4) / 0.6, 1.0)) * 100.0

    # Min ratio: 0.5 → 50+, 0 → 0
    min_score = max(0.0, min(stab.min_ratio / 0.5, 1.0)) * 100.0

    return 0.7 * mean_score + 0.3 * min_score


def score_dsr(dsr_value: float) -> float:
    """Score DSR value (0-100).

    DSR >= 0.99 → 100
    DSR = 0.95 → 80  (at gate threshold)
    DSR = 0.5  → 0
    """
    if dsr_value <= 0.5:
        return 0.0
    if dsr_value >= 0.99:
        return 100.0
    if dsr_value >= 0.95:
        # 80-100 for 0.95-0.99
        return 80.0 + 20.0 * (dsr_value - 0.95) / 0.04
    # 0-80 for 0.5-0.95
    return 80.0 * (dsr_value - 0.5) / 0.45


def score_backtest_quality(candidate: CandidateResult) -> float:
    """Score raw backtest quality (0-100).

    Based on quality score, trade count, and Sharpe.
    """
    # Quality score: cap at 50 → 100
    qs = min(candidate.back_quality / 50.0, 1.0) * 100.0

    # Trade count: 100+ → full score
    trade_score = min(candidate.back_trades / 100.0, 1.0) * 100.0

    # Sharpe: cap at 2.0 → 100
    sharpe_score = min(max(candidate.back_sharpe, 0.0) / 2.0, 1.0) * 100.0

    return 0.50 * qs + 0.25 * trade_score + 0.25 * sharpe_score


# ---------------------------------------------------------------------------
# Composite Score + Rating
# ---------------------------------------------------------------------------

def compute_confidence(
    candidate: CandidateResult,
    config: PipelineConfig,
) -> ConfidenceResult:
    """Compute full confidence result: gates + composite score + rating."""
    result = ConfidenceResult()

    # Step 1: Apply gates
    result.gates_passed = apply_gates(candidate, config)
    result.all_gates_passed = all(result.gates_passed.values())

    # Step 2: Compute sub-scores
    result.walk_forward_score = score_walk_forward(candidate.walk_forward)
    result.monte_carlo_score = score_monte_carlo(candidate.monte_carlo)
    result.forward_back_score = score_forward_back(candidate.forward_back_ratio)
    result.stability_score = score_stability(candidate.stability)

    dsr_val = candidate.monte_carlo.dsr if candidate.monte_carlo else 0.0
    result.dsr_score = score_dsr(dsr_val)
    result.backtest_quality_score = score_backtest_quality(candidate)

    # Step 3: Weighted composite
    result.composite_score = (
        config.conf_weight_walk_forward * result.walk_forward_score
        + config.conf_weight_monte_carlo * result.monte_carlo_score
        + config.conf_weight_forward_back * result.forward_back_score
        + config.conf_weight_stability * result.stability_score
        + config.conf_weight_dsr * result.dsr_score
        + config.conf_weight_backtest * result.backtest_quality_score
    )

    # Step 4: Assign rating
    result.rating = assign_rating(
        result.composite_score,
        result.all_gates_passed,
        config,
    )

    return result


def assign_rating(
    composite_score: float,
    all_gates_passed: bool,
    config: PipelineConfig,
) -> Rating:
    """Assign RED/YELLOW/GREEN rating.

    RED if any gate failed, regardless of composite score.
    """
    if not all_gates_passed:
        return Rating.RED
    if composite_score >= config.conf_green_threshold:
        return Rating.GREEN
    if composite_score >= config.conf_yellow_threshold:
        return Rating.YELLOW
    return Rating.RED
