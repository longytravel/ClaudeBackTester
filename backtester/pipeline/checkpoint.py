"""Checkpoint save/load for pipeline state.

Saves PipelineState to JSON. Supports resume from any completed stage.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any

from backtester.pipeline.types import (
    CPCVFoldResult,
    CPCVResult,
    CandidateResult,
    ConfidenceResult,
    MonteCarloResult,
    PipelineState,
    Rating,
    StabilityRating,
    StabilityResult,
    WalkForwardResult,
    WindowResult,
    PerturbationResult,
)

logger = logging.getLogger(__name__)


def save_checkpoint(state: PipelineState, filepath: str) -> None:
    """Save pipeline state to JSON file.

    Uses atomic write (write to temp, then rename) for crash safety.
    """
    data = asdict(state)
    # Convert enums to their string values for JSON
    _convert_enums(data)

    tmp_path = filepath + ".tmp"
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    # Atomic rename (on Windows, need to remove target first)
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(tmp_path, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str) -> PipelineState:
    """Load pipeline state from JSON file."""
    with open(filepath) as f:
        data = json.load(f)

    state = PipelineState(
        strategy_name=data.get("strategy_name", ""),
        strategy_version=data.get("strategy_version", ""),
        pair=data.get("pair", ""),
        timeframe=data.get("timeframe", ""),
        completed_stages=data.get("completed_stages", []),
        current_stage=data.get("current_stage", 0),
        config_dict=data.get("config_dict", {}),
        seed=data.get("seed", 42),
    )

    # Reconstruct candidates
    for cd in data.get("candidates", []):
        candidate = _reconstruct_candidate(cd)
        state.candidates.append(candidate)

    logger.info(
        f"Checkpoint loaded from {filepath}: "
        f"stage={state.current_stage}, "
        f"{len(state.candidates)} candidates"
    )
    return state


def _convert_enums(obj: Any) -> None:
    """Recursively convert enum values to strings in a dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if hasattr(v, "value") and hasattr(v, "name"):
                obj[k] = v.value
            elif isinstance(v, (dict, list)):
                _convert_enums(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if hasattr(v, "value") and hasattr(v, "name"):
                obj[i] = v.value
            elif isinstance(v, (dict, list)):
                _convert_enums(v)


def _reconstruct_candidate(cd: dict) -> CandidateResult:
    """Reconstruct a CandidateResult from a dict."""
    c = CandidateResult(
        candidate_index=cd.get("candidate_index", 0),
        params=cd.get("params", {}),
        back_quality=cd.get("back_quality", 0.0),
        forward_quality=cd.get("forward_quality", 0.0),
        forward_back_ratio=cd.get("forward_back_ratio", 0.0),
        back_sharpe=cd.get("back_sharpe", 0.0),
        back_trades=cd.get("back_trades", 0),
        n_trials=cd.get("n_trials", 0),
        eliminated=cd.get("eliminated", False),
        eliminated_at_stage=cd.get("eliminated_at_stage", ""),
        elimination_reason=cd.get("elimination_reason", ""),
    )

    # Walk-forward
    wf_data = cd.get("walk_forward")
    if wf_data is not None:
        windows = []
        for wd in wf_data.get("windows", []):
            windows.append(WindowResult(**{
                k: wd[k] for k in WindowResult.__dataclass_fields__
                if k in wd
            }))
        c.walk_forward = WalkForwardResult(
            windows=windows,
            n_windows=wf_data.get("n_windows", 0),
            n_oos_windows=wf_data.get("n_oos_windows", 0),
            n_passed=wf_data.get("n_passed", 0),
            pass_rate=wf_data.get("pass_rate", 0.0),
            mean_sharpe=wf_data.get("mean_sharpe", 0.0),
            mean_quality=wf_data.get("mean_quality", 0.0),
            geo_mean_quality=wf_data.get("geo_mean_quality", 0.0),
            min_quality=wf_data.get("min_quality", 0.0),
            quality_cv=wf_data.get("quality_cv", 0.0),
            wfe=wf_data.get("wfe", 0.0),
            passed_gate=wf_data.get("passed_gate", False),
        )

    # Stability
    stab_data = cd.get("stability")
    if stab_data is not None:
        perturbations = []
        for pd in stab_data.get("perturbations", []):
            perturbations.append(PerturbationResult(**{
                k: pd[k] for k in PerturbationResult.__dataclass_fields__
                if k in pd
            }))
        rating_str = stab_data.get("rating", "OVERFIT")
        c.stability = StabilityResult(
            perturbations=perturbations,
            mean_ratio=stab_data.get("mean_ratio", 0.0),
            min_ratio=stab_data.get("min_ratio", 0.0),
            worst_param=stab_data.get("worst_param", ""),
            rating=StabilityRating(rating_str),
        )

    # Monte Carlo
    mc_data = cd.get("monte_carlo")
    if mc_data is not None:
        c.monte_carlo = MonteCarloResult(
            bootstrap_sharpe_mean=mc_data.get("bootstrap_sharpe_mean", 0.0),
            bootstrap_sharpe_std=mc_data.get("bootstrap_sharpe_std", 0.0),
            bootstrap_sharpe_ci_low=mc_data.get("bootstrap_sharpe_ci_low", 0.0),
            bootstrap_sharpe_ci_high=mc_data.get("bootstrap_sharpe_ci_high", 0.0),
            permutation_p_value=mc_data.get("permutation_p_value", 1.0),
            observed_sharpe=mc_data.get("observed_sharpe", 0.0),
            skip_results=mc_data.get("skip_results", {}),
            stress_quality=mc_data.get("stress_quality", 0.0),
            stress_quality_ratio=mc_data.get("stress_quality_ratio", 0.0),
            dsr=mc_data.get("dsr", 0.0),
            passed_gate=mc_data.get("passed_gate", False),
        )

    # CPCV
    cpcv_data = cd.get("cpcv")
    if cpcv_data is not None:
        folds = []
        for fd in cpcv_data.get("folds", []):
            folds.append(CPCVFoldResult(
                fold_index=fd.get("fold_index", 0),
                train_blocks=tuple(fd.get("train_blocks", ())),
                test_blocks=tuple(fd.get("test_blocks", ())),
                n_purged=fd.get("n_purged", 0),
                n_trades=fd.get("n_trades", 0),
                sharpe=fd.get("sharpe", 0.0),
                quality_score=fd.get("quality_score", 0.0),
            ))
        c.cpcv = CPCVResult(
            n_blocks=cpcv_data.get("n_blocks", 0),
            k_test=cpcv_data.get("k_test", 0),
            n_folds=cpcv_data.get("n_folds", 0),
            mean_sharpe=cpcv_data.get("mean_sharpe", 0.0),
            median_sharpe=cpcv_data.get("median_sharpe", 0.0),
            std_sharpe=cpcv_data.get("std_sharpe", 0.0),
            sharpe_ci_low=cpcv_data.get("sharpe_ci_low", 0.0),
            sharpe_ci_high=cpcv_data.get("sharpe_ci_high", 0.0),
            pct_positive_sharpe=cpcv_data.get("pct_positive_sharpe", 0.0),
            mean_quality=cpcv_data.get("mean_quality", 0.0),
            median_quality=cpcv_data.get("median_quality", 0.0),
            passed_gate=cpcv_data.get("passed_gate", False),
            folds=folds,
        )

    # Regime
    regime_data = cd.get("regime")
    if regime_data is not None:
        c.regime = _reconstruct_regime(regime_data)

    # Trade stats (plain dict, no reconstruction needed)
    c.trade_stats = cd.get("trade_stats")

    # Confidence
    conf_data = cd.get("confidence")
    if conf_data is not None:
        rating_str = conf_data.get("rating", "RED")
        c.confidence = ConfidenceResult(
            walk_forward_score=conf_data.get("walk_forward_score", 0.0),
            cpcv_score=conf_data.get("cpcv_score", 0.0),
            monte_carlo_score=conf_data.get("monte_carlo_score", 0.0),
            forward_back_score=conf_data.get("forward_back_score", 0.0),
            stability_score=conf_data.get("stability_score", 0.0),
            dsr_score=conf_data.get("dsr_score", 0.0),
            backtest_quality_score=conf_data.get("backtest_quality_score", 0.0),
            composite_score=conf_data.get("composite_score", 0.0),
            rating=Rating(rating_str),
            gates_passed=conf_data.get("gates_passed", {}),
            all_gates_passed=conf_data.get("all_gates_passed", False),
        )

    return c


def _reconstruct_regime(data: dict) -> Any:
    """Reconstruct a RegimeResult from a dict."""
    from backtester.pipeline.regime import RegimeResult, RegimeStats

    per_regime = []
    for rs in data.get("per_regime", []):
        per_regime.append(RegimeStats(
            regime=rs.get("regime", 0),
            regime_name=rs.get("regime_name", ""),
            n_bars=rs.get("n_bars", 0),
            bar_pct=rs.get("bar_pct", 0.0),
            n_trades=rs.get("n_trades", 0),
            sharpe=rs.get("sharpe", 0.0),
            profit_factor=rs.get("profit_factor", 0.0),
            max_dd_pct=rs.get("max_dd_pct", 0.0),
            win_rate=rs.get("win_rate", 0.0),
            mean_pnl_pips=rs.get("mean_pnl_pips", 0.0),
            sufficient_data=rs.get("sufficient_data", False),
        ))

    return RegimeResult(
        regime_distribution=data.get("regime_distribution", {}),
        per_regime=per_regime,
        regime_weighted_sharpe=data.get("regime_weighted_sharpe", 0.0),
        worst_regime_max_dd=data.get("worst_regime_max_dd", 0.0),
        n_profitable_regimes=data.get("n_profitable_regimes", 0),
        n_scored_regimes=data.get("n_scored_regimes", 0),
        advisory=data.get("advisory", ""),
        robustness_score=data.get("robustness_score", 0.0),
    )
