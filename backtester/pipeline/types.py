"""Pipeline result types — dataclasses for all pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Rating(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class StabilityRating(Enum):
    ROBUST = "ROBUST"
    MODERATE = "MODERATE"
    FRAGILE = "FRAGILE"
    OVERFIT = "OVERFIT"


# ---------------------------------------------------------------------------
# Walk-Forward (Stage 3)
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Result of evaluating a candidate on one walk-forward window."""
    window_index: int
    start_bar: int
    end_bar: int
    is_oos: bool         # True = out-of-sample, False = in-sample
    n_trades: int = 0
    sharpe: float = 0.0
    quality_score: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    return_pct: float = 0.0
    passed: bool = False  # True if window passes minimum thresholds


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward result for one candidate."""
    windows: list[WindowResult] = field(default_factory=list)
    n_windows: int = 0
    n_oos_windows: int = 0
    n_passed: int = 0
    pass_rate: float = 0.0
    mean_sharpe: float = 0.0
    mean_quality: float = 0.0
    geo_mean_quality: float = 0.0
    min_quality: float = 0.0
    quality_cv: float = 0.0       # Coefficient of variation
    wfe: float = 0.0              # Walk-forward efficiency
    passed_gate: bool = False


# ---------------------------------------------------------------------------
# CPCV (sub-step of Stage 3)
# ---------------------------------------------------------------------------

@dataclass
class CPCVFoldResult:
    """Result of evaluating one candidate on one CPCV fold."""
    fold_index: int
    train_blocks: tuple[int, ...]       # Block indices used for training
    test_blocks: tuple[int, ...]        # Block indices used for testing
    n_purged: int = 0                   # Bars removed by purging
    n_trades: int = 0
    sharpe: float = 0.0
    quality_score: float = 0.0


@dataclass
class CPCVResult:
    """Aggregate CPCV result for one candidate."""
    n_blocks: int = 0
    k_test: int = 0
    n_folds: int = 0

    mean_sharpe: float = 0.0
    median_sharpe: float = 0.0
    std_sharpe: float = 0.0
    sharpe_ci_low: float = 0.0
    sharpe_ci_high: float = 0.0

    pct_positive_sharpe: float = 0.0     # Fraction of folds with Sharpe > 0
    mean_quality: float = 0.0
    median_quality: float = 0.0

    passed_gate: bool = False
    folds: list[CPCVFoldResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stability (Stage 4)
# ---------------------------------------------------------------------------

@dataclass
class PerturbationResult:
    """Result of perturbing one parameter."""
    param_name: str
    original_value: Any
    perturbed_value: Any
    original_quality: float
    perturbed_quality: float
    ratio: float  # perturbed / original (clamped to 0 if original <= 0)


@dataclass
class StabilityResult:
    """Stability analysis for one candidate."""
    perturbations: list[PerturbationResult] = field(default_factory=list)
    mean_ratio: float = 0.0
    min_ratio: float = 0.0
    worst_param: str = ""
    rating: StabilityRating = StabilityRating.OVERFIT


# ---------------------------------------------------------------------------
# Monte Carlo (Stage 5)
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """Monte Carlo analysis for one candidate."""
    # Block bootstrap
    bootstrap_sharpe_mean: float = 0.0
    bootstrap_sharpe_std: float = 0.0
    bootstrap_sharpe_ci_low: float = 0.0
    bootstrap_sharpe_ci_high: float = 0.0

    # Permutation test (sign-flip)
    permutation_p_value: float = 1.0
    observed_sharpe: float = 0.0

    # Trade skip
    skip_results: dict[str, float] = field(default_factory=dict)
    # e.g. {"5%": quality_after_skip, "10%": quality_after_skip}

    # Execution stress
    stress_quality: float = 0.0
    stress_quality_ratio: float = 0.0  # stress / original

    # DSR
    dsr: float = 0.0

    passed_gate: bool = False


# ---------------------------------------------------------------------------
# Confidence (Stage 6)
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceResult:
    """Final confidence scoring for one candidate."""
    # Sub-scores (0-100 each)
    walk_forward_score: float = 0.0
    cpcv_score: float = 0.0
    monte_carlo_score: float = 0.0
    forward_back_score: float = 0.0
    stability_score: float = 0.0
    dsr_score: float = 0.0
    backtest_quality_score: float = 0.0

    # Composite
    composite_score: float = 0.0
    rating: Rating = Rating.RED

    # Gate results
    gates_passed: dict[str, bool] = field(default_factory=dict)
    all_gates_passed: bool = False


# ---------------------------------------------------------------------------
# Full Candidate Result (across all stages)
# ---------------------------------------------------------------------------

@dataclass
class CandidateResult:
    """Full pipeline result for one candidate."""
    candidate_index: int
    params: dict[str, Any] = field(default_factory=dict)

    # Stage 2 (from optimizer)
    back_quality: float = 0.0
    forward_quality: float = 0.0
    forward_back_ratio: float = 0.0
    back_sharpe: float = 0.0
    back_trades: int = 0
    n_trials: int = 0  # Total trials explored during optimization

    # Stage 3
    walk_forward: WalkForwardResult | None = None

    # Stage 3b (CPCV sub-step)
    cpcv: CPCVResult | None = None

    # Stage 4
    stability: StabilityResult | None = None

    # Stage 5
    monte_carlo: MonteCarloResult | None = None

    # Stage 6
    confidence: ConfidenceResult | None = None

    # Elimination tracking
    eliminated: bool = False
    eliminated_at_stage: str = ""
    elimination_reason: str = ""


# ---------------------------------------------------------------------------
# Pipeline State (for checkpoint/resume)
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """Full pipeline state for checkpoint/resume."""
    strategy_name: str = ""
    strategy_version: str = ""
    pair: str = ""
    timeframe: str = ""

    completed_stages: list[int] = field(default_factory=list)
    current_stage: int = 0

    candidates: list[CandidateResult] = field(default_factory=list)

    # Config snapshot (for reproducibility)
    config_dict: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
