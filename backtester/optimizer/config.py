"""Optimization configuration: trial counts, batch sizes, thresholds, presets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OptimizationConfig:
    """Configuration for optimization runs."""

    # --- Trial counts per stage ---
    trials_per_stage: int = 5000
    refinement_trials: int = 10000

    # --- Batch size ---
    batch_size: int = 512

    # --- Sampler settings ---
    exploration_pct: float = 0.4   # First 40% of budget uses Sobol/random
    eda_learning_rate: float = 0.3
    eda_min_prob: float = 0.01
    elite_pct: float = 0.1        # Top 10% used as elites for EDA update

    # --- Post-filter thresholds ---
    min_trades: int = 20
    max_dd_pct: float = 30.0
    min_r_squared: float = 0.5

    # --- Ranking ---
    forward_weight: float = 1.5
    min_forward_back_ratio: float = 0.4
    top_n_candidates: int = 50

    # --- DSR ---
    dsr_threshold: float = 0.95   # DSR must exceed this to be "significant"

    # --- Execution ---
    max_trades_per_trial: int = 5000
    seed: int | None = None


# Speed presets
TURBO = OptimizationConfig(
    trials_per_stage=1000,
    refinement_trials=2000,
    batch_size=256,
    exploration_pct=0.5,
    min_trades=10,
)

FAST = OptimizationConfig(
    trials_per_stage=4000,
    refinement_trials=8000,
    batch_size=512,
)

DEFAULT = OptimizationConfig(
    trials_per_stage=10000,
    refinement_trials=20000,
    batch_size=512,
)

PRESETS: dict[str, OptimizationConfig] = {
    "turbo": TURBO,
    "fast": FAST,
    "default": DEFAULT,
}


def get_preset(name: str) -> OptimizationConfig:
    """Get a preset config by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Choose from: {list(PRESETS.keys())}")
    return PRESETS[name]
