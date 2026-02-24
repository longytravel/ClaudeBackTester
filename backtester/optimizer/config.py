"""Optimization configuration: trial counts, batch sizes, thresholds, presets."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OptimizationConfig:
    """Configuration for optimization runs."""

    # --- Trial counts per stage ---
    trials_per_stage: int = 200_000
    refinement_trials: int = 400_000

    # --- Batch size ---
    batch_size: int = 4096

    # --- Sampler settings ---
    exploration_pct: float = 0.4   # First 40% of budget uses Sobol/random
    eda_learning_rate: float = 0.3
    eda_lr_decay: float = 0.95     # LR decay per update (effective_lr decays toward floor)
    eda_lr_floor: float = 0.05     # Minimum LR after decay
    eda_min_prob: float = 0.01
    elite_pct: float = 0.1        # Top 10% used as elites for EDA update

    # --- Post-filter thresholds ---
    min_trades: int = 20
    max_dd_pct: float = 30.0
    min_r_squared: float = 0.5

    # --- Ranking ---
    forward_weight: float = 1.5
    min_forward_back_ratio: float = 0.4
    top_n_candidates: int = 10

    # --- DSR ---
    dsr_threshold: float = 0.95   # DSR must exceed this to be "significant"

    # --- Execution ---
    max_trades_per_trial: int = 50000
    seed: int | None = None


# Presets scaled for i9-14900HX (24 cores, 64GB RAM)
# Batch sizes tuned for cache efficiency on 24-core systems

TURBO = OptimizationConfig(
    trials_per_stage=50_000,
    refinement_trials=100_000,
    batch_size=2048,
    exploration_pct=0.35,
)

STANDARD = OptimizationConfig(
    trials_per_stage=200_000,
    refinement_trials=400_000,
    batch_size=4096,
    exploration_pct=0.30,
)

DEEP = OptimizationConfig(
    trials_per_stage=500_000,
    refinement_trials=1_000_000,
    batch_size=4096,
    exploration_pct=0.25,
)

MAX = OptimizationConfig(
    trials_per_stage=1_000_000,
    refinement_trials=2_000_000,
    batch_size=8192,
    exploration_pct=0.20,
)

PRESETS: dict[str, OptimizationConfig] = {
    "turbo": TURBO,
    "standard": STANDARD,
    "deep": DEEP,
    "max": MAX,
}


def get_preset(name: str) -> OptimizationConfig:
    """Get a preset config by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Choose from: {list(PRESETS.keys())}")
    return PRESETS[name]
