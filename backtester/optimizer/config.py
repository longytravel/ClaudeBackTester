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
    min_trades_per_year: float = 30.0   # Minimum annual trade frequency (timeframe-agnostic)
    min_total_trades: int = 200          # Absolute floor — ensures statistical significance
    max_dd_pct: float = 30.0
    min_r_squared: float = 0.5

    # --- Ranking (legacy, kept for backward compat) ---
    forward_weight: float = 1.5
    min_forward_back_ratio: float = 0.4  # No longer used as hard gate (soft score only)
    top_n_candidates: int = 25  # Deprecated: use max_pipeline_candidates
    top_n_candidates_pct: float | None = None  # Deprecated

    # --- DSR ---
    dsr_threshold: float = 0.95   # DSR must exceed this to be "significant"

    # --- Candidate selection (post-refinement) ---
    max_pipeline_candidates: int = 20     # Max candidates sent to validation pipeline
    dsr_prefilter_threshold: float = 0.95  # DSR threshold for IS prefilter
    dsr_prefilter_fallback: float = 0.90   # Relaxed threshold if zero pass at 0.95
    max_per_dedup_group: int = 3           # Max candidates per signal+risk param group

    # --- Refinement ---
    refinement_neighborhood_radius: int = 5  # ±5 index steps around locked best

    # --- Cyclic passes ---
    max_cyclic_passes: int = 0           # Additional passes through signal + trade profile stages
    cyclic_budget_fraction: float = 0.5  # Budget per cyclic stage relative to normal
    cyclic_improvement_threshold: float = 0.01  # Stop if improvement < 1%

    # --- Exploitation method ---
    exploitation_method: str = "cmaes"    # "cmaes" or "eda"
    cmaes_sigma0: float = 0.3            # Initial step size (fraction of param range)
    cmaes_population_size: int | None = None  # None = use batch_size (acts as implicit regularization)

    # --- Execution ---
    # PnL buffer = batch_size × max_trades × 8 bytes. At batch_size=4096:
    #   50K → 1.6GB (segfaults on Windows), 25K → 781MB (safe).
    # 25K trades covers M15 over 16 years (~3 trades/day).
    # The batch_size cap in run.py further guards against extreme cases.
    max_trades_per_trial: int = 25_000
    seed: int | None = None


# Presets scaled for i9-14900HX (24 cores, 64GB RAM)
# Batch sizes tuned for cache efficiency on 24-core systems

TURBO = OptimizationConfig(
    trials_per_stage=50_000,
    refinement_trials=100_000,
    batch_size=2048,
    exploration_pct=0.35,
    max_pipeline_candidates=10,
    max_cyclic_passes=0,  # Speed priority
    exploitation_method="cmaes",
)

STANDARD = OptimizationConfig(
    trials_per_stage=200_000,
    refinement_trials=400_000,
    batch_size=4096,
    exploration_pct=0.30,
    max_cyclic_passes=1,
    exploitation_method="cmaes",
)

DEEP = OptimizationConfig(
    trials_per_stage=500_000,
    refinement_trials=1_000_000,
    batch_size=4096,
    exploration_pct=0.25,
    max_pipeline_candidates=30,
    max_cyclic_passes=2,
    exploitation_method="cmaes",
)

MAX = OptimizationConfig(
    trials_per_stage=1_000_000,
    refinement_trials=2_000_000,
    batch_size=8192,
    exploration_pct=0.20,
    max_pipeline_candidates=50,
    max_cyclic_passes=2,
    exploitation_method="cmaes",
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
