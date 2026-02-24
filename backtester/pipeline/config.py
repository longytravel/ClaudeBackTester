"""Pipeline configuration with all thresholds and defaults."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    """Configuration for the full validation pipeline.

    All thresholds are configurable with sensible defaults based on
    research recommendations.
    """

    # --- Execution costs (must match optimizer defaults) ---
    commission_pips: float = 0.7    # IC Markets Raw, ~$7/lot RT ≈ 0.7 pips EUR/USD
    max_spread_pips: float = 3.0    # Reject signals with spread > 3 pips

    # --- General ---
    output_dir: str = "pipeline_output"
    n_candidates: int = 20  # Number of candidates to validate from optimizer

    # --- Walk-Forward (Stage 3) ---
    wf_window_bars: int = 8760  # ~1.45 years at H1 (365 * 24)
    wf_step_bars: int = 4380    # ~0.72 years at H1 (step = half window)
    wf_embargo_bars: int = 168  # 1 week of H1 bars between train/test
    wf_anchored: bool = False   # Rolling (False) vs anchored (True) windows
    wf_lookback_prefix: int = 200  # Extra bars before test window for indicator warmup
    wf_min_trades_per_window: int = 10
    wf_pass_rate_gate: float = 0.6      # Hard gate: >= 60% windows must pass
    wf_mean_sharpe_gate: float = 0.3    # Hard gate: mean OOS Sharpe >= 0.3

    # --- CPCV (sub-step of Stage 3) ---
    cpcv_enabled: bool = True
    cpcv_n_blocks: int = 10
    cpcv_k_test: int = 2
    cpcv_purge_bars: int = 200         # Max trade duration (bars removed near test)
    cpcv_embargo_bars: int = 168       # 1 week H1 (bars removed after test)
    cpcv_min_block_bars: int = 500     # Skip CPCV if blocks smaller than this
    cpcv_pct_positive_sharpe_gate: float = 0.6   # >= 60% folds positive Sharpe
    cpcv_mean_sharpe_gate: float = 0.2           # Mean Sharpe >= 0.2

    # --- Stability (Stage 4) ---
    stab_perturbation_steps: int = 3      # +-3 steps per numeric param
    stab_use_forward_data: bool = True    # Test perturbations on forward data
    # Rating thresholds (advisory only, no hard gate)
    stab_robust_mean: float = 0.8
    stab_robust_min: float = 0.5
    stab_moderate_mean: float = 0.6
    stab_fragile_mean: float = 0.4

    # --- Monte Carlo (Stage 5) ---
    mc_n_bootstrap: int = 1000       # Block bootstrap iterations
    mc_block_size: int = 10          # Bootstrap block size (trades)
    mc_n_permutations: int = 1000    # Sign-flip permutation iterations
    mc_skip_levels: list[float] = field(default_factory=lambda: [0.05, 0.10])
    mc_stress_slippage_mult: float = 1.5   # +50% slippage
    mc_stress_commission_mult: float = 1.3  # +30% commission
    mc_dsr_gate: float = 0.95             # Hard gate: DSR >= 0.95
    mc_permutation_p_gate: float = 0.05   # Hard gate: p-value <= 0.05
    mc_bootstrap_ci: float = 0.95         # Confidence interval level

    # --- Confidence Scoring (Stage 6) ---
    # Weights for composite score (must sum to 1.0)
    conf_weight_walk_forward: float = 0.30
    conf_weight_monte_carlo: float = 0.25
    conf_weight_forward_back: float = 0.15
    conf_weight_stability: float = 0.10
    conf_weight_dsr: float = 0.10
    conf_weight_backtest: float = 0.10

    # Rating thresholds
    conf_green_threshold: float = 70.0
    conf_yellow_threshold: float = 40.0
    # Below yellow = RED

    # --- Regime Analysis (after Monte Carlo, advisory) ---
    regime_enabled: bool = True
    regime_adx_period: int = 14
    regime_atr_period: int = 14
    regime_adx_trending: float = 25.0
    regime_adx_ranging: float = 20.0
    regime_natr_lookback: int = 100
    regime_natr_high_pctile: float = 75.0
    regime_min_bars: int = 8
    regime_min_trades: int = 30

    # --- Pipeline execution ---
    checkpoint_enabled: bool = True
    seed: int = 42

    def validate(self) -> list[str]:
        """Validate config, return list of error strings."""
        errors: list[str] = []
        weights = (
            self.conf_weight_walk_forward
            + self.conf_weight_monte_carlo
            + self.conf_weight_forward_back
            + self.conf_weight_stability
            + self.conf_weight_dsr
            + self.conf_weight_backtest
        )
        if abs(weights - 1.0) > 0.01:
            errors.append(f"Confidence weights sum to {weights:.3f}, expected 1.0")
        if self.wf_pass_rate_gate < 0 or self.wf_pass_rate_gate > 1:
            errors.append(f"wf_pass_rate_gate must be 0-1, got {self.wf_pass_rate_gate}")
        if self.mc_dsr_gate < 0 or self.mc_dsr_gate > 1:
            errors.append(f"mc_dsr_gate must be 0-1, got {self.mc_dsr_gate}")
        if self.mc_permutation_p_gate < 0 or self.mc_permutation_p_gate > 1:
            errors.append(f"mc_permutation_p_gate must be 0-1, got {self.mc_permutation_p_gate}")
        return errors
