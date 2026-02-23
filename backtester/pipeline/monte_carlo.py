"""Monte Carlo simulation for strategy validation (Stage 5).

Provides block bootstrap, permutation testing, trade skip resilience,
and execution stress testing. All operations work on per-trade PnL
arrays (1D float64). No Numba/JIT — pure numpy for clarity.
"""

from __future__ import annotations

import math

import numpy as np

from backtester.core.metrics import compute_metrics, sharpe_ratio
from backtester.optimizer.ranking import deflated_sharpe_ratio
from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.types import MonteCarloResult


def block_bootstrap(
    pnl: np.ndarray,
    n_iterations: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block bootstrap resampling of trade PnL to estimate Sharpe distribution.

    Preserves serial correlation by resampling in contiguous blocks rather
    than individual trades.

    Args:
        pnl: 1D array of per-trade PnL (pips).
        n_iterations: Number of bootstrap samples to draw.
        block_size: Length of each contiguous block.
        rng: numpy random Generator for reproducibility.

    Returns:
        (n_iterations,) array of Sharpe ratios from bootstrapped samples.
    """
    n = len(pnl)
    if n == 0:
        return np.zeros(n_iterations)

    block_size = max(1, min(block_size, n))
    n_blocks = math.ceil(n / block_size)

    sharpes = np.empty(n_iterations)
    for i in range(n_iterations):
        # Sample random block start indices
        starts = rng.integers(0, n, size=n_blocks)
        # Build bootstrapped sample by concatenating blocks
        pieces = []
        for s in starts:
            end = min(s + block_size, n)
            pieces.append(pnl[s:end])
            # If block wraps past end, we just take what's available
        bootstrapped = np.concatenate(pieces)[:n]
        sharpes[i] = sharpe_ratio(bootstrapped)

    return sharpes


def permutation_test(
    pnl: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    """Sign-flip permutation test for null hypothesis: mean PnL = 0.

    Randomly flips the sign of each trade PnL and recomputes Sharpe.
    The p-value is the fraction of permuted Sharpes >= the observed Sharpe.

    Args:
        pnl: 1D array of per-trade PnL (pips).
        n_permutations: Number of permutations.
        rng: numpy random Generator for reproducibility.

    Returns:
        p-value (0 to 1). Low values indicate the observed Sharpe is
        unlikely under the null.
    """
    n = len(pnl)
    if n == 0:
        return 1.0

    observed = sharpe_ratio(pnl)

    count_ge = 0
    for _ in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        permuted = pnl * signs
        perm_sharpe = sharpe_ratio(permuted)
        if perm_sharpe >= observed:
            count_ge += 1

    return count_ge / n_permutations


def trade_skip_test(
    pnl: np.ndarray,
    skip_fraction: float,
    rng: np.random.Generator,
) -> float:
    """Randomly drop a fraction of trades and compute quality score.

    Tests resilience: a robust strategy should maintain quality even
    when some trades are removed (simulating missed fills, etc.).

    Args:
        pnl: 1D array of per-trade PnL (pips).
        skip_fraction: Fraction of trades to remove (0 to 1).
        rng: numpy random Generator for reproducibility.

    Returns:
        quality_score of the remaining trades. 0.0 if too few trades remain.
    """
    n = len(pnl)
    if n == 0:
        return 0.0

    n_keep = max(1, int(n * (1.0 - skip_fraction)))
    indices = rng.choice(n, size=n_keep, replace=False)
    indices.sort()  # Maintain original order
    remaining = pnl[indices]

    metrics = compute_metrics(remaining)
    return metrics["quality_score"]


def execution_stress_test(
    pnl: np.ndarray,
    original_slippage: float,
    original_commission: float,
    stress_slippage_mult: float,
    stress_commission_mult: float,
) -> np.ndarray:
    """Apply increased execution costs to all trades.

    Deterministic: every trade gets additional slippage and commission
    deducted from its PnL.

    Args:
        pnl: 1D array of per-trade PnL (pips).
        original_slippage: Base slippage per trade (pips).
        original_commission: Base commission per trade (pips).
        stress_slippage_mult: Multiplier for slippage (e.g. 1.5 = +50%).
        stress_commission_mult: Multiplier for commission (e.g. 1.3 = +30%).

    Returns:
        Stressed PnL array (same length as input).
    """
    if len(pnl) == 0:
        return pnl.copy()

    extra_slippage = original_slippage * (stress_slippage_mult - 1.0)
    extra_commission = original_commission * (stress_commission_mult - 1.0)
    return pnl - extra_slippage - extra_commission


def run_monte_carlo(
    pnl: np.ndarray,
    n_trials: int,
    n_trades: int,
    config: PipelineConfig,
    original_slippage: float = 0.5,
    original_commission: float = 0.7,
) -> MonteCarloResult:
    """Run full Monte Carlo validation suite on a trade PnL array.

    Orchestrates block bootstrap, permutation test, trade skip, execution
    stress, and DSR computation.

    Args:
        pnl: 1D array of per-trade PnL (pips).
        n_trials: Total optimizer trials attempted (for DSR correction).
        n_trades: Number of trades in the strategy (can differ from len(pnl)
                  if pnl was already filtered).
        config: Pipeline configuration with MC parameters.
        original_slippage: Base slippage per trade (pips).
        original_commission: Base commission per trade (pips).

    Returns:
        MonteCarloResult with all fields populated.
    """
    result = MonteCarloResult()

    # Handle empty PnL gracefully
    if len(pnl) == 0:
        return result

    rng = np.random.default_rng(config.seed)

    # Observed Sharpe
    observed = sharpe_ratio(pnl)
    result.observed_sharpe = observed

    # --- Block bootstrap ---
    bootstrap_sharpes = block_bootstrap(
        pnl, config.mc_n_bootstrap, config.mc_block_size, rng
    )
    result.bootstrap_sharpe_mean = float(np.mean(bootstrap_sharpes))
    result.bootstrap_sharpe_std = float(np.std(bootstrap_sharpes, ddof=1))

    alpha = 1.0 - config.mc_bootstrap_ci
    ci_low = float(np.percentile(bootstrap_sharpes, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2)))
    result.bootstrap_sharpe_ci_low = ci_low
    result.bootstrap_sharpe_ci_high = ci_high

    # --- Permutation test ---
    result.permutation_p_value = permutation_test(
        pnl, config.mc_n_permutations, rng
    )

    # --- Trade skip test ---
    for level in config.mc_skip_levels:
        label = f"{level * 100:.0f}%"
        qs = trade_skip_test(pnl, level, rng)
        result.skip_results[label] = qs

    # --- Execution stress test ---
    stressed_pnl = execution_stress_test(
        pnl,
        original_slippage,
        original_commission,
        config.mc_stress_slippage_mult,
        config.mc_stress_commission_mult,
    )
    stressed_metrics = compute_metrics(stressed_pnl)
    result.stress_quality = stressed_metrics["quality_score"]

    original_metrics = compute_metrics(pnl)
    original_quality = original_metrics["quality_score"]
    if original_quality > 0:
        result.stress_quality_ratio = result.stress_quality / original_quality
    else:
        result.stress_quality_ratio = 0.0

    # --- Deflated Sharpe Ratio ---
    result.dsr = deflated_sharpe_ratio(observed, n_trials, n_trades)

    # --- Gate decision ---
    result.passed_gate = (
        result.dsr >= config.mc_dsr_gate
        and result.permutation_p_value <= config.mc_permutation_p_gate
    )

    return result
