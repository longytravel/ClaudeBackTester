"""Ranking and overfitting detection for optimization results.

Ranks candidates on back-test and forward-test quality scores,
applies forward/back ratio gate, and computes Deflated Sharpe Ratio.
"""

from __future__ import annotations

import numpy as np

from backtester.core.dtypes import (
    M_MAX_DD_PCT,
    M_PROFIT_FACTOR,
    M_QUALITY,
    M_R_SQUARED,
    M_SHARPE,
    M_SORTINO,
    M_TRADES,
    M_WIN_RATE,
    NUM_METRICS,
)


def rank_by_quality(metrics: np.ndarray) -> np.ndarray:
    """Rank trials by quality score (descending). Returns rank indices (0=best).

    Args:
        metrics: (N, NUM_METRICS) array.

    Returns:
        (N,) int64 array of ranks (0 = best quality score).
    """
    quality = metrics[:, M_QUALITY]
    # argsort ascending, then invert for descending rank
    order = np.argsort(-quality)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks


def combined_rank(
    back_metrics: np.ndarray,
    forward_metrics: np.ndarray,
    forward_weight: float = 1.5,
) -> np.ndarray:
    """Combined rank: back_rank + forward_rank * forward_weight.

    Lower combined rank = better candidate.

    Args:
        back_metrics: (N, NUM_METRICS) from back-test period.
        forward_metrics: (N, NUM_METRICS) from forward-test period.
        forward_weight: Weight for forward rank (>1 favors forward performance).

    Returns:
        (N,) float64 array of combined rank scores.
    """
    back_rank = rank_by_quality(back_metrics).astype(np.float64)
    fwd_rank = rank_by_quality(forward_metrics).astype(np.float64)
    return back_rank + fwd_rank * forward_weight


def forward_back_ratio(
    back_metrics: np.ndarray,
    forward_metrics: np.ndarray,
) -> np.ndarray:
    """Compute forward/back quality ratio per trial.

    Ratio < threshold suggests overfitting to back-test data.

    Returns:
        (N,) float64 array of ratios. 0 if back quality is 0.
    """
    back_q = back_metrics[:, M_QUALITY]
    fwd_q = forward_metrics[:, M_QUALITY]

    ratios = np.zeros_like(back_q)
    nonzero = back_q != 0
    ratios[nonzero] = fwd_q[nonzero] / back_q[nonzero]
    return ratios


def forward_back_gate(
    back_metrics: np.ndarray,
    forward_metrics: np.ndarray,
    min_ratio: float = 0.4,
) -> np.ndarray:
    """Boolean mask: True if forward/back ratio >= min_ratio."""
    ratios = forward_back_ratio(back_metrics, forward_metrics)
    return ratios >= min_ratio


def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_trades: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio (DSR) — overfitting-adjusted significance.

    Accounts for multiple testing by deflating the observed Sharpe ratio
    based on the number of trials attempted.

    Returns probability that the Sharpe is genuinely positive (0-1 scale).
    Values > 0.95 suggest the Sharpe is unlikely due to overfitting.

    Reference: Bailey & Lopez de Prado (2014).
    """
    if n_trades < 2 or n_trials < 1:
        return 0.0

    # Expected maximum z-score from N iid standard normals
    # E[max(Z_1,...,Z_n)] ≈ sqrt(2 * ln(n))
    import math
    e_max_z = math.sqrt(2.0 * math.log(max(n_trials, 2)))

    # Variance of Sharpe estimator with non-normal returns
    # Var(SR) ≈ (1 + 0.5*SR² - skew*SR + (kurt-3)/4 * SR²) / (N-1)
    sr = sharpe
    var_sr = (1.0 + 0.5 * sr**2 - skewness * sr
              + (kurtosis - 3.0) / 4.0 * sr**2) / max(n_trades - 1, 1)
    std_sr = math.sqrt(max(var_sr, 1e-10))

    # DSR z-score: observed SR in standard errors minus expected max z-score
    # z = SR*/σ(SR) - E[max(Z)] per Bailey & Lopez de Prado (2014)
    if std_sr == 0:
        return 0.0
    z = sr / std_sr - e_max_z

    # Convert to probability using normal CDF
    # Approximation: Phi(z) ≈ 1/(1+exp(-1.7*z - 0.73*z^3))
    # For more accuracy, use scipy if available
    try:
        from scipy.stats import norm
        return float(norm.cdf(z))
    except ImportError:
        # Logistic approximation with overflow protection
        exponent = -1.7 * z - 0.73 * z**3
        if exponent > 500:
            return 0.0
        if exponent < -500:
            return 1.0
        return 1.0 / (1.0 + math.exp(exponent))


def select_top_n(
    metrics: np.ndarray,
    n: int = 50,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Select top N trials by quality score.

    Args:
        metrics: (M, NUM_METRICS) array.
        n: Number of top trials to select.
        valid_mask: (M,) bool — only consider True entries.

    Returns:
        (K,) int64 array of trial indices (K <= n).
    """
    if valid_mask is not None:
        candidates = np.where(valid_mask)[0]
    else:
        candidates = np.arange(metrics.shape[0])

    if len(candidates) == 0:
        return np.array([], dtype=np.int64)

    quality = metrics[candidates, M_QUALITY]
    top_k = min(n, len(candidates))
    top_indices = np.argsort(-quality)[:top_k]
    return candidates[top_indices]
