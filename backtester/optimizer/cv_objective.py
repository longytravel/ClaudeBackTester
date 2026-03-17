"""Cross-Validation Objective for optimizer.

Instead of evaluating each trial on a single IS period, evaluates across
K time folds and aggregates for robustness. Prevents overfitting to any
single time period.

Usage:
    fold_config = auto_configure_folds(n_bars, bars_per_year, timeframe, ...)
    cv = CVObjective(engine, fold_config)
    metrics = cv.evaluate_batch(param_matrix, exec_mode)  # drop-in replacement
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from backtester.core.dtypes import M_QUALITY, M_TRADES, NUM_METRICS

logger = logging.getLogger(__name__)

# Bars per trading day by timeframe (for embargo conversion)
BARS_PER_DAY: dict[str, float] = {
    "M1": 1440, "M5": 288, "M15": 96, "M30": 48,
    "H1": 24, "H4": 6, "D1": 1, "W": 0.2,
}


@dataclass
class CVFoldConfig:
    """Auto-calculated fold configuration."""
    n_folds: int
    fold_boundaries: list[tuple[int, int]]  # (start_bar, end_bar) per fold
    embargo_bars: int
    min_trades_per_fold: int
    aggregation: str = "mean_std"
    aggregation_lambda: float = 1.0
    early_stopping: bool = True


def auto_configure_folds(
    n_bars: int,
    bars_per_year: float,
    timeframe: str,
    expected_trades_per_year: float,
    embargo_days: int = 5,
    min_trades_per_fold: int = 30,
    aggregation: str = "mean_std",
    aggregation_lambda: float = 1.0,
    early_stopping: bool = True,
    n_folds_override: int | None = None,
) -> CVFoldConfig:
    """Auto-calculate fold configuration from data and strategy characteristics.

    Args:
        n_bars: Total bars in the IS dataset.
        bars_per_year: Bars per year for this timeframe.
        timeframe: "M15", "H1", "H4", "D1", etc.
        expected_trades_per_year: Estimated from signal_count / data_years.
        embargo_days: Gap between folds in calendar days.
        min_trades_per_fold: Minimum trades for meaningful quality score.
        aggregation: "mean_std", "cvar", or "geometric_mean".
        aggregation_lambda: Penalty weight for mean_std.
        early_stopping: Enable progressive culling of bad trials.
        n_folds_override: Force specific K (bypasses auto-calculation).

    Returns:
        CVFoldConfig with fold boundaries and settings.
    """
    data_years = n_bars / bars_per_year if bars_per_year > 0 else 1.0
    expected_total_trades = expected_trades_per_year * data_years

    if n_folds_override is not None:
        n_folds = max(2, min(n_folds_override, 10))
    else:
        # Auto-calculate K
        max_folds_by_trades = expected_total_trades / min_trades_per_fold
        max_folds_by_years = data_years / 2.0  # each fold needs ≥2 years
        n_folds = int(round(min(max_folds_by_trades, max_folds_by_years)))
        n_folds = max(3, min(n_folds, 7))

    # Embargo in bars (calendar days → bars)
    bpd = BARS_PER_DAY.get(timeframe, 24)
    embargo_bars = max(1, int(embargo_days * bpd))

    # Equal-width fold boundaries with embargo gaps
    usable_bars = n_bars - embargo_bars * (n_folds - 1)
    fold_size = max(1, usable_bars // n_folds)

    boundaries: list[tuple[int, int]] = []
    offset = 0
    for i in range(n_folds):
        start = offset
        end = start + fold_size
        if i == n_folds - 1:
            end = n_bars  # last fold gets remainder
        end = min(end, n_bars)
        boundaries.append((start, end))
        offset = end + embargo_bars

    logger.info(
        f"CV folds: K={n_folds}, fold_size={fold_size:,} bars, "
        f"embargo={embargo_bars} bars ({embargo_days}d), "
        f"expected {expected_trades_per_year:.0f} trades/yr, "
        f"aggregation={aggregation}"
    )

    return CVFoldConfig(
        n_folds=n_folds,
        fold_boundaries=boundaries,
        embargo_bars=embargo_bars,
        min_trades_per_fold=min_trades_per_fold,
        aggregation=aggregation,
        aggregation_lambda=aggregation_lambda,
        early_stopping=early_stopping,
    )


def aggregate_fold_scores(
    fold_qualities: np.ndarray,
    fold_trades: np.ndarray,
    method: str,
    lam: float,
    min_trades: int,
) -> np.ndarray:
    """Aggregate per-fold quality scores into a single robustness score.

    Args:
        fold_qualities: (N, K) quality scores per trial per fold.
        fold_trades: (N, K) trade counts per trial per fold.
        method: "mean_std", "cvar", or "geometric_mean".
        lam: Penalty weight for mean_std.
        min_trades: Minimum trades for a fold to count.

    Returns:
        (N,) aggregated quality scores.
    """
    n, k = fold_qualities.shape
    result = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # Exclude folds with insufficient trades
        valid = fold_trades[i] >= min_trades
        if valid.sum() < 2:
            result[i] = 0.0
            continue

        scores = fold_qualities[i, valid]

        if method == "mean_std":
            mean = scores.mean()
            std = scores.std()
            val = mean - lam * std
            # Floor: if any valid fold is <= 0, cap at 0
            if scores.min() <= 0:
                val = min(val, 0.0)
            result[i] = max(val, 0.0)

        elif method == "cvar":
            # Worst 40% of valid folds
            sorted_s = np.sort(scores)
            n_tail = max(1, int(len(sorted_s) * 0.4))
            result[i] = max(sorted_s[:n_tail].mean(), 0.0)

        elif method == "geometric_mean":
            if np.any(scores <= 0):
                result[i] = 0.0
            else:
                result[i] = np.exp(np.mean(np.log(scores)))

    return result


class CVObjective:
    """Cross-validation objective wrapper for the optimizer.

    Drop-in replacement for engine.evaluate_batch(). Evaluates each trial
    across K folds and returns aggregated metrics.
    """

    def __init__(self, engine, fold_config: CVFoldConfig):
        self.engine = engine
        self.config = fold_config
        self._fold_evals = 0  # total fold evaluations for diagnostics

    def evaluate_batch(
        self,
        param_matrix: np.ndarray,
        exec_mode: int,
    ) -> np.ndarray:
        """Evaluate batch across K folds with early stopping.

        Args:
            param_matrix: (N, P) float64 value matrix.
            exec_mode: EXEC_BASIC or EXEC_FULL.

        Returns:
            (N, NUM_METRICS) with quality column replaced by aggregated CV score.
        """
        n = param_matrix.shape[0]
        k = self.config.n_folds
        min_trades = self.config.min_trades_per_fold

        fold_qualities = np.zeros((n, k), dtype=np.float64)
        fold_trades = np.zeros((n, k), dtype=np.float64)

        # Keep the last fold's full metrics as the "representative" metrics
        # (for Sharpe, trades, etc. in the output row)
        last_metrics = np.zeros((n, NUM_METRICS), dtype=np.float64)

        alive = np.ones(n, dtype=np.bool_)

        for fold_idx in range(k):
            fold_start, fold_end = self.config.fold_boundaries[fold_idx]

            if alive.sum() == 0:
                break

            # Evaluate alive trials on this fold
            alive_indices = np.where(alive)[0]
            alive_params = param_matrix[alive_indices]

            metrics_k = self.engine.evaluate_batch_windowed(
                alive_params, fold_start, fold_end, exec_mode,
            )
            self._fold_evals += len(alive_indices)

            fold_qualities[alive_indices, fold_idx] = metrics_k[:, M_QUALITY]
            fold_trades[alive_indices, fold_idx] = metrics_k[:, M_TRADES]

            # Update representative metrics (last evaluated fold)
            last_metrics[alive_indices] = metrics_k

            # Early stopping (softer approach per Codex review)
            if self.config.early_stopping and fold_idx < k - 1:
                if fold_idx == 0:
                    # Kill only clearly broken: zero quality with enough trades
                    has_trades = fold_trades[:, 0] >= min_trades
                    kill = has_trades & (fold_qualities[:, 0] <= 0)
                    alive &= ~kill
                elif fold_idx >= 1:
                    # Soft cull: bottom 30% by running mean
                    running_mean = np.zeros(n, dtype=np.float64)
                    for a_idx in np.where(alive)[0]:
                        valid_so_far = fold_trades[a_idx, :fold_idx + 1] >= min_trades
                        if valid_so_far.sum() > 0:
                            running_mean[a_idx] = fold_qualities[a_idx, :fold_idx + 1][valid_so_far].mean()
                    alive_means = running_mean[alive]
                    if len(alive_means) > 10:
                        cutoff = np.percentile(alive_means, 30)
                        for a_idx in np.where(alive)[0]:
                            if running_mean[a_idx] < cutoff:
                                alive[a_idx] = False

        # Aggregate fold scores
        aggregated = aggregate_fold_scores(
            fold_qualities, fold_trades,
            self.config.aggregation,
            self.config.aggregation_lambda,
            min_trades,
        )

        # Replace quality in output metrics with aggregated CV score
        output = last_metrics.copy()
        output[:, M_QUALITY] = aggregated

        # Dead trials (killed by early stopping) get zero
        dead = ~alive & (aggregated == 0)
        output[dead] = 0.0

        return output

    @property
    def fold_evals(self) -> int:
        """Total fold evaluations performed (for throughput diagnostics)."""
        return self._fold_evals
