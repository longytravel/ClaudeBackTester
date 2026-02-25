"""Combinatorial Purged Cross-Validation (CPCV) for the validation pipeline.

Divides data into N blocks, picks k blocks for testing in all C(N,k)
combinations. Each split purges training bars near test boundaries and
applies embargo. Produces a distribution of OOS Sharpe ratios.

Uses a SINGLE BacktestEngine on the full dataset with windowed evaluation
to avoid memory issues from creating hundreds of engine instances.

Reference: Bailey, Borwein, Lopez de Prado & Zhu (2017).
"""

from __future__ import annotations

import gc
import logging
import math
from itertools import combinations
from typing import Any

import numpy as np

from backtester.core.dtypes import EXEC_FULL
from backtester.core.encoding import encode_params
from backtester.core.engine import BacktestEngine
from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.types import CPCVFoldResult, CPCVResult
from backtester.pipeline.walk_forward import (
    _metrics_row_to_window_result,
    build_engine,
)
from backtester.strategies.base import Strategy

logger = logging.getLogger(__name__)


def generate_blocks(n_bars: int, n_blocks: int) -> list[tuple[int, int]]:
    """Divide data into N approximately equal blocks.

    Args:
        n_bars: Total number of bars.
        n_blocks: Number of blocks to create.

    Returns:
        List of (start, end) tuples. end is exclusive.
    """
    if n_blocks <= 0 or n_bars <= 0:
        return []

    block_size = n_bars // n_blocks
    remainder = n_bars % n_blocks

    blocks: list[tuple[int, int]] = []
    start = 0
    for i in range(n_blocks):
        # Distribute remainder across first `remainder` blocks
        size = block_size + (1 if i < remainder else 0)
        blocks.append((start, start + size))
        start += size

    return blocks


def generate_folds(
    n_blocks: int,
    k_test: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Generate all C(n_blocks, k_test) fold assignments.

    Each fold is a (train_block_indices, test_block_indices) tuple.

    Args:
        n_blocks: Total number of blocks.
        k_test: Number of blocks to use for testing per fold.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    all_indices = list(range(n_blocks))
    folds = []

    for test_combo in combinations(all_indices, k_test):
        test_set = set(test_combo)
        train_indices = tuple(i for i in all_indices if i not in test_set)
        folds.append((train_indices, tuple(test_combo)))

    return folds


def build_fold_masks(
    blocks: list[tuple[int, int]],
    train_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
    n_bars: int,
    purge_bars: int,
    embargo_bars: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build boolean masks for one fold with purging and embargo.

    Purging: remove `purge_bars` from train data on BOTH sides of each
    test block boundary.

    Embargo: remove `embargo_bars` from train data AFTER each test block.

    Args:
        blocks: List of (start, end) block boundaries.
        train_indices: Block indices assigned to training.
        test_indices: Block indices assigned to testing.
        n_bars: Total bars in dataset.
        purge_bars: Number of bars to purge near test boundaries.
        embargo_bars: Number of bars of embargo after test blocks.

    Returns:
        (train_mask, test_mask, n_purged) where masks are boolean arrays
        of length n_bars, and n_purged is the number of bars removed.
    """
    train_mask = np.zeros(n_bars, dtype=np.bool_)
    test_mask = np.zeros(n_bars, dtype=np.bool_)

    # Mark test blocks
    for ti in test_indices:
        start, end = blocks[ti]
        test_mask[start:end] = True

    # Mark initial train blocks
    for ti in train_indices:
        start, end = blocks[ti]
        train_mask[start:end] = True

    # Count initial train bars before purging
    initial_train = int(train_mask.sum())

    # Purge: remove from train near test boundaries
    for ti in test_indices:
        test_start, test_end = blocks[ti]

        # Purge before test block
        purge_start = max(0, test_start - purge_bars)
        train_mask[purge_start:test_start] = False

        # Purge after test block
        purge_end = min(n_bars, test_end + purge_bars)
        train_mask[test_end:purge_end] = False

        # Embargo after test block (additional to purge)
        embargo_end = min(n_bars, test_end + purge_bars + embargo_bars)
        train_mask[test_end:embargo_end] = False

    # Ensure no overlap
    train_mask &= ~test_mask

    n_purged = initial_train - int(train_mask.sum())
    return train_mask, test_mask, n_purged


def evaluate_candidate_on_fold(
    engine: BacktestEngine,
    params_dict: dict[str, Any],
    blocks: list[tuple[int, int]],
    test_indices: tuple[int, ...],
    config: PipelineConfig,
    fold_index: int = 0,
    train_indices: tuple[int, ...] = (),
    n_bars: int = 0,
) -> CPCVFoldResult:
    """Evaluate one candidate on one CPCV fold using windowed evaluation.

    Uses the pre-built engine's evaluate_batch_windowed() for each test
    block, avoiding per-block engine creation.
    """
    # Build masks for purge/embargo stats
    _, _, n_purged = build_fold_masks(
        blocks, train_indices, test_indices,
        n_bars, config.cpcv_purge_bars, config.cpcv_embargo_bars,
    )

    # Encode params
    param_row = encode_params(engine.encoding, params_dict)
    param_matrix = param_row.reshape(1, -1)

    # Evaluate each test block as a separate window
    all_sharpes: list[float] = []
    all_qualities: list[float] = []
    total_trades = 0

    for ti in test_indices:
        block_start, block_end = blocks[ti]
        metrics = engine.evaluate_batch_windowed(
            param_matrix, block_start, block_end, exec_mode=EXEC_FULL,
        )
        row = metrics[0]

        sharpe = float(row[3])    # M_SHARPE
        quality = float(row[9])   # M_QUALITY
        n_trades = int(row[0])    # M_TRADES

        all_sharpes.append(sharpe)
        all_qualities.append(quality)
        total_trades += n_trades

    # Aggregate across test blocks in this fold
    if all_sharpes:
        fold_sharpe = float(np.mean(all_sharpes))
        fold_quality = float(np.mean(all_qualities))
    else:
        fold_sharpe = 0.0
        fold_quality = 0.0

    return CPCVFoldResult(
        fold_index=fold_index,
        train_blocks=train_indices,
        test_blocks=test_indices,
        n_purged=n_purged,
        n_trades=total_trades,
        sharpe=fold_sharpe,
        quality_score=fold_quality,
    )


def cpcv_validate(
    strategy: Strategy,
    candidates: list[dict[str, Any]],
    data_arrays: dict[str, np.ndarray],
    config: PipelineConfig,
    pip_value: float = 0.0001,
    slippage_pips: float = 0.5,
    engine: BacktestEngine | None = None,
) -> list[CPCVResult]:
    """Run CPCV validation for a list of candidates.

    Creates ONE BacktestEngine on the full dataset, then evaluates each
    fold's test blocks using windowed evaluation.

    Args:
        strategy: Strategy instance.
        candidates: List of parameter dictionaries.
        data_arrays: Dict with keys: open, high, low, close, volume, spread,
                     bar_hour, bar_day_of_week.
        config: Pipeline configuration.

    Returns:
        List of CPCVResult, one per candidate.
    """
    n_bars = len(data_arrays["close"])
    n_blocks = config.cpcv_n_blocks
    k_test = config.cpcv_k_test

    # Generate blocks
    blocks = generate_blocks(n_bars, n_blocks)
    if not blocks:
        logger.warning("CPCV: no blocks generated")
        return [CPCVResult(passed_gate=False) for _ in candidates]

    # Check minimum block size
    min_block_size = min(end - start for start, end in blocks)
    if min_block_size < config.cpcv_min_block_bars:
        logger.warning(
            f"CPCV: smallest block ({min_block_size} bars) < "
            f"min_block_bars ({config.cpcv_min_block_bars}), skipping"
        )
        return [CPCVResult(
            n_blocks=n_blocks, k_test=k_test, n_folds=0, passed_gate=False,
        ) for _ in candidates]

    # Generate all folds
    folds = generate_folds(n_blocks, k_test)
    n_folds = len(folds)
    logger.info(
        f"CPCV: {n_blocks} blocks, k={k_test}, "
        f"C({n_blocks},{k_test})={n_folds} folds, "
        f"block sizes {min_block_size}-{max(e - s for s, e in blocks)} bars"
    )

    # Use provided engine or create one
    if engine is None:
        engine = build_engine(strategy, data_arrays, config, pip_value, slippage_pips)
    logger.info("CPCV engine: %d signals on %d bars", engine.n_signals, engine.n_bars)

    results: list[CPCVResult] = []

    for cand_idx, params_dict in enumerate(candidates):
        fold_results: list[CPCVFoldResult] = []

        for fold_idx, (train_idxs, test_idxs) in enumerate(folds):
            fr = evaluate_candidate_on_fold(
                engine=engine,
                params_dict=params_dict,
                blocks=blocks,
                test_indices=test_idxs,
                config=config,
                fold_index=fold_idx,
                train_indices=train_idxs,
                n_bars=n_bars,
            )
            fold_results.append(fr)

        # Compute distribution statistics
        sharpes = np.array([f.sharpe for f in fold_results])
        qualities = np.array([f.quality_score for f in fold_results])

        mean_sharpe = float(np.mean(sharpes))
        median_sharpe = float(np.median(sharpes))
        std_sharpe = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0

        # 95% CI using t-distribution approximation
        if len(sharpes) > 1 and std_sharpe > 0:
            se = std_sharpe / math.sqrt(len(sharpes))
            # For 45 folds, t_0.025 ≈ 2.015; use 1.96 as approximation
            ci_half = 1.96 * se
            ci_low = mean_sharpe - ci_half
            ci_high = mean_sharpe + ci_half
        else:
            ci_low = mean_sharpe
            ci_high = mean_sharpe

        pct_positive = float(np.mean(sharpes > 0))
        mean_quality = float(np.mean(qualities))
        median_quality = float(np.median(qualities))

        # Apply gates
        passed_gate = (
            pct_positive >= config.cpcv_pct_positive_sharpe_gate
            and mean_sharpe >= config.cpcv_mean_sharpe_gate
        )

        cpcv_result = CPCVResult(
            n_blocks=n_blocks,
            k_test=k_test,
            n_folds=n_folds,
            mean_sharpe=mean_sharpe,
            median_sharpe=median_sharpe,
            std_sharpe=std_sharpe,
            sharpe_ci_low=ci_low,
            sharpe_ci_high=ci_high,
            pct_positive_sharpe=pct_positive,
            mean_quality=mean_quality,
            median_quality=median_quality,
            passed_gate=passed_gate,
            folds=fold_results,
        )
        results.append(cpcv_result)

        logger.info(
            "CPCV candidate %d: mean_sharpe=%.3f, pct_positive=%.1f%%, "
            "CI=[%.3f, %.3f], gate=%s",
            cand_idx, mean_sharpe, pct_positive * 100,
            ci_low, ci_high,
            "PASS" if passed_gate else "FAIL",
        )

    return results
