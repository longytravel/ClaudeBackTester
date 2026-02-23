"""Top-level optimization entry point.

Usage:
    from backtester.optimizer.run import optimize
    result = optimize(strategy, data_back, data_forward, config)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backtester.core.dtypes import (
    EXEC_BASIC,
    EXEC_FULL,
    M_QUALITY,
    M_SHARPE,
    M_TRADES,
    NUM_METRICS,
)
from backtester.core.encoding import (
    build_encoding_spec,
    decode_params,
    indices_to_values,
)
from backtester.core.engine import BacktestEngine
from backtester.optimizer.archive import DiversityArchive, select_top_n_diverse
from backtester.optimizer.config import OptimizationConfig, get_preset
from backtester.optimizer.prefilter import postfilter_results
from backtester.optimizer.ranking import (
    combined_rank,
    deflated_sharpe_ratio,
    forward_back_gate,
    select_top_n,
)
from backtester.optimizer.staged import StagedOptimizer, StagedResult
from backtester.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A single optimization candidate with back + forward metrics."""
    index: int
    params: dict[str, Any]
    back_metrics: dict[str, float]
    forward_metrics: dict[str, float] | None = None
    combined_rank: float = 0.0
    dsr: float = 0.0
    forward_back_ratio: float = 0.0


@dataclass
class OptimizationResult:
    """Full optimization result."""
    candidates: list[Candidate] = field(default_factory=list)
    staged_result: StagedResult | None = None
    total_trials: int = 0
    elapsed_seconds: float = 0.0
    evals_per_second: float = 0.0


def _estimate_memory_mb(n_bars: int, n_signals: int, batch_size: int, n_params: int) -> float:
    """Estimate peak memory usage in MB."""
    # Price arrays: 4 × n_bars × 8 bytes
    price_mb = 4 * n_bars * 8 / 1e6
    # Signal arrays: 7 × n_signals × 8 bytes
    signal_mb = 7 * n_signals * 8 / 1e6
    # Param matrix: batch_size × n_params × 8 bytes
    param_mb = batch_size * n_params * 8 / 1e6
    # Metrics output: batch_size × NUM_METRICS × 8 bytes
    metrics_mb = batch_size * NUM_METRICS * 8 / 1e6
    # PnL buffers: batch_size × max_trades × 8 bytes
    pnl_mb = batch_size * 5000 * 8 / 1e6

    return price_mb + signal_mb + param_mb + metrics_mb + pnl_mb


def optimize(
    strategy: Strategy,
    open_back: np.ndarray,
    high_back: np.ndarray,
    low_back: np.ndarray,
    close_back: np.ndarray,
    volume_back: np.ndarray,
    spread_back: np.ndarray,
    open_fwd: np.ndarray | None = None,
    high_fwd: np.ndarray | None = None,
    low_fwd: np.ndarray | None = None,
    close_fwd: np.ndarray | None = None,
    volume_fwd: np.ndarray | None = None,
    spread_fwd: np.ndarray | None = None,
    config: OptimizationConfig | None = None,
    pip_value: float = 0.0001,
    slippage_pips: float = 0.5,
) -> OptimizationResult:
    """Run full optimization pipeline.

    1. Staged optimization on back-test data
    2. Forward-test evaluation of top candidates
    3. Ranking and diversity selection
    4. DSR overfitting gate

    Args:
        strategy: Strategy instance.
        open_back..spread_back: Back-test price data.
        open_fwd..spread_fwd: Forward-test price data (optional).
        config: Optimization configuration.
        pip_value: Pip value (0.0001 for most pairs).
        slippage_pips: Slippage in pips.

    Returns:
        OptimizationResult with ranked candidates.
    """
    config = config or OptimizationConfig()
    t0 = time.time()

    # Memory check
    mem_mb = _estimate_memory_mb(
        len(high_back), 0, config.batch_size,
        len(list(strategy.param_space())),
    )
    logger.info(f"Estimated peak memory: {mem_mb:.1f} MB")

    # --- Stage 1: Build engine and run staged optimization ---
    logger.info("Building back-test engine...")
    engine_back = BacktestEngine(
        strategy, open_back, high_back, low_back, close_back,
        volume_back, spread_back, pip_value, slippage_pips,
        config.max_trades_per_trial,
    )
    logger.info(f"Generated {engine_back.n_signals} signals")

    logger.info("Running staged optimization...")
    staged = StagedOptimizer(engine_back, config)
    staged_result = staged.optimize()
    logger.info(
        f"Staged optimization complete: {staged_result.total_trials} trials, "
        f"best quality={staged_result.best_quality:.2f}"
    )

    # --- Stage 2: Build final candidate list ---
    # For now, return the best from staged optimization
    result = OptimizationResult()
    result.staged_result = staged_result
    result.total_trials = staged_result.total_trials

    if staged_result.best_indices is not None:
        # Decode best params
        spec = engine_back.encoding
        value_row = indices_to_values(spec, staged_result.best_indices.reshape(1, -1))
        params_dict = decode_params(spec, value_row[0])

        # Back metrics
        back_metrics_dict = {
            "trades": float(staged_result.best_metrics[M_TRADES]),
            "quality_score": float(staged_result.best_metrics[M_QUALITY]),
            "sharpe": float(staged_result.best_metrics[M_SHARPE]),
        }

        candidate = Candidate(
            index=0,
            params=params_dict,
            back_metrics=back_metrics_dict,
        )

        # Forward-test if data provided
        if high_fwd is not None:
            logger.info("Evaluating on forward-test data...")
            engine_fwd = BacktestEngine(
                strategy, open_fwd, high_fwd, low_fwd, close_fwd,
                volume_fwd, spread_fwd, pip_value, slippage_pips,
                config.max_trades_per_trial,
            )
            from backtester.core.encoding import encode_params
            fwd_row = encode_params(spec, params_dict).reshape(1, -1)
            fwd_metrics = engine_fwd.evaluate_batch(fwd_row, EXEC_FULL)

            candidate.forward_metrics = {
                "trades": float(fwd_metrics[0, M_TRADES]),
                "quality_score": float(fwd_metrics[0, M_QUALITY]),
                "sharpe": float(fwd_metrics[0, M_SHARPE]),
            }

            # DSR
            candidate.dsr = deflated_sharpe_ratio(
                float(staged_result.best_metrics[M_SHARPE]),
                staged_result.total_trials,
                int(staged_result.best_metrics[M_TRADES]),
            )

        result.candidates.append(candidate)

    elapsed = time.time() - t0
    result.elapsed_seconds = elapsed
    if elapsed > 0:
        result.evals_per_second = staged_result.total_trials / elapsed

    logger.info(
        f"Optimization complete: {elapsed:.1f}s, "
        f"{result.evals_per_second:.0f} evals/sec, "
        f"{len(result.candidates)} candidates"
    )

    return result
