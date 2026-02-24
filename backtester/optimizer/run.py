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
    DEFAULT_COMMISSION_PIPS,
    DEFAULT_MAX_SPREAD_PIPS,
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
    encode_params,
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
    pnl_mb = batch_size * 50000 * 8 / 1e6

    return price_mb + signal_mb + param_mb + metrics_mb + pnl_mb


def _add_single_best_candidate(
    result: OptimizationResult,
    staged_result: StagedResult,
    spec: Any,
    engine_fwd: BacktestEngine | None,
    config: OptimizationConfig,
) -> None:
    """Add the single best candidate from staged optimization (fallback)."""
    value_row = indices_to_values(spec, staged_result.best_indices.reshape(1, -1))
    params_dict = decode_params(spec, value_row[0])

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

    if engine_fwd is not None:
        fwd_row = encode_params(spec, params_dict).reshape(1, -1)
        fwd_metrics = engine_fwd.evaluate_batch(fwd_row, EXEC_FULL)

        candidate.forward_metrics = {
            "trades": float(fwd_metrics[0, M_TRADES]),
            "quality_score": float(fwd_metrics[0, M_QUALITY]),
            "sharpe": float(fwd_metrics[0, M_SHARPE]),
        }

        candidate.dsr = deflated_sharpe_ratio(
            float(staged_result.best_metrics[M_SHARPE]),
            staged_result.total_trials,
            int(staged_result.best_metrics[M_TRADES]),
        )

        back_q = candidate.back_metrics.get("quality_score", 0)
        fwd_q = candidate.forward_metrics.get("quality_score", 0)
        candidate.forward_back_ratio = fwd_q / back_q if back_q > 0 else 0.0

    result.candidates.append(candidate)
    logger.info("Single-best candidate selected (fallback)")


def _add_multi_candidates(
    result: OptimizationResult,
    staged_result: StagedResult,
    spec: Any,
    engine_fwd: BacktestEngine | None,
    config: OptimizationConfig,
) -> bool:
    """Select multiple diverse candidates from refinement passing trials.

    Returns True if at least one candidate was added, False otherwise.
    """
    ref_indices = staged_result.refinement_indices
    ref_metrics = staged_result.refinement_metrics

    if ref_indices is None or ref_metrics is None or len(ref_indices) == 0:
        logger.info("No refinement passing trials for multi-candidate selection")
        return False

    n_candidates = min(config.top_n_candidates, len(ref_indices))
    logger.info(
        f"Multi-candidate: {len(ref_indices)} passing trials, "
        f"selecting top {n_candidates} diverse"
    )

    # Step 1: Diversity selection from refinement passing trials
    selected_local = select_top_n_diverse(
        ref_metrics, n=n_candidates, params=ref_indices,
    )

    if not selected_local:
        return False

    # Build selected arrays (indices are into ref_indices/ref_metrics)
    sel_back_indices = ref_indices[selected_local]  # (K, P) index-space
    sel_back_metrics = ref_metrics[selected_local]  # (K, NUM_METRICS)

    # Step 2: Convert to value space for forward evaluation
    sel_value_matrix = indices_to_values(spec, sel_back_indices)

    # Step 3: Forward-test all if forward engine available
    sel_fwd_metrics = None
    if engine_fwd is not None:
        logger.info(f"Forward-testing {len(sel_value_matrix)} candidates...")
        sel_fwd_metrics = engine_fwd.evaluate_batch(sel_value_matrix, EXEC_FULL)

        # Step 4: Forward/back gate
        gate_mask = forward_back_gate(
            sel_back_metrics, sel_fwd_metrics,
            min_ratio=config.min_forward_back_ratio,
        )
        n_passed = int(gate_mask.sum())
        logger.info(
            f"Forward/back gate: {n_passed}/{len(gate_mask)} passed "
            f"(min ratio={config.min_forward_back_ratio})"
        )

        if n_passed == 0:
            # No candidates pass gate — return False to fallback
            logger.warning("All multi-candidates failed forward/back gate")
            return False

        # Filter to passing candidates only
        passing_idx = np.where(gate_mask)[0]
        sel_back_indices = sel_back_indices[passing_idx]
        sel_back_metrics = sel_back_metrics[passing_idx]
        sel_value_matrix = sel_value_matrix[passing_idx]
        sel_fwd_metrics = sel_fwd_metrics[passing_idx]

        # Step 5: Compute DSR and combined rank
        dsrs = np.array([
            deflated_sharpe_ratio(
                float(sel_back_metrics[i, M_SHARPE]),
                staged_result.total_trials,
                int(sel_back_metrics[i, M_TRADES]),
            )
            for i in range(len(sel_back_metrics))
        ])
        comb_ranks = combined_rank(
            sel_back_metrics, sel_fwd_metrics,
            forward_weight=config.forward_weight,
        )

        # Sort by combined rank (lower = better)
        sort_order = np.argsort(comb_ranks)
    else:
        # No forward data — sort by back quality descending
        sort_order = np.argsort(-sel_back_metrics[:, M_QUALITY])
        dsrs = np.zeros(len(sel_back_metrics))

    # Step 6: Build Candidate objects
    for rank, orig_idx in enumerate(sort_order):
        params_dict = decode_params(spec, sel_value_matrix[orig_idx])

        back_metrics_dict = {
            "trades": float(sel_back_metrics[orig_idx, M_TRADES]),
            "quality_score": float(sel_back_metrics[orig_idx, M_QUALITY]),
            "sharpe": float(sel_back_metrics[orig_idx, M_SHARPE]),
        }

        candidate = Candidate(
            index=rank,
            params=params_dict,
            back_metrics=back_metrics_dict,
            dsr=float(dsrs[orig_idx]),
        )

        if sel_fwd_metrics is not None:
            candidate.forward_metrics = {
                "trades": float(sel_fwd_metrics[orig_idx, M_TRADES]),
                "quality_score": float(sel_fwd_metrics[orig_idx, M_QUALITY]),
                "sharpe": float(sel_fwd_metrics[orig_idx, M_SHARPE]),
            }
            back_q = back_metrics_dict.get("quality_score", 0)
            fwd_q = candidate.forward_metrics.get("quality_score", 0)
            candidate.forward_back_ratio = fwd_q / back_q if back_q > 0 else 0.0
            candidate.combined_rank = float(comb_ranks[orig_idx])

        result.candidates.append(candidate)

    logger.info(f"Multi-candidate selection: {len(result.candidates)} candidates")
    return True


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
    bar_hour_back: np.ndarray | None = None,
    bar_day_back: np.ndarray | None = None,
    bar_hour_fwd: np.ndarray | None = None,
    bar_day_fwd: np.ndarray | None = None,
    # M1 sub-bar arrays (optional)
    m1_back: dict[str, np.ndarray] | None = None,
    m1_fwd: dict[str, np.ndarray] | None = None,
    # Execution cost overrides (default to engine defaults)
    commission_pips: float | None = None,
    max_spread_pips: float | None = None,
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

    # Resolve cost params: explicit overrides > defaults from dtypes
    eff_commission = commission_pips if commission_pips is not None else DEFAULT_COMMISSION_PIPS
    eff_max_spread = max_spread_pips if max_spread_pips is not None else DEFAULT_MAX_SPREAD_PIPS

    # Memory check
    mem_mb = _estimate_memory_mb(
        len(high_back), 0, config.batch_size,
        len(list(strategy.param_space())),
    )
    logger.info(f"Estimated peak memory: {mem_mb:.1f} MB")

    # --- Stage 1: Build engine and run staged optimization ---
    logger.info("Building back-test engine...")
    m1_back_kwargs: dict = {}
    if m1_back is not None:
        for key in ("m1_high", "m1_low", "m1_close", "m1_spread",
                    "h1_to_m1_start", "h1_to_m1_end"):
            if key in m1_back:
                m1_back_kwargs[key] = m1_back[key]
    engine_back = BacktestEngine(
        strategy, open_back, high_back, low_back, close_back,
        volume_back, spread_back, pip_value, slippage_pips,
        config.max_trades_per_trial,
        commission_pips=eff_commission, max_spread_pips=eff_max_spread,
        bar_hour=bar_hour_back, bar_day_of_week=bar_day_back,
        **m1_back_kwargs,
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
    result = OptimizationResult()
    result.staged_result = staged_result
    result.total_trials = staged_result.total_trials
    spec = engine_back.encoding

    if staged_result.best_indices is not None:
        # Build forward engine if data provided
        engine_fwd = None
        if high_fwd is not None:
            m1_fwd_kwargs: dict = {}
            if m1_fwd is not None:
                for key in ("m1_high", "m1_low", "m1_close", "m1_spread",
                            "h1_to_m1_start", "h1_to_m1_end"):
                    if key in m1_fwd:
                        m1_fwd_kwargs[key] = m1_fwd[key]
            engine_fwd = BacktestEngine(
                strategy, open_fwd, high_fwd, low_fwd, close_fwd,
                volume_fwd, spread_fwd, pip_value, slippage_pips,
                config.max_trades_per_trial,
                commission_pips=eff_commission, max_spread_pips=eff_max_spread,
                bar_hour=bar_hour_fwd, bar_day_of_week=bar_day_fwd,
                **m1_fwd_kwargs,
            )

        # Try multi-candidate path first
        multi_ok = _add_multi_candidates(
            result, staged_result, spec, engine_fwd, config,
        )

        # Fallback to single-best if multi-candidate didn't produce results
        if not multi_ok:
            _add_single_best_candidate(
                result, staged_result, spec, engine_fwd, config,
            )

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
