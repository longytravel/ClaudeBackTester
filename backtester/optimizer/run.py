"""Top-level optimization entry point.

Usage:
    from backtester.optimizer.run import optimize
    result = optimize(strategy, data_back, data_forward, config)
"""

from __future__ import annotations

import logging
import time
import gc
from dataclasses import dataclass, field
from typing import Any, Callable

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
from backtester.optimizer.archive import DiversityArchive  # Used for optimizer search diversity
from backtester.optimizer.config import OptimizationConfig, get_preset
from backtester.optimizer.prefilter import postfilter_results
from backtester.optimizer.ranking import (
    deflated_sharpe_ratio,
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
    forward_gate_passed: bool = False


@dataclass
class OptimizationResult:
    """Full optimization result."""
    candidates: list[Candidate] = field(default_factory=list)
    staged_result: StagedResult | None = None
    total_trials: int = 0
    elapsed_seconds: float = 0.0
    evals_per_second: float = 0.0
    optimizer_funnel: dict = field(default_factory=dict)


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
    result.optimizer_funnel["sent_to_pipeline"] = 1
    result.optimizer_funnel.setdefault("dsr_surviving", 0)
    result.optimizer_funnel.setdefault("dedup_groups", 0)
    result.optimizer_funnel.setdefault("after_dedup", 0)
    result.optimizer_funnel.setdefault("pipeline_candidates", 1)
    result.optimizer_funnel.setdefault("forward_tested", 0)
    logger.info("Single-best candidate selected (fallback)")


def _build_dedup_key(spec: Any, index_row: np.ndarray) -> tuple:
    """Build deduplication key from signal params + sl_mode + tp_mode.

    Candidates with the same key produce identical trade entries (they differ
    only in management params like trailing/breakeven/partial close).
    """
    key_parts = []
    for col in spec.columns:
        if col.group == "signal":
            key_parts.append(int(index_row[col.index]))
        elif col.name in ("sl_mode", "tp_mode"):
            key_parts.append(int(index_row[col.index]))
    return tuple(key_parts)


def _select_pipeline_candidates(
    result: OptimizationResult,
    staged_result: StagedResult,
    spec: Any,
    engine_fwd: BacktestEngine | None,
    config: OptimizationConfig,
) -> bool:
    """Select candidates for validation pipeline using DSR prefilter + dedup.

    Flow:
    1. DSR prefilter on IS Sharpe (statistical significance gate)
    2. Group by signal+risk params, keep top K per group (deduplication)
    3. Take top N by IS quality (pipeline capacity limit)
    4. Forward-test for reporting only (NOT for selection/elimination)

    Returns True if at least one candidate was added, False otherwise.
    """
    ref_indices = staged_result.refinement_indices
    ref_metrics = staged_result.refinement_metrics

    if ref_indices is None or ref_metrics is None or len(ref_indices) == 0:
        logger.info("No refinement passing trials for candidate selection")
        result.optimizer_funnel["refinement_passing"] = 0
        return False

    n_passing = len(ref_indices)
    result.optimizer_funnel["refinement_passing"] = n_passing
    logger.info(f"Candidate selection: {n_passing:,} refinement passing trials")

    # --- Step 1: DSR prefilter on IS data ---
    dsr_threshold = config.dsr_prefilter_threshold
    dsr_values = np.array([
        deflated_sharpe_ratio(
            float(ref_metrics[i, M_SHARPE]),
            staged_result.total_trials,
            int(ref_metrics[i, M_TRADES]),
        )
        for i in range(n_passing)
    ])
    dsr_mask = dsr_values >= dsr_threshold
    n_dsr = int(dsr_mask.sum())

    if n_dsr == 0:
        # Fallback to relaxed threshold
        dsr_threshold = config.dsr_prefilter_fallback
        dsr_mask = dsr_values >= dsr_threshold
        n_dsr = int(dsr_mask.sum())
        if n_dsr > 0:
            logger.warning(
                f"DSR prefilter: 0 passed at {config.dsr_prefilter_threshold}, "
                f"relaxed to {dsr_threshold} → {n_dsr:,} passed"
            )
        else:
            logger.warning(
                f"DSR prefilter: 0 passed even at fallback {dsr_threshold}. "
                "Skipping DSR filter, using top candidates by quality."
            )
            # Use all passing trials, sorted by quality
            dsr_mask = np.ones(n_passing, dtype=np.bool_)
            n_dsr = n_passing

    logger.info(f"DSR prefilter: {n_dsr:,}/{n_passing:,} passed (threshold={dsr_threshold})")
    result.optimizer_funnel["dsr_surviving"] = n_dsr

    # Get DSR-surviving subset
    dsr_indices = np.where(dsr_mask)[0]
    surviving_ref_indices = ref_indices[dsr_indices]
    surviving_ref_metrics = ref_metrics[dsr_indices]
    surviving_dsr_values = dsr_values[dsr_indices]

    # --- Step 2: Signal+risk param deduplication ---
    groups: dict[tuple, list[int]] = {}
    for local_idx in range(len(surviving_ref_indices)):
        key = _build_dedup_key(spec, surviving_ref_indices[local_idx])
        groups.setdefault(key, []).append(local_idx)

    result.optimizer_funnel["dedup_groups"] = len(groups)
    logger.info(
        f"Dedup: {len(surviving_ref_indices):,} trials → "
        f"{len(groups):,} unique signal+risk groups"
    )

    # Keep top K per group by IS quality
    max_per_group = config.max_per_dedup_group
    deduped_local: list[int] = []
    for group_members in groups.values():
        # Sort by quality descending within group
        group_members.sort(
            key=lambda idx: -float(surviving_ref_metrics[idx, M_QUALITY])
        )
        deduped_local.extend(group_members[:max_per_group])

    result.optimizer_funnel["after_dedup"] = len(deduped_local)
    logger.info(
        f"Dedup: kept top {max_per_group} per group → "
        f"{len(deduped_local):,} candidates"
    )

    # --- Step 3: Top N by IS quality ---
    max_n = config.max_pipeline_candidates
    # Sort all deduped candidates by quality descending
    deduped_local.sort(
        key=lambda idx: -float(surviving_ref_metrics[idx, M_QUALITY])
    )
    selected_local = deduped_local[:max_n]

    logger.info(
        f"Pipeline candidates: top {len(selected_local)} "
        f"(max {max_n}) by IS quality"
    )
    result.optimizer_funnel["pipeline_candidates"] = len(selected_local)

    # Build selected arrays
    sel_back_indices = surviving_ref_indices[selected_local]
    sel_back_metrics = surviving_ref_metrics[selected_local]
    sel_dsr_values = surviving_dsr_values[selected_local]

    # Convert to value space
    sel_value_matrix = indices_to_values(spec, sel_back_indices)

    # --- Step 4: Forward-test for reporting only (NOT for selection) ---
    sel_fwd_metrics = None
    if engine_fwd is not None:
        logger.info(f"Forward-testing {len(sel_value_matrix)} candidates (reporting only)...")
        sel_fwd_metrics = engine_fwd.evaluate_batch(sel_value_matrix, EXEC_FULL)
        result.optimizer_funnel["forward_tested"] = len(sel_value_matrix)
    else:
        result.optimizer_funnel["forward_tested"] = 0

    # --- Step 5: Build Candidate objects sorted by IS quality ---
    sort_order = np.argsort(-sel_back_metrics[:, M_QUALITY])

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
            dsr=float(sel_dsr_values[orig_idx]),
            forward_gate_passed=True,  # No longer used as gate
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

        result.candidates.append(candidate)

    result.optimizer_funnel["sent_to_pipeline"] = len(result.candidates)
    logger.info(f"Candidate selection complete: {len(result.candidates)} candidates")
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
    # Progress callbacks
    on_batch: Callable | None = None,
    on_stage: Callable | None = None,
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

    # Build CV objective if enabled
    cv_objective = None
    if config.use_cv_objective:
        from backtester.optimizer.cv_objective import CVObjective, auto_configure_folds
        n_years = len(high_back) / engine_back.bars_per_year
        fold_config = auto_configure_folds(
            n_bars=len(high_back),
            bars_per_year=engine_back.bars_per_year,
            timeframe=config.timeframe,
            expected_trades_per_year=engine_back.n_signals / n_years if n_years > 0 else 0,
            embargo_days=config.cv_embargo_days,
            min_trades_per_fold=config.cv_min_trades_per_fold,
            aggregation=config.cv_aggregation,
            aggregation_lambda=config.cv_lambda,
            early_stopping=config.cv_early_stopping,
            n_folds_override=config.cv_n_folds,
        )
        cv_objective = CVObjective(engine_back, fold_config)
        logger.info(f"CV objective enabled: {fold_config.n_folds} folds, {config.cv_aggregation}")

    logger.info("Running staged optimization...")
    staged = StagedOptimizer(engine_back, config, on_batch=on_batch, on_stage=on_stage,
                             cv_objective=cv_objective)
    staged_result = staged.optimize()
    logger.info(
        f"Staged optimization complete: {staged_result.total_trials} trials, "
        f"best quality={staged_result.best_quality:.2f}"
    )

    # --- Stage 2: Build final candidate list ---
    result = OptimizationResult()
    result.staged_result = staged_result
    result.total_trials = staged_result.total_trials
    result.optimizer_funnel["total_trials"] = staged_result.total_trials
    spec = engine_back.encoding

    # Free back-test engine BEFORE creating forward engine to keep peak RSS bounded.
    del staged   # releases staged.engine reference to engine_back
    del engine_back
    gc.collect()

    engine_fwd: BacktestEngine | None = None
    if staged_result.best_indices is not None:
        # Build forward engine if data provided
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

        # Select candidates via DSR prefilter + dedup + top N
        multi_ok = _select_pipeline_candidates(
            result, staged_result, spec, engine_fwd, config,
        )

        # Fallback to single-best if multi-candidate didn't produce results
        if not multi_ok:
            _add_single_best_candidate(
                result, staged_result, spec, engine_fwd, config,
            )

    # Free forward engine + Numba-allocated buffers before returning.
    if engine_fwd is not None:
        del engine_fwd
    gc.collect()

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
