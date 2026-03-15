"""Staged optimization orchestrator.

Optimizes one parameter group at a time, locking best values between stages.
Stage order is strategy-defined via optimization_stages(), NOT hard-coded.

Default stages: signal → time → risk → management → refinement
- Stages 1-3: Basic execution mode (SL/TP only, fast)
- Stage 4 (management): Full execution mode (all management features)
- Refinement: Full mode, all params active with narrowed ranges

The staged approach is our biggest leverage — it reduces the effective
search space exponentially at each stage.
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
from backtester.optimizer.progress import BatchProgress, StageComplete
from backtester.core.encoding import EncodingSpec, build_encoding_spec, indices_to_values
from backtester.core.engine import BacktestEngine
from backtester.optimizer.config import OptimizationConfig
from backtester.optimizer.prefilter import postfilter_results, prefilter_invalid_combos
from backtester.optimizer.sampler import (
    EDASampler, NeighborhoodSpec, RandomSampler, SobolSampler, build_neighborhood,
)
# CMAESSampler imported lazily inside _create_exploiter() to avoid hard
# dependency when not yet available (built by separate agent).

from backtester.strategies.base import Strategy

logger = logging.getLogger(__name__)


def _normalize_stages(
    raw_stages: list[str | tuple[str, list[str]]],
) -> list[tuple[str, list[str]]]:
    """Convert stage entries to uniform (name, groups) form.

    Accepts:
        - ``"signal"`` → ``("signal", ["signal"])``
        - ``("core_trade_profile", ["risk", "exit_trailing"])`` → as-is

    Returns:
        List of ``(stage_name, group_list)`` tuples.
    """
    normalized: list[tuple[str, list[str]]] = []
    for entry in raw_stages:
        if isinstance(entry, str):
            normalized.append((entry, [entry]))
        elif isinstance(entry, tuple) and len(entry) == 2:
            name, groups = entry
            normalized.append((name, list(groups)))
        else:
            raise ValueError(
                f"Invalid stage entry: {entry!r}. "
                "Expected str or (str, list[str])."
            )
    return normalized


def compute_stage_budgets(
    strategy: Strategy,
    config: OptimizationConfig,
) -> list[dict]:
    """Compute per-stage unique combo counts vs budget for reporting.

    Returns a list of dicts, one per stage (including refinement):
        [{"stage": "signal", "unique_combos": 75, "budget": 200000, "coverage": 2667.0}, ...]
    """
    ps = strategy.param_space()
    raw_stages = strategy.optimization_stages()
    spec = build_encoding_spec(ps)
    stages = _normalize_stages(raw_stages)

    results = []
    for stage_name, stage_groups in stages:
        # Collect indices from all groups in this (possibly composite) stage
        group_idxs: list[int] = []
        for g in stage_groups:
            group_idxs.extend(spec.group_indices(g))
        if not group_idxs:
            continue

        # Count unique combos = product of value counts for all params in group
        unique_combos = 1
        for idx in group_idxs:
            unique_combos *= len(spec.columns[idx].values)

        budget = config.trials_per_stage
        # Apply auto-cap (mirrors _run_stage logic)
        max_coverage = 10
        if unique_combos > 0 and budget > unique_combos * max_coverage:
            budget = unique_combos * max_coverage
        coverage = budget / unique_combos if unique_combos > 0 else 0.0

        results.append({
            "stage": stage_name,
            "unique_combos": unique_combos,
            "budget": budget,
            "coverage": coverage,
        })

    # Refinement: neighborhood-constrained combos
    radius = config.refinement_neighborhood_radius
    if radius > 0:
        ref_combos = 1
        for col in spec.columns:
            n_vals = len(col.values)
            # Neighborhood span: min(2*radius+1, n_vals)
            span = min(2 * radius + 1, n_vals)
            ref_combos *= span
    else:
        ref_combos = 1
        for col in spec.columns:
            ref_combos *= len(col.values)

    results.append({
        "stage": "refinement",
        "unique_combos": ref_combos,
        "budget": config.refinement_trials,
        "coverage": config.refinement_trials / ref_combos if ref_combos > 0 else 0.0,
    })

    return results


@dataclass
class StageResult:
    """Result of a single optimization stage."""
    stage_name: str
    best_indices: np.ndarray     # (P,) best parameter indices
    best_quality: float
    best_metrics: np.ndarray     # (NUM_METRICS,)
    trials_evaluated: int
    valid_count: int
    # When collect_all=True, all post-filter passing trials are stored
    all_passing_indices: np.ndarray | None = None   # (K, P)
    all_passing_metrics: np.ndarray | None = None   # (K, NUM_METRICS)


@dataclass
class StagedResult:
    """Full staged optimization result."""
    stages: list[StageResult] = field(default_factory=list)
    best_indices: np.ndarray | None = None
    best_quality: float = 0.0
    best_metrics: np.ndarray | None = None
    total_trials: int = 0
    # Refinement stage passing trials (for multi-candidate selection)
    refinement_indices: np.ndarray | None = None   # (K, P)
    refinement_metrics: np.ndarray | None = None   # (K, NUM_METRICS)


class StagedOptimizer:
    """Staged optimization: optimize one param group at a time."""

    def __init__(
        self,
        engine: BacktestEngine,
        config: OptimizationConfig | None = None,
        on_batch: Any = None,
        on_stage: Any = None,
    ):
        self.engine = engine
        self.config = config or OptimizationConfig()
        self.spec = engine.encoding
        self._on_batch = on_batch
        self._on_stage = on_stage

    def optimize(self) -> StagedResult:
        """Run staged optimization.

        Reads stage order from strategy.optimization_stages().
        Supports both simple string stages and composite (name, [groups]) stages.
        """
        strategy = self.engine.strategy
        raw_stages = strategy.optimization_stages()
        stages = _normalize_stages(raw_stages)

        result = StagedResult()

        # Start with all params unlocked
        locked = np.full(self.spec.num_params, -1, dtype=np.int64)

        # Build set of groups that require EXEC_FULL from management modules
        full_mode_groups: set[str] = set()
        if hasattr(strategy, "management_modules"):
            for mod in strategy.management_modules():
                if mod.requires_full_mode:
                    full_mode_groups.add(mod.group)
        else:
            full_mode_groups.add("management")  # legacy fallback

        for stage_idx, (stage_name, stage_groups) in enumerate(stages):
            # Determine execution mode: EXEC_FULL if ANY group requires it
            exec_mode = EXEC_BASIC
            for g in stage_groups:
                if g in full_mode_groups:
                    exec_mode = EXEC_FULL
                    break

            # Build mask: which params are active this stage (all groups combined)
            group_indices: list[int] = []
            for g in stage_groups:
                group_indices.extend(self.spec.group_indices(g))
            if not group_indices:
                logger.info(f"Stage '{stage_name}': no params in group(s) {stage_groups}, skipping")
                continue

            active_mask = np.zeros(self.spec.num_params, dtype=np.bool_)
            for idx in group_indices:
                active_mask[idx] = True

            t_stage = time.time()
            stage_result = self._run_stage(
                stage_name=stage_name,
                active_mask=active_mask,
                locked=locked.copy(),
                exec_mode=exec_mode,
                trials=self.config.trials_per_stage,
                stage_index=stage_idx,
                total_stages=len(stages) + 1,
            )

            # Lock best values from this stage (only if stage found passing candidates)
            if stage_result.best_quality > -np.inf:
                for idx in group_indices:
                    locked[idx] = stage_result.best_indices[idx]
            else:
                logger.warning(
                    f"Stage '{stage_name}': no passing candidates, "
                    f"leaving params unlocked for refinement"
                )

            result.stages.append(stage_result)
            result.total_trials += stage_result.trials_evaluated

            logger.info(
                f"Stage '{stage_name}': quality={stage_result.best_quality:.2f}, "
                f"valid={stage_result.valid_count}/{stage_result.trials_evaluated}"
            )

            if self._on_stage:
                self._on_stage(StageComplete(
                    stage_name=stage_name,
                    stage_index=stage_idx,
                    total_stages=len(stages) + 1,  # +1 for refinement
                    best_quality=stage_result.best_quality,
                    best_metrics={
                        "sharpe": float(stage_result.best_metrics[M_SHARPE]),
                        "trades": float(stage_result.best_metrics[M_TRADES]),
                        "quality": float(stage_result.best_quality),
                    },
                    trials_evaluated=stage_result.trials_evaluated,
                    valid_count=stage_result.valid_count,
                    elapsed_secs=time.time() - t_stage,
                ))

        # --- Cyclic passes: re-optimize interacting stages ---
        # Signal and composite stages have strong cross-stage interactions.
        # Re-optimizing them after all stages are locked captures interactions
        # that single-pass staged optimization misses.
        if self.config.max_cyclic_passes > 0:
            # Identify cyclic stages: "signal" and any composite stage
            # (one with multiple groups in its group list).
            cyclic_stage_entries: list[tuple[str, list[str], int]] = []
            for _stage_name, _stage_groups in stages:
                is_signal = _stage_name == "signal"
                is_composite = isinstance(_stage_groups, list) and len(_stage_groups) > 1
                if is_signal or is_composite:
                    # Determine exec_mode: EXEC_FULL if ANY group requires it
                    _exec_mode = EXEC_BASIC
                    for g in _stage_groups:
                        if g in full_mode_groups:
                            _exec_mode = EXEC_FULL
                            break
                    cyclic_stage_entries.append((_stage_name, _stage_groups, _exec_mode))

            if cyclic_stage_entries:
                # Track best quality across all stages so far for convergence check
                cyclic_best_quality = max(
                    (s.best_quality for s in result.stages if s.best_quality > -np.inf),
                    default=-np.inf,
                )

                for cycle in range(self.config.max_cyclic_passes):
                    prev_quality = cyclic_best_quality

                    for c_name, c_groups, c_exec_mode in cyclic_stage_entries:
                        # Collect group indices for this cyclic stage
                        group_idxs: list[int] = []
                        for g in c_groups:
                            group_idxs.extend(self.spec.group_indices(g))

                        if not group_idxs:
                            continue

                        # Unlock this stage's params for re-optimization
                        cyclic_locked = locked.copy()
                        for idx in group_idxs:
                            cyclic_locked[idx] = -1

                        # Reduced budget for cyclic stages
                        cyclic_budget = int(
                            self.config.trials_per_stage * self.config.cyclic_budget_fraction
                        )

                        # Build active mask
                        active_mask = np.zeros(self.spec.num_params, dtype=np.bool_)
                        for idx in group_idxs:
                            active_mask[idx] = True

                        t_cyclic = time.time()
                        cyclic_result = self._run_stage(
                            stage_name=f"{c_name}_cycle{cycle + 1}",
                            active_mask=active_mask,
                            locked=cyclic_locked,
                            exec_mode=c_exec_mode,
                            trials=cyclic_budget,
                            stage_index=len(stages) + cycle,
                            total_stages=len(stages) + 1 + self.config.max_cyclic_passes,
                        )

                        # Re-lock with new best if stage found anything
                        if cyclic_result.best_quality > -np.inf:
                            for idx in group_idxs:
                                locked[idx] = cyclic_result.best_indices[idx]
                            if cyclic_result.best_quality > cyclic_best_quality:
                                cyclic_best_quality = cyclic_result.best_quality

                        result.stages.append(cyclic_result)
                        result.total_trials += cyclic_result.trials_evaluated

                        logger.info(
                            f"Cyclic pass {cycle + 1}, stage '{c_name}': "
                            f"quality={cyclic_result.best_quality:.2f}, "
                            f"valid={cyclic_result.valid_count}/{cyclic_result.trials_evaluated}"
                        )

                        if self._on_stage:
                            self._on_stage(StageComplete(
                                stage_name=f"{c_name}_cycle{cycle + 1}",
                                stage_index=len(stages) + cycle,
                                total_stages=len(stages) + 1 + self.config.max_cyclic_passes,
                                best_quality=cyclic_result.best_quality,
                                best_metrics={
                                    "sharpe": float(cyclic_result.best_metrics[M_SHARPE]),
                                    "trades": float(cyclic_result.best_metrics[M_TRADES]),
                                    "quality": float(cyclic_result.best_quality),
                                },
                                trials_evaluated=cyclic_result.trials_evaluated,
                                valid_count=cyclic_result.valid_count,
                                elapsed_secs=time.time() - t_cyclic,
                            ))

                    # Convergence check
                    if prev_quality > 0:
                        improvement = (cyclic_best_quality - prev_quality) / abs(prev_quality)
                    else:
                        improvement = 1.0  # Always do at least one pass if quality was 0

                    logger.info(
                        f"Cyclic pass {cycle + 1} improvement: {improvement:.4f} "
                        f"(threshold: {self.config.cyclic_improvement_threshold})"
                    )

                    if improvement < self.config.cyclic_improvement_threshold:
                        logger.info(
                            f"Cyclic passes converged after {cycle + 1} passes"
                        )
                        break  # Converged

        # --- Refinement stage: all params active, full mode ---
        t_stage = time.time()
        refinement_result = self._run_stage(
            stage_name="refinement",
            active_mask=np.ones(self.spec.num_params, dtype=np.bool_),
            locked=locked.copy(),  # Start from locked best, but allow variation
            exec_mode=EXEC_FULL,
            trials=self.config.refinement_trials,
            use_locked_as_center=True,
            neighborhood_radius=self.config.refinement_neighborhood_radius,
            collect_all=True,  # Collect all passing for multi-candidate selection
            stage_index=len(stages),
            total_stages=len(stages) + 1,
        )
        result.stages.append(refinement_result)
        result.total_trials += refinement_result.trials_evaluated

        # Set overall best from refinement (the final stage with all params active).
        # Earlier stages use partial param sets, so their quality scores aren't
        # directly comparable to refinement which evaluates the full configuration.
        result.best_indices = refinement_result.best_indices
        result.best_quality = refinement_result.best_quality
        result.best_metrics = refinement_result.best_metrics

        # Copy refinement passing data for multi-candidate selection
        result.refinement_indices = refinement_result.all_passing_indices
        result.refinement_metrics = refinement_result.all_passing_metrics

        if self._on_stage:
            self._on_stage(StageComplete(
                stage_name="refinement",
                stage_index=len(stages),
                total_stages=len(stages) + 1,
                best_quality=refinement_result.best_quality,
                best_metrics={
                    "sharpe": float(refinement_result.best_metrics[M_SHARPE]),
                    "trades": float(refinement_result.best_metrics[M_TRADES]),
                    "quality": float(refinement_result.best_quality),
                },
                trials_evaluated=refinement_result.trials_evaluated,
                valid_count=refinement_result.valid_count,
                elapsed_secs=time.time() - t_stage,
            ))

        return result

    def _create_exploiter(self, batch_size: int) -> tuple:
        """Create the exploitation sampler based on config.

        Returns:
            (exploiter, use_cmaes): The sampler instance and whether it's CMA-ES.
        """
        if self.config.exploitation_method == "cmaes":
            try:
                from backtester.optimizer.sampler import CMAESSampler
                pop_size = self.config.cmaes_population_size or batch_size
                exploiter = CMAESSampler(
                    self.spec,
                    sigma0=self.config.cmaes_sigma0,
                    population_size=pop_size,
                    seed=self.config.seed,
                )
                logger.info(
                    f"Using CMA-ES exploitation (sigma0={self.config.cmaes_sigma0}, "
                    f"pop_size={pop_size})"
                )
                return exploiter, True
            except ImportError:
                logger.warning(
                    "CMAESSampler not available, falling back to EDA exploitation"
                )
                # Fall through to EDA

        exploiter = EDASampler(
            self.spec,
            learning_rate=self.config.eda_learning_rate,
            lr_decay=self.config.eda_lr_decay,
            lr_floor=self.config.eda_lr_floor,
            min_prob=self.config.eda_min_prob,
            seed=self.config.seed,
        )
        logger.info("Using EDA exploitation")
        return exploiter, False

    def _run_stage(
        self,
        stage_name: str,
        active_mask: np.ndarray,
        locked: np.ndarray,
        exec_mode: int,
        trials: int,
        use_locked_as_center: bool = False,
        neighborhood_radius: int = 0,
        collect_all: bool = False,
        stage_index: int = 0,
        total_stages: int = 1,
    ) -> StageResult:
        """Run a single optimization stage.

        Args:
            collect_all: When True, accumulate all post-filter passing
                trials (indices + metrics) for multi-candidate selection.
            neighborhood_radius: When >0 and use_locked_as_center=True,
                constrain sampling to ±radius index steps around locked values
                instead of unlocking to full range.
        """
        # For refinement with neighborhood, build constrained bounds instead of
        # unlocking to full range. For normal stages, no neighborhood.
        neighborhood: NeighborhoodSpec | None = None
        stage_locked = locked.copy()

        if use_locked_as_center and neighborhood_radius > 0:
            # Build neighborhood bounds — keeps params locked but constrains range
            neighborhood = build_neighborhood(
                self.spec, locked, neighborhood_radius,
            )
            # Unlock active params so samplers will sample within neighborhood
            for i in range(self.spec.num_params):
                if active_mask[i]:
                    stage_locked[i] = -1
        elif use_locked_as_center:
            # Legacy fallback: unlock to full range (no neighborhood)
            for i in range(self.spec.num_params):
                if active_mask[i]:
                    stage_locked[i] = -1

        # Compute unique combos AFTER stage_locked/neighborhood are set (Fix 5)
        active_indices = [i for i in range(self.spec.num_params) if active_mask[i]]
        if neighborhood is not None:
            unique_combos = 1
            for idx in active_indices:
                if stage_locked[idx] == -1:
                    span = int(neighborhood.max_bounds[idx]) - int(neighborhood.min_bounds[idx]) + 1
                    unique_combos *= span
        else:
            unique_combos = 1
            for idx in active_indices:
                n_vals = len(self.spec.columns[idx].values)
                if stage_locked[idx] == -1:  # Only count unlocked params
                    unique_combos *= n_vals

        # Fix 2: Auto-cap budget for non-refinement stages when space is small
        max_coverage = 10
        if not use_locked_as_center and unique_combos > 0 and trials > unique_combos * max_coverage:
            capped_trials = unique_combos * max_coverage
            logger.info(
                f"Stage '{stage_name}': budget capped {trials:,} -> {capped_trials:,} "
                f"({unique_combos:,} combos x {max_coverage}x)"
            )
            trials = capped_trials

        coverage = trials / unique_combos if unique_combos > 0 else 0.0
        logger.info(
            f"Stage '{stage_name}': {unique_combos:,} unique combos, "
            f"{trials:,} budget ({coverage:,.1f}x coverage)"
        )

        t_stage_start = time.time()
        batch_size = self.config.batch_size
        exploration_budget = int(trials * self.config.exploration_pct)

        # Initialize samplers
        sobol = SobolSampler(self.spec, seed=self.config.seed)
        exploiter, use_cmaes = self._create_exploiter(batch_size)

        best_quality = -np.inf
        best_indices = np.zeros(self.spec.num_params, dtype=np.int64)
        best_metrics = np.zeros(NUM_METRICS, dtype=np.float64)
        # Track best trial regardless of gates (for "best of bad bunch" fallback)
        ungated_best_quality = -np.inf
        ungated_best_indices = np.zeros(self.spec.num_params, dtype=np.int64)
        ungated_best_metrics = np.zeros(NUM_METRICS, dtype=np.float64)
        total_evaluated = 0
        total_valid = 0

        # Accumulators for collect_all mode
        all_indices_list: list[np.ndarray] = []
        all_metrics_list: list[np.ndarray] = []

        # Compute once — doesn't change per batch
        n_years = self.engine.n_bars / self.engine.bars_per_year

        while total_evaluated < trials:
            remaining = trials - total_evaluated
            n = min(batch_size, remaining)

            # Choose sampler: exploration phase uses Sobol, exploitation uses exploiter
            if total_evaluated < exploration_budget:
                index_batch = sobol.sample(
                    n, mask=active_mask, locked=stage_locked, neighborhood=neighborhood,
                )
            else:
                index_batch = exploiter.sample(
                    n, mask=active_mask, locked=stage_locked, neighborhood=neighborhood,
                )

            # Pre-filter invalid combinations
            valid_pre = prefilter_invalid_combos(index_batch, self.spec)
            # Keep only valid rows
            valid_indices = np.where(valid_pre)[0]
            if len(valid_indices) == 0:
                total_evaluated += n
                continue

            valid_batch = index_batch[valid_indices]

            # Convert to value space and evaluate
            value_matrix = indices_to_values(self.spec, valid_batch)
            metrics = self.engine.evaluate_batch(value_matrix, exec_mode)

            # Track ungated best (before post-filter) for fallback
            batch_qualities = metrics[:, M_QUALITY]
            batch_best_idx = int(np.argmax(batch_qualities))
            if batch_qualities[batch_best_idx] > ungated_best_quality:
                ungated_best_quality = float(batch_qualities[batch_best_idx])
                ungated_best_indices = valid_batch[batch_best_idx].copy()
                ungated_best_metrics = metrics[batch_best_idx].copy()

            # Post-filter
            valid_post = postfilter_results(
                metrics,
                min_trades_per_year=self.config.min_trades_per_year,
                min_total_trades=self.config.min_total_trades,
                n_years=n_years,
                max_dd_pct=self.config.max_dd_pct,
                min_r_squared=self.config.min_r_squared,
            )

            passing = np.where(valid_post)[0]
            total_valid += len(passing)
            total_evaluated += n

            if len(passing) > 0:
                # Update best
                qualities = metrics[passing, M_QUALITY]
                best_in_batch = passing[np.argmax(qualities)]
                if metrics[best_in_batch, M_QUALITY] > best_quality:
                    best_quality = float(metrics[best_in_batch, M_QUALITY])
                    best_indices = valid_batch[best_in_batch].copy()
                    best_metrics = metrics[best_in_batch].copy()

                # Collect all passing for multi-candidate selection
                if collect_all:
                    all_indices_list.append(valid_batch[passing].copy())
                    all_metrics_list.append(metrics[passing].copy())

            # Update exploiter (outside passing check — CMA-ES needs all results)
            if total_evaluated >= exploration_budget:
                if use_cmaes:
                    # CMA-ES update: pass full batch + quality scores
                    # Quality 0.0 for filtered-out rows (post-filter failures)
                    quality_scores = np.zeros(len(valid_batch), dtype=np.float64)
                    if len(passing) > 0:
                        quality_scores[passing] = metrics[passing, M_QUALITY]
                    exploiter.update(valid_batch, quality_scores, mask=active_mask)

                    # Check convergence — if converged, do IPOP restart
                    if exploiter.converged:
                        logger.info(
                            f"Stage '{stage_name}': CMA-ES converged, "
                            f"performing IPOP restart"
                        )
                        exploiter.reset()

                    # Log diagnostics
                    ent = exploiter.entropy(mask=active_mask)
                    logger.debug(
                        f"Stage '{stage_name}' CMA-ES update: entropy={ent:.3f}"
                    )
                elif len(passing) > 0:
                    # EDA update: pass elite subset only
                    qualities = metrics[passing, M_QUALITY]
                    n_elite = max(1, int(len(passing) * self.config.elite_pct))
                    elite_order = np.argsort(-qualities)[:n_elite]
                    elite_indices_batch = valid_batch[passing[elite_order]]
                    exploiter.update(elite_indices_batch, mask=active_mask)

                    # Log entropy diagnostics
                    ent = exploiter.entropy(mask=active_mask)
                    logger.debug(
                        f"Stage '{stage_name}' EDA update #{exploiter.update_count}: "
                        f"lr={exploiter.effective_lr:.3f}, entropy={ent:.3f}"
                    )

            # Fire progress callback AFTER best update so dashboard sees latest
            if self._on_batch:
                elapsed = time.time() - t_stage_start
                evals_per_sec = total_evaluated / elapsed if elapsed > 0 else 0

                # Determine phase
                phase = "exploration" if total_evaluated < exploration_budget else "exploitation"

                # Get entropy if in exploitation phase
                ent = None
                lr = None
                if phase == "exploitation":
                    ent = float(exploiter.entropy(mask=active_mask))
                    lr = float(exploiter.effective_lr) if hasattr(exploiter, 'effective_lr') else None

                # Batch quality stats (cap to prevent inf/garbage from
                # combos with 1-2 trades producing infinite Sharpe/PF)
                batch_qualities_all = metrics[:, M_QUALITY]
                batch_valid_mask = (batch_qualities_all > -1e9) & (batch_qualities_all < 1e6)
                batch_mean_q = float(np.mean(batch_qualities_all[batch_valid_mask])) if batch_valid_mask.any() else 0.0
                batch_best_q = float(np.max(batch_qualities_all[batch_valid_mask])) if batch_valid_mask.any() else 0.0

                # Prefer gated best (post-filtered). If no gated trials pass,
                # fall back to ungated best so the dashboard shows quality
                # progression during early stages (where post-filter is strict
                # because non-active params are random).
                if best_quality > -np.inf:
                    live_best_q = float(best_quality)
                    live_best_sharpe = float(best_metrics[M_SHARPE])
                    live_best_trades = int(best_metrics[M_TRADES])
                elif ungated_best_quality > -np.inf:
                    live_best_q = float(ungated_best_quality)
                    live_best_sharpe = float(ungated_best_metrics[M_SHARPE])
                    live_best_trades = int(ungated_best_metrics[M_TRADES])
                else:
                    live_best_q = 0.0
                    live_best_sharpe = 0.0
                    live_best_trades = 0

                self._on_batch(BatchProgress(
                    stage_name=stage_name,
                    stage_index=stage_index,
                    total_stages=total_stages,
                    trials_done=total_evaluated,
                    trials_total=trials,
                    best_quality=live_best_q,
                    best_sharpe=live_best_sharpe,
                    best_trades=live_best_trades,
                    valid_count=total_valid,
                    valid_rate=total_valid / total_evaluated if total_evaluated > 0 else 0.0,
                    batch_best_quality=batch_best_q,
                    batch_mean_quality=batch_mean_q,
                    phase=phase,
                    entropy=ent,
                    effective_lr=lr,
                    evals_per_sec=evals_per_sec,
                    elapsed_secs=elapsed,
                ))

        # Build collected arrays
        all_passing_indices = None
        all_passing_metrics = None
        if collect_all and all_indices_list:
            all_passing_indices = np.vstack(all_indices_list)
            all_passing_metrics = np.vstack(all_metrics_list)

        # Fallback: if no gated trials passed, use ungated best ("best of bad bunch")
        if best_quality == -np.inf and ungated_best_quality > -np.inf:
            best_quality = ungated_best_quality
            best_indices = ungated_best_indices
            best_metrics = ungated_best_metrics
            logger.info(
                f"Stage '{stage_name}': using ungated best "
                f"(quality={ungated_best_quality:.4f}, "
                f"trades={int(ungated_best_metrics[M_TRADES])})"
            )

        return StageResult(
            stage_name=stage_name,
            best_indices=best_indices,
            best_quality=best_quality,
            best_metrics=best_metrics,
            trials_evaluated=total_evaluated,
            valid_count=total_valid,
            all_passing_indices=all_passing_indices,
            all_passing_metrics=all_passing_metrics,
        )
