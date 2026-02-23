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
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backtester.core.dtypes import (
    EXEC_BASIC,
    EXEC_FULL,
    M_QUALITY,
    M_TRADES,
    NUM_METRICS,
)
from backtester.core.encoding import EncodingSpec, indices_to_values
from backtester.core.engine import BacktestEngine
from backtester.optimizer.config import OptimizationConfig
from backtester.optimizer.prefilter import postfilter_results, prefilter_invalid_combos
from backtester.optimizer.sampler import EDASampler, RandomSampler, SobolSampler

logger = logging.getLogger(__name__)


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
    ):
        self.engine = engine
        self.config = config or OptimizationConfig()
        self.spec = engine.encoding

    def optimize(self) -> StagedResult:
        """Run staged optimization.

        Reads stage order from strategy.optimization_stages().
        """
        strategy = self.engine.strategy
        stages = strategy.optimization_stages()

        result = StagedResult()

        # Start with all params unlocked
        locked = np.full(self.spec.num_params, -1, dtype=np.int64)

        for stage_idx, stage_name in enumerate(stages):
            # Determine execution mode
            exec_mode = EXEC_BASIC
            if stage_name == "management":
                exec_mode = EXEC_FULL

            # Build mask: which params are active this stage
            group_indices = self.spec.group_indices(stage_name)
            if not group_indices:
                logger.info(f"Stage '{stage_name}': no params in group, skipping")
                continue

            active_mask = np.zeros(self.spec.num_params, dtype=np.bool_)
            for idx in group_indices:
                active_mask[idx] = True

            stage_result = self._run_stage(
                stage_name=stage_name,
                active_mask=active_mask,
                locked=locked.copy(),
                exec_mode=exec_mode,
                trials=self.config.trials_per_stage,
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

        # --- Refinement stage: all params active, full mode ---
        refinement_result = self._run_stage(
            stage_name="refinement",
            active_mask=np.ones(self.spec.num_params, dtype=np.bool_),
            locked=locked.copy(),  # Start from locked best, but allow variation
            exec_mode=EXEC_FULL,
            trials=self.config.refinement_trials,
            use_locked_as_center=True,
            collect_all=True,  # Collect all passing for multi-candidate selection
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

        return result

    def _run_stage(
        self,
        stage_name: str,
        active_mask: np.ndarray,
        locked: np.ndarray,
        exec_mode: int,
        trials: int,
        use_locked_as_center: bool = False,
        collect_all: bool = False,
    ) -> StageResult:
        """Run a single optimization stage.

        Args:
            collect_all: When True, accumulate all post-filter passing
                trials (indices + metrics) for multi-candidate selection.
        """
        batch_size = self.config.batch_size
        exploration_budget = int(trials * self.config.exploration_pct)

        # Initialize samplers
        sobol = SobolSampler(self.spec, seed=self.config.seed)
        eda = EDASampler(
            self.spec,
            learning_rate=self.config.eda_learning_rate,
            min_prob=self.config.eda_min_prob,
            seed=self.config.seed,
        )

        best_quality = -np.inf
        best_indices = np.zeros(self.spec.num_params, dtype=np.int64)
        best_metrics = np.zeros(NUM_METRICS, dtype=np.float64)
        total_evaluated = 0
        total_valid = 0

        # Accumulators for collect_all mode
        all_indices_list: list[np.ndarray] = []
        all_metrics_list: list[np.ndarray] = []

        # For refinement, unlock all but use locked values as starting point
        stage_locked = locked.copy()
        if use_locked_as_center:
            # Unlock active params but keep locked values as reference
            for i in range(self.spec.num_params):
                if active_mask[i]:
                    stage_locked[i] = -1  # Unlock for sampling

        while total_evaluated < trials:
            remaining = trials - total_evaluated
            n = min(batch_size, remaining)

            # Choose sampler: exploration phase uses Sobol, exploitation uses EDA
            if total_evaluated < exploration_budget:
                index_batch = sobol.sample(n, mask=active_mask, locked=stage_locked)
            else:
                index_batch = eda.sample(n, mask=active_mask, locked=stage_locked)

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

            # Post-filter
            valid_post = postfilter_results(
                metrics,
                min_trades=self.config.min_trades,
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

                # Update EDA with elite subset
                n_elite = max(1, int(len(passing) * self.config.elite_pct))
                elite_order = np.argsort(-qualities)[:n_elite]
                elite_indices = valid_batch[passing[elite_order]]
                eda.update(elite_indices, mask=active_mask)

        # Build collected arrays
        all_passing_indices = None
        all_passing_metrics = None
        if collect_all and all_indices_list:
            all_passing_indices = np.vstack(all_indices_list)
            all_passing_metrics = np.vstack(all_metrics_list)

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
