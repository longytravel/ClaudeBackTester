"""Pipeline runner — orchestrates the full validation pipeline.

Stages:
1. Data loading (wraps existing data pipeline)
2. Optimization (wraps optimizer/run.py)
3. Walk-forward validation
4. Parameter stability
5. Monte Carlo simulation
6. Confidence scoring
7. Report generation (JSON output for MVP)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any

import numpy as np

from backtester.pipeline.checkpoint import load_checkpoint, save_checkpoint
from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.types import (
    CandidateResult,
    PipelineState,
    Rating,
)

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the multi-stage validation pipeline.

    Supports checkpoint/resume: saves state after each stage,
    can resume from any completed stage.
    """

    def __init__(
        self,
        strategy: Any,  # Strategy instance
        data_arrays: dict[str, np.ndarray],
        config: PipelineConfig | None = None,
        pair: str = "",
        timeframe: str = "",
        pip_value: float = 0.0001,
        slippage_pips: float = 0.5,
        output_dir: str | None = None,
    ):
        self.strategy = strategy
        self.data = data_arrays
        self.config = config or PipelineConfig()
        self.pair = pair
        self.timeframe = timeframe
        self.pip_value = pip_value
        self.slippage_pips = slippage_pips
        self.output_dir = output_dir or self.config.output_dir

        self.state = PipelineState(
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            pair=pair,
            timeframe=timeframe,
            config_dict=asdict(self.config),
            seed=self.config.seed,
        )

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, "checkpoint.json")

    def run(
        self,
        candidates: list[dict[str, Any]] | None = None,
        candidate_results: list[CandidateResult] | None = None,
        resume_from: int | None = None,
        stop_after: int | None = None,
    ) -> PipelineState:
        """Run the pipeline, optionally resuming from a checkpoint.

        Args:
            candidates: List of param dicts from optimizer (for stage 3+).
            candidate_results: Pre-built CandidateResult objects (alternative to candidates).
            resume_from: Stage number to resume from (loads checkpoint).
            stop_after: Stage number to stop after.

        Returns:
            PipelineState with all results.
        """
        t0 = time.time()

        # Resume from checkpoint if requested
        if resume_from is not None:
            if os.path.exists(self.checkpoint_path):
                self.state = load_checkpoint(self.checkpoint_path)
                logger.info(f"Resumed from stage {resume_from}")
            else:
                logger.warning(
                    f"No checkpoint at {self.checkpoint_path}, starting fresh"
                )

        # Initialize candidates if provided
        if candidate_results is not None:
            self.state.candidates = candidate_results
        elif candidates is not None and not self.state.candidates:
            self.state.candidates = [
                CandidateResult(candidate_index=i, params=p)
                for i, p in enumerate(candidates)
            ]

        # Run stages
        stages = [
            (3, "Walk-Forward", self._run_walk_forward),
            (4, "Stability", self._run_stability),
            (5, "Monte Carlo", self._run_monte_carlo),
            (6, "Confidence", self._run_confidence),
            (7, "Report", self._run_report),
        ]

        for stage_num, stage_name, stage_fn in stages:
            if resume_from is not None and stage_num < resume_from:
                continue
            if stop_after is not None and stage_num > stop_after:
                break
            if stage_num in self.state.completed_stages:
                logger.info(f"Stage {stage_num} ({stage_name}) already completed, skipping")
                continue

            logger.info(f"--- Stage {stage_num}: {stage_name} ---")
            self.state.current_stage = stage_num

            stage_fn()

            self.state.completed_stages.append(stage_num)
            if self.config.checkpoint_enabled:
                save_checkpoint(self.state, self.checkpoint_path)

            # Log survivors
            active = [c for c in self.state.candidates if not c.eliminated]
            logger.info(
                f"Stage {stage_num} complete: "
                f"{len(active)}/{len(self.state.candidates)} candidates surviving"
            )

        elapsed = time.time() - t0
        logger.info(f"Pipeline complete in {elapsed:.1f}s")
        return self.state

    def _run_walk_forward(self) -> None:
        """Stage 3: Walk-forward validation + optional CPCV sub-step."""
        from backtester.pipeline.walk_forward import walk_forward_validate

        active = [c for c in self.state.candidates if not c.eliminated]
        if not active:
            logger.warning("No active candidates for walk-forward")
            return

        # Determine optimization range (back-test portion = first 80%)
        n_bars = len(self.data["close"])
        opt_end = int(n_bars * 0.8)

        param_dicts = [c.params for c in active]
        results = walk_forward_validate(
            self.strategy, param_dicts, self.data,
            opt_start=0, opt_end=opt_end, config=self.config,
            pip_value=self.pip_value, slippage_pips=self.slippage_pips,
        )

        for candidate, wf_result in zip(active, results):
            candidate.walk_forward = wf_result
            if not wf_result.passed_gate:
                candidate.eliminated = True
                candidate.eliminated_at_stage = "walk_forward"
                candidate.elimination_reason = (
                    f"pass_rate={wf_result.pass_rate:.2f}, "
                    f"mean_sharpe={wf_result.mean_sharpe:.2f}"
                )

        # Run CPCV sub-step on survivors if enabled
        if self.config.cpcv_enabled:
            self._run_cpcv()

    def _run_cpcv(self) -> None:
        """Stage 3b: CPCV sub-step (runs after walk-forward)."""
        from backtester.pipeline.cpcv import cpcv_validate

        active = [c for c in self.state.candidates if not c.eliminated]
        if not active:
            logger.info("No active candidates for CPCV")
            return

        param_dicts = [c.params for c in active]
        results = cpcv_validate(
            self.strategy, param_dicts, self.data,
            config=self.config,
            pip_value=self.pip_value,
            slippage_pips=self.slippage_pips,
        )

        for candidate, cpcv_result in zip(active, results):
            candidate.cpcv = cpcv_result
            if not cpcv_result.passed_gate:
                candidate.eliminated = True
                candidate.eliminated_at_stage = "cpcv"
                candidate.elimination_reason = (
                    f"pct_positive={cpcv_result.pct_positive_sharpe:.2f}, "
                    f"mean_sharpe={cpcv_result.mean_sharpe:.3f}"
                )

    def _run_stability(self) -> None:
        """Stage 4: Parameter stability analysis."""
        from backtester.pipeline.stability import run_stability

        active = [c for c in self.state.candidates if not c.eliminated]
        if not active:
            logger.warning("No active candidates for stability")
            return

        # Use forward data if configured, else full data
        if self.config.stab_use_forward_data:
            n_bars = len(self.data["close"])
            fwd_start = int(n_bars * 0.8)
            eval_data = {}
            for k, v in self.data.items():
                if k.startswith("m1_"):
                    # M1 arrays are not sliceable by H1 index — keep full
                    eval_data[k] = v
                elif k.startswith("h1_to_m1_"):
                    # Rebase H1→M1 mapping for the forward slice
                    eval_data[k] = v[fwd_start:]
                else:
                    eval_data[k] = v[fwd_start:]
        else:
            eval_data = self.data

        param_dicts = [c.params for c in active]
        results = run_stability(
            self.strategy, param_dicts, eval_data, self.config,
            pip_value=self.pip_value, slippage_pips=self.slippage_pips,
        )

        for candidate, stab_result in zip(active, results):
            candidate.stability = stab_result
            # Stability is advisory only — no elimination

    def _run_monte_carlo(self) -> None:
        """Stage 5: Monte Carlo simulation."""
        from backtester.pipeline.monte_carlo import run_monte_carlo
        from backtester.core.telemetry import run_telemetry
        from backtester.core.engine import BacktestEngine
        from backtester.core.dtypes import EXEC_FULL

        active = [c for c in self.state.candidates if not c.eliminated]
        if not active:
            logger.warning("No active candidates for Monte Carlo")
            return

        # Build engine on full data for telemetry
        m1_kwargs: dict = {}
        for key in ("m1_high", "m1_low", "m1_close", "m1_spread",
                    "h1_to_m1_start", "h1_to_m1_end"):
            if key in self.data:
                m1_kwargs[key] = self.data[key]
        engine = BacktestEngine(
            self.strategy,
            self.data["open"], self.data["high"],
            self.data["low"], self.data["close"],
            self.data["volume"], self.data["spread"],
            pip_value=self.pip_value, slippage_pips=self.slippage_pips,
            bar_hour=self.data.get("bar_hour"),
            bar_day_of_week=self.data.get("bar_day_of_week"),
            **m1_kwargs,
        )

        for candidate in active:
            # Get per-trade PnL via telemetry
            telemetry = run_telemetry(engine, candidate.params, EXEC_FULL)
            pnl = np.array(
                [t.pnl_pips for t in telemetry.trades], dtype=np.float64
            )

            mc_result = run_monte_carlo(
                pnl,
                n_trials=candidate.n_trials,
                n_trades=len(pnl),
                config=self.config,
                original_slippage=self.slippage_pips,
                original_commission=self.config.__dict__.get(
                    "commission_pips", 0.7
                ),
            )
            candidate.monte_carlo = mc_result

            if not mc_result.passed_gate:
                candidate.eliminated = True
                candidate.eliminated_at_stage = "monte_carlo"
                candidate.elimination_reason = (
                    f"dsr={mc_result.dsr:.3f}, "
                    f"p={mc_result.permutation_p_value:.3f}"
                )

    def _run_confidence(self) -> None:
        """Stage 6: Confidence scoring."""
        from backtester.pipeline.confidence import compute_confidence

        for candidate in self.state.candidates:
            if candidate.eliminated:
                continue
            candidate.confidence = compute_confidence(candidate, self.config)

    def _run_report(self) -> None:
        """Stage 7: Generate JSON report (MVP)."""
        os.makedirs(self.output_dir, exist_ok=True)
        report_path = os.path.join(self.output_dir, "report.json")

        report = {
            "strategy": self.state.strategy_name,
            "version": self.state.strategy_version,
            "pair": self.state.pair,
            "timeframe": self.state.timeframe,
            "candidates": [],
        }

        # Sort by composite score (survivors first, then eliminated)
        sorted_candidates = sorted(
            self.state.candidates,
            key=lambda c: (
                not c.eliminated,
                c.confidence.composite_score if c.confidence else 0,
            ),
            reverse=True,
        )

        for c in sorted_candidates:
            entry: dict[str, Any] = {
                "index": c.candidate_index,
                "params": c.params,
                "back_quality": c.back_quality,
                "forward_quality": c.forward_quality,
                "forward_back_ratio": c.forward_back_ratio,
                "eliminated": c.eliminated,
            }
            if c.eliminated:
                entry["eliminated_at"] = c.eliminated_at_stage
                entry["elimination_reason"] = c.elimination_reason
            if c.confidence:
                entry["composite_score"] = c.confidence.composite_score
                entry["rating"] = c.confidence.rating.value
                entry["gates_passed"] = c.confidence.gates_passed
            if c.walk_forward:
                entry["wf_pass_rate"] = c.walk_forward.pass_rate
                entry["wf_mean_sharpe"] = c.walk_forward.mean_sharpe
            if c.monte_carlo:
                entry["dsr"] = c.monte_carlo.dsr
                entry["permutation_p"] = c.monte_carlo.permutation_p_value
            if c.cpcv:
                entry["cpcv_n_folds"] = c.cpcv.n_folds
                entry["cpcv_mean_sharpe"] = c.cpcv.mean_sharpe
                entry["cpcv_pct_positive"] = c.cpcv.pct_positive_sharpe
                entry["cpcv_ci_low"] = c.cpcv.sharpe_ci_low
                entry["cpcv_ci_high"] = c.cpcv.sharpe_ci_high
                entry["cpcv_gate"] = c.cpcv.passed_gate
            if c.stability:
                entry["stability_rating"] = c.stability.rating.value
                entry["stability_mean_ratio"] = c.stability.mean_ratio

            report["candidates"].append(entry)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {report_path}")
