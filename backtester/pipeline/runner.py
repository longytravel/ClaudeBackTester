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

import gc
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

# Pip value per standard lot (1.0 lot) for $ conversion
# At 0.01 lots: multiply by 0.01
_PIP_VALUE_USD = {
    "EUR/USD": 10.0, "GBP/USD": 10.0, "AUD/USD": 10.0, "NZD/USD": 10.0,
    "USD/CHF": 10.0, "USD/CAD": 10.0, "EUR/GBP": 10.0,
    "USD/JPY": 7.0, "EUR/JPY": 7.0, "GBP/JPY": 7.0, "AUD/JPY": 7.0,
    "CAD/JPY": 7.0, "CHF/JPY": 7.0, "NZD/JPY": 7.0,
    "EUR/AUD": 6.5, "GBP/AUD": 6.5, "AUD/NZD": 6.5, "EUR/NZD": 6.5,
    "GBP/NZD": 6.5, "EUR/CHF": 10.0, "GBP/CHF": 10.0, "AUD/CAD": 7.0,
    "EUR/CAD": 7.0, "GBP/CAD": 7.0,
    "XAU/USD": 10.0,
}
LOT_SIZE = 0.01  # Standard micro lot for all strategies


def _compute_trade_stats(trades: list, pair: str) -> dict[str, Any]:
    """Compute trade P&L statistics from telemetry trades."""
    from collections import Counter

    if not trades:
        return {"n_trades": 0, "total_pnl_pips": 0.0, "total_pnl_usd": 0.0}

    pnls = [t.pnl_pips for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    pip_val = _PIP_VALUE_USD.get(pair, 10.0) * LOT_SIZE
    total_usd = total_pnl * pip_val

    exit_counts = Counter(t.exit_reason for t in trades)
    exit_breakdown = {}
    for reason, count in exit_counts.most_common():
        reason_pnl = sum(t.pnl_pips for t in trades if t.exit_reason == reason)
        exit_breakdown[reason] = {
            "count": count,
            "pct": round(count / len(trades) * 100, 1),
            "pnl_pips": round(reason_pnl, 2),
            "pnl_usd": round(reason_pnl * pip_val, 2),
        }

    buys = [t for t in trades if t.direction == "BUY"]
    sells = [t for t in trades if t.direction == "SELL"]

    direction_stats = {}
    for label, group in [("BUY", buys), ("SELL", sells)]:
        if group:
            g_pnls = [t.pnl_pips for t in group]
            direction_stats[label] = {
                "n_trades": len(group),
                "win_rate": round(sum(1 for p in g_pnls if p > 0) / len(group) * 100, 1),
                "total_pnl_pips": round(sum(g_pnls), 2),
                "total_pnl_usd": round(sum(g_pnls) * pip_val, 2),
            }

    return {
        "n_trades": len(trades),
        "total_pnl_pips": round(total_pnl, 2),
        "total_pnl_usd": round(total_usd, 2),
        "lot_size": LOT_SIZE,
        "pip_value_usd": pip_val,
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 0,
        "avg_pnl_pips": round(float(np.mean(pnls)), 2),
        "median_pnl_pips": round(float(np.median(pnls)), 2),
        "best_trade_pips": round(max(pnls), 2),
        "worst_trade_pips": round(min(pnls), 2),
        "std_dev_pips": round(float(np.std(pnls)), 2),
        "avg_bars_held": round(float(np.mean([t.bars_held for t in trades])), 1),
        "exit_breakdown": exit_breakdown,
        "direction_stats": direction_stats,
    }


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

        # Guard: shared engine assumes causal signals (pre-computed once on
        # full dataset). Non-causal strategies need per-window signal generation,
        # which is not yet supported.
        from backtester.strategies.base import SignalCausality
        if self.strategy.signal_causality() == SignalCausality.REQUIRES_TRAIN_FIT:
            raise NotImplementedError(
                f"Strategy '{self.strategy.name}' declares REQUIRES_TRAIN_FIT "
                f"signal causality. The shared-engine pipeline pre-computes "
                f"signals once on the full dataset, which is only correct for "
                f"causal indicators. Per-window signal generation is not yet "
                f"supported."
            )

        # Create ONE shared engine for all stages that need evaluation.
        # Creating multiple BacktestEngine instances accumulates Numba NRT
        # memory that Windows never returns to the OS, causing ACCESS_VIOLATION.
        from backtester.pipeline.walk_forward import build_engine
        self._shared_engine = build_engine(
            self.strategy, self.data, self.config,
            self.pip_value, self.slippage_pips,
        )

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

        # Free shared engine after all stages complete
        del self._shared_engine
        gc.collect()

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
            engine=self._shared_engine,
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
            engine=self._shared_engine,
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

        # Use shared engine with windowed evaluation for forward portion
        n_bars = len(self.data["close"])
        if self.config.stab_use_forward_data:
            fwd_start = int(n_bars * 0.8)
            window_start = fwd_start
            window_end = n_bars
        else:
            window_start = None
            window_end = None

        param_dicts = [c.params for c in active]
        results = run_stability(
            self.strategy, param_dicts, self.data, self.config,
            pip_value=self.pip_value, slippage_pips=self.slippage_pips,
            engine=self._shared_engine,
            window_start=window_start, window_end=window_end,
        )

        for candidate, stab_result in zip(active, results):
            candidate.stability = stab_result
            # Stability is advisory only — no elimination

    def _run_monte_carlo(self) -> None:
        """Stage 5: Monte Carlo simulation + regime analysis."""
        from backtester.pipeline.monte_carlo import run_monte_carlo
        from backtester.core.telemetry import run_telemetry
        from backtester.core.dtypes import EXEC_FULL

        active = [c for c in self.state.candidates if not c.eliminated]
        if not active:
            logger.warning("No active candidates for Monte Carlo")
            return

        # Use shared engine for telemetry (no new engine creation)
        engine = self._shared_engine

        # Pre-compute regime labels once if enabled
        regime_labels = None
        if self.config.regime_enabled:
            from backtester.pipeline.regime import classify_bars
            regime_labels = classify_bars(
                self.data["high"], self.data["low"], self.data["close"],
                adx_period=self.config.regime_adx_period,
                atr_period=self.config.regime_atr_period,
                adx_trending_threshold=self.config.regime_adx_trending,
                adx_ranging_threshold=self.config.regime_adx_ranging,
                natr_percentile_lookback=self.config.regime_natr_lookback,
                natr_high_percentile=self.config.regime_natr_high_pctile,
                min_regime_bars=self.config.regime_min_bars,
            )
            logger.info("Regime labels computed for %d bars", len(regime_labels))

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
                original_commission=self.config.commission_pips,
            )
            candidate.monte_carlo = mc_result

            # Compute trade statistics from telemetry
            candidate.trade_stats = _compute_trade_stats(
                telemetry.trades, self.state.pair
            )

            if not mc_result.passed_gate:
                candidate.eliminated = True
                candidate.eliminated_at_stage = "monte_carlo"
                candidate.elimination_reason = (
                    f"dsr={mc_result.dsr:.3f}, "
                    f"p={mc_result.permutation_p_value:.3f}"
                )

            # Regime analysis (advisory, runs on all candidates including eliminated)
            if regime_labels is not None:
                from backtester.pipeline.regime import compute_regime_stats
                candidate.regime = compute_regime_stats(
                    regime_labels, telemetry.trades,
                    min_trades_per_regime=self.config.regime_min_trades,
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
            if c.regime:
                entry["regime_distribution"] = c.regime.regime_distribution
                entry["regime_robustness_score"] = c.regime.robustness_score
                entry["regime_advisory"] = c.regime.advisory
                entry["per_regime_stats"] = [asdict(rs) for rs in c.regime.per_regime]
            if c.trade_stats:
                entry["trade_stats"] = c.trade_stats

            report["candidates"].append(entry)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {report_path}")
