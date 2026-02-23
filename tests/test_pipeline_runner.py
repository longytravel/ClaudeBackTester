"""Tests for pipeline checkpoint and runner."""

import json
import os
import tempfile

import numpy as np
import pytest

from backtester.pipeline.checkpoint import load_checkpoint, save_checkpoint
from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.types import (
    CandidateResult,
    ConfidenceResult,
    MonteCarloResult,
    PipelineState,
    Rating,
    StabilityRating,
    StabilityResult,
    WalkForwardResult,
    WindowResult,
    PerturbationResult,
)


# ---------------------------------------------------------------------------
# Checkpoint Tests
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_save_load_empty_state(self, tmp_path):
        state = PipelineState(
            strategy_name="test", strategy_version="1.0",
            pair="EURUSD", timeframe="H1",
        )
        path = str(tmp_path / "ckpt.json")
        save_checkpoint(state, path)

        loaded = load_checkpoint(path)
        assert loaded.strategy_name == "test"
        assert loaded.pair == "EURUSD"
        assert loaded.candidates == []

    def test_save_load_with_candidates(self, tmp_path):
        state = PipelineState(
            strategy_name="rsi", strategy_version="2.0",
            completed_stages=[1, 2, 3], current_stage=4,
        )
        c = CandidateResult(
            candidate_index=0,
            params={"sl_mode": "fixed_pips", "sl_fixed_pips": 30},
            back_quality=42.0,
            forward_quality=35.0,
            forward_back_ratio=0.83,
            back_sharpe=1.5,
            back_trades=100,
        )
        c.walk_forward = WalkForwardResult(
            n_windows=4, n_oos_windows=3, n_passed=3,
            pass_rate=1.0, mean_sharpe=0.8,
            mean_quality=30.0, geo_mean_quality=28.0,
            min_quality=20.0, quality_cv=0.3, wfe=0.7,
            passed_gate=True,
            windows=[WindowResult(
                window_index=0, start_bar=0, end_bar=1000,
                is_oos=True, n_trades=20, sharpe=0.9,
                quality_score=35.0, passed=True,
            )],
        )
        c.stability = StabilityResult(
            mean_ratio=0.85, min_ratio=0.6,
            worst_param="sl_fixed_pips",
            rating=StabilityRating.ROBUST,
        )
        c.monte_carlo = MonteCarloResult(
            bootstrap_sharpe_mean=1.2, bootstrap_sharpe_std=0.3,
            bootstrap_sharpe_ci_low=0.6, bootstrap_sharpe_ci_high=1.8,
            permutation_p_value=0.01, observed_sharpe=1.5,
            skip_results={"5%": 38.0, "10%": 33.0},
            stress_quality=30.0, stress_quality_ratio=0.71,
            dsr=0.97, passed_gate=True,
        )
        c.confidence = ConfidenceResult(
            walk_forward_score=85.0, monte_carlo_score=75.0,
            forward_back_score=70.0, stability_score=80.0,
            dsr_score=90.0, backtest_quality_score=65.0,
            composite_score=78.0, rating=Rating.GREEN,
            gates_passed={"forward_back_ratio": True, "dsr": True},
            all_gates_passed=True,
        )
        state.candidates.append(c)

        path = str(tmp_path / "ckpt.json")
        save_checkpoint(state, path)

        loaded = load_checkpoint(path)
        assert loaded.strategy_name == "rsi"
        assert loaded.completed_stages == [1, 2, 3]
        assert len(loaded.candidates) == 1

        lc = loaded.candidates[0]
        assert lc.params["sl_mode"] == "fixed_pips"
        assert lc.back_quality == 42.0

        # Walk-forward
        assert lc.walk_forward is not None
        assert lc.walk_forward.pass_rate == 1.0
        assert len(lc.walk_forward.windows) == 1

        # Stability
        assert lc.stability is not None
        assert lc.stability.rating == StabilityRating.ROBUST

        # Monte Carlo
        assert lc.monte_carlo is not None
        assert lc.monte_carlo.dsr == 0.97
        assert lc.monte_carlo.skip_results["5%"] == 38.0

        # Confidence
        assert lc.confidence is not None
        assert lc.confidence.rating == Rating.GREEN
        assert lc.confidence.composite_score == 78.0

    def test_atomic_write(self, tmp_path):
        """Checkpoint should not leave partial files on crash."""
        state = PipelineState(strategy_name="test")
        path = str(tmp_path / "ckpt.json")
        save_checkpoint(state, path)
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")

    def test_overwrite_existing(self, tmp_path):
        path = str(tmp_path / "ckpt.json")
        state1 = PipelineState(strategy_name="v1", current_stage=1)
        save_checkpoint(state1, path)

        state2 = PipelineState(strategy_name="v2", current_stage=3)
        save_checkpoint(state2, path)

        loaded = load_checkpoint(path)
        assert loaded.strategy_name == "v2"
        assert loaded.current_stage == 3

    def test_eliminated_candidates(self, tmp_path):
        state = PipelineState(strategy_name="test")
        c = CandidateResult(
            candidate_index=0, eliminated=True,
            eliminated_at_stage="walk_forward",
            elimination_reason="pass_rate=0.25",
        )
        state.candidates.append(c)

        path = str(tmp_path / "ckpt.json")
        save_checkpoint(state, path)
        loaded = load_checkpoint(path)

        lc = loaded.candidates[0]
        assert lc.eliminated is True
        assert lc.eliminated_at_stage == "walk_forward"
