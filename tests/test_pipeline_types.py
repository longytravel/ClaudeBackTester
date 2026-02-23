"""Tests for pipeline types and config."""

import json
from dataclasses import asdict

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
)


class TestPipelineConfig:
    def test_defaults_valid(self):
        cfg = PipelineConfig()
        errors = cfg.validate()
        assert errors == []

    def test_weights_must_sum_to_one(self):
        cfg = PipelineConfig(conf_weight_walk_forward=0.5)
        errors = cfg.validate()
        assert any("weights sum" in e for e in errors)

    def test_invalid_gate_range(self):
        cfg = PipelineConfig(wf_pass_rate_gate=1.5)
        errors = cfg.validate()
        assert any("wf_pass_rate_gate" in e for e in errors)

    def test_serialization_round_trip(self):
        cfg = PipelineConfig(seed=123, wf_anchored=True)
        d = asdict(cfg)
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        cfg2 = PipelineConfig(**loaded)
        assert cfg2.seed == 123
        assert cfg2.wf_anchored is True


class TestTypes:
    def test_window_result_defaults(self):
        wr = WindowResult(window_index=0, start_bar=0, end_bar=1000, is_oos=True)
        assert wr.sharpe == 0.0
        assert wr.is_oos is True

    def test_walk_forward_result_defaults(self):
        wf = WalkForwardResult()
        assert wf.n_windows == 0
        assert wf.passed_gate is False

    def test_stability_result_defaults(self):
        sr = StabilityResult()
        assert sr.rating == StabilityRating.OVERFIT

    def test_monte_carlo_result_defaults(self):
        mc = MonteCarloResult()
        assert mc.permutation_p_value == 1.0
        assert mc.passed_gate is False

    def test_confidence_result_defaults(self):
        cr = ConfidenceResult()
        assert cr.rating == Rating.RED

    def test_candidate_result_serialization(self):
        c = CandidateResult(
            candidate_index=0,
            params={"sl_mode": "fixed_pips", "sl_fixed_pips": 30},
            back_quality=42.5,
            forward_quality=35.0,
            forward_back_ratio=0.82,
        )
        d = asdict(c)
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["back_quality"] == 42.5
        assert loaded["params"]["sl_mode"] == "fixed_pips"

    def test_pipeline_state_serialization(self):
        state = PipelineState(
            strategy_name="rsi_mean_reversion",
            strategy_version="1.0",
            pair="EURUSD",
            timeframe="H1",
            completed_stages=[1, 2, 3],
            current_stage=4,
        )
        d = asdict(state)
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["strategy_name"] == "rsi_mean_reversion"
        assert loaded["completed_stages"] == [1, 2, 3]

    def test_rating_values(self):
        assert Rating.RED.value == "RED"
        assert Rating.GREEN.value == "GREEN"
        assert StabilityRating.ROBUST.value == "ROBUST"
