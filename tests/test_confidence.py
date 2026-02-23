"""Tests for confidence scoring module."""

import pytest

from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.confidence import (
    apply_gates,
    assign_rating,
    compute_confidence,
    score_backtest_quality,
    score_dsr,
    score_forward_back,
    score_monte_carlo,
    score_stability,
    score_walk_forward,
)
from backtester.pipeline.types import (
    CandidateResult,
    ConfidenceResult,
    MonteCarloResult,
    Rating,
    StabilityRating,
    StabilityResult,
    WalkForwardResult,
    WindowResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_candidate() -> CandidateResult:
    """A candidate that should pass all gates and score well."""
    c = CandidateResult(
        candidate_index=0,
        params={"sl_mode": "fixed_pips"},
        back_quality=45.0,
        forward_quality=38.0,
        forward_back_ratio=0.84,
        back_sharpe=1.5,
        back_trades=120,
        n_trials=5000,
    )
    c.walk_forward = WalkForwardResult(
        n_windows=6,
        n_oos_windows=4,
        n_passed=4,
        pass_rate=1.0,
        mean_sharpe=0.8,
        mean_quality=35.0,
        geo_mean_quality=32.0,
        min_quality=20.0,
        quality_cv=0.25,
        wfe=0.75,
        passed_gate=True,
    )
    c.stability = StabilityResult(
        mean_ratio=0.85,
        min_ratio=0.6,
        worst_param="sl_fixed_pips",
        rating=StabilityRating.ROBUST,
    )
    c.monte_carlo = MonteCarloResult(
        bootstrap_sharpe_mean=1.3,
        bootstrap_sharpe_std=0.3,
        bootstrap_sharpe_ci_low=0.7,
        bootstrap_sharpe_ci_high=1.9,
        permutation_p_value=0.01,
        observed_sharpe=1.5,
        skip_results={"5%": 40.0, "10%": 35.0},
        stress_quality=32.0,
        stress_quality_ratio=0.71,
        dsr=0.97,
        passed_gate=True,
    )
    return c


def _bad_candidate() -> CandidateResult:
    """A candidate that should fail gates."""
    c = CandidateResult(
        candidate_index=1,
        params={"sl_mode": "fixed_pips"},
        back_quality=15.0,
        forward_quality=3.0,
        forward_back_ratio=0.2,  # Fails forward/back gate
        back_sharpe=0.3,
        back_trades=25,
        n_trials=5000,
    )
    c.walk_forward = WalkForwardResult(
        n_windows=6,
        n_oos_windows=4,
        n_passed=1,
        pass_rate=0.25,  # Fails pass rate gate
        mean_sharpe=0.1,  # Fails mean sharpe gate
        mean_quality=5.0,
        geo_mean_quality=0.0,
        min_quality=-2.0,
        quality_cv=1.5,
        wfe=0.2,
        passed_gate=False,
    )
    c.stability = StabilityResult(
        mean_ratio=0.3,
        min_ratio=0.05,
        worst_param="sl_fixed_pips",
        rating=StabilityRating.OVERFIT,
    )
    c.monte_carlo = MonteCarloResult(
        bootstrap_sharpe_mean=0.2,
        bootstrap_sharpe_std=0.8,
        bootstrap_sharpe_ci_low=-0.5,
        bootstrap_sharpe_ci_high=0.9,
        permutation_p_value=0.35,  # Fails permutation gate
        observed_sharpe=0.3,
        skip_results={"5%": 3.0, "10%": 1.0},
        stress_quality=2.0,
        stress_quality_ratio=0.13,
        dsr=0.4,  # Fails DSR gate
        passed_gate=False,
    )
    return c


# ---------------------------------------------------------------------------
# Gate Tests
# ---------------------------------------------------------------------------

class TestGates:
    def test_good_candidate_passes_all_gates(self):
        cfg = PipelineConfig()
        gates = apply_gates(_good_candidate(), cfg)
        assert all(gates.values())

    def test_bad_candidate_fails_gates(self):
        cfg = PipelineConfig()
        gates = apply_gates(_bad_candidate(), cfg)
        assert not gates["forward_back_ratio"]
        assert not gates["wf_pass_rate"]
        assert not gates["wf_mean_sharpe"]
        assert not gates["dsr"]
        assert not gates["permutation_p"]

    def test_missing_stages_fail_gates(self):
        c = CandidateResult(candidate_index=0, forward_back_ratio=0.5)
        cfg = PipelineConfig()
        gates = apply_gates(c, cfg)
        assert gates["forward_back_ratio"] is True
        assert gates["wf_pass_rate"] is False
        assert gates["dsr"] is False

    def test_borderline_forward_back(self):
        cfg = PipelineConfig()
        c = CandidateResult(candidate_index=0, forward_back_ratio=0.4)
        gates = apply_gates(c, cfg)
        assert gates["forward_back_ratio"] is True

        c2 = CandidateResult(candidate_index=0, forward_back_ratio=0.39)
        gates2 = apply_gates(c2, cfg)
        assert gates2["forward_back_ratio"] is False


# ---------------------------------------------------------------------------
# Sub-Scorer Tests
# ---------------------------------------------------------------------------

class TestScoreWalkForward:
    def test_good_wf_scores_high(self):
        wf = WalkForwardResult(
            n_oos_windows=5, n_passed=5, pass_rate=1.0,
            mean_sharpe=1.0, mean_quality=40.0,
            geo_mean_quality=35.0, min_quality=25.0,
            quality_cv=0.2, wfe=0.8,
        )
        score = score_walk_forward(wf)
        assert score > 70

    def test_none_wf_scores_zero(self):
        assert score_walk_forward(None) == 0.0

    def test_no_oos_windows_scores_zero(self):
        wf = WalkForwardResult(n_oos_windows=0)
        assert score_walk_forward(wf) == 0.0

    def test_partial_pass_rate(self):
        wf = WalkForwardResult(
            n_oos_windows=4, n_passed=2, pass_rate=0.5,
            mean_sharpe=0.5, mean_quality=15.0,
            geo_mean_quality=10.0, min_quality=5.0,
            quality_cv=0.6,
        )
        score = score_walk_forward(wf)
        assert 20 < score < 60


class TestScoreMonteCarlo:
    def test_strong_mc_scores_high(self):
        mc = MonteCarloResult(
            bootstrap_sharpe_ci_low=0.8,
            bootstrap_sharpe_ci_high=2.0,
            permutation_p_value=0.005,
            skip_results={"5%": 40.0, "10%": 35.0},
            stress_quality_ratio=0.85,
        )
        score = score_monte_carlo(mc)
        assert score > 60

    def test_weak_mc_scores_low(self):
        mc = MonteCarloResult(
            bootstrap_sharpe_ci_low=-0.5,
            bootstrap_sharpe_ci_high=0.5,
            permutation_p_value=0.4,
            skip_results={"5%": 3.0},
            stress_quality_ratio=0.1,
        )
        score = score_monte_carlo(mc)
        assert score < 40

    def test_none_mc_scores_zero(self):
        assert score_monte_carlo(None) == 0.0


class TestScoreForwardBack:
    def test_ratio_1_0_is_100(self):
        assert score_forward_back(1.0) == 100.0

    def test_ratio_above_1_is_100(self):
        assert score_forward_back(1.5) == 100.0

    def test_ratio_0_4_is_0(self):
        assert score_forward_back(0.4) == 0.0

    def test_ratio_below_0_4_is_0(self):
        assert score_forward_back(0.2) == 0.0

    def test_ratio_0_7_is_middle(self):
        score = score_forward_back(0.7)
        assert 40 < score < 60


class TestScoreStability:
    def test_robust_scores_high(self):
        stab = StabilityResult(mean_ratio=0.9, min_ratio=0.7)
        score = score_stability(stab)
        assert score > 70

    def test_overfit_scores_low(self):
        stab = StabilityResult(mean_ratio=0.3, min_ratio=0.1)
        score = score_stability(stab)
        assert score < 10

    def test_none_scores_zero(self):
        assert score_stability(None) == 0.0


class TestScoreDSR:
    def test_high_dsr(self):
        assert score_dsr(0.99) == 100.0

    def test_gate_threshold_dsr(self):
        score = score_dsr(0.95)
        assert score == pytest.approx(80.0)

    def test_low_dsr(self):
        assert score_dsr(0.5) == 0.0

    def test_below_threshold(self):
        assert score_dsr(0.3) == 0.0


class TestScoreBacktestQuality:
    def test_good_backtest(self):
        c = CandidateResult(
            candidate_index=0,
            back_quality=50.0, back_trades=150, back_sharpe=2.0,
        )
        score = score_backtest_quality(c)
        assert score == 100.0

    def test_poor_backtest(self):
        c = CandidateResult(
            candidate_index=0,
            back_quality=5.0, back_trades=15, back_sharpe=0.2,
        )
        score = score_backtest_quality(c)
        assert score < 20


# ---------------------------------------------------------------------------
# Composite Scoring Tests
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_good_candidate_green(self):
        cfg = PipelineConfig()
        c = _good_candidate()
        result = compute_confidence(c, cfg)
        assert result.all_gates_passed is True
        assert result.rating == Rating.GREEN
        assert result.composite_score >= 70

    def test_bad_candidate_red(self):
        cfg = PipelineConfig()
        c = _bad_candidate()
        result = compute_confidence(c, cfg)
        assert result.all_gates_passed is False
        assert result.rating == Rating.RED

    def test_gate_failure_forces_red(self):
        """Even high composite score is RED if gates fail."""
        cfg = PipelineConfig()
        c = _good_candidate()
        # Break one gate
        c.forward_back_ratio = 0.2
        result = compute_confidence(c, cfg)
        assert result.rating == Rating.RED
        # Composite score may still be high, but rating is RED
        assert result.gates_passed["forward_back_ratio"] is False

    def test_composite_score_range(self):
        cfg = PipelineConfig()
        result = compute_confidence(_good_candidate(), cfg)
        assert 0 <= result.composite_score <= 100

    def test_sub_scores_populated(self):
        cfg = PipelineConfig()
        result = compute_confidence(_good_candidate(), cfg)
        assert result.walk_forward_score > 0
        assert result.monte_carlo_score > 0
        assert result.forward_back_score > 0
        assert result.stability_score > 0
        assert result.dsr_score > 0
        assert result.backtest_quality_score > 0


class TestAssignRating:
    def test_green(self):
        cfg = PipelineConfig()
        assert assign_rating(80.0, True, cfg) == Rating.GREEN

    def test_yellow(self):
        cfg = PipelineConfig()
        assert assign_rating(55.0, True, cfg) == Rating.YELLOW

    def test_red_low_score(self):
        cfg = PipelineConfig()
        assert assign_rating(30.0, True, cfg) == Rating.RED

    def test_red_gate_failure(self):
        cfg = PipelineConfig()
        assert assign_rating(90.0, False, cfg) == Rating.RED

    def test_custom_thresholds(self):
        cfg = PipelineConfig(conf_green_threshold=90.0, conf_yellow_threshold=60.0)
        assert assign_rating(85.0, True, cfg) == Rating.YELLOW
        assert assign_rating(95.0, True, cfg) == Rating.GREEN
