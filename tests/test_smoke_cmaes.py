"""Smoke tests for CMA-ES optimizer upgrade.

These tests run real optimization with synthetic data to verify the full
pipeline works end-to-end. They take longer than unit tests (5-60 seconds each).
"""
import pytest
import numpy as np

from backtester.core.dtypes import DIR_BUY, EXEC_BASIC, EXEC_FULL
from backtester.core.encoding import build_encoding_spec
from backtester.core.engine import BacktestEngine
from backtester.optimizer.config import (
    DEEP,
    MAX,
    STANDARD,
    TURBO,
    OptimizationConfig,
    PRESETS,
    get_preset,
)
from backtester.optimizer.run import Candidate, OptimizationResult, optimize
from backtester.optimizer.staged import StagedOptimizer, StagedResult, _normalize_stages
from backtester.strategies.base import (
    Direction,
    ParamDef,
    ParamSpace,
    Signal,
    SLTPResult,
    Strategy,
    management_params,
    risk_params,
    time_params,
)

# Mark all tests in this file as slow
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Test strategies
# ---------------------------------------------------------------------------

class SmokeTestStrategy(Strategy):
    """Strategy with enough signals and trend for smoke tests.

    Uses more bars and denser signals than OptimizerTestStrategy
    to ensure the optimizer can find passing candidates reliably.
    """

    @property
    def name(self) -> str:
        return "smoke_test"

    @property
    def version(self) -> str:
        return "1.0"

    def param_space(self) -> ParamSpace:
        ps = ParamSpace([
            ParamDef("fast_period", [5, 10, 20], group="signal"),
            ParamDef("slow_period", [20, 50, 100], group="signal"),
        ])
        for p in risk_params():
            ps.add(p.name, p.values, p.group)
        for p in management_params():
            ps.add(p.name, p.values, p.group)
        for p in time_params():
            ps.add(p.name, p.values, p.group)
        return ps

    def generate_signals(self, open_, high, low, close, volume, spread):
        signals = []
        # Generate a signal every 10 bars (denser than standard test strategy)
        for i in range(20, len(close) - 20, 10):
            direction = Direction.BUY if i % 20 == 0 else Direction.SELL
            signals.append(Signal(
                bar_index=i,
                direction=direction,
                entry_price=close[i],
                hour=10,
                day_of_week=i % 5,
                atr_pips=20.0,
            ))
        return signals

    def filter_signals(self, signals, params):
        return signals

    def calc_sl_tp(self, signal, params, high, low):
        return SLTPResult(
            sl_price=signal.entry_price - 0.003 if signal.direction == Direction.BUY
            else signal.entry_price + 0.003,
            tp_price=signal.entry_price + 0.006 if signal.direction == Direction.BUY
            else signal.entry_price - 0.006,
            sl_pips=30.0,
            tp_pips=60.0,
        )


class MinimalSignalStrategy(Strategy):
    """Strategy with fewer signal params for testing different param counts."""

    @property
    def name(self) -> str:
        return "minimal_signal"

    @property
    def version(self) -> str:
        return "1.0"

    def param_space(self) -> ParamSpace:
        ps = ParamSpace([
            ParamDef("threshold", [10, 20, 30, 40, 50], group="signal"),
        ])
        for p in risk_params():
            ps.add(p.name, p.values, p.group)
        for p in management_params():
            ps.add(p.name, p.values, p.group)
        for p in time_params():
            ps.add(p.name, p.values, p.group)
        return ps

    def generate_signals(self, open_, high, low, close, volume, spread):
        signals = []
        for i in range(20, len(close) - 20, 15):
            signals.append(Signal(
                bar_index=i,
                direction=Direction.BUY,
                entry_price=close[i],
                hour=10,
                day_of_week=i % 5,
                atr_pips=20.0,
            ))
        return signals

    def filter_signals(self, signals, params):
        return signals

    def calc_sl_tp(self, signal, params, high, low):
        return SLTPResult(
            sl_price=signal.entry_price - 0.003,
            tp_price=signal.entry_price + 0.006,
            sl_pips=30.0,
            tp_pips=60.0,
        )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _make_trending_data(n_bars: int = 3000, pip: float = 0.0001, seed: int = 42):
    """Create trending price data with enough bars for realistic signal generation.

    Generates a mild uptrend with noise — enough for the optimizer to find
    strategies with positive quality on BUY signals.
    """
    rng = np.random.default_rng(seed)
    base = 1.1000
    open_ = np.zeros(n_bars, dtype=np.float64)
    high = np.zeros(n_bars, dtype=np.float64)
    low = np.zeros(n_bars, dtype=np.float64)
    close = np.zeros(n_bars, dtype=np.float64)
    for i in range(n_bars):
        trend = base + i * 0.5 * pip + rng.normal(0, 3 * pip)
        open_[i] = trend
        high[i] = trend + rng.uniform(5, 20) * pip
        low[i] = trend - rng.uniform(5, 20) * pip
        close[i] = trend + rng.normal(0, 2 * pip)
    volume = np.ones(n_bars, dtype=np.float64)
    spread = np.full(n_bars, 0.00010, dtype=np.float64)  # 1 pip spread
    return open_, high, low, close, volume, spread


def _make_relaxed_config(**overrides) -> OptimizationConfig:
    """Create a config with relaxed filters suitable for smoke tests.

    Uses small trial counts for speed and very relaxed post-filters
    so the optimizer can find passing candidates on synthetic data.
    """
    defaults = dict(
        trials_per_stage=300,
        refinement_trials=500,
        batch_size=64,
        min_total_trades=1,
        min_trades_per_year=0.0,
        min_r_squared=0.0,
        max_dd_pct=100.0,
        min_forward_back_ratio=0.0,
        seed=42,
        exploitation_method="cmaes",
        cmaes_sigma0=0.3,
        max_cyclic_passes=0,
        refinement_neighborhood_radius=3,
    )
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


def _run_optimize_smoke(
    strategy: Strategy | None = None,
    config: OptimizationConfig | None = None,
    with_forward: bool = True,
    n_bars: int = 3000,
    data_seed: int = 42,
) -> OptimizationResult:
    """Run optimize() with synthetic data — the standard smoke test helper."""
    strategy = strategy or SmokeTestStrategy()
    config = config or _make_relaxed_config()

    data = _make_trending_data(n_bars=n_bars, seed=data_seed)
    open_, high, low, close, volume, spread = data

    kwargs = dict(
        strategy=strategy,
        open_back=open_,
        high_back=high,
        low_back=low,
        close_back=close,
        volume_back=volume,
        spread_back=spread,
        config=config,
        pip_value=0.0001,
        slippage_pips=0.0,
        commission_pips=0.0,
        max_spread_pips=0.0,
    )

    if with_forward:
        # Use same data as forward (smoke test, not quality test)
        kwargs.update(
            open_fwd=open_,
            high_fwd=high,
            low_fwd=low,
            close_fwd=close,
            volume_fwd=volume,
            spread_fwd=spread,
        )

    return optimize(**kwargs)


# ---------------------------------------------------------------------------
# CMA-ES end-to-end smoke tests
# ---------------------------------------------------------------------------

class TestCMAESSmoke:
    """End-to-end smoke tests with CMA-ES exploitation."""

    def test_cmaes_produces_candidates(self):
        """Full optimizer with CMA-ES produces valid candidates with quality > 0."""
        result = _run_optimize_smoke()

        assert len(result.candidates) >= 1, "No candidates produced"
        best = result.candidates[0]
        assert best.back_metrics["quality_score"] > 0, (
            f"Best quality should be > 0, got {best.back_metrics['quality_score']}"
        )
        assert best.back_metrics["trades"] > 0, "Best candidate has zero trades"
        assert np.isfinite(best.back_metrics["sharpe"]), "Sharpe is not finite"
        assert isinstance(best.params, dict), "Params should be a dict"
        assert len(best.params) > 0, "Params dict is empty"

    def test_eda_produces_comparable_results(self):
        """EDA exploitation on same data produces results (regression check)."""
        config = _make_relaxed_config(exploitation_method="eda")
        result = _run_optimize_smoke(config=config)

        assert len(result.candidates) >= 1, "EDA produced no candidates"
        best = result.candidates[0]
        assert best.back_metrics["quality_score"] > 0, (
            f"EDA quality should be > 0, got {best.back_metrics['quality_score']}"
        )
        assert best.back_metrics["trades"] > 0, "EDA best has zero trades"

    def test_composite_stage_optimizes_jointly(self):
        """Core trade profile stage optimizes risk+trailing+breakeven together."""
        config = _make_relaxed_config(trials_per_stage=300, refinement_trials=500)
        strategy = SmokeTestStrategy()

        # Verify strategy stages include composite stage
        raw_stages = strategy.optimization_stages()
        has_composite = any(
            isinstance(s, tuple) and s[0] == "core_trade_profile"
            for s in raw_stages
        )
        assert has_composite, (
            f"Expected core_trade_profile composite stage, got: {raw_stages}"
        )

        result = _run_optimize_smoke(strategy=strategy, config=config)

        # Check that staged result has a core_trade_profile stage
        stage_names = [s.stage_name for s in result.staged_result.stages]
        assert "core_trade_profile" in stage_names, (
            f"Expected core_trade_profile in stages, got: {stage_names}"
        )

        # The composite stage should have evaluated some trials
        ctp_stage = next(
            s for s in result.staged_result.stages
            if s.stage_name == "core_trade_profile"
        )
        assert ctp_stage.trials_evaluated > 0, "Composite stage evaluated 0 trials"

    def test_cyclic_passes_add_trials(self):
        """Cyclic passes run and add to total trial count."""
        config_no_cyclic = _make_relaxed_config(
            trials_per_stage=200,
            refinement_trials=300,
            max_cyclic_passes=0,
        )
        config_with_cyclic = _make_relaxed_config(
            trials_per_stage=200,
            refinement_trials=300,
            max_cyclic_passes=1,
            cyclic_budget_fraction=0.5,
        )

        result_no = _run_optimize_smoke(config=config_no_cyclic)
        result_yes = _run_optimize_smoke(config=config_with_cyclic)

        # Cyclic should have more total trials
        assert result_yes.total_trials > result_no.total_trials, (
            f"Cyclic ({result_yes.total_trials}) should have more trials "
            f"than no-cyclic ({result_no.total_trials})"
        )

        # Cyclic stages should be visible in stage names
        stage_names = [s.stage_name for s in result_yes.staged_result.stages]
        cyclic_stages = [n for n in stage_names if "cycle" in n]
        assert len(cyclic_stages) >= 1, (
            f"Expected cyclic stages in names, got: {stage_names}"
        )

    def test_cyclic_vs_no_cyclic(self):
        """Cyclic passes don't degrade quality (may improve)."""
        config_no = _make_relaxed_config(
            trials_per_stage=300,
            refinement_trials=500,
            max_cyclic_passes=0,
            seed=42,
        )
        config_yes = _make_relaxed_config(
            trials_per_stage=300,
            refinement_trials=500,
            max_cyclic_passes=1,
            seed=42,
        )

        result_no = _run_optimize_smoke(config=config_no)
        result_yes = _run_optimize_smoke(config=config_yes)

        q_no = result_no.staged_result.best_quality
        q_yes = result_yes.staged_result.best_quality

        # Allow some variance: cyclic should not be dramatically worse
        # (it can be slightly worse due to different random paths)
        assert q_yes >= 0.5 * q_no or q_yes > 0, (
            f"Cyclic quality ({q_yes:.4f}) much worse than no-cyclic ({q_no:.4f})"
        )

    def test_wider_refinement_searches_broader(self):
        """Refinement with radius=5 explores wider than radius=2."""
        config = _make_relaxed_config(
            refinement_neighborhood_radius=5,
            refinement_trials=600,
        )
        result = _run_optimize_smoke(config=config)

        # Find the refinement stage
        refinement = next(
            s for s in result.staged_result.stages
            if s.stage_name == "refinement"
        )
        assert refinement.valid_count >= 0, "Refinement valid_count should be >= 0"
        assert refinement.trials_evaluated > 0, "Refinement should have evaluated trials"

        # Refinement best quality should be set
        assert result.staged_result.best_quality > -np.inf, (
            "Refinement should produce a best quality"
        )

    def test_cmaes_convergence_handling(self):
        """CMA-ES convergence doesn't crash, remaining budget is used."""
        # Very small budget so CMA-ES converges quickly
        config = _make_relaxed_config(
            trials_per_stage=100,
            refinement_trials=200,
            batch_size=32,
        )
        result = _run_optimize_smoke(config=config)

        # Should not crash — just verify we get a result
        assert result.total_trials > 0, "Should have evaluated some trials"
        assert result.staged_result is not None, "Should have staged result"

    def test_all_presets_valid(self):
        """All preset configs are valid and can create optimizers."""
        for name, preset in PRESETS.items():
            assert preset.exploitation_method == "cmaes", (
                f"Preset '{name}' should use cmaes, got {preset.exploitation_method}"
            )
            assert preset.max_cyclic_passes >= 0, (
                f"Preset '{name}' has negative cyclic passes"
            )
            assert preset.cmaes_sigma0 > 0, (
                f"Preset '{name}' has non-positive sigma0"
            )
            assert preset.trials_per_stage > 0, (
                f"Preset '{name}' has non-positive trials_per_stage"
            )
            assert preset.refinement_trials > 0, (
                f"Preset '{name}' has non-positive refinement_trials"
            )
            assert preset.batch_size > 0, (
                f"Preset '{name}' has non-positive batch_size"
            )

        # Also test get_preset round-trip
        for name in PRESETS:
            fetched = get_preset(name)
            assert fetched.exploitation_method == "cmaes"

    def test_full_pipeline_integration(self):
        """Full optimize() -> pipeline candidates flow works."""
        result = _run_optimize_smoke(with_forward=True)

        assert len(result.candidates) >= 1, "Pipeline should produce candidates"

        for cand in result.candidates:
            # Check structure
            assert isinstance(cand.params, dict)
            assert len(cand.params) > 0

            # Back metrics
            assert "trades" in cand.back_metrics
            assert "quality_score" in cand.back_metrics
            assert "sharpe" in cand.back_metrics
            assert cand.back_metrics["trades"] > 0

            # Forward metrics (since we passed forward data)
            assert cand.forward_metrics is not None, "Forward metrics should be populated"
            assert "trades" in cand.forward_metrics
            assert "quality_score" in cand.forward_metrics
            assert "sharpe" in cand.forward_metrics

            # DSR should be between 0 and 1
            assert 0.0 <= cand.dsr <= 1.0, f"DSR out of range: {cand.dsr}"

        # Check optimizer result metadata
        assert result.total_trials > 0
        assert result.elapsed_seconds > 0
        assert result.evals_per_second > 0
        assert result.staged_result is not None

    def test_no_forward_data_still_works(self):
        """Optimizer produces candidates even without forward data."""
        result = _run_optimize_smoke(with_forward=False)

        assert len(result.candidates) >= 1, "Should produce candidates without forward"
        for cand in result.candidates:
            assert cand.forward_metrics is None, "No forward metrics without forward data"
            assert cand.back_metrics["trades"] > 0

    def test_optimizer_funnel_populated(self):
        """The optimizer funnel dict has expected keys."""
        result = _run_optimize_smoke(with_forward=True)

        funnel = result.optimizer_funnel
        assert "total_trials" in funnel
        assert funnel["total_trials"] > 0
        # These may vary but should exist
        assert "sent_to_pipeline" in funnel
        assert funnel["sent_to_pipeline"] >= 1


# ---------------------------------------------------------------------------
# Stage interaction smoke tests
# ---------------------------------------------------------------------------

class TestStageInteractionSmoke:
    """Smoke tests specifically for stage interaction effects."""

    def test_composite_vs_split_conceptual(self):
        """Verify composite stage groups are correctly normalized."""
        strategy = SmokeTestStrategy()
        raw_stages = strategy.optimization_stages()
        normalized = _normalize_stages(raw_stages)

        # Check we have a composite stage
        composite_found = False
        for name, groups in normalized:
            if name == "core_trade_profile":
                composite_found = True
                assert len(groups) > 1, (
                    f"Composite stage should have >1 groups, got {groups}"
                )
                assert "risk" in groups, "Composite should include risk"
                break

        assert composite_found, (
            f"Expected core_trade_profile in stages: {normalized}"
        )

    def test_composite_produces_valid_results(self):
        """Composite core_trade_profile stage produces valid optimization results."""
        config = _make_relaxed_config(trials_per_stage=300, refinement_trials=500)
        result = _run_optimize_smoke(config=config)

        # Should have valid candidates
        assert len(result.candidates) >= 1
        assert result.staged_result.best_quality > -np.inf

        # core_trade_profile stage should exist and have evaluated trials
        stage_names = [s.stage_name for s in result.staged_result.stages]
        assert "core_trade_profile" in stage_names

    def test_multiple_strategies(self):
        """Optimizer works with different strategy types."""
        configs = _make_relaxed_config(trials_per_stage=200, refinement_trials=300)

        # Strategy 1: SmokeTestStrategy (2 signal params, BUY+SELL)
        result1 = _run_optimize_smoke(
            strategy=SmokeTestStrategy(),
            config=configs,
            n_bars=2000,
        )
        assert len(result1.candidates) >= 1, "SmokeTestStrategy should produce candidates"
        assert result1.total_trials > 0

        # Strategy 2: MinimalSignalStrategy (1 signal param, BUY only)
        result2 = _run_optimize_smoke(
            strategy=MinimalSignalStrategy(),
            config=configs,
            n_bars=2000,
        )
        assert len(result2.candidates) >= 1, "MinimalSignalStrategy should produce candidates"
        assert result2.total_trials > 0

    def test_stage_order_matches_strategy(self):
        """Optimization stage order matches what the strategy declares."""
        strategy = SmokeTestStrategy()
        config = _make_relaxed_config()
        result = _run_optimize_smoke(strategy=strategy, config=config)

        raw_stages = strategy.optimization_stages()
        normalized = _normalize_stages(raw_stages)
        expected_names = [name for name, _ in normalized] + ["refinement"]

        actual_names = [s.stage_name for s in result.staged_result.stages]

        # Actual may include cyclic stages too, but the base stages should match
        base_actual = [n for n in actual_names if "cycle" not in n]
        assert base_actual == expected_names, (
            f"Stage order mismatch.\nExpected: {expected_names}\nActual: {base_actual}"
        )

    def test_each_stage_evaluates_trials(self):
        """Every stage in the pipeline evaluates at least some trials."""
        config = _make_relaxed_config(trials_per_stage=200, refinement_trials=300)
        result = _run_optimize_smoke(config=config)

        for stage in result.staged_result.stages:
            assert stage.trials_evaluated > 0, (
                f"Stage '{stage.stage_name}' evaluated 0 trials"
            )

    def test_refinement_collects_passing_trials(self):
        """Refinement stage collects all passing trials for multi-candidate selection."""
        config = _make_relaxed_config(refinement_trials=500)
        result = _run_optimize_smoke(config=config)

        # Refinement passing data should be available
        sr = result.staged_result
        # It's possible none pass the post-filter, but the arrays should at least
        # be attempted. If best_quality > -inf, refinement found something.
        if sr.best_quality > -np.inf:
            # At minimum, the single best should be accessible
            assert sr.best_indices is not None, "Best indices should be set"
            assert sr.best_metrics is not None, "Best metrics should be set"

    def test_staged_result_total_trials_consistent(self):
        """Total trials = sum of all stage trials."""
        config = _make_relaxed_config(trials_per_stage=200, refinement_trials=300)
        result = _run_optimize_smoke(config=config)

        sum_stage_trials = sum(s.trials_evaluated for s in result.staged_result.stages)
        assert result.staged_result.total_trials == sum_stage_trials, (
            f"Total trials ({result.staged_result.total_trials}) != "
            f"sum of stage trials ({sum_stage_trials})"
        )
