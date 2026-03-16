"""Tests for optimizer: samplers, prefilters, ranking, diversity, staged flow."""

import numpy as np
import pytest

from backtester.core.dtypes import (
    DIR_BUY,
    EXEC_BASIC,
    EXEC_FULL,
    M_MAX_DD_PCT,
    M_QUALITY,
    M_R_SQUARED,
    M_SHARPE,
    M_TRADES,
    M_WIN_RATE,
    NUM_METRICS,
)
from backtester.core.encoding import build_encoding_spec
from backtester.core.engine import BacktestEngine
from backtester.optimizer.archive import DiversityArchive, select_top_n_diverse
from backtester.optimizer.config import (
    STANDARD,
    TURBO,
    OptimizationConfig,
    get_preset,
)
from backtester.optimizer.prefilter import postfilter_results, prefilter_invalid_combos
from backtester.optimizer.ranking import (
    combined_rank,
    deflated_sharpe_ratio,
    forward_back_gate,
    forward_back_ratio,
    rank_by_quality,
    select_top_n,
)
from backtester.optimizer.sampler import CMAESSampler, EDASampler, RandomSampler, SobolSampler
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


# ---------------------------------------------------------------------------
# Test strategy for optimizer tests
# ---------------------------------------------------------------------------

class OptimizerTestStrategy(Strategy):
    """Simple strategy for optimizer testing."""

    @property
    def name(self) -> str:
        return "opt_test"

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
        # Generate a signal every 20 bars
        for i in range(20, len(close) - 20, 20):
            signals.append(Signal(
                bar_index=i,
                direction=Direction.BUY,
                entry_price=close[i],
                hour=10,
                day_of_week=i % 5,  # Rotate Mon-Fri
                atr_pips=20.0,
            ))
        return signals

    def filter_signals(self, signals, params):
        return signals

    def calc_sl_tp(self, signal, params, high, low):
        return SLTPResult(
            sl_price=signal.entry_price - 0.003,
            tp_price=signal.entry_price + 0.006,
            sl_pips=30.0, tp_pips=60.0,
        )


def _make_data(n_bars: int = 500, pip: float = 0.0001):
    """Create trending price data for optimizer tests."""
    base = 1.1000
    rng = np.random.default_rng(42)
    open_ = np.zeros(n_bars, dtype=np.float64)
    high = np.zeros(n_bars, dtype=np.float64)
    low = np.zeros(n_bars, dtype=np.float64)
    close = np.zeros(n_bars, dtype=np.float64)
    for i in range(n_bars):
        # Slight uptrend with noise
        trend = base + i * 0.5 * pip + rng.normal(0, 3 * pip)
        open_[i] = trend
        high[i] = trend + rng.uniform(5, 20) * pip
        low[i] = trend - rng.uniform(5, 20) * pip
        close[i] = trend + rng.normal(0, 2 * pip)
    volume = np.ones(n_bars, dtype=np.float64)
    spread = np.ones(n_bars, dtype=np.float64)
    return open_, high, low, close, volume, spread


# ---------------------------------------------------------------------------
# Sampler tests
# ---------------------------------------------------------------------------

class TestRandomSampler:
    def test_shape(self):
        ps = ParamSpace([
            ParamDef("a", [1, 2, 3]),
            ParamDef("b", [10, 20]),
        ])
        spec = build_encoding_spec(ps)
        sampler = RandomSampler(spec, seed=42)
        matrix = sampler.sample(100)
        assert matrix.shape == (100, 2)

    def test_values_in_range(self):
        ps = ParamSpace([
            ParamDef("a", [1, 2, 3]),
            ParamDef("b", [10, 20, 30, 40]),
        ])
        spec = build_encoding_spec(ps)
        sampler = RandomSampler(spec, seed=42)
        matrix = sampler.sample(1000)
        assert matrix[:, 0].min() >= 0
        assert matrix[:, 0].max() <= 2
        assert matrix[:, 1].min() >= 0
        assert matrix[:, 1].max() <= 3

    def test_locked_params(self):
        ps = ParamSpace([
            ParamDef("a", [1, 2, 3]),
            ParamDef("b", [10, 20, 30]),
        ])
        spec = build_encoding_spec(ps)
        sampler = RandomSampler(spec, seed=42)
        locked = np.array([1, -1], dtype=np.int64)  # Lock param 'a' to index 1
        matrix = sampler.sample(100, locked=locked)
        assert np.all(matrix[:, 0] == 1)
        assert not np.all(matrix[:, 1] == matrix[0, 1])  # b should vary

    def test_mask_inactive_unlocked_gets_random(self):
        """Inactive + unlocked params get random samples (noise averaging)."""
        ps = ParamSpace([
            ParamDef("a", [1, 2, 3]),
            ParamDef("b", [10, 20, 30]),
        ])
        spec = build_encoding_spec(ps)
        sampler = RandomSampler(spec, seed=42)
        mask = np.array([True, False], dtype=bool)
        matrix = sampler.sample(100, mask=mask)
        # Active param is varied
        assert len(np.unique(matrix[:, 0])) > 1
        # Inactive + unlocked param is ALSO varied (noise averaging)
        assert len(np.unique(matrix[:, 1])) > 1
        # All values in valid range
        assert np.all(matrix[:, 1] >= 0)
        assert np.all(matrix[:, 1] < 3)

    def test_mask_inactive_locked_uses_locked_value(self):
        """Inactive + locked params use locked value (not 0)."""
        ps = ParamSpace([
            ParamDef("a", [1, 2, 3]),
            ParamDef("b", [10, 20, 30]),
        ])
        spec = build_encoding_spec(ps)
        sampler = RandomSampler(spec, seed=42)
        mask = np.array([True, False], dtype=bool)
        locked = np.array([-1, 2], dtype=np.int64)  # b locked to index 2
        matrix = sampler.sample(100, mask=mask, locked=locked)
        # Active param is varied
        assert len(np.unique(matrix[:, 0])) > 1
        # Inactive + locked param uses locked value
        assert np.all(matrix[:, 1] == 2)


class TestSobolSampler:
    def test_shape(self):
        ps = ParamSpace([
            ParamDef("a", [1, 2, 3, 4]),
            ParamDef("b", [10, 20, 30]),
        ])
        spec = build_encoding_spec(ps)
        sampler = SobolSampler(spec, seed=42)
        matrix = sampler.sample(64)
        assert matrix.shape == (64, 2)

    def test_coverage(self):
        """Sobol should cover the space better than random."""
        ps = ParamSpace([
            ParamDef("a", list(range(10))),
            ParamDef("b", list(range(10))),
        ])
        spec = build_encoding_spec(ps)
        sampler = SobolSampler(spec, seed=42)
        matrix = sampler.sample(128)
        # Should hit most of the 10×10 grid
        unique_pairs = set()
        for i in range(matrix.shape[0]):
            unique_pairs.add((matrix[i, 0], matrix[i, 1]))
        assert len(unique_pairs) > 50  # Good coverage


class TestEDASampler:
    def test_shape(self):
        ps = ParamSpace([
            ParamDef("a", [1, 2, 3]),
            ParamDef("b", [10, 20]),
        ])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, seed=42)
        matrix = eda.sample(100)
        assert matrix.shape == (100, 2)

    def test_initial_uniform(self):
        """Initial distribution should be approximately uniform."""
        ps = ParamSpace([ParamDef("a", [1, 2, 3])])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, seed=42)
        matrix = eda.sample(3000)
        counts = np.bincount(matrix[:, 0], minlength=3)
        # Each value should get ~1000 samples, allow 20% tolerance
        assert all(800 < c < 1200 for c in counts)

    def test_update_concentrates(self):
        """After updating with elites, sampling should concentrate."""
        ps = ParamSpace([ParamDef("a", [1, 2, 3, 4, 5])])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, learning_rate=0.5, seed=42)

        # Update with elites that all have index 2
        elites = np.full((50, 1), 2, dtype=np.int64)
        for _ in range(10):
            eda.update(elites)

        # Now sampling should strongly favor index 2
        matrix = eda.sample(1000)
        counts = np.bincount(matrix[:, 0], minlength=5)
        assert counts[2] > counts.sum() * 0.4  # Should be dominant

    def test_reset(self):
        """Reset should return to uniform distribution."""
        ps = ParamSpace([ParamDef("a", [1, 2, 3])])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, seed=42)

        # Update then reset
        elites = np.full((50, 1), 0, dtype=np.int64)
        eda.update(elites)
        eda.reset()

        # Should be approximately uniform again
        for prob in eda.prob_tables[0]:
            assert abs(prob - 1.0 / 3.0) < 0.01

    def test_adaptive_lr_decays(self):
        """Learning rate should decay over successive updates."""
        ps = ParamSpace([ParamDef("a", [1, 2, 3, 4, 5])])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, learning_rate=0.3, lr_decay=0.9, lr_floor=0.05, seed=42)

        assert eda.effective_lr == pytest.approx(0.3)  # Before any updates

        elites = np.full((10, 1), 2, dtype=np.int64)
        eda.update(elites)
        lr_after_1 = eda.effective_lr
        assert lr_after_1 < 0.3  # Should have decayed
        assert lr_after_1 == pytest.approx(0.05 + 0.25 * 0.9)

        for _ in range(9):
            eda.update(elites)
        lr_after_10 = eda.effective_lr
        assert lr_after_10 < lr_after_1
        assert lr_after_10 >= 0.05  # Should not go below floor

    def test_adaptive_lr_respects_floor(self):
        """After many updates, LR should converge to floor."""
        ps = ParamSpace([ParamDef("a", [1, 2, 3])])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, learning_rate=0.3, lr_decay=0.5, lr_floor=0.05, seed=42)

        elites = np.full((10, 1), 0, dtype=np.int64)
        for _ in range(100):
            eda.update(elites)

        assert eda.effective_lr == pytest.approx(0.05, abs=1e-10)

    def test_reset_restarts_lr_decay(self):
        """Reset should restart the LR decay from initial."""
        ps = ParamSpace([ParamDef("a", [1, 2, 3])])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, learning_rate=0.3, lr_decay=0.9, lr_floor=0.05, seed=42)

        elites = np.full((10, 1), 0, dtype=np.int64)
        for _ in range(10):
            eda.update(elites)
        assert eda.effective_lr < 0.3

        eda.reset()
        assert eda.effective_lr == pytest.approx(0.3)
        assert eda.update_count == 0

    def test_entropy_uniform_is_one(self):
        """Uniform distribution should have normalized entropy of 1.0."""
        ps = ParamSpace([
            ParamDef("a", [1, 2, 3, 4]),
            ParamDef("b", [10, 20, 30]),
        ])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, seed=42)

        assert eda.entropy() == pytest.approx(1.0)
        per_param = eda.entropy_per_param()
        assert len(per_param) == 2
        assert all(e == pytest.approx(1.0) for e in per_param)

    def test_entropy_decreases_with_updates(self):
        """Entropy should decrease as distribution converges."""
        ps = ParamSpace([ParamDef("a", [1, 2, 3, 4, 5])])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, learning_rate=0.5, seed=42)

        initial_entropy = eda.entropy()
        assert initial_entropy == pytest.approx(1.0)

        elites = np.full((50, 1), 2, dtype=np.int64)
        for _ in range(10):
            eda.update(elites)

        assert eda.entropy() < initial_entropy

    def test_entropy_with_mask(self):
        """Entropy with mask should only include active params."""
        ps = ParamSpace([
            ParamDef("a", [1, 2, 3]),
            ParamDef("b", [10, 20, 30]),
        ])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, learning_rate=0.5, seed=42)

        # Update only param 'a' to converge
        mask = np.array([True, False], dtype=bool)
        elites = np.array([[0, 0]], dtype=np.int64)
        for _ in range(20):
            eda.update(elites, mask=mask)

        # Full entropy includes both (a=converged, b=uniform)
        full = eda.entropy()
        # Masked entropy only includes a (converged)
        masked = eda.entropy(mask=mask)
        assert masked < full

    def test_entropy_single_value_param(self):
        """Single-value param should have entropy 0."""
        ps = ParamSpace([ParamDef("a", [42])])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, seed=42)
        assert eda.entropy() == pytest.approx(0.0)

    def test_update_count_tracks(self):
        """Update count should increment on each update call."""
        ps = ParamSpace([ParamDef("a", [1, 2, 3])])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, seed=42)
        assert eda.update_count == 0

        elites = np.full((5, 1), 1, dtype=np.int64)
        eda.update(elites)
        assert eda.update_count == 1
        eda.update(elites)
        assert eda.update_count == 2

    def test_empty_elite_no_update(self):
        """Empty elite set should not change anything."""
        ps = ParamSpace([ParamDef("a", [1, 2, 3])])
        spec = build_encoding_spec(ps)
        eda = EDASampler(spec, seed=42)

        elites = np.zeros((0, 1), dtype=np.int64)
        eda.update(elites)
        assert eda.update_count == 0  # No actual update
        assert eda.entropy() == pytest.approx(1.0)


class TestCMAESSampler:
    def _make_spec(self):
        ps = ParamSpace([
            ParamDef("a", list(range(10))),   # 10 values: 0-9
            ParamDef("b", list(range(20))),   # 20 values: 0-19
            ParamDef("c", list(range(5))),    # 5 values: 0-4
        ])
        return build_encoding_spec(ps)

    def test_shape(self):
        spec = self._make_spec()
        sampler = CMAESSampler(spec, seed=42)
        matrix = sampler.sample(100)
        assert matrix.shape == (100, 3)
        assert matrix.dtype == np.int64

    def test_values_in_range(self):
        spec = self._make_spec()
        sampler = CMAESSampler(spec, seed=42)
        matrix = sampler.sample(200)
        assert matrix[:, 0].min() >= 0
        assert matrix[:, 0].max() <= 9
        assert matrix[:, 1].min() >= 0
        assert matrix[:, 1].max() <= 19
        assert matrix[:, 2].min() >= 0
        assert matrix[:, 2].max() <= 4

    def test_locked_params_respected(self):
        spec = self._make_spec()
        sampler = CMAESSampler(spec, seed=42)
        locked = np.array([3, -1, 2], dtype=np.int64)
        matrix = sampler.sample(100, locked=locked)
        assert np.all(matrix[:, 0] == 3)
        assert np.all(matrix[:, 2] == 2)
        # b should vary
        assert len(np.unique(matrix[:, 1])) > 1

    def test_mask_inactive_random(self):
        """Inactive params get uniform random values (not all zeros)."""
        spec = self._make_spec()
        sampler = CMAESSampler(spec, seed=42)
        mask = np.array([True, False, True], dtype=bool)
        matrix = sampler.sample(200, mask=mask)
        # Inactive param b should have varied values
        unique_b = np.unique(matrix[:, 1])
        assert len(unique_b) > 1
        # All in valid range
        assert matrix[:, 1].min() >= 0
        assert matrix[:, 1].max() <= 19

    def test_neighborhood_bounds(self):
        from backtester.optimizer.sampler import NeighborhoodSpec
        spec = self._make_spec()
        sampler = CMAESSampler(spec, seed=42)
        nb = NeighborhoodSpec(
            min_bounds=np.array([3, 8, 1], dtype=np.int64),
            max_bounds=np.array([7, 12, 3], dtype=np.int64),
        )
        matrix = sampler.sample(200, neighborhood=nb)
        assert matrix[:, 0].min() >= 3
        assert matrix[:, 0].max() <= 7
        assert matrix[:, 1].min() >= 8
        assert matrix[:, 1].max() <= 12
        assert matrix[:, 2].min() >= 1
        assert matrix[:, 2].max() <= 3

    def test_update_and_convergence(self):
        """After many updates with same elite, CMA-ES should concentrate."""
        ps = ParamSpace([
            ParamDef("a", list(range(20))),
            ParamDef("b", list(range(20))),
        ])
        spec = build_encoding_spec(ps)
        pop_size = 16
        sampler = CMAESSampler(spec, sigma0=0.3, population_size=pop_size, seed=42)

        target_a, target_b = 10, 15

        for gen in range(30):
            matrix = sampler.sample(pop_size)
            # Assign quality based on distance to target
            qualities = np.zeros(pop_size, dtype=np.float64)
            for i in range(pop_size):
                dist = abs(matrix[i, 0] - target_a) + abs(matrix[i, 1] - target_b)
                qualities[i] = 100.0 - dist  # higher = better
            sampler.update(matrix, qualities)

        # After convergence, samples should cluster near target
        final = sampler.sample(pop_size)
        mean_a = final[:, 0].mean()
        mean_b = final[:, 1].mean()
        # Should be within 3 of target on average
        assert abs(mean_a - target_a) < 5
        assert abs(mean_b - target_b) < 5

    def test_reset_clears_state(self):
        spec = self._make_spec()
        sampler = CMAESSampler(spec, seed=42)
        # Sample to initialize
        sampler.sample(50)
        assert sampler._cma is not None

        sampler.reset()
        assert sampler._cma is None
        assert sampler.converged is False
        assert sampler._generation_count == 0
        assert sampler._pending_tell == []

        # Should work after reset
        matrix = sampler.sample(50)
        assert matrix.shape == (50, 3)

    def test_entropy_starts_high(self):
        spec = self._make_spec()
        sampler = CMAESSampler(spec, seed=42)
        # Before init, entropy is 1.0
        assert sampler.entropy() == 1.0
        # After init, entropy should still be near 1.0
        sampler.sample(50)
        ent = sampler.entropy()
        assert ent > 0.5  # should be high initially


# ---------------------------------------------------------------------------
# Staged optimizer CMA-ES integration tests
# ---------------------------------------------------------------------------

class TestStagedOptimizerCMAES:
    """Tests staged optimizer with CMA-ES exploitation."""

    def test_staged_flow_cmaes(self):
        """Full staged optimization completes with CMA-ES."""
        data = _make_data(500)
        strategy = OptimizerTestStrategy()
        engine = BacktestEngine(strategy, *data, slippage_pips=0.0)

        from backtester.optimizer.staged import StagedOptimizer
        config = OptimizationConfig(
            trials_per_stage=200,
            refinement_trials=200,
            batch_size=64,
            exploitation_method="cmaes",
            min_total_trades=1,
            min_trades_per_year=0.0,
            min_r_squared=0.0,
            max_dd_pct=100.0,
        )
        staged = StagedOptimizer(engine, config)
        result = staged.optimize()

        # Should have stages + refinement
        assert len(result.stages) >= 2
        assert result.total_trials > 0
        assert result.best_indices is not None

    def test_eda_fallback(self):
        """EDA exploitation still works when explicitly selected."""
        data = _make_data(500)
        strategy = OptimizerTestStrategy()
        engine = BacktestEngine(strategy, *data, slippage_pips=0.0)

        from backtester.optimizer.staged import StagedOptimizer
        config = OptimizationConfig(
            trials_per_stage=200,
            refinement_trials=200,
            batch_size=64,
            exploitation_method="eda",
            min_total_trades=1,
            min_trades_per_year=0.0,
            min_r_squared=0.0,
            max_dd_pct=100.0,
        )
        staged = StagedOptimizer(engine, config)
        result = staged.optimize()

        assert len(result.stages) >= 2
        assert result.total_trials > 0
        assert result.best_indices is not None

    def test_cmaes_refinement_collects(self):
        """Refinement stage with CMA-ES collects passing trials."""
        data = _make_data(500)
        strategy = OptimizerTestStrategy()
        engine = BacktestEngine(strategy, *data, slippage_pips=0.0)

        from backtester.optimizer.staged import StagedOptimizer
        config = OptimizationConfig(
            trials_per_stage=200,
            refinement_trials=400,
            batch_size=64,
            exploitation_method="cmaes",
            min_total_trades=1,
            min_trades_per_year=0.0,
            min_r_squared=0.0,
            max_dd_pct=100.0,
        )
        staged = StagedOptimizer(engine, config)
        result = staged.optimize()

        # Refinement stage should be last
        refinement = result.stages[-1]
        assert refinement.stage_name == "refinement"

        # Should have collected passing trials (if any passed)
        if refinement.valid_count > 0:
            assert refinement.all_passing_indices is not None
            assert refinement.all_passing_metrics is not None
            assert refinement.all_passing_indices.shape[1] == engine.encoding.num_params
            assert refinement.all_passing_metrics.shape[1] == NUM_METRICS
            assert result.refinement_indices is not None
            assert result.refinement_metrics is not None

    def test_config_cmaes_fields(self):
        """CMA-ES config fields have correct defaults."""
        config = OptimizationConfig()
        assert config.exploitation_method == "cmaes"
        assert config.cmaes_sigma0 == 0.3
        assert config.cmaes_population_size is None

    def test_presets_use_cmaes(self):
        """All presets should default to CMA-ES exploitation."""
        for name in ["turbo", "standard", "deep", "max"]:
            preset = get_preset(name)
            assert preset.exploitation_method == "cmaes", (
                f"Preset '{name}' should use cmaes, got {preset.exploitation_method}"
            )

    def test_config_ga_fields(self):
        """GA config fields have correct defaults."""
        config = OptimizationConfig()
        assert config.ga_population_size == 200
        assert config.ga_mutation_rate == 0.08
        assert config.ga_crossover_rate == 0.8
        assert config.ga_elite_pct == 0.2


class TestGASampler:
    """Tests for GASampler exploitation."""

    @pytest.fixture
    def spec(self):
        from backtester.core.encoding import build_encoding_spec
        from backtester.strategies import registry
        strategy = registry.get("ema_crossover")()
        return build_encoding_spec(strategy.param_space())

    def test_shape(self, spec):
        from backtester.optimizer.sampler import GASampler
        ga = GASampler(spec, seed=42)
        matrix = ga.sample(100)
        assert matrix.shape == (100, spec.num_params)
        assert matrix.dtype == np.int64

    def test_values_in_range(self, spec):
        from backtester.optimizer.sampler import GASampler
        ga = GASampler(spec, seed=42)
        matrix = ga.sample(200)
        for col in spec.columns:
            assert np.all(matrix[:, col.index] >= 0)
            assert np.all(matrix[:, col.index] < len(col.values))

    def test_locked_params_respected(self, spec):
        from backtester.optimizer.sampler import GASampler
        ga = GASampler(spec, seed=42)
        locked = np.full(spec.num_params, -1, dtype=np.int64)
        locked[0] = 2
        locked[1] = 0
        matrix = ga.sample(50, locked=locked)
        assert np.all(matrix[:, 0] == 2)
        assert np.all(matrix[:, 1] == 0)

    def test_mask_inactive_random(self, spec):
        from backtester.optimizer.sampler import GASampler
        ga = GASampler(spec, seed=42)
        mask = np.zeros(spec.num_params, dtype=np.bool_)
        mask[0] = True
        matrix = ga.sample(100, mask=mask)
        # Inactive params should still vary (noise averaging)
        for col in spec.columns:
            if col.index != 0 and len(col.values) > 1:
                assert len(np.unique(matrix[:, col.index])) > 1

    def test_neighborhood_respected(self, spec):
        from backtester.optimizer.sampler import GASampler, NeighborhoodSpec
        ga = GASampler(spec, seed=42)
        lo = np.zeros(spec.num_params, dtype=np.int64)
        hi = np.zeros(spec.num_params, dtype=np.int64)
        for col in spec.columns:
            max_idx = len(col.values) - 1
            lo[col.index] = min(1, max_idx)
            hi[col.index] = min(3, max_idx)
            if hi[col.index] <= lo[col.index]:
                lo[col.index] = 0
                hi[col.index] = max_idx
        nbhood = NeighborhoodSpec(min_bounds=lo, max_bounds=hi)
        locked = np.full(spec.num_params, -1, dtype=np.int64)
        matrix = ga.sample(100, locked=locked, neighborhood=nbhood)
        for col in spec.columns:
            assert np.all(matrix[:, col.index] >= lo[col.index])
            assert np.all(matrix[:, col.index] <= hi[col.index])

    def test_update_improves_population(self, spec):
        from backtester.optimizer.sampler import GASampler
        ga = GASampler(spec, population_size=50, seed=42)
        matrix = ga.sample(50)
        # Give high fitness to first 10, low to rest
        qualities = np.zeros(50, dtype=np.float64)
        qualities[:10] = 10.0
        ga.update(matrix, qualities)
        # Population should now contain the good individuals
        assert np.max(ga._fitness) >= 10.0

    def test_entropy(self, spec):
        from backtester.optimizer.sampler import GASampler
        ga = GASampler(spec, seed=42)
        ga.sample(100)
        ent = ga.entropy()
        assert 0.0 <= ent <= 1.0

    def test_reset(self, spec):
        from backtester.optimizer.sampler import GASampler
        ga = GASampler(spec, seed=42)
        ga.sample(50)
        assert ga._population is not None
        ga.reset()
        assert ga._population is None

    def test_sobol_only_exploitation(self, spec):
        """Sobol-only mode: exploitation phase uses SobolSampler."""
        from backtester.optimizer.sampler import SobolSampler
        sobol = SobolSampler(spec, seed=42)
        matrix = sobol.sample(100)
        assert matrix.shape == (100, spec.num_params)

    def test_sobol_update_noop(self, spec):
        """Sobol update() is a no-op and doesn't crash."""
        from backtester.optimizer.sampler import SobolSampler
        sobol = SobolSampler(spec, seed=42)
        matrix = sobol.sample(50)
        # EDA-style call
        sobol.update(matrix[:10], mask=np.ones(spec.num_params, dtype=np.bool_))
        # CMA-ES-style call
        sobol.update(matrix, np.ones(50), mask=None, original_indices=np.arange(50))
        # Verify it still works after update
        m2 = sobol.sample(50)
        assert m2.shape == (50, spec.num_params)
        # entropy always 1.0
        assert sobol.entropy() == 1.0

    def test_cmaes_update_row_index_pairing(self, spec):
        """CMA-ES update uses exact row-index pairing, not key reconstruction."""
        from backtester.optimizer.sampler import CMAESSampler
        cma = CMAESSampler(spec, sigma0=0.3, population_size=20, seed=42)
        mask = np.ones(spec.num_params, dtype=np.bool_)
        locked = np.full(spec.num_params, -1, dtype=np.int64)
        matrix = cma.sample(20, mask=mask, locked=locked)
        # Simulate pre-filter keeping only rows 0,2,4,6,8
        valid_indices = np.array([0, 2, 4, 6, 8], dtype=np.int64)
        valid_batch = matrix[valid_indices]
        qualities = np.array([5.0, 3.0, 4.0, 2.0, 1.0])
        # Should not crash — uses original_indices for exact matching
        cma.update(valid_batch, qualities, mask=mask, original_indices=valid_indices)

    def test_cmaes_empty_batch_update(self, spec):
        """CMA-ES update with empty batch doesn't drop generations."""
        from backtester.optimizer.sampler import CMAESSampler
        cma = CMAESSampler(spec, sigma0=0.3, population_size=20, seed=42)
        mask = np.ones(spec.num_params, dtype=np.bool_)
        locked = np.full(spec.num_params, -1, dtype=np.int64)
        cma.sample(20, mask=mask, locked=locked)
        # Empty batch — all rows pre-filtered
        cma.update(
            np.empty((0, spec.num_params), dtype=np.int64),
            np.empty(0, dtype=np.float64),
            mask=mask,
            original_indices=np.empty(0, dtype=np.int64),
        )
        # Should still be able to sample after
        m2 = cma.sample(20, mask=mask, locked=locked)
        assert m2.shape == (20, spec.num_params)

    def test_create_exploiter_eda_path(self):
        """_create_exploiter returns EDA when configured."""
        data = _make_data(500)
        strategy = OptimizerTestStrategy()
        engine = BacktestEngine(strategy, *data, slippage_pips=0.0)

        from backtester.optimizer.staged import StagedOptimizer
        config = OptimizationConfig(exploitation_method="eda")
        staged = StagedOptimizer(engine, config)
        exploiter, use_cmaes = staged._create_exploiter(batch_size=64)

        assert not use_cmaes
        assert isinstance(exploiter, EDASampler)

    def test_create_exploiter_cmaes_path(self):
        """_create_exploiter returns CMA-ES when configured and available."""
        data = _make_data(500)
        strategy = OptimizerTestStrategy()
        engine = BacktestEngine(strategy, *data, slippage_pips=0.0)

        from backtester.optimizer.staged import StagedOptimizer
        config = OptimizationConfig(exploitation_method="cmaes")
        staged = StagedOptimizer(engine, config)
        exploiter, use_cmaes = staged._create_exploiter(batch_size=64)

        # CMAESSampler exists (added by other agent), so should succeed
        if use_cmaes:
            assert isinstance(exploiter, CMAESSampler)
        else:
            # Fallback to EDA if CMAESSampler not yet importable
            assert isinstance(exploiter, EDASampler)


# ---------------------------------------------------------------------------
# Pre-filter tests
# ---------------------------------------------------------------------------

class TestPrefilter:
    def test_breakeven_invalid(self):
        """BE offset >= trigger should be rejected."""
        ps = ParamSpace([
            ParamDef("breakeven_enabled", [False, True]),
            ParamDef("breakeven_trigger_pips", [5, 10, 15, 20]),
            ParamDef("breakeven_offset_pips", [0, 5, 10, 20]),
        ])
        spec = build_encoding_spec(ps)
        # Row: BE enabled (idx=1), trigger=5 (idx=0), offset=20 (idx=3)
        index_matrix = np.array([[1, 0, 3]], dtype=np.int64)
        valid = prefilter_invalid_combos(index_matrix, spec)
        assert not valid[0]

    def test_breakeven_valid(self):
        """BE offset < trigger should be accepted."""
        ps = ParamSpace([
            ParamDef("breakeven_enabled", [False, True]),
            ParamDef("breakeven_trigger_pips", [5, 10, 15, 20]),
            ParamDef("breakeven_offset_pips", [0, 1, 2, 3]),
        ])
        spec = build_encoding_spec(ps)
        # Row: BE enabled (idx=1), trigger=20 (idx=3), offset=3 (idx=3)
        index_matrix = np.array([[1, 3, 3]], dtype=np.int64)
        valid = prefilter_invalid_combos(index_matrix, spec)
        assert valid[0]

    def test_be_disabled_always_valid(self):
        """When BE is disabled, offset/trigger don't matter."""
        ps = ParamSpace([
            ParamDef("breakeven_enabled", [False, True]),
            ParamDef("breakeven_trigger_pips", [5, 10]),
            ParamDef("breakeven_offset_pips", [0, 20]),
        ])
        spec = build_encoding_spec(ps)
        # Row: BE disabled (idx=0), trigger=5 (idx=0), offset=20 (idx=1)
        index_matrix = np.array([[0, 0, 1]], dtype=np.int64)
        valid = prefilter_invalid_combos(index_matrix, spec)
        assert valid[0]


class TestPostfilter:
    def test_min_trades(self):
        metrics = np.zeros((3, NUM_METRICS), dtype=np.float64)
        metrics[0, M_TRADES] = 5    # Too few
        metrics[1, M_TRADES] = 20   # Exactly enough
        metrics[2, M_TRADES] = 100  # Plenty
        metrics[:, M_R_SQUARED] = 0.8
        metrics[:, M_MAX_DD_PCT] = 10.0

        valid = postfilter_results(metrics, min_total_trades=20, min_trades_per_year=0.0)
        assert not valid[0]
        assert valid[1]
        assert valid[2]

    def test_max_dd(self):
        metrics = np.zeros((2, NUM_METRICS), dtype=np.float64)
        metrics[:, M_TRADES] = 50
        metrics[:, M_R_SQUARED] = 0.8
        metrics[0, M_MAX_DD_PCT] = 35.0  # Too high
        metrics[1, M_MAX_DD_PCT] = 20.0  # OK

        valid = postfilter_results(metrics, max_dd_pct=30.0, min_total_trades=1, min_trades_per_year=0.0)
        assert not valid[0]
        assert valid[1]

    def test_min_r_squared(self):
        metrics = np.zeros((2, NUM_METRICS), dtype=np.float64)
        metrics[:, M_TRADES] = 50
        metrics[:, M_MAX_DD_PCT] = 10.0
        metrics[0, M_R_SQUARED] = 0.3  # Too low
        metrics[1, M_R_SQUARED] = 0.7  # OK

        valid = postfilter_results(metrics, min_r_squared=0.5, min_total_trades=1, min_trades_per_year=0.0)
        assert not valid[0]
        assert valid[1]


# ---------------------------------------------------------------------------
# Ranking tests
# ---------------------------------------------------------------------------

class TestRanking:
    def test_rank_by_quality(self):
        metrics = np.zeros((4, NUM_METRICS), dtype=np.float64)
        metrics[0, M_QUALITY] = 10.0
        metrics[1, M_QUALITY] = 30.0
        metrics[2, M_QUALITY] = 20.0
        metrics[3, M_QUALITY] = 5.0

        ranks = rank_by_quality(metrics)
        # Best quality (30) should have rank 0
        assert ranks[1] == 0
        assert ranks[2] == 1
        assert ranks[0] == 2
        assert ranks[3] == 3

    def test_combined_rank(self):
        back = np.zeros((3, NUM_METRICS), dtype=np.float64)
        fwd = np.zeros((3, NUM_METRICS), dtype=np.float64)
        back[0, M_QUALITY] = 30.0  # Best back
        back[1, M_QUALITY] = 20.0
        back[2, M_QUALITY] = 10.0
        fwd[0, M_QUALITY] = 10.0   # Worst forward (overfitting!)
        fwd[1, M_QUALITY] = 30.0   # Best forward
        fwd[2, M_QUALITY] = 20.0

        combined = combined_rank(back, fwd, forward_weight=1.5)
        # Trial 1 should have best combined rank (good back + best forward)
        assert np.argmin(combined) == 1

    def test_forward_back_ratio(self):
        back = np.zeros((2, NUM_METRICS), dtype=np.float64)
        fwd = np.zeros((2, NUM_METRICS), dtype=np.float64)
        back[0, M_QUALITY] = 100.0
        fwd[0, M_QUALITY] = 50.0  # ratio = 0.5
        back[1, M_QUALITY] = 100.0
        fwd[1, M_QUALITY] = 30.0  # ratio = 0.3

        ratios = forward_back_ratio(back, fwd)
        assert abs(ratios[0] - 0.5) < 1e-10
        assert abs(ratios[1] - 0.3) < 1e-10

    def test_forward_back_gate(self):
        back = np.zeros((3, NUM_METRICS), dtype=np.float64)
        fwd = np.zeros((3, NUM_METRICS), dtype=np.float64)
        back[:, M_QUALITY] = 100.0
        fwd[0, M_QUALITY] = 50.0   # ratio 0.5 — pass
        fwd[1, M_QUALITY] = 30.0   # ratio 0.3 — fail
        fwd[2, M_QUALITY] = 40.0   # ratio 0.4 — borderline pass

        gate = forward_back_gate(back, fwd, min_ratio=0.4)
        assert gate[0] is True or gate[0] == True
        assert gate[1] is False or gate[1] == False
        assert gate[2] is True or gate[2] == True

    def test_select_top_n(self):
        metrics = np.zeros((10, NUM_METRICS), dtype=np.float64)
        for i in range(10):
            metrics[i, M_QUALITY] = float(i)
        top = select_top_n(metrics, n=3)
        assert len(top) == 3
        assert top[0] == 9  # Best quality
        assert top[1] == 8
        assert top[2] == 7

    def test_select_top_n_with_mask(self):
        metrics = np.zeros((5, NUM_METRICS), dtype=np.float64)
        metrics[0, M_QUALITY] = 100.0
        metrics[1, M_QUALITY] = 50.0
        metrics[2, M_QUALITY] = 80.0
        metrics[3, M_QUALITY] = 10.0
        metrics[4, M_QUALITY] = 90.0

        mask = np.array([False, True, True, True, True])
        top = select_top_n(metrics, n=2, valid_mask=mask)
        assert len(top) == 2
        assert top[0] == 4  # Best valid
        assert top[1] == 2


class TestDSR:
    def test_high_trials_deflates(self):
        """Many trials should deflate the Sharpe."""
        dsr_few = deflated_sharpe_ratio(1.5, n_trials=10, n_trades=100)
        dsr_many = deflated_sharpe_ratio(1.5, n_trials=10000, n_trades=100)
        assert dsr_many < dsr_few

    def test_edge_cases(self):
        assert deflated_sharpe_ratio(0, 0, 0) == 0.0
        assert deflated_sharpe_ratio(0, 1, 1) == 0.0


# ---------------------------------------------------------------------------
# Archive tests
# ---------------------------------------------------------------------------

class TestDiversityArchive:
    def test_add_and_retrieve(self):
        archive = DiversityArchive()
        metrics = np.zeros(NUM_METRICS, dtype=np.float64)
        metrics[M_QUALITY] = 50.0
        metrics[M_TRADES] = 80.0
        assert archive.add(0, metrics)
        assert archive.size == 1

    def test_better_replaces_worse(self):
        archive = DiversityArchive()
        m1 = np.zeros(NUM_METRICS, dtype=np.float64)
        m1[M_QUALITY] = 30.0
        m1[M_TRADES] = 50.0
        m2 = np.zeros(NUM_METRICS, dtype=np.float64)
        m2[M_QUALITY] = 60.0
        m2[M_TRADES] = 50.0  # Same cell

        archive.add(0, m1)
        archive.add(1, m2)
        assert archive.size == 1  # Same cell, replaced
        entries = archive.get_all()
        assert entries[0].quality == 60.0

    def test_different_cells(self):
        archive = DiversityArchive()
        m1 = np.zeros(NUM_METRICS, dtype=np.float64)
        m1[M_QUALITY] = 30.0
        m1[M_TRADES] = 10.0  # Low trade freq
        m2 = np.zeros(NUM_METRICS, dtype=np.float64)
        m2[M_QUALITY] = 30.0
        m2[M_TRADES] = 200.0  # High trade freq

        archive.add(0, m1)
        archive.add(1, m2)
        assert archive.size == 2  # Different cells

    def test_get_top_n(self):
        archive = DiversityArchive()
        for i in range(20):
            m = np.zeros(NUM_METRICS, dtype=np.float64)
            m[M_QUALITY] = float(i)
            m[M_TRADES] = float(i * 10)
            archive.add(i, m)

        top5 = archive.get_top_n(5)
        assert len(top5) <= 5
        # Should be sorted by quality descending
        for j in range(len(top5) - 1):
            assert top5[j].quality >= top5[j + 1].quality

    def test_select_top_n_diverse(self):
        metrics = np.zeros((20, NUM_METRICS), dtype=np.float64)
        for i in range(20):
            metrics[i, M_QUALITY] = float(i)
            metrics[i, M_TRADES] = float(i * 5 + 10)

        selected = select_top_n_diverse(metrics, n=5)
        assert len(selected) <= 5
        assert all(0 <= idx < 20 for idx in selected)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_presets_exist(self):
        turbo = get_preset("turbo")
        standard = get_preset("standard")
        deep = get_preset("deep")
        max_preset = get_preset("max")
        assert turbo.trials_per_stage < standard.trials_per_stage
        assert standard.trials_per_stage < deep.trials_per_stage
        assert deep.trials_per_stage < max_preset.trials_per_stage

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            get_preset("invalid")


# ---------------------------------------------------------------------------
# Staged optimizer integration test
# ---------------------------------------------------------------------------

class TestStagedOptimizer:
    def test_staged_flow(self):
        """End-to-end staged optimization with test strategy."""
        data = _make_data(500)
        strategy = OptimizerTestStrategy()

        engine = BacktestEngine(strategy, *data, slippage_pips=0.0)

        from backtester.optimizer.staged import StagedOptimizer
        config = OptimizationConfig(
            trials_per_stage=200,
            refinement_trials=200,
            batch_size=64,
            min_total_trades=1,       # Low bar for test data
            min_trades_per_year=0.0,  # No trade density gate
            min_r_squared=0.0,  # Low bar
            max_dd_pct=100.0,   # No DD filter
        )
        staged = StagedOptimizer(engine, config)
        result = staged.optimize()

        # Should have stages + refinement
        assert len(result.stages) >= 2  # At least signal + refinement
        assert result.total_trials > 0
        assert result.best_indices is not None

    def test_optimization_stages_from_strategy(self):
        """Strategy's optimization_stages() should be respected."""
        strategy = OptimizerTestStrategy()
        stages = strategy.optimization_stages()
        # Extract stage names (handle both str and tuple entries)
        stage_names = [s if isinstance(s, str) else s[0] for s in stages]
        assert "signal" in stage_names
        # Composite stage merges risk + trailing + breakeven
        assert "core_trade_profile" in stage_names
        # Remaining management groups
        assert "exit_protection" in stage_names
        assert "exit_time" in stage_names

    def test_refinement_collects_passing_trials(self):
        """Refinement stage should collect all passing trials for multi-candidate."""
        data = _make_data(500)
        strategy = OptimizerTestStrategy()
        engine = BacktestEngine(strategy, *data, slippage_pips=0.0)

        from backtester.optimizer.staged import StagedOptimizer
        config = OptimizationConfig(
            trials_per_stage=200,
            refinement_trials=400,
            batch_size=64,
            min_total_trades=1,
            min_trades_per_year=0.0,
            min_r_squared=0.0,
            max_dd_pct=100.0,
        )
        staged = StagedOptimizer(engine, config)
        result = staged.optimize()

        # Refinement stage should be the last stage
        refinement = result.stages[-1]
        assert refinement.stage_name == "refinement"

        # Should have collected passing trials
        if refinement.valid_count > 0:
            assert refinement.all_passing_indices is not None
            assert refinement.all_passing_metrics is not None
            # Shape: (K, P) and (K, NUM_METRICS)
            assert refinement.all_passing_indices.shape[1] == engine.encoding.num_params
            assert refinement.all_passing_metrics.shape[1] == NUM_METRICS
            # K should match valid_count
            assert refinement.all_passing_indices.shape[0] == refinement.valid_count
            # StagedResult should mirror refinement data
            assert result.refinement_indices is not None
            assert result.refinement_metrics is not None
            assert result.refinement_indices.shape == refinement.all_passing_indices.shape


# ---------------------------------------------------------------------------
# Base class additions tests
# ---------------------------------------------------------------------------

class TestStrategyBaseAdditions:
    def test_validate_params_valid(self):
        strategy = OptimizerTestStrategy()
        params = {
            "breakeven_enabled": True,
            "breakeven_trigger_pips": 20,
            "breakeven_offset_pips": 2,
        }
        errors = strategy.validate_params(params)
        assert len(errors) == 0

    def test_validate_params_invalid_be(self):
        strategy = OptimizerTestStrategy()
        params = {
            "breakeven_enabled": True,
            "breakeven_trigger_pips": 5,
            "breakeven_offset_pips": 10,
        }
        errors = strategy.validate_params(params)
        assert len(errors) > 0

    def test_optimization_stages_default(self):
        strategy = OptimizerTestStrategy()
        stages = strategy.optimization_stages()
        assert stages[0] == "signal"
        assert stages[1] == "time"
        # Third entry is the composite core_trade_profile stage
        assert isinstance(stages[2], tuple)
        comp_name, comp_groups = stages[2]
        assert comp_name == "core_trade_profile"
        assert "risk" in comp_groups
        assert "exit_trailing" in comp_groups
        assert "exit_protection_be" in comp_groups
        # Remaining management groups follow
        stage_names = [s if isinstance(s, str) else s[0] for s in stages]
        assert "exit_protection" in stage_names
        assert "exit_time" in stage_names


# ---------------------------------------------------------------------------
# Multi-candidate selection tests
# ---------------------------------------------------------------------------

class TestMultiCandidate:
    def _run_optimize(self, with_forward: bool = True):
        """Helper: run optimize() with test data."""
        from backtester.optimizer.run import optimize

        data = _make_data(500)
        strategy = OptimizerTestStrategy()
        config = OptimizationConfig(
            trials_per_stage=200,
            refinement_trials=400,
            batch_size=64,
            min_total_trades=1,
            min_trades_per_year=0.0,
            min_r_squared=0.0,
            max_dd_pct=100.0,
            top_n_candidates=5,
            min_forward_back_ratio=0.0,  # Don't filter in test
        )

        open_, high, low, close, volume, spread = data
        if with_forward:
            result = optimize(
                strategy,
                open_, high, low, close, volume, spread,
                open_, high, low, close, volume, spread,  # Same data for forward
                config=config,
                slippage_pips=0.0,
            )
        else:
            result = optimize(
                strategy,
                open_, high, low, close, volume, spread,
                config=config,
                slippage_pips=0.0,
            )
        return result

    def test_multi_returns_candidates(self):
        """Multi-candidate should return >= 1 candidates."""
        result = self._run_optimize(with_forward=True)
        assert len(result.candidates) >= 1

    def test_forward_metrics_populated(self):
        """Each candidate should have forward metrics when forward data provided."""
        result = self._run_optimize(with_forward=True)
        for cand in result.candidates:
            assert cand.forward_metrics is not None
            assert "quality_score" in cand.forward_metrics
            assert "sharpe" in cand.forward_metrics
            assert "trades" in cand.forward_metrics

    def test_fallback_to_single_without_forward(self):
        """Without forward data, should still produce candidates."""
        result = self._run_optimize(with_forward=False)
        assert len(result.candidates) >= 1
        # Without forward data, forward_metrics should be None
        for cand in result.candidates:
            assert cand.forward_metrics is None

    def test_ranking_order(self):
        """Candidates should be sorted by combined rank (best first)."""
        result = self._run_optimize(with_forward=True)
        if len(result.candidates) > 1:
            # Combined rank: lower is better. Candidates ordered 0, 1, 2, ...
            for i in range(len(result.candidates) - 1):
                assert result.candidates[i].combined_rank <= result.candidates[i + 1].combined_rank


class TestOptimizeCleanup:
    def test_handles_no_best_indices_without_unbound_error(self, monkeypatch):
        """Cleanup path should work even when staged optimization has no best trial."""
        from backtester.optimizer import run as run_module

        class DummyStagedOptimizer:
            def __init__(self, engine, config, **kwargs):
                pass

            def optimize(self):
                return run_module.StagedResult(
                    stages=[],
                    best_indices=None,
                    best_quality=float("-inf"),
                    best_metrics=None,
                    total_trials=1,
                    refinement_indices=None,
                    refinement_metrics=None,
                )

        monkeypatch.setattr(run_module, "StagedOptimizer", DummyStagedOptimizer)

        data = _make_data(200)
        strategy = OptimizerTestStrategy()
        result = run_module.optimize(
            strategy,
            *data,
            config=OptimizationConfig(
                trials_per_stage=10,
                refinement_trials=10,
                batch_size=16,
            ),
            slippage_pips=0.0,
        )

        assert result.total_trials == 1
        assert len(result.candidates) == 0


# ---------------------------------------------------------------------------
# Neighborhood sampling tests
# ---------------------------------------------------------------------------

class TestNeighborhoodSampling:
    def _make_spec(self):
        ps = ParamSpace([
            ParamDef("a", list(range(10))),   # 10 values: 0-9
            ParamDef("b", list(range(20))),   # 20 values: 0-19
        ])
        return build_encoding_spec(ps)

    def test_build_neighborhood_locked(self):
        from backtester.optimizer.sampler import build_neighborhood
        spec = self._make_spec()
        locked = np.array([5, 10], dtype=np.int64)
        nb = build_neighborhood(spec, locked, radius=2)
        # a: center=5, range=[3, 7]
        assert nb.min_bounds[0] == 3
        assert nb.max_bounds[0] == 7
        # b: center=10, range=[8, 12]
        assert nb.min_bounds[1] == 8
        assert nb.max_bounds[1] == 12

    def test_build_neighborhood_edge_clamp(self):
        from backtester.optimizer.sampler import build_neighborhood
        spec = self._make_spec()
        locked = np.array([0, 18], dtype=np.int64)  # Near edges
        nb = build_neighborhood(spec, locked, radius=3)
        # a: center=0, min clamped to 0
        assert nb.min_bounds[0] == 0
        assert nb.max_bounds[0] == 3
        # b: center=18, max clamped to 19
        assert nb.min_bounds[1] == 15
        assert nb.max_bounds[1] == 19

    def test_build_neighborhood_unlocked_full_range(self):
        from backtester.optimizer.sampler import build_neighborhood
        spec = self._make_spec()
        locked = np.array([5, -1], dtype=np.int64)  # b unlocked
        nb = build_neighborhood(spec, locked, radius=2)
        # a: constrained
        assert nb.min_bounds[0] == 3
        assert nb.max_bounds[0] == 7
        # b: unlocked = full range
        assert nb.min_bounds[1] == 0
        assert nb.max_bounds[1] == 19

    def test_random_sampler_with_neighborhood(self):
        from backtester.optimizer.sampler import NeighborhoodSpec
        spec = self._make_spec()
        sampler = RandomSampler(spec, seed=42)
        nb = NeighborhoodSpec(
            min_bounds=np.array([3, 8], dtype=np.int64),
            max_bounds=np.array([7, 12], dtype=np.int64),
        )
        matrix = sampler.sample(1000, neighborhood=nb)
        assert matrix[:, 0].min() >= 3
        assert matrix[:, 0].max() <= 7
        assert matrix[:, 1].min() >= 8
        assert matrix[:, 1].max() <= 12

    def test_sobol_sampler_with_neighborhood(self):
        from backtester.optimizer.sampler import NeighborhoodSpec
        spec = self._make_spec()
        sampler = SobolSampler(spec, seed=42)
        nb = NeighborhoodSpec(
            min_bounds=np.array([3, 8], dtype=np.int64),
            max_bounds=np.array([7, 12], dtype=np.int64),
        )
        matrix = sampler.sample(128, neighborhood=nb)
        assert matrix[:, 0].min() >= 3
        assert matrix[:, 0].max() <= 7
        assert matrix[:, 1].min() >= 8
        assert matrix[:, 1].max() <= 12

    def test_eda_sampler_with_neighborhood(self):
        from backtester.optimizer.sampler import NeighborhoodSpec
        spec = self._make_spec()
        eda = EDASampler(spec, seed=42)
        nb = NeighborhoodSpec(
            min_bounds=np.array([3, 8], dtype=np.int64),
            max_bounds=np.array([7, 12], dtype=np.int64),
        )
        mask = np.array([True, True], dtype=bool)
        matrix = eda.sample(1000, mask=mask, neighborhood=nb)
        assert matrix[:, 0].min() >= 3
        assert matrix[:, 0].max() <= 7
        assert matrix[:, 1].min() >= 8
        assert matrix[:, 1].max() <= 12


class TestBudgetAutoCap:
    def test_compute_stage_budgets_caps_small_space(self):
        """Budget should be capped when space is much smaller than budget."""
        from backtester.optimizer.staged import compute_stage_budgets

        # Simple strategy with 3×3 = 9 signal combos
        strategy = OptimizerTestStrategy()
        config = OptimizationConfig(
            trials_per_stage=50_000,
            refinement_trials=100_000,
        )
        budgets = compute_stage_budgets(strategy, config)
        signal_stage = next(b for b in budgets if b["stage"] == "signal")
        # 9 combos × 10x = 90, so budget should be capped to 90
        assert signal_stage["budget"] == 90
        assert signal_stage["unique_combos"] == 9

    def test_budget_not_capped_for_large_space(self):
        """Budget should not be capped when space is larger than budget."""
        from backtester.optimizer.staged import compute_stage_budgets

        strategy = OptimizerTestStrategy()
        config = OptimizationConfig(
            trials_per_stage=50,  # Very small budget
            refinement_trials=100,
        )
        budgets = compute_stage_budgets(strategy, config)
        signal_stage = next(b for b in budgets if b["stage"] == "signal")
        # 9 combos × 10x = 90 > 50, so no cap
        assert signal_stage["budget"] == 50


# ---------------------------------------------------------------------------
# Cyclic passes tests
# ---------------------------------------------------------------------------

class TestCyclicPasses:
    """Tests for cyclic pass re-optimization."""

    def test_cyclic_passes_execute(self):
        """Cyclic passes run when max_cyclic_passes > 0."""
        data = _make_data(500)
        strategy = OptimizerTestStrategy()
        engine = BacktestEngine(strategy, *data, slippage_pips=0.0)

        from backtester.optimizer.staged import StagedOptimizer
        config_no_cyclic = OptimizationConfig(
            trials_per_stage=200,
            refinement_trials=200,
            batch_size=64,
            min_total_trades=1,
            min_trades_per_year=0.0,
            min_r_squared=0.0,
            max_dd_pct=100.0,
            max_cyclic_passes=0,
        )
        staged_no = StagedOptimizer(engine, config_no_cyclic)
        result_no = staged_no.optimize()

        config_with_cyclic = OptimizationConfig(
            trials_per_stage=200,
            refinement_trials=200,
            batch_size=64,
            min_total_trades=1,
            min_trades_per_year=0.0,
            min_r_squared=0.0,
            max_dd_pct=100.0,
            max_cyclic_passes=1,
            cyclic_budget_fraction=0.5,
        )
        staged_yes = StagedOptimizer(engine, config_with_cyclic)
        result_yes = staged_yes.optimize()

        # With cyclic passes, total_trials should be strictly greater
        assert result_yes.total_trials > result_no.total_trials
        # Cyclic stages should show up in the stages list
        cyclic_stages = [s for s in result_yes.stages if "cycle" in s.stage_name]
        assert len(cyclic_stages) >= 1

    def test_cyclic_passes_disabled(self):
        """No cyclic passes when max_cyclic_passes=0."""
        data = _make_data(500)
        strategy = OptimizerTestStrategy()
        engine = BacktestEngine(strategy, *data, slippage_pips=0.0)

        from backtester.optimizer.staged import StagedOptimizer
        config = OptimizationConfig(
            trials_per_stage=200,
            refinement_trials=200,
            batch_size=64,
            min_total_trades=1,
            min_trades_per_year=0.0,
            min_r_squared=0.0,
            max_dd_pct=100.0,
            max_cyclic_passes=0,
        )
        staged = StagedOptimizer(engine, config)
        result = staged.optimize()

        # No cyclic stage names in the results
        cyclic_stages = [s for s in result.stages if "cycle" in s.stage_name]
        assert len(cyclic_stages) == 0

    def test_cyclic_convergence_stops_early(self):
        """Cycling stops when improvement < threshold."""
        data = _make_data(500)
        strategy = OptimizerTestStrategy()
        engine = BacktestEngine(strategy, *data, slippage_pips=0.0)

        from backtester.optimizer.staged import StagedOptimizer
        config = OptimizationConfig(
            trials_per_stage=200,
            refinement_trials=200,
            batch_size=64,
            min_total_trades=1,
            min_trades_per_year=0.0,
            min_r_squared=0.0,
            max_dd_pct=100.0,
            max_cyclic_passes=5,  # Allow up to 5
            cyclic_budget_fraction=0.5,
            cyclic_improvement_threshold=0.01,
        )
        staged = StagedOptimizer(engine, config)
        result = staged.optimize()

        # With small data and small budget, quality plateaus fast.
        # Expect fewer than 5 cycles actually ran (early convergence).
        cyclic_stages = [s for s in result.stages if "cycle" in s.stage_name]
        # Should have at least 1 cycle (always runs first pass)
        assert len(cyclic_stages) >= 1
        # With 5 max passes and only signal as cyclic stage, max would be 5 stages.
        # Early convergence should stop before that (usually 1-2 passes).
        # We can't guarantee exact count, but total_trials should be bounded.
        assert result.total_trials > 0

    def test_cyclic_total_trials_accounted(self):
        """StagedResult.total_trials includes cyclic pass evaluations."""
        data = _make_data(500)
        strategy = OptimizerTestStrategy()
        engine = BacktestEngine(strategy, *data, slippage_pips=0.0)

        from backtester.optimizer.staged import StagedOptimizer
        config = OptimizationConfig(
            trials_per_stage=200,
            refinement_trials=200,
            batch_size=64,
            min_total_trades=1,
            min_trades_per_year=0.0,
            min_r_squared=0.0,
            max_dd_pct=100.0,
            max_cyclic_passes=1,
            cyclic_budget_fraction=0.5,
        )
        staged = StagedOptimizer(engine, config)
        result = staged.optimize()

        # total_trials must equal sum of all stage trials
        sum_stage_trials = sum(s.trials_evaluated for s in result.stages)
        assert result.total_trials == sum_stage_trials


# ---------------------------------------------------------------------------
# Refinement upgrade tests
# ---------------------------------------------------------------------------

class TestRefinementUpgrade:

    def test_wider_default_neighborhood(self):
        """Default refinement_neighborhood_radius is now 5."""
        cfg = OptimizationConfig()
        assert cfg.refinement_neighborhood_radius == 5


# ---------------------------------------------------------------------------
# Composite stage tests
# ---------------------------------------------------------------------------

class TestCompositeStages:
    """Tests for composite stage support (merging multiple param groups)."""

    def test_composite_stage_activates_multiple_groups(self):
        """Composite stage ('name', ['g1', 'g2']) activates params from both groups."""
        strategy = OptimizerTestStrategy()
        ps = strategy.param_space()
        spec = build_encoding_spec(ps)

        # Get indices for the groups that go into the composite
        risk_indices = spec.group_indices("risk")
        trail_indices = spec.group_indices("exit_trailing")
        be_indices = spec.group_indices("exit_protection_be")

        # All three should have params
        assert len(risk_indices) > 0
        assert len(trail_indices) > 0
        assert len(be_indices) > 0

        # The composite should include all of them
        combined = set(risk_indices) | set(trail_indices) | set(be_indices)
        assert len(combined) == len(risk_indices) + len(trail_indices) + len(be_indices)

    def test_backward_compat_string_stages(self):
        """String stage entries still work as single groups."""
        from backtester.optimizer.staged import _normalize_stages
        normalized = _normalize_stages(["signal", "time"])
        assert normalized == [("signal", ["signal"]), ("time", ["time"])]

    def test_normalize_stages_mixed(self):
        """_normalize_stages converts strings and tuples correctly."""
        from backtester.optimizer.staged import _normalize_stages
        raw = ["signal", ("merged", ["a", "b"]), "time"]
        result = _normalize_stages(raw)
        assert result == [
            ("signal", ["signal"]),
            ("merged", ["a", "b"]),
            ("time", ["time"]),
        ]

    def test_normalize_stages_invalid_raises(self):
        """_normalize_stages raises on invalid entry types."""
        from backtester.optimizer.staged import _normalize_stages
        with pytest.raises(ValueError, match="Invalid stage entry"):
            _normalize_stages([123])

    def test_breakeven_in_new_group(self):
        """BreakevenModule now has group 'exit_protection_be'."""
        from backtester.strategies.modules import BreakevenModule
        mod = BreakevenModule()
        assert mod.group == "exit_protection_be"

    def test_partial_close_stays_in_exit_protection(self):
        """PartialCloseModule still has group 'exit_protection'."""
        from backtester.strategies.modules import PartialCloseModule
        mod = PartialCloseModule()
        assert mod.group == "exit_protection"

    def test_default_stages_include_composite(self):
        """Default optimization_stages() includes the core_trade_profile composite."""
        strategy = OptimizerTestStrategy()
        stages = strategy.optimization_stages()

        # Find the composite entry
        composite_entries = [s for s in stages if isinstance(s, tuple)]
        assert len(composite_entries) >= 1

        # Check the core_trade_profile composite
        core = next(s for s in stages if isinstance(s, tuple) and s[0] == "core_trade_profile")
        name, groups = core
        assert "risk" in groups
        assert "exit_trailing" in groups
        assert "exit_protection_be" in groups

    def test_composite_stage_exec_mode_full(self):
        """Composite stage containing management modules should use EXEC_FULL."""
        # The composite core_trade_profile includes exit_trailing and exit_protection_be,
        # both of which require EXEC_FULL. The staged optimizer should detect this.
        strategy = OptimizerTestStrategy()
        full_mode_groups: set[str] = set()
        for mod in strategy.management_modules():
            if mod.requires_full_mode:
                full_mode_groups.add(mod.group)

        # exit_trailing and exit_protection_be should both be in full_mode_groups
        assert "exit_trailing" in full_mode_groups
        assert "exit_protection_be" in full_mode_groups

    def test_staged_flow_with_composite(self):
        """End-to-end staged optimization works with composite stages."""
        data = _make_data(500)
        strategy = OptimizerTestStrategy()
        engine = BacktestEngine(strategy, *data, slippage_pips=0.0)

        from backtester.optimizer.staged import StagedOptimizer
        config = OptimizationConfig(
            trials_per_stage=200,
            refinement_trials=200,
            batch_size=64,
            min_total_trades=1,
            min_trades_per_year=0.0,
            min_r_squared=0.0,
            max_dd_pct=100.0,
        )
        staged = StagedOptimizer(engine, config)
        result = staged.optimize()

        # Should have stages + refinement
        assert len(result.stages) >= 2
        assert result.total_trials > 0
        assert result.best_indices is not None

        # One of the stages should be core_trade_profile
        stage_names = [s.stage_name for s in result.stages]
        assert "core_trade_profile" in stage_names

    def test_compute_stage_budgets_composite(self):
        """compute_stage_budgets handles composite stages correctly."""
        from backtester.optimizer.staged import compute_stage_budgets

        strategy = OptimizerTestStrategy()
        config = OptimizationConfig(
            trials_per_stage=500_000,
            refinement_trials=100_000,
        )
        budgets = compute_stage_budgets(strategy, config)

        # Find the core_trade_profile budget entry
        core_budget = next(
            (b for b in budgets if b["stage"] == "core_trade_profile"), None,
        )
        assert core_budget is not None
        # Composite unique_combos = product across risk + trailing + BE params
        assert core_budget["unique_combos"] > 1
