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
from backtester.optimizer.sampler import EDASampler, RandomSampler, SobolSampler
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

        valid = postfilter_results(metrics, min_trades=20)
        assert not valid[0]
        assert valid[1]
        assert valid[2]

    def test_max_dd(self):
        metrics = np.zeros((2, NUM_METRICS), dtype=np.float64)
        metrics[:, M_TRADES] = 50
        metrics[:, M_R_SQUARED] = 0.8
        metrics[0, M_MAX_DD_PCT] = 35.0  # Too high
        metrics[1, M_MAX_DD_PCT] = 20.0  # OK

        valid = postfilter_results(metrics, max_dd_pct=30.0)
        assert not valid[0]
        assert valid[1]

    def test_min_r_squared(self):
        metrics = np.zeros((2, NUM_METRICS), dtype=np.float64)
        metrics[:, M_TRADES] = 50
        metrics[:, M_MAX_DD_PCT] = 10.0
        metrics[0, M_R_SQUARED] = 0.3  # Too low
        metrics[1, M_R_SQUARED] = 0.7  # OK

        valid = postfilter_results(metrics, min_r_squared=0.5)
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
            min_trades=1,       # Low bar for test data
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
        assert "signal" in stages
        assert "management" in stages

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
            min_trades=1,
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
        assert stages == ["signal", "time", "risk", "management"]


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
            min_trades=1,
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
