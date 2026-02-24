"""Tests for CPCV (Combinatorial Purged Cross-Validation)."""

import math

import numpy as np
import pytest

from backtester.core.dtypes import NUM_METRICS
from backtester.pipeline.cpcv import (
    build_fold_masks,
    cpcv_validate,
    evaluate_candidate_on_fold,
    generate_blocks,
    generate_folds,
)
from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.types import CPCVFoldResult, CPCVResult


# ---------------------------------------------------------------------------
# Block generation
# ---------------------------------------------------------------------------

class TestGenerateBlocks:
    def test_equal_division(self):
        """10 blocks from 1000 bars should be exactly 100 each."""
        blocks = generate_blocks(1000, 10)
        assert len(blocks) == 10
        for start, end in blocks:
            assert end - start == 100
        # Contiguous: each start == previous end
        for i in range(1, len(blocks)):
            assert blocks[i][0] == blocks[i - 1][1]
        # Full coverage
        assert blocks[0][0] == 0
        assert blocks[-1][1] == 1000

    def test_remainder_handling(self):
        """1003 bars / 10 blocks: first 3 blocks get 101, rest get 100."""
        blocks = generate_blocks(1003, 10)
        assert len(blocks) == 10
        sizes = [end - start for start, end in blocks]
        assert sum(sizes) == 1003
        # First 3 blocks should be 101, rest 100
        assert sizes[:3] == [101, 101, 101]
        assert sizes[3:] == [100] * 7

    def test_single_block(self):
        blocks = generate_blocks(500, 1)
        assert len(blocks) == 1
        assert blocks[0] == (0, 500)

    def test_blocks_equal_bars(self):
        """N blocks from N bars = each block is 1 bar."""
        blocks = generate_blocks(5, 5)
        assert len(blocks) == 5
        for i, (start, end) in enumerate(blocks):
            assert start == i
            assert end == i + 1

    def test_empty_cases(self):
        assert generate_blocks(0, 10) == []
        assert generate_blocks(100, 0) == []


# ---------------------------------------------------------------------------
# Fold generation
# ---------------------------------------------------------------------------

class TestGenerateFolds:
    def test_c_10_2(self):
        """C(10,2) = 45 folds."""
        folds = generate_folds(10, 2)
        assert len(folds) == 45

    def test_c_5_1(self):
        """C(5,1) = 5 folds (leave-one-out)."""
        folds = generate_folds(5, 1)
        assert len(folds) == 5

    def test_c_4_2(self):
        """C(4,2) = 6 folds."""
        folds = generate_folds(4, 2)
        assert len(folds) == 6

    def test_train_test_disjoint(self):
        """Train and test block indices should never overlap."""
        folds = generate_folds(8, 3)
        for train_idxs, test_idxs in folds:
            assert set(train_idxs) & set(test_idxs) == set()

    def test_all_blocks_covered(self):
        """Each block should appear as test in at least one fold."""
        folds = generate_folds(6, 2)
        test_blocks_seen = set()
        for _, test_idxs in folds:
            test_blocks_seen.update(test_idxs)
        assert test_blocks_seen == set(range(6))

    def test_correct_train_size(self):
        """Train should contain n_blocks - k_test blocks."""
        folds = generate_folds(10, 3)
        for train_idxs, test_idxs in folds:
            assert len(train_idxs) == 7
            assert len(test_idxs) == 3


# ---------------------------------------------------------------------------
# Fold masks (purging + embargo)
# ---------------------------------------------------------------------------

class TestBuildFoldMasks:
    def _make_blocks(self, n_bars, n_blocks):
        return generate_blocks(n_bars, n_blocks)

    def test_basic_no_purge_no_embargo(self):
        """Without purge/embargo, train + test should cover all bars."""
        blocks = self._make_blocks(100, 5)  # 5 blocks of 20
        train_mask, test_mask, n_purged = build_fold_masks(
            blocks, train_indices=(0, 1, 2), test_indices=(3, 4),
            n_bars=100, purge_bars=0, embargo_bars=0,
        )
        assert test_mask.sum() == 40  # 2 blocks × 20
        assert train_mask.sum() == 60  # 3 blocks × 20
        assert n_purged == 0
        # No overlap
        assert not np.any(train_mask & test_mask)

    def test_purge_removes_near_test(self):
        """Purge should remove bars from train near test boundaries."""
        blocks = self._make_blocks(100, 5)  # 5 blocks of 20
        # Test block 2 = [40, 60)
        train_mask, test_mask, n_purged = build_fold_masks(
            blocks, train_indices=(0, 1, 3, 4), test_indices=(2,),
            n_bars=100, purge_bars=5, embargo_bars=0,
        )
        # Should purge 5 bars before test (35-39) and 5 bars after (60-64)
        assert not train_mask[35:40].any()  # Purged before test
        assert not train_mask[60:65].any()  # Purged after test
        # Test block should be intact
        assert test_mask[40:60].all()
        assert n_purged > 0

    def test_embargo_removes_after_test(self):
        """Embargo should remove additional bars after test + purge."""
        blocks = self._make_blocks(200, 4)  # 4 blocks of 50
        # Test block 1 = [50, 100)
        train_mask, _, _ = build_fold_masks(
            blocks, train_indices=(0, 2, 3), test_indices=(1,),
            n_bars=200, purge_bars=5, embargo_bars=10,
        )
        # After test end (100): purge 5 + embargo 10 = 15 bars removed
        assert not train_mask[100:115].any()
        # But bar 115+ should be train (block 2 starts at 100, block 3 at 150)
        assert train_mask[115:150].any()

    def test_no_overlap(self):
        """Train and test masks should never overlap."""
        blocks = self._make_blocks(500, 10)
        for test_combo in [(0, 5), (2, 7), (4, 9)]:
            train_idxs = tuple(i for i in range(10) if i not in test_combo)
            train_mask, test_mask, _ = build_fold_masks(
                blocks, train_idxs, test_combo,
                n_bars=500, purge_bars=20, embargo_bars=10,
            )
            assert not np.any(train_mask & test_mask)

    def test_edge_blocks(self):
        """Purge near block 0 and last block shouldn't crash."""
        blocks = self._make_blocks(100, 5)
        # Test block 0 (start of data)
        train_mask, test_mask, _ = build_fold_masks(
            blocks, train_indices=(1, 2, 3, 4), test_indices=(0,),
            n_bars=100, purge_bars=10, embargo_bars=5,
        )
        assert test_mask[:20].all()
        assert not np.any(train_mask & test_mask)

        # Test last block (end of data)
        train_mask, test_mask, _ = build_fold_masks(
            blocks, train_indices=(0, 1, 2, 3), test_indices=(4,),
            n_bars=100, purge_bars=10, embargo_bars=5,
        )
        assert test_mask[80:100].all()
        assert not np.any(train_mask & test_mask)


# ---------------------------------------------------------------------------
# Strategy fixture for CPCV evaluation tests
# ---------------------------------------------------------------------------

from backtester.strategies.base import (
    Direction, ParamDef, ParamSpace, Signal, SLTPResult, Strategy,
    risk_params, management_params, time_params,
)


class CPCVTestStrategy(Strategy):
    """Simple strategy for CPCV tests — generates a signal every 50 bars."""

    @property
    def name(self) -> str:
        return "cpcv_test"

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
        for i in range(50, len(close) - 50, 50):
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
            sl_pips=30.0, tp_pips=60.0,
        )


def _make_trending_data(n_bars: int = 2000, pip: float = 0.0001):
    """Create uptrending price data for CPCV tests."""
    rng = np.random.default_rng(42)
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
    spread = np.ones(n_bars, dtype=np.float64)
    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "spread": spread,
        "bar_hour": np.tile(np.arange(24), n_bars // 24 + 1)[:n_bars].astype(np.int64),
        "bar_day_of_week": np.tile(np.arange(5), n_bars // 5 + 1)[:n_bars].astype(np.int64),
    }


def _make_flat_data(n_bars: int = 2000, pip: float = 0.0001):
    """Create flat/random price data (no trend)."""
    rng = np.random.default_rng(99)
    base = 1.1000
    open_ = np.full(n_bars, base, dtype=np.float64)
    high = np.full(n_bars, base + 10 * pip, dtype=np.float64)
    low = np.full(n_bars, base - 10 * pip, dtype=np.float64)
    close = base + rng.normal(0, 2 * pip, n_bars)
    volume = np.ones(n_bars, dtype=np.float64)
    spread = np.ones(n_bars, dtype=np.float64)
    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "spread": spread,
        "bar_hour": np.tile(np.arange(24), n_bars // 24 + 1)[:n_bars].astype(np.int64),
        "bar_day_of_week": np.tile(np.arange(5), n_bars // 5 + 1)[:n_bars].astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Fold evaluation tests
# ---------------------------------------------------------------------------

def _cpcv_default_params():
    """Default parameter dict matching CPCVTestStrategy's param_space."""
    return {
        "fast_period": 10, "slow_period": 50,
        "sl_mode": "fixed_pips", "sl_fixed_pips": 30,
        "sl_atr_mult": 1.5,
        "tp_mode": "rr_ratio", "tp_rr_ratio": 2.0,
        "tp_atr_mult": 2.0, "tp_fixed_pips": 60,
        "trailing_mode": "off",
        "trail_activate_pips": 0, "trail_distance_pips": 10,
        "trail_atr_mult": 2.0,
        "breakeven_enabled": False,
        "breakeven_trigger_pips": 20, "breakeven_offset_pips": 2,
        "partial_close_enabled": False,
        "partial_close_pct": 50, "partial_close_trigger_pips": 30,
        "max_bars": 0,
        "stale_exit_enabled": False,
        "stale_exit_bars": 50, "stale_exit_atr_threshold": 0.5,
        "allowed_hours_start": 0, "allowed_hours_end": 23,
        "allowed_days": [0, 1, 2, 3, 4],
    }


class TestEvaluateCandidateOnFold:
    def test_single_block_fold(self):
        """Evaluate on a fold with 1 test block."""
        strategy = CPCVTestStrategy()
        data = _make_trending_data(2000)
        blocks = generate_blocks(2000, 5)
        config = PipelineConfig(wf_min_trades_per_window=1, commission_pips=0.0, max_spread_pips=0.0)

        params = _cpcv_default_params()

        result = evaluate_candidate_on_fold(
            strategy, params, data,
            blocks=blocks,
            test_indices=(2,),
            train_indices=(0, 1, 3, 4),
            lookback_prefix=100,
            config=config,
            fold_index=0,
        )

        assert isinstance(result, CPCVFoldResult)
        assert result.fold_index == 0
        assert result.test_blocks == (2,)
        assert result.train_blocks == (0, 1, 3, 4)

    def test_two_block_fold(self):
        """Evaluate on a fold with 2 test blocks."""
        strategy = CPCVTestStrategy()
        data = _make_trending_data(2000)
        blocks = generate_blocks(2000, 5)
        config = PipelineConfig(wf_min_trades_per_window=1, commission_pips=0.0, max_spread_pips=0.0)

        params = _cpcv_default_params()

        result = evaluate_candidate_on_fold(
            strategy, params, data,
            blocks=blocks,
            test_indices=(1, 3),
            train_indices=(0, 2, 4),
            lookback_prefix=100,
            config=config,
            fold_index=5,
        )

        assert result.fold_index == 5
        assert result.test_blocks == (1, 3)


# ---------------------------------------------------------------------------
# Full CPCV validation tests
# ---------------------------------------------------------------------------

class TestCPCVValidate:
    def _default_params(self):
        return _cpcv_default_params()

    def test_trending_data_produces_results(self):
        """CPCV on trending data should produce non-zero results."""
        strategy = CPCVTestStrategy()
        data = _make_trending_data(2000)
        config = PipelineConfig(
            cpcv_n_blocks=5, cpcv_k_test=1,
            cpcv_purge_bars=50, cpcv_embargo_bars=20,
            cpcv_min_block_bars=100,
            cpcv_pct_positive_sharpe_gate=0.0,
            cpcv_mean_sharpe_gate=-999.0,
            wf_min_trades_per_window=1,
            commission_pips=0.0, max_spread_pips=0.0,
        )

        results = cpcv_validate(
            strategy, [self._default_params()], data, config,
            slippage_pips=0.0,
        )

        assert len(results) == 1
        r = results[0]
        assert r.n_blocks == 5
        assert r.k_test == 1
        assert r.n_folds == 5  # C(5,1) = 5
        assert len(r.folds) == 5
        # Should have some non-zero metrics
        assert r.mean_sharpe != 0 or r.n_folds == 0

    def test_small_blocks_skip(self):
        """Data too small for min_block_bars should skip CPCV."""
        strategy = CPCVTestStrategy()
        data = _make_trending_data(500)
        config = PipelineConfig(
            cpcv_n_blocks=10, cpcv_k_test=2,
            cpcv_min_block_bars=200,  # 500/10 = 50 < 200
            commission_pips=0.0, max_spread_pips=0.0,
        )

        results = cpcv_validate(
            strategy, [self._default_params()], data, config,
        )

        assert len(results) == 1
        assert results[0].n_folds == 0
        assert not results[0].passed_gate

    def test_multi_candidate(self):
        """CPCV should produce one result per candidate."""
        strategy = CPCVTestStrategy()
        data = _make_trending_data(2000)
        config = PipelineConfig(
            cpcv_n_blocks=4, cpcv_k_test=1,
            cpcv_purge_bars=20, cpcv_embargo_bars=10,
            cpcv_min_block_bars=100,
            cpcv_pct_positive_sharpe_gate=0.0,
            cpcv_mean_sharpe_gate=-999.0,
            wf_min_trades_per_window=1,
            commission_pips=0.0, max_spread_pips=0.0,
        )

        params = self._default_params()
        results = cpcv_validate(
            strategy, [params, params, params], data, config,
            slippage_pips=0.0,
        )

        assert len(results) == 3
        # All candidates get same params so should get same results
        assert results[0].n_folds == results[1].n_folds == results[2].n_folds

    def test_gate_strict_fails(self):
        """Very strict gates should cause failure."""
        strategy = CPCVTestStrategy()
        data = _make_trending_data(2000)
        config = PipelineConfig(
            cpcv_n_blocks=5, cpcv_k_test=1,
            cpcv_purge_bars=50, cpcv_embargo_bars=20,
            cpcv_min_block_bars=100,
            cpcv_pct_positive_sharpe_gate=1.0,  # 100% — very strict
            cpcv_mean_sharpe_gate=10.0,          # Very strict
            wf_min_trades_per_window=1,
            commission_pips=0.0, max_spread_pips=0.0,
        )

        results = cpcv_validate(
            strategy, [self._default_params()], data, config,
            slippage_pips=0.0,
        )

        assert len(results) == 1
        assert not results[0].passed_gate

    def test_ci_computed(self):
        """Confidence interval should be computed."""
        strategy = CPCVTestStrategy()
        data = _make_trending_data(3000)
        config = PipelineConfig(
            cpcv_n_blocks=6, cpcv_k_test=2,
            cpcv_purge_bars=30, cpcv_embargo_bars=10,
            cpcv_min_block_bars=100,
            cpcv_pct_positive_sharpe_gate=0.0,
            cpcv_mean_sharpe_gate=-999.0,
            wf_min_trades_per_window=1,
            commission_pips=0.0, max_spread_pips=0.0,
        )

        results = cpcv_validate(
            strategy, [self._default_params()], data, config,
            slippage_pips=0.0,
        )

        r = results[0]
        assert r.n_folds == 15  # C(6,2) = 15
        # CI should be computed
        assert r.sharpe_ci_low <= r.mean_sharpe
        assert r.sharpe_ci_high >= r.mean_sharpe
        # Std should be non-negative
        assert r.std_sharpe >= 0


# ---------------------------------------------------------------------------
# Dataclass and checkpoint tests
# ---------------------------------------------------------------------------

class TestCPCVTypes:
    def test_defaults(self):
        r = CPCVResult()
        assert r.n_blocks == 0
        assert r.n_folds == 0
        assert r.folds == []
        assert not r.passed_gate

    def test_fold_defaults(self):
        f = CPCVFoldResult(fold_index=0, train_blocks=(0, 1), test_blocks=(2,))
        assert f.sharpe == 0.0
        assert f.n_trades == 0


class TestCPCVCheckpoint:
    def test_round_trip(self, tmp_path):
        """Save and load checkpoint with CPCV data."""
        from backtester.pipeline.checkpoint import save_checkpoint, load_checkpoint
        from backtester.pipeline.types import CandidateResult, PipelineState

        state = PipelineState(
            strategy_name="test", pair="EUR/USD", timeframe="H1",
            completed_stages=[3],
        )
        cand = CandidateResult(candidate_index=0, params={"a": 1})
        cand.cpcv = CPCVResult(
            n_blocks=10, k_test=2, n_folds=45,
            mean_sharpe=0.85, median_sharpe=0.9,
            std_sharpe=0.3,
            sharpe_ci_low=0.6, sharpe_ci_high=1.1,
            pct_positive_sharpe=0.8,
            mean_quality=25.0, median_quality=23.0,
            passed_gate=True,
            folds=[
                CPCVFoldResult(
                    fold_index=0, train_blocks=(0, 1, 2, 3, 4, 5, 6, 7),
                    test_blocks=(8, 9), n_purged=100,
                    n_trades=50, sharpe=1.2, quality_score=30.0,
                ),
            ],
        )
        state.candidates.append(cand)

        filepath = str(tmp_path / "checkpoint.json")
        save_checkpoint(state, filepath)
        loaded = load_checkpoint(filepath)

        assert len(loaded.candidates) == 1
        c = loaded.candidates[0]
        assert c.cpcv is not None
        assert c.cpcv.n_blocks == 10
        assert c.cpcv.k_test == 2
        assert c.cpcv.n_folds == 45
        assert abs(c.cpcv.mean_sharpe - 0.85) < 1e-6
        assert abs(c.cpcv.pct_positive_sharpe - 0.8) < 1e-6
        assert c.cpcv.passed_gate is True
        assert len(c.cpcv.folds) == 1
        assert c.cpcv.folds[0].train_blocks == (0, 1, 2, 3, 4, 5, 6, 7)
        assert c.cpcv.folds[0].test_blocks == (8, 9)
        assert c.cpcv.folds[0].n_purged == 100
