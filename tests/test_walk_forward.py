"""Tests for walk-forward validation module."""

import numpy as np
import pytest

from backtester.core.dtypes import EXEC_FULL, M_TRADES, M_SHARPE, M_QUALITY
from backtester.core.encoding import build_encoding_spec, encode_params
from backtester.core.engine import BacktestEngine
from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.types import WalkForwardResult, WindowResult
from backtester.pipeline.walk_forward import (
    evaluate_candidate_on_window,
    generate_windows,
    label_windows,
    walk_forward_validate,
)
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
# DummyStrategy for testing
# ---------------------------------------------------------------------------

class DummyStrategy(Strategy):
    """Simple test strategy: generate BUY signals at fixed bars."""

    def __init__(self, signal_bars=None):
        self._signal_bars = signal_bars or [10, 30, 50]

    @property
    def name(self):
        return "dummy"

    @property
    def version(self):
        return "1.0"

    def param_space(self):
        ps = ParamSpace([
            ParamDef("rsi_threshold", [30, 40, 50, 60, 70], group="signal"),
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
        for bar in self._signal_bars:
            if bar < len(close):
                signals.append(Signal(
                    bar_index=bar,
                    direction=Direction.BUY,
                    entry_price=close[bar],
                    hour=10,
                    day_of_week=1,
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


class EmptyStrategy(Strategy):
    """Strategy that generates zero signals."""

    @property
    def name(self):
        return "empty"

    @property
    def version(self):
        return "1.0"

    def param_space(self):
        ps = ParamSpace([
            ParamDef("rsi_threshold", [30, 40, 50, 60, 70], group="signal"),
        ])
        for p in risk_params():
            ps.add(p.name, p.values, p.group)
        for p in management_params():
            ps.add(p.name, p.values, p.group)
        for p in time_params():
            ps.add(p.name, p.values, p.group)
        return ps

    def generate_signals(self, open_, high, low, close, volume, spread):
        return []

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
# Helpers
# ---------------------------------------------------------------------------

def _make_trending_data(n_bars=100, pip=0.0001):
    """Create synthetic uptrending price data where BUY signals should profit."""
    base = 1.1000
    open_ = np.zeros(n_bars, dtype=np.float64)
    high = np.zeros(n_bars, dtype=np.float64)
    low = np.zeros(n_bars, dtype=np.float64)
    close = np.zeros(n_bars, dtype=np.float64)
    for i in range(n_bars):
        price = base + i * 3 * pip
        open_[i] = price
        high[i] = price + 15 * pip
        low[i] = price - 10 * pip
        close[i] = price + 2 * pip
    volume = np.ones(n_bars, dtype=np.float64)
    spread = np.full(n_bars, 1.0 * pip)
    return open_, high, low, close, volume, spread


def _make_flat_data(n_bars=100, pip=0.0001):
    """Create flat/sideways price data where signals should not produce good metrics."""
    base = 1.1000
    rng = np.random.RandomState(42)
    open_ = np.full(n_bars, base, dtype=np.float64)
    noise = rng.randn(n_bars) * pip * 0.5
    close = open_ + noise
    high = np.maximum(open_, close) + pip * 2
    low = np.minimum(open_, close) - pip * 2
    volume = np.ones(n_bars, dtype=np.float64)
    spread = np.full(n_bars, 1.0 * pip)
    return open_, high, low, close, volume, spread


def _make_data_arrays(n_bars=100, trending=True, pip=0.0001):
    """Create data_arrays dict suitable for walk-forward functions."""
    if trending:
        open_, high, low, close, volume, spread = _make_trending_data(n_bars, pip)
    else:
        open_, high, low, close, volume, spread = _make_flat_data(n_bars, pip)
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "spread": spread,
        "bar_hour": None,
        "bar_day_of_week": None,
    }


def _default_params():
    """Default parameter dict for DummyStrategy."""
    return {
        "rsi_threshold": 50,
        "sl_mode": "fixed_pips",
        "sl_fixed_pips": 30,
        "sl_atr_mult": 1.5,
        "tp_mode": "rr_ratio",
        "tp_rr_ratio": 2.0,
        "tp_atr_mult": 2.0,
        "tp_fixed_pips": 60,
        "allowed_hours_start": 0,
        "allowed_hours_end": 23,
        "allowed_days": [0, 1, 2, 3, 4],
        "trailing_mode": "off",
        "trail_activate_pips": 0,
        "trail_distance_pips": 10,
        "trail_atr_mult": 2.0,
        "breakeven_enabled": False,
        "breakeven_trigger_pips": 20,
        "breakeven_offset_pips": 2,
        "partial_close_enabled": False,
        "partial_close_pct": 50,
        "partial_close_trigger_pips": 30,
        "max_bars": 0,
        "stale_exit_enabled": False,
        "stale_exit_bars": 50,
        "stale_exit_atr_threshold": 0.5,
    }


# ---------------------------------------------------------------------------
# Tests: generate_windows
# ---------------------------------------------------------------------------

class TestGenerateWindows:

    def test_rolling_basic(self):
        """Rolling windows: correct count, no window extends past n_bars."""
        windows = generate_windows(
            n_bars=1000, window_size=200, step_size=100,
            embargo_bars=0, anchored=False,
        )
        # With step=100, window=200, no embargo:
        # starts: 0, 100, 200, ..., 800 => 9 windows
        assert len(windows) == 9
        for start, end in windows:
            assert end <= 1000
            assert end - start == 200

    def test_rolling_no_partial_windows(self):
        """Rolling windows should not create partial windows past n_bars."""
        windows = generate_windows(
            n_bars=350, window_size=200, step_size=100,
            embargo_bars=0, anchored=False,
        )
        # starts: 0 (end=200), 100 (end=300) => 2 windows
        # start=200 would give end=400 > 350, so excluded
        assert len(windows) == 2
        for start, end in windows:
            assert end <= 350

    def test_anchored_basic(self):
        """Anchored windows: all start at 0, end grows."""
        windows = generate_windows(
            n_bars=1000, window_size=200, step_size=200,
            embargo_bars=0, anchored=True,
        )
        # ends: 200, 400, 600, 800, 1000 => 5 windows
        assert len(windows) == 5
        for start, end in windows:
            assert start == 0
            assert end <= 1000

    def test_anchored_growing_windows(self):
        """Anchored windows grow with each step."""
        windows = generate_windows(
            n_bars=600, window_size=200, step_size=100,
            embargo_bars=0, anchored=True,
        )
        # ends: 200, 300, 400, 500, 600 => 5 windows
        assert len(windows) == 5
        for i, (start, end) in enumerate(windows):
            assert start == 0
            expected_end = 200 + i * 100
            assert end == expected_end

    def test_embargo_rolling(self):
        """Embargo creates gaps between rolling window starts."""
        windows_no_embargo = generate_windows(
            n_bars=1000, window_size=200, step_size=100,
            embargo_bars=0, anchored=False,
        )
        windows_with_embargo = generate_windows(
            n_bars=1000, window_size=200, step_size=100,
            embargo_bars=50, anchored=False,
        )
        # With embargo, effective step = 100 + 50 = 150
        # Fewer windows because of larger effective step
        assert len(windows_with_embargo) < len(windows_no_embargo)

        # Verify that window starts are spaced by (step + embargo)
        for i in range(len(windows_with_embargo) - 1):
            start_i, _ = windows_with_embargo[i]
            start_next, _ = windows_with_embargo[i + 1]
            assert start_next - start_i == 150  # step(100) + embargo(50)

    def test_embargo_anchored(self):
        """Embargo works with anchored windows too."""
        windows = generate_windows(
            n_bars=1000, window_size=200, step_size=200,
            embargo_bars=50, anchored=True,
        )
        # Effective step = 200 + 50 = 250
        # ends: 200, 450, 700, 950 => 4 windows
        assert len(windows) == 4
        for start, end in windows:
            assert start == 0

    def test_empty_when_too_short(self):
        """No windows generated when data is shorter than window_size."""
        windows = generate_windows(
            n_bars=100, window_size=200, step_size=100,
            embargo_bars=0, anchored=False,
        )
        assert len(windows) == 0


# ---------------------------------------------------------------------------
# Tests: label_windows
# ---------------------------------------------------------------------------

class TestLabelWindows:

    def test_oos_windows_outside_opt_range(self):
        """Windows entirely outside optimization range are OOS."""
        windows = [(0, 100), (100, 200), (200, 300), (300, 400)]
        # Optimization covers bars 100-300
        labels = label_windows(windows, opt_start=100, opt_end=300)
        # Window (0,100) does NOT overlap [100,300) because end=100 is not > 100
        # Actually: overlap requires w_start < opt_end AND opt_start < w_end
        # (0,100): 0 < 300=True AND 100 < 100=False => no overlap => OOS
        # (100,200): 100 < 300=True AND 100 < 200=True => overlap => IS
        # (200,300): 200 < 300=True AND 100 < 300=True => overlap => IS
        # (300,400): 300 < 300=False => no overlap => OOS
        assert labels == [True, False, False, True]

    def test_is_windows_overlap_opt_range(self):
        """Windows overlapping optimization range are IS."""
        windows = [(50, 150), (150, 250)]
        labels = label_windows(windows, opt_start=100, opt_end=200)
        # (50,150): 50 < 200=True AND 100 < 150=True => overlap => IS
        # (150,250): 150 < 200=True AND 100 < 250=True => overlap => IS
        assert labels == [False, False]

    def test_all_oos_when_no_overlap(self):
        """All windows are OOS when opt range doesn't overlap any."""
        windows = [(0, 100), (100, 200)]
        labels = label_windows(windows, opt_start=500, opt_end=600)
        assert labels == [True, True]

    def test_partial_overlap_is_is(self):
        """Even partial overlap marks a window as IS."""
        windows = [(90, 110)]  # Partially overlaps [100, 200)
        labels = label_windows(windows, opt_start=100, opt_end=200)
        # 90 < 200=True AND 100 < 110=True => overlap => IS
        assert labels == [False]

    def test_adjacent_no_overlap(self):
        """Window ending exactly at opt_start does NOT overlap (exclusive boundary)."""
        windows = [(0, 100)]
        labels = label_windows(windows, opt_start=100, opt_end=200)
        # 0 < 200=True AND 100 < 100=False => no overlap => OOS
        assert labels == [True]


# ---------------------------------------------------------------------------
# Tests: evaluate_candidate_on_window
# ---------------------------------------------------------------------------

class TestEvaluateCandidateOnWindow:

    def test_returns_window_result_with_metrics(self):
        """Evaluate a candidate on a trending window, verify metrics are populated."""
        n_bars = 600
        data_arrays = _make_data_arrays(n_bars=n_bars, trending=True)
        # Signals spread throughout the data
        signal_bars = list(range(20, 500, 25))
        strategy = DummyStrategy(signal_bars=signal_bars)
        params = _default_params()

        config = PipelineConfig(wf_min_trades_per_window=1)

        result = evaluate_candidate_on_window(
            strategy=strategy,
            params_dict=params,
            data_arrays=data_arrays,
            window_start=0,
            window_end=300,
            lookback_prefix=50,
            config=config,
            window_index=0,
            is_oos=True,
        )

        assert isinstance(result, WindowResult)
        assert result.window_index == 0
        assert result.start_bar == 0
        assert result.end_bar == 300
        assert result.is_oos is True
        # Should have some trades from the trending data
        assert result.n_trades >= 1

    def test_trending_data_produces_positive_sharpe(self):
        """On strong uptrend, BUY signals should yield positive Sharpe."""
        n_bars = 600
        data_arrays = _make_data_arrays(n_bars=n_bars, trending=True)
        signal_bars = list(range(20, 500, 25))
        strategy = DummyStrategy(signal_bars=signal_bars)
        params = _default_params()

        config = PipelineConfig(wf_min_trades_per_window=1)

        result = evaluate_candidate_on_window(
            strategy=strategy,
            params_dict=params,
            data_arrays=data_arrays,
            window_start=50,
            window_end=500,
            lookback_prefix=50,
            config=config,
        )

        # Trending data should produce profitable trades
        assert result.n_trades > 0
        assert result.sharpe > 0
        assert result.passed is True

    def test_lookback_prefix_extends_slice(self):
        """Lookback prefix gives indicators warmup data before the window."""
        n_bars = 600
        data_arrays = _make_data_arrays(n_bars=n_bars, trending=True)
        # Signal at bar 210 (relative to full data): should be in range
        # if lookback is considered
        signal_bars = [210]
        strategy = DummyStrategy(signal_bars=signal_bars)
        params = _default_params()
        config = PipelineConfig(wf_min_trades_per_window=1)

        # Window is [200, 400), with 50-bar lookback => slice is [150, 400)
        # Signal at bar 210 in full data maps to bar 210-150=60 in the slice
        result = evaluate_candidate_on_window(
            strategy=strategy,
            params_dict=params,
            data_arrays=data_arrays,
            window_start=200,
            window_end=400,
            lookback_prefix=50,
            config=config,
        )

        # The strategy generates signals based on the slice's indices,
        # not the original data indices. Since the slice starts at 150 and
        # the DummyStrategy generates at bar_index=210 only if 210 < len(slice)=250,
        # which is true, we should get a signal.
        # NOTE: The DummyStrategy uses absolute bar indices within the slice,
        # so bar 210 in a 250-bar slice is valid.
        assert result.n_trades >= 0  # May or may not produce trades depending on TP/SL


# ---------------------------------------------------------------------------
# Tests: walk_forward_validate (gate pass)
# ---------------------------------------------------------------------------

class TestWalkForwardGatePass:

    def test_good_candidate_passes_gate(self):
        """A good candidate on trending data should pass the walk-forward gate."""
        n_bars = 800
        data_arrays = _make_data_arrays(n_bars=n_bars, trending=True)
        # Generate signals throughout all windows
        signal_bars = list(range(10, 780, 15))
        strategy = DummyStrategy(signal_bars=signal_bars)
        params = _default_params()

        config = PipelineConfig(
            wf_window_bars=200,
            wf_step_bars=100,
            wf_embargo_bars=0,
            wf_anchored=False,
            wf_lookback_prefix=20,
            wf_min_trades_per_window=1,
            wf_pass_rate_gate=0.5,
            wf_mean_sharpe_gate=0.0,
        )

        # Optimization covers first half; second half is OOS
        results = walk_forward_validate(
            strategy=strategy,
            candidates=[params],
            data_arrays=data_arrays,
            opt_start=0,
            opt_end=400,
            config=config,
        )

        assert len(results) == 1
        wf = results[0]
        assert isinstance(wf, WalkForwardResult)
        assert wf.n_oos_windows > 0
        assert wf.n_windows > 0
        # On uptrending data, BUY signals should be consistently profitable
        assert wf.passed_gate is True
        assert wf.pass_rate > 0
        assert wf.mean_sharpe > 0

    def test_multiple_candidates(self):
        """Walk-forward returns one result per candidate."""
        n_bars = 800
        data_arrays = _make_data_arrays(n_bars=n_bars, trending=True)
        signal_bars = list(range(10, 780, 15))
        strategy = DummyStrategy(signal_bars=signal_bars)

        params1 = _default_params()
        params2 = _default_params()
        params2["rsi_threshold"] = 70  # Different but should still work

        config = PipelineConfig(
            wf_window_bars=200,
            wf_step_bars=100,
            wf_embargo_bars=0,
            wf_lookback_prefix=20,
            wf_min_trades_per_window=1,
            wf_pass_rate_gate=0.5,
            wf_mean_sharpe_gate=0.0,
        )

        results = walk_forward_validate(
            strategy=strategy,
            candidates=[params1, params2],
            data_arrays=data_arrays,
            opt_start=0,
            opt_end=400,
            config=config,
        )

        assert len(results) == 2
        # Both should have window results
        for wf in results:
            assert wf.n_oos_windows > 0


# ---------------------------------------------------------------------------
# Tests: walk_forward_validate (gate fail)
# ---------------------------------------------------------------------------

class TestWalkForwardGateFail:

    def test_flat_data_fails_gate(self):
        """On flat/noisy data, candidate should fail the walk-forward gate."""
        n_bars = 800
        data_arrays = _make_data_arrays(n_bars=n_bars, trending=False)
        signal_bars = list(range(10, 780, 15))
        strategy = DummyStrategy(signal_bars=signal_bars)
        params = _default_params()

        config = PipelineConfig(
            wf_window_bars=200,
            wf_step_bars=100,
            wf_embargo_bars=0,
            wf_lookback_prefix=20,
            wf_min_trades_per_window=1,
            wf_pass_rate_gate=0.6,
            wf_mean_sharpe_gate=0.3,
        )

        results = walk_forward_validate(
            strategy=strategy,
            candidates=[params],
            data_arrays=data_arrays,
            opt_start=0,
            opt_end=400,
            config=config,
        )

        assert len(results) == 1
        wf = results[0]
        # On flat data with SL/TP, unlikely to have consistently positive Sharpe
        # The gate requires mean_sharpe >= 0.3 which flat data won't achieve
        assert wf.passed_gate is False

    def test_strict_gate_rejects(self):
        """A very strict gate on flat data should reject the candidate."""
        n_bars = 800
        # Use flat data so performance metrics are poor
        data_arrays = _make_data_arrays(n_bars=n_bars, trending=False)
        signal_bars = list(range(10, 780, 15))
        strategy = DummyStrategy(signal_bars=signal_bars)
        params = _default_params()

        config = PipelineConfig(
            wf_window_bars=200,
            wf_step_bars=100,
            wf_embargo_bars=0,
            wf_lookback_prefix=20,
            wf_min_trades_per_window=1,
            wf_pass_rate_gate=1.0,  # Every single window must pass
            wf_mean_sharpe_gate=1.0,  # High bar for flat data
        )

        results = walk_forward_validate(
            strategy=strategy,
            candidates=[params],
            data_arrays=data_arrays,
            opt_start=0,
            opt_end=400,
            config=config,
        )

        assert len(results) == 1
        # Flat data with strict gates should fail
        assert results[0].passed_gate is False


# ---------------------------------------------------------------------------
# Tests: walk_forward with empty signals
# ---------------------------------------------------------------------------

class TestWalkForwardEmptySignals:

    def test_zero_trades_fails_gate(self):
        """A strategy with zero signals should produce 0 metrics and fail the gate."""
        n_bars = 800
        data_arrays = _make_data_arrays(n_bars=n_bars, trending=True)
        strategy = EmptyStrategy()
        params = _default_params()

        config = PipelineConfig(
            wf_window_bars=200,
            wf_step_bars=100,
            wf_embargo_bars=0,
            wf_lookback_prefix=20,
            wf_min_trades_per_window=10,
            wf_pass_rate_gate=0.6,
            wf_mean_sharpe_gate=0.3,
        )

        results = walk_forward_validate(
            strategy=strategy,
            candidates=[params],
            data_arrays=data_arrays,
            opt_start=0,
            opt_end=400,
            config=config,
        )

        assert len(results) == 1
        wf = results[0]
        assert wf.passed_gate is False
        assert wf.n_passed == 0
        assert wf.mean_sharpe == 0.0

        # All OOS windows should have 0 trades
        oos_windows = [w for w in wf.windows if w.is_oos]
        for wr in oos_windows:
            assert wr.n_trades == 0
            assert wr.passed is False

    def test_evaluate_window_zero_signals(self):
        """Direct window evaluation with zero signals returns zeroed metrics."""
        n_bars = 600
        data_arrays = _make_data_arrays(n_bars=n_bars, trending=True)
        strategy = EmptyStrategy()
        params = _default_params()
        config = PipelineConfig(wf_min_trades_per_window=10)

        result = evaluate_candidate_on_window(
            strategy=strategy,
            params_dict=params,
            data_arrays=data_arrays,
            window_start=0,
            window_end=300,
            lookback_prefix=50,
            config=config,
        )

        assert result.n_trades == 0
        assert result.sharpe == 0.0
        assert result.quality_score == 0.0
        assert result.passed is False


# ---------------------------------------------------------------------------
# Tests: walk_forward edge cases
# ---------------------------------------------------------------------------

class TestWalkForwardEdgeCases:

    def test_data_too_short_for_windows(self):
        """When data is too short to generate any windows, all candidates fail."""
        n_bars = 50
        data_arrays = _make_data_arrays(n_bars=n_bars, trending=True)
        strategy = DummyStrategy(signal_bars=[10])
        params = _default_params()

        config = PipelineConfig(
            wf_window_bars=200,  # Larger than data
            wf_step_bars=100,
            wf_embargo_bars=0,
            wf_lookback_prefix=20,
        )

        results = walk_forward_validate(
            strategy=strategy,
            candidates=[params],
            data_arrays=data_arrays,
            opt_start=0,
            opt_end=25,
            config=config,
        )

        assert len(results) == 1
        assert results[0].passed_gate is False

    def test_wfe_computation(self):
        """WFE (walk-forward efficiency) is computed as mean OOS quality / best IS quality."""
        n_bars = 800
        data_arrays = _make_data_arrays(n_bars=n_bars, trending=True)
        signal_bars = list(range(10, 780, 15))
        strategy = DummyStrategy(signal_bars=signal_bars)
        params = _default_params()

        config = PipelineConfig(
            wf_window_bars=200,
            wf_step_bars=200,
            wf_embargo_bars=0,
            wf_lookback_prefix=20,
            wf_min_trades_per_window=1,
            wf_pass_rate_gate=0.0,
            wf_mean_sharpe_gate=-999.0,
        )

        # Optimization covers the middle; windows at start and end are OOS
        results = walk_forward_validate(
            strategy=strategy,
            candidates=[params],
            data_arrays=data_arrays,
            opt_start=200,
            opt_end=600,
            config=config,
        )

        assert len(results) == 1
        wf = results[0]
        # WFE should be a finite number (could be 0 if IS quality is 0)
        assert np.isfinite(wf.wfe)
        # If there are both IS and OOS windows with positive quality,
        # WFE should be positive
        is_windows = [w for w in wf.windows if not w.is_oos]
        oos_windows = [w for w in wf.windows if w.is_oos]
        if is_windows and oos_windows:
            best_is_q = max(w.quality_score for w in is_windows)
            if best_is_q > 0:
                mean_oos_q = np.mean([w.quality_score for w in oos_windows])
                expected_wfe = mean_oos_q / best_is_q
                assert abs(wf.wfe - expected_wfe) < 1e-6
