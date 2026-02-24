"""Tests for regime classification and analysis (VP-3)."""

import numpy as np
import pytest
from dataclasses import dataclass

from backtester.pipeline.regime import (
    REGIME_TREND_QUIET,
    REGIME_TREND_VOLATILE,
    REGIME_RANGE_QUIET,
    REGIME_RANGE_VOLATILE,
    REGIME_UNKNOWN,
    REGIME_NAMES,
    RegimeResult,
    RegimeStats,
    classify_bars,
    compute_regime_stats,
    score_regime_robustness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trending_data(n: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic uptrending data with high ADX."""
    # Strong directional move: each bar adds 0.001 to close
    close = 1.1000 + np.arange(n) * 0.001
    high = close + 0.0005
    low = close - 0.0005
    return high, low, close


def _make_ranging_data(n: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic ranging data with low ADX."""
    # Oscillate around 1.1 with small amplitude
    t = np.arange(n, dtype=np.float64)
    close = 1.1 + 0.001 * np.sin(t * 0.5)
    high = close + 0.0003
    low = close - 0.0003
    return high, low, close


@dataclass
class MockTrade:
    """Minimal mock trade for testing regime stats."""
    bar_entry: int
    pnl_pips: float
    direction: str = "BUY"


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestClassifyBars:
    """Tests for classify_bars()."""

    def test_all_trending(self):
        """Strong trend should classify most bars as TREND_*."""
        high, low, close = _make_trending_data(500)
        labels = classify_bars(high, low, close)

        valid = labels[labels != REGIME_UNKNOWN]
        assert len(valid) > 0

        # Most valid bars should be trending
        trending = np.sum(
            (valid == REGIME_TREND_QUIET) | (valid == REGIME_TREND_VOLATILE)
        )
        # With strong directional data, at least 50% should be trending
        assert trending / len(valid) >= 0.5, (
            f"Only {trending}/{len(valid)} bars classified as trending"
        )

    def test_all_ranging(self):
        """Ranging (oscillating) data should produce RANGE_* labels."""
        high, low, close = _make_ranging_data(500)
        labels = classify_bars(high, low, close)

        valid = labels[labels != REGIME_UNKNOWN]
        assert len(valid) > 0

        ranging = np.sum(
            (valid == REGIME_RANGE_QUIET) | (valid == REGIME_RANGE_VOLATILE)
        )
        # Most should be ranging
        assert ranging / len(valid) >= 0.5, (
            f"Only {ranging}/{len(valid)} bars classified as ranging"
        )

    def test_hysteresis(self):
        """ADX near threshold should not cause flickering due to hysteresis."""
        # Create data where ADX hovers between 20-25 (hysteresis zone)
        # Use real data patterns: gentle trend then gentle range
        n = 400
        # First half: mild uptrend (ADX likely > 20 but variable)
        close1 = 1.1 + np.arange(200) * 0.0002
        # Second half: same mild uptrend continuing
        close2 = close1[-1] + np.arange(200) * 0.0002
        close = np.concatenate([close1, close2])
        high = close + 0.0004
        low = close - 0.0004

        labels = classify_bars(high, low, close, min_regime_bars=1)
        valid = labels[labels != REGIME_UNKNOWN]

        if len(valid) < 10:
            pytest.skip("Not enough valid bars for hysteresis test")

        # Count regime transitions
        transitions = np.sum(np.diff(valid) != 0)
        n_valid = len(valid)

        # With hysteresis, should have relatively few transitions
        # (less than 1 transition per 5 valid bars)
        assert transitions / n_valid < 0.2, (
            f"Too many transitions: {transitions} in {n_valid} bars "
            f"({transitions/n_valid:.2f} per bar)"
        )

    def test_min_duration_suppresses_short_bursts(self):
        """Short regime bursts should be suppressed by cooldown."""
        high, low, close = _make_trending_data(500)

        # With min_regime_bars=1 (no cooldown) vs min_regime_bars=20
        labels_no_cool = classify_bars(high, low, close, min_regime_bars=1)
        labels_cool = classify_bars(high, low, close, min_regime_bars=20)

        valid_no_cool = labels_no_cool[labels_no_cool != REGIME_UNKNOWN]
        valid_cool = labels_cool[labels_cool != REGIME_UNKNOWN]

        if len(valid_no_cool) < 10 or len(valid_cool) < 10:
            pytest.skip("Not enough valid bars")

        # Cooldown should produce fewer or equal transitions
        trans_no_cool = np.sum(np.diff(valid_no_cool) != 0)
        trans_cool = np.sum(np.diff(valid_cool) != 0)
        assert trans_cool <= trans_no_cool

    def test_natr_percentile_is_causal(self):
        """NATR percentile should NOT use future data (no lookahead)."""
        high, low, close = _make_trending_data(500)

        # Classify full array
        labels_full = classify_bars(high, low, close)

        # Classify truncated array (first 300 bars)
        labels_trunc = classify_bars(high[:300], low[:300], close[:300])

        # The first 300 labels should be identical (causal = no future info)
        # They should match wherever both are valid (not UNKNOWN)
        both_valid = (labels_full[:300] != REGIME_UNKNOWN) & (labels_trunc != REGIME_UNKNOWN)
        if np.sum(both_valid) > 0:
            assert np.array_equal(
                labels_full[:300][both_valid],
                labels_trunc[both_valid],
            ), "Classification changed when future data was added (lookahead detected)"

    def test_warmup_is_unknown(self):
        """Bars in warmup period should be UNKNOWN."""
        high, low, close = _make_trending_data(500)
        labels = classify_bars(high, low, close, adx_period=14, natr_percentile_lookback=100)

        # First ~100 bars should be UNKNOWN (need both ADX warmup and NATR lookback)
        # ADX needs ~2*period bars, NATR needs natr_percentile_lookback
        # At minimum, first 28 bars (2*14 for ADX) should be UNKNOWN
        assert np.all(labels[:28] == REGIME_UNKNOWN), (
            f"Expected first 28 bars to be UNKNOWN, got: {labels[:28]}"
        )

    def test_short_data(self):
        """Very short data should return all UNKNOWN."""
        high = np.array([1.1, 1.2, 1.3])
        low = np.array([1.0, 1.1, 1.2])
        close = np.array([1.05, 1.15, 1.25])

        labels = classify_bars(high, low, close)
        assert np.all(labels == REGIME_UNKNOWN)

    def test_output_shape_and_dtype(self):
        """Output should match input length and be int64."""
        high, low, close = _make_trending_data(500)
        labels = classify_bars(high, low, close)

        assert labels.shape == (500,)
        assert labels.dtype == np.int64

    def test_all_labels_are_valid_values(self):
        """All labels should be valid regime codes."""
        high, low, close = _make_trending_data(500)
        labels = classify_bars(high, low, close)

        valid_codes = {
            REGIME_TREND_QUIET, REGIME_TREND_VOLATILE,
            REGIME_RANGE_QUIET, REGIME_RANGE_VOLATILE,
            REGIME_UNKNOWN,
        }
        unique = set(labels)
        assert unique.issubset(valid_codes), (
            f"Invalid label values: {unique - valid_codes}"
        )


# ---------------------------------------------------------------------------
# Regime stats tests
# ---------------------------------------------------------------------------

class TestComputeRegimeStats:
    """Tests for compute_regime_stats()."""

    def test_basic_stats(self):
        """Test per-regime Sharpe/PF computation with known trades."""
        # 200 bars: first 100 trending, last 100 ranging
        labels = np.array(
            [REGIME_TREND_QUIET] * 100 + [REGIME_RANGE_QUIET] * 100,
            dtype=np.int64,
        )
        # 60 trades: 40 in trend (profitable), 20 in range (mixed)
        trades = []
        for i in range(40):
            # Varying positive PnL so std > 0
            trades.append(MockTrade(bar_entry=i * 2, pnl_pips=3.0 + i * 0.1))
        for i in range(20):
            # Range trades: half win, half lose
            pnl = 3.0 if i < 10 else -4.0
            trades.append(MockTrade(bar_entry=100 + i * 3, pnl_pips=pnl))

        result = compute_regime_stats(labels, trades, min_trades_per_regime=10)

        # Trend regime should be scored and profitable
        trend_stats = [s for s in result.per_regime if s.regime == REGIME_TREND_QUIET][0]
        assert trend_stats.sufficient_data is True
        assert trend_stats.n_trades == 40
        assert trend_stats.win_rate == 100.0
        assert trend_stats.sharpe > 0

        # Range regime should also be scored
        range_stats = [s for s in result.per_regime if s.regime == REGIME_RANGE_QUIET][0]
        assert range_stats.sufficient_data is True
        assert range_stats.n_trades == 20

        assert result.n_scored_regimes >= 2
        assert result.n_profitable_regimes >= 1

    def test_insufficient_trades(self):
        """Regime with < min_trades should be marked insufficient."""
        labels = np.array(
            [REGIME_TREND_QUIET] * 50 + [REGIME_RANGE_QUIET] * 50,
            dtype=np.int64,
        )
        # Only 5 trades in range regime (below min_trades=30)
        trades = [MockTrade(bar_entry=i, pnl_pips=3.0) for i in range(30)]  # all in trend
        trades += [MockTrade(bar_entry=55, pnl_pips=1.0) for _ in range(5)]  # few in range

        result = compute_regime_stats(labels, trades, min_trades_per_regime=30)

        trend_stats = [s for s in result.per_regime if s.regime == REGIME_TREND_QUIET][0]
        assert trend_stats.sufficient_data is True

        range_stats = [s for s in result.per_regime if s.regime == REGIME_RANGE_QUIET][0]
        assert range_stats.sufficient_data is False
        assert range_stats.n_trades == 5

    def test_no_trades(self):
        """Empty trade list should be handled gracefully."""
        labels = np.full(100, REGIME_TREND_QUIET, dtype=np.int64)
        result = compute_regime_stats(labels, [], min_trades_per_regime=30)

        assert result.n_scored_regimes == 0
        assert result.n_profitable_regimes == 0
        assert len(result.per_regime) == 4  # All 4 regimes present
        assert all(s.n_trades == 0 for s in result.per_regime)

    def test_regime_distribution(self):
        """Bar distribution percentages should sum to ~100%."""
        labels = np.array(
            [REGIME_TREND_QUIET] * 30
            + [REGIME_TREND_VOLATILE] * 20
            + [REGIME_RANGE_QUIET] * 40
            + [REGIME_RANGE_VOLATILE] * 10,
            dtype=np.int64,
        )
        result = compute_regime_stats(labels, [], min_trades_per_regime=30)

        total_pct = sum(result.regime_distribution.values())
        assert abs(total_pct - 100.0) < 0.1, f"Distribution sums to {total_pct}"

        assert abs(result.regime_distribution["Trend + Quiet"] - 30.0) < 0.1
        assert abs(result.regime_distribution["Range + Quiet"] - 40.0) < 0.1

    def test_trades_outside_range(self):
        """Trades with bar_entry outside label array should be ignored."""
        labels = np.full(50, REGIME_TREND_QUIET, dtype=np.int64)
        trades = [
            MockTrade(bar_entry=999, pnl_pips=10.0),  # out of range
            MockTrade(bar_entry=-1, pnl_pips=10.0),   # negative
        ]
        result = compute_regime_stats(labels, trades, min_trades_per_regime=1)
        assert all(s.n_trades == 0 for s in result.per_regime)

    def test_advisory_message_generated(self):
        """Advisory message should be non-empty when trades exist."""
        labels = np.full(100, REGIME_TREND_QUIET, dtype=np.int64)
        trades = [MockTrade(bar_entry=i, pnl_pips=2.0) for i in range(50)]
        result = compute_regime_stats(labels, trades, min_trades_per_regime=10)
        assert len(result.advisory) > 0


# ---------------------------------------------------------------------------
# Robustness scoring tests
# ---------------------------------------------------------------------------

class TestScoreRegimeRobustness:
    """Tests for score_regime_robustness()."""

    def test_all_profitable(self):
        """All regimes profitable -> high score."""
        result = RegimeResult(
            per_regime=[
                RegimeStats(regime=0, regime_name="A", sharpe=1.5, max_dd_pct=10.0,
                            sufficient_data=True),
                RegimeStats(regime=1, regime_name="B", sharpe=1.2, max_dd_pct=12.0,
                            sufficient_data=True),
                RegimeStats(regime=2, regime_name="C", sharpe=0.8, max_dd_pct=15.0,
                            sufficient_data=True),
            ],
            n_profitable_regimes=3,
            n_scored_regimes=3,
            worst_regime_max_dd=15.0,
        )
        score = score_regime_robustness(result)
        assert score >= 70.0, f"Expected >= 70, got {score}"

    def test_one_catastrophic(self):
        """One regime with MaxDD > 60% -> lower score."""
        result = RegimeResult(
            per_regime=[
                RegimeStats(regime=0, regime_name="A", sharpe=1.5, max_dd_pct=10.0,
                            sufficient_data=True),
                RegimeStats(regime=1, regime_name="B", sharpe=-0.5, max_dd_pct=65.0,
                            sufficient_data=True),
            ],
            n_profitable_regimes=1,
            n_scored_regimes=2,
            worst_regime_max_dd=65.0,
        )
        score = score_regime_robustness(result)
        # 1/2 profitable (50% * 0.40 = 20) + catastrophic DD (0) + high variance
        assert score < 40.0, f"Expected < 40, got {score}"

    def test_low_variance(self):
        """Similar Sharpes across regimes -> high variance component."""
        result = RegimeResult(
            per_regime=[
                RegimeStats(regime=0, regime_name="A", sharpe=1.0, max_dd_pct=10.0,
                            sufficient_data=True),
                RegimeStats(regime=1, regime_name="B", sharpe=1.1, max_dd_pct=12.0,
                            sufficient_data=True),
                RegimeStats(regime=2, regime_name="C", sharpe=0.9, max_dd_pct=8.0,
                            sufficient_data=True),
            ],
            n_profitable_regimes=3,
            n_scored_regimes=3,
            worst_regime_max_dd=12.0,
        )
        score = score_regime_robustness(result)
        assert score >= 80.0, f"Expected >= 80, got {score}"

    def test_no_scored_regimes(self):
        """No scored regimes -> 0 score."""
        result = RegimeResult(n_scored_regimes=0, n_profitable_regimes=0)
        assert score_regime_robustness(result) == 0.0

    def test_score_range(self):
        """Score should always be 0-100."""
        for n_prof, n_scored, worst_dd in [
            (0, 3, 80.0), (3, 3, 5.0), (1, 4, 45.0), (2, 2, 30.0),
        ]:
            result = RegimeResult(
                per_regime=[
                    RegimeStats(regime=i, regime_name=f"R{i}", sharpe=float(i) - 1.0,
                                max_dd_pct=worst_dd - i * 5,
                                sufficient_data=True)
                    for i in range(n_scored)
                ],
                n_profitable_regimes=n_prof,
                n_scored_regimes=n_scored,
                worst_regime_max_dd=worst_dd,
            )
            score = score_regime_robustness(result)
            assert 0.0 <= score <= 100.0, f"Score out of range: {score}"


# ---------------------------------------------------------------------------
# Serialization test
# ---------------------------------------------------------------------------

class TestRegimeResultSerialization:
    """Test that RegimeResult can be serialized for checkpoints."""

    def test_to_dict(self):
        """RegimeResult should be convertible to dict via dataclasses.asdict()."""
        from dataclasses import asdict

        result = RegimeResult(
            regime_distribution={"Trend + Quiet": 30.0, "Range + Quiet": 70.0},
            per_regime=[
                RegimeStats(
                    regime=0, regime_name="Trend + Quiet", n_bars=100,
                    bar_pct=30.0, n_trades=50, sharpe=1.5, profit_factor=2.0,
                    max_dd_pct=8.0, win_rate=60.0, mean_pnl_pips=3.5,
                    sufficient_data=True,
                ),
            ],
            regime_weighted_sharpe=1.2,
            worst_regime_max_dd=8.0,
            n_profitable_regimes=1,
            n_scored_regimes=1,
            advisory="Test advisory",
            robustness_score=75.0,
        )

        d = asdict(result)
        assert isinstance(d, dict)
        assert d["regime_weighted_sharpe"] == 1.2
        assert d["per_regime"][0]["regime_name"] == "Trend + Quiet"
        assert d["per_regime"][0]["sharpe"] == 1.5
