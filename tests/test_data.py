"""Tests for data download, timeframe conversion, and validation."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest


# --- Timeframe conversion tests ---

def _make_m1_data(minutes: int = 1440, include_spread: bool = True) -> pd.DataFrame:
    """Create synthetic M1 OHLCV data with optional spread column."""
    idx = pd.date_range("2025-01-06", periods=minutes, freq="1min", tz="UTC")
    rng = np.random.default_rng(42)
    base = 1.1000 + np.cumsum(rng.normal(0, 0.0001, minutes))
    data = {
        "open": base,
        "high": base + rng.uniform(0, 0.0005, minutes),
        "low": base - rng.uniform(0, 0.0005, minutes),
        "close": base + rng.normal(0, 0.0002, minutes),
        "volume": rng.uniform(10, 500, minutes),
    }
    if include_spread:
        # Realistic spread: 0.5 to 2.5 pips (0.00005 to 0.00025 for EUR/USD)
        data["spread"] = rng.uniform(0.00005, 0.00025, minutes)
    return pd.DataFrame(data, index=idx)


def test_resample_ohlcv_h1():
    """Test M1 -> H1 uses correct OHLCV aggregation."""
    from backtester.data.timeframes import resample_ohlcv

    df = _make_m1_data(120)  # 2 hours
    h1 = resample_ohlcv(df, "1h")

    assert len(h1) == 2
    # First hour: open should be first M1 open
    first_hour_m1 = df.iloc[:60]
    assert h1["open"].iloc[0] == first_hour_m1["open"].iloc[0]
    assert h1["high"].iloc[0] == first_hour_m1["high"].max()
    assert h1["low"].iloc[0] == first_hour_m1["low"].min()
    assert h1["close"].iloc[0] == first_hour_m1["close"].iloc[-1]
    assert abs(h1["volume"].iloc[0] - first_hour_m1["volume"].sum()) < 0.01


def test_resample_ohlcv_d():
    """Test M1 -> Daily."""
    from backtester.data.timeframes import resample_ohlcv

    df = _make_m1_data(1440)  # 1 day
    daily = resample_ohlcv(df, "1D")

    assert len(daily) == 1
    assert daily["open"].iloc[0] == df["open"].iloc[0]
    assert daily["high"].iloc[0] == df["high"].max()
    assert daily["low"].iloc[0] == df["low"].min()
    assert daily["close"].iloc[0] == df["close"].iloc[-1]


def test_convert_timeframes_roundtrip(tmp_path):
    """Test full conversion pipeline with file I/O."""
    from backtester.data.timeframes import convert_timeframes

    df = _make_m1_data(1440 * 5)  # 5 days
    m1_path = tmp_path / "TEST_PAIR_M1.parquet"
    df.to_parquet(m1_path)

    results = convert_timeframes("TEST/PAIR", str(tmp_path), ["M5", "H1", "D"])

    assert "M5" in results
    assert "H1" in results
    assert "D" in results

    # Check M5 has ~5x fewer rows
    m5 = pd.read_parquet(results["M5"])
    assert len(m5) < len(df)
    assert len(m5) > len(df) // 6


# --- Validation tests ---

def test_detect_gaps():
    """Test gap detection finds unexpected gaps but ignores weekends."""
    from backtester.data.validation import detect_gaps

    # Create M1 data with a 5-hour gap on a Wednesday
    idx = pd.date_range("2025-01-06 00:00", periods=60, freq="1min", tz="UTC")
    idx2 = pd.date_range("2025-01-06 05:00", periods=60, freq="1min", tz="UTC")
    full_idx = idx.append(idx2)

    df = pd.DataFrame({
        "open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15, "volume": 100.0,
    }, index=full_idx)

    gaps = detect_gaps(df, "M1")
    assert gaps["unexpected_gaps"] >= 1


def test_detect_zeros_nans():
    """Test zero and NaN detection."""
    from backtester.data.validation import detect_zeros_nans

    idx = pd.date_range("2025-01-06", periods=100, freq="1min", tz="UTC")
    df = pd.DataFrame({
        "open": np.ones(100),
        "high": np.ones(100),
        "low": np.ones(100),
        "close": np.ones(100),
        "volume": np.ones(100),
    }, index=idx)

    # Add some zeros and NaNs
    df.loc[df.index[5], "open"] = 0
    df.loc[df.index[10], "close"] = np.nan

    result = detect_zeros_nans(df)
    assert result["zero_count"] == 1
    assert result["nan_count"] == 1


def test_quality_score_clean_data():
    """Clean data should score high."""
    from backtester.data.validation import compute_quality_score

    df = _make_m1_data(10000)
    result = compute_quality_score(df, "M1")
    assert result["quality_score"] >= 80


def test_validate_rejects_empty():
    """Empty dataframe should fail validation."""
    from backtester.data.validation import validate_data

    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    result = validate_data(df)
    assert not result["passed"]


def test_validate_rejects_too_few_candles():
    """Less than min_candles should fail."""
    from backtester.data.validation import validate_data

    df = _make_m1_data(100)
    result = validate_data(df, min_candles=5000)
    assert not result["passed"]
    assert "insufficient" in result["reason"]


# --- Downloader unit tests (no network) ---

def test_pair_to_filename():
    """Test pair name conversion."""
    from backtester.data.downloader import _pair_to_filename

    assert _pair_to_filename("EUR/USD") == "EUR_USD"
    assert _pair_to_filename("XAU/USD") == "XAU_USD"


def test_get_downloaded_years_empty(tmp_path):
    """No chunks dir = no years."""
    from backtester.data.downloader import get_downloaded_years

    result = get_downloaded_years(str(tmp_path), "EUR/USD")
    assert result == []


def test_save_and_read_chunk(tmp_path):
    """Test atomic save of a yearly chunk."""
    from backtester.data.downloader import save_chunk

    df = _make_m1_data(100)
    path = save_chunk(df, str(tmp_path), "EUR/USD", 2025)

    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == 100


# --- Spread computation tests ---

def test_add_spread_aligned():
    """Spread computed correctly when bid and ask have matching timestamps."""
    from backtester.data.downloader import _add_spread

    idx = pd.date_range("2025-01-06", periods=100, freq="1min", tz="UTC")
    bid = pd.DataFrame({
        "open": np.full(100, 1.1000),
        "high": np.full(100, 1.1010),
        "low": np.full(100, 1.0990),
        "close": np.full(100, 1.1005),
        "volume": np.full(100, 100.0),
    }, index=idx)

    # Ask is exactly 1.5 pips above bid
    ask = pd.DataFrame({
        "open": np.full(100, 1.10015),
        "high": np.full(100, 1.10115),
        "low": np.full(100, 1.09915),
        "close": np.full(100, 1.10065),
        "volume": np.full(100, 100.0),
    }, index=idx)

    result = _add_spread(bid, ask)

    assert "spread" in result.columns
    assert result["spread"].notna().all()
    # Spread should be ~0.00015 (1.5 pips)
    assert abs(result["spread"].iloc[0] - 0.00015) < 1e-10


def test_add_spread_partial_overlap():
    """Spread is NaN where ask data is missing."""
    from backtester.data.downloader import _add_spread

    bid_idx = pd.date_range("2025-01-06 00:00", periods=100, freq="1min", tz="UTC")
    ask_idx = pd.date_range("2025-01-06 00:30", periods=70, freq="1min", tz="UTC")

    bid = pd.DataFrame({
        "open": 1.1, "high": 1.101, "low": 1.099, "close": 1.1005, "volume": 100.0,
    }, index=bid_idx)
    ask = pd.DataFrame({
        "open": 1.10015, "high": 1.10115, "low": 1.09915, "close": 1.10065, "volume": 100.0,
    }, index=ask_idx)

    result = _add_spread(bid, ask)

    assert result["spread"].isna().sum() == 30  # first 30 min have no ask
    assert result["spread"].notna().sum() == 70


def test_resample_preserves_spread():
    """Spread column survives timeframe conversion with median aggregation."""
    from backtester.data.timeframes import resample_ohlcv

    df = _make_m1_data(120, include_spread=True)  # 2 hours
    h1 = resample_ohlcv(df, "1h")

    assert "spread" in h1.columns
    assert len(h1) == 2
    # Median of first hour's spreads
    first_hour_spreads = df["spread"].iloc[:60]
    assert abs(h1["spread"].iloc[0] - first_hour_spreads.median()) < 1e-10


def test_resample_works_without_spread():
    """Resampling still works if spread column is absent (backward compat)."""
    from backtester.data.timeframes import resample_ohlcv

    df = _make_m1_data(120, include_spread=False)
    h1 = resample_ohlcv(df, "1h")

    assert len(h1) == 2
    assert "spread" not in h1.columns


def test_save_chunk_with_spread(tmp_path):
    """Chunk with spread column round-trips through Parquet."""
    from backtester.data.downloader import save_chunk

    df = _make_m1_data(100, include_spread=True)
    path = save_chunk(df, str(tmp_path), "EUR/USD", 2025)

    loaded = pd.read_parquet(path)
    assert "spread" in loaded.columns
    assert len(loaded) == 100
    assert loaded["spread"].notna().all()


# --- Data splitting tests ---

def test_split_backforward_default():
    """80/20 split by row count."""
    from backtester.data.splitting import split_backforward

    df = _make_m1_data(1000)
    back, forward = split_backforward(df)

    assert len(back) == 800
    assert len(forward) == 200
    # Back ends before forward starts (chronological)
    assert back.index[-1] < forward.index[0]


def test_split_backforward_custom_pct():
    """Custom split ratio."""
    from backtester.data.splitting import split_backforward

    df = _make_m1_data(1000)
    back, forward = split_backforward(df, back_pct=0.70)

    assert len(back) == 700
    assert len(forward) == 300


def test_split_holdout():
    """Holdout reserves last N months."""
    from backtester.data.splitting import split_holdout

    # 5 days of data, 1-month holdout → all data is in holdout (training empty)
    df = _make_m1_data(1440 * 5)
    training, holdout = split_holdout(df, holdout_months=1)

    assert len(training) + len(holdout) == len(df)
    # With only 5 days, the 1-month cutoff is before all data
    assert len(training) == 0
    assert len(holdout) == len(df)

    # More realistic: use a very short holdout on same data
    training2, holdout2 = split_holdout(df, holdout_months=0)
    # 0 months = cutoff at last timestamp, so holdout is empty or minimal
    assert len(training2) + len(holdout2) == len(df)


def test_split_data_modes():
    """split_data dispatches correctly."""
    from backtester.data.splitting import split_data

    df = _make_m1_data(1000)

    result = split_data(df, mode="backforward")
    assert "back" in result
    assert "forward" in result
    assert len(result["back"]) + len(result["forward"]) == 1000

    result2 = split_data(df, mode="holdout", holdout_months=1)
    assert "back" in result2
    assert "forward" in result2


def test_split_empty_dataframe():
    """Splitting empty data returns two empty frames."""
    from backtester.data.splitting import split_backforward

    df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    back, forward = split_backforward(df)
    assert len(back) == 0
    assert len(forward) == 0
