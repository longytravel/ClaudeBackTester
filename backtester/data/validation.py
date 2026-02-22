"""Data validation: gap detection, anomaly checks, and quality scoring.

Validates downloaded price data for integrity before backtesting.
"""

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

# Expected intervals in minutes for each timeframe
EXPECTED_INTERVALS = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D": 1440, "W": 10080,
}

# Weekend hours (Forex market closed Fri 22:00 UTC to Sun 22:00 UTC)
WEEKEND_CLOSE_HOUR = 22  # Friday
WEEKEND_OPEN_HOUR = 22   # Sunday


def _is_weekend_gap(ts1: pd.Timestamp, ts2: pd.Timestamp) -> bool:
    """Check if a gap spans a weekend (expected market closure)."""
    # Friday after 22:00 UTC to Sunday 22:00 UTC is expected
    if ts1.weekday() == 4 and ts1.hour >= WEEKEND_CLOSE_HOUR:
        return True
    if ts1.weekday() == 5:  # Saturday
        return True
    if ts1.weekday() == 6 and ts2.weekday() == 0:  # Sunday to Monday
        return True
    return False


def detect_gaps(
    df: pd.DataFrame,
    timeframe: str = "M1",
    gap_multiplier: float = 3.0,
) -> dict:
    """Detect timestamp gaps exceeding gap_multiplier * expected interval.

    Returns dict with gap counts and details.
    """
    expected_mins = EXPECTED_INTERVALS.get(timeframe, 1)
    threshold = pd.Timedelta(minutes=expected_mins * gap_multiplier)

    diffs = df.index.to_series().diff()
    large_gaps = diffs[diffs > threshold]

    weekend_gaps = 0
    unexpected_gaps = 0
    gap_details = []

    for idx, gap in large_gaps.items():
        prev_idx = df.index[df.index.get_loc(idx) - 1]
        if _is_weekend_gap(prev_idx, idx):
            weekend_gaps += 1
        else:
            unexpected_gaps += 1
            gap_details.append({
                "from": str(prev_idx),
                "to": str(idx),
                "duration_minutes": gap.total_seconds() / 60,
            })

    return {
        "total_gaps": len(large_gaps),
        "weekend_gaps": weekend_gaps,
        "unexpected_gaps": unexpected_gaps,
        "gap_details": gap_details[:20],  # Limit detail output
    }


def detect_zeros_nans(df: pd.DataFrame) -> dict:
    """Detect zero or NaN values in OHLC fields."""
    ohlc = df[["open", "high", "low", "close"]]

    nan_count = int(ohlc.isna().sum().sum())
    zero_count = int((ohlc == 0).sum().sum())
    total_cells = ohlc.size

    return {
        "nan_count": nan_count,
        "zero_count": zero_count,
        "total_cells": total_cells,
        "nan_pct": round(nan_count / total_cells * 100, 4) if total_cells else 0,
        "zero_pct": round(zero_count / total_cells * 100, 4) if total_cells else 0,
    }


def detect_anomalies(df: pd.DataFrame) -> dict:
    """Detect price anomalies: extreme range, zero-range, OHLC violations."""
    ranges = df["high"] - df["low"]
    median_range = ranges.median()

    # Extreme range candles (>10x median)
    extreme = int((ranges > median_range * 10).sum()) if median_range > 0 else 0

    # Zero-range candles (high == low)
    zero_range = int((ranges == 0).sum())

    # OHLC violations (high < low)
    violations = int((df["high"] < df["low"]).sum())

    return {
        "extreme_range_candles": extreme,
        "zero_range_candles": zero_range,
        "ohlc_violations": violations,
        "median_range": float(median_range) if not np.isnan(median_range) else 0,
    }


def compute_quality_score(
    df: pd.DataFrame,
    timeframe: str = "M1",
) -> dict:
    """Compute data quality score (0-100).

    Deducts heavily for critical issues (unexpected gaps, zeros, NaNs)
    and lightly for minor issues (weekend gaps, anomalies).
    """
    score = 100.0

    gaps = detect_gaps(df, timeframe)
    zeros = detect_zeros_nans(df)
    anomalies = detect_anomalies(df)

    # Critical deductions
    score -= min(30, gaps["unexpected_gaps"] * 2)  # Up to -30 for gaps
    score -= min(20, zeros["nan_count"] * 5)        # Up to -20 for NaNs
    score -= min(20, zeros["zero_count"] * 5)       # Up to -20 for zeros
    score -= min(10, anomalies["ohlc_violations"] * 10)  # Up to -10 for violations

    # Minor deductions
    score -= min(5, anomalies["extreme_range_candles"] * 0.5)
    score -= min(5, anomalies["zero_range_candles"] * 0.01)

    score = max(0, min(100, score))

    result = {
        "quality_score": round(score, 1),
        "total_candles": len(df),
        "date_range": f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "empty",
        "gaps": gaps,
        "zeros_nans": zeros,
        "anomalies": anomalies,
    }

    log.info("quality_score", score=result["quality_score"], candles=result["total_candles"])
    return result


def validate_data(
    df: pd.DataFrame,
    timeframe: str = "M1",
    min_score: float = 50.0,
    min_candles: int = 5000,
) -> dict:
    """Full validation: compute quality score and check thresholds.

    Returns validation result dict with pass/fail and details.
    """
    if df.empty:
        return {"passed": False, "reason": "empty_dataframe", "quality_score": 0}

    if len(df) < min_candles:
        return {
            "passed": False,
            "reason": f"insufficient_candles ({len(df)} < {min_candles})",
            "quality_score": 0,
            "total_candles": len(df),
        }

    quality = compute_quality_score(df, timeframe)

    passed = quality["quality_score"] >= min_score
    quality["passed"] = passed
    if not passed:
        quality["reason"] = f"quality_score {quality['quality_score']} < {min_score}"

    return quality
