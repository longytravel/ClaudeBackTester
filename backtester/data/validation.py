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


def _is_weekend_gap(ts1: pd.Timestamp, ts2: pd.Timestamp) -> bool:
    """Check if a gap spans a weekend (expected market closure).

    Forex market closes Friday ~21:00-22:00 UTC and reopens Sunday ~21:00-22:00 UTC.
    We use a generous window: any gap where ts1 is Fri after 20:00 or ts2 is
    Mon before 01:00, or the gap spans Sat/Sun, counts as a weekend gap.
    """
    d1 = ts1.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
    d2 = ts2.weekday()

    # Gap starts on Friday evening (20:00+ to be generous)
    if d1 == 4 and ts1.hour >= 20:
        return True
    # Gap starts on Saturday (market fully closed)
    if d1 == 5:
        return True
    # Gap starts on Sunday (market closed until ~21:00-22:00 UTC)
    if d1 == 6:
        return True
    # Gap ends on Monday early morning (market just reopened)
    if d2 == 0 and ts2.hour <= 1:
        return True
    # Cross-weekend: Thursday/Friday to Sunday/Monday (multi-day gap including weekend)
    if d1 <= 4 and d2 >= 0 and (ts2 - ts1).days >= 2:
        # Check if a Saturday falls within the gap
        days_ahead = 5 - d1  # days until Saturday
        if days_ahead <= (ts2 - ts1).days:
            return True

    return False


def _is_holiday_gap(ts1: pd.Timestamp, ts2: pd.Timestamp) -> bool:
    """Check if a gap is likely a holiday closure (Christmas, New Year)."""
    # Christmas: Dec 24-26
    if ts1.month == 12 and ts1.day >= 24 and ts1.day <= 26:
        return True
    if ts2.month == 12 and ts2.day >= 24 and ts2.day <= 26:
        return True
    # New Year: Dec 31 - Jan 2
    if ts1.month == 12 and ts1.day >= 31:
        return True
    if ts2.month == 1 and ts2.day <= 2:
        return True
    return False


def detect_gaps(
    df: pd.DataFrame,
    timeframe: str = "M1",
    gap_multiplier: float = 3.0,
) -> dict:
    """Detect timestamp gaps exceeding gap_multiplier * expected interval.

    Distinguishes between expected gaps (weekends, holidays) and unexpected gaps.
    """
    expected_mins = EXPECTED_INTERVALS.get(timeframe, 1)
    threshold = pd.Timedelta(minutes=expected_mins * gap_multiplier)

    diffs = df.index.to_series().diff()
    large_gaps = diffs[diffs > threshold]

    weekend_gaps = 0
    holiday_gaps = 0
    unexpected_gaps = 0
    gap_details = []

    for idx, gap in large_gaps.items():
        loc = df.index.get_loc(idx)
        if loc == 0:
            continue
        prev_idx = df.index[loc - 1]

        if _is_weekend_gap(prev_idx, idx):
            weekend_gaps += 1
        elif _is_holiday_gap(prev_idx, idx):
            holiday_gaps += 1
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
        "holiday_gaps": holiday_gaps,
        "unexpected_gaps": unexpected_gaps,
        "gap_details": gap_details[:20],
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


def detect_anomalies(df: pd.DataFrame, timeframe: str = "M1") -> dict:
    """Detect price anomalies: extreme range, zero-range, OHLC violations.

    Zero-range candles (high == low) are NORMAL for M1 data in quiet periods
    and are only flagged as informational, not penalised.
    """
    ranges = df["high"] - df["low"]
    median_range = ranges.median()

    # Extreme range: use a higher threshold for M1 (50x) vs higher TFs (10x)
    # M1 has naturally tiny median ranges, so 10x catches normal active candles
    extreme_multiplier = 50.0 if timeframe == "M1" else 20.0 if timeframe == "M5" else 10.0
    extreme = int((ranges > median_range * extreme_multiplier).sum()) if median_range > 0 else 0

    # Zero-range: informational for M1 (completely normal)
    zero_range = int((ranges == 0).sum())
    zero_range_pct = round(zero_range / len(df) * 100, 2) if len(df) > 0 else 0

    # OHLC violations (high < low) — this is always a data error
    violations = int((df["high"] < df["low"]).sum())

    return {
        "extreme_range_candles": extreme,
        "extreme_range_threshold": f"{extreme_multiplier}x median",
        "zero_range_candles": zero_range,
        "zero_range_pct": zero_range_pct,
        "ohlc_violations": violations,
        "median_range": float(median_range) if not np.isnan(median_range) else 0,
    }


def check_yearly_coverage(df: pd.DataFrame, expected_start_year: int = 2005) -> dict:
    """Check for missing or sparse years in the data.

    A year is 'sparse' if it has less than 50% of the expected M1 candles
    (~365 * 5 * 24 * 60 * 0.5 ≈ 130,000 candles, accounting for weekends).
    """
    if df.empty:
        return {"missing_years": [], "sparse_years": [], "good_years": []}

    actual_start = df.index[0].year
    actual_end = df.index[-1].year
    min_candles_per_year = 130_000  # ~50% of expected M1 candles

    missing = []
    sparse = []
    good = []

    for year in range(expected_start_year, actual_end + 1):
        year_data = df[df.index.year == year]
        count = len(year_data)

        if count == 0:
            missing.append(year)
        elif count < min_candles_per_year:
            sparse.append({"year": year, "candles": count})
        else:
            good.append({"year": year, "candles": count})

    return {
        "missing_years": missing,
        "sparse_years": sparse,
        "good_years": good,
        "usable_from": good[0]["year"] if good else None,
    }


def compute_quality_score(
    df: pd.DataFrame,
    timeframe: str = "M1",
    expected_start_year: int = 2005,
) -> dict:
    """Compute data quality score (0-100).

    Scoring approach:
    - Start at 100
    - Deduct for REAL problems: unexpected gaps, zeros, NaNs, OHLC violations
    - DO NOT penalise: weekend gaps, holiday gaps, zero-range M1 candles (all normal)
    - Separate coverage assessment: missing/sparse years flagged but scored independently
    """
    score = 100.0

    gaps = detect_gaps(df, timeframe)
    zeros = detect_zeros_nans(df)
    anomalies = detect_anomalies(df, timeframe)
    coverage = check_yearly_coverage(df, expected_start_year)

    # --- Critical deductions ---
    # Unexpected gaps: 0.5 points each, up to -20
    # (most "gaps" in good data are just low-liquidity hours, not real problems)
    score -= min(20, gaps["unexpected_gaps"] * 0.5)

    # NaN values: 5 points each, up to -20 (these are real data errors)
    score -= min(20, zeros["nan_count"] * 5)

    # Zero OHLC values: 5 points each, up to -20 (price can never be 0)
    score -= min(20, zeros["zero_count"] * 5)

    # OHLC violations: 10 points each, up to -20 (high < low = corrupt data)
    score -= min(20, anomalies["ohlc_violations"] * 10)

    # Extreme range candles: 0.1 points each, up to -5 (mild concern, could be real volatility)
    score -= min(5, anomalies["extreme_range_candles"] * 0.1)

    # --- Coverage penalty (separate from data quality) ---
    # Missing years reduce score proportionally but aren't catastrophic
    # (you can still backtest on the years you have)
    total_expected_years = max(1, datetime_now_year() - expected_start_year + 1)
    years_with_data = len(coverage["good_years"])
    coverage_ratio = years_with_data / total_expected_years
    coverage_penalty = max(0, (1 - coverage_ratio) * 15)  # Up to -15 for poor coverage
    score -= coverage_penalty

    score = max(0, min(100, score))

    result = {
        "quality_score": round(score, 1),
        "total_candles": len(df),
        "date_range": f"{df.index[0]} to {df.index[-1]}" if len(df) > 0 else "empty",
        "gaps": gaps,
        "zeros_nans": zeros,
        "anomalies": anomalies,
        "coverage": coverage,
        "coverage_penalty": round(coverage_penalty, 1),
    }

    # Log warnings for things the user should know about
    if coverage["missing_years"]:
        log.warning(
            "missing_years",
            years=coverage["missing_years"],
            usable_from=coverage.get("usable_from"),
        )
    if coverage["sparse_years"]:
        log.warning("sparse_years", years=coverage["sparse_years"])

    log.info("quality_score", score=result["quality_score"], candles=result["total_candles"])
    return result


def datetime_now_year() -> int:
    """Get current year (separated for testability)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).year


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
