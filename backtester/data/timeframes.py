"""Timeframe conversion: build higher timeframes from M1 data.

M1 is the single source of truth. All higher timeframes are built
using correct OHLCV aggregation: first open, max high, min low,
last close, sum volume.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()

# Mapping of our timeframe names to pandas resample rules
TIMEFRAME_RULES = {
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "1h",
    "H4": "4h",
    "D": "1D",
    "W": "W-MON",
}

ALL_TIMEFRAMES = list(TIMEFRAME_RULES.keys())


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M1 OHLCV data to a higher timeframe.

    Uses correct aggregation: first open, max high, min low, last close, sum volume.
    If spread column exists, takes the median spread for the period.
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    if "spread" in df.columns:
        agg["spread"] = "median"

    resampled = df.resample(rule).agg(agg)
    # Drop rows with no price data (volume sum=0 survives dropna(how="all"))
    resampled = resampled.dropna(subset=["close"])
    return resampled


def convert_single_timeframe(
    m1_path: Path,
    output_path: Path,
    rule: str,
) -> Path:
    """Convert M1 parquet to a single higher timeframe."""
    df = pd.read_parquet(m1_path)
    resampled = resample_ohlcv(df, rule)

    tmp = output_path.with_suffix(".parquet.tmp")
    resampled.to_parquet(tmp, engine="pyarrow", compression="snappy")
    tmp.replace(output_path)

    log.info(
        "converted",
        source=m1_path.name,
        target=output_path.name,
        input_rows=len(df),
        output_rows=len(resampled),
    )
    return output_path


def convert_timeframes(
    pair: str,
    data_dir: str,
    timeframes: list[str] | None = None,
) -> dict[str, Path]:
    """Build all higher timeframes from consolidated M1 data.

    Returns dict mapping timeframe name to output file path.
    """
    if timeframes is None:
        timeframes = ALL_TIMEFRAMES

    pair_file = pair.replace("/", "_")
    m1_path = Path(data_dir) / f"{pair_file}_M1.parquet"

    if not m1_path.exists():
        raise FileNotFoundError(f"M1 data not found: {m1_path}")

    results = {}
    for tf in timeframes:
        rule = TIMEFRAME_RULES.get(tf)
        if rule is None:
            log.warning("unknown_timeframe", timeframe=tf)
            continue

        output_path = Path(data_dir) / f"{pair_file}_{tf}.parquet"
        convert_single_timeframe(m1_path, output_path, rule)
        results[tf] = output_path

    log.info("convert_timeframes_complete", pair=pair, timeframes=list(results.keys()))
    return results


def build_h1_to_m1_mapping(
    h1_timestamps: np.ndarray,
    m1_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map each H1 bar to its range of M1 sub-bars.

    For each H1 bar at index i, finds all M1 bars whose timestamp falls
    within [h1_timestamps[i], h1_timestamps[i] + 1 hour). Uses
    np.searchsorted for O(n log n) performance.

    Args:
        h1_timestamps: (N,) int64 array of H1 bar timestamps (Unix nanoseconds).
        m1_timestamps: (M,) int64 array of M1 bar timestamps (Unix nanoseconds).

    Returns:
        (start_idx, end_idx): Two int64 arrays of length N.
        start_idx[i] = first M1 bar index in H1 bar i.
        end_idx[i] = one past the last M1 bar index in H1 bar i.
        If an H1 bar has no M1 bars, start_idx[i] == end_idx[i].
    """
    n_h1 = len(h1_timestamps)
    start_idx = np.empty(n_h1, dtype=np.int64)
    end_idx = np.empty(n_h1, dtype=np.int64)

    for i in range(n_h1):
        h1_start = h1_timestamps[i]
        # H1 bar covers [h1_start, h1_start + 1 hour)
        if i + 1 < n_h1:
            h1_end = h1_timestamps[i + 1]
        else:
            # Last H1 bar: assume 1 hour duration (3600 seconds in nanoseconds)
            h1_end = h1_start + np.int64(3600_000_000_000)

        start_idx[i] = np.searchsorted(m1_timestamps, h1_start, side="left")
        end_idx[i] = np.searchsorted(m1_timestamps, h1_end, side="left")

    return start_idx, end_idx
