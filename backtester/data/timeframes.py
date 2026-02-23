"""Timeframe conversion: build higher timeframes from M1 data.

M1 is the single source of truth. All higher timeframes are built
using correct OHLCV aggregation: first open, max high, min low,
last close, sum volume.
"""

from pathlib import Path

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
    # Drop rows where all values are NaN (no data in that period)
    resampled = resampled.dropna(how="all")
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
