"""Dukascopy historical data downloader.

Downloads M1 OHLCV data from Dukascopy's free data feed, stores as yearly
Parquet chunks on Google Drive, then consolidates into a single file per pair.
Supports resume (skips already-downloaded years) and incremental updates.
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import dukascopy_python as dk
import pandas as pd
import structlog

log = structlog.get_logger()

# Default data directory (Google Drive)
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "G:/My Drive/BackTestData")

# All target pairs (Dukascopy instrument format)
ALL_PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD",
    "NZD/USD", "USD/CHF", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "AUD/JPY", "EUR/AUD", "EUR/CAD", "EUR/CHF", "GBP/AUD",
    "GBP/CAD", "GBP/CHF", "AUD/CAD", "AUD/NZD", "NZD/JPY",
    "CAD/JPY", "CHF/JPY", "EUR/NZD", "GBP/NZD", "XAU/USD",
]

# Start year for historical data
DEFAULT_START_YEAR = 2005


def _pair_to_filename(pair: str) -> str:
    """Convert 'EUR/USD' to 'EUR_USD'."""
    return pair.replace("/", "_")


def _chunk_dir(data_dir: str, pair: str) -> Path:
    """Get the yearly chunks directory for a pair."""
    return Path(data_dir) / f"{_pair_to_filename(pair)}_M1_chunks"


def _chunk_path(data_dir: str, pair: str, year: int) -> Path:
    """Get path for a specific yearly chunk."""
    return _chunk_dir(data_dir, pair) / f"{_pair_to_filename(pair)}_M1_{year}.parquet"


def _consolidated_path(data_dir: str, pair: str) -> Path:
    """Get path for the consolidated M1 file."""
    return Path(data_dir) / f"{_pair_to_filename(pair)}_M1.parquet"


def download_year(
    pair: str,
    year: int,
    data_dir: str = DEFAULT_DATA_DIR,
    offer_side: str = dk.OFFER_SIDE_BID,
) -> pd.DataFrame | None:
    """Download one year of M1 data for a pair.

    Returns the DataFrame, or None if no data available for that year.
    """
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    # End at Jan 1 of next year (exclusive)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    # Don't try to download future data
    now = datetime.now(timezone.utc)
    if start > now:
        return None
    if end > now:
        end = now

    log.info("downloading", pair=pair, year=year)
    t0 = time.time()

    df = dk.fetch(
        instrument=pair,
        interval=dk.INTERVAL_MIN_1,
        offer_side=offer_side,
        start=start,
        end=end,
        max_retries=7,
    )

    elapsed = time.time() - t0

    if df is None or df.empty:
        log.warning("no_data", pair=pair, year=year)
        return None

    log.info(
        "downloaded",
        pair=pair,
        year=year,
        rows=len(df),
        elapsed_s=round(elapsed, 1),
    )
    return df


def save_chunk(df: pd.DataFrame, data_dir: str, pair: str, year: int) -> Path:
    """Save a yearly chunk as Parquet with atomic write."""
    chunk_d = _chunk_dir(data_dir, pair)
    chunk_d.mkdir(parents=True, exist_ok=True)

    target = _chunk_path(data_dir, pair, year)
    tmp = target.with_suffix(".parquet.tmp")

    df.to_parquet(tmp, engine="pyarrow", compression="snappy")
    tmp.replace(target)

    log.info("saved_chunk", path=str(target), rows=len(df))
    return target


def consolidate_chunks(data_dir: str, pair: str) -> Path:
    """Merge all yearly chunks into a single consolidated M1 Parquet file."""
    chunk_d = _chunk_dir(data_dir, pair)
    if not chunk_d.exists():
        raise FileNotFoundError(f"No chunks directory: {chunk_d}")

    chunks = sorted(chunk_d.glob("*.parquet"))
    if not chunks:
        raise FileNotFoundError(f"No chunk files in {chunk_d}")

    dfs = []
    for c in chunks:
        dfs.append(pd.read_parquet(c))

    df = pd.concat(dfs)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    target = _consolidated_path(data_dir, pair)
    tmp = target.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, engine="pyarrow", compression="snappy")
    tmp.replace(target)

    log.info(
        "consolidated",
        pair=pair,
        total_rows=len(df),
        from_date=str(df.index[0]),
        to_date=str(df.index[-1]),
    )
    return target


def get_downloaded_years(data_dir: str, pair: str) -> list[int]:
    """Return list of years already downloaded for a pair."""
    chunk_d = _chunk_dir(data_dir, pair)
    if not chunk_d.exists():
        return []

    years = []
    for f in chunk_d.glob("*.parquet"):
        # Parse year from filename like EUR_USD_M1_2025.parquet
        try:
            year = int(f.stem.split("_")[-1])
            years.append(year)
        except ValueError:
            continue
    return sorted(years)


def download_pair(
    pair: str,
    data_dir: str = DEFAULT_DATA_DIR,
    start_year: int = DEFAULT_START_YEAR,
    force: bool = False,
) -> Path:
    """Download full M1 history for a pair, with resume support.

    Skips years already downloaded unless force=True.
    Returns path to consolidated Parquet file.
    """
    current_year = datetime.now(timezone.utc).year
    years_needed = list(range(start_year, current_year + 1))

    if not force:
        existing = set(get_downloaded_years(data_dir, pair))
        # Always re-download current year (for incremental updates)
        years_to_download = [y for y in years_needed if y not in existing or y == current_year]
    else:
        years_to_download = years_needed

    fname = _pair_to_filename(pair)
    log.info(
        "download_pair_start",
        pair=pair,
        years_total=len(years_needed),
        years_to_download=len(years_to_download),
        skipping=len(years_needed) - len(years_to_download),
    )

    for i, year in enumerate(years_to_download):
        log.info(
            "progress",
            pair=pair,
            step=f"{i + 1}/{len(years_to_download)}",
            year=year,
        )
        df = download_year(pair, year, data_dir)
        if df is not None and not df.empty:
            save_chunk(df, data_dir, pair, year)

    # Consolidate all chunks
    consolidated = consolidate_chunks(data_dir, pair)
    log.info("download_pair_complete", pair=pair, path=str(consolidated))
    return consolidated


def download_all_pairs(
    pairs: list[str] | None = None,
    data_dir: str = DEFAULT_DATA_DIR,
    start_year: int = DEFAULT_START_YEAR,
    force: bool = False,
) -> dict[str, Path]:
    """Download M1 data for all configured pairs.

    Returns dict mapping pair name to consolidated file path.
    """
    if pairs is None:
        pairs = ALL_PAIRS

    results = {}
    total = len(pairs)

    for i, pair in enumerate(pairs):
        log.info("pair_progress", pair=pair, step=f"{i + 1}/{total}")
        try:
            path = download_pair(pair, data_dir, start_year, force)
            results[pair] = path
        except Exception:
            log.exception("download_failed", pair=pair)
            continue

    log.info("download_all_complete", successful=len(results), total=total)
    return results


def get_latest_timestamp(data_dir: str, pair: str) -> datetime | None:
    """Get the latest timestamp in the consolidated M1 file."""
    path = _consolidated_path(data_dir, pair)
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    if df.empty:
        return None

    last_ts = df.index[-1]
    if hasattr(last_ts, "to_pydatetime"):
        return last_ts.to_pydatetime()
    return last_ts


def update_pair(
    pair: str,
    data_dir: str = DEFAULT_DATA_DIR,
) -> Path:
    """Incremental update: only download data newer than what we have."""
    # Re-download current year's chunk (contains latest data)
    current_year = datetime.now(timezone.utc).year
    df = download_year(pair, current_year, data_dir)
    if df is not None and not df.empty:
        save_chunk(df, data_dir, pair, current_year)

    return consolidate_chunks(data_dir, pair)


def is_stale(pair: str, data_dir: str = DEFAULT_DATA_DIR, max_age_hours: float = 2.0) -> bool:
    """Check if data for a pair is stale (older than max_age_hours)."""
    latest = get_latest_timestamp(data_dir, pair)
    if latest is None:
        return True

    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)

    age = datetime.now(timezone.utc) - latest
    return age.total_seconds() > max_age_hours * 3600


def ensure_fresh(
    pair: str,
    data_dir: str = DEFAULT_DATA_DIR,
    max_age_hours: float = 2.0,
) -> Path:
    """Ensure data is fresh before backtest. Downloads if stale or missing."""
    consolidated = _consolidated_path(data_dir, pair)

    if not consolidated.exists():
        log.info("no_data_found_downloading", pair=pair)
        return download_pair(pair, data_dir)

    if is_stale(pair, data_dir, max_age_hours):
        log.info("data_stale_updating", pair=pair)
        return update_pair(pair, data_dir)

    log.info("data_fresh", pair=pair)
    return consolidated
