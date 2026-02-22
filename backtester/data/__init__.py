"""Data management: download, cache, validate, split, and serve historical price data."""

from backtester.data.downloader import (
    ALL_PAIRS,
    DEFAULT_DATA_DIR,
    download_pair,
    ensure_fresh,
)
from backtester.data.splitting import split_backforward, split_data, split_holdout
from backtester.data.timeframes import convert_timeframes, resample_ohlcv
from backtester.data.validation import compute_quality_score, validate_data

__all__ = [
    "ALL_PAIRS",
    "DEFAULT_DATA_DIR",
    "compute_quality_score",
    "convert_timeframes",
    "download_pair",
    "ensure_fresh",
    "resample_ohlcv",
    "split_backforward",
    "split_data",
    "split_holdout",
    "validate_data",
]
