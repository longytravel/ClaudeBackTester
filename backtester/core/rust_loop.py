"""Rust backend (backtester_core) — the sole hot loop implementation.

Imports the Rust native extension built with PyO3 + Rayon.
Build with: cd rust && bash build.sh
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import backtester_core

    batch_evaluate = backtester_core.batch_evaluate
    logger.info("Using Rust native backend (backtester_core)")
except ImportError as e:
    raise ImportError(
        f"backtester_core not found: {e}. "
        "Build with: cd rust && bash build.sh"
    ) from e

# Re-export PL_* constants from Rust module
NUM_PL = backtester_core.NUM_PL
PL_SL_MODE = backtester_core.PL_SL_MODE
PL_SL_FIXED_PIPS = backtester_core.PL_SL_FIXED_PIPS
PL_SL_ATR_MULT = backtester_core.PL_SL_ATR_MULT
PL_TP_MODE = backtester_core.PL_TP_MODE
PL_TP_RR_RATIO = backtester_core.PL_TP_RR_RATIO
PL_TP_ATR_MULT = backtester_core.PL_TP_ATR_MULT
PL_TP_FIXED_PIPS = backtester_core.PL_TP_FIXED_PIPS
PL_HOURS_START = backtester_core.PL_HOURS_START
PL_HOURS_END = backtester_core.PL_HOURS_END
PL_DAYS_BITMASK = backtester_core.PL_DAYS_BITMASK
PL_TRAILING_MODE = backtester_core.PL_TRAILING_MODE
PL_TRAIL_ACTIVATE = backtester_core.PL_TRAIL_ACTIVATE
PL_TRAIL_DISTANCE = backtester_core.PL_TRAIL_DISTANCE
PL_TRAIL_ATR_MULT = backtester_core.PL_TRAIL_ATR_MULT
PL_BREAKEVEN_ENABLED = backtester_core.PL_BREAKEVEN_ENABLED
PL_BREAKEVEN_TRIGGER = backtester_core.PL_BREAKEVEN_TRIGGER
PL_BREAKEVEN_OFFSET = backtester_core.PL_BREAKEVEN_OFFSET
PL_PARTIAL_ENABLED = backtester_core.PL_PARTIAL_ENABLED
PL_PARTIAL_PCT = backtester_core.PL_PARTIAL_PCT
PL_PARTIAL_TRIGGER = backtester_core.PL_PARTIAL_TRIGGER
PL_MAX_BARS = backtester_core.PL_MAX_BARS
PL_STALE_ENABLED = backtester_core.PL_STALE_ENABLED
PL_STALE_BARS = backtester_core.PL_STALE_BARS
PL_STALE_ATR_THRESH = backtester_core.PL_STALE_ATR_THRESH
PL_SIGNAL_VARIANT = backtester_core.PL_SIGNAL_VARIANT
PL_BUY_FILTER_MAX = backtester_core.PL_BUY_FILTER_MAX
PL_SELL_FILTER_MIN = backtester_core.PL_SELL_FILTER_MIN

# Generic signal param slots (for expanded signal filtering)
PL_SIGNAL_P0 = backtester_core.PL_SIGNAL_P0
PL_SIGNAL_P1 = backtester_core.PL_SIGNAL_P1
PL_SIGNAL_P2 = backtester_core.PL_SIGNAL_P2
PL_SIGNAL_P3 = backtester_core.PL_SIGNAL_P3
PL_SIGNAL_P4 = backtester_core.PL_SIGNAL_P4
PL_SIGNAL_P5 = backtester_core.PL_SIGNAL_P5
PL_SIGNAL_P6 = backtester_core.PL_SIGNAL_P6
PL_SIGNAL_P7 = backtester_core.PL_SIGNAL_P7
PL_SIGNAL_P8 = backtester_core.PL_SIGNAL_P8
PL_SIGNAL_P9 = backtester_core.PL_SIGNAL_P9
NUM_SIGNAL_PARAMS = backtester_core.NUM_SIGNAL_PARAMS


def get_backend_name() -> str:
    """Return the name of the active backend."""
    return "rust"
