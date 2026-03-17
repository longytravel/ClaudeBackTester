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
    _HAS_RUST = True
except ImportError:
    batch_evaluate = None  # type: ignore[assignment]
    _HAS_RUST = False
    logger.warning(
        "backtester_core not available — PL constants use Python fallbacks. "
        "batch_evaluate disabled. Build with: cd rust && bash build.sh"
    )

# PL_* constants — from Rust if available, Python fallback otherwise.
# Values must match rust/src/constants.rs exactly.
NUM_PL = getattr(backtester_core, "NUM_PL", 64) if _HAS_RUST else 64
PL_SL_MODE = getattr(backtester_core, "PL_SL_MODE", 0) if _HAS_RUST else 0
PL_SL_FIXED_PIPS = getattr(backtester_core, "PL_SL_FIXED_PIPS", 1) if _HAS_RUST else 1
PL_SL_ATR_MULT = getattr(backtester_core, "PL_SL_ATR_MULT", 2) if _HAS_RUST else 2
PL_TP_MODE = getattr(backtester_core, "PL_TP_MODE", 3) if _HAS_RUST else 3
PL_TP_RR_RATIO = getattr(backtester_core, "PL_TP_RR_RATIO", 4) if _HAS_RUST else 4
PL_TP_ATR_MULT = getattr(backtester_core, "PL_TP_ATR_MULT", 5) if _HAS_RUST else 5
PL_TP_FIXED_PIPS = getattr(backtester_core, "PL_TP_FIXED_PIPS", 6) if _HAS_RUST else 6
PL_HOURS_START = getattr(backtester_core, "PL_HOURS_START", 7) if _HAS_RUST else 7
PL_HOURS_END = getattr(backtester_core, "PL_HOURS_END", 8) if _HAS_RUST else 8
PL_DAYS_BITMASK = getattr(backtester_core, "PL_DAYS_BITMASK", 9) if _HAS_RUST else 9
PL_TRAILING_MODE = getattr(backtester_core, "PL_TRAILING_MODE", 10) if _HAS_RUST else 10
PL_TRAIL_ACTIVATE = getattr(backtester_core, "PL_TRAIL_ACTIVATE", 11) if _HAS_RUST else 11
PL_TRAIL_DISTANCE = getattr(backtester_core, "PL_TRAIL_DISTANCE", 12) if _HAS_RUST else 12
PL_TRAIL_ATR_MULT = getattr(backtester_core, "PL_TRAIL_ATR_MULT", 13) if _HAS_RUST else 13
PL_BREAKEVEN_ENABLED = getattr(backtester_core, "PL_BREAKEVEN_ENABLED", 14) if _HAS_RUST else 14
PL_BREAKEVEN_TRIGGER = getattr(backtester_core, "PL_BREAKEVEN_TRIGGER", 15) if _HAS_RUST else 15
PL_BREAKEVEN_OFFSET = getattr(backtester_core, "PL_BREAKEVEN_OFFSET", 16) if _HAS_RUST else 16
PL_PARTIAL_ENABLED = getattr(backtester_core, "PL_PARTIAL_ENABLED", 17) if _HAS_RUST else 17
PL_PARTIAL_PCT = getattr(backtester_core, "PL_PARTIAL_PCT", 18) if _HAS_RUST else 18
PL_PARTIAL_TRIGGER = getattr(backtester_core, "PL_PARTIAL_TRIGGER", 19) if _HAS_RUST else 19
PL_MAX_BARS = getattr(backtester_core, "PL_MAX_BARS", 20) if _HAS_RUST else 20
PL_STALE_ENABLED = getattr(backtester_core, "PL_STALE_ENABLED", 21) if _HAS_RUST else 21
PL_STALE_BARS = getattr(backtester_core, "PL_STALE_BARS", 22) if _HAS_RUST else 22
PL_STALE_ATR_THRESH = getattr(backtester_core, "PL_STALE_ATR_THRESH", 23) if _HAS_RUST else 23
PL_SIGNAL_VARIANT = getattr(backtester_core, "PL_SIGNAL_VARIANT", 24) if _HAS_RUST else 24
PL_BUY_FILTER_MAX = getattr(backtester_core, "PL_BUY_FILTER_MAX", 25) if _HAS_RUST else 25
PL_SELL_FILTER_MIN = getattr(backtester_core, "PL_SELL_FILTER_MIN", 26) if _HAS_RUST else 26

# Generic signal param slots (for expanded signal filtering)
PL_SIGNAL_P0 = getattr(backtester_core, "PL_SIGNAL_P0", 27) if _HAS_RUST else 27
PL_SIGNAL_P1 = getattr(backtester_core, "PL_SIGNAL_P1", 28) if _HAS_RUST else 28
PL_SIGNAL_P2 = getattr(backtester_core, "PL_SIGNAL_P2", 29) if _HAS_RUST else 29
PL_SIGNAL_P3 = getattr(backtester_core, "PL_SIGNAL_P3", 30) if _HAS_RUST else 30
PL_SIGNAL_P4 = getattr(backtester_core, "PL_SIGNAL_P4", 31) if _HAS_RUST else 31
PL_SIGNAL_P5 = getattr(backtester_core, "PL_SIGNAL_P5", 32) if _HAS_RUST else 32
PL_SIGNAL_P6 = getattr(backtester_core, "PL_SIGNAL_P6", 33) if _HAS_RUST else 33
PL_SIGNAL_P7 = getattr(backtester_core, "PL_SIGNAL_P7", 34) if _HAS_RUST else 34
PL_SIGNAL_P8 = getattr(backtester_core, "PL_SIGNAL_P8", 35) if _HAS_RUST else 35
PL_SIGNAL_P9 = getattr(backtester_core, "PL_SIGNAL_P9", 36) if _HAS_RUST else 36
NUM_SIGNAL_PARAMS = getattr(backtester_core, "NUM_SIGNAL_PARAMS", 10) if _HAS_RUST else 10


def get_backend_name() -> str:
    """Return the name of the active backend."""
    return "rust"
