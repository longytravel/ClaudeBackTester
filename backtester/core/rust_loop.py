"""Backend dispatcher: Rust (backtester_core) with Numba fallback.

Tries to import the Rust native extension first. If not available,
falls back to the Numba JIT implementation.

Control via environment variable:
    BACKTESTER_BACKEND=rust   — force Rust (error if not available)
    BACKTESTER_BACKEND=numba  — force Numba
    BACKTESTER_BACKEND=auto   — try Rust, fall back to Numba (default)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_backend = os.environ.get("BACKTESTER_BACKEND", "auto").lower()

_using_rust = False

if _backend == "numba":
    logger.info("BACKTESTER_BACKEND=numba — using Numba JIT backend")
    from backtester.core.jit_loop import batch_evaluate  # noqa: F401
    from backtester.core.jit_loop import (  # noqa: F401
        NUM_PL,
        PL_BREAKEVEN_ENABLED,
        PL_BREAKEVEN_OFFSET,
        PL_BREAKEVEN_TRIGGER,
        PL_BUY_FILTER_MAX,
        PL_DAYS_BITMASK,
        PL_HOURS_END,
        PL_HOURS_START,
        PL_MAX_BARS,
        PL_PARTIAL_ENABLED,
        PL_PARTIAL_PCT,
        PL_PARTIAL_TRIGGER,
        PL_SELL_FILTER_MIN,
        PL_SIGNAL_VARIANT,
        PL_SL_ATR_MULT,
        PL_SL_FIXED_PIPS,
        PL_SL_MODE,
        PL_STALE_ATR_THRESH,
        PL_STALE_BARS,
        PL_STALE_ENABLED,
        PL_TP_ATR_MULT,
        PL_TP_FIXED_PIPS,
        PL_TP_MODE,
        PL_TP_RR_RATIO,
        PL_TRAILING_MODE,
        PL_TRAIL_ACTIVATE,
        PL_TRAIL_ATR_MULT,
        PL_TRAIL_DISTANCE,
    )
elif _backend == "rust":
    try:
        import backtester_core

        batch_evaluate = backtester_core.batch_evaluate  # noqa: F811
        _using_rust = True
        logger.info("BACKTESTER_BACKEND=rust — using Rust native backend")
    except ImportError as e:
        raise ImportError(
            f"BACKTESTER_BACKEND=rust but backtester_core not found: {e}. "
            "Build with: cd rust && maturin develop --release"
        ) from e

    # Re-export PL_* constants from Rust module
    NUM_PL = backtester_core.NUM_PL  # noqa: F811
    PL_SL_MODE = backtester_core.PL_SL_MODE  # noqa: F811
    PL_SL_FIXED_PIPS = backtester_core.PL_SL_FIXED_PIPS  # noqa: F811
    PL_SL_ATR_MULT = backtester_core.PL_SL_ATR_MULT  # noqa: F811
    PL_TP_MODE = backtester_core.PL_TP_MODE  # noqa: F811
    PL_TP_RR_RATIO = backtester_core.PL_TP_RR_RATIO  # noqa: F811
    PL_TP_ATR_MULT = backtester_core.PL_TP_ATR_MULT  # noqa: F811
    PL_TP_FIXED_PIPS = backtester_core.PL_TP_FIXED_PIPS  # noqa: F811
    PL_HOURS_START = backtester_core.PL_HOURS_START  # noqa: F811
    PL_HOURS_END = backtester_core.PL_HOURS_END  # noqa: F811
    PL_DAYS_BITMASK = backtester_core.PL_DAYS_BITMASK  # noqa: F811
    PL_TRAILING_MODE = backtester_core.PL_TRAILING_MODE  # noqa: F811
    PL_TRAIL_ACTIVATE = backtester_core.PL_TRAIL_ACTIVATE  # noqa: F811
    PL_TRAIL_DISTANCE = backtester_core.PL_TRAIL_DISTANCE  # noqa: F811
    PL_TRAIL_ATR_MULT = backtester_core.PL_TRAIL_ATR_MULT  # noqa: F811
    PL_BREAKEVEN_ENABLED = backtester_core.PL_BREAKEVEN_ENABLED  # noqa: F811
    PL_BREAKEVEN_TRIGGER = backtester_core.PL_BREAKEVEN_TRIGGER  # noqa: F811
    PL_BREAKEVEN_OFFSET = backtester_core.PL_BREAKEVEN_OFFSET  # noqa: F811
    PL_PARTIAL_ENABLED = backtester_core.PL_PARTIAL_ENABLED  # noqa: F811
    PL_PARTIAL_PCT = backtester_core.PL_PARTIAL_PCT  # noqa: F811
    PL_PARTIAL_TRIGGER = backtester_core.PL_PARTIAL_TRIGGER  # noqa: F811
    PL_MAX_BARS = backtester_core.PL_MAX_BARS  # noqa: F811
    PL_STALE_ENABLED = backtester_core.PL_STALE_ENABLED  # noqa: F811
    PL_STALE_BARS = backtester_core.PL_STALE_BARS  # noqa: F811
    PL_STALE_ATR_THRESH = backtester_core.PL_STALE_ATR_THRESH  # noqa: F811
    PL_SIGNAL_VARIANT = backtester_core.PL_SIGNAL_VARIANT  # noqa: F811
    PL_BUY_FILTER_MAX = backtester_core.PL_BUY_FILTER_MAX  # noqa: F811
    PL_SELL_FILTER_MIN = backtester_core.PL_SELL_FILTER_MIN  # noqa: F811
else:
    # auto mode: try Rust first, fall back to Numba
    try:
        import backtester_core

        batch_evaluate = backtester_core.batch_evaluate  # noqa: F811
        _using_rust = True
        logger.info("Using Rust native backend (backtester_core)")

        # Re-export PL_* constants from Rust module
        NUM_PL = backtester_core.NUM_PL  # noqa: F811
        PL_SL_MODE = backtester_core.PL_SL_MODE  # noqa: F811
        PL_SL_FIXED_PIPS = backtester_core.PL_SL_FIXED_PIPS  # noqa: F811
        PL_SL_ATR_MULT = backtester_core.PL_SL_ATR_MULT  # noqa: F811
        PL_TP_MODE = backtester_core.PL_TP_MODE  # noqa: F811
        PL_TP_RR_RATIO = backtester_core.PL_TP_RR_RATIO  # noqa: F811
        PL_TP_ATR_MULT = backtester_core.PL_TP_ATR_MULT  # noqa: F811
        PL_TP_FIXED_PIPS = backtester_core.PL_TP_FIXED_PIPS  # noqa: F811
        PL_HOURS_START = backtester_core.PL_HOURS_START  # noqa: F811
        PL_HOURS_END = backtester_core.PL_HOURS_END  # noqa: F811
        PL_DAYS_BITMASK = backtester_core.PL_DAYS_BITMASK  # noqa: F811
        PL_TRAILING_MODE = backtester_core.PL_TRAILING_MODE  # noqa: F811
        PL_TRAIL_ACTIVATE = backtester_core.PL_TRAIL_ACTIVATE  # noqa: F811
        PL_TRAIL_DISTANCE = backtester_core.PL_TRAIL_DISTANCE  # noqa: F811
        PL_TRAIL_ATR_MULT = backtester_core.PL_TRAIL_ATR_MULT  # noqa: F811
        PL_BREAKEVEN_ENABLED = backtester_core.PL_BREAKEVEN_ENABLED  # noqa: F811
        PL_BREAKEVEN_TRIGGER = backtester_core.PL_BREAKEVEN_TRIGGER  # noqa: F811
        PL_BREAKEVEN_OFFSET = backtester_core.PL_BREAKEVEN_OFFSET  # noqa: F811
        PL_PARTIAL_ENABLED = backtester_core.PL_PARTIAL_ENABLED  # noqa: F811
        PL_PARTIAL_PCT = backtester_core.PL_PARTIAL_PCT  # noqa: F811
        PL_PARTIAL_TRIGGER = backtester_core.PL_PARTIAL_TRIGGER  # noqa: F811
        PL_MAX_BARS = backtester_core.PL_MAX_BARS  # noqa: F811
        PL_STALE_ENABLED = backtester_core.PL_STALE_ENABLED  # noqa: F811
        PL_STALE_BARS = backtester_core.PL_STALE_BARS  # noqa: F811
        PL_STALE_ATR_THRESH = backtester_core.PL_STALE_ATR_THRESH  # noqa: F811
        PL_SIGNAL_VARIANT = backtester_core.PL_SIGNAL_VARIANT  # noqa: F811
        PL_BUY_FILTER_MAX = backtester_core.PL_BUY_FILTER_MAX  # noqa: F811
        PL_SELL_FILTER_MIN = backtester_core.PL_SELL_FILTER_MIN  # noqa: F811
    except ImportError:
        logger.info("Rust backend not available, using Numba JIT fallback")
        from backtester.core.jit_loop import batch_evaluate  # noqa: F401, F811
        from backtester.core.jit_loop import (  # noqa: F401, F811
            NUM_PL,
            PL_BREAKEVEN_ENABLED,
            PL_BREAKEVEN_OFFSET,
            PL_BREAKEVEN_TRIGGER,
            PL_BUY_FILTER_MAX,
            PL_DAYS_BITMASK,
            PL_HOURS_END,
            PL_HOURS_START,
            PL_MAX_BARS,
            PL_PARTIAL_ENABLED,
            PL_PARTIAL_PCT,
            PL_PARTIAL_TRIGGER,
            PL_SELL_FILTER_MIN,
            PL_SIGNAL_VARIANT,
            PL_SL_ATR_MULT,
            PL_SL_FIXED_PIPS,
            PL_SL_MODE,
            PL_STALE_ATR_THRESH,
            PL_STALE_BARS,
            PL_STALE_ENABLED,
            PL_TP_ATR_MULT,
            PL_TP_FIXED_PIPS,
            PL_TP_MODE,
            PL_TP_RR_RATIO,
            PL_TRAILING_MODE,
            PL_TRAIL_ACTIVATE,
            PL_TRAIL_ATR_MULT,
            PL_TRAIL_DISTANCE,
        )


def get_backend_name() -> str:
    """Return the name of the active backend."""
    return "rust" if _using_rust else "numba"
