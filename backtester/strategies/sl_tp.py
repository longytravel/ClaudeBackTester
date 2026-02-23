"""Stop-Loss and Take-Profit calculation modes (REQ-S14 through REQ-S17).

Supports: fixed pips, ATR-based, swing-based SL; RR ratio, ATR-based, fixed pips TP.
Enforces minimum TP >= SL constraint (REQ-S17).
"""

from __future__ import annotations

import numpy as np

from backtester.strategies.base import Direction, SLTPResult, Signal


def _find_recent_swing(
    data: np.ndarray,
    bar_index: int,
    lookback: int,
    find_high: bool,
) -> float | None:
    """Find most recent swing high/low before bar_index."""
    start = max(0, bar_index - lookback)
    window = data[start:bar_index]
    if len(window) == 0:
        return None
    return float(np.max(window)) if find_high else float(np.min(window))


def calc_sl_tp(
    signal: Signal,
    params: dict,
    high: np.ndarray,
    low: np.ndarray,
    pip_value: float = 0.0001,
    swing_lookback: int = 50,
) -> SLTPResult:
    """Calculate SL and TP prices for a signal given parameters.

    Args:
        signal: The entry signal with direction, price, atr_pips.
        params: Parameter dict with sl_mode, tp_mode, and related values.
        high: Full high price array (for swing-based SL).
        low: Full low price array (for swing-based SL).
        pip_value: Value of one pip (0.0001 for most pairs, 0.01 for JPY).
        swing_lookback: Number of bars to look back for swing points.
    """
    entry = signal.entry_price
    atr_price = signal.atr_pips * pip_value  # Convert ATR pips to price units
    is_buy = signal.direction == Direction.BUY

    # --- Stop Loss ---
    sl_mode = params.get("sl_mode", "fixed_pips")

    if sl_mode == "atr_based":
        sl_mult = params.get("sl_atr_mult", 1.5)
        sl_distance = atr_price * sl_mult
    elif sl_mode == "swing":
        if is_buy:
            swing = _find_recent_swing(low, signal.bar_index, swing_lookback, find_high=False)
        else:
            swing = _find_recent_swing(high, signal.bar_index, swing_lookback, find_high=True)

        if swing is not None:
            sl_distance = abs(entry - swing)
            # Minimum SL: 5 pips
            sl_distance = max(sl_distance, 5 * pip_value)
        else:
            # Fallback to ATR-based if no swing found
            sl_distance = atr_price * params.get("sl_atr_mult", 1.5)
    else:  # fixed_pips
        sl_pips = params.get("sl_fixed_pips", 30)
        sl_distance = sl_pips * pip_value

    sl_pips = sl_distance / pip_value

    if is_buy:
        sl_price = entry - sl_distance
    else:
        sl_price = entry + sl_distance

    # --- Take Profit ---
    tp_mode = params.get("tp_mode", "rr_ratio")

    if tp_mode == "atr_based":
        tp_mult = params.get("tp_atr_mult", 2.0)
        tp_distance = atr_price * tp_mult
    elif tp_mode == "fixed_pips":
        tp_pips_val = params.get("tp_fixed_pips", 50)
        tp_distance = tp_pips_val * pip_value
    else:  # rr_ratio
        rr = params.get("tp_rr_ratio", 2.0)
        tp_distance = sl_distance * rr

    # Enforce TP >= SL (REQ-S17)
    tp_distance = max(tp_distance, sl_distance)
    tp_pips = tp_distance / pip_value

    if is_buy:
        tp_price = entry + tp_distance
    else:
        tp_price = entry - tp_distance

    return SLTPResult(
        sl_price=sl_price,
        tp_price=tp_price,
        sl_pips=sl_pips,
        tp_pips=tp_pips,
    )
