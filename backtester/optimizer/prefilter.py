"""Pre-batch and post-batch filtering for optimization.

Pre-filter: reject invalid parameter combinations before engine evaluation.
Post-filter: reject results that fail hard metric gates after evaluation.
"""

from __future__ import annotations

import numpy as np

from backtester.core.dtypes import (
    M_MAX_DD_PCT,
    M_R_SQUARED,
    M_TRADES,
    SL_ATR_BASED,
    SL_FIXED_PIPS,
    SL_SWING,
    TP_ATR_BASED,
    TP_FIXED_PIPS,
    TP_RR_RATIO,
    TRAIL_ATR_CHANDELIER,
    TRAIL_FIXED_PIP,
    TRAIL_OFF,
)
from backtester.core.encoding import EncodingSpec


def prefilter_invalid_combos(
    index_matrix: np.ndarray,
    spec: EncodingSpec,
) -> np.ndarray:
    """Return boolean mask: True = valid combination, False = invalid.

    Checks for logically impossible parameter combinations:
    - breakeven_offset >= breakeven_trigger (BE can never lock)
    - trailing OFF but trail params nonzero (harmless but wasteful)
    - SL mode mismatch with active SL params
    """
    n = index_matrix.shape[0]
    valid = np.ones(n, dtype=np.bool_)

    # Check breakeven: offset must be less than trigger
    if ("breakeven_enabled" in spec.name_to_index
            and "breakeven_trigger_pips" in spec.name_to_index
            and "breakeven_offset_pips" in spec.name_to_index):
        be_col = spec.column("breakeven_enabled")
        trigger_col = spec.column("breakeven_trigger_pips")
        offset_col = spec.column("breakeven_offset_pips")

        for i in range(n):
            be_idx = int(index_matrix[i, be_col.index])
            if be_idx < len(be_col.values) and be_col.values[be_idx]:
                # BE is enabled — check offset < trigger
                trig_idx = int(index_matrix[i, trigger_col.index])
                off_idx = int(index_matrix[i, offset_col.index])
                if trig_idx < len(trigger_col.values) and off_idx < len(offset_col.values):
                    trigger_val = trigger_col.values[trig_idx]
                    offset_val = offset_col.values[off_idx]
                    if offset_val >= trigger_val:
                        valid[i] = False

    return valid


def postfilter_results(
    metrics: np.ndarray,
    min_trades: int = 20,
    max_dd_pct: float = 30.0,
    min_r_squared: float = 0.5,
) -> np.ndarray:
    """Return boolean mask: True = passes hard gates, False = rejected.

    Hard rejection criteria:
    - Fewer than min_trades trades
    - Max drawdown exceeds max_dd_pct
    - R-squared below min_r_squared
    """
    n = metrics.shape[0]
    valid = np.ones(n, dtype=np.bool_)

    valid &= metrics[:, M_TRADES] >= min_trades
    valid &= metrics[:, M_MAX_DD_PCT] <= max_dd_pct
    valid &= metrics[:, M_R_SQUARED] >= min_r_squared

    return valid
