"""Backtest engine orchestrator.

Connects Strategy → signal generation → encoding → JIT batch evaluation.

Usage:
    engine = BacktestEngine(strategy, open_, high, low, close, volume, spread)
    metrics = engine.evaluate_batch(param_matrix)  # (N, NUM_METRICS)
    result = engine.evaluate_single(params_dict)   # dict
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backtester.core.dtypes import (
    DEFAULT_BARS_PER_YEAR,
    DEFAULT_COMMISSION_PIPS,
    DEFAULT_MAX_SPREAD_PIPS,
    DEFAULT_PIP_VALUE,
    DEFAULT_SLIPPAGE_PIPS,
    EXEC_BASIC,
    EXEC_FULL,
    NUM_METRICS,
)
from backtester.core.encoding import (
    EncodingSpec,
    build_encoding_spec,
    decode_params,
    encode_params,
    indices_to_values,
)
from backtester.core.jit_loop import (
    NUM_PL,
    PL_BREAKEVEN_ENABLED,
    PL_BREAKEVEN_OFFSET,
    PL_BREAKEVEN_TRIGGER,
    PL_DAYS_BITMASK,
    PL_HOURS_END,
    PL_HOURS_START,
    PL_MAX_BARS,
    PL_PARTIAL_ENABLED,
    PL_PARTIAL_PCT,
    PL_PARTIAL_TRIGGER,
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
    batch_evaluate,
)
from backtester.core.metrics import compute_metrics as compute_metrics_python
from backtester.strategies.base import Strategy
from backtester.strategies.sl_tp import _find_recent_swing


# Standard param name → PL_* layout index mapping
_PARAM_TO_PL: dict[str, int] = {
    "sl_mode": PL_SL_MODE,
    "sl_fixed_pips": PL_SL_FIXED_PIPS,
    "sl_atr_mult": PL_SL_ATR_MULT,
    "tp_mode": PL_TP_MODE,
    "tp_rr_ratio": PL_TP_RR_RATIO,
    "tp_atr_mult": PL_TP_ATR_MULT,
    "tp_fixed_pips": PL_TP_FIXED_PIPS,
    "allowed_hours_start": PL_HOURS_START,
    "allowed_hours_end": PL_HOURS_END,
    "allowed_days": PL_DAYS_BITMASK,
    "trailing_mode": PL_TRAILING_MODE,
    "trail_activate_pips": PL_TRAIL_ACTIVATE,
    "trail_distance_pips": PL_TRAIL_DISTANCE,
    "trail_atr_mult": PL_TRAIL_ATR_MULT,
    "breakeven_enabled": PL_BREAKEVEN_ENABLED,
    "breakeven_trigger_pips": PL_BREAKEVEN_TRIGGER,
    "breakeven_offset_pips": PL_BREAKEVEN_OFFSET,
    "partial_close_enabled": PL_PARTIAL_ENABLED,
    "partial_close_pct": PL_PARTIAL_PCT,
    "partial_close_trigger_pips": PL_PARTIAL_TRIGGER,
    "max_bars": PL_MAX_BARS,
    "stale_exit_enabled": PL_STALE_ENABLED,
    "stale_exit_bars": PL_STALE_BARS,
    "stale_exit_atr_threshold": PL_STALE_ATR_THRESH,
}


class BacktestEngine:
    """Orchestrates strategy signal generation and batch evaluation.

    Generates signals ONCE, then evaluates batches of parameter sets
    against those signals via the JIT-compiled loop.
    """

    def __init__(
        self,
        strategy: Strategy,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        spread: np.ndarray,
        pip_value: float = DEFAULT_PIP_VALUE,
        slippage_pips: float = DEFAULT_SLIPPAGE_PIPS,
        max_trades_per_trial: int = 50000,
        swing_lookback: int = 50,
        bars_per_year: float = DEFAULT_BARS_PER_YEAR,
        commission_pips: float = DEFAULT_COMMISSION_PIPS,
        max_spread_pips: float = DEFAULT_MAX_SPREAD_PIPS,
    ):
        self.strategy = strategy
        self.high = high
        self.low = low
        self.close = close
        self.spread = spread
        self.pip_value = pip_value
        self.slippage_pips = slippage_pips
        self.max_trades = max_trades_per_trial
        self.n_bars = len(high)
        self.bars_per_year = bars_per_year
        self.commission_pips = commission_pips
        self.max_spread_pips = max_spread_pips

        # Build encoding spec from strategy's param space
        self.param_space = strategy.param_space()
        self.encoding = build_encoding_spec(self.param_space)

        # Build param layout: maps PL_* constants to encoding column indices
        self.param_layout = self._build_param_layout()

        # Generate signals ONCE (the expensive step)
        sig_dict = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread,
        )
        self._unpack_signals(sig_dict, high, low, swing_lookback)

    def _build_param_layout(self) -> np.ndarray:
        """Build the param_layout array that maps PL_* to encoding column indices."""
        layout = np.zeros(NUM_PL, dtype=np.int64)

        for param_name, pl_index in _PARAM_TO_PL.items():
            if param_name in self.encoding.name_to_index:
                layout[pl_index] = self.encoding.name_to_index[param_name]
            # If param not in space, layout stays 0 — param will read from column 0
            # which is fine since the default value will be used

        return layout

    def _unpack_signals(
        self,
        sig_dict: dict[str, np.ndarray],
        high: np.ndarray,
        low: np.ndarray,
        swing_lookback: int,
    ) -> None:
        """Unpack vectorized signals into arrays for JIT."""
        self.sig_bar_index = sig_dict["bar_index"]
        self.sig_direction = sig_dict["direction"]
        self.sig_entry_price = sig_dict["entry_price"]
        self.sig_hour = sig_dict["hour"]
        self.sig_day = sig_dict["day_of_week"]
        self.sig_atr_pips = sig_dict["atr_pips"]
        self.n_signals = len(self.sig_bar_index)

        # Pre-compute swing SL prices for each signal (for swing SL mode)
        self.sig_swing_sl = np.full(self.n_signals, np.nan, dtype=np.float64)
        for i in range(self.n_signals):
            bar_idx = int(self.sig_bar_index[i])
            direction = int(self.sig_direction[i])
            from backtester.core.dtypes import DIR_BUY
            if direction == DIR_BUY:
                swing = _find_recent_swing(low, bar_idx, swing_lookback, find_high=False)
            else:
                swing = _find_recent_swing(high, bar_idx, swing_lookback, find_high=True)
            if swing is not None:
                self.sig_swing_sl[i] = swing

        # Store attr keys for telemetry
        self.attr_keys = [k[5:] for k in sig_dict if k.startswith("attr_")]
        self.sig_attrs = {k: sig_dict[f"attr_{k}"] for k in self.attr_keys}

    def evaluate_batch(
        self,
        param_matrix: np.ndarray,
        exec_mode: int = EXEC_BASIC,
    ) -> np.ndarray:
        """Evaluate N parameter sets. Returns (N, NUM_METRICS) float64 array.

        Args:
            param_matrix: (N, P) float64 matrix of encoded parameter values.
            exec_mode: EXEC_BASIC (SL/TP only) or EXEC_FULL (all management).

        Returns:
            (N, NUM_METRICS) array with metric values per trial.
        """
        n_trials = param_matrix.shape[0]
        metrics_out = np.zeros((n_trials, NUM_METRICS), dtype=np.float64)

        if self.n_signals == 0:
            return metrics_out

        batch_evaluate(
            self.high, self.low, self.close, self.spread,
            self.pip_value, self.slippage_pips,
            self.sig_bar_index, self.sig_direction, self.sig_entry_price,
            self.sig_hour, self.sig_day, self.sig_atr_pips, self.sig_swing_sl,
            param_matrix, self.param_layout, exec_mode,
            metrics_out, self.max_trades, self.bars_per_year,
            self.commission_pips, self.max_spread_pips,
        )

        return metrics_out

    def evaluate_batch_from_indices(
        self,
        index_matrix: np.ndarray,
        exec_mode: int = EXEC_BASIC,
    ) -> np.ndarray:
        """Evaluate from index-space matrix (optimizer interface).

        Converts indices to values, then calls evaluate_batch.
        """
        value_matrix = indices_to_values(self.encoding, index_matrix)
        return self.evaluate_batch(value_matrix, exec_mode)

    def evaluate_single(
        self,
        params: dict[str, Any],
        exec_mode: int = EXEC_BASIC,
    ) -> dict[str, float]:
        """Evaluate a single parameter set. Returns metric dict.

        Convenience method for single-trial evaluation.
        """
        row = encode_params(self.encoding, params)
        matrix = row.reshape(1, -1)
        metrics = self.evaluate_batch(matrix, exec_mode)

        return {
            "trades": float(metrics[0, 0]),
            "win_rate": float(metrics[0, 1]),
            "profit_factor": float(metrics[0, 2]),
            "sharpe": float(metrics[0, 3]),
            "sortino": float(metrics[0, 4]),
            "max_dd_pct": float(metrics[0, 5]),
            "return_pct": float(metrics[0, 6]),
            "r_squared": float(metrics[0, 7]),
            "ulcer": float(metrics[0, 8]),
            "quality_score": float(metrics[0, 9]),
        }
