"""Backtest engine orchestrator.

Connects Strategy → signal generation → encoding → JIT batch evaluation.

Usage:
    engine = BacktestEngine(strategy, open_, high, low, close, volume, spread)
    metrics = engine.evaluate_batch(param_matrix)  # (N, NUM_METRICS)
    result = engine.evaluate_single(params_dict)   # dict
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

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
from backtester.core.rust_loop import (
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
    PL_SIGNAL_VARIANT,
    PL_BUY_FILTER_MAX,
    PL_SELL_FILTER_MIN,
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
    # Strategy-specific filter params (mapped to generic JIT filter mechanism)
    # Any strategy can use these PL slots by defining params with matching names.
    "signal_variant": PL_SIGNAL_VARIANT,
    "buy_filter_max": PL_BUY_FILTER_MAX,
    "sell_filter_min": PL_SELL_FILTER_MIN,
    # RSI strategy maps its params to the generic filter:
    "rsi_period": PL_SIGNAL_VARIANT,       # variant = RSI period value
    "rsi_oversold": PL_BUY_FILTER_MAX,     # BUY if RSI <= oversold
    "rsi_overbought": PL_SELL_FILTER_MIN,  # SELL if RSI >= overbought
    # EMA crossover: composite variant = fast*1000 + slow
    "ema_combo": PL_SIGNAL_VARIANT,
    # MACD crossover: composite variant = fast*10000 + slow*100 + signal
    "macd_combo": PL_SIGNAL_VARIANT,
    # Bollinger reversion: composite variant = period*100 + int(std*10)
    "bb_combo": PL_SIGNAL_VARIANT,
    # Stochastic crossover: variant = k*100 + d, thresholds via filter
    "stoch_combo": PL_SIGNAL_VARIANT,
    "stoch_oversold": PL_BUY_FILTER_MAX,
    "stoch_overbought": PL_SELL_FILTER_MIN,
    # Donchian breakout: variant = period directly
    "donchian_period": PL_SIGNAL_VARIANT,
    # ADX trend: composite variant = period*100 + threshold
    "adx_combo": PL_SIGNAL_VARIANT,
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
        bar_hour: np.ndarray | None = None,
        bar_day_of_week: np.ndarray | None = None,
        # M1 sub-bar arrays (optional — identity fallback when None)
        m1_high: np.ndarray | None = None,
        m1_low: np.ndarray | None = None,
        m1_close: np.ndarray | None = None,
        m1_spread: np.ndarray | None = None,
        h1_to_m1_start: np.ndarray | None = None,
        h1_to_m1_end: np.ndarray | None = None,
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

        # Per-bar time arrays for time filtering (from timestamps)
        # Default: all zeros — passes default time filters (hours 0-23, all weekdays)
        n = len(high)
        if bar_hour is None or bar_day_of_week is None:
            logger.warning(
                "bar_hour/bar_day_of_week not provided — time filters will be "
                "ineffective (all bars treated as hour=0, day=Monday)"
            )
        self.bar_hour = bar_hour if bar_hour is not None else np.zeros(n, dtype=np.int64)
        self.bar_day_of_week = (
            bar_day_of_week if bar_day_of_week is not None
            else np.zeros(n, dtype=np.int64)
        )

        # Sub-bar arrays for M1 trade simulation
        if m1_high is not None and h1_to_m1_start is not None:
            # M1 data provided — use it for sub-bar simulation
            self.sub_high = m1_high
            self.sub_low = m1_low
            self.sub_close = m1_close
            self.sub_spread = m1_spread
            self.h1_to_sub_start = h1_to_m1_start
            self.h1_to_sub_end = h1_to_m1_end
        else:
            # Identity fallback: each H1 bar maps to itself
            self.sub_high = high
            self.sub_low = low
            self.sub_close = close
            self.sub_spread = spread
            self.h1_to_sub_start = np.arange(n, dtype=np.int64)
            self.h1_to_sub_end = np.arange(n, dtype=np.int64) + 1

        # Guard: BacktestEngine pre-computes signals once on the full dataset,
        # which is only valid for causal strategies.
        from backtester.strategies.base import SignalCausality
        if strategy.signal_causality() == SignalCausality.REQUIRES_TRAIN_FIT:
            raise NotImplementedError(
                f"Strategy '{strategy.name}' declares REQUIRES_TRAIN_FIT "
                f"signal causality. BacktestEngine pre-computes signals once "
                f"on the full dataset, which produces incorrect results for "
                f"non-causal indicators. Per-window signal generation is not "
                f"yet supported."
            )

        # Build encoding spec from strategy's param space
        self.param_space = strategy.param_space()
        self.encoding = build_encoding_spec(self.param_space)

        # Build param layout: maps PL_* constants to encoding column indices
        self.param_layout = self._build_param_layout()

        # Generate signals ONCE (the expensive step)
        sig_dict = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread, pip_value,
            self.bar_hour, self.bar_day_of_week,
        )
        self._unpack_signals(sig_dict, high, low, swing_lookback)
        logger.debug("Generated %d signals for %d bars", self.n_signals, len(high))

    def _build_param_layout(self) -> np.ndarray:
        """Build the param_layout array that maps PL_* to encoding column indices."""
        layout = np.zeros(NUM_PL, dtype=np.int64)

        # Filter PL slots default to -1 (disabled) when strategy doesn't define them.
        # Without this, unmapped filter slots point to column 0 which contains
        # an unrelated param value, causing incorrect signal filtering.
        from backtester.core.rust_loop import (
            PL_BUY_FILTER_MAX,
            PL_SELL_FILTER_MIN,
            PL_SIGNAL_VARIANT,
        )
        layout[PL_SIGNAL_VARIANT] = -1
        layout[PL_BUY_FILTER_MAX] = -1
        layout[PL_SELL_FILTER_MIN] = -1

        for param_name, pl_index in _PARAM_TO_PL.items():
            if param_name in self.encoding.name_to_index:
                layout[pl_index] = self.encoding.name_to_index[param_name]
            # If param not in space, layout stays at its default (0 or -1 for filters)

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

        # Unpack filter arrays (for strategy-specific signal filtering in JIT)
        if "filter_value" in sig_dict:
            self.sig_filter_value = sig_dict["filter_value"]
        else:
            self.sig_filter_value = np.full(self.n_signals, 0.0, dtype=np.float64)
        if "variant" in sig_dict:
            self.sig_variant = sig_dict["variant"]
        else:
            self.sig_variant = np.full(self.n_signals, -1, dtype=np.int64)

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

        # Pre-allocate PnL buffer on Python side (not inside Numba JIT).
        # Numba's NRT allocator pools large arrays and never returns pages
        # to the OS, causing segfaults on Windows after many batches.
        pnl_buffers = np.empty((n_trials, self.max_trades), dtype=np.float64)

        batch_evaluate(
            self.high, self.low, self.close, self.spread,
            self.pip_value, self.slippage_pips,
            self.sig_bar_index, self.sig_direction, self.sig_entry_price,
            self.sig_hour, self.sig_day, self.sig_atr_pips, self.sig_swing_sl,
            self.sig_filter_value, self.sig_variant,
            param_matrix, self.param_layout, exec_mode,
            metrics_out, self.max_trades, self.bars_per_year,
            self.commission_pips, self.max_spread_pips,
            # Sub-bar arrays for M1 trade simulation
            self.sub_high, self.sub_low, self.sub_close, self.sub_spread,
            self.h1_to_sub_start, self.h1_to_sub_end,
            pnl_buffers,
        )

        return metrics_out

    def evaluate_batch_windowed(
        self,
        param_matrix: np.ndarray,
        window_start: int,
        window_end: int,
        exec_mode: int = EXEC_BASIC,
    ) -> np.ndarray:
        """Evaluate N parameter sets on a sub-window [window_start, window_end).

        Filters pre-computed signals to the window range, slices data arrays,
        and rebases bar indices so n_bars = window_size (correct for Sharpe
        annualization). No engine recreation needed — uses the same signals.

        Args:
            param_matrix: (N, P) float64 matrix of encoded parameter values.
            window_start: First bar of the window (inclusive).
            window_end: Last bar of the window (exclusive).
            exec_mode: EXEC_BASIC or EXEC_FULL.

        Returns:
            (N, NUM_METRICS) array with metric values per trial.
        """
        window_start = max(0, window_start)
        window_end = min(self.n_bars, window_end)

        n_trials = param_matrix.shape[0]
        metrics_out = np.zeros((n_trials, NUM_METRICS), dtype=np.float64)

        # Filter signals to window range
        mask = (self.sig_bar_index >= window_start) & (self.sig_bar_index < window_end)

        if not mask.any():
            return metrics_out

        # Rebase signal bar indices to 0-based within window
        sig_bar_index = self.sig_bar_index[mask] - window_start
        sig_direction = self.sig_direction[mask]
        sig_entry_price = self.sig_entry_price[mask]
        sig_hour = self.sig_hour[mask]
        sig_day = self.sig_day[mask]
        sig_atr_pips = self.sig_atr_pips[mask]
        sig_swing_sl = self.sig_swing_sl[mask]
        sig_filter_value = self.sig_filter_value[mask]
        sig_variant = self.sig_variant[mask]

        # Slice data arrays to window (numpy views — no copy)
        high = self.high[window_start:window_end]
        low = self.low[window_start:window_end]
        close = self.close[window_start:window_end]
        spread = self.spread[window_start:window_end]

        # Sub-bar mapping: slice to window range. M1 data stays full —
        # the mapping indices still point to correct M1 positions.
        h1_to_sub_start = self.h1_to_sub_start[window_start:window_end]
        h1_to_sub_end = self.h1_to_sub_end[window_start:window_end]

        pnl_buffers = np.empty((n_trials, self.max_trades), dtype=np.float64)

        batch_evaluate(
            high, low, close, spread,
            self.pip_value, self.slippage_pips,
            sig_bar_index, sig_direction, sig_entry_price,
            sig_hour, sig_day, sig_atr_pips, sig_swing_sl,
            sig_filter_value, sig_variant,
            param_matrix, self.param_layout, exec_mode,
            metrics_out, self.max_trades, self.bars_per_year,
            self.commission_pips, self.max_spread_pips,
            self.sub_high, self.sub_low, self.sub_close, self.sub_spread,
            h1_to_sub_start, h1_to_sub_end,
            pnl_buffers,
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
