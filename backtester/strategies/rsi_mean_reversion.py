"""RSI Mean Reversion strategy — first real strategy implementation.

Buys when RSI dips below oversold threshold, sells when RSI rises above overbought.
Uses ATR for volatility-aware SL/TP sizing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backtester.strategies.base import (
    Direction,
    ParamDef,
    ParamSpace,
    Signal,
    SLTPResult,
    Strategy,
    management_params,
    risk_params,
    time_params,
)
from backtester.strategies.indicators import atr, rsi
from backtester.strategies.registry import register
from backtester.strategies.sl_tp import calc_sl_tp


# RSI period values — used as variant identifiers in signal filtering
RSI_PERIODS = [5, 7, 9, 11, 14, 18, 21, 28]

# Threshold values — signals generated at each crossing, filtered by exact match
OVERSOLD_THRESHOLDS = [15, 20, 25, 30, 35, 40]
OVERBOUGHT_THRESHOLDS = [60, 65, 70, 75, 80, 85]


@register
class RSIMeanReversion(Strategy):
    """Buy oversold RSI, sell overbought RSI."""

    @property
    def name(self) -> str:
        return "rsi_mean_reversion"

    @property
    def version(self) -> str:
        return "1.0.0"

    def param_space(self) -> ParamSpace:
        params = [
            # Signal params
            # rsi_period maps to PL_SIGNAL_VARIANT in engine (variant filtering)
            # rsi_oversold maps to PL_BUY_FILTER_MAX (BUY accepted when threshold == this)
            # rsi_overbought maps to PL_SELL_FILTER_MIN (SELL accepted when threshold == this)
            ParamDef("rsi_period", RSI_PERIODS, group="signal"),
            ParamDef("rsi_oversold", OVERSOLD_THRESHOLDS, group="signal"),
            ParamDef("rsi_overbought", OVERBOUGHT_THRESHOLDS, group="signal"),
        ]
        params += risk_params()
        params += management_params()
        params += time_params()
        return ParamSpace(params)

    def optimization_stages(self) -> list[str]:
        return ["signal", "time", "risk", "management"]

    def generate_signals(
        self,
        open: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        spread: np.ndarray,
    ) -> list[Signal]:
        # Not used — we override the vectorized path below
        return []

    def generate_signals_vectorized(
        self,
        open: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        spread: np.ndarray,
        pip_value: float = 0.0001,
        bar_hour: np.ndarray | None = None,
        bar_day_of_week: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        n = len(close)
        if bar_hour is None:
            bar_hour = np.zeros(n, dtype=np.int64)
        if bar_day_of_week is None:
            bar_day_of_week = np.zeros(n, dtype=np.int64)

        rsi_arrays = {p: rsi(close, p) for p in RSI_PERIODS}
        atr_14 = atr(high, low, close, 14)

        warmup = max(RSI_PERIODS) + 1
        idx = np.arange(warmup, n)

        parts_idx: list[np.ndarray] = []
        parts_dir: list[np.ndarray] = []
        parts_price: list[np.ndarray] = []
        parts_hour: list[np.ndarray] = []
        parts_day: list[np.ndarray] = []
        parts_atr: list[np.ndarray] = []
        parts_filt: list[np.ndarray] = []
        parts_var: list[np.ndarray] = []

        if len(idx) == 0:
            return {
                "bar_index": np.array([], dtype=np.int64),
                "direction": np.array([], dtype=np.int64),
                "entry_price": np.array([], dtype=np.float64),
                "hour": np.array([], dtype=np.int64),
                "day_of_week": np.array([], dtype=np.int64),
                "atr_pips": np.array([], dtype=np.float64),
                "filter_value": np.array([], dtype=np.float64),
                "variant": np.array([], dtype=np.int64),
            }

        a_val = atr_14[idx]
        valid_atr = np.isfinite(a_val) & (a_val > 0)

        for rp in RSI_PERIODS:
            r_cur = rsi_arrays[rp][idx]
            r_prev = rsi_arrays[rp][idx - 1]
            valid = valid_atr & np.isfinite(r_cur) & np.isfinite(r_prev)

            # Buy signals: RSI crosses below each oversold threshold
            for thresh in OVERSOLD_THRESHOLDS:
                buy = valid & (r_cur < thresh) & (r_prev >= thresh)
                bar_idx = idx[buy]
                if len(bar_idx) == 0:
                    continue
                parts_idx.append(bar_idx)
                parts_dir.append(np.full(len(bar_idx), Direction.BUY.value, dtype=np.int64))
                parts_price.append(close[bar_idx])
                parts_hour.append(bar_hour[bar_idx])
                parts_day.append(bar_day_of_week[bar_idx])
                parts_atr.append(atr_14[bar_idx] / pip_value)
                parts_filt.append(np.full(len(bar_idx), float(thresh)))
                parts_var.append(np.full(len(bar_idx), rp, dtype=np.int64))

            # Sell signals: RSI crosses above each overbought threshold
            for thresh in OVERBOUGHT_THRESHOLDS:
                sell = valid & (r_cur > thresh) & (r_prev <= thresh)
                bar_idx = idx[sell]
                if len(bar_idx) == 0:
                    continue
                parts_idx.append(bar_idx)
                parts_dir.append(np.full(len(bar_idx), Direction.SELL.value, dtype=np.int64))
                parts_price.append(close[bar_idx])
                parts_hour.append(bar_hour[bar_idx])
                parts_day.append(bar_day_of_week[bar_idx])
                parts_atr.append(atr_14[bar_idx] / pip_value)
                parts_filt.append(np.full(len(bar_idx), float(thresh)))
                parts_var.append(np.full(len(bar_idx), rp, dtype=np.int64))

        if not parts_idx:
            return {
                "bar_index": np.array([], dtype=np.int64),
                "direction": np.array([], dtype=np.int64),
                "entry_price": np.array([], dtype=np.float64),
                "hour": np.array([], dtype=np.int64),
                "day_of_week": np.array([], dtype=np.int64),
                "atr_pips": np.array([], dtype=np.float64),
                "filter_value": np.array([], dtype=np.float64),
                "variant": np.array([], dtype=np.int64),
            }

        return {
            "bar_index": np.concatenate(parts_idx),
            "direction": np.concatenate(parts_dir),
            "entry_price": np.concatenate(parts_price),
            "hour": np.concatenate(parts_hour),
            "day_of_week": np.concatenate(parts_day),
            "atr_pips": np.concatenate(parts_atr),
            "filter_value": np.concatenate(parts_filt),
            "variant": np.concatenate(parts_var),
        }

    def filter_signals(
        self,
        signals: list[Signal],
        params: dict[str, Any],
    ) -> list[Signal]:
        # Filtering handled by JIT: variant exact-match + filter_value exact-match
        return signals

    def calc_sl_tp(
        self,
        signal: Signal,
        params: dict[str, Any],
        high: np.ndarray,
        low: np.ndarray,
    ) -> SLTPResult:
        return calc_sl_tp(signal, params, high, low)
