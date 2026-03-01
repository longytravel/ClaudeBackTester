"""Stochastic Crossover strategy — momentum-based %K/%D crossover.

Buys when %K crosses above %D in oversold zone, sells when %K crosses
below %D in overbought zone. Uses ATR for volatility-aware SL/TP sizing.
Variant encodes K/D period combo, filter_value encodes zone threshold.
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
from backtester.strategies.indicators import atr, stochastic
from backtester.strategies.registry import register
from backtester.strategies.sl_tp import calc_sl_tp


# Stochastic parameter options
STOCH_K_PERIODS = [5, 7, 9, 14, 21]
STOCH_D_PERIODS = [3, 5, 7]
OVERSOLD_THRESHOLDS = [15, 20, 25, 30]
OVERBOUGHT_THRESHOLDS = [70, 75, 80, 85]

# Encode K/D combo as k * 100 + d  (e.g. 1403 = K=14, D=3)
STOCH_COMBOS = sorted(
    k * 100 + d
    for k in STOCH_K_PERIODS
    for d in STOCH_D_PERIODS
)


def decode_stoch_combo(combo: int) -> tuple[int, int]:
    """Decode stoch_combo into (k_period, d_period)."""
    return combo // 100, combo % 100


@register
class StochasticCrossover(Strategy):
    """Momentum: buy on bullish %K/%D crossover in oversold, sell in overbought."""

    @property
    def name(self) -> str:
        return "stochastic_crossover"

    @property
    def version(self) -> str:
        return "1.0.0"

    def param_space(self) -> ParamSpace:
        params = [
            ParamDef("stoch_combo", STOCH_COMBOS, group="signal"),
            ParamDef("stoch_oversold", OVERSOLD_THRESHOLDS, group="signal"),
            ParamDef("stoch_overbought", OVERBOUGHT_THRESHOLDS, group="signal"),
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

        atr_14 = atr(high, low, close, 14)

        parts_idx: list[np.ndarray] = []
        parts_dir: list[np.ndarray] = []
        parts_price: list[np.ndarray] = []
        parts_hour: list[np.ndarray] = []
        parts_day: list[np.ndarray] = []
        parts_atr: list[np.ndarray] = []
        parts_filt: list[np.ndarray] = []
        parts_var: list[np.ndarray] = []

        for combo in STOCH_COMBOS:
            k_period, d_period = decode_stoch_combo(combo)
            k_line, d_line = stochastic(high, low, close, k_period, d_period)

            warmup = k_period + d_period + 1
            idx = np.arange(warmup, n - 1)
            if len(idx) == 0:
                continue

            k_cur = k_line[idx]
            k_prev = k_line[idx - 1]
            d_cur = d_line[idx]
            d_prev = d_line[idx - 1]
            a_val = atr_14[idx]

            valid = (np.isfinite(a_val) & (a_val > 0)
                     & np.isfinite(k_cur) & np.isfinite(k_prev)
                     & np.isfinite(d_cur) & np.isfinite(d_prev))

            cross_up = valid & (k_prev <= d_prev) & (k_cur > d_cur)
            cross_down = valid & (k_prev >= d_prev) & (k_cur < d_cur)

            # Bullish crossover + oversold threshold filter
            for thresh in OVERSOLD_THRESHOLDS:
                buy = cross_up & (k_cur < thresh)
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
                parts_var.append(np.full(len(bar_idx), combo, dtype=np.int64))

            # Bearish crossover + overbought threshold filter
            for thresh in OVERBOUGHT_THRESHOLDS:
                sell = cross_down & (k_cur > thresh)
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
                parts_var.append(np.full(len(bar_idx), combo, dtype=np.int64))

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
        return signals

    def calc_sl_tp(
        self,
        signal: Signal,
        params: dict[str, Any],
        high: np.ndarray,
        low: np.ndarray,
    ) -> SLTPResult:
        return calc_sl_tp(signal, params, high, low)
