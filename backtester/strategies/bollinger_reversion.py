"""Bollinger Band Mean Reversion strategy.

Buys when price crosses below lower Bollinger Band (oversold),
sells when price crosses above upper Bollinger Band (overbought).
Uses ATR for volatility-aware SL/TP sizing.
"""

from __future__ import annotations

import math
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
from backtester.strategies.indicators import atr, bollinger_bands
from backtester.strategies.registry import register
from backtester.strategies.sl_tp import calc_sl_tp


# Bollinger Band parameter options
BB_PERIODS = [10, 15, 20, 25, 30]
BB_STD_DEVS = [1.5, 2.0, 2.5, 3.0]

# Encode as period * 100 + int(std * 10)  e.g. 2020 = period=20, std=2.0
BB_COMBOS = sorted(
    p * 100 + int(s * 10)
    for p in BB_PERIODS
    for s in BB_STD_DEVS
)


def decode_bb_combo(combo: int) -> tuple[int, float]:
    """Decode bb_combo into (period, std_dev)."""
    period = combo // 100
    std_x10 = combo % 100
    return period, std_x10 / 10.0


@register
class BollingerReversion(Strategy):
    """Mean reversion: buy at lower band, sell at upper band."""

    @property
    def name(self) -> str:
        return "bollinger_reversion"

    @property
    def version(self) -> str:
        return "1.0.0"

    def param_space(self) -> ParamSpace:
        params = [
            ParamDef("bb_combo", BB_COMBOS, group="signal"),
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

        bar_indices = []
        directions = []
        entry_prices = []
        hours_list = []
        days_list = []
        atr_pips_list = []
        variants = []

        for combo in BB_COMBOS:
            period, std_dev = decode_bb_combo(combo)
            upper, middle, lower = bollinger_bands(close, period, std_dev)

            warmup = period + 1
            for i in range(warmup, n - 1):
                atr_val = float(atr_14[i])
                if math.isnan(atr_val) or atr_val <= 0:
                    continue

                c_cur = close[i]
                c_prev = close[i - 1]
                lo_cur = float(lower[i])
                lo_prev = float(lower[i - 1])
                up_cur = float(upper[i])
                up_prev = float(upper[i - 1])

                if math.isnan(lo_cur) or math.isnan(up_cur):
                    continue
                if math.isnan(lo_prev) or math.isnan(up_prev):
                    continue

                atr_p = atr_val / pip_value

                # Price crosses below lower band → BUY (mean reversion)
                if c_prev >= lo_prev and c_cur < lo_cur:
                    bar_indices.append(i)
                    directions.append(Direction.BUY.value)
                    entry_prices.append(c_cur)
                    hours_list.append(int(bar_hour[i]))
                    days_list.append(int(bar_day_of_week[i]))
                    atr_pips_list.append(atr_p)
                    variants.append(combo)

                # Price crosses above upper band → SELL (mean reversion)
                if c_prev <= up_prev and c_cur > up_cur:
                    bar_indices.append(i)
                    directions.append(Direction.SELL.value)
                    entry_prices.append(c_cur)
                    hours_list.append(int(bar_hour[i]))
                    days_list.append(int(bar_day_of_week[i]))
                    atr_pips_list.append(atr_p)
                    variants.append(combo)

        if not bar_indices:
            return {
                "bar_index": np.array([], dtype=np.int64),
                "direction": np.array([], dtype=np.int64),
                "entry_price": np.array([], dtype=np.float64),
                "hour": np.array([], dtype=np.int64),
                "day_of_week": np.array([], dtype=np.int64),
                "atr_pips": np.array([], dtype=np.float64),
                "variant": np.array([], dtype=np.int64),
            }

        return {
            "bar_index": np.array(bar_indices, dtype=np.int64),
            "direction": np.array(directions, dtype=np.int64),
            "entry_price": np.array(entry_prices, dtype=np.float64),
            "hour": np.array(hours_list, dtype=np.int64),
            "day_of_week": np.array(days_list, dtype=np.int64),
            "atr_pips": np.array(atr_pips_list, dtype=np.float64),
            "variant": np.array(variants, dtype=np.int64),
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
