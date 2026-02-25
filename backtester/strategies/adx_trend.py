"""ADX Trend strategy — directional movement with trend strength filter.

Buys when +DI crosses above -DI with ADX above threshold (strong uptrend),
sells when -DI crosses above +DI with ADX above threshold (strong downtrend).
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
from backtester.strategies.indicators import adx, atr
from backtester.strategies.registry import register
from backtester.strategies.sl_tp import calc_sl_tp


# ADX parameter options
ADX_PERIODS = [10, 14, 20, 28]
ADX_THRESHOLDS = [20, 25, 30, 35, 40]

# Encode as period * 100 + threshold  (e.g. 1425 = period=14, threshold=25)
ADX_COMBOS = sorted(
    p * 100 + t
    for p in ADX_PERIODS
    for t in ADX_THRESHOLDS
)


def decode_adx_combo(combo: int) -> tuple[int, int]:
    """Decode adx_combo into (period, threshold)."""
    return combo // 100, combo % 100


@register
class ADXTrend(Strategy):
    """Trend strength: buy on +DI > -DI with strong ADX, sell on -DI > +DI."""

    @property
    def name(self) -> str:
        return "adx_trend"

    @property
    def version(self) -> str:
        return "1.0.0"

    def param_space(self) -> ParamSpace:
        params = [
            ParamDef("adx_combo", ADX_COMBOS, group="signal"),
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

        for combo in ADX_COMBOS:
            adx_period, adx_threshold = decode_adx_combo(combo)
            adx_line, plus_di, minus_di = adx(high, low, close, adx_period)

            warmup = adx_period * 3 + 1
            for i in range(warmup, n - 1):
                atr_val = float(atr_14[i])
                if math.isnan(atr_val) or atr_val <= 0:
                    continue

                adx_cur = float(adx_line[i])
                pdi_cur = float(plus_di[i])
                pdi_prev = float(plus_di[i - 1])
                mdi_cur = float(minus_di[i])
                mdi_prev = float(minus_di[i - 1])

                if math.isnan(adx_cur) or math.isnan(pdi_cur) or math.isnan(mdi_cur):
                    continue
                if math.isnan(pdi_prev) or math.isnan(mdi_prev):
                    continue

                # ADX must be above threshold (strong trend)
                if adx_cur < adx_threshold:
                    continue

                atr_p = atr_val / pip_value

                # +DI crosses above -DI → BUY (bullish trend)
                if pdi_prev <= mdi_prev and pdi_cur > mdi_cur:
                    bar_indices.append(i)
                    directions.append(Direction.BUY.value)
                    entry_prices.append(close[i])
                    hours_list.append(int(bar_hour[i]))
                    days_list.append(int(bar_day_of_week[i]))
                    atr_pips_list.append(atr_p)
                    variants.append(combo)

                # -DI crosses above +DI → SELL (bearish trend)
                if mdi_prev <= pdi_prev and mdi_cur > pdi_cur:
                    bar_indices.append(i)
                    directions.append(Direction.SELL.value)
                    entry_prices.append(close[i])
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
