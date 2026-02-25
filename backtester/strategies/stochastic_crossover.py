"""Stochastic Crossover strategy — momentum-based %K/%D crossover.

Buys when %K crosses above %D in oversold zone, sells when %K crosses
below %D in overbought zone. Uses ATR for volatility-aware SL/TP sizing.
Variant encodes K/D period combo, filter_value encodes zone threshold.
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

        bar_indices = []
        directions = []
        entry_prices = []
        hours_list = []
        days_list = []
        atr_pips_list = []
        filter_values = []
        variants = []

        for combo in STOCH_COMBOS:
            k_period, d_period = decode_stoch_combo(combo)
            k_line, d_line = stochastic(high, low, close, k_period, d_period)

            warmup = k_period + d_period + 1
            for i in range(warmup, n - 1):
                atr_val = float(atr_14[i])
                if math.isnan(atr_val) or atr_val <= 0:
                    continue

                k_cur = float(k_line[i])
                k_prev = float(k_line[i - 1])
                d_cur = float(d_line[i])
                d_prev = float(d_line[i - 1])

                if math.isnan(k_cur) or math.isnan(k_prev):
                    continue
                if math.isnan(d_cur) or math.isnan(d_prev):
                    continue

                atr_p = atr_val / pip_value

                # Bullish crossover: %K crosses above %D
                if k_prev <= d_prev and k_cur > d_cur:
                    # Generate BUY at each oversold threshold where K is below
                    for thresh in OVERSOLD_THRESHOLDS:
                        if k_cur < thresh:
                            bar_indices.append(i)
                            directions.append(Direction.BUY.value)
                            entry_prices.append(close[i])
                            hours_list.append(int(bar_hour[i]))
                            days_list.append(int(bar_day_of_week[i]))
                            atr_pips_list.append(atr_p)
                            filter_values.append(float(thresh))
                            variants.append(combo)

                # Bearish crossover: %K crosses below %D
                if k_prev >= d_prev and k_cur < d_cur:
                    # Generate SELL at each overbought threshold where K is above
                    for thresh in OVERBOUGHT_THRESHOLDS:
                        if k_cur > thresh:
                            bar_indices.append(i)
                            directions.append(Direction.SELL.value)
                            entry_prices.append(close[i])
                            hours_list.append(int(bar_hour[i]))
                            days_list.append(int(bar_day_of_week[i]))
                            atr_pips_list.append(atr_p)
                            filter_values.append(float(thresh))
                            variants.append(combo)

        if not bar_indices:
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
            "bar_index": np.array(bar_indices, dtype=np.int64),
            "direction": np.array(directions, dtype=np.int64),
            "entry_price": np.array(entry_prices, dtype=np.float64),
            "hour": np.array(hours_list, dtype=np.int64),
            "day_of_week": np.array(days_list, dtype=np.int64),
            "atr_pips": np.array(atr_pips_list, dtype=np.float64),
            "filter_value": np.array(filter_values, dtype=np.float64),
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
