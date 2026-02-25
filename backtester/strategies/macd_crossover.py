"""MACD Crossover strategy — trend-following via MACD/signal line crosses.

Buys when MACD line crosses above signal line, sells when MACD crosses below.
Uses ATR for volatility-aware SL/TP sizing. Variant encodes the MACD parameter
combination (fast, slow, signal period) for JIT filtering.
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
from backtester.strategies.indicators import atr, macd
from backtester.strategies.registry import register
from backtester.strategies.sl_tp import calc_sl_tp


# MACD parameter options
MACD_FAST_PERIODS = [5, 8, 12, 16]
MACD_SLOW_PERIODS = [21, 26, 30, 40, 50]
MACD_SIGNAL_PERIODS = [5, 7, 9, 12]

# Encode as fast * 10000 + slow * 100 + signal
MACD_COMBOS = sorted(
    f * 10000 + s * 100 + sig
    for f in MACD_FAST_PERIODS
    for s in MACD_SLOW_PERIODS
    for sig in MACD_SIGNAL_PERIODS
    if f < s
)


def decode_macd_combo(combo: int) -> tuple[int, int, int]:
    """Decode macd_combo into (fast, slow, signal_period)."""
    fast = combo // 10000
    remainder = combo % 10000
    slow = remainder // 100
    signal = remainder % 100
    return fast, slow, signal


@register
class MACDCrossover(Strategy):
    """Trend-following: buy when MACD crosses above signal, sell when below."""

    @property
    def name(self) -> str:
        return "macd_crossover"

    @property
    def version(self) -> str:
        return "1.0.0"

    def param_space(self) -> ParamSpace:
        params = [
            ParamDef("macd_combo", MACD_COMBOS, group="signal"),
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

        for combo in MACD_COMBOS:
            fast_p, slow_p, sig_p = decode_macd_combo(combo)
            macd_line, signal_line, _ = macd(close, fast_p, slow_p, sig_p)

            warmup = slow_p + sig_p + 1
            for i in range(warmup, n - 1):
                atr_val = float(atr_14[i])
                if math.isnan(atr_val) or atr_val <= 0:
                    continue

                m_cur = float(macd_line[i])
                m_prev = float(macd_line[i - 1])
                s_cur = float(signal_line[i])
                s_prev = float(signal_line[i - 1])

                if math.isnan(m_cur) or math.isnan(m_prev):
                    continue
                if math.isnan(s_cur) or math.isnan(s_prev):
                    continue

                atr_p = atr_val / pip_value

                # MACD crosses above signal → BUY
                if m_prev <= s_prev and m_cur > s_cur:
                    bar_indices.append(i)
                    directions.append(Direction.BUY.value)
                    entry_prices.append(close[i])
                    hours_list.append(int(bar_hour[i]))
                    days_list.append(int(bar_day_of_week[i]))
                    atr_pips_list.append(atr_p)
                    variants.append(combo)

                # MACD crosses below signal → SELL
                if m_prev >= s_prev and m_cur < s_cur:
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
