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
        """Generate RSI crossover signals for all RSI periods and thresholds.

        For each RSI period, generates a separate signal at each threshold
        crossing (20, 25, 30, 35 for oversold; 65, 70, 75, 80 for overbought).
        The JIT loop uses exact-match filtering: only signals whose threshold
        matches the trial's rsi_oversold/rsi_overbought are accepted.
        """
        n = len(close)
        if bar_hour is None:
            bar_hour = np.zeros(n, dtype=np.int64)
        if bar_day_of_week is None:
            bar_day_of_week = np.zeros(n, dtype=np.int64)

        rsi_arrays = {p: rsi(close, p) for p in RSI_PERIODS}
        atr_14 = atr(high, low, close, 14)

        bar_indices = []
        directions = []
        entry_prices = []
        hours_list = []
        days_list = []
        atr_pips_list = []
        filter_values = []   # threshold value that was crossed (for exact match)
        variants = []         # RSI period value (7, 9, 14, 21) matching rsi_period param

        for i in range(max(RSI_PERIODS) + 1, n - 1):
            atr_val = atr_14[i]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            atr_p = atr_val / pip_value

            for rp in RSI_PERIODS:
                r_cur = rsi_arrays[rp][i]
                r_prev = rsi_arrays[rp][i - 1]
                if np.isnan(r_cur) or np.isnan(r_prev):
                    continue

                # Buy signals: generate at EACH oversold threshold crossing
                for thresh in OVERSOLD_THRESHOLDS:
                    if r_cur < thresh and r_prev >= thresh:
                        bar_indices.append(i)
                        directions.append(Direction.BUY.value)
                        entry_prices.append(close[i])
                        hours_list.append(int(bar_hour[i]))
                        days_list.append(int(bar_day_of_week[i]))
                        atr_pips_list.append(atr_p)
                        filter_values.append(float(thresh))
                        variants.append(rp)

                # Sell signals: generate at EACH overbought threshold crossing
                for thresh in OVERBOUGHT_THRESHOLDS:
                    if r_cur > thresh and r_prev <= thresh:
                        bar_indices.append(i)
                        directions.append(Direction.SELL.value)
                        entry_prices.append(close[i])
                        hours_list.append(int(bar_hour[i]))
                        days_list.append(int(bar_day_of_week[i]))
                        atr_pips_list.append(atr_p)
                        filter_values.append(float(thresh))
                        variants.append(rp)

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
