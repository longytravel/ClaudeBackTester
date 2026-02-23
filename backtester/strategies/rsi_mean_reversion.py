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
from backtester.strategies.indicators import atr, rsi, sma
from backtester.strategies.registry import register
from backtester.strategies.sl_tp import calc_sl_tp


# RSI period values — used as variant identifiers in signal filtering
RSI_PERIODS = [7, 9, 14, 21]


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
            # rsi_oversold maps to PL_BUY_FILTER_MAX (BUY accepted when RSI <= this)
            # rsi_overbought maps to PL_SELL_FILTER_MIN (SELL accepted when RSI >= this)
            ParamDef("rsi_period", RSI_PERIODS, group="signal"),
            ParamDef("rsi_oversold", [20, 25, 30, 35], group="signal"),
            ParamDef("rsi_overbought", [65, 70, 75, 80], group="signal"),
            ParamDef("atr_period", [10, 14, 20], group="signal"),
            ParamDef("sma_filter_period", [0, 50, 100, 200], group="signal"),
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
    ) -> dict[str, np.ndarray]:
        """Generate RSI crossover signals for all RSI periods.

        Each RSI period generates its own signals tagged with a variant index.
        The JIT loop filters signals by variant (matching trial's rsi_period)
        and by threshold (matching trial's rsi_oversold/overbought).
        """
        n = len(close)

        # Precompute indicators at all parameter periods
        atr_periods = [10, 14, 20]

        rsi_arrays = {p: rsi(close, p) for p in RSI_PERIODS}
        atr_arrays = {p: atr(high, low, close, p) for p in atr_periods}

        # Use the default ATR period (14) for signal atr_pips
        atr_14 = atr_arrays[14]

        # Use widest thresholds to capture ALL potential crossover signals.
        # The JIT's signal filter (buy_filter_max / sell_filter_min) handles
        # per-trial threshold filtering.
        min_oversold = 35       # widest oversold threshold
        max_overbought = 65     # widest overbought threshold

        bar_indices = []
        directions = []
        entry_prices = []
        hours_list = []
        days_list = []
        atr_pips_list = []
        filter_values = []   # RSI value at signal's period
        variants = []         # RSI period value (7, 9, 14, 21) matching rsi_period param

        for i in range(max(RSI_PERIODS) + 1, n - 1):
            atr_val = atr_14[i]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            atr_p = atr_val / pip_value

            # Generate signals for EACH RSI period (no break — one per period)
            for period_idx, rp in enumerate(RSI_PERIODS):
                r_cur = rsi_arrays[rp][i]
                r_prev = rsi_arrays[rp][i - 1]
                if np.isnan(r_cur) or np.isnan(r_prev):
                    continue

                # Buy signal: RSI crosses below oversold (widest threshold)
                if r_cur < min_oversold and r_prev >= min_oversold:
                    bar_indices.append(i)
                    directions.append(Direction.BUY.value)
                    entry_prices.append(close[i])
                    hours_list.append(i % 24)  # placeholder
                    days_list.append((i // 24) % 5)  # placeholder
                    atr_pips_list.append(atr_p)
                    filter_values.append(r_cur)  # actual RSI value
                    variants.append(rp)           # actual period value (matches rsi_period param)

                # Sell signal: RSI crosses above overbought (widest threshold)
                if r_cur > max_overbought and r_prev <= max_overbought:
                    bar_indices.append(i)
                    directions.append(Direction.SELL.value)
                    entry_prices.append(close[i])
                    hours_list.append(i % 24)
                    days_list.append((i // 24) % 5)
                    atr_pips_list.append(atr_p)
                    filter_values.append(r_cur)
                    variants.append(rp)           # actual period value (matches rsi_period param)

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
        # Filtering is handled by the JIT via filter_value/variant
        return signals

    def calc_sl_tp(
        self,
        signal: Signal,
        params: dict[str, Any],
        high: np.ndarray,
        low: np.ndarray,
    ) -> SLTPResult:
        return calc_sl_tp(signal, params, high, low)
