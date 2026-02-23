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
            ParamDef("rsi_period", [7, 9, 14, 21], group="signal"),
            ParamDef("rsi_oversold", [20, 25, 30, 35], group="signal"),
            ParamDef("rsi_overbought", [65, 70, 75, 80], group="signal"),
            ParamDef("atr_period", [10, 14, 20], group="signal"),
            ParamDef("sma_filter_period", [0, 50, 100, 200], group="signal"),
        ]
        params += risk_params()
        params += management_params()
        params += time_params()
        return ParamSpace(params)

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
    ) -> dict[str, np.ndarray]:
        """Generate RSI crossover signals using all RSI period variants.

        We precompute RSI at all periods so the filter step is cheap.
        A signal fires when RSI crosses below oversold (buy) or above
        overbought (sell) at ANY of the RSI periods. The filter_signals
        step then selects which period/threshold to keep per trial.
        """
        n = len(close)

        # Precompute indicators at all parameter periods
        rsi_periods = [7, 9, 14, 21]
        atr_periods = [10, 14, 20]
        sma_periods = [50, 100, 200]

        rsi_arrays = {p: rsi(close, p) for p in rsi_periods}
        atr_arrays = {p: atr(high, low, close, p) for p in atr_periods}
        sma_arrays = {p: sma(close, p) for p in sma_periods}

        # Use the default ATR period (14) for signal atr_pips
        atr_14 = atr_arrays[14]
        pip_value = 0.0001  # standard forex

        # Detect oversold/overbought at the widest thresholds to capture all signals
        # Filter step will narrow based on actual params
        min_oversold = 35  # widest oversold threshold
        max_overbought = 65  # widest overbought threshold

        bar_indices = []
        directions = []
        entry_prices = []
        hours = []
        days = []
        atr_pips_list = []
        rsi_values = {p: [] for p in rsi_periods}

        # Generate timestamps array for hour/day extraction
        # We don't have timestamps, so we'll use bar index patterns
        # For now, set hour/day from bar index (will be overridden by real data)

        for i in range(max(rsi_periods) + 1, n - 1):
            atr_val = atr_14[i]
            if np.isnan(atr_val) or atr_val <= 0:
                continue

            atr_p = atr_val / pip_value

            # Check ALL RSI periods for crossover signals
            for rp in rsi_periods:
                r_cur = rsi_arrays[rp][i]
                r_prev = rsi_arrays[rp][i - 1]
                if np.isnan(r_cur) or np.isnan(r_prev):
                    continue

                # Buy signal: RSI crosses below oversold
                if r_cur < min_oversold and r_prev >= min_oversold:
                    bar_indices.append(i)
                    directions.append(Direction.BUY.value)
                    entry_prices.append(close[i])
                    hours.append(i % 24)  # placeholder
                    days.append((i // 24) % 5)  # placeholder
                    atr_pips_list.append(atr_p)
                    for p2 in rsi_periods:
                        rv = rsi_arrays[p2][i]
                        rsi_values[p2].append(rv if not np.isnan(rv) else 50.0)
                    break  # one signal per bar

                # Sell signal: RSI crosses above overbought
                if r_cur > max_overbought and r_prev <= max_overbought:
                    bar_indices.append(i)
                    directions.append(Direction.SELL.value)
                    entry_prices.append(close[i])
                    hours.append(i % 24)
                    days.append((i // 24) % 5)
                    atr_pips_list.append(atr_p)
                    for p2 in rsi_periods:
                        rv = rsi_arrays[p2][i]
                        rsi_values[p2].append(rv if not np.isnan(rv) else 50.0)
                    break

        if not bar_indices:
            return {
                "bar_index": np.array([], dtype=np.int64),
                "direction": np.array([], dtype=np.int64),
                "entry_price": np.array([], dtype=np.float64),
                "hour": np.array([], dtype=np.int64),
                "day_of_week": np.array([], dtype=np.int64),
                "atr_pips": np.array([], dtype=np.float64),
            }

        result = {
            "bar_index": np.array(bar_indices, dtype=np.int64),
            "direction": np.array(directions, dtype=np.int64),
            "entry_price": np.array(entry_prices, dtype=np.float64),
            "hour": np.array(hours, dtype=np.int64),
            "day_of_week": np.array(days, dtype=np.int64),
            "atr_pips": np.array(atr_pips_list, dtype=np.float64),
        }

        # Store RSI values as attrs for filtering
        for p in rsi_periods:
            result[f"attr_rsi_{p}"] = np.array(rsi_values[p], dtype=np.float64)

        return result

    def filter_signals(
        self,
        signals: list[Signal],
        params: dict[str, Any],
    ) -> list[Signal]:
        # Default implementation — not used in hot path
        return signals

    def calc_sl_tp(
        self,
        signal: Signal,
        params: dict[str, Any],
        high: np.ndarray,
        low: np.ndarray,
    ) -> SLTPResult:
        return calc_sl_tp(signal, params, high, low)
