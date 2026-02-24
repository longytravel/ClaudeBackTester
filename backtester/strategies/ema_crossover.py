"""EMA Crossover strategy — trend-following via fast/slow EMA crosses.

Buys when fast EMA crosses above slow EMA, sells when fast crosses below.
Uses ATR for volatility-aware SL/TP sizing. Generates far more signals than
RSI mean reversion, making it suitable for walk-forward validation.
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
from backtester.strategies.indicators import atr, ema
from backtester.strategies.registry import register
from backtester.strategies.sl_tp import calc_sl_tp


# EMA period options
EMA_FAST_PERIODS = [5, 8, 10, 13, 20, 25, 30]
EMA_SLOW_PERIODS = [30, 50, 75, 100, 150, 200]

# Build all valid (fast < slow) combos encoded as fast * 1000 + slow
EMA_COMBOS = sorted(
    f * 1000 + s
    for f in EMA_FAST_PERIODS
    for s in EMA_SLOW_PERIODS
    if f < s
)


def decode_combo(combo: int) -> tuple[int, int]:
    """Decode ema_combo into (fast_period, slow_period)."""
    return combo // 1000, combo % 1000


@register
class EMACrossover(Strategy):
    """Trend-following: buy on golden cross, sell on death cross."""

    @property
    def name(self) -> str:
        return "ema_crossover"

    @property
    def version(self) -> str:
        return "1.0.0"

    def param_space(self) -> ParamSpace:
        params = [
            # ema_combo encodes fast*1000+slow; maps to PL_SIGNAL_VARIANT
            ParamDef("ema_combo", EMA_COMBOS, group="signal"),
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
        """Generate EMA crossover signals for all fast/slow period combos.

        For each (fast, slow) pair, generates a BUY signal when fast EMA
        crosses above slow EMA, and a SELL signal when fast crosses below.
        The JIT loop uses variant exact-match to filter by ema_combo.
        """
        n = len(close)
        if bar_hour is None:
            bar_hour = np.zeros(n, dtype=np.int64)
        if bar_day_of_week is None:
            bar_day_of_week = np.zeros(n, dtype=np.int64)

        # Pre-compute all EMA arrays
        all_periods = sorted(set(EMA_FAST_PERIODS + EMA_SLOW_PERIODS))
        ema_arrays = {p: ema(close, p) for p in all_periods}
        atr_14 = atr(high, low, close, 14)

        bar_indices = []
        directions = []
        entry_prices = []
        hours_list = []
        days_list = []
        atr_pips_list = []
        variants = []

        for combo in EMA_COMBOS:
            fast_p, slow_p = decode_combo(combo)
            ema_fast = ema_arrays[fast_p]
            ema_slow = ema_arrays[slow_p]

            # Start after the slowest indicator is valid
            warmup = slow_p + 1
            for i in range(warmup, n - 1):
                atr_val = atr_14[i]
                if np.isnan(atr_val) or atr_val <= 0:
                    continue

                fast_cur = ema_fast[i]
                fast_prev = ema_fast[i - 1]
                slow_cur = ema_slow[i]
                slow_prev = ema_slow[i - 1]

                if np.isnan(fast_cur) or np.isnan(fast_prev):
                    continue
                if np.isnan(slow_cur) or np.isnan(slow_prev):
                    continue

                atr_p = atr_val / pip_value

                # Golden cross: fast crosses above slow → BUY
                if fast_prev <= slow_prev and fast_cur > slow_cur:
                    bar_indices.append(i)
                    directions.append(Direction.BUY.value)
                    entry_prices.append(close[i])
                    hours_list.append(int(bar_hour[i]))
                    days_list.append(int(bar_day_of_week[i]))
                    atr_pips_list.append(atr_p)
                    variants.append(combo)

                # Death cross: fast crosses below slow → SELL
                if fast_prev >= slow_prev and fast_cur < slow_cur:
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
