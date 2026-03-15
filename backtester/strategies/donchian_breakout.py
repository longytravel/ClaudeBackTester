"""Donchian Channel Breakout strategy — classic channel breakout.

Buys when price breaks above the N-period high (upper channel),
sells when price breaks below the N-period low (lower channel).
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
from backtester.strategies.indicators import atr, donchian
from backtester.strategies.registry import register
from backtester.strategies.sl_tp import calc_sl_tp


# Donchian channel period options
DONCHIAN_PERIODS = [10, 15, 20, 30, 40, 50, 60]


@register
class DonchianBreakout(Strategy):
    """Breakout: buy above channel high, sell below channel low."""

    @property
    def name(self) -> str:
        return "donchian_breakout"

    @property
    def version(self) -> str:
        return "1.0.0"

    def param_space(self) -> ParamSpace:
        params = [
            ParamDef("donchian_period", DONCHIAN_PERIODS, group="signal"),
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
        parts_var: list[np.ndarray] = []

        for period in DONCHIAN_PERIODS:
            upper, middle, lower = donchian(high, low, period)

            warmup = period + 1
            idx = np.arange(warmup, n)
            if len(idx) == 0:
                continue

            c_cur = close[idx]
            c_prev = close[idx - 1]
            up_prev = upper[idx - 1]
            lo_prev = lower[idx - 1]
            a_val = atr_14[idx]

            valid = (np.isfinite(a_val) & (a_val > 0)
                     & np.isfinite(up_prev) & np.isfinite(lo_prev))

            # Close breaks above previous upper channel → BUY
            buy = valid & (c_cur > up_prev) & (c_prev <= up_prev)
            # Close breaks below previous lower channel → SELL
            sell = valid & (c_cur < lo_prev) & (c_prev >= lo_prev)

            for mask, direction in [(buy, Direction.BUY.value),
                                    (sell, Direction.SELL.value)]:
                bar_idx = idx[mask]
                # Filter out signals on the last bar (no next-bar open available)
                bar_idx = bar_idx[bar_idx < (n - 1)]
                if len(bar_idx) == 0:
                    continue
                parts_idx.append(bar_idx)
                parts_dir.append(np.full(len(bar_idx), direction, dtype=np.int64))
                parts_price.append(open[bar_idx + 1])
                parts_hour.append(bar_hour[bar_idx])
                parts_day.append(bar_day_of_week[bar_idx])
                parts_atr.append(atr_14[bar_idx] / pip_value)
                parts_var.append(np.full(len(bar_idx), period, dtype=np.int64))

        if not parts_idx:
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
            "bar_index": np.concatenate(parts_idx),
            "direction": np.concatenate(parts_dir),
            "entry_price": np.concatenate(parts_price),
            "hour": np.concatenate(parts_hour),
            "day_of_week": np.concatenate(parts_day),
            "atr_pips": np.concatenate(parts_atr),
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
