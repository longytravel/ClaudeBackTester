"""Bollinger Band Mean Reversion strategy.

Buys when price crosses below lower Bollinger Band (oversold),
sells when price crosses above upper Bollinger Band (overbought).
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

        parts_idx: list[np.ndarray] = []
        parts_dir: list[np.ndarray] = []
        parts_price: list[np.ndarray] = []
        parts_hour: list[np.ndarray] = []
        parts_day: list[np.ndarray] = []
        parts_atr: list[np.ndarray] = []
        parts_var: list[np.ndarray] = []

        for combo in BB_COMBOS:
            period, std_dev = decode_bb_combo(combo)
            upper, middle, lower = bollinger_bands(close, period, std_dev)

            warmup = period + 1
            idx = np.arange(warmup, n)
            if len(idx) == 0:
                continue

            c_cur = close[idx]
            c_prev = close[idx - 1]
            lo_cur = lower[idx]
            lo_prev = lower[idx - 1]
            up_cur = upper[idx]
            up_prev = upper[idx - 1]
            a_val = atr_14[idx]

            valid = (np.isfinite(a_val) & (a_val > 0)
                     & np.isfinite(lo_cur) & np.isfinite(lo_prev)
                     & np.isfinite(up_cur) & np.isfinite(up_prev))

            # Price crosses below lower band → BUY (mean reversion)
            buy = valid & (c_prev >= lo_prev) & (c_cur < lo_cur)
            # Price crosses above upper band → SELL (mean reversion)
            sell = valid & (c_prev <= up_prev) & (c_cur > up_cur)

            for mask, direction in [(buy, Direction.BUY.value),
                                    (sell, Direction.SELL.value)]:
                bar_idx = idx[mask]
                if len(bar_idx) == 0:
                    continue
                parts_idx.append(bar_idx)
                parts_dir.append(np.full(len(bar_idx), direction, dtype=np.int64))
                parts_price.append(close[bar_idx])
                parts_hour.append(bar_hour[bar_idx])
                parts_day.append(bar_day_of_week[bar_idx])
                parts_atr.append(atr_14[bar_idx] / pip_value)
                parts_var.append(np.full(len(bar_idx), combo, dtype=np.int64))

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
