"""ADX Trend strategy — directional movement with trend strength filter.

Buys when +DI crosses above -DI with ADX above threshold (strong uptrend),
sells when -DI crosses above +DI with ADX above threshold (strong downtrend).
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

        parts_idx: list[np.ndarray] = []
        parts_dir: list[np.ndarray] = []
        parts_price: list[np.ndarray] = []
        parts_hour: list[np.ndarray] = []
        parts_day: list[np.ndarray] = []
        parts_atr: list[np.ndarray] = []
        parts_var: list[np.ndarray] = []

        for combo in ADX_COMBOS:
            adx_period, adx_threshold = decode_adx_combo(combo)
            adx_line, plus_di, minus_di = adx(high, low, close, adx_period)

            warmup = adx_period * 3 + 1
            idx = np.arange(warmup, n)
            if len(idx) == 0:
                continue

            adx_cur = adx_line[idx]
            pdi_cur = plus_di[idx]
            pdi_prev = plus_di[idx - 1]
            mdi_cur = minus_di[idx]
            mdi_prev = minus_di[idx - 1]
            a_val = atr_14[idx]

            valid = (np.isfinite(a_val) & (a_val > 0)
                     & np.isfinite(adx_cur) & np.isfinite(pdi_cur)
                     & np.isfinite(pdi_prev) & np.isfinite(mdi_cur)
                     & np.isfinite(mdi_prev))

            # ADX must be above threshold (strong trend)
            strong = valid & (adx_cur >= adx_threshold)

            # +DI crosses above -DI → BUY
            buy = strong & (pdi_prev <= mdi_prev) & (pdi_cur > mdi_cur)
            # -DI crosses above +DI → SELL
            sell = strong & (mdi_prev <= pdi_prev) & (mdi_cur > pdi_cur)

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
