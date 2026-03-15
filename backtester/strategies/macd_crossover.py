"""MACD Crossover strategy — trend-following via MACD/signal line crosses.

Buys when MACD line crosses above signal line, sells when MACD crosses below.
Uses ATR for volatility-aware SL/TP sizing. Variant encodes the MACD parameter
combination (fast, slow, signal period) for JIT filtering.
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

        for combo in MACD_COMBOS:
            fast_p, slow_p, sig_p = decode_macd_combo(combo)
            macd_line, signal_line, _ = macd(close, fast_p, slow_p, sig_p)

            warmup = slow_p + sig_p + 1
            idx = np.arange(warmup, n)
            if len(idx) == 0:
                continue

            m_cur = macd_line[idx]
            m_prev = macd_line[idx - 1]
            s_cur = signal_line[idx]
            s_prev = signal_line[idx - 1]
            a_val = atr_14[idx]

            valid = (np.isfinite(a_val) & (a_val > 0)
                     & np.isfinite(m_cur) & np.isfinite(m_prev)
                     & np.isfinite(s_cur) & np.isfinite(s_prev))

            buy = valid & (m_prev <= s_prev) & (m_cur > s_cur)
            sell = valid & (m_prev >= s_prev) & (m_cur < s_cur)

            for mask, direction in [(buy, Direction.BUY.value),
                                    (sell, Direction.SELL.value)]:
                bar_idx = idx[mask]
                if len(bar_idx) == 0:
                    continue
                # Use next-bar open for entry, fall back to close for last bar
                # (live trading needs last-bar signals)
                next_idx = np.minimum(bar_idx + 1, n - 1)
                entry_price = np.where(
                    bar_idx < (n - 1), open[next_idx], close[bar_idx]
                )
                parts_idx.append(bar_idx)
                parts_dir.append(np.full(len(bar_idx), direction, dtype=np.int64))
                parts_price.append(entry_price)
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
