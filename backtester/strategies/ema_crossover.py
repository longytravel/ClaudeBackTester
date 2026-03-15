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
EMA_FAST_PERIODS = [3, 5, 8, 10, 13, 15, 20, 25, 30, 40, 50]
EMA_SLOW_PERIODS = [30, 50, 75, 100, 150, 200, 250, 300]

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
        # Fixed time params: EMA crossover is time-agnostic (trend signal).
        # Optimizing hours cherry-picks 1-2 hours and kills 90%+ of signals.
        params += [
            ParamDef("allowed_hours_start", [0], group="time"),
            ParamDef("allowed_hours_end", [23], group="time"),
            ParamDef("allowed_days", [[0, 1, 2, 3, 4]], group="time"),
        ]
        return ParamSpace(params)

    def optimization_stages(self) -> list[str]:
        # Skip time — EMA crossover has fixed time params (1 value each)
        stages = ["signal", "risk"]
        seen: set[str] = set()
        for mod in self.management_modules():
            if mod.group not in seen:
                stages.append(mod.group)
                seen.add(mod.group)
        return stages

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

        # Pre-compute all EMA arrays
        all_periods = sorted(set(EMA_FAST_PERIODS + EMA_SLOW_PERIODS))
        ema_arrays = {p: ema(close, p) for p in all_periods}
        atr_14 = atr(high, low, close, 14)

        parts_idx: list[np.ndarray] = []
        parts_dir: list[np.ndarray] = []
        parts_price: list[np.ndarray] = []
        parts_hour: list[np.ndarray] = []
        parts_day: list[np.ndarray] = []
        parts_atr: list[np.ndarray] = []
        parts_var: list[np.ndarray] = []

        for combo in EMA_COMBOS:
            fast_p, slow_p = decode_combo(combo)
            ema_fast = ema_arrays[fast_p]
            ema_slow = ema_arrays[slow_p]

            warmup = slow_p + 1
            idx = np.arange(warmup, n)
            if len(idx) == 0:
                continue

            f_cur = ema_fast[idx]
            f_prev = ema_fast[idx - 1]
            s_cur = ema_slow[idx]
            s_prev = ema_slow[idx - 1]
            a_val = atr_14[idx]

            valid = (np.isfinite(a_val) & (a_val > 0)
                     & np.isfinite(f_cur) & np.isfinite(f_prev)
                     & np.isfinite(s_cur) & np.isfinite(s_prev))

            # Golden cross: fast crosses above slow → BUY
            buy = valid & (f_prev <= s_prev) & (f_cur > s_cur)
            # Death cross: fast crosses below slow → SELL
            sell = valid & (f_prev >= s_prev) & (f_cur < s_cur)

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
