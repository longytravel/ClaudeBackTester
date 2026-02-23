"""Always-buy test strategy — fires a BUY every bar. For demo/testing only."""

from __future__ import annotations

from typing import Any

import numpy as np

from backtester.strategies.base import (
    Direction,
    ParamDef,
    ParamSpace,
    SLTPResult,
    Signal,
    Strategy,
)
from backtester.strategies.registry import register


class AlwaysBuyStrategy(Strategy):

    @property
    def name(self) -> str:
        return "always_buy"

    @property
    def version(self) -> str:
        return "1.0.0"

    def param_space(self) -> ParamSpace:
        return ParamSpace([
            ParamDef("sl_pips", [30], group="risk"),
            ParamDef("tp_pips", [60], group="risk"),
        ])

    def generate_signals(
        self,
        open: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        spread: np.ndarray,
    ) -> list[Signal]:
        signals = []
        for i in range(1, len(close)):
            signals.append(Signal(
                bar_index=i,
                direction=Direction.BUY,
                entry_price=close[i],
                hour=0,
                day_of_week=0,
                atr_pips=30.0,
            ))
        return signals

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
        sl_pips = params.get("sl_pips", 30)
        tp_pips = params.get("tp_pips", 60)
        pip = 0.0001
        return SLTPResult(
            sl_price=signal.entry_price - sl_pips * pip,
            tp_price=signal.entry_price + tp_pips * pip,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
        )


register(AlwaysBuyStrategy)
