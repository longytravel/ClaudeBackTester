"""Hidden Smash Day reversal strategy — based on Larry Williams.

Source: MQL5 article 21391 (Larry Williams Market Secrets Part 13)
Book: "Long Term Secrets to Short Term Trading" by Larry Williams

A Hidden Smash Day is a bar that closes in the EXTREME portion of its own
range despite closing in the OPPOSITE direction vs the previous bar. This
signals internal exhaustion — the market moved one way but finished weakly.

Entry is NOT on the smash bar itself. It requires CONFIRMATION: the next
bar must close beyond the smash bar's extreme (above high for buy, below
low for sell).

Parameters to optimize:
- close_pct_threshold: what % of range counts as "extreme" (default 25)
- require_opposite_body: require close vs open to oppose direction
- atr_period: for ATR-based stops and filtering
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
from backtester.strategies.indicators import atr
from backtester.strategies.registry import register
from backtester.strategies.sl_tp import calc_sl_tp


# Close percentage thresholds — how deep into the range the close must be
CLOSE_PCT_THRESHOLDS = [15, 20, 25, 30, 35, 40]

# ATR periods for stop calculation and signal quality filtering
ATR_PERIODS = [10, 14, 20]

# Require opposite body filter (0 = no filter, 1 = require opposite body)
BODY_FILTER = [0, 1]

# Encode: atr_period * 100 + body_filter * 10 + threshold_index
# This gives us a unique variant per combination
def _encode_variant(atr_period: int, body_filter: int, pct_idx: int) -> int:
    return atr_period * 100 + body_filter * 10 + pct_idx


def _build_variants() -> list[int]:
    variants = []
    for ap in ATR_PERIODS:
        for bf in BODY_FILTER:
            for pi, _ in enumerate(CLOSE_PCT_THRESHOLDS):
                variants.append(_encode_variant(ap, bf, pi))
    return variants


VARIANTS = _build_variants()


@register
class HiddenSmashDay(Strategy):
    """Larry Williams Hidden Smash Day reversal pattern.

    Detects bars that close in an extreme portion of their range despite
    closing in the opposite direction vs the prior bar, then enters on
    confirmation (next bar closes beyond the smash bar's extreme).
    """

    @property
    def name(self) -> str:
        return "hidden_smash_day"

    @property
    def version(self) -> str:
        return "1.0.0"

    def param_space(self) -> ParamSpace:
        params = [
            ParamDef("hsd_variant", VARIANTS, group="signal"),
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

        # Precompute ATR for all periods
        atr_arrays = {p: atr(high, low, close, p) for p in ATR_PERIODS}

        # Need at least 2 bars for pattern + 1 for confirmation = 3 bars
        # Plus ATR warmup
        warmup = max(ATR_PERIODS) + 2

        # Precompute bar-level properties used in pattern detection
        bar_range = high - low  # (n,)
        close_vs_prev = np.zeros(n, dtype=np.float64)
        close_vs_prev[1:] = close[1:] - close[:-1]

        # Position of close within bar range (0 = at low, 100 = at high)
        # close_pct_from_low[i] = ((close[i] - low[i]) / range[i]) * 100
        with np.errstate(divide="ignore", invalid="ignore"):
            close_pct_from_low = np.where(
                bar_range > 0,
                ((close - low) / bar_range) * 100.0,
                50.0,  # zero-range bar — neutral
            )
            close_pct_from_high = np.where(
                bar_range > 0,
                ((high - close) / bar_range) * 100.0,
                50.0,
            )

        # Body direction: close > open = bullish body, close < open = bearish body
        bullish_body = close > open
        bearish_body = close < open

        parts_idx: list[np.ndarray] = []
        parts_dir: list[np.ndarray] = []
        parts_price: list[np.ndarray] = []
        parts_hour: list[np.ndarray] = []
        parts_day: list[np.ndarray] = []
        parts_atr: list[np.ndarray] = []
        parts_var: list[np.ndarray] = []

        # Confirmation bars: idx where we CHECK if bar (idx-1) confirmed smash at (idx-2)
        # Signal fires at idx (the confirmation bar), entry at close of idx
        confirm_idx = np.arange(warmup, n)
        if len(confirm_idx) == 0:
            return self._empty_signals()

        # Smash bar is at (confirm_idx - 1), pattern bar is at (confirm_idx - 2)
        smash_bars = confirm_idx - 1

        for atr_period in ATR_PERIODS:
            atr_vals = atr_arrays[atr_period]
            valid_atr = np.isfinite(atr_vals[smash_bars]) & (atr_vals[smash_bars] > 0)

            for body_filter in BODY_FILTER:
                for pct_idx, pct_thresh in enumerate(CLOSE_PCT_THRESHOLDS):
                    variant = _encode_variant(atr_period, body_filter, pct_idx)

                    # --- BULLISH Hidden Smash Day ---
                    # Smash bar: closes HIGHER than prev, but in LOWER pct% of its range
                    bull_smash = (
                        valid_atr
                        & (close_vs_prev[smash_bars] > 0)  # close > prev close
                        & (bar_range[smash_bars] > 0)  # non-zero range
                        & (close_pct_from_low[smash_bars] <= pct_thresh)  # close in lower portion
                    )
                    if body_filter == 1:
                        # Require bearish body (close < open) despite up close
                        bull_smash = bull_smash & bearish_body[smash_bars]

                    # Confirmation: confirm bar closes ABOVE smash bar's high
                    bull_confirmed = bull_smash & (close[confirm_idx] > high[smash_bars])

                    bar_idx = confirm_idx[bull_confirmed]
                    # Filter out signals on the last bar (no next-bar open available)
                    bar_idx = bar_idx[bar_idx < (n - 1)]
                    if len(bar_idx) > 0:
                        parts_idx.append(bar_idx)
                        parts_dir.append(np.full(len(bar_idx), Direction.BUY.value, dtype=np.int64))
                        parts_price.append(open[bar_idx + 1])
                        parts_hour.append(bar_hour[bar_idx])
                        parts_day.append(bar_day_of_week[bar_idx])
                        parts_atr.append(atr_vals[bar_idx] / pip_value)
                        parts_var.append(np.full(len(bar_idx), variant, dtype=np.int64))

                    # --- BEARISH Hidden Smash Day ---
                    # Smash bar: closes LOWER than prev, but in UPPER pct% of its range
                    bear_smash = (
                        valid_atr
                        & (close_vs_prev[smash_bars] < 0)  # close < prev close
                        & (bar_range[smash_bars] > 0)
                        & (close_pct_from_high[smash_bars] <= pct_thresh)  # close in upper portion
                    )
                    if body_filter == 1:
                        # Require bullish body (close > open) despite down close
                        bear_smash = bear_smash & bullish_body[smash_bars]

                    # Confirmation: confirm bar closes BELOW smash bar's low
                    bear_confirmed = bear_smash & (close[confirm_idx] < low[smash_bars])

                    bar_idx = confirm_idx[bear_confirmed]
                    # Filter out signals on the last bar (no next-bar open available)
                    bar_idx = bar_idx[bar_idx < (n - 1)]
                    if len(bar_idx) > 0:
                        parts_idx.append(bar_idx)
                        parts_dir.append(np.full(len(bar_idx), Direction.SELL.value, dtype=np.int64))
                        parts_price.append(open[bar_idx + 1])
                        parts_hour.append(bar_hour[bar_idx])
                        parts_day.append(bar_day_of_week[bar_idx])
                        parts_atr.append(atr_vals[bar_idx] / pip_value)
                        parts_var.append(np.full(len(bar_idx), variant, dtype=np.int64))

        if not parts_idx:
            return self._empty_signals()

        return {
            "bar_index": np.concatenate(parts_idx),
            "direction": np.concatenate(parts_dir),
            "entry_price": np.concatenate(parts_price),
            "hour": np.concatenate(parts_hour),
            "day_of_week": np.concatenate(parts_day),
            "atr_pips": np.concatenate(parts_atr),
            "filter_value": np.concatenate([np.zeros(len(p), dtype=np.float64) for p in parts_idx]),
            "variant": np.concatenate(parts_var),
        }

    def _empty_signals(self) -> dict[str, np.ndarray]:
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
