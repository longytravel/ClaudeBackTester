"""Verification Test strategy — multi-signal high-frequency generator.

Designed to generate LOTS of trades for backtest-to-live parity verification.
Supports 7 fast signal modes (EMA, RSI, MACD, Bollinger, Stochastic, ADX,
Donchian) — each tuned for maximum signal frequency. Full risk and management
param ranges exercise trailing, breakeven, partial close, etc.

Signal quality is irrelevant — trade volume and system coverage matter.
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
)
from backtester.strategies.indicators import atr, ema
from backtester.strategies.registry import register
from backtester.strategies.sl_tp import calc_sl_tp

# ---------------------------------------------------------------------------
# Signal mode constants — each mode gets a unique variant integer
# ---------------------------------------------------------------------------
MODE_EMA = 1         # EMA(3) vs EMA(8) crossover
MODE_RSI = 2         # RSI(5), oversold<30 / overbought>70
MODE_MACD = 3        # MACD(5,13,5) signal-line crossover
MODE_BOLLINGER = 4   # Bollinger(10, 1.5σ) band touch
MODE_STOCHASTIC = 5  # Stochastic(5,3), %K/%D cross in zones
MODE_ADX = 6         # ADX(10) + DI cross, threshold=15
MODE_DONCHIAN = 7    # Donchian(10) channel breakout

ALL_MODES = [MODE_EMA, MODE_RSI, MODE_MACD, MODE_BOLLINGER,
             MODE_STOCHASTIC, MODE_ADX, MODE_DONCHIAN]

MODE_NAMES = {
    MODE_EMA: "EMA(3/8)",
    MODE_RSI: "RSI(5)",
    MODE_MACD: "MACD(5/13/5)",
    MODE_BOLLINGER: "BB(10/1.5σ)",
    MODE_STOCHASTIC: "Stoch(5/3)",
    MODE_ADX: "ADX(10/15)",
    MODE_DONCHIAN: "Donch(10)",
}


# ---------------------------------------------------------------------------
# Indicator helpers (minimal, self-contained)
# ---------------------------------------------------------------------------

def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """RSI via exponential moving average of gains/losses."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[period] = gain[1:period + 1].mean()
    avg_loss[period] = loss[1:period + 1].mean()
    for i in range(period + 1, len(close)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[:period + 1] = np.nan
    return rsi


def _macd(close: np.ndarray, fast: int, slow: int, signal: int
          ) -> tuple[np.ndarray, np.ndarray]:
    """MACD line and signal line."""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    # ema() can't handle NaN-prefixed input — compute signal EMA manually
    n = len(close)
    signal_line = np.full(n, np.nan)
    # First finite MACD value is at index (slow - 1)
    start = slow - 1
    if start + signal <= n:
        alpha = 2.0 / (signal + 1)
        signal_line[start + signal - 1] = np.mean(macd_line[start:start + signal])
        for i in range(start + signal, n):
            signal_line[i] = alpha * macd_line[i] + (1 - alpha) * signal_line[i - 1]
    return macd_line, signal_line


def _bollinger(close: np.ndarray, period: int, std_mult: float
               ) -> tuple[np.ndarray, np.ndarray]:
    """Bollinger upper and lower bands."""
    sma = np.convolve(close, np.ones(period) / period, mode='full')[:len(close)]
    sma[:period - 1] = np.nan
    # Rolling std
    std = np.full_like(close, np.nan)
    for i in range(period - 1, len(close)):
        std[i] = close[i - period + 1:i + 1].std(ddof=0)
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower


def _stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                k_period: int, d_period: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """%K and %D stochastic oscillator."""
    n = len(close)
    pct_k = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        hh = high[i - k_period + 1:i + 1].max()
        ll = low[i - k_period + 1:i + 1].min()
        if hh - ll > 0:
            pct_k[i] = 100.0 * (close[i] - ll) / (hh - ll)
        else:
            pct_k[i] = 50.0
    # %D = SMA of %K
    pct_d = np.full(n, np.nan)
    for i in range(k_period + d_period - 2, n):
        window = pct_k[i - d_period + 1:i + 1]
        if np.all(np.isfinite(window)):
            pct_d[i] = window.mean()
    return pct_k, pct_d


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX, +DI, -DI."""
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        h_l = high[i] - low[i]
        h_cp = abs(high[i] - close[i - 1])
        l_cp = abs(low[i] - close[i - 1])
        tr[i] = max(h_l, h_cp, l_cp)
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    # Wilder smoothing
    atr_arr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    adx_arr = np.full(n, np.nan)

    atr_arr[period] = tr[1:period + 1].sum()
    s_plus = plus_dm[1:period + 1].sum()
    s_minus = minus_dm[1:period + 1].sum()

    for i in range(period + 1, n):
        atr_arr[i] = atr_arr[i - 1] - atr_arr[i - 1] / period + tr[i]
        s_plus = s_plus - s_plus / period + plus_dm[i]
        s_minus = s_minus - s_minus / period + minus_dm[i]
        if atr_arr[i] > 0:
            plus_di[i] = 100.0 * s_plus / atr_arr[i]
            minus_di[i] = 100.0 * s_minus / atr_arr[i]

    # ADX = smoothed DX
    dx = np.zeros(n)
    for i in range(period + 1, n):
        di_sum = plus_di[i] + minus_di[i]
        dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum if di_sum > 0 else 0.0

    start = period * 2 + 1
    if start < n:
        adx_arr[start] = dx[period + 1:start + 1].mean()
        for i in range(start + 1, n):
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period

    return adx_arr, plus_di, minus_di


def _donchian(high: np.ndarray, low: np.ndarray, period: int
              ) -> tuple[np.ndarray, np.ndarray]:
    """Donchian channel upper/lower."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period, n):
        upper[i] = high[i - period:i].max()
        lower[i] = low[i - period:i].min()
    return upper, lower


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@register
class VerificationTest(Strategy):
    """Multi-signal high-frequency strategy for full-system verification."""

    @property
    def name(self) -> str:
        return "verification_test"

    @property
    def version(self) -> str:
        return "2.0.0"

    def param_space(self) -> ParamSpace:
        params = [
            # signal_variant selects which signal mode to use
            # Maps to PL_SIGNAL_VARIANT via _PARAM_TO_PL["signal_variant"]
            ParamDef("signal_variant", ALL_MODES, group="signal"),
        ]
        # Full risk params — all SL/TP modes, ATR multipliers, fixed pips
        params += risk_params()
        # Full management params — trailing, breakeven, partial close, etc.
        params += management_params()
        # All hours, all weekdays — no time filtering
        params += [
            ParamDef("allowed_hours_start", [0], group="time"),
            ParamDef("allowed_hours_end", [23], group="time"),
            ParamDef("allowed_days", [[0, 1, 2, 3, 4]], group="time"),
        ]
        return ParamSpace(params)

    def optimization_stages(self) -> list[str | tuple[str, list[str]]]:
        # Skip time — verification test has fixed time params
        composite_groups_ordered = ["risk", "exit_trailing", "exit_protection_be"]
        module_groups = {mod.group for mod in self.management_modules()}
        composite_groups = [
            g for g in composite_groups_ordered
            if g == "risk" or g in module_groups
        ]

        stages: list[str | tuple[str, list[str]]] = ["signal"]
        if len(composite_groups) > 1:
            stages.append(("core_trade_profile", composite_groups))
        else:
            stages.append("risk")

        composite_set = set(composite_groups)
        seen: set[str] = set()
        for mod in self.management_modules():
            if mod.group not in composite_set and mod.group not in seen:
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

        atr_14 = atr(high, low, close, 14)

        parts_idx: list[np.ndarray] = []
        parts_dir: list[np.ndarray] = []
        parts_price: list[np.ndarray] = []
        parts_hour: list[np.ndarray] = []
        parts_day: list[np.ndarray] = []
        parts_atr: list[np.ndarray] = []
        parts_var: list[np.ndarray] = []

        def _append(bar_idx: np.ndarray, direction: int, mode: int) -> None:
            if len(bar_idx) == 0:
                return
            parts_idx.append(bar_idx)
            parts_dir.append(np.full(len(bar_idx), direction, dtype=np.int64))
            parts_price.append(close[bar_idx])
            parts_hour.append(bar_hour[bar_idx])
            parts_day.append(bar_day_of_week[bar_idx])
            parts_atr.append(atr_14[bar_idx] / pip_value)
            parts_var.append(np.full(len(bar_idx), mode, dtype=np.int64))

        BUY = Direction.BUY.value
        SELL = Direction.SELL.value

        # --- Mode 1: EMA(3) vs EMA(8) crossover ---
        ema3 = ema(close, 3)
        ema8 = ema(close, 8)
        warmup = 9
        idx = np.arange(warmup, n)
        if len(idx) > 0:
            f, fp = ema3[idx], ema3[idx - 1]
            s, sp = ema8[idx], ema8[idx - 1]
            valid = np.isfinite(f) & np.isfinite(s) & np.isfinite(atr_14[idx]) & (atr_14[idx] > 0)
            _append(idx[valid & (fp <= sp) & (f > s)], BUY, MODE_EMA)
            _append(idx[valid & (fp >= sp) & (f < s)], SELL, MODE_EMA)

        # --- Mode 2: RSI(5), oversold<30 / overbought>70 ---
        rsi5 = _rsi(close, 5)
        warmup = 7
        idx = np.arange(warmup, n)
        if len(idx) > 0:
            r, rp = rsi5[idx], rsi5[idx - 1]
            valid = np.isfinite(r) & np.isfinite(rp) & np.isfinite(atr_14[idx]) & (atr_14[idx] > 0)
            _append(idx[valid & (rp >= 30) & (r < 30)], BUY, MODE_RSI)
            _append(idx[valid & (rp <= 70) & (r > 70)], SELL, MODE_RSI)

        # --- Mode 3: MACD(5,13,5) signal-line crossover ---
        macd_line, sig_line = _macd(close, 5, 13, 5)
        warmup = 20
        idx = np.arange(warmup, n)
        if len(idx) > 0:
            m, mp = macd_line[idx], macd_line[idx - 1]
            sl, slp = sig_line[idx], sig_line[idx - 1]
            valid = np.isfinite(m) & np.isfinite(sl) & np.isfinite(atr_14[idx]) & (atr_14[idx] > 0)
            _append(idx[valid & (mp <= slp) & (m > sl)], BUY, MODE_MACD)
            _append(idx[valid & (mp >= slp) & (m < sl)], SELL, MODE_MACD)

        # --- Mode 4: Bollinger(10, 1.5σ) band touch ---
        bb_upper, bb_lower = _bollinger(close, 10, 1.5)
        warmup = 11
        idx = np.arange(warmup, n)
        if len(idx) > 0:
            valid = (np.isfinite(bb_upper[idx]) & np.isfinite(bb_lower[idx])
                     & np.isfinite(atr_14[idx]) & (atr_14[idx] > 0))
            # Price crosses below lower band → BUY (reversion)
            _append(idx[valid & (close[idx - 1] >= bb_lower[idx - 1]) & (close[idx] < bb_lower[idx])], BUY, MODE_BOLLINGER)
            # Price crosses above upper band → SELL (reversion)
            _append(idx[valid & (close[idx - 1] <= bb_upper[idx - 1]) & (close[idx] > bb_upper[idx])], SELL, MODE_BOLLINGER)

        # --- Mode 5: Stochastic(5,3) %K/%D cross in zones ---
        pct_k, pct_d = _stochastic(high, low, close, 5, 3)
        warmup = 8
        idx = np.arange(warmup, n)
        if len(idx) > 0:
            k, kp = pct_k[idx], pct_k[idx - 1]
            d, dp = pct_d[idx], pct_d[idx - 1]
            valid = (np.isfinite(k) & np.isfinite(d) & np.isfinite(kp) & np.isfinite(dp)
                     & np.isfinite(atr_14[idx]) & (atr_14[idx] > 0))
            # %K crosses above %D in oversold zone (<25) → BUY
            _append(idx[valid & (kp <= dp) & (k > d) & (k < 25)], BUY, MODE_STOCHASTIC)
            # %K crosses below %D in overbought zone (>75) → SELL
            _append(idx[valid & (kp >= dp) & (k < d) & (k > 75)], SELL, MODE_STOCHASTIC)

        # --- Mode 6: ADX(10) + DI cross, threshold=15 ---
        adx_val, plus_di, minus_di = _adx(high, low, close, 10)
        warmup = 22
        idx = np.arange(warmup, n)
        if len(idx) > 0:
            valid = (np.isfinite(adx_val[idx]) & (adx_val[idx] >= 15)
                     & np.isfinite(atr_14[idx]) & (atr_14[idx] > 0))
            pi, pip_ = plus_di[idx], plus_di[idx - 1]
            mi, mip = minus_di[idx], minus_di[idx - 1]
            # +DI crosses above -DI → BUY
            _append(idx[valid & (pip_ <= mip) & (pi > mi)], BUY, MODE_ADX)
            # -DI crosses above +DI → SELL
            _append(idx[valid & (pip_ >= mip) & (pi < mi)], SELL, MODE_ADX)

        # --- Mode 7: Donchian(10) channel breakout ---
        don_upper, don_lower = _donchian(high, low, 10)
        warmup = 11
        idx = np.arange(warmup, n)
        if len(idx) > 0:
            valid = (np.isfinite(don_upper[idx]) & np.isfinite(don_lower[idx])
                     & np.isfinite(atr_14[idx]) & (atr_14[idx] > 0))
            # Close breaks above previous upper channel → BUY
            _append(idx[valid & (close[idx] > don_upper[idx])], BUY, MODE_DONCHIAN)
            # Close breaks below previous lower channel → SELL
            _append(idx[valid & (close[idx] < don_lower[idx])], SELL, MODE_DONCHIAN)

        # --- Combine all signals ---
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
