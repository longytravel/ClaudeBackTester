"""Numpy-based indicator library (REQ-S24, REQ-S25).

All functions accept and return numpy arrays. No pandas, no external TA
libraries in the hot path. Designed for direct use in Numba-compatible
strategy code.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Rolling helpers (vectorized, no Python loops)
# ---------------------------------------------------------------------------

def _rolling_max(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling maximum over a window. Output valid from index period-1."""
    out = np.full(len(data), np.nan)
    if period > len(data):
        return out
    windows = sliding_window_view(data, period)
    out[period - 1:] = np.max(windows, axis=1)
    return out


def _rolling_min(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling minimum over a window. Output valid from index period-1."""
    out = np.full(len(data), np.nan)
    if period > len(data):
        return out
    windows = sliding_window_view(data, period)
    out[period - 1:] = np.min(windows, axis=1)
    return out


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    out = np.full(len(data), np.nan)
    if period > len(data):
        return out
    cumsum = np.cumsum(data)
    out[period - 1:] = (cumsum[period - 1:] - np.concatenate(([0.0], cumsum[:-period]))) / period
    return out


def ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    out = np.full(len(data), np.nan)
    if period > len(data):
        return out
    alpha = 2.0 / (period + 1)
    out[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# ATR (Average True Range)
# ---------------------------------------------------------------------------

def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range for each bar (first bar uses high-low)."""
    hl = high - low
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr = np.empty(len(high))
    tr[0] = hl[0]
    tr[1:] = np.maximum(hl[1:], np.maximum(hc, lc))
    return tr


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range (Wilder's smoothing)."""
    tr = true_range(high, low, close)
    out = np.full(len(high), np.nan)
    if period > len(high):
        return out
    out[period - 1] = np.mean(tr[:period])
    multiplier = (period - 1) / period
    for i in range(period, len(high)):
        out[i] = out[i - 1] * multiplier + tr[i] / period
    return out


# ---------------------------------------------------------------------------
# RSI (Relative Strength Index)
# ---------------------------------------------------------------------------

def rsi(data: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI using Wilder's smoothing method."""
    out = np.full(len(data), np.nan)
    if period + 1 > len(data):
        return out

    deltas = np.diff(data)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return out


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    data: np.ndarray, period: int = 20, num_std: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands. Returns (upper, middle, lower)."""
    middle = sma(data, period)
    std = np.full(len(data), np.nan)
    if period <= len(data):
        # Rolling std via cumsum: std = sqrt(mean(x²) - mean(x)²)
        cumsum = np.cumsum(data)
        cumsum2 = np.cumsum(data * data)
        s = cumsum[period - 1:] - np.concatenate(([0.0], cumsum[:-period]))
        s2 = cumsum2[period - 1:] - np.concatenate(([0.0], cumsum2[:-period]))
        variance = s2 / period - (s / period) ** 2
        # Clamp tiny negatives from floating-point to zero before sqrt
        np.maximum(variance, 0.0, out=variance)
        std[period - 1:] = np.sqrt(variance)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------

def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator. Returns (%K, %D)."""
    n = len(close)
    k = np.full(n, np.nan)
    if k_period <= n:
        hh = _rolling_max(high, k_period)
        ll = _rolling_min(low, k_period)
        denom = hh - ll
        valid = denom > 0
        k[k_period - 1:] = np.where(
            valid[k_period - 1:],
            100.0 * (close[k_period - 1:] - ll[k_period - 1:]) / np.where(valid[k_period - 1:], denom[k_period - 1:], 1.0),
            50.0,
        )
    # %D = SMA of %K (only over valid portion, since sma can't handle NaN)
    d_full = np.full(n, np.nan)
    k_start = k_period - 1  # first valid %K index
    if k_start < n:
        k_valid = k[k_start:]  # all valid %K values
        d_valid = sma(k_valid, d_period)
        d_full[k_start:] = d_valid
    return k, d_full


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(
    data: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD. Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow

    # Signal line: EMA of the MACD line (only where MACD is valid)
    valid_mask = ~np.isnan(macd_line)
    signal_line = np.full(len(data), np.nan)
    if np.sum(valid_mask) >= signal_period:
        valid_macd = macd_line[valid_mask]
        sig = ema(valid_macd, signal_period)
        signal_line[valid_mask] = sig

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# ADX (Average Directional Index)
# ---------------------------------------------------------------------------

def adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ADX with +DI and -DI. Returns (adx, plus_di, minus_di)."""
    n = len(high)
    # Vectorized directional movement
    up = np.empty(n)
    up[0] = 0.0
    up[1:] = high[1:] - high[:-1]
    down = np.empty(n)
    down[0] = 0.0
    down[1:] = low[:-1] - low[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    plus_dm[0] = 0.0
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    minus_dm[0] = 0.0

    atr_val = atr(high, low, close, period)

    # Smoothed DM using Wilder's method
    smooth_plus = np.full(n, np.nan)
    smooth_minus = np.full(n, np.nan)

    if period > n:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    smooth_plus[period] = np.sum(plus_dm[1 : period + 1])
    smooth_minus[period] = np.sum(minus_dm[1 : period + 1])

    for i in range(period + 1, n):
        smooth_plus[i] = smooth_plus[i - 1] - smooth_plus[i - 1] / period + plus_dm[i]
        smooth_minus[i] = smooth_minus[i - 1] - smooth_minus[i - 1] / period + minus_dm[i]

    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)

    for i in range(period, n):
        if not np.isnan(atr_val[i]) and atr_val[i] > 0:
            plus_di[i] = 100.0 * smooth_plus[i] / (atr_val[i] * period)
            minus_di[i] = 100.0 * smooth_minus[i] / (atr_val[i] * period)
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX: Wilder's smoothed DX
    adx_out = np.full(n, np.nan)
    # Find first valid DX stretch of length period
    valid_dx = ~np.isnan(dx)
    first_valid = -1
    count = 0
    for i in range(n):
        if valid_dx[i]:
            if count == 0:
                first_valid = i
            count += 1
            if count == period:
                break
        else:
            count = 0
            first_valid = -1

    if first_valid >= 0 and count >= period:
        start = first_valid + period - 1
        adx_out[start] = np.nanmean(dx[first_valid : first_valid + period])
        for i in range(start + 1, n):
            if not np.isnan(dx[i]):
                adx_out[i] = (adx_out[i - 1] * (period - 1) + dx[i]) / period

    return adx_out, plus_di, minus_di


# ---------------------------------------------------------------------------
# Donchian Channel
# ---------------------------------------------------------------------------

def donchian(
    high: np.ndarray, low: np.ndarray, period: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Donchian Channel. Returns (upper, middle, lower)."""
    upper = _rolling_max(high, period)
    lower = _rolling_min(low, period)
    middle = (upper + lower) / 2
    return upper, middle, lower


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------

def supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Supertrend indicator. Returns (supertrend_line, direction).

    direction: 1 = uptrend (bullish), -1 = downtrend (bearish).
    """
    atr_val = atr(high, low, close, period)
    n = len(close)

    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=np.int64)

    basic_upper = np.full(n, np.nan)
    basic_lower = np.full(n, np.nan)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)

    for i in range(period - 1, n):
        if np.isnan(atr_val[i]):
            continue
        hl2 = (high[i] + low[i]) / 2
        basic_upper[i] = hl2 + multiplier * atr_val[i]
        basic_lower[i] = hl2 - multiplier * atr_val[i]

    # Initialize
    start = period - 1
    while start < n and np.isnan(basic_upper[start]):
        start += 1
    if start >= n:
        return st, direction

    final_upper[start] = basic_upper[start]
    final_lower[start] = basic_lower[start]
    st[start] = basic_upper[start]
    direction[start] = -1

    for i in range(start + 1, n):
        if np.isnan(basic_upper[i]):
            continue

        # Final upper band
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Final lower band
        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction and supertrend value
        if direction[i - 1] == 1:
            if close[i] < final_lower[i]:
                direction[i] = -1
                st[i] = final_upper[i]
            else:
                direction[i] = 1
                st[i] = final_lower[i]
        else:
            if close[i] > final_upper[i]:
                direction[i] = 1
                st[i] = final_lower[i]
            else:
                direction[i] = -1
                st[i] = final_upper[i]

    return st, direction


# ---------------------------------------------------------------------------
# Keltner Channel
# ---------------------------------------------------------------------------

def keltner(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channel. Returns (upper, middle, lower)."""
    middle = ema(close, ema_period)
    atr_val = atr(high, low, close, atr_period)
    upper = middle + multiplier * atr_val
    lower = middle - multiplier * atr_val
    return upper, middle, lower


# ---------------------------------------------------------------------------
# Williams %R
# ---------------------------------------------------------------------------

def williams_r(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14,
) -> np.ndarray:
    """Williams %R oscillator. Range: -100 to 0."""
    n = len(close)
    out = np.full(n, np.nan)
    if period > n:
        return out
    hh = _rolling_max(high, period)
    ll = _rolling_min(low, period)
    denom = hh - ll
    valid = denom > 0
    s = period - 1
    out[s:] = np.where(
        valid[s:],
        -100.0 * (hh[s:] - close[s:]) / np.where(valid[s:], denom[s:], 1.0),
        -50.0,
    )
    return out


# ---------------------------------------------------------------------------
# CCI (Commodity Channel Index)
# ---------------------------------------------------------------------------

def cci(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20,
) -> np.ndarray:
    """Commodity Channel Index."""
    n = len(close)
    tp = (high + low + close) / 3.0
    tp_sma = sma(tp, period)
    out = np.full(n, np.nan)
    if period > n:
        return out
    # Rolling mean absolute deviation: mean(|tp - sma|) over window
    # Computed via sliding_window_view of tp, subtract corresponding sma
    windows = sliding_window_view(tp, period)  # shape (n-period+1, period)
    # tp_sma is valid from period-1 onward, align with windows
    sma_vals = tp_sma[period - 1:]  # shape (n-period+1,)
    deviations = np.abs(windows - sma_vals[:, np.newaxis])
    mean_dev = np.mean(deviations, axis=1)
    s = period - 1
    diff = tp[s:] - tp_sma[s:]
    valid = mean_dev > 0
    out[s:] = np.where(valid, diff / (0.015 * np.where(valid, mean_dev, 1.0)), 0.0)
    return out


# ---------------------------------------------------------------------------
# Swing High / Low Detection
# ---------------------------------------------------------------------------

def swing_highs(high: np.ndarray, lookback: int = 5) -> np.ndarray:
    """Detect swing highs. Returns array of NaN except at swing high bars.

    A swing high at bar i means high[i] is the highest of bars [i-lookback, i+lookback].
    """
    n = len(high)
    out = np.full(n, np.nan)
    window_size = 2 * lookback + 1
    if window_size > n:
        return out
    windows = sliding_window_view(high, window_size)  # (n - window_size + 1, window_size)
    win_max = np.max(windows, axis=1)
    # Center of each window is at index lookback within the window,
    # which corresponds to global index i = lookback + row_index
    center_vals = high[lookback: n - lookback]
    is_swing = center_vals == win_max
    out[lookback: n - lookback] = np.where(is_swing, center_vals, np.nan)
    return out


def swing_lows(low: np.ndarray, lookback: int = 5) -> np.ndarray:
    """Detect swing lows. Returns array of NaN except at swing low bars."""
    n = len(low)
    out = np.full(n, np.nan)
    window_size = 2 * lookback + 1
    if window_size > n:
        return out
    windows = sliding_window_view(low, window_size)
    win_min = np.min(windows, axis=1)
    center_vals = low[lookback: n - lookback]
    is_swing = center_vals == win_min
    out[lookback: n - lookback] = np.where(is_swing, center_vals, np.nan)
    return out
