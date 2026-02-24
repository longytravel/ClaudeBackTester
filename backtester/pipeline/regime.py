"""Regime-aware validation (VP-3).

Classifies every bar into one of 4 market regimes using ADX + normalized ATR,
then breaks down strategy performance by regime. Advisory only — does not
eliminate candidates or affect composite confidence score.

Regimes:
  0 = Trend + Quiet
  1 = Trend + Volatile
  2 = Range + Quiet
  3 = Range + Volatile
 -1 = Unknown (warmup period)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backtester.strategies.indicators import adx, atr

# ---------------------------------------------------------------------------
# Regime constants
# ---------------------------------------------------------------------------

REGIME_TREND_QUIET = 0
REGIME_TREND_VOLATILE = 1
REGIME_RANGE_QUIET = 2
REGIME_RANGE_VOLATILE = 3
REGIME_UNKNOWN = -1

REGIME_NAMES: dict[int, str] = {
    REGIME_TREND_QUIET: "Trend + Quiet",
    REGIME_TREND_VOLATILE: "Trend + Volatile",
    REGIME_RANGE_QUIET: "Range + Quiet",
    REGIME_RANGE_VOLATILE: "Range + Volatile",
    REGIME_UNKNOWN: "Unknown",
}


# ---------------------------------------------------------------------------
# Bar classification
# ---------------------------------------------------------------------------

def classify_bars(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    adx_period: int = 14,
    atr_period: int = 14,
    adx_trending_threshold: float = 25.0,
    adx_ranging_threshold: float = 20.0,
    natr_percentile_lookback: int = 100,
    natr_high_percentile: float = 75.0,
    min_regime_bars: int = 8,
) -> np.ndarray:
    """Classify each bar into one of 4 regimes.

    Returns int array: 0=TREND_QUIET, 1=TREND_VOLATILE,
                       2=RANGE_QUIET, 3=RANGE_VOLATILE, -1=UNKNOWN.

    Logic:
    - ADX >= adx_trending_threshold -> trending
    - ADX < adx_ranging_threshold -> ranging
    - Between thresholds -> maintain previous state (hysteresis)
    - NATR percentile rank over trailing window (causal, no lookahead)
    - NATR pctile >= natr_high_percentile -> high volatility
    - Combine: trend/range x high/low vol = 4 regimes
    - Apply min_regime_bars cooldown to suppress flicker
    - Warmup bars -> UNKNOWN (-1)
    """
    n = len(close)
    labels = np.full(n, REGIME_UNKNOWN, dtype=np.int64)

    if n < max(adx_period * 2, atr_period, natr_percentile_lookback):
        return labels

    # Compute indicators
    adx_vals, _, _ = adx(high, low, close, adx_period)
    atr_vals = atr(high, low, close, atr_period)

    # Normalized ATR (ATR / close)
    natr = np.full(n, np.nan)
    valid_close = close != 0
    natr[valid_close] = atr_vals[valid_close] / close[valid_close]

    # NATR percentile rank (causal: expanding window with min_periods)
    natr_pctile = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(natr[i]):
            continue
        # Expanding window with min_periods = natr_percentile_lookback
        start = max(0, i - natr_percentile_lookback + 1)
        window = natr[start:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < natr_percentile_lookback:
            continue
        # Percentile rank: fraction of values <= current value
        natr_pctile[i] = np.sum(valid <= natr[i]) / len(valid) * 100.0

    # Classify each bar
    is_trending = False  # hysteresis state

    for i in range(n):
        if np.isnan(adx_vals[i]) or np.isnan(natr_pctile[i]):
            continue

        # Trend/range with hysteresis
        if adx_vals[i] >= adx_trending_threshold:
            is_trending = True
        elif adx_vals[i] < adx_ranging_threshold:
            is_trending = False
        # else: maintain previous state

        # Volatility
        is_volatile = natr_pctile[i] >= natr_high_percentile

        # Combine
        if is_trending:
            labels[i] = REGIME_TREND_VOLATILE if is_volatile else REGIME_TREND_QUIET
        else:
            labels[i] = REGIME_RANGE_VOLATILE if is_volatile else REGIME_RANGE_QUIET

    # Apply min_regime_bars cooldown: suppress transitions shorter than threshold
    if min_regime_bars > 1:
        labels = _apply_min_duration(labels, min_regime_bars)

    return labels


def _apply_min_duration(labels: np.ndarray, min_bars: int) -> np.ndarray:
    """Suppress regime transitions shorter than min_bars.

    Short bursts are replaced with the preceding regime.
    """
    n = len(labels)
    out = labels.copy()

    # Find runs of each regime
    i = 0
    while i < n:
        if out[i] == REGIME_UNKNOWN:
            i += 1
            continue

        # Find end of this run
        run_start = i
        current = out[i]
        j = i + 1
        while j < n and out[j] == current:
            j += 1
        run_len = j - run_start

        if run_len < min_bars:
            # Find previous valid regime
            prev_regime = REGIME_UNKNOWN
            for k in range(run_start - 1, -1, -1):
                if out[k] != REGIME_UNKNOWN and out[k] != current:
                    prev_regime = out[k]
                    break
            if prev_regime == REGIME_UNKNOWN:
                # No previous regime — check if there's a regime before the unknown block
                for k in range(run_start - 1, -1, -1):
                    if out[k] != REGIME_UNKNOWN:
                        prev_regime = out[k]
                        break
            if prev_regime != REGIME_UNKNOWN:
                out[run_start:j] = prev_regime

        i = j

    return out


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RegimeStats:
    """Per-regime statistics."""
    regime: int
    regime_name: str
    n_bars: int = 0
    bar_pct: float = 0.0
    n_trades: int = 0
    sharpe: float = 0.0
    profit_factor: float = 0.0
    max_dd_pct: float = 0.0
    win_rate: float = 0.0
    mean_pnl_pips: float = 0.0
    sufficient_data: bool = False


@dataclass
class RegimeResult:
    """Full regime analysis for one candidate."""
    regime_distribution: dict[str, float] = field(default_factory=dict)
    per_regime: list[RegimeStats] = field(default_factory=list)
    regime_weighted_sharpe: float = 0.0
    worst_regime_max_dd: float = 0.0
    n_profitable_regimes: int = 0
    n_scored_regimes: int = 0
    advisory: str = ""
    robustness_score: float = 0.0


# ---------------------------------------------------------------------------
# Per-regime metrics computation
# ---------------------------------------------------------------------------

def compute_regime_stats(
    regime_labels: np.ndarray,
    trades: list,
    min_trades_per_regime: int = 30,
) -> RegimeResult:
    """Compute per-regime performance breakdown from telemetry trades.

    Tags each trade by regime at entry bar.
    For each regime with >= min_trades: compute Sharpe, PF, MaxDD%, win rate.
    Regimes with < min_trades: mark as insufficient.
    """
    result = RegimeResult()
    n_bars = len(regime_labels)

    # Compute bar distribution (excluding UNKNOWN)
    valid_mask = regime_labels != REGIME_UNKNOWN
    n_valid = int(np.sum(valid_mask))

    for regime_code in (REGIME_TREND_QUIET, REGIME_TREND_VOLATILE,
                        REGIME_RANGE_QUIET, REGIME_RANGE_VOLATILE):
        n_regime_bars = int(np.sum(regime_labels == regime_code))
        bar_pct = n_regime_bars / n_valid * 100.0 if n_valid > 0 else 0.0
        result.regime_distribution[REGIME_NAMES[regime_code]] = bar_pct

    if not trades:
        # No trades — return empty stats
        for regime_code in (REGIME_TREND_QUIET, REGIME_TREND_VOLATILE,
                            REGIME_RANGE_QUIET, REGIME_RANGE_VOLATILE):
            result.per_regime.append(RegimeStats(
                regime=regime_code,
                regime_name=REGIME_NAMES[regime_code],
                n_bars=int(np.sum(regime_labels == regime_code)),
                bar_pct=result.regime_distribution.get(REGIME_NAMES[regime_code], 0.0),
            ))
        return result

    # Group trades by entry-bar regime
    regime_trades: dict[int, list] = {
        REGIME_TREND_QUIET: [],
        REGIME_TREND_VOLATILE: [],
        REGIME_RANGE_QUIET: [],
        REGIME_RANGE_VOLATILE: [],
    }
    for t in trades:
        bar = t.bar_entry
        if 0 <= bar < n_bars:
            regime = int(regime_labels[bar])
            if regime in regime_trades:
                regime_trades[regime].append(t)
        # Trades with entry bar in UNKNOWN are not counted

    # Compute per-regime stats
    scored_sharpes = []
    scored_time_pcts = []

    for regime_code in (REGIME_TREND_QUIET, REGIME_TREND_VOLATILE,
                        REGIME_RANGE_QUIET, REGIME_RANGE_VOLATILE):
        regime_name = REGIME_NAMES[regime_code]
        n_regime_bars = int(np.sum(regime_labels == regime_code))
        bar_pct = result.regime_distribution.get(regime_name, 0.0)
        rtrades = regime_trades[regime_code]

        stats = RegimeStats(
            regime=regime_code,
            regime_name=regime_name,
            n_bars=n_regime_bars,
            bar_pct=bar_pct,
            n_trades=len(rtrades),
        )

        if len(rtrades) >= min_trades_per_regime:
            stats.sufficient_data = True
            pnls = np.array([t.pnl_pips for t in rtrades], dtype=np.float64)

            # Win rate
            stats.win_rate = float(np.sum(pnls > 0) / len(pnls) * 100.0)

            # Mean PnL
            stats.mean_pnl_pips = float(np.mean(pnls))

            # Sharpe (annualized, assume ~6048 H1 bars/year)
            if np.std(pnls) > 0:
                trades_per_year = len(pnls)  # approximate
                stats.sharpe = float(
                    np.mean(pnls) / np.std(pnls) * np.sqrt(trades_per_year)
                )
            else:
                stats.sharpe = 0.0

            # Profit factor
            gross_win = float(np.sum(pnls[pnls > 0]))
            gross_loss = float(abs(np.sum(pnls[pnls <= 0])))
            stats.profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

            # Max drawdown %
            cumulative = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = running_max - cumulative
            if running_max[-1] > 0:
                stats.max_dd_pct = float(np.max(drawdowns) / max(np.max(running_max), 1e-10) * 100.0)
            else:
                stats.max_dd_pct = 100.0 if np.max(drawdowns) > 0 else 0.0

            # Track for aggregate metrics
            scored_sharpes.append(stats.sharpe)
            scored_time_pcts.append(bar_pct / 100.0)

        result.per_regime.append(stats)

    # Aggregate metrics
    result.n_scored_regimes = len(scored_sharpes)
    result.n_profitable_regimes = sum(1 for s in scored_sharpes if s > 0)

    if scored_sharpes and scored_time_pcts:
        total_time = sum(scored_time_pcts)
        if total_time > 0:
            result.regime_weighted_sharpe = sum(
                s * t / total_time for s, t in zip(scored_sharpes, scored_time_pcts)
            )

    # Worst regime MaxDD
    scored_stats = [s for s in result.per_regime if s.sufficient_data]
    if scored_stats:
        result.worst_regime_max_dd = max(s.max_dd_pct for s in scored_stats)

    # Advisory message
    result.advisory = _build_advisory(result)

    # Robustness score
    result.robustness_score = score_regime_robustness(result)

    return result


# ---------------------------------------------------------------------------
# Robustness scoring
# ---------------------------------------------------------------------------

def score_regime_robustness(regime_result: RegimeResult) -> float:
    """Score regime consistency (0-100). Advisory sub-score.

    Components:
    - Profitable in multiple regimes (40%): n_profitable / n_scored * 100
    - No catastrophic regime (30%): worst MaxDD < 40% = full, 40-60% = partial, >60% = 0
    - Low cross-regime variance (30%): inverse of Sharpe CV across scored regimes
    """
    if regime_result.n_scored_regimes == 0:
        return 0.0

    # Component 1: Profitable regimes (40%)
    profitable_pct = (
        regime_result.n_profitable_regimes / regime_result.n_scored_regimes * 100.0
    )
    comp1 = min(profitable_pct, 100.0) * 0.40

    # Component 2: No catastrophic regime (30%)
    worst_dd = regime_result.worst_regime_max_dd
    if worst_dd < 40.0:
        comp2 = 100.0 * 0.30
    elif worst_dd < 60.0:
        comp2 = (1.0 - (worst_dd - 40.0) / 20.0) * 100.0 * 0.30
    else:
        comp2 = 0.0

    # Component 3: Low cross-regime Sharpe variance (30%)
    scored_sharpes = [
        s.sharpe for s in regime_result.per_regime if s.sufficient_data
    ]
    if len(scored_sharpes) >= 2:
        mean_s = np.mean(scored_sharpes)
        std_s = np.std(scored_sharpes)
        if abs(mean_s) > 1e-10:
            cv = abs(std_s / mean_s)
            # CV < 0.5 = full marks, CV > 2.0 = 0
            variance_score = max(0.0, min(1.0, 1.0 - (cv - 0.5) / 1.5))
        else:
            variance_score = 0.0
        comp3 = variance_score * 100.0 * 0.30
    else:
        # Only 1 scored regime — can't compute variance
        comp3 = 50.0 * 0.30

    return comp1 + comp2 + comp3


# ---------------------------------------------------------------------------
# Advisory message builder
# ---------------------------------------------------------------------------

def _build_advisory(result: RegimeResult) -> str:
    """Build human-readable advisory message."""
    parts = []

    if result.n_scored_regimes == 0:
        return "Insufficient trades per regime for analysis."

    if result.n_profitable_regimes == 0:
        parts.append("Strategy is unprofitable in ALL scored regimes.")
    elif result.n_profitable_regimes == 1:
        # Find the profitable one
        profitable = [
            s for s in result.per_regime if s.sufficient_data and s.sharpe > 0
        ]
        if profitable:
            parts.append(
                f"Strategy only profitable in {profitable[0].regime_name} regime. "
                f"High regime concentration risk."
            )
    elif result.n_profitable_regimes == result.n_scored_regimes:
        parts.append("Strategy profitable across all scored regimes.")

    # Check for catastrophic regime
    catastrophic = [
        s for s in result.per_regime
        if s.sufficient_data and s.max_dd_pct > 40.0
    ]
    if catastrophic:
        worst = max(catastrophic, key=lambda s: s.max_dd_pct)
        parts.append(
            f"Catastrophic drawdown in {worst.regime_name} "
            f"({worst.max_dd_pct:.1f}% MaxDD)."
        )

    # Check for weak regimes (negative Sharpe)
    weak = [
        s for s in result.per_regime
        if s.sufficient_data and s.sharpe < 0
    ]
    if weak:
        names = [s.regime_name for s in weak]
        parts.append(f"Strategy underperforms in: {', '.join(names)}.")

    if not parts:
        parts.append("Regime analysis complete.")

    return " ".join(parts)
