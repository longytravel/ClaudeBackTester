"""Python-side metrics computation for reporting and telemetry.

Input: 1D array of trade PnL values (in pips).
Output: dict of named metric values.

These are the reference implementations — the JIT loop has inline equivalents
that produce matching results. Use this module for:
- Post-optimization reporting
- Telemetry detailed analysis
- Cross-validation against JIT metrics
"""

from __future__ import annotations

import numpy as np

from backtester.core.dtypes import TRADING_DAYS_PER_YEAR


def compute_metrics(
    pnl_pips: np.ndarray,
    avg_sl_pips: float = 30.0,
    trades_per_year: float | None = None,
) -> dict[str, float]:
    """Compute all metrics from an array of trade PnL values (pips).

    Args:
        pnl_pips: 1D array of per-trade profit/loss in pips.
        avg_sl_pips: Average stop-loss in pips (for Return% calculation).
        trades_per_year: Estimated trades per year (for annualization).
            If None, uses len(pnl_pips) as-is (assumes 1 year of data).

    Returns:
        Dict with keys: trades, win_rate, profit_factor, sharpe, sortino,
        max_dd_pct, return_pct, r_squared, ulcer, quality_score.
    """
    n = len(pnl_pips)

    if n == 0:
        return _empty_metrics()

    trades = float(n)
    wr = win_rate(pnl_pips)
    pf = profit_factor(pnl_pips)

    # Use provided trades_per_year, or default to 252 (daily)
    ann_factor = trades_per_year if trades_per_year else TRADING_DAYS_PER_YEAR
    sh = sharpe_ratio(pnl_pips, ann_factor)
    so = sortino_ratio(pnl_pips, ann_factor)

    dd = max_drawdown_pct(pnl_pips)
    ret = return_pct(pnl_pips, avg_sl_pips)
    rsq = r_squared(pnl_pips)
    ulc = ulcer_index(pnl_pips)
    qs = quality_score(pnl_pips, so, rsq, pf, dd, ret, ulc)

    return {
        "trades": trades,
        "win_rate": wr,
        "profit_factor": pf,
        "sharpe": sh,
        "sortino": so,
        "max_dd_pct": dd,
        "return_pct": ret,
        "r_squared": rsq,
        "ulcer": ulc,
        "quality_score": qs,
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "trades": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "max_dd_pct": 0.0,
        "return_pct": 0.0,
        "r_squared": 0.0,
        "ulcer": 0.0,
        "quality_score": 0.0,
    }


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def win_rate(pnl: np.ndarray) -> float:
    """Fraction of winning trades (PnL > 0)."""
    if len(pnl) == 0:
        return 0.0
    return float(np.sum(pnl > 0) / len(pnl))


def profit_factor(pnl: np.ndarray) -> float:
    """Gross profit / gross loss. Returns 0 if no losses, inf-safe."""
    gross_profit = float(np.sum(pnl[pnl > 0]))
    gross_loss = float(np.abs(np.sum(pnl[pnl < 0])))
    if gross_loss == 0:
        return 10.0 if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def sharpe_ratio(pnl: np.ndarray, trades_per_year: float | None = None) -> float:
    """Annualized Sharpe ratio (risk-free rate = 0 for FX)."""
    if len(pnl) < 2:
        return 0.0
    mean = float(np.mean(pnl))
    std = float(np.std(pnl, ddof=1))
    if std == 0:
        return 0.0
    ann = np.sqrt(trades_per_year) if trades_per_year else 1.0
    return (mean / std) * ann


def sortino_ratio(pnl: np.ndarray, trades_per_year: float | None = None) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    if len(pnl) < 2:
        return 0.0
    mean = float(np.mean(pnl))
    downside = pnl[pnl < 0]
    if len(downside) == 0:
        return 10.0 if mean > 0 else 0.0
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    if downside_std == 0:
        return 0.0
    ann = np.sqrt(trades_per_year) if trades_per_year else 1.0
    return (mean / downside_std) * ann


def max_drawdown_pct(pnl: np.ndarray) -> float:
    """Maximum drawdown as percentage of peak equity.

    Equity curve is computed from cumulative PnL.
    Peak starts at 0 (the baseline before any trades).
    Returns 0-100+ scale (can exceed 100% if drawdown exceeds peak).

    Matches JIT inline implementation exactly.
    """
    if len(pnl) == 0:
        return 0.0

    equity = np.cumsum(pnl)
    # Peak starts at 0 (baseline), matching JIT which initializes equity_peak=0
    peak = np.maximum(np.maximum.accumulate(equity), 0.0)

    drawdowns = peak - equity
    max_dd = float(np.max(drawdowns))

    # base_val = max(max_abs_equity_ever, equity_peak) — matching JIT
    base_val = max(float(np.max(np.abs(equity))), float(np.max(peak)))
    if base_val <= 0:
        base_val = 1.0

    return (max_dd / base_val) * 100.0


def return_pct(pnl: np.ndarray, avg_sl_pips: float = 30.0) -> float:
    """Risk-adjusted return percentage.

    Defined as total_pip_gain / avg_SL_pips * 100.
    With fixed lot sizing there's no capital base, so this measures
    how many SL-equivalents were gained.
    """
    if len(pnl) == 0 or avg_sl_pips <= 0:
        return 0.0
    total = float(np.sum(pnl))
    return (total / avg_sl_pips) * 100.0


def r_squared(pnl: np.ndarray) -> float:
    """R-squared of the equity curve vs ideal straight line.

    Measures consistency of returns. R²=1 means perfectly linear equity growth.
    """
    n = len(pnl)
    if n < 2:
        return 0.0

    equity = np.cumsum(pnl)
    x = np.arange(n, dtype=np.float64)

    # Linear regression: y = mx + b
    x_mean = np.mean(x)
    y_mean = np.mean(equity)
    ss_xy = np.sum((x - x_mean) * (equity - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx == 0:
        return 0.0

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    y_pred = slope * x + intercept
    ss_res = np.sum((equity - y_pred) ** 2)
    ss_tot = np.sum((equity - y_mean) ** 2)

    if ss_tot == 0:
        return 0.0

    rsq = 1.0 - (ss_res / ss_tot)
    return max(0.0, float(rsq))


def ulcer_index(pnl: np.ndarray) -> float:
    """Ulcer Index — RMS of percentage drawdowns from equity peak.

    Computed from trade-level equity curve (one point per trade close).
    Lower is better. Measures both depth and duration of drawdowns.

    Uses a running base_val (max of abs(equity) and peak seen so far)
    to match the JIT inline implementation exactly.
    """
    if len(pnl) == 0:
        return 0.0

    n = len(pnl)
    equity = 0.0
    equity_peak = 0.0
    base_val = 0.0
    sum_sq_dd = 0.0

    for i in range(n):
        equity += pnl[i]
        if equity > equity_peak:
            equity_peak = equity
        dd = equity_peak - equity

        if abs(equity) > base_val:
            base_val = abs(equity)
        if equity_peak > base_val:
            base_val = equity_peak

        if base_val > 0:
            pct_dd = (dd / base_val) * 100.0
        else:
            pct_dd = 0.0
        sum_sq_dd += pct_dd * pct_dd

    return float(np.sqrt(sum_sq_dd / n))


def quality_score(
    pnl: np.ndarray,
    sortino: float | None = None,
    rsq: float | None = None,
    pf: float | None = None,
    max_dd: float | None = None,
    ret_pct: float | None = None,
    ulc: float | None = None,
) -> float:
    """Combined quality score for ranking strategies.

    Formula: (Sortino * R² * min(PF,5) * sqrt(min(Trades,200))
              * (1 + clamp(Return%,0,200)/100)) / (Ulcer + MaxDD/2 + 5)

    Guards:
    - Sortino <= 0 → quality = 0 (losing strategies have no quality)
    - Return% clamped to [0, 200] (negative returns give no bonus, never
      flip the sign via double-negative with Sortino)

    Pre-computed metric values can be passed to avoid recalculation.
    """
    n = len(pnl)
    if n == 0:
        return 0.0

    so = sortino if sortino is not None else sortino_ratio(pnl)

    # Losing strategies score zero — no ranking needed
    if so <= 0:
        return 0.0

    r2 = rsq if rsq is not None else r_squared(pnl)
    p = pf if pf is not None else profit_factor(pnl)
    dd = max_dd if max_dd is not None else max_drawdown_pct(pnl)
    ret = ret_pct if ret_pct is not None else return_pct(pnl)
    u = ulc if ulc is not None else ulcer_index(pnl)

    # Clamp components
    p_clamped = min(p, 5.0)
    trades_factor = np.sqrt(min(n, 200))
    # Return% bonus: only positive returns contribute, clamped to [0, 200]
    ret_factor = 1.0 + max(0.0, min(ret, 200.0)) / 100.0

    numerator = so * r2 * p_clamped * trades_factor * ret_factor
    denominator = u + dd / 2.0 + 5.0

    if denominator <= 0:
        return 0.0

    return float(numerator / denominator)
