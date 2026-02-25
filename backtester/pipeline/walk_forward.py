"""Walk-forward validation for the validation pipeline (Stage 3).

Evaluates strategy candidates on rolling/anchored out-of-sample windows.
Uses a SINGLE BacktestEngine on the full dataset with windowed evaluation
to avoid memory issues from creating hundreds of engine instances.

Functions:
    generate_windows   - Create rolling or anchored window boundaries
    label_windows      - Label windows as in-sample or out-of-sample
    build_engine       - Create a BacktestEngine from data_arrays dict
    evaluate_candidate_on_window - Evaluate one candidate on one window
    walk_forward_validate - Full walk-forward validation for candidate list
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from backtester.core.dtypes import (
    EXEC_FULL,
    M_MAX_DD_PCT,
    M_PROFIT_FACTOR,
    M_QUALITY,
    M_RETURN_PCT,
    M_SHARPE,
    M_TRADES,
)
from backtester.core.encoding import encode_params
from backtester.core.engine import BacktestEngine
from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.types import WalkForwardResult, WindowResult
from backtester.strategies.base import Strategy

logger = logging.getLogger(__name__)


def generate_windows(
    n_bars: int,
    window_size: int,
    step_size: int,
    embargo_bars: int,
    anchored: bool = False,
) -> list[tuple[int, int]]:
    """Generate walk-forward window boundaries.

    Args:
        n_bars: Total number of bars in the dataset.
        window_size: Size of each test window in bars.
        step_size: Step size between window starts (rolling) or end increments (anchored).
        embargo_bars: Number of bars to skip between windows.
        anchored: If True, all windows start at bar 0 and grow by step_size.
                  If False, windows roll forward by step_size.

    Returns:
        List of (start_bar, end_bar) tuples. end_bar is exclusive.
    """
    windows: list[tuple[int, int]] = []

    if anchored:
        # Anchored: start always at 0, end grows by step_size each iteration.
        # First window covers [0, window_size), then [0, window_size + step_size), etc.
        end = window_size
        while end <= n_bars:
            windows.append((0, end))
            end += step_size + embargo_bars
    else:
        # Rolling: each window is window_size bars, stepping forward.
        start = 0
        while start + window_size <= n_bars:
            windows.append((start, start + window_size))
            start += step_size + embargo_bars

    return windows


def label_windows(
    windows: list[tuple[int, int]],
    opt_start: int,
    opt_end: int,
) -> list[bool]:
    """Label each window as OOS (True) or IS (False).

    A window is in-sample (IS) if it overlaps with [opt_start, opt_end).
    A window is out-of-sample (OOS) if it does NOT overlap.

    Args:
        windows: List of (start_bar, end_bar) tuples.
        opt_start: Start bar of the optimization range (inclusive).
        opt_end: End bar of the optimization range (exclusive).

    Returns:
        List of booleans, True = OOS, False = IS.
    """
    labels: list[bool] = []
    for w_start, w_end in windows:
        # Two ranges overlap iff: w_start < opt_end AND opt_start < w_end
        overlaps = w_start < opt_end and opt_start < w_end
        labels.append(not overlaps)
    return labels


def build_engine(
    strategy: Strategy,
    data_arrays: dict[str, np.ndarray],
    config: PipelineConfig,
    pip_value: float = 0.0001,
    slippage_pips: float = 0.5,
) -> BacktestEngine:
    """Create a BacktestEngine from a data_arrays dict.

    Shared helper for walk-forward, CPCV, and other pipeline stages
    that need to build an engine on the full dataset.
    """
    m1_kwargs: dict[str, np.ndarray] = {}
    for key in ("m1_high", "m1_low", "m1_close", "m1_spread",
                "h1_to_m1_start", "h1_to_m1_end"):
        if key in data_arrays and data_arrays[key] is not None:
            m1_kwargs[key] = data_arrays[key]

    return BacktestEngine(
        strategy=strategy,
        open_=data_arrays["open"],
        high=data_arrays["high"],
        low=data_arrays["low"],
        close=data_arrays["close"],
        volume=data_arrays["volume"],
        spread=data_arrays["spread"],
        pip_value=pip_value,
        slippage_pips=slippage_pips,
        max_trades_per_trial=config.wf_max_trades_per_trial,
        bars_per_year=config.bars_per_year,
        commission_pips=config.commission_pips,
        max_spread_pips=config.max_spread_pips,
        bar_hour=data_arrays.get("bar_hour"),
        bar_day_of_week=data_arrays.get("bar_day_of_week"),
        **m1_kwargs,
    )


def _metrics_row_to_window_result(
    row: np.ndarray,
    window_start: int,
    window_end: int,
    window_index: int,
    is_oos: bool,
    min_trades: int,
) -> WindowResult:
    """Extract metrics from a JIT output row and build a WindowResult."""
    n_trades = int(row[M_TRADES])
    sharpe = float(row[M_SHARPE])
    quality = float(row[M_QUALITY])
    profit_factor = float(row[M_PROFIT_FACTOR])
    max_dd_pct = float(row[M_MAX_DD_PCT])
    return_pct = float(row[M_RETURN_PCT])

    passed = (
        n_trades >= min_trades
        and sharpe > 0
        and quality > 0
    )

    return WindowResult(
        window_index=window_index,
        start_bar=window_start,
        end_bar=window_end,
        is_oos=is_oos,
        n_trades=n_trades,
        sharpe=sharpe,
        quality_score=quality,
        profit_factor=profit_factor,
        max_dd_pct=max_dd_pct,
        return_pct=return_pct,
        passed=passed,
    )


def evaluate_candidate_on_window(
    strategy: Strategy,
    params_dict: dict[str, Any],
    data_arrays: dict[str, np.ndarray],
    window_start: int,
    window_end: int,
    lookback_prefix: int,  # Kept for API compat, unused by shared-engine path
    config: PipelineConfig,
    window_index: int = 0,
    is_oos: bool = True,
    pip_value: float = 0.0001,
    slippage_pips: float = 0.5,
    engine: BacktestEngine | None = None,
) -> WindowResult:
    """Evaluate one candidate on one walk-forward window.

    Uses windowed evaluation on a shared BacktestEngine. If no engine is
    provided, builds one on the full dataset on the fly.

    Args:
        strategy: Strategy instance.
        params_dict: Parameter dictionary for the candidate.
        data_arrays: Dict with keys: open, high, low, close, volume, spread,
                     bar_hour, bar_day_of_week. All full-length numpy arrays.
        window_start: Start bar of the test window (inclusive).
        window_end: End bar of the test window (exclusive).
        lookback_prefix: Deprecated — unused by shared-engine pipeline.
                         Kept for API compatibility.
        config: Pipeline configuration.
        window_index: Index of this window (for result tracking).
        is_oos: Whether this window is out-of-sample.
        pip_value: Pip value for the instrument.
        slippage_pips: Slippage in pips.
        engine: Pre-built BacktestEngine on full data. If None, one is
                created from data_arrays.

    Returns:
        WindowResult with metrics from this window evaluation.
    """
    if engine is None:
        engine = build_engine(strategy, data_arrays, config, pip_value, slippage_pips)

    encoding = engine.encoding
    param_row = encode_params(encoding, params_dict)
    param_matrix = param_row.reshape(1, -1)
    metrics = engine.evaluate_batch_windowed(
        param_matrix, window_start, window_end, exec_mode=EXEC_FULL,
    )
    row = metrics[0]

    return _metrics_row_to_window_result(
        row, window_start, window_end, window_index, is_oos,
        config.wf_min_trades_per_window,
    )


def walk_forward_validate(
    strategy: Strategy,
    candidates: list[dict[str, Any]],
    data_arrays: dict[str, np.ndarray],
    opt_start: int,
    opt_end: int,
    config: PipelineConfig | None = None,
    pip_value: float = 0.0001,
    slippage_pips: float = 0.5,
    engine: BacktestEngine | None = None,
) -> list[WalkForwardResult]:
    """Run full walk-forward validation for a list of candidates.

    Creates ONE BacktestEngine on the full dataset, then evaluates each
    candidate on each window using windowed evaluation (signal filtering +
    data slicing). This avoids creating hundreds of engine instances which
    caused segfaults on large datasets (M15+).

    Args:
        strategy: Strategy instance.
        candidates: List of parameter dictionaries (one per candidate).
        data_arrays: Dict with keys: open, high, low, close, volume, spread,
                     bar_hour, bar_day_of_week. All full-length numpy arrays.
        opt_start: Start bar of the optimization range (inclusive).
        opt_end: End bar of the optimization range (exclusive).
        config: Pipeline configuration. Uses defaults if None.

    Returns:
        List of WalkForwardResult, one per candidate.
    """
    if config is None:
        config = PipelineConfig()

    n_bars = len(data_arrays["close"])

    # Generate windows over the full data range
    windows = generate_windows(
        n_bars=n_bars,
        window_size=config.wf_window_bars,
        step_size=config.wf_step_bars,
        embargo_bars=config.wf_embargo_bars,
        anchored=config.wf_anchored,
    )

    if not windows:
        logger.warning("No walk-forward windows generated (data too short?)")
        return [WalkForwardResult(passed_gate=False) for _ in candidates]

    # Label IS/OOS
    oos_labels = label_windows(windows, opt_start, opt_end)

    oos_windows = [(i, w) for i, (w, is_oos) in enumerate(zip(windows, oos_labels)) if is_oos]
    is_windows = [(i, w) for i, (w, is_oos) in enumerate(zip(windows, oos_labels)) if not is_oos]

    logger.info(
        "Walk-forward: %d total windows (%d OOS, %d IS)",
        len(windows), len(oos_windows), len(is_windows),
    )

    if not oos_windows:
        logger.warning("No OOS windows — all windows overlap optimization range")
        return [WalkForwardResult(
            n_windows=len(windows),
            n_oos_windows=0,
            passed_gate=False,
        ) for _ in candidates]

    # Use provided engine or create one — signals generated once
    if engine is None:
        engine = build_engine(strategy, data_arrays, config, pip_value, slippage_pips)
    logger.info("Walk-forward engine: %d signals on %d bars", engine.n_signals, engine.n_bars)

    results: list[WalkForwardResult] = []

    for cand_idx, params_dict in enumerate(candidates):
        # Encode params ONCE per candidate
        param_row = encode_params(engine.encoding, params_dict)
        param_matrix = param_row.reshape(1, -1)

        window_results: list[WindowResult] = []

        # Evaluate on OOS windows
        for i, (win_idx, (w_start, w_end)) in enumerate(oos_windows):
            metrics = engine.evaluate_batch_windowed(
                param_matrix, w_start, w_end, exec_mode=EXEC_FULL,
            )
            wr = _metrics_row_to_window_result(
                metrics[0], w_start, w_end, win_idx, True,
                config.wf_min_trades_per_window,
            )
            window_results.append(wr)

        # Also evaluate on IS windows (needed for WFE calculation)
        is_results: list[WindowResult] = []
        for i, (win_idx, (w_start, w_end)) in enumerate(is_windows):
            metrics = engine.evaluate_batch_windowed(
                param_matrix, w_start, w_end, exec_mode=EXEC_FULL,
            )
            wr = _metrics_row_to_window_result(
                metrics[0], w_start, w_end, win_idx, False,
                config.wf_min_trades_per_window,
            )
            is_results.append(wr)

        # Compute aggregate OOS stats
        n_oos = len(window_results)
        oos_sharpes = [wr.sharpe for wr in window_results]
        oos_qualities = [wr.quality_score for wr in window_results]
        n_passed = sum(1 for wr in window_results if wr.passed)

        pass_rate = n_passed / n_oos if n_oos > 0 else 0.0
        mean_sharpe = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        mean_quality = float(np.mean(oos_qualities)) if oos_qualities else 0.0

        # Geometric mean of quality scores from passed OOS windows
        passed_qualities = [wr.quality_score for wr in window_results if wr.passed]
        if passed_qualities and all(q > 0 for q in passed_qualities):
            log_sum = sum(math.log(q) for q in passed_qualities)
            geo_mean_quality = math.exp(log_sum / len(passed_qualities))
        else:
            geo_mean_quality = 0.0

        # Min quality across all OOS windows
        min_quality = min(oos_qualities) if oos_qualities else 0.0

        # Quality CV (coefficient of variation) across OOS windows
        if oos_qualities and mean_quality != 0:
            quality_std = float(np.std(oos_qualities, ddof=0))
            quality_cv = quality_std / abs(mean_quality)
        else:
            quality_cv = 0.0

        # WFE: mean OOS quality / best IS quality
        if is_results:
            best_is_quality = max(wr.quality_score for wr in is_results)
            wfe = mean_quality / best_is_quality if best_is_quality > 0 else 0.0
        else:
            wfe = 0.0

        # Apply gates
        passed_gate = (
            pass_rate >= config.wf_pass_rate_gate
            and mean_sharpe >= config.wf_mean_sharpe_gate
        )

        # Combine all window results (OOS + IS)
        all_windows = window_results + is_results

        wf_result = WalkForwardResult(
            windows=all_windows,
            n_windows=len(windows),
            n_oos_windows=n_oos,
            n_passed=n_passed,
            pass_rate=pass_rate,
            mean_sharpe=mean_sharpe,
            mean_quality=mean_quality,
            geo_mean_quality=geo_mean_quality,
            min_quality=min_quality,
            quality_cv=quality_cv,
            wfe=wfe,
            passed_gate=passed_gate,
        )
        results.append(wf_result)

        logger.info(
            "Candidate %d: %d/%d OOS windows passed (%.0f%%), "
            "mean Sharpe=%.3f, mean quality=%.3f, WFE=%.3f, gate=%s",
            cand_idx, n_passed, n_oos, pass_rate * 100,
            mean_sharpe, mean_quality, wfe,
            "PASS" if passed_gate else "FAIL",
        )

    return results
