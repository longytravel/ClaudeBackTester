"""Parameter stability analysis for the validation pipeline (Stage 4).

Tests how sensitive a candidate's performance is to small parameter changes.
For each parameter, perturb +-N steps in the values list, evaluate on
(forward) data, and measure how much quality drops.

Functions:
    generate_perturbations  - Create perturbed parameter dicts for one candidate
    evaluate_stability      - Evaluate stability for one candidate
    run_stability           - Run stability analysis for a list of candidates
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from backtester.core.dtypes import EXEC_FULL, M_QUALITY
from backtester.core.encoding import build_encoding_spec, encode_params
from backtester.core.engine import BacktestEngine
from backtester.pipeline.config import PipelineConfig
from backtester.pipeline.types import PerturbationResult, StabilityRating, StabilityResult
from backtester.strategies.base import ParamSpace, Strategy

logger = logging.getLogger(__name__)


def generate_perturbations(
    params_dict: dict[str, Any],
    param_space: ParamSpace,
    n_steps: int = 3,
) -> list[tuple[str, Any, dict[str, Any]]]:
    """Generate perturbed parameter dicts by shifting each param +-N steps.

    For each parameter in param_space:
    - Numeric: shift index +-1, +-2, ..., +-n_steps in the values list.
      Skip perturbations that would go out of bounds.
    - Boolean: flip the value (exactly one perturbation).
    - Categorical (string values): try each OTHER value in the values list.
    - Bitmask (list-of-lists like allowed_days): try each OTHER value.

    Args:
        params_dict: The original parameter dictionary.
        param_space: The strategy's parameter space.
        n_steps: Number of steps to perturb in each direction (numeric only).

    Returns:
        List of (param_name, perturbed_value, perturbed_params_dict) tuples.
    """
    perturbations: list[tuple[str, Any, dict[str, Any]]] = []

    for pdef in param_space:
        name = pdef.name
        values = pdef.values
        current_value = params_dict.get(name)

        if current_value is None:
            continue

        is_boolean = (
            len(values) == 2
            and not isinstance(values[0], list)
            and set(values) == {True, False}
        )
        is_bitmask = len(values) > 0 and isinstance(values[0], list)
        is_categorical = len(values) > 0 and isinstance(values[0], str)

        if is_boolean:
            # Flip the boolean — exactly one perturbation
            flipped = not current_value
            perturbed = dict(params_dict)
            perturbed[name] = flipped
            perturbations.append((name, flipped, perturbed))

        elif is_categorical or is_bitmask:
            # Try each OTHER value in the values list
            for v in values:
                if v != current_value:
                    perturbed = dict(params_dict)
                    perturbed[name] = v
                    perturbations.append((name, v, perturbed))

        else:
            # Numeric: find current index, shift +-1..n_steps
            try:
                current_idx = values.index(current_value)
            except ValueError:
                # Current value not in values list — try closest match
                logger.warning(
                    "Value %r for param %s not in values list, skipping perturbation",
                    current_value, name,
                )
                continue

            for step in range(1, n_steps + 1):
                # Positive direction
                new_idx = current_idx + step
                if 0 <= new_idx < len(values):
                    perturbed = dict(params_dict)
                    perturbed[name] = values[new_idx]
                    perturbations.append((name, values[new_idx], perturbed))

                # Negative direction
                new_idx = current_idx - step
                if 0 <= new_idx < len(values):
                    perturbed = dict(params_dict)
                    perturbed[name] = values[new_idx]
                    perturbations.append((name, values[new_idx], perturbed))

    return perturbations


def evaluate_stability(
    strategy: Strategy,
    params_dict: dict[str, Any],
    perturbations: list[tuple[str, Any, dict[str, Any]]],
    data_arrays: dict[str, np.ndarray],
    config: PipelineConfig,
    pip_value: float = 0.0001,
    slippage_pips: float = 0.5,
) -> StabilityResult:
    """Evaluate parameter stability for one candidate.

    Creates a BacktestEngine on the provided data, evaluates the original
    params and all perturbations in a single batch, then computes ratios
    and assigns a stability rating.

    Args:
        strategy: Strategy instance.
        params_dict: Original parameter dictionary for the candidate.
        perturbations: List of (param_name, perturbed_value, perturbed_params_dict)
                       from generate_perturbations().
        data_arrays: Dict with keys: open, high, low, close, volume, spread,
                     bar_hour, bar_day_of_week.
        config: Pipeline configuration.

    Returns:
        StabilityResult with perturbation details, ratios, and rating.
    """
    if not perturbations:
        return StabilityResult(
            perturbations=[],
            mean_ratio=1.0,
            min_ratio=1.0,
            worst_param="",
            rating=StabilityRating.ROBUST,
        )

    # Create engine on the evaluation data
    bar_hour = data_arrays.get("bar_hour")
    bar_day_of_week = data_arrays.get("bar_day_of_week")

    # Pass M1 sub-bar arrays if present
    m1_kwargs: dict[str, np.ndarray] = {}
    for key in ("m1_high", "m1_low", "m1_close", "m1_spread",
                "h1_to_m1_start", "h1_to_m1_end"):
        if key in data_arrays:
            m1_kwargs[key] = data_arrays[key]

    engine = BacktestEngine(
        strategy=strategy,
        open_=data_arrays["open"],
        high=data_arrays["high"],
        low=data_arrays["low"],
        close=data_arrays["close"],
        volume=data_arrays["volume"],
        spread=data_arrays["spread"],
        pip_value=pip_value,
        slippage_pips=slippage_pips,
        commission_pips=config.commission_pips,
        max_spread_pips=config.max_spread_pips,
        bar_hour=bar_hour,
        bar_day_of_week=bar_day_of_week,
        **m1_kwargs,
    )

    encoding = build_encoding_spec(strategy.param_space())

    # Build batch: original params (index 0) + all perturbations
    all_dicts = [params_dict] + [p[2] for p in perturbations]
    n_total = len(all_dicts)

    param_matrix = np.zeros((n_total, encoding.num_params), dtype=np.float64)
    for i, d in enumerate(all_dicts):
        param_matrix[i] = encode_params(encoding, d)

    # Evaluate all at once
    metrics = engine.evaluate_batch(param_matrix, exec_mode=EXEC_FULL)

    original_quality = float(metrics[0, M_QUALITY])

    # Build perturbation results
    perturbation_results: list[PerturbationResult] = []
    for i, (param_name, perturbed_value, _) in enumerate(perturbations):
        perturbed_quality = float(metrics[i + 1, M_QUALITY])

        if original_quality <= 0:
            ratio = 0.0
        else:
            ratio = perturbed_quality / original_quality

        perturbation_results.append(PerturbationResult(
            param_name=param_name,
            original_value=params_dict.get(param_name),
            perturbed_value=perturbed_value,
            original_quality=original_quality,
            perturbed_quality=perturbed_quality,
            ratio=ratio,
        ))

    # Compute aggregate statistics
    ratios = [pr.ratio for pr in perturbation_results]
    mean_ratio = float(np.mean(ratios)) if ratios else 0.0
    min_ratio = float(np.min(ratios)) if ratios else 0.0

    # Find worst parameter (lowest ratio)
    worst_idx = int(np.argmin(ratios)) if ratios else 0
    worst_param = perturbation_results[worst_idx].param_name if perturbation_results else ""

    # Assign rating based on thresholds
    if mean_ratio >= config.stab_robust_mean and min_ratio >= config.stab_robust_min:
        rating = StabilityRating.ROBUST
    elif mean_ratio >= config.stab_moderate_mean:
        rating = StabilityRating.MODERATE
    elif mean_ratio >= config.stab_fragile_mean:
        rating = StabilityRating.FRAGILE
    else:
        rating = StabilityRating.OVERFIT

    logger.info(
        "Stability: mean_ratio=%.3f, min_ratio=%.3f, worst_param=%s, rating=%s",
        mean_ratio, min_ratio, worst_param, rating.value,
    )

    return StabilityResult(
        perturbations=perturbation_results,
        mean_ratio=mean_ratio,
        min_ratio=min_ratio,
        worst_param=worst_param,
        rating=rating,
    )


def run_stability(
    strategy: Strategy,
    candidates: list[dict[str, Any]],
    data_arrays: dict[str, np.ndarray],
    config: PipelineConfig | None = None,
    pip_value: float = 0.0001,
    slippage_pips: float = 0.5,
) -> list[StabilityResult]:
    """Run stability analysis for a list of candidates.

    For each candidate, generates perturbations from the strategy's param space,
    evaluates them on the provided data, and returns stability results.

    Args:
        strategy: Strategy instance.
        candidates: List of parameter dictionaries (one per candidate).
        data_arrays: Dict with keys: open, high, low, close, volume, spread,
                     bar_hour, bar_day_of_week.
        config: Pipeline configuration. Uses defaults if None.

    Returns:
        List of StabilityResult, one per candidate.
    """
    if config is None:
        config = PipelineConfig()

    param_space = strategy.param_space()
    results: list[StabilityResult] = []

    for cand_idx, params_dict in enumerate(candidates):
        logger.info("Stability analysis for candidate %d", cand_idx)

        perturbations = generate_perturbations(
            params_dict=params_dict,
            param_space=param_space,
            n_steps=config.stab_perturbation_steps,
        )

        result = evaluate_stability(
            strategy=strategy,
            params_dict=params_dict,
            perturbations=perturbations,
            data_arrays=data_arrays,
            config=config,
            pip_value=pip_value,
            slippage_pips=slippage_pips,
        )
        results.append(result)

    return results
