"""Tests verifying that strategies declaring CAUSAL actually are causal.

Two properties are checked for every registered causal strategy:

1. Truncation invariance: signals at bars <= t are identical whether
   computed on [0, N) or [0, t+1).
2. Future perturbation: wildly changing data after bar t does not affect
   signals at bars <= t.

These tests are parameterized across all registered strategies via the
_REGISTRY, so adding a new @register strategy automatically gets tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from backtester.strategies.base import SignalCausality, Strategy
from backtester.strategies.registry import _REGISTRY

# Trigger strategy registration by importing the strategies package
import backtester.strategies  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(
    n_bars: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic FX random-walk OHLCV data for causality tests.

    Returns:
        (open, high, low, close, volume, spread, bar_hour, bar_day_of_week)
    """
    rng = np.random.default_rng(seed)

    # Random walk for close prices
    base = 1.1000
    returns = rng.normal(0, 0.0005, n_bars)
    close = np.empty(n_bars, dtype=np.float64)
    close[0] = base
    for i in range(1, n_bars):
        close[i] = close[i - 1] * (1 + returns[i])

    # Build OHLC from close
    noise = rng.uniform(0.0001, 0.0010, n_bars)
    open_ = np.roll(close, 1)
    open_[0] = base
    high = np.maximum(open_, close) + noise
    low = np.minimum(open_, close) - noise
    volume = rng.uniform(100, 10000, n_bars).astype(np.float64)
    spread = rng.uniform(0.00005, 0.00020, n_bars).astype(np.float64)

    # Time arrays (cycle through hours and weekdays)
    bar_hour = np.array([i % 24 for i in range(n_bars)], dtype=np.int64)
    bar_day_of_week = np.array([(i // 24) % 5 for i in range(n_bars)], dtype=np.int64)

    return open_, high, low, close, volume, spread, bar_hour, bar_day_of_week


def _get_causal_strategies() -> list[pytest.param]:
    """Get all registered strategies that declare CAUSAL, as pytest params."""
    params = []
    for name, cls in sorted(_REGISTRY.items()):
        try:
            instance = cls()
        except TypeError:
            instance = cls.__new__(cls)
        if instance.signal_causality() == SignalCausality.CAUSAL:
            params.append(pytest.param(cls, id=name))
    return params


def _signal_arrays_at_mask(
    sig: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    """Extract signal arrays where mask is True."""
    return {k: v[mask] for k, v in sig.items()}


def _signals_match(
    sig_a: dict[str, np.ndarray],
    sig_b: dict[str, np.ndarray],
    keys: list[str] | None = None,
) -> bool:
    """Check that two signal dicts match on the given keys."""
    if keys is None:
        keys = ["bar_index", "direction", "entry_price", "atr_pips"]
        # Also check variant and filter_value if present in both
        for extra in ("variant", "filter_value"):
            if extra in sig_a and extra in sig_b:
                keys.append(extra)

    for k in keys:
        a = sig_a.get(k)
        b = sig_b.get(k)
        if a is None and b is None:
            continue
        if a is None or b is None:
            return False
        if len(a) != len(b):
            return False
        if a.dtype.kind == 'f':
            if not np.allclose(a, b, rtol=1e-12, atol=1e-15, equal_nan=True):
                return False
        else:
            if not np.array_equal(a, b):
                return False
    return True


# ---------------------------------------------------------------------------
# Test A: Truncation Invariance
# ---------------------------------------------------------------------------

class TestTruncationInvariance:
    """Signals at bars <= t must be identical whether computed on
    [0, N) or [0, t+1)."""

    @pytest.mark.parametrize("strategy_cls", _get_causal_strategies())
    def test_truncation_invariance(self, strategy_cls: type[Strategy]):
        strategy = strategy_cls()
        n_bars = 2000
        open_, high, low, close, volume, spread, bar_hour, bar_dow = (
            _make_synthetic_data(n_bars, seed=123)
        )

        # Full signal computation
        full_sig = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread,
            pip_value=0.0001,
            bar_hour=bar_hour,
            bar_day_of_week=bar_dow,
        )

        if len(full_sig["bar_index"]) == 0:
            pytest.skip(f"{strategy.name} produced 0 signals on synthetic data")

        # Pick 3 truncation points after warmup zone (bar 500+)
        max_sig_bar = int(full_sig["bar_index"].max())
        min_trunc = max(500, int(full_sig["bar_index"].min()) + 1)
        if min_trunc >= max_sig_bar:
            pytest.skip(
                f"{strategy.name}: insufficient signal range for truncation test "
                f"(min_trunc={min_trunc}, max_sig_bar={max_sig_bar})"
            )

        rng = np.random.default_rng(42)
        trunc_points = sorted(rng.integers(min_trunc, min(max_sig_bar, n_bars - 1), size=3))

        for t in trunc_points:
            t = int(t)
            # Compute signals on truncated data [0, t+1)
            trunc_sig = strategy.generate_signals_vectorized(
                open_[:t + 1], high[:t + 1], low[:t + 1], close[:t + 1],
                volume[:t + 1], spread[:t + 1],
                pip_value=0.0001,
                bar_hour=bar_hour[:t + 1],
                bar_day_of_week=bar_dow[:t + 1],
            )

            # Compare signals at bars STRICTLY before the truncation point.
            # We use < t (not <= t) because strategies legitimately skip the
            # last bar of the data (no next bar for trade entry). Causality
            # means signals at interior bars are unaffected by data length.
            full_mask = full_sig["bar_index"] < t
            trunc_mask = trunc_sig["bar_index"] < t

            full_subset = _signal_arrays_at_mask(full_sig, full_mask)
            trunc_subset = _signal_arrays_at_mask(trunc_sig, trunc_mask)

            assert _signals_match(full_subset, trunc_subset), (
                f"{strategy.name}: truncation invariance violated at t={t}. "
                f"Full has {full_mask.sum()} signals < t, "
                f"truncated has {trunc_mask.sum()} signals < t."
            )


# ---------------------------------------------------------------------------
# Test B: Future Perturbation
# ---------------------------------------------------------------------------

class TestFuturePerturbation:
    """Aggressively changing data after bar t must not affect signals at bars <= t."""

    @pytest.mark.parametrize("strategy_cls", _get_causal_strategies())
    def test_future_perturbation(self, strategy_cls: type[Strategy]):
        strategy = strategy_cls()
        n_bars = 2000
        open_, high, low, close, volume, spread, bar_hour, bar_dow = (
            _make_synthetic_data(n_bars, seed=456)
        )

        # Compute original signals
        orig_sig = strategy.generate_signals_vectorized(
            open_, high, low, close, volume, spread,
            pip_value=0.0001,
            bar_hour=bar_hour,
            bar_day_of_week=bar_dow,
        )

        if len(orig_sig["bar_index"]) == 0:
            pytest.skip(f"{strategy.name} produced 0 signals on synthetic data")

        # Pick split point near midpoint of signal range
        sig_bars = orig_sig["bar_index"]
        mid_idx = len(sig_bars) // 2
        t = int(sig_bars[mid_idx])

        # Ensure t is reasonable
        if t < 100 or t >= n_bars - 100:
            t = n_bars // 2

        # Create perturbed copy — wildly different data after bar t
        rng = np.random.default_rng(999)
        open_p = open_.copy()
        high_p = high.copy()
        low_p = low.copy()
        close_p = close.copy()
        volume_p = volume.copy()
        spread_p = spread.copy()

        # Perturb everything after t with completely different values
        n_after = n_bars - t - 1
        if n_after > 0:
            open_p[t + 1:] = 2.0 + rng.uniform(-0.1, 0.1, n_after)
            close_p[t + 1:] = 2.0 + rng.uniform(-0.1, 0.1, n_after)
            high_p[t + 1:] = np.maximum(open_p[t + 1:], close_p[t + 1:]) + rng.uniform(0.01, 0.05, n_after)
            low_p[t + 1:] = np.minimum(open_p[t + 1:], close_p[t + 1:]) - rng.uniform(0.01, 0.05, n_after)
            volume_p[t + 1:] = rng.uniform(50000, 100000, n_after)
            spread_p[t + 1:] = rng.uniform(0.001, 0.005, n_after)

        # Compute signals on perturbed data
        pert_sig = strategy.generate_signals_vectorized(
            open_p, high_p, low_p, close_p, volume_p, spread_p,
            pip_value=0.0001,
            bar_hour=bar_hour,
            bar_day_of_week=bar_dow,
        )

        # Compare signals at bars <= t
        orig_mask = orig_sig["bar_index"] <= t
        pert_mask = pert_sig["bar_index"] <= t

        orig_subset = _signal_arrays_at_mask(orig_sig, orig_mask)
        pert_subset = _signal_arrays_at_mask(pert_sig, pert_mask)

        assert _signals_match(orig_subset, pert_subset), (
            f"{strategy.name}: future perturbation affected signals at bars <= {t}. "
            f"Original has {orig_mask.sum()} signals, "
            f"perturbed has {pert_mask.sum()} signals."
        )
