"""Strategy base class and parameter space definitions (REQ-S01 through REQ-S13).

Strategies implement the precompute-once, filter-many pattern:
1. generate_signals() — expensive, runs ONCE per dataset
2. filter_signals() — cheap, runs ONCE PER TRIAL during optimization
3. calc_sl_tp() — compute SL/TP for each signal given parameters
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Parameter Space
# ---------------------------------------------------------------------------

@dataclass
class ParamDef:
    """A single parameter definition with allowed values."""
    name: str
    values: list[Any]
    group: str = "signal"

    def random_value(self, rng: np.random.Generator | None = None) -> Any:
        rng = rng or np.random.default_rng()
        return self.values[rng.integers(0, len(self.values))]


class ParamSpace:
    """Parameter space for optimizer search (REQ-S02, REQ-S11)."""

    def __init__(self, params: list[ParamDef] | None = None):
        self._params: dict[str, ParamDef] = {}
        if params:
            for p in params:
                self._params[p.name] = p

    def add(self, name: str, values: list[Any], group: str = "signal") -> None:
        self._params[name] = ParamDef(name=name, values=values, group=group)

    def get(self, name: str) -> ParamDef:
        return self._params[name]

    @property
    def names(self) -> list[str]:
        return list(self._params.keys())

    @property
    def groups(self) -> dict[str, list[str]]:
        """Group parameter names by their group."""
        out: dict[str, list[str]] = {}
        for p in self._params.values():
            out.setdefault(p.group, []).append(p.name)
        return out

    def random_sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """Sample a random parameter set."""
        rng = rng or np.random.default_rng()
        return {name: p.random_value(rng) for name, p in self._params.items()}

    def total_combinations(self) -> int:
        result = 1
        for p in self._params.values():
            result *= len(p.values)
        return result

    def __len__(self) -> int:
        return len(self._params)

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __iter__(self):
        return iter(self._params.values())


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

class Direction(Enum):
    BUY = 1
    SELL = -1


@dataclass
class Signal:
    """A single entry signal (REQ-S03, REQ-S09, REQ-S10)."""
    bar_index: int
    direction: Direction
    entry_price: float
    hour: int
    day_of_week: int  # 0=Monday, 6=Sunday
    atr_pips: float  # mandatory (REQ-S09)
    attrs: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SL/TP Modes (REQ-S14 through REQ-S17)
# ---------------------------------------------------------------------------

class SLMode(Enum):
    FIXED_PIPS = "fixed_pips"
    ATR_BASED = "atr_based"
    SWING = "swing"


class TPMode(Enum):
    RR_RATIO = "rr_ratio"
    ATR_BASED = "atr_based"
    FIXED_PIPS = "fixed_pips"


@dataclass
class SLTPResult:
    """Computed SL and TP price levels for a trade."""
    sl_price: float
    tp_price: float
    sl_pips: float
    tp_pips: float


# ---------------------------------------------------------------------------
# Standard Parameter Groups (REQ-S12, REQ-S13)
# ---------------------------------------------------------------------------

def risk_params() -> list[ParamDef]:
    """Standard Risk parameter group — shared across all strategies."""
    return [
        ParamDef("sl_mode", ["fixed_pips", "atr_based", "swing"], group="risk"),
        ParamDef("sl_fixed_pips", list(range(10, 101, 5)), group="risk"),
        ParamDef("sl_atr_mult", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0], group="risk"),
        ParamDef("tp_mode", ["rr_ratio", "atr_based", "fixed_pips"], group="risk"),
        ParamDef("tp_rr_ratio", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], group="risk"),
        ParamDef("tp_atr_mult", [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], group="risk"),
        ParamDef("tp_fixed_pips", list(range(10, 201, 10)), group="risk"),
    ]


def management_params() -> list[ParamDef]:
    """Standard Management parameter group (REQ-S13: all default to OFF)."""
    return [
        # Trailing stop (REQ-S18)
        ParamDef("trailing_mode", ["off", "fixed_pip", "atr_chandelier"], group="management"),
        ParamDef("trail_activate_pips", [0, 10, 15, 20, 30, 40, 50], group="management"),
        ParamDef("trail_distance_pips", [5, 10, 15, 20, 30], group="management"),
        ParamDef("trail_atr_mult", [1.0, 1.5, 2.0, 2.5, 3.0], group="management"),
        # Breakeven (REQ-S19)
        ParamDef("breakeven_enabled", [False, True], group="management"),
        ParamDef("breakeven_trigger_pips", [5, 10, 15, 20, 30], group="management"),
        ParamDef("breakeven_offset_pips", [0, 1, 2, 3, 5], group="management"),
        # Partial close (REQ-S20)
        ParamDef("partial_close_enabled", [False, True], group="management"),
        ParamDef("partial_close_pct", [30, 40, 50, 60, 70], group="management"),
        ParamDef("partial_close_trigger_pips", [10, 15, 20, 30, 50], group="management"),
        # Max bars exit (REQ-S21)
        ParamDef("max_bars", [0, 50, 100, 200, 500, 1000], group="management"),
        # Stale exit (REQ-S22)
        ParamDef("stale_exit_enabled", [False, True], group="management"),
        ParamDef("stale_exit_bars", [20, 50, 100], group="management"),
        ParamDef("stale_exit_atr_threshold", [0.3, 0.5, 0.75, 1.0], group="management"),
    ]


def time_params() -> list[ParamDef]:
    """Standard Time parameter group — session/day filters."""
    return [
        ParamDef("allowed_hours_start", list(range(0, 24)), group="time"),
        ParamDef("allowed_hours_end", list(range(0, 24)), group="time"),
        ParamDef("allowed_days", [
            [0, 1, 2, 3, 4],           # Mon-Fri (default)
            [0, 1, 2, 3],              # Mon-Thu
            [1, 2, 3],                 # Tue-Thu
            [0, 1, 2, 3, 4, 5, 6],    # All days
        ], group="time"),
    ]


# ---------------------------------------------------------------------------
# Strategy Lifecycle (REQ-S27)
# ---------------------------------------------------------------------------

class StrategyStage(Enum):
    BUILT = "built"
    VALIDATED = "validated"
    PIPELINE_RUN = "pipeline_run"
    REFINED = "refined"
    LIVE = "live"
    ARCHIVED = "archived"


# ---------------------------------------------------------------------------
# Strategy Base Class (REQ-S01 through REQ-S08)
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """Abstract base class for all trading strategies.

    Implements precompute-once, filter-many pattern (REQ-S06):
    - generate_signals(): Expensive. Runs ONCE per dataset.
    - filter_signals(): Cheap. Runs per-trial during optimization.
    - calc_sl_tp(): Compute SL/TP for each filtered signal.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name (REQ-S01)."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Strategy version (REQ-S01)."""
        ...

    @abstractmethod
    def param_space(self) -> ParamSpace:
        """Define the full parameter space (REQ-S02).

        Should include strategy-specific signal/filter params plus
        standard risk/management/time params via risk_params(), etc.
        """
        ...

    @abstractmethod
    def generate_signals(
        self,
        open: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        spread: np.ndarray,
    ) -> list[Signal]:
        """Generate all possible entry signals from price data (REQ-S03).

        This is the EXPENSIVE step — runs ONCE per dataset. Compute all
        indicators and detect all potential entries.
        """
        ...

    @abstractmethod
    def filter_signals(
        self,
        signals: list[Signal],
        params: dict[str, Any],
    ) -> list[Signal]:
        """Filter signals by parameter thresholds (REQ-S04).

        This is the CHEAP step — runs per-trial. Must NOT recalculate
        indicators. Only applies parameter-based boolean masks.
        """
        ...

    @abstractmethod
    def calc_sl_tp(
        self,
        signal: Signal,
        params: dict[str, Any],
        high: np.ndarray,
        low: np.ndarray,
    ) -> SLTPResult:
        """Calculate SL and TP prices for a signal (REQ-S05)."""
        ...

    # --- Optimization support ---

    def optimization_stages(self) -> list[str]:
        """Define the optimization stage order for this strategy.

        Each stage optimizes one parameter group at a time, locking
        best values from prior stages. Override to customize.

        Default: signal → time → risk → management
        Any param group NOT listed is included in the final refinement stage.
        """
        return ["signal", "time", "risk", "management"]

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate a parameter combination, returning list of error strings.

        Override to add strategy-specific validation rules.
        Empty list = valid. Non-empty = invalid (with reasons).
        """
        errors: list[str] = []

        # Check breakeven offset < trigger
        be_enabled = params.get("breakeven_enabled", False)
        if be_enabled:
            trigger = params.get("breakeven_trigger_pips", 20)
            offset = params.get("breakeven_offset_pips", 0)
            if offset >= trigger:
                errors.append(f"breakeven_offset ({offset}) >= breakeven_trigger ({trigger})")

        return errors

    # --- Vectorized fast path (REQ-S08) ---

    def generate_signals_vectorized(
        self,
        open: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
        spread: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Vectorized signal generation returning numpy arrays.

        Override this for 10-50x speedup over list-of-Signal approach.
        Returns dict with keys: bar_index, direction, entry_price,
        hour, day_of_week, atr_pips, plus any strategy-specific attrs.

        Default implementation wraps generate_signals().
        """
        signals = self.generate_signals(open, high, low, close, volume, spread)
        if not signals:
            return {
                "bar_index": np.array([], dtype=np.int64),
                "direction": np.array([], dtype=np.int64),
                "entry_price": np.array([], dtype=np.float64),
                "hour": np.array([], dtype=np.int64),
                "day_of_week": np.array([], dtype=np.int64),
                "atr_pips": np.array([], dtype=np.float64),
            }
        result = {
            "bar_index": np.array([s.bar_index for s in signals], dtype=np.int64),
            "direction": np.array([s.direction.value for s in signals], dtype=np.int64),
            "entry_price": np.array([s.entry_price for s in signals], dtype=np.float64),
            "hour": np.array([s.hour for s in signals], dtype=np.int64),
            "day_of_week": np.array([s.day_of_week for s in signals], dtype=np.int64),
            "atr_pips": np.array([s.atr_pips for s in signals], dtype=np.float64),
        }
        # Propagate strategy-specific attrs into the vectorized dict (REQ-S10)
        all_keys: set[str] = set()
        for s in signals:
            all_keys.update(s.attrs.keys())
        for key in all_keys:
            result[f"attr_{key}"] = np.array(
                [s.attrs.get(key, np.nan) for s in signals], dtype=np.float64,
            )
        return result

    def filter_signals_vectorized(
        self,
        signals: dict[str, np.ndarray],
        params: dict[str, Any],
    ) -> np.ndarray:
        """Vectorized signal filtering returning boolean mask (REQ-S08).

        Override this for fast-path filtering using numpy boolean operations.
        Returns boolean mask array (True = signal passes filter).

        Default implementation wraps filter_signals().
        """
        # Reconstruct Signal objects (including attrs) for the default path
        n = len(signals["bar_index"])
        attr_keys = [k[5:] for k in signals if k.startswith("attr_")]
        sig_list = []
        for i in range(n):
            attrs = {k: float(signals[f"attr_{k}"][i]) for k in attr_keys}
            sig_list.append(Signal(
                bar_index=int(signals["bar_index"][i]),
                direction=Direction(int(signals["direction"][i])),
                entry_price=float(signals["entry_price"][i]),
                hour=int(signals["hour"][i]),
                day_of_week=int(signals["day_of_week"][i]),
                atr_pips=float(signals["atr_pips"][i]),
                attrs=attrs,
            ))
        filtered = self.filter_signals(sig_list, params)
        filtered_indices = {s.bar_index for s in filtered}
        return np.array([signals["bar_index"][i] in filtered_indices for i in range(n)])
