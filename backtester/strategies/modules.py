"""Reusable management modules for trade exit/management.

Each module is a self-contained component that declares its parameters,
PL slot mapping, and optimization group. Strategies compose modules via
management_modules() — a bug fix or improvement to a module automatically
applies to every strategy that uses it.

Review guide: this file + rust/src/trade_full.rs covers all management logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from backtester.core.rust_loop import (
    PL_BREAKEVEN_ENABLED,
    PL_BREAKEVEN_OFFSET,
    PL_BREAKEVEN_TRIGGER,
    PL_MAX_BARS,
    PL_PARTIAL_ENABLED,
    PL_PARTIAL_PCT,
    PL_PARTIAL_TRIGGER,
    PL_STALE_ATR_THRESH,
    PL_STALE_BARS,
    PL_STALE_ENABLED,
    PL_TRAIL_ACTIVATE,
    PL_TRAIL_ATR_MULT,
    PL_TRAIL_DISTANCE,
    PL_TRAILING_MODE,
)
from backtester.strategies.base import ParamDef


class ManagementModule(ABC):
    """A reusable trade management component.

    Subclass to define a new exit/management technique. Each module:
    - Declares its own ParamDefs with value ranges
    - Maps param names to Rust PL_* slots
    - Declares its optimization group name
    - Can be composed with any strategy via management_modules()
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Module name, e.g., 'trailing_stop'."""
        ...

    @property
    @abstractmethod
    def group(self) -> str:
        """Optimization group name, e.g., 'exit_trailing'."""
        ...

    @property
    def requires_full_mode(self) -> bool:
        """Whether this module needs EXEC_FULL (default True)."""
        return True

    @abstractmethod
    def param_defs(self) -> list[ParamDef]:
        """Return ParamDefs for this module's parameters."""
        ...

    @abstractmethod
    def pl_mapping(self) -> dict[str, int]:
        """Map param names to PL_* slot indices."""
        ...


# ---------------------------------------------------------------------------
# Built-in modules (wrap existing management functionality)
# ---------------------------------------------------------------------------


class TrailingStopModule(ManagementModule):
    """Trailing stop exit — fixed pip or ATR chandelier."""

    @property
    def name(self) -> str:
        return "trailing_stop"

    @property
    def group(self) -> str:
        return "exit_trailing"

    def param_defs(self) -> list[ParamDef]:
        return [
            ParamDef("trailing_mode", ["off", "fixed_pip", "atr_chandelier"], group=self.group),
            ParamDef("trail_activate_pips", [0, 10, 15, 20, 30, 40, 50], group=self.group),
            ParamDef("trail_distance_pips", [5, 10, 15, 20, 30], group=self.group),
            ParamDef("trail_atr_mult", [1.0, 1.5, 2.0, 2.5, 3.0], group=self.group),
        ]

    def pl_mapping(self) -> dict[str, int]:
        return {
            "trailing_mode": PL_TRAILING_MODE,
            "trail_activate_pips": PL_TRAIL_ACTIVATE,
            "trail_distance_pips": PL_TRAIL_DISTANCE,
            "trail_atr_mult": PL_TRAIL_ATR_MULT,
        }


class BreakevenModule(ManagementModule):
    """Breakeven lock — move SL to entry after trigger distance."""

    @property
    def name(self) -> str:
        return "breakeven"

    @property
    def group(self) -> str:
        return "exit_protection"

    def param_defs(self) -> list[ParamDef]:
        return [
            ParamDef("breakeven_enabled", [False, True], group=self.group),
            ParamDef("breakeven_trigger_pips", [7, 10, 15, 20, 30], group=self.group),
            ParamDef("breakeven_offset_pips", [2, 3, 5, 7, 10], group=self.group),
        ]

    def pl_mapping(self) -> dict[str, int]:
        return {
            "breakeven_enabled": PL_BREAKEVEN_ENABLED,
            "breakeven_trigger_pips": PL_BREAKEVEN_TRIGGER,
            "breakeven_offset_pips": PL_BREAKEVEN_OFFSET,
        }


class PartialCloseModule(ManagementModule):
    """Partial close — close a percentage at trigger distance."""

    @property
    def name(self) -> str:
        return "partial_close"

    @property
    def group(self) -> str:
        return "exit_protection"

    def param_defs(self) -> list[ParamDef]:
        return [
            ParamDef("partial_close_enabled", [False, True], group=self.group),
            ParamDef("partial_close_pct", [30, 40, 50, 60, 70], group=self.group),
            ParamDef("partial_close_trigger_pips", [10, 15, 20, 30, 50], group=self.group),
        ]

    def pl_mapping(self) -> dict[str, int]:
        return {
            "partial_close_enabled": PL_PARTIAL_ENABLED,
            "partial_close_pct": PL_PARTIAL_PCT,
            "partial_close_trigger_pips": PL_PARTIAL_TRIGGER,
        }


class MaxBarsModule(ManagementModule):
    """Max bars exit — close trade after N bars."""

    @property
    def name(self) -> str:
        return "max_bars"

    @property
    def group(self) -> str:
        return "exit_time"

    def param_defs(self) -> list[ParamDef]:
        return [
            ParamDef("max_bars", [0, 50, 100, 200, 500, 1000], group=self.group),
        ]

    def pl_mapping(self) -> dict[str, int]:
        return {
            "max_bars": PL_MAX_BARS,
        }


class StaleExitModule(ManagementModule):
    """Stale exit — close trade if price movement below ATR threshold."""

    @property
    def name(self) -> str:
        return "stale_exit"

    @property
    def group(self) -> str:
        return "exit_time"

    def param_defs(self) -> list[ParamDef]:
        return [
            ParamDef("stale_exit_enabled", [False, True], group=self.group),
            ParamDef("stale_exit_bars", [20, 50, 100], group=self.group),
            ParamDef("stale_exit_atr_threshold", [0.3, 0.5, 0.75, 1.0], group=self.group),
        ]

    def pl_mapping(self) -> dict[str, int]:
        return {
            "stale_exit_enabled": PL_STALE_ENABLED,
            "stale_exit_bars": PL_STALE_BARS,
            "stale_exit_atr_threshold": PL_STALE_ATR_THRESH,
        }


# ---------------------------------------------------------------------------
# Default module set
# ---------------------------------------------------------------------------

DEFAULT_MODULES: list[ManagementModule] = [
    TrailingStopModule(),
    BreakevenModule(),
    PartialCloseModule(),
    MaxBarsModule(),
    StaleExitModule(),
]
