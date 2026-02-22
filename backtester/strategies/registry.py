"""Central strategy registry (REQ-S26, REQ-S27).

Maps strategy names (and aliases) to implementations.
Tracks lifecycle stages.
"""

from __future__ import annotations

from typing import Any

import structlog

from backtester.strategies.base import Strategy, StrategyStage

log = structlog.get_logger()

# Global registry
_REGISTRY: dict[str, type[Strategy]] = {}
_ALIASES: dict[str, str] = {}
_STAGES: dict[str, StrategyStage] = {}


def register(
    cls: type[Strategy],
    aliases: list[str] | None = None,
) -> type[Strategy]:
    """Register a strategy class. Can be used as a decorator."""
    # Create a temporary instance to safely read the name property.
    # Use try/except: __new__ alone may not set up state needed by the property,
    # so fall back to full __init__() if needed.
    try:
        instance = cls.__new__(cls)
        name = instance.name
    except (AttributeError, TypeError):
        instance = cls()
        name = instance.name
    _REGISTRY[name] = cls
    _STAGES[name] = StrategyStage.BUILT

    if aliases:
        for alias in aliases:
            _ALIASES[alias] = name

    log.debug("strategy_registered", name=name, aliases=aliases or [])
    return cls


def get(name: str) -> type[Strategy]:
    """Look up a strategy by name or alias (REQ-S26)."""
    canonical = _ALIASES.get(name, name)
    if canonical not in _REGISTRY:
        available = list(_REGISTRY.keys()) + list(_ALIASES.keys())
        raise KeyError(f"Unknown strategy '{name}'. Available: {available}")
    return _REGISTRY[canonical]


def create(name: str, **kwargs: Any) -> Strategy:
    """Create a strategy instance by name."""
    cls = get(name)
    return cls(**kwargs)


def list_strategies() -> list[dict[str, str]]:
    """List all registered strategies with their stages."""
    return [
        {"name": name, "stage": _STAGES.get(name, StrategyStage.BUILT).value}
        for name in _REGISTRY
    ]


def set_stage(name: str, stage: StrategyStage) -> None:
    """Update lifecycle stage for a strategy (REQ-S27)."""
    canonical = _ALIASES.get(name, name)
    if canonical not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'")
    _STAGES[canonical] = stage
    log.info("strategy_stage_changed", name=canonical, stage=stage.value)


def get_stage(name: str) -> StrategyStage:
    """Get current lifecycle stage."""
    canonical = _ALIASES.get(name, name)
    return _STAGES.get(canonical, StrategyStage.BUILT)


def clear() -> None:
    """Clear the registry (for testing)."""
    _REGISTRY.clear()
    _ALIASES.clear()
    _STAGES.clear()
