"""Strategy framework: base classes, indicators, SL/TP, registry."""

# Import concrete strategies so @register decorators fire
import backtester.strategies.always_buy  # noqa: F401
import backtester.strategies.ema_crossover  # noqa: F401
import backtester.strategies.rsi_mean_reversion  # noqa: F401

from backtester.strategies.base import (
    Direction,
    ParamDef,
    ParamSpace,
    SLMode,
    SLTPResult,
    Signal,
    SignalCausality,
    Strategy,
    StrategyStage,
    TPMode,
    management_params,
    risk_params,
    time_params,
)
from backtester.strategies.registry import (
    clear as clear_registry,
    create as create_strategy,
    get as get_strategy,
    list_strategies,
    register,
    set_stage,
)

__all__ = [
    "Direction",
    "ParamDef",
    "ParamSpace",
    "SLMode",
    "SLTPResult",
    "Signal",
    "SignalCausality",
    "Strategy",
    "StrategyStage",
    "TPMode",
    "clear_registry",
    "create_strategy",
    "get_strategy",
    "list_strategies",
    "management_params",
    "register",
    "risk_params",
    "set_stage",
    "time_params",
]
