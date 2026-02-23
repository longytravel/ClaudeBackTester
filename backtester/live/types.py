"""Data types for the live trading engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TradingMode(Enum):
    DRY_RUN = "dry_run"      # Log signals only, no orders
    PRACTICE = "practice"    # Trade on demo account
    LIVE = "live"            # Trade on live account


class ManagementActionType(Enum):
    CLOSE = "close"
    MODIFY_SL = "modify_sl"
    MODIFY_TP = "modify_tp"
    PARTIAL_CLOSE = "partial_close"
    NONE = "none"


@dataclass
class ManagementAction:
    """An action to take on a position, produced by position_manager."""
    action_type: ManagementActionType
    reason: str = ""
    new_sl: float = 0.0
    new_tp: float = 0.0
    close_volume: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class LivePosition:
    """Tracks a live open position with management state."""
    ticket: int
    symbol: str
    direction: int          # 1=BUY, -1=SELL
    volume: float           # current volume (may decrease after partial close)
    original_volume: float
    entry_price: float
    entry_time: datetime
    sl_price: float
    tp_price: float
    pip_value: float        # e.g. 0.0001 for EUR/USD

    # Management state (mirrors telemetry.py)
    current_sl: float = 0.0
    trailing_active: bool = False
    be_locked: bool = False
    partial_done: bool = False
    bars_held: int = 0
    position_pct: float = 1.0
    realized_pnl_pips: float = 0.0

    # Strategy params for this trade's management
    params: dict[str, Any] = field(default_factory=dict)
    atr_pips: float = 0.0

    # Signal bar timestamp for dedup
    signal_bar_time: datetime | None = None

    def __post_init__(self):
        if self.current_sl == 0.0:
            self.current_sl = self.sl_price

    def to_dict(self) -> dict:
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "direction": self.direction,
            "volume": self.volume,
            "original_volume": self.original_volume,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "pip_value": self.pip_value,
            "current_sl": self.current_sl,
            "trailing_active": self.trailing_active,
            "be_locked": self.be_locked,
            "partial_done": self.partial_done,
            "bars_held": self.bars_held,
            "position_pct": self.position_pct,
            "realized_pnl_pips": self.realized_pnl_pips,
            "params": self.params,
            "atr_pips": self.atr_pips,
            "signal_bar_time": self.signal_bar_time.isoformat() if self.signal_bar_time else None,
        }

    @staticmethod
    def from_dict(d: dict) -> LivePosition:
        pos = LivePosition(
            ticket=d["ticket"],
            symbol=d["symbol"],
            direction=d["direction"],
            volume=d["volume"],
            original_volume=d["original_volume"],
            entry_price=d["entry_price"],
            entry_time=datetime.fromisoformat(d["entry_time"]),
            sl_price=d["sl_price"],
            tp_price=d["tp_price"],
            pip_value=d["pip_value"],
            current_sl=d.get("current_sl", d["sl_price"]),
            trailing_active=d.get("trailing_active", False),
            be_locked=d.get("be_locked", False),
            partial_done=d.get("partial_done", False),
            bars_held=d.get("bars_held", 0),
            position_pct=d.get("position_pct", 1.0),
            realized_pnl_pips=d.get("realized_pnl_pips", 0.0),
            params=d.get("params", {}),
            atr_pips=d.get("atr_pips", 0.0),
        )
        sbt = d.get("signal_bar_time")
        if sbt:
            pos.signal_bar_time = datetime.fromisoformat(sbt)
        return pos


@dataclass
class DailyStats:
    """Daily trading statistics for risk management."""
    date: str = ""                # YYYY-MM-DD
    starting_balance: float = 0.0
    trades_opened: int = 0
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0       # in account currency
    net_pnl: float = 0.0         # after commission/swap

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "starting_balance": self.starting_balance,
            "trades_opened": self.trades_opened,
            "trades_closed": self.trades_closed,
            "wins": self.wins,
            "losses": self.losses,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
        }

    @staticmethod
    def from_dict(d: dict) -> DailyStats:
        return DailyStats(**{k: d[k] for k in DailyStats.__dataclass_fields__ if k in d})


@dataclass
class TraderState:
    """Full persisted state for crash recovery."""
    instance_id: str = ""
    positions: dict[int, LivePosition] = field(default_factory=dict)  # ticket -> pos
    daily_stats: DailyStats = field(default_factory=DailyStats)
    circuit_breaker_active: bool = False
    circuit_breaker_reason: str = ""
    last_bar_time: str | None = None       # ISO timestamp of last processed bar
    last_signal_time: str | None = None    # ISO timestamp of last signal (for dedup)
    peak_balance: float = 0.0
    total_errors: int = 0
    last_error: str = ""

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "positions": {str(k): v.to_dict() for k, v in self.positions.items()},
            "daily_stats": self.daily_stats.to_dict(),
            "circuit_breaker_active": self.circuit_breaker_active,
            "circuit_breaker_reason": self.circuit_breaker_reason,
            "last_bar_time": self.last_bar_time,
            "last_signal_time": self.last_signal_time,
            "peak_balance": self.peak_balance,
            "total_errors": self.total_errors,
            "last_error": self.last_error,
        }

    @staticmethod
    def from_dict(d: dict) -> TraderState:
        state = TraderState(
            instance_id=d.get("instance_id", ""),
            circuit_breaker_active=d.get("circuit_breaker_active", False),
            circuit_breaker_reason=d.get("circuit_breaker_reason", ""),
            last_bar_time=d.get("last_bar_time"),
            last_signal_time=d.get("last_signal_time"),
            peak_balance=d.get("peak_balance", 0.0),
            total_errors=d.get("total_errors", 0),
            last_error=d.get("last_error", ""),
        )
        # Reconstruct positions
        for ticket_str, pos_dict in d.get("positions", {}).items():
            pos = LivePosition.from_dict(pos_dict)
            state.positions[int(ticket_str)] = pos
        # Reconstruct daily stats
        ds = d.get("daily_stats")
        if ds:
            state.daily_stats = DailyStats.from_dict(ds)
        return state
