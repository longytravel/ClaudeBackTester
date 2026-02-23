"""Risk management for live trading (FR-7).

Pre-trade checks, position sizing, and circuit breaker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import structlog

from backtester.live.config import LiveConfig
from backtester.live.types import DailyStats, LivePosition

log = structlog.get_logger()


@dataclass
class RiskCheckResult:
    """Result of a pre-trade risk check."""
    allowed: bool
    reason: str = ""


class RiskManager:
    """Manages risk limits and position sizing."""

    def __init__(self, config: LiveConfig):
        self.config = config
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""

    def check_pre_trade(
        self,
        daily_stats: DailyStats,
        open_positions: dict[int, LivePosition],
        balance: float,
        peak_balance: float,
        spread_pips: float,
        symbol: str,
    ) -> RiskCheckResult:
        """Run all pre-trade risk checks in order. Returns first failure."""
        # 1. Circuit breaker (REQ-R10)
        if self._circuit_breaker_active:
            return RiskCheckResult(
                allowed=False,
                reason=f"Circuit breaker active: {self._circuit_breaker_reason}",
            )

        # 2. Daily trade limit (REQ-R05)
        if daily_stats.trades_opened >= self.config.max_daily_trades:
            return RiskCheckResult(
                allowed=False,
                reason=f"Daily trade limit reached: {daily_stats.trades_opened}/{self.config.max_daily_trades}",
            )

        # 3. Daily loss limit (REQ-R06)
        if daily_stats.starting_balance > 0:
            daily_loss_pct = abs(min(0, daily_stats.net_pnl)) / daily_stats.starting_balance * 100
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                self.trip_circuit_breaker(
                    f"Daily loss {daily_loss_pct:.1f}% >= {self.config.max_daily_loss_pct}%"
                )
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Daily loss limit: {daily_loss_pct:.1f}% >= {self.config.max_daily_loss_pct}%",
                )

        # 4. Drawdown limit (REQ-R08)
        if peak_balance > 0:
            dd_pct = (peak_balance - balance) / peak_balance * 100
            if dd_pct >= self.config.max_drawdown_pct:
                self.trip_circuit_breaker(
                    f"Drawdown {dd_pct:.1f}% >= {self.config.max_drawdown_pct}%"
                )
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Max drawdown: {dd_pct:.1f}% >= {self.config.max_drawdown_pct}%",
                )

        # 5. Max open positions (REQ-R09)
        if len(open_positions) >= self.config.max_open_positions:
            return RiskCheckResult(
                allowed=False,
                reason=f"Max open positions: {len(open_positions)}/{self.config.max_open_positions}",
            )

        # 6. Pair occupancy — only one position per symbol
        for pos in open_positions.values():
            if pos.symbol == symbol:
                return RiskCheckResult(
                    allowed=False,
                    reason=f"Already have open position on {symbol} (ticket {pos.ticket})",
                )

        # 7. Spread filter (REQ-R07)
        if self.config.max_spread_pips > 0 and spread_pips > self.config.max_spread_pips:
            return RiskCheckResult(
                allowed=False,
                reason=f"Spread too wide: {spread_pips:.1f} > {self.config.max_spread_pips} pips",
            )

        return RiskCheckResult(allowed=True)

    def calculate_position_size(
        self,
        balance: float,
        sl_pips: float,
        symbol_info: dict,
    ) -> float:
        """Calculate position size in lots based on risk percentage.

        lots = (balance * risk_pct / 100) / (sl_pips * pip_value_per_lot)
        Then round to volume_step and clamp to [volume_min, volume_max].
        """
        if sl_pips <= 0:
            log.warning("zero_sl_pips", sl_pips=sl_pips)
            return 0.0

        # pip_value_per_lot: for standard lot (100k units), 1 pip = $10 for most pairs
        # contract_size is typically 100000
        contract_size = symbol_info.get("trade_contract_size", 100000)
        point = symbol_info.get("point", 0.00001)
        digits = symbol_info.get("digits", 5)

        # pip size in price: 0.0001 for 5-digit, 0.01 for 3-digit
        pip_price = point * 10 if digits in (3, 5) else point

        # Value of 1 pip for 1 lot
        pip_value_per_lot = pip_price * contract_size

        risk_amount = balance * self.config.risk_pct / 100.0
        raw_lots = risk_amount / (sl_pips * pip_value_per_lot)

        # Round to volume_step
        volume_step = symbol_info.get("volume_step", 0.01)
        volume_min = symbol_info.get("volume_min", 0.01)
        volume_max = symbol_info.get("volume_max", 100.0)

        lots = math.floor(raw_lots / volume_step) * volume_step
        lots = round(lots, 8)  # avoid float artifacts
        lots = max(volume_min, min(volume_max, lots))

        log.info(
            "position_size",
            balance=balance,
            risk_pct=self.config.risk_pct,
            sl_pips=sl_pips,
            pip_value_per_lot=pip_value_per_lot,
            raw_lots=raw_lots,
            lots=lots,
        )
        return lots

    def trip_circuit_breaker(self, reason: str) -> None:
        """Activate circuit breaker (REQ-R10)."""
        self._circuit_breaker_active = True
        self._circuit_breaker_reason = reason
        log.warning("circuit_breaker_tripped", reason=reason)

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (REQ-R11)."""
        self._circuit_breaker_active = False
        self._circuit_breaker_reason = ""
        log.info("circuit_breaker_reset")

    @property
    def circuit_breaker_active(self) -> bool:
        return self._circuit_breaker_active

    @property
    def circuit_breaker_reason(self) -> str:
        return self._circuit_breaker_reason

    def get_status(self) -> dict:
        """Get current risk manager status (REQ-R13)."""
        return {
            "circuit_breaker_active": self._circuit_breaker_active,
            "circuit_breaker_reason": self._circuit_breaker_reason,
            "risk_pct": self.config.risk_pct,
            "max_daily_trades": self.config.max_daily_trades,
            "max_daily_loss_pct": self.config.max_daily_loss_pct,
            "max_drawdown_pct": self.config.max_drawdown_pct,
            "max_open_positions": self.config.max_open_positions,
            "max_spread_pips": self.config.max_spread_pips,
        }
