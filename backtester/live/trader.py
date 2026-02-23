"""Main live trading loop.

Connects to MT5, loads strategy + params from pipeline checkpoint,
and runs a candle-close loop: manage positions → generate signals → place orders.
"""

from __future__ import annotations

import json
import math
import signal
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import structlog

from backtester.broker import mt5 as broker_data
from backtester.broker import mt5_orders
from backtester.live.config import LiveConfig
from backtester.live.position_manager import manage_position
from backtester.live.state import append_audit, load_state, save_heartbeat, save_state
from backtester.live.types import (
    DailyStats,
    LivePosition,
    ManagementActionType,
    TraderState,
    TradingMode,
)
from backtester.risk.manager import RiskManager
from backtester.strategies.registry import create as create_strategy

log = structlog.get_logger()

# Timeframe -> minutes per bar
TF_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D": 1440, "W": 10080,
}


class LiveTrader:
    """Runs one strategy on one pair in an event loop."""

    def __init__(self, config: LiveConfig):
        self.config = config
        self.magic = abs(hash(config.instance_id)) % (2**31)
        self.strategy = create_strategy(config.strategy_name)
        self.risk_manager = RiskManager(config)
        self.params: dict[str, Any] = {}
        self.state = TraderState(instance_id=config.instance_id)
        self._running = False
        self._recent_bars: deque[tuple[float, float]] = deque(maxlen=200)
        self._bar_minutes = TF_MINUTES.get(config.timeframe, 60)
        self._pip_value = 0.01 if "JPY" in config.pair else 0.0001

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to MT5, restore state, and enter the trading loop."""
        log.info(
            "trader_starting",
            instance_id=self.config.instance_id,
            strategy=self.config.strategy_name,
            pair=self.config.pair,
            timeframe=self.config.timeframe,
            mode=self.config.mode.value,
            magic=self.magic,
        )

        # Connect to MT5
        if self.config.mode != TradingMode.DRY_RUN:
            ok = broker_data.connect()
            if not ok:
                raise RuntimeError("Failed to connect to MT5")

        # Load strategy params from pipeline checkpoint
        self.params = self._load_params()
        log.info("params_loaded", params=self.params)

        # Restore persisted state
        saved = load_state(self.config.state_dir)
        if saved is not None:
            self.state = saved
            self.risk_manager._circuit_breaker_active = saved.circuit_breaker_active
            self.risk_manager._circuit_breaker_reason = saved.circuit_breaker_reason
            log.info(
                "state_restored",
                positions=len(saved.positions),
                last_bar=saved.last_bar_time,
            )

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self._running = True
        self._run_loop()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        """Sleep until candle close, then run a cycle."""
        consecutive_errors = 0
        max_consecutive_errors = 10

        while self._running:
            wait_secs = self._seconds_until_next_bar()
            log.info("waiting_for_bar", wait_seconds=round(wait_secs, 1))

            # Sleep in chunks so we can respond to SIGINT
            sleep_until = time.time() + wait_secs
            while time.time() < sleep_until and self._running:
                time.sleep(min(1.0, sleep_until - time.time()))

            if not self._running:
                break

            try:
                self._cycle()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                self.state.total_errors += 1
                self.state.last_error = str(e)
                log.error(
                    "cycle_error",
                    error=str(e),
                    consecutive=consecutive_errors,
                    exc_info=True,
                )
                # Exponential backoff
                if consecutive_errors >= max_consecutive_errors:
                    log.critical("too_many_errors", count=consecutive_errors)
                    self.risk_manager.trip_circuit_breaker(
                        f"{consecutive_errors} consecutive errors"
                    )
                backoff = min(300, 2 ** consecutive_errors)
                time.sleep(backoff)

        # Save final state on exit
        self._save_all()
        if self.config.mode != TradingMode.DRY_RUN:
            broker_data.disconnect()
        log.info("trader_stopped")

    def _cycle(self) -> None:
        """One trading cycle: manage existing, sync, signal, order."""
        now = datetime.now(tz=timezone.utc)
        log.info("cycle_start", time=now.isoformat())

        # 1. Daily stats rollover
        today_str = now.strftime("%Y-%m-%d")
        if self.state.daily_stats.date != today_str:
            self._daily_rollover(today_str)

        # 2. Fetch candles from MT5
        if self.config.mode == TradingMode.DRY_RUN:
            log.info("dry_run_skip_candles")
            return

        candles = broker_data.fetch_candles(
            self.config.pair,
            self.config.timeframe,
            self.config.lookback_bars,
        )
        if candles.empty:
            log.warning("no_candles_received")
            return

        latest_bar_time = str(candles.index[-1])

        # Dedup: skip if we already processed this bar
        if self.state.last_bar_time == latest_bar_time:
            log.debug("bar_already_processed", bar=latest_bar_time)
            return

        # Update recent bars for stale exit
        for _, row in candles.tail(10).iterrows():
            self._recent_bars.append((row["high"], row["low"]))

        latest = candles.iloc[-1]
        bar_high = float(latest["high"])
        bar_low = float(latest["low"])
        bar_close = float(latest["close"])

        # 3. Manage existing positions
        for ticket, pos in list(self.state.positions.items()):
            actions = manage_position(
                pos, bar_high, bar_low, bar_close, self._recent_bars,
            )
            for action in actions:
                self._execute_action(pos, action)

        # 4. Sync with broker — detect SL/TP hits
        self._sync_with_broker()

        # 5. Generate signals on latest bar
        open_arr = candles["open"].values
        high_arr = candles["high"].values
        low_arr = candles["low"].values
        close_arr = candles["close"].values
        volume_arr = candles["volume"].values
        # Spread: MT5 gives spread_points, convert to price units
        symbol_info = broker_data.get_symbol_info(self.config.pair)
        point = symbol_info["point"] if symbol_info else 0.00001
        spread_arr = candles["spread_points"].values * point

        signals = self.strategy.generate_signals(
            open_arr, high_arr, low_arr, close_arr, volume_arr, spread_arr,
        )

        # Filter signals — only take the last bar's signal
        filtered = self.strategy.filter_signals(signals, self.params)
        last_bar_idx = len(candles) - 1
        new_signals = [s for s in filtered if s.bar_index == last_bar_idx]

        # Dedup by signal bar time
        if self.state.last_signal_time == latest_bar_time:
            new_signals = []

        # 6. Risk checks + place orders
        for sig in new_signals:
            self._process_signal(sig, high_arr, low_arr, symbol_info, latest_bar_time)

        # 7. Update state
        self.state.last_bar_time = latest_bar_time
        if new_signals:
            self.state.last_signal_time = latest_bar_time

        self._save_all()
        log.info(
            "cycle_complete",
            positions=len(self.state.positions),
            signals_found=len(new_signals),
        )

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    def _process_signal(
        self,
        sig,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        symbol_info: dict | None,
        bar_time: str,
    ) -> None:
        """Run risk checks and place order for a signal."""
        from backtester.strategies.base import Direction

        direction = 1 if sig.direction == Direction.BUY else -1

        # Get current tick for spread check
        tick = mt5_orders.get_current_tick(self.config.pair)
        spread_pips = tick["spread_pips"] if tick else 0.0

        # Get account info for balance
        acct = broker_data.get_account_info()
        if acct is None:
            log.error("no_account_info")
            return
        balance = acct["balance"]

        # Update peak balance
        if balance > self.state.peak_balance:
            self.state.peak_balance = balance

        # Pre-trade risk check
        check = self.risk_manager.check_pre_trade(
            self.state.daily_stats,
            self.state.positions,
            balance,
            self.state.peak_balance,
            spread_pips,
            self.config.pair,
        )
        if not check.allowed:
            log.info("signal_rejected", reason=check.reason)
            append_audit(self.config.state_dir, {
                "event": "signal_rejected",
                "reason": check.reason,
                "direction": "BUY" if direction == 1 else "SELL",
                "bar_time": bar_time,
            })
            return

        # Compute SL/TP
        sltp = self.strategy.calc_sl_tp(sig, self.params, high_arr, low_arr)
        sl_pips = sltp.sl_pips

        # Position sizing
        if symbol_info is None:
            log.error("no_symbol_info")
            return
        volume = self.risk_manager.calculate_position_size(balance, sl_pips, symbol_info)
        if volume <= 0:
            log.warning("zero_volume", sl_pips=sl_pips)
            return

        if self.config.mode == TradingMode.DRY_RUN:
            log.info(
                "dry_run_order",
                direction="BUY" if direction == 1 else "SELL",
                volume=volume,
                sl=sltp.sl_price,
                tp=sltp.tp_price,
            )
            return

        # Place the order
        result = mt5_orders.place_market_order(
            symbol=self.config.pair,
            direction=direction,
            volume=volume,
            sl=sltp.sl_price,
            tp=sltp.tp_price,
            magic=self.magic,
            comment=f"{self.config.strategy_name}",
        )

        if result.success:
            pos = LivePosition(
                ticket=result.ticket,
                symbol=self.config.pair,
                direction=direction,
                volume=volume,
                original_volume=volume,
                entry_price=result.price,
                entry_time=datetime.now(tz=timezone.utc),
                sl_price=sltp.sl_price,
                tp_price=sltp.tp_price,
                pip_value=self._pip_value,
                params=self.params,
                atr_pips=sig.atr_pips,
                signal_bar_time=datetime.fromisoformat(bar_time) if bar_time else None,
            )
            self.state.positions[result.ticket] = pos
            self.state.daily_stats.trades_opened += 1

            append_audit(self.config.state_dir, {
                "event": "order_placed",
                "ticket": result.ticket,
                "direction": "BUY" if direction == 1 else "SELL",
                "volume": volume,
                "price": result.price,
                "sl": sltp.sl_price,
                "tp": sltp.tp_price,
                "bar_time": bar_time,
            })
        else:
            append_audit(self.config.state_dir, {
                "event": "order_failed",
                "retcode": result.retcode,
                "comment": result.comment,
                "bar_time": bar_time,
            })

    # ------------------------------------------------------------------
    # Position management execution
    # ------------------------------------------------------------------

    def _execute_action(self, pos: LivePosition, action) -> None:
        """Execute a management action against MT5."""
        if self.config.mode == TradingMode.DRY_RUN:
            log.info("dry_run_action", action=action.action_type.value, ticket=pos.ticket)
            return

        if action.action_type == ManagementActionType.CLOSE:
            result = mt5_orders.close_position(
                pos.ticket, pos.symbol, pos.direction, pos.volume, self.magic,
            )
            if result.success:
                self._record_close(pos, action.reason, result.price)

        elif action.action_type == ManagementActionType.MODIFY_SL:
            mt5_orders.modify_sl_tp(pos.ticket, pos.symbol, action.new_sl, action.new_tp)

        elif action.action_type == ManagementActionType.PARTIAL_CLOSE:
            result = mt5_orders.partial_close(
                pos.ticket, pos.symbol, pos.direction, action.close_volume, self.magic,
            )
            if result.success:
                append_audit(self.config.state_dir, {
                    "event": "partial_close",
                    "ticket": pos.ticket,
                    "volume_closed": action.close_volume,
                    "reason": action.reason,
                })

    def _record_close(self, pos: LivePosition, reason: str, exit_price: float) -> None:
        """Record a position close in stats and audit."""
        pip = pos.pip_value
        if pos.direction == 1:
            pnl_pips = (exit_price - pos.entry_price) / pip * pos.position_pct
        else:
            pnl_pips = (pos.entry_price - exit_price) / pip * pos.position_pct
        pnl_pips += pos.realized_pnl_pips

        self.state.daily_stats.trades_closed += 1
        if pnl_pips >= 0:
            self.state.daily_stats.wins += 1
        else:
            self.state.daily_stats.losses += 1

        # Remove from tracked positions
        self.state.positions.pop(pos.ticket, None)

        append_audit(self.config.state_dir, {
            "event": "position_closed",
            "ticket": pos.ticket,
            "reason": reason,
            "exit_price": exit_price,
            "pnl_pips": round(pnl_pips, 2),
            "bars_held": pos.bars_held,
        })
        log.info(
            "position_closed",
            ticket=pos.ticket,
            reason=reason,
            pnl_pips=round(pnl_pips, 2),
        )

    # ------------------------------------------------------------------
    # Broker sync
    # ------------------------------------------------------------------

    def _sync_with_broker(self) -> None:
        """Compare local positions with MT5. Detect SL/TP/external closes."""
        if self.config.mode == TradingMode.DRY_RUN:
            return

        broker_positions = mt5_orders.get_open_positions(magic=self.magic)
        broker_tickets = {p["ticket"] for p in broker_positions}

        # Find positions we track but broker no longer has (= closed by SL/TP/external)
        for ticket in list(self.state.positions.keys()):
            if ticket not in broker_tickets:
                pos = self.state.positions[ticket]
                log.info("position_closed_by_broker", ticket=ticket)

                # Try to get exit details from deal history
                deals = mt5_orders.get_closed_deals(magic=self.magic)
                exit_price = pos.entry_price  # fallback
                reason = "broker_close"
                for deal in reversed(deals):
                    if deal.get("position_id") == ticket:
                        exit_price = deal["price"]
                        comment = deal.get("comment", "")
                        if "sl" in comment.lower():
                            reason = "stop_loss"
                        elif "tp" in comment.lower():
                            reason = "take_profit"
                        break

                self._record_close(pos, reason, exit_price)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _seconds_until_next_bar(self) -> float:
        """Calculate seconds until the next bar close.

        Aligns to timeframe boundaries (e.g., H1 bars close at :00).
        """
        now = datetime.now(tz=timezone.utc)
        minutes = self._bar_minutes

        # Current bar's start: floor to nearest timeframe boundary
        total_minutes = now.hour * 60 + now.minute
        bar_start_minutes = (total_minutes // minutes) * minutes
        next_bar_minutes = bar_start_minutes + minutes

        next_bar = now.replace(
            hour=0, minute=0, second=0, microsecond=0,
        ) + timedelta(minutes=next_bar_minutes)

        # Add a small buffer (5 seconds) to ensure the bar is fully closed
        wait = (next_bar - now).total_seconds() + 5

        if wait <= 0:
            wait = minutes * 60  # full bar period
        return wait

    def _daily_rollover(self, today_str: str) -> None:
        """Reset daily stats for a new trading day."""
        log.info("daily_rollover", date=today_str)
        acct = broker_data.get_account_info() if self.config.mode != TradingMode.DRY_RUN else None
        balance = acct["balance"] if acct else 0.0

        self.state.daily_stats = DailyStats(
            date=today_str,
            starting_balance=balance,
        )
        # Reset circuit breaker on new day (if triggered by daily loss)
        if self.risk_manager.circuit_breaker_active:
            reason = self.risk_manager.circuit_breaker_reason
            if "Daily loss" in reason:
                self.risk_manager.reset_circuit_breaker()

    def _load_params(self) -> dict[str, Any]:
        """Load optimized params from pipeline checkpoint JSON."""
        path = self.config.pipeline_json
        if not path:
            log.warning("no_pipeline_json, using strategy defaults")
            return self.strategy.param_space().random_sample()

        with open(path) as f:
            data = json.load(f)

        candidates = data.get("candidates", [])
        idx = self.config.candidate_index
        if idx >= len(candidates):
            raise ValueError(
                f"Candidate index {idx} out of range (have {len(candidates)} candidates)"
            )

        params = candidates[idx].get("params", {})
        if not params:
            raise ValueError(f"Candidate {idx} has no params")
        return params

    def _save_all(self) -> None:
        """Save state + heartbeat."""
        self.state.circuit_breaker_active = self.risk_manager.circuit_breaker_active
        self.state.circuit_breaker_reason = self.risk_manager.circuit_breaker_reason

        save_state(self.state, self.config.state_dir)
        save_heartbeat(self.config.state_dir, {
            "instance_id": self.config.instance_id,
            "mode": self.config.mode.value,
            "positions": len(self.state.positions),
            "daily_trades": self.state.daily_stats.trades_opened,
            "circuit_breaker": self.risk_manager.circuit_breaker_active,
            "last_bar": self.state.last_bar_time,
            "errors": self.state.total_errors,
        })

    def _handle_shutdown(self, signum, frame) -> None:
        """Graceful shutdown on SIGINT/SIGTERM."""
        log.info("shutdown_signal", signal=signum)
        self._running = False
