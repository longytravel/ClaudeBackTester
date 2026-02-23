"""Live position management — mirrors telemetry.py check order exactly.

Check order (telemetry.py lines 273-367):
1. Max bars exit
2. Stale exit (low ATR range)
3. Floating PnL calculation
4. Breakeven lock
5. Trailing stop update
6. Partial close
7. SL/TP — handled by broker (detected in sync)
"""

from __future__ import annotations

from collections import deque

import structlog

from backtester.live.types import (
    LivePosition,
    ManagementAction,
    ManagementActionType,
)

log = structlog.get_logger()


def manage_position(
    pos: LivePosition,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    recent_bars: deque | None = None,
) -> list[ManagementAction]:
    """Check all management rules for one position on a new bar.

    Args:
        pos: The live position with current management state.
        bar_high: High of the latest closed bar.
        bar_low: Low of the latest closed bar.
        bar_close: Close of the latest closed bar.
        recent_bars: Deque of recent (high, low) tuples for stale exit check.

    Returns:
        List of actions to execute. The caller (trader.py) is responsible
        for executing these against MT5.

    Side effects:
        Mutates pos in-place (bars_held, trailing_active, be_locked, etc.)
        to keep management state up to date.
    """
    actions: list[ManagementAction] = []
    params = pos.params
    pip = pos.pip_value
    is_buy = pos.direction == 1

    pos.bars_held += 1

    # --- Floating PnL (in pips) ---
    if is_buy:
        float_pnl = (bar_high - pos.entry_price) / pip  # MFE-like
    else:
        float_pnl = (pos.entry_price - bar_low) / pip

    # ----------------------------------------------------------------
    # 1. Max bars exit
    # ----------------------------------------------------------------
    max_bars = params.get("max_bars", 0)
    if max_bars > 0 and pos.bars_held >= max_bars:
        actions.append(ManagementAction(
            action_type=ManagementActionType.CLOSE,
            reason="max_bars",
            details={"bars_held": pos.bars_held, "max_bars": max_bars},
        ))
        log.info("exit_max_bars", ticket=pos.ticket, bars_held=pos.bars_held)
        return actions  # Exit — no further checks

    # ----------------------------------------------------------------
    # 2. Stale exit
    # ----------------------------------------------------------------
    stale_en = params.get("stale_exit_enabled", False)
    stale_bars = params.get("stale_exit_bars", 50)
    stale_atr_thresh = params.get("stale_exit_atr_threshold", 0.5)
    atr_pips = pos.atr_pips

    if stale_en and pos.bars_held >= stale_bars and recent_bars is not None:
        # Check max range over the last stale_bars bars
        lookback = min(stale_bars, len(recent_bars))
        max_range = 0.0
        for i in range(lookback):
            h, l = recent_bars[-(i + 1)]
            r = (h - l) / pip
            if r > max_range:
                max_range = r
        if atr_pips > 0 and max_range < stale_atr_thresh * atr_pips:
            actions.append(ManagementAction(
                action_type=ManagementActionType.CLOSE,
                reason="stale_exit",
                details={
                    "max_range_pips": max_range,
                    "threshold": stale_atr_thresh * atr_pips,
                },
            ))
            log.info("exit_stale", ticket=pos.ticket, max_range=max_range)
            return actions

    # ----------------------------------------------------------------
    # 3. Breakeven lock (one-shot)
    # ----------------------------------------------------------------
    be_enabled = params.get("breakeven_enabled", False)
    be_trigger = params.get("breakeven_trigger_pips", 20)
    be_offset = params.get("breakeven_offset_pips", 2)

    if be_enabled and not pos.be_locked and float_pnl >= be_trigger:
        pos.be_locked = True
        if is_buy:
            be_price = pos.entry_price + be_offset * pip
            if be_price > pos.current_sl:
                pos.current_sl = be_price
                actions.append(ManagementAction(
                    action_type=ManagementActionType.MODIFY_SL,
                    reason="breakeven",
                    new_sl=pos.current_sl,
                    new_tp=pos.tp_price,
                    details={"be_price": be_price},
                ))
                log.info("breakeven_locked", ticket=pos.ticket, new_sl=pos.current_sl)
        else:
            be_price = pos.entry_price - be_offset * pip
            if be_price < pos.current_sl:
                pos.current_sl = be_price
                actions.append(ManagementAction(
                    action_type=ManagementActionType.MODIFY_SL,
                    reason="breakeven",
                    new_sl=pos.current_sl,
                    new_tp=pos.tp_price,
                    details={"be_price": be_price},
                ))
                log.info("breakeven_locked", ticket=pos.ticket, new_sl=pos.current_sl)

    # ----------------------------------------------------------------
    # 4. Trailing stop
    # ----------------------------------------------------------------
    trailing_mode = params.get("trailing_mode", "off")
    trail_activate = params.get("trail_activate_pips", 0)
    trail_distance = params.get("trail_distance_pips", 10)
    trail_atr_m = params.get("trail_atr_mult", 2.0)

    if trailing_mode != "off":
        if not pos.trailing_active and float_pnl >= trail_activate:
            pos.trailing_active = True

        if pos.trailing_active:
            if trailing_mode == "fixed_pip":
                t_dist = trail_distance * pip
            else:  # atr_chandelier
                t_dist = trail_atr_m * atr_pips * pip

            sl_changed = False
            if is_buy:
                new_sl = bar_high - t_dist
                if new_sl > pos.current_sl:
                    pos.current_sl = new_sl
                    sl_changed = True
            else:
                new_sl = bar_low + t_dist
                if new_sl < pos.current_sl:
                    pos.current_sl = new_sl
                    sl_changed = True

            if sl_changed:
                actions.append(ManagementAction(
                    action_type=ManagementActionType.MODIFY_SL,
                    reason="trailing_stop",
                    new_sl=pos.current_sl,
                    new_tp=pos.tp_price,
                    details={"trailing_mode": trailing_mode, "t_dist": t_dist},
                ))
                log.debug("trailing_updated", ticket=pos.ticket, new_sl=pos.current_sl)

    # ----------------------------------------------------------------
    # 5. Partial close
    # ----------------------------------------------------------------
    partial_en = params.get("partial_close_enabled", False)
    partial_pct = params.get("partial_close_pct", 50)
    partial_trig = params.get("partial_close_trigger_pips", 30)

    if partial_en and not pos.partial_done and float_pnl >= partial_trig:
        pos.partial_done = True
        close_pct = partial_pct / 100.0
        close_volume = round(pos.original_volume * close_pct, 8)

        if is_buy:
            partial_pnl = (bar_close - pos.entry_price) / pip * close_pct
        else:
            partial_pnl = (pos.entry_price - bar_close) / pip * close_pct

        pos.realized_pnl_pips += partial_pnl
        pos.position_pct -= close_pct
        pos.volume = round(pos.volume - close_volume, 8)

        actions.append(ManagementAction(
            action_type=ManagementActionType.PARTIAL_CLOSE,
            reason="partial_close",
            close_volume=close_volume,
            details={
                "partial_pnl_pips": partial_pnl,
                "remaining_pct": pos.position_pct,
            },
        ))
        log.info(
            "partial_close",
            ticket=pos.ticket,
            close_volume=close_volume,
            remaining_pct=pos.position_pct,
        )

    # SL/TP handled by broker — detected in _sync_with_broker()

    return actions
