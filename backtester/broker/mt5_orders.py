"""IC Markets MT5 order management — trade execution and position queries.

Separated from mt5.py (data fetching) for clarity.
All order functions assume MT5 is already connected via broker.connect().
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import structlog

log = structlog.get_logger()


def _get_mt5():
    """Lazy import of MetaTrader5 (only available on Windows with terminal installed)."""
    import MetaTrader5 as mt5
    return mt5


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    ticket: int = 0
    retcode: int = 0
    comment: str = ""
    price: float = 0.0
    volume: float = 0.0

    @staticmethod
    def from_mt5(result) -> OrderResult:
        """Build from mt5.order_send() result."""
        if result is None:
            return OrderResult(success=False, comment="order_send returned None")
        return OrderResult(
            success=result.retcode == 10009,  # TRADE_RETCODE_DONE
            ticket=result.order if hasattr(result, "order") else 0,
            retcode=result.retcode,
            comment=result.comment if hasattr(result, "comment") else "",
            price=result.price if hasattr(result, "price") else 0.0,
            volume=result.volume if hasattr(result, "volume") else 0.0,
        )


def place_market_order(
    symbol: str,
    direction: int,
    volume: float,
    sl: float,
    tp: float,
    magic: int = 0,
    comment: str = "",
) -> OrderResult:
    """Place a market order (BUY or SELL).

    Args:
        direction: 1 for BUY, -1 for SELL
        volume: lot size
        sl: stop-loss price (0.0 to skip)
        tp: take-profit price (0.0 to skip)
        magic: magic number for identifying our orders
    """
    mt5 = _get_mt5()

    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return OrderResult(success=False, comment=f"No tick data for {symbol}")

    price = tick.ask if direction == 1 else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": magic,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    log.info(
        "placing_order",
        symbol=symbol,
        direction="BUY" if direction == 1 else "SELL",
        volume=volume,
        price=price,
        sl=sl,
        tp=tp,
        magic=magic,
    )

    result = mt5.order_send(request)
    order_result = OrderResult.from_mt5(result)

    if order_result.success:
        log.info("order_placed", ticket=order_result.ticket, price=order_result.price)
    else:
        log.error(
            "order_failed",
            retcode=order_result.retcode,
            comment=order_result.comment,
        )

    return order_result


def modify_sl_tp(
    ticket: int,
    symbol: str,
    sl: float,
    tp: float,
) -> OrderResult:
    """Modify SL/TP on an existing position."""
    mt5 = _get_mt5()

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": ticket,
        "sl": sl,
        "tp": tp,
    }

    log.debug("modifying_sl_tp", ticket=ticket, sl=sl, tp=tp)

    result = mt5.order_send(request)
    order_result = OrderResult.from_mt5(result)

    if not order_result.success:
        log.warning(
            "modify_sl_tp_failed",
            ticket=ticket,
            retcode=order_result.retcode,
            comment=order_result.comment,
        )

    return order_result


def close_position(
    ticket: int,
    symbol: str,
    direction: int,
    volume: float,
    magic: int = 0,
) -> OrderResult:
    """Close a position by sending an opposite-direction deal."""
    mt5 = _get_mt5()

    # Close BUY with SELL and vice versa
    close_type = mt5.ORDER_TYPE_SELL if direction == 1 else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return OrderResult(success=False, comment=f"No tick data for {symbol}")

    price = tick.bid if direction == 1 else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "price": price,
        "position": ticket,
        "magic": magic,
        "comment": "close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    log.info("closing_position", ticket=ticket, symbol=symbol, volume=volume)

    result = mt5.order_send(request)
    order_result = OrderResult.from_mt5(result)

    if order_result.success:
        log.info("position_closed", ticket=ticket, price=order_result.price)
    else:
        log.error(
            "close_failed",
            ticket=ticket,
            retcode=order_result.retcode,
            comment=order_result.comment,
        )

    return order_result


def partial_close(
    ticket: int,
    symbol: str,
    direction: int,
    volume: float,
    magic: int = 0,
) -> OrderResult:
    """Partially close a position (reduce volume)."""
    return close_position(ticket, symbol, direction, volume, magic)


def get_open_positions(magic: int | None = None) -> list[dict]:
    """Get all open positions, optionally filtered by magic number."""
    mt5 = _get_mt5()

    if magic is not None:
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [
            _position_to_dict(p) for p in positions
            if p.magic == magic
        ]
    else:
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [_position_to_dict(p) for p in positions]


def _position_to_dict(pos) -> dict:
    """Convert MT5 position to dict."""
    return {
        "ticket": pos.ticket,
        "symbol": pos.symbol,
        "type": pos.type,  # 0=BUY, 1=SELL
        "volume": pos.volume,
        "price_open": pos.price_open,
        "sl": pos.sl,
        "tp": pos.tp,
        "profit": pos.profit,
        "magic": pos.magic,
        "comment": pos.comment,
        "time": datetime.fromtimestamp(pos.time, tz=timezone.utc),
    }


def get_closed_deals(
    magic: int | None = None,
    from_date: datetime | None = None,
) -> list[dict]:
    """Get closed deals from history."""
    mt5 = _get_mt5()

    if from_date is None:
        from_date = datetime(2020, 1, 1, tzinfo=timezone.utc)

    now = datetime.now(tz=timezone.utc)
    deals = mt5.history_deals_get(from_date, now)
    if deals is None:
        return []

    result = []
    for d in deals:
        if magic is not None and d.magic != magic:
            continue
        result.append({
            "ticket": d.ticket,
            "order": d.order,
            "symbol": d.symbol,
            "type": d.type,
            "volume": d.volume,
            "price": d.price,
            "profit": d.profit,
            "commission": d.commission,
            "swap": d.swap,
            "magic": d.magic,
            "comment": d.comment,
            "time": datetime.fromtimestamp(d.time, tz=timezone.utc),
            "position_id": d.position_id,
        })
    return result


def get_current_tick(symbol: str) -> dict | None:
    """Get current bid/ask/spread for a symbol."""
    mt5 = _get_mt5()

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    info = mt5.symbol_info(symbol)
    point = info.point if info else 0.00001

    return {
        "bid": tick.bid,
        "ask": tick.ask,
        "spread_price": (tick.ask - tick.bid),
        "spread_pips": (tick.ask - tick.bid) / point / 10,  # points to pips
        "time": datetime.fromtimestamp(tick.time, tz=timezone.utc),
    }
