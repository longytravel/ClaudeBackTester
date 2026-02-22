"""Broker abstraction layer — connects to trading platforms for data and execution."""

from backtester.broker.mt5 import (
    connect,
    disconnect,
    fetch_candles,
    fetch_candles_range,
    get_account_info,
    get_symbol_info,
)

__all__ = [
    "connect",
    "disconnect",
    "fetch_candles",
    "fetch_candles_range",
    "get_account_info",
    "get_symbol_info",
]
