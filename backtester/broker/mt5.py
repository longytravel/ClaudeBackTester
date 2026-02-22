"""IC Markets MT5 integration — data fetching and account access.

All timestamps are converted to UTC on ingestion. MT5 server uses EET
(UTC+2 winter, UTC+3 summer) — handled automatically via zoneinfo.
"""

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import structlog

log = structlog.get_logger()

# MT5 server timezone (IC Markets uses EET — handles DST automatically)
MT5_TZ = ZoneInfo("EET")
UTC = ZoneInfo("UTC")

# Default terminal path
DEFAULT_MT5_PATH = r"C:\Program Files\MetaTrader 5 IC Markets Global\terminal64.exe"


def _get_mt5():
    """Lazy import of MetaTrader5 (only available on Windows with terminal installed)."""
    import MetaTrader5 as mt5
    return mt5


def connect(
    login: int | None = None,
    password: str | None = None,
    server: str | None = None,
    path: str | None = None,
) -> bool:
    """Connect to MT5 terminal.

    If terminal is already running and logged in, connects without credentials.
    Otherwise reads credentials from params or env vars.
    """
    mt5 = _get_mt5()

    # First try connecting to already-running terminal
    ok = mt5.initialize()
    if ok:
        acct = mt5.account_info()
        if acct and acct.login > 0:
            log.info(
                "mt5_connected",
                server=acct.server,
                login=acct.login,
                broker=acct.company,
                balance=acct.balance,
            )
            return True

    # Terminal not running or not logged in — try with credentials
    path = path or os.environ.get("MT5_PATH", DEFAULT_MT5_PATH)
    login = login or int(os.environ.get("MT5_LOGIN", 0))
    password = password or os.environ.get("MT5_PASSWORD", "")
    server = server or os.environ.get("MT5_SERVER", "ICMarketsSC-Demo")

    ok = mt5.initialize(path, login=login, password=password, server=server)
    if not ok:
        error = mt5.last_error()
        log.error("mt5_connect_failed", error=error)
        return False

    acct = mt5.account_info()
    log.info(
        "mt5_connected",
        server=acct.server,
        login=acct.login,
        broker=acct.company,
        balance=acct.balance,
    )
    return True


def disconnect():
    """Shutdown MT5 connection."""
    mt5 = _get_mt5()
    mt5.shutdown()


def _mt5_epoch_to_utc(epoch_seconds: int) -> pd.Timestamp:
    """Convert MT5 epoch timestamp to UTC.

    MT5 returns Unix-style epoch seconds but they represent server-local
    time (EET), not actual UTC. We interpret them as EET then convert.
    """
    # Interpret the epoch as a naive datetime, then localize to EET
    naive = datetime.utcfromtimestamp(epoch_seconds)
    local = naive.replace(tzinfo=MT5_TZ)
    return pd.Timestamp(local.astimezone(UTC))


def _rates_to_dataframe(rates) -> pd.DataFrame:
    """Convert MT5 rates array to a UTC-indexed DataFrame.

    Columns: open, high, low, close, volume, spread
    Index: DatetimeIndex in UTC
    """
    df = pd.DataFrame(rates)

    # Convert timestamps: MT5 'time' is epoch seconds in server timezone (EET)
    df["time"] = df["time"].apply(_mt5_epoch_to_utc)
    df = df.set_index("time")
    df.index.name = "timestamp"

    # Rename tick_volume to volume, keep spread
    df = df.rename(columns={"tick_volume": "volume"})

    # Convert spread from points (integer) to price units
    # For 5-digit pairs (EURUSD etc), point = 0.00001
    # For 3-digit pairs (USDJPY etc), point = 0.001
    # We'll store raw spread in points and let consumers convert with symbol info
    df["spread_points"] = df["spread"]

    # Keep only the columns we use
    df = df[["open", "high", "low", "close", "volume", "spread_points"]]
    return df


def fetch_candles(
    symbol: str,
    timeframe: str = "M1",
    count: int = 1000,
) -> pd.DataFrame:
    """Fetch the most recent N candles for a symbol. Returns UTC-indexed DataFrame."""
    mt5 = _get_mt5()

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D": mt5.TIMEFRAME_D1,
        "W": mt5.TIMEFRAME_W1,
    }
    mt5_tf = tf_map.get(timeframe)
    if mt5_tf is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
    if rates is None or len(rates) == 0:
        log.warning("no_candles", symbol=symbol, timeframe=timeframe)
        return pd.DataFrame()

    df = _rates_to_dataframe(rates)
    log.info("fetched_candles", symbol=symbol, timeframe=timeframe, count=len(df))
    return df


def fetch_candles_range(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "M1",
) -> pd.DataFrame:
    """Fetch candles for a date range. start/end must be UTC datetimes.

    Internally converts UTC to EET for MT5 query, then converts results back to UTC.
    """
    mt5 = _get_mt5()

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D": mt5.TIMEFRAME_D1,
        "W": mt5.TIMEFRAME_W1,
    }
    mt5_tf = tf_map.get(timeframe)
    if mt5_tf is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    # Convert UTC datetimes to EET for MT5 query
    if start.tzinfo is not None:
        start_eet = start.astimezone(MT5_TZ).replace(tzinfo=None)
    else:
        start_eet = start.replace(tzinfo=UTC).astimezone(MT5_TZ).replace(tzinfo=None)

    if end.tzinfo is not None:
        end_eet = end.astimezone(MT5_TZ).replace(tzinfo=None)
    else:
        end_eet = end.replace(tzinfo=UTC).astimezone(MT5_TZ).replace(tzinfo=None)

    rates = mt5.copy_rates_range(symbol, mt5_tf, start_eet, end_eet)
    if rates is None or len(rates) == 0:
        log.warning("no_candles_range", symbol=symbol, start=str(start), end=str(end))
        return pd.DataFrame()

    df = _rates_to_dataframe(rates)
    log.info(
        "fetched_candles_range",
        symbol=symbol,
        timeframe=timeframe,
        count=len(df),
        from_date=str(df.index[0]),
        to_date=str(df.index[-1]),
    )
    return df


def get_account_info() -> dict | None:
    """Get current account info."""
    mt5 = _get_mt5()
    acct = mt5.account_info()
    if acct is None:
        return None
    return {
        "login": acct.login,
        "server": acct.server,
        "balance": acct.balance,
        "equity": acct.equity,
        "margin": acct.margin,
        "free_margin": acct.margin_free,
        "leverage": acct.leverage,
        "broker": acct.company,
    }


def get_symbol_info(symbol: str) -> dict | None:
    """Get symbol details (point size, digits, spread, etc.)."""
    mt5 = _get_mt5()
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    return {
        "name": info.name,
        "point": info.point,
        "digits": info.digits,
        "spread": info.spread,
        "trade_contract_size": info.trade_contract_size,
        "volume_min": info.volume_min,
        "volume_max": info.volume_max,
        "volume_step": info.volume_step,
    }
