"""Backtest-to-live trade verification.

Compares live MT5 trades against backtest replay to identify
discrepancies in signal timing, entry/exit prices, SL/TP levels,
exit reasons, and PnL. Produces a detailed comparison report
showing where backtest and live diverge so the system can be tuned.

Workflow:
    1. Fetch closed trades from MT5 history (by magic number or time range)
    2. Download/refresh Dukascopy data to cover the trade period
    3. Replay the backtest with identical params (via telemetry)
    4. Match live trades to backtest trades by direction + time proximity
    5. Compare each matched pair across all dimensions
    6. Generate a structured report
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default data directory
DATA_DIR = Path("G:/My Drive/BackTestData")

# Maximum time gap (in minutes) to consider a live trade matched to a backtest trade
MAX_MATCH_GAP_MINUTES = 120  # 2 H1 bars tolerance


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LiveTrade:
    """A completed live trade reconstructed from MT5 deal history."""
    position_id: int
    symbol: str
    direction: int          # 1=BUY, -1=SELL
    volume: float
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    profit: float           # account currency
    commission: float
    swap: float
    pnl_pips: float
    spread_at_entry: float  # pips (if available)
    exit_reason: str        # "stop_loss", "take_profit", "manual", "unknown"
    sl_price: float = 0.0
    tp_price: float = 0.0
    magic: int = 0


@dataclass
class BacktestTrade:
    """A single trade from backtest telemetry replay."""
    signal_index: int
    bar_entry: int
    bar_exit: int
    bars_held: int
    direction: str          # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    sl_price: float
    tp_price: float
    sl_pips: float
    tp_pips: float
    pnl_pips: float
    exit_reason: str
    mfe_pips: float
    mae_pips: float
    entry_time: datetime | None = None
    exit_time: datetime | None = None


@dataclass
class TradeComparison:
    """Side-by-side comparison of one matched live vs backtest trade."""
    live: LiveTrade
    backtest: BacktestTrade

    # Deltas (live - backtest)
    entry_price_delta: float = 0.0      # pips
    exit_price_delta: float = 0.0       # pips
    sl_delta: float = 0.0               # pips
    tp_delta: float = 0.0               # pips
    pnl_delta: float = 0.0              # pips
    entry_time_delta_minutes: float = 0.0
    exit_time_delta_minutes: float = 0.0
    exit_reason_match: bool = True
    direction_match: bool = True

    @property
    def entry_slippage_pips(self) -> float:
        """Positive = worse fill for trader (paid more for BUY, got less for SELL)."""
        return self.entry_price_delta

    @property
    def summary(self) -> str:
        status = "MATCH" if self.exit_reason_match and abs(self.pnl_delta) < 2.0 else "DIVERGED"
        return (
            f"[{status}] {self.live.direction:>4s} | "
            f"entry_slip={self.entry_slippage_pips:+.1f}pip | "
            f"pnl_delta={self.pnl_delta:+.1f}pip | "
            f"exit: live={self.live.exit_reason} bt={self.backtest.exit_reason}"
        )


@dataclass
class VerificationReport:
    """Full verification report for a strategy/pair/timeframe."""
    strategy_name: str
    pair: str
    timeframe: str
    params: dict[str, Any]
    run_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    # Trade counts
    live_trades: int = 0
    backtest_trades: int = 0
    matched_trades: int = 0
    unmatched_live: int = 0
    unmatched_backtest: int = 0

    # Comparisons
    comparisons: list[TradeComparison] = field(default_factory=list)

    # Aggregate stats
    mean_entry_slippage_pips: float = 0.0
    mean_pnl_delta_pips: float = 0.0
    exit_reason_match_rate: float = 0.0
    direction_match_rate: float = 0.0

    # Unmatched trades (for investigation)
    unmatched_live_trades: list[LiveTrade] = field(default_factory=list)
    unmatched_backtest_trades: list[BacktestTrade] = field(default_factory=list)

    # Data freshness
    data_latest_timestamp: datetime | None = None
    data_covers_all_trades: bool = True

    def compute_aggregates(self) -> None:
        """Compute aggregate statistics from individual comparisons."""
        if not self.comparisons:
            return
        n = len(self.comparisons)
        self.mean_entry_slippage_pips = sum(c.entry_slippage_pips for c in self.comparisons) / n
        self.mean_pnl_delta_pips = sum(c.pnl_delta for c in self.comparisons) / n
        self.exit_reason_match_rate = sum(1 for c in self.comparisons if c.exit_reason_match) / n
        self.direction_match_rate = sum(1 for c in self.comparisons if c.direction_match) / n

    def to_dict(self) -> dict:
        """Serialize for JSON output."""
        return {
            "strategy_name": self.strategy_name,
            "pair": self.pair,
            "timeframe": self.timeframe,
            "run_time": self.run_time.isoformat(),
            "params": self.params,
            "trade_counts": {
                "live": self.live_trades,
                "backtest": self.backtest_trades,
                "matched": self.matched_trades,
                "unmatched_live": self.unmatched_live,
                "unmatched_backtest": self.unmatched_backtest,
            },
            "aggregates": {
                "mean_entry_slippage_pips": round(self.mean_entry_slippage_pips, 2),
                "mean_pnl_delta_pips": round(self.mean_pnl_delta_pips, 2),
                "exit_reason_match_rate": round(self.exit_reason_match_rate, 3),
                "direction_match_rate": round(self.direction_match_rate, 3),
            },
            "comparisons": [_comparison_to_dict(c) for c in self.comparisons],
            "unmatched_live_trades": [_live_trade_to_dict(t) for t in self.unmatched_live_trades],
            "unmatched_backtest_trades": [
                _bt_trade_to_dict(t) for t in self.unmatched_backtest_trades
            ],
            "data_latest_timestamp": (
                self.data_latest_timestamp.isoformat() if self.data_latest_timestamp else None
            ),
            "data_covers_all_trades": self.data_covers_all_trades,
        }


# ---------------------------------------------------------------------------
# Step 1: Fetch live trades from MT5
# ---------------------------------------------------------------------------

def fetch_live_trades(
    pair: str,
    magic: int,
    from_date: datetime | None = None,
) -> list[LiveTrade]:
    """Fetch closed trades from MT5 history and reconstruct full trades.

    MT5 records deals (individual fills), not trades. A complete trade
    has an entry deal and an exit deal sharing the same position_id.
    """
    from backtester.broker.mt5_orders import get_closed_deals

    deals = get_closed_deals(magic=magic, from_date=from_date)
    if not deals:
        logger.warning("No closed deals found for magic=%d", magic)
        return []

    # Filter to the pair we care about
    deals = [d for d in deals if d["symbol"] == pair]

    # Group by position_id
    positions: dict[int, list[dict]] = {}
    for deal in deals:
        pos_id = deal.get("position_id", 0)
        if pos_id == 0:
            continue
        positions.setdefault(pos_id, []).append(deal)

    pip_value = 0.01 if "JPY" in pair else 0.0001
    trades = []

    for pos_id, pos_deals in positions.items():
        if len(pos_deals) < 2:
            # Need at least entry + exit
            continue

        # Sort by time
        pos_deals.sort(key=lambda d: d["time"])

        entry_deal = pos_deals[0]
        exit_deal = pos_deals[-1]

        # Direction: type 0=BUY, 1=SELL for deals
        # Entry BUY deal type=0, entry SELL deal type=1
        direction = 1 if entry_deal["type"] == 0 else -1

        # PnL in pips
        if direction == 1:
            pnl_pips = (exit_deal["price"] - entry_deal["price"]) / pip_value
        else:
            pnl_pips = (entry_deal["price"] - exit_deal["price"]) / pip_value

        # Determine exit reason from comment
        exit_comment = exit_deal.get("comment", "").lower()
        if "sl" in exit_comment or "stop" in exit_comment:
            exit_reason = "stop_loss"
        elif "tp" in exit_comment or "profit" in exit_comment:
            exit_reason = "take_profit"
        elif "close" in exit_comment:
            exit_reason = "manual"
        else:
            exit_reason = "unknown"

        # Sum up commission and swap across all deals
        total_commission = sum(d.get("commission", 0.0) for d in pos_deals)
        total_swap = sum(d.get("swap", 0.0) for d in pos_deals)
        total_profit = sum(d.get("profit", 0.0) for d in pos_deals)

        trades.append(LiveTrade(
            position_id=pos_id,
            symbol=pair,
            direction=direction,
            volume=entry_deal["volume"],
            entry_price=entry_deal["price"],
            entry_time=entry_deal["time"],
            exit_price=exit_deal["price"],
            exit_time=exit_deal["time"],
            profit=total_profit,
            commission=total_commission,
            swap=total_swap,
            pnl_pips=pnl_pips,
            spread_at_entry=0.0,  # Not available from deal history
            exit_reason=exit_reason,
            magic=magic,
        ))

    trades.sort(key=lambda t: t.entry_time)
    logger.info("Fetched %d live trades for %s (magic=%d)", len(trades), pair, magic)
    return trades


def fetch_live_trades_from_audit(
    audit_path: str | Path,
    pair: str | None = None,
) -> list[LiveTrade]:
    """Fetch trade history from the audit.jsonl file (alternative to MT5).

    Useful when MT5 is not connected or for offline analysis.
    """
    audit_path = Path(audit_path)
    if not audit_path.exists():
        logger.warning("Audit file not found: %s", audit_path)
        return []

    pip_value = 0.01 if pair and "JPY" in pair else 0.0001

    # Parse all events
    events: list[dict] = []
    with open(audit_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    # Match order_placed with position_closed events by ticket
    opens: dict[int, dict] = {}
    trades: list[LiveTrade] = []

    for ev in events:
        event_type = ev.get("event", "")

        if event_type == "order_placed":
            ticket = ev.get("ticket", 0)
            opens[ticket] = ev

        elif event_type == "position_closed":
            ticket = ev.get("ticket", 0)
            open_ev = opens.pop(ticket, None)
            if open_ev is None:
                continue

            direction = 1 if open_ev.get("direction") == "BUY" else -1
            entry_price = open_ev.get("price", 0.0)
            exit_price = ev.get("exit_price", 0.0)

            if direction == 1:
                pnl_pips = (exit_price - entry_price) / pip_value
            else:
                pnl_pips = (entry_price - exit_price) / pip_value

            entry_time = datetime.fromisoformat(open_ev.get("timestamp", ""))
            exit_time = datetime.fromisoformat(ev.get("timestamp", ""))

            trades.append(LiveTrade(
                position_id=ticket,
                symbol=open_ev.get("symbol", pair or ""),
                direction=direction,
                volume=open_ev.get("volume", 0.01),
                entry_price=entry_price,
                entry_time=entry_time,
                exit_price=exit_price,
                exit_time=exit_time,
                profit=0.0,  # not in audit
                commission=0.0,
                swap=0.0,
                pnl_pips=pnl_pips,
                spread_at_entry=0.0,
                exit_reason=ev.get("reason", "unknown"),
                sl_price=open_ev.get("sl", 0.0),
                tp_price=open_ev.get("tp", 0.0),
                magic=0,
            ))

    trades.sort(key=lambda t: t.entry_time)
    logger.info("Loaded %d trades from audit file %s", len(trades), audit_path)
    return trades


# ---------------------------------------------------------------------------
# Step 2: Ensure fresh data
# ---------------------------------------------------------------------------

def ensure_fresh_data(
    pair: str,
    data_dir: str | Path = DATA_DIR,
) -> Path:
    """Download/update data so we have the very latest bars.

    Returns path to consolidated M1 parquet file.
    """
    from backtester.data.downloader import ensure_fresh
    from backtester.data.timeframes import convert_timeframes

    data_dir = str(data_dir)

    # Update M1 data
    m1_path = ensure_fresh(pair, data_dir, max_age_hours=1.0)

    # Rebuild higher timeframes from fresh M1
    convert_timeframes(pair, data_dir, ["H1", "M15", "M5"])

    logger.info("Data refreshed for %s", pair)
    return m1_path


# ---------------------------------------------------------------------------
# Step 3: Replay backtest with telemetry
# ---------------------------------------------------------------------------

def _load_data_arrays(
    pair: str,
    timeframe: str,
    data_dir: str | Path = DATA_DIR,
) -> tuple[dict[str, np.ndarray], pd.DatetimeIndex, dict | None]:
    """Load OHLCV arrays + M1 sub-bar arrays for backtest replay.

    Returns (arrays_dict, timestamps, m1_dict_or_none).
    """
    from backtester.data.timeframes import build_h1_to_m1_mapping

    data_dir = Path(data_dir)
    pair_file = pair.replace("/", "_")

    # Load timeframe data
    tf_path = data_dir / f"{pair_file}_{timeframe}.parquet"
    if not tf_path.exists():
        raise FileNotFoundError(f"Timeframe data not found: {tf_path}")

    df = pd.read_parquet(tf_path)
    timestamps = df.index

    arrays = {
        "open": df["open"].to_numpy(dtype=np.float64),
        "high": df["high"].to_numpy(dtype=np.float64),
        "low": df["low"].to_numpy(dtype=np.float64),
        "close": df["close"].to_numpy(dtype=np.float64),
        "volume": df["volume"].to_numpy(dtype=np.float64),
        "spread": df["spread"].to_numpy(dtype=np.float64),
        "bar_hour": df.index.hour.to_numpy(dtype=np.int64),
        "bar_day_of_week": df.index.dayofweek.to_numpy(dtype=np.int64),
    }

    # Load M1 sub-bar data
    m1_path = data_dir / f"{pair_file}_M1.parquet"
    m1_dict = None
    if m1_path.exists():
        m1_df = pd.read_parquet(m1_path)
        h1_ts = timestamps.astype(np.int64).to_numpy()
        m1_ts = m1_df.index.astype(np.int64).to_numpy()
        start_idx, end_idx = build_h1_to_m1_mapping(h1_ts, m1_ts)

        m1_dict = {
            "m1_high": m1_df["high"].to_numpy(dtype=np.float64),
            "m1_low": m1_df["low"].to_numpy(dtype=np.float64),
            "m1_close": m1_df["close"].to_numpy(dtype=np.float64),
            "m1_spread": m1_df["spread"].to_numpy(dtype=np.float64),
            "h1_to_m1_start": start_idx,
            "h1_to_m1_end": end_idx,
        }
        logger.info("M1 sub-bar data loaded: %d bars", len(m1_df))

    return arrays, timestamps, m1_dict


def replay_backtest(
    strategy_name: str,
    pair: str,
    timeframe: str,
    params: dict[str, Any],
    data_dir: str | Path = DATA_DIR,
    from_date: datetime | None = None,
    to_date: datetime | None = None,
) -> tuple[list[BacktestTrade], pd.DatetimeIndex]:
    """Replay a backtest with telemetry to get per-trade details.

    Uses the exact same params as the live trader.
    Returns (trades, bar_timestamps) so trades can be dated.
    """
    from backtester.core.dtypes import EXEC_FULL
    from backtester.core.engine import BacktestEngine
    from backtester.core.telemetry import run_telemetry
    from backtester.strategies.registry import create as create_strategy

    arrays, timestamps, m1_dict = _load_data_arrays(pair, timeframe, data_dir)

    # Optionally slice to date range
    if from_date or to_date:
        mask = np.ones(len(timestamps), dtype=bool)
        if from_date:
            if from_date.tzinfo is None:
                from_date = from_date.replace(tzinfo=timezone.utc)
            mask &= timestamps >= pd.Timestamp(from_date)
        if to_date:
            if to_date.tzinfo is None:
                to_date = to_date.replace(tzinfo=timezone.utc)
            mask &= timestamps <= pd.Timestamp(to_date)

        idx = np.where(mask)[0]
        if len(idx) == 0:
            logger.warning("No bars in date range %s to %s", from_date, to_date)
            return [], timestamps

        start, end = idx[0], idx[-1] + 1
        timestamps = timestamps[start:end]
        for key in arrays:
            arrays[key] = arrays[key][start:end]

        # Re-slice M1 mapping isn't needed — the H1→M1 indices still point
        # to correct M1 bars since M1 data is not sliced

    pip_value = 0.01 if "JPY" in pair else 0.0001

    # Determine bars_per_year based on timeframe
    tf_bpy = {
        "M1": 252 * 24 * 60, "M5": 252 * 24 * 12, "M15": 252 * 24 * 4,
        "M30": 252 * 24 * 2, "H1": 252 * 24, "H4": 252 * 6, "D": 252,
    }
    bars_per_year = tf_bpy.get(timeframe, 6048)

    strategy = create_strategy(strategy_name)

    engine_kwargs: dict[str, Any] = {
        "strategy": strategy,
        "open_": arrays["open"],
        "high": arrays["high"],
        "low": arrays["low"],
        "close": arrays["close"],
        "volume": arrays["volume"],
        "spread": arrays["spread"],
        "pip_value": pip_value,
        "slippage_pips": 0.5,
        "bars_per_year": bars_per_year,
        "commission_pips": 0.7,
        "max_spread_pips": 3.0,
        "bar_hour": arrays["bar_hour"],
        "bar_day_of_week": arrays["bar_day_of_week"],
    }

    if m1_dict:
        engine_kwargs.update({
            "m1_high": m1_dict["m1_high"],
            "m1_low": m1_dict["m1_low"],
            "m1_close": m1_dict["m1_close"],
            "m1_spread": m1_dict["m1_spread"],
            "h1_to_m1_start": m1_dict["h1_to_m1_start"],
            "h1_to_m1_end": m1_dict["h1_to_m1_end"],
        })

    engine = BacktestEngine(**engine_kwargs)
    telemetry = run_telemetry(engine, params, exec_mode=EXEC_FULL)

    # Convert telemetry trades to BacktestTrade with timestamps
    bt_trades = []
    for t in telemetry.trades:
        entry_time = timestamps[t.bar_entry] if t.bar_entry < len(timestamps) else None
        exit_time = timestamps[t.bar_exit] if t.bar_exit < len(timestamps) else None

        bt_trades.append(BacktestTrade(
            signal_index=t.signal_index,
            bar_entry=t.bar_entry,
            bar_exit=t.bar_exit,
            bars_held=t.bars_held,
            direction=t.direction,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            sl_price=t.sl_price,
            tp_price=t.tp_price,
            sl_pips=t.sl_pips,
            tp_pips=t.tp_pips,
            pnl_pips=t.pnl_pips,
            exit_reason=t.exit_reason,
            mfe_pips=t.mfe_pips,
            mae_pips=t.mae_pips,
            entry_time=entry_time.to_pydatetime() if entry_time is not None else None,
            exit_time=exit_time.to_pydatetime() if exit_time is not None else None,
        ))

    logger.info(
        "Backtest replay: %d signals, %d filtered, %d trades",
        telemetry.n_signals_total,
        telemetry.n_signals_filtered,
        len(bt_trades),
    )
    return bt_trades, timestamps


# ---------------------------------------------------------------------------
# Step 4: Match trades
# ---------------------------------------------------------------------------

def match_trades(
    live_trades: list[LiveTrade],
    bt_trades: list[BacktestTrade],
    max_gap_minutes: float = MAX_MATCH_GAP_MINUTES,
) -> tuple[list[tuple[LiveTrade, BacktestTrade]], list[LiveTrade], list[BacktestTrade]]:
    """Match live trades to backtest trades by direction + time proximity.

    Returns (matched_pairs, unmatched_live, unmatched_bt).
    Each backtest trade is matched to at most one live trade (greedy nearest).
    """
    matched: list[tuple[LiveTrade, BacktestTrade]] = []
    used_bt: set[int] = set()  # indices into bt_trades

    unmatched_live: list[LiveTrade] = []

    for live in live_trades:
        live_dir = "BUY" if live.direction == 1 else "SELL"
        best_idx = -1
        best_gap = float("inf")

        for j, bt in enumerate(bt_trades):
            if j in used_bt:
                continue
            if bt.direction != live_dir:
                continue
            if bt.entry_time is None:
                continue

            # Make timezone-aware comparison
            bt_time = bt.entry_time
            live_time = live.entry_time
            if bt_time.tzinfo is None:
                bt_time = bt_time.replace(tzinfo=timezone.utc)
            if live_time.tzinfo is None:
                live_time = live_time.replace(tzinfo=timezone.utc)

            gap = abs((live_time - bt_time).total_seconds()) / 60.0

            if gap < best_gap and gap <= max_gap_minutes:
                best_gap = gap
                best_idx = j

        if best_idx >= 0:
            matched.append((live, bt_trades[best_idx]))
            used_bt.add(best_idx)
        else:
            unmatched_live.append(live)

    unmatched_bt = [bt for j, bt in enumerate(bt_trades) if j not in used_bt]

    logger.info(
        "Trade matching: %d matched, %d unmatched live, %d unmatched backtest",
        len(matched), len(unmatched_live), len(unmatched_bt),
    )
    return matched, unmatched_live, unmatched_bt


# ---------------------------------------------------------------------------
# Step 5: Compare matched trades
# ---------------------------------------------------------------------------

def compare_matched_trades(
    matched: list[tuple[LiveTrade, BacktestTrade]],
    pip_value: float,
) -> list[TradeComparison]:
    """Compute detailed deltas for each matched pair."""
    comparisons = []

    for live, bt in matched:
        # Direction check
        live_dir = "BUY" if live.direction == 1 else "SELL"
        dir_match = live_dir == bt.direction

        # Price deltas in pips
        entry_delta = (live.entry_price - bt.entry_price) / pip_value
        exit_delta = (live.exit_price - bt.exit_price) / pip_value
        sl_delta = (live.sl_price - bt.sl_price) / pip_value if live.sl_price > 0 else 0.0
        tp_delta = (live.tp_price - bt.tp_price) / pip_value if live.tp_price > 0 else 0.0
        pnl_delta = live.pnl_pips - bt.pnl_pips

        # For SELL trades, entry slippage sign is inverted
        # (higher entry = worse for SELL, but entry_delta is positive)
        if live.direction == -1:
            entry_delta = -entry_delta

        # Time deltas
        entry_time_delta = 0.0
        exit_time_delta = 0.0
        if bt.entry_time:
            bt_entry = bt.entry_time.replace(tzinfo=timezone.utc) if bt.entry_time.tzinfo is None else bt.entry_time
            live_entry = live.entry_time.replace(tzinfo=timezone.utc) if live.entry_time.tzinfo is None else live.entry_time
            entry_time_delta = (live_entry - bt_entry).total_seconds() / 60.0
        if bt.exit_time:
            bt_exit = bt.exit_time.replace(tzinfo=timezone.utc) if bt.exit_time.tzinfo is None else bt.exit_time
            live_exit = live.exit_time.replace(tzinfo=timezone.utc) if live.exit_time.tzinfo is None else live.exit_time
            exit_time_delta = (live_exit - bt_exit).total_seconds() / 60.0

        # Exit reason match (normalize names)
        live_reason = _normalize_exit_reason(live.exit_reason)
        bt_reason = _normalize_exit_reason(bt.exit_reason)
        reason_match = live_reason == bt_reason

        comparisons.append(TradeComparison(
            live=live,
            backtest=bt,
            entry_price_delta=round(entry_delta, 2),
            exit_price_delta=round(exit_delta, 2),
            sl_delta=round(sl_delta, 2),
            tp_delta=round(tp_delta, 2),
            pnl_delta=round(pnl_delta, 2),
            entry_time_delta_minutes=round(entry_time_delta, 1),
            exit_time_delta_minutes=round(exit_time_delta, 1),
            exit_reason_match=reason_match,
            direction_match=dir_match,
        ))

    return comparisons


def _normalize_exit_reason(reason: str) -> str:
    """Normalize exit reason strings for comparison."""
    reason = reason.lower().strip()
    mapping = {
        "stop_loss": "sl",
        "sl": "sl",
        "take_profit": "tp",
        "tp": "tp",
        "trailing_stop": "trailing",
        "trailing": "trailing",
        "breakeven": "be",
        "max_bars": "max_bars",
        "stale_exit": "stale",
        "stale": "stale",
        "manual": "manual",
        "broker_close": "broker",
    }
    return mapping.get(reason, reason)


# ---------------------------------------------------------------------------
# Step 6: Full orchestration
# ---------------------------------------------------------------------------

def run_verification(
    strategy_name: str,
    pair: str,
    timeframe: str,
    checkpoint_path: str | Path,
    candidate_index: int = 0,
    from_date: datetime | None = None,
    magic: int | None = None,
    refresh_data: bool = True,
    use_audit: str | None = None,
    data_dir: str | Path = DATA_DIR,
) -> VerificationReport:
    """Run the full backtest-to-live verification pipeline.

    Args:
        strategy_name: Name of the strategy (e.g., "ema_crossover")
        pair: Trading pair (e.g., "EURUSD" for MT5, "EUR/USD" for data)
        timeframe: Timeframe (e.g., "H1")
        checkpoint_path: Path to pipeline checkpoint.json
        candidate_index: Which candidate's params to use (default 0)
        from_date: Only consider trades after this date
        magic: MT5 magic number (computed from instance_id if not provided)
        refresh_data: Whether to download latest data first
        use_audit: Path to audit.jsonl (alternative to MT5)
        data_dir: Data directory for parquet files

    Returns:
        VerificationReport with all comparisons
    """
    # Normalize pair formats
    pair_dukascopy = pair.replace("_", "/") if "_" not in pair and "/" not in pair else pair
    if "/" not in pair_dukascopy:
        # EURUSD -> EUR/USD
        pair_dukascopy = pair_dukascopy[:3] + "/" + pair_dukascopy[3:]
    pair_mt5 = pair_dukascopy.replace("/", "")

    logger.info(
        "Starting verification: %s %s %s (candidate %d)",
        strategy_name, pair_dukascopy, timeframe, candidate_index,
    )

    # 1. Load params from checkpoint
    checkpoint_path = Path(checkpoint_path)
    with open(checkpoint_path) as f:
        checkpoint = json.load(f)

    candidates = checkpoint.get("candidates", [])
    if candidate_index >= len(candidates):
        raise ValueError(
            f"Candidate {candidate_index} not found (have {len(candidates)} candidates)"
        )
    params = candidates[candidate_index].get("params", {})
    if not params:
        raise ValueError(f"Candidate {candidate_index} has empty params")

    logger.info("Loaded params: %s", params)

    # 2. Refresh data
    if refresh_data:
        logger.info("Refreshing data for %s...", pair_dukascopy)
        ensure_fresh_data(pair_dukascopy, data_dir)

    # Get data freshness info
    from backtester.data.downloader import get_latest_timestamp
    data_latest = get_latest_timestamp(str(data_dir), pair_dukascopy)

    # 3. Fetch live trades
    if use_audit:
        live_trades = fetch_live_trades_from_audit(use_audit, pair_mt5)
    else:
        if magic is None:
            instance_id = f"{strategy_name}_{pair_mt5}_{timeframe}"
            digest = hashlib.md5(instance_id.encode()).hexdigest()
            magic = int(digest[:8], 16) % (2**31)
        live_trades = fetch_live_trades(pair_mt5, magic, from_date)

    if not live_trades:
        logger.warning("No live trades found — cannot compare")

    # Determine replay date range (cover all live trades + some margin)
    replay_from = from_date
    replay_to = None
    if live_trades:
        earliest = min(t.entry_time for t in live_trades)
        latest = max(t.exit_time for t in live_trades)
        # Add buffer: 500 bars before earliest trade for indicator warmup
        replay_from = earliest - timedelta(days=30)
        replay_to = latest + timedelta(hours=2)

    # 4. Replay backtest
    bt_trades, bar_timestamps = replay_backtest(
        strategy_name=strategy_name,
        pair=pair_dukascopy,
        timeframe=timeframe,
        params=params,
        data_dir=data_dir,
        from_date=replay_from,
        to_date=replay_to,
    )

    # Filter backtest trades to only those in the live trading period
    if live_trades and bt_trades:
        earliest_live = min(t.entry_time for t in live_trades)
        if earliest_live.tzinfo is None:
            earliest_live = earliest_live.replace(tzinfo=timezone.utc)
        bt_trades = [
            t for t in bt_trades
            if t.entry_time is not None and (
                t.entry_time.replace(tzinfo=timezone.utc) if t.entry_time.tzinfo is None
                else t.entry_time
            ) >= earliest_live - timedelta(hours=2)
        ]

    pip_value = 0.01 if "JPY" in pair_dukascopy else 0.0001

    # 5. Match and compare
    matched, unmatched_live, unmatched_bt = match_trades(live_trades, bt_trades)
    comparisons = compare_matched_trades(matched, pip_value)

    # 6. Build report
    report = VerificationReport(
        strategy_name=strategy_name,
        pair=pair_dukascopy,
        timeframe=timeframe,
        params=params,
        live_trades=len(live_trades),
        backtest_trades=len(bt_trades),
        matched_trades=len(matched),
        unmatched_live=len(unmatched_live),
        unmatched_backtest=len(unmatched_bt),
        comparisons=comparisons,
        unmatched_live_trades=unmatched_live,
        unmatched_backtest_trades=unmatched_bt,
        data_latest_timestamp=data_latest,
        data_covers_all_trades=(
            data_latest is not None and live_trades and
            data_latest >= max(t.exit_time for t in live_trades).replace(tzinfo=None)
        ) if live_trades and data_latest else True,
    )
    report.compute_aggregates()

    logger.info(
        "Verification complete: %d matched, %.1f%% exit reason match, "
        "mean slippage=%.2f pips, mean PnL delta=%.2f pips",
        report.matched_trades,
        report.exit_reason_match_rate * 100,
        report.mean_entry_slippage_pips,
        report.mean_pnl_delta_pips,
    )

    return report


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_report(report: VerificationReport) -> None:
    """Print a human-readable verification report to stdout."""
    print("\n" + "=" * 80)
    print("  BACKTEST vs LIVE TRADE VERIFICATION REPORT")
    print("=" * 80)
    print(f"  Strategy:   {report.strategy_name}")
    print(f"  Pair:       {report.pair}")
    print(f"  Timeframe:  {report.timeframe}")
    print(f"  Run time:   {report.run_time.strftime('%Y-%m-%d %H:%M UTC')}")
    if report.data_latest_timestamp:
        print(f"  Data up to: {report.data_latest_timestamp}")
    print()

    # Trade counts
    print("  TRADE COUNTS")
    print(f"    Live trades:       {report.live_trades}")
    print(f"    Backtest trades:   {report.backtest_trades}")
    print(f"    Matched:           {report.matched_trades}")
    print(f"    Unmatched live:    {report.unmatched_live}")
    print(f"    Unmatched backtest:{report.unmatched_backtest}")
    print()

    if not report.comparisons:
        print("  No matched trades to compare.")
        if report.unmatched_live_trades:
            print("\n  UNMATCHED LIVE TRADES:")
            for t in report.unmatched_live_trades:
                dir_str = "BUY" if t.direction == 1 else "SELL"
                print(f"    {t.entry_time} {dir_str} @ {t.entry_price:.5f} -> "
                      f"{t.exit_price:.5f} ({t.exit_reason}) PnL={t.pnl_pips:+.1f} pips")
        if report.unmatched_backtest_trades:
            print(f"\n  UNMATCHED BACKTEST TRADES (showing first 20):")
            for t in report.unmatched_backtest_trades[:20]:
                print(f"    {t.entry_time} {t.direction} @ {t.entry_price:.5f} -> "
                      f"{t.exit_price:.5f} ({t.exit_reason}) PnL={t.pnl_pips:+.1f} pips")
        print("=" * 80)
        return

    # Aggregate stats
    print("  AGGREGATE STATISTICS")
    print(f"    Mean entry slippage:   {report.mean_entry_slippage_pips:+.2f} pips")
    print(f"    Mean PnL delta:        {report.mean_pnl_delta_pips:+.2f} pips")
    print(f"    Exit reason match:     {report.exit_reason_match_rate:.0%}")
    print(f"    Direction match:       {report.direction_match_rate:.0%}")
    print()

    # Per-trade comparisons
    print("  TRADE-BY-TRADE COMPARISON")
    print("  " + "-" * 76)
    print(f"  {'#':>3} {'Dir':>4} {'Entry Time':>19} {'Entry Slip':>10} "
          f"{'PnL Delta':>9} {'Exit Match':>10} {'Live Exit':>12} {'BT Exit':>12}")
    print("  " + "-" * 76)

    for i, c in enumerate(report.comparisons):
        dir_str = "BUY" if c.live.direction == 1 else "SELL"
        entry_time = c.live.entry_time.strftime("%Y-%m-%d %H:%M") if c.live.entry_time else "N/A"
        exit_match = "YES" if c.exit_reason_match else "NO"

        print(f"  {i+1:3d} {dir_str:>4} {entry_time:>19} "
              f"{c.entry_slippage_pips:+10.2f} {c.pnl_delta:+9.2f} "
              f"{exit_match:>10} {c.live.exit_reason:>12} {c.backtest.exit_reason:>12}")

    print("  " + "-" * 76)

    # Detail section for diverged trades
    diverged = [c for c in report.comparisons if not c.exit_reason_match or abs(c.pnl_delta) >= 2.0]
    if diverged:
        print(f"\n  DIVERGED TRADES ({len(diverged)} trades)")
        print("  " + "-" * 76)
        for c in diverged:
            dir_str = "BUY" if c.live.direction == 1 else "SELL"
            print(f"\n  Trade: {dir_str} @ {c.live.entry_time}")
            print(f"    Entry:  live={c.live.entry_price:.5f}  bt={c.backtest.entry_price:.5f}  "
                  f"slip={c.entry_slippage_pips:+.2f} pips")
            print(f"    Exit:   live={c.live.exit_price:.5f}  bt={c.backtest.exit_price:.5f}  "
                  f"delta={c.exit_price_delta:+.2f} pips")
            print(f"    SL:     live={c.live.sl_price:.5f}  bt={c.backtest.sl_price:.5f}  "
                  f"delta={c.sl_delta:+.2f} pips")
            print(f"    TP:     live={c.live.tp_price:.5f}  bt={c.backtest.tp_price:.5f}  "
                  f"delta={c.tp_delta:+.2f} pips")
            print(f"    Reason: live={c.live.exit_reason}  bt={c.backtest.exit_reason}")
            print(f"    PnL:    live={c.live.pnl_pips:+.1f}  bt={c.backtest.pnl_pips:+.1f}  "
                  f"delta={c.pnl_delta:+.1f} pips")
            print(f"    Time:   entry_gap={c.entry_time_delta_minutes:+.0f}min  "
                  f"exit_gap={c.exit_time_delta_minutes:+.0f}min")
            if c.backtest.mfe_pips > 0:
                print(f"    BT MFE: {c.backtest.mfe_pips:.1f} pips  "
                      f"MAE: {c.backtest.mae_pips:.1f} pips  "
                      f"Held: {c.backtest.bars_held} bars")

    # Unmatched trades
    if report.unmatched_live_trades:
        print(f"\n  UNMATCHED LIVE TRADES ({len(report.unmatched_live_trades)})")
        print("  These trades happened live but have no backtest match:")
        for t in report.unmatched_live_trades:
            dir_str = "BUY" if t.direction == 1 else "SELL"
            print(f"    {t.entry_time} {dir_str} @ {t.entry_price:.5f} -> "
                  f"{t.exit_price:.5f} ({t.exit_reason}) PnL={t.pnl_pips:+.1f} pips")

    if report.unmatched_backtest_trades:
        n_show = min(20, len(report.unmatched_backtest_trades))
        total = len(report.unmatched_backtest_trades)
        print(f"\n  UNMATCHED BACKTEST TRADES (showing {n_show}/{total})")
        print("  These trades appear in backtest but not live:")
        for t in report.unmatched_backtest_trades[:n_show]:
            print(f"    {t.entry_time} {t.direction} @ {t.entry_price:.5f} -> "
                  f"{t.exit_price:.5f} ({t.exit_reason}) PnL={t.pnl_pips:+.1f} pips")

    print("\n" + "=" * 80)


def save_report(report: VerificationReport, output_dir: str | Path) -> Path:
    """Save verification report as JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = report.run_time.strftime("%Y%m%d_%H%M%S")
    filename = f"verification_{report.strategy_name}_{timestamp}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    logger.info("Report saved to %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _comparison_to_dict(c: TradeComparison) -> dict:
    return {
        "live": _live_trade_to_dict(c.live),
        "backtest": _bt_trade_to_dict(c.backtest),
        "deltas": {
            "entry_price_pips": c.entry_price_delta,
            "exit_price_pips": c.exit_price_delta,
            "sl_pips": c.sl_delta,
            "tp_pips": c.tp_delta,
            "pnl_pips": c.pnl_delta,
            "entry_time_minutes": c.entry_time_delta_minutes,
            "exit_time_minutes": c.exit_time_delta_minutes,
            "exit_reason_match": c.exit_reason_match,
            "direction_match": c.direction_match,
        },
    }


def _live_trade_to_dict(t: LiveTrade) -> dict:
    return {
        "position_id": t.position_id,
        "symbol": t.symbol,
        "direction": "BUY" if t.direction == 1 else "SELL",
        "volume": t.volume,
        "entry_price": t.entry_price,
        "entry_time": t.entry_time.isoformat() if t.entry_time else None,
        "exit_price": t.exit_price,
        "exit_time": t.exit_time.isoformat() if t.exit_time else None,
        "pnl_pips": round(t.pnl_pips, 2),
        "profit": t.profit,
        "commission": t.commission,
        "swap": t.swap,
        "exit_reason": t.exit_reason,
        "sl_price": t.sl_price,
        "tp_price": t.tp_price,
    }


def _bt_trade_to_dict(t: BacktestTrade) -> dict:
    return {
        "signal_index": t.signal_index,
        "direction": t.direction,
        "entry_price": t.entry_price,
        "entry_time": t.entry_time.isoformat() if t.entry_time else None,
        "exit_price": t.exit_price,
        "exit_time": t.exit_time.isoformat() if t.exit_time else None,
        "sl_price": t.sl_price,
        "tp_price": t.tp_price,
        "sl_pips": round(t.sl_pips, 2),
        "tp_pips": round(t.tp_pips, 2),
        "pnl_pips": round(t.pnl_pips, 2),
        "exit_reason": t.exit_reason,
        "bars_held": t.bars_held,
        "mfe_pips": round(t.mfe_pips, 2),
        "mae_pips": round(t.mae_pips, 2),
    }
