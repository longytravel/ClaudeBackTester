"""Configuration for the live trading engine."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from backtester.live.types import TradingMode


@dataclass
class LiveConfig:
    """Configuration for a live trading instance."""
    # Identity
    strategy_name: str = ""
    pair: str = ""             # e.g. "EURUSD"
    timeframe: str = "H1"
    pipeline_json: str = ""    # path to pipeline checkpoint JSON
    candidate_index: int = 0   # which candidate from the pipeline
    instance_id: str = ""
    mode: TradingMode = TradingMode.DRY_RUN

    # Risk settings (FR-7)
    risk_pct: float = 1.0              # % of balance to risk per trade
    max_daily_trades: int = 10         # REQ-R05
    max_daily_loss_pct: float = 3.0    # REQ-R06 — % of starting daily balance
    max_drawdown_pct: float = 10.0     # REQ-R08 — % from peak balance
    max_open_positions: int = 3        # REQ-R09
    max_spread_pips: float = 3.0       # REQ-R07

    # Position sizing override — set > 0 to use fixed lot size instead of risk-based
    fixed_lot_size: float = 0.0        # 0 = use risk-based sizing

    # Execution costs
    slippage_pips: float = 0.5
    commission_per_lot: float = 7.0    # USD per round trip

    # State persistence
    state_dir: str = ""

    # Candle lookback for signal generation (how many bars to fetch)
    lookback_bars: int = 500

    def __post_init__(self):
        if not self.instance_id:
            self.instance_id = f"{self.strategy_name}_{self.pair}_{self.timeframe}_{uuid.uuid4().hex[:8]}"
        if not self.state_dir:
            self.state_dir = f"state/{self.instance_id}"
