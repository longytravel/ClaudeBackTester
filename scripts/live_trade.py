"""CLI entry point for the live trading engine.

Usage:
    uv run python scripts/live_trade.py \
        --strategy rsi_mean_reversion \
        --pair EURUSD --timeframe H1 \
        --pipeline results/report.json \
        --mode practice
"""

from __future__ import annotations

import argparse
import json
import sys

# Ensure strategy registry is populated — import all strategies
import backtester.strategies  # noqa: F401
import backtester.strategies.rsi_mean_reversion  # noqa: F401
import backtester.strategies.always_buy  # noqa: F401

from backtester.live.config import LiveConfig
from backtester.live.trader import LiveTrader
from backtester.live.types import TradingMode


def main():
    parser = argparse.ArgumentParser(description="Live trading engine")
    parser.add_argument("--strategy", required=True, help="Strategy name (e.g. rsi_mean_reversion)")
    parser.add_argument("--pair", required=True, help="Symbol (e.g. EURUSD)")
    parser.add_argument("--timeframe", default="H1", help="Timeframe (M1/M5/M15/M30/H1/H4/D)")
    parser.add_argument("--pipeline", required=True, help="Path to pipeline checkpoint JSON")
    parser.add_argument("--candidate", type=int, default=0, help="Candidate index (default 0)")
    parser.add_argument(
        "--mode",
        choices=["dry_run", "practice", "live"],
        default="dry_run",
        help="Trading mode",
    )
    parser.add_argument("--config", help="JSON config file (overrides CLI args)")
    parser.add_argument("--risk-pct", type=float, default=1.0, help="Risk %% per trade")
    parser.add_argument("--max-spread", type=float, default=3.0, help="Max spread in pips")
    parser.add_argument("--fixed-lots", type=float, default=0.0, help="Fixed lot size (0 = risk-based)")
    parser.add_argument("--state-dir", default="", help="State directory")
    parser.add_argument("--lookback", type=int, default=500, help="Candle lookback count")

    args = parser.parse_args()

    # Build config
    if args.config:
        with open(args.config) as f:
            cfg_data = json.load(f)
        config = LiveConfig(
            strategy_name=cfg_data.get("strategy_name", args.strategy),
            pair=cfg_data.get("pair", args.pair),
            timeframe=cfg_data.get("timeframe", args.timeframe),
            pipeline_json=cfg_data.get("pipeline_json", args.pipeline),
            candidate_index=cfg_data.get("candidate_index", args.candidate),
            mode=TradingMode(cfg_data.get("mode", args.mode)),
            risk_pct=cfg_data.get("risk_pct", args.risk_pct),
            fixed_lot_size=cfg_data.get("fixed_lot_size", args.fixed_lots),
            max_spread_pips=cfg_data.get("max_spread_pips", args.max_spread),
            state_dir=cfg_data.get("state_dir", args.state_dir),
            lookback_bars=cfg_data.get("lookback_bars", args.lookback),
        )
    else:
        config = LiveConfig(
            strategy_name=args.strategy,
            pair=args.pair,
            timeframe=args.timeframe,
            pipeline_json=args.pipeline,
            candidate_index=args.candidate,
            mode=TradingMode(args.mode),
            risk_pct=args.risk_pct,
            fixed_lot_size=args.fixed_lots,
            max_spread_pips=args.max_spread,
            state_dir=args.state_dir,
            lookback_bars=args.lookback,
        )

    # LIVE mode safety gate
    if config.mode == TradingMode.LIVE:
        print("\n" + "=" * 60)
        print("WARNING: LIVE TRADING MODE")
        print("This will place real orders with real money.")
        print("=" * 60)
        confirm = input("\nType LIVE to confirm: ")
        if confirm.strip() != "LIVE":
            print("Aborted.")
            sys.exit(1)

    print(f"\nStarting live trader:")
    print(f"  Strategy:  {config.strategy_name}")
    print(f"  Pair:      {config.pair}")
    print(f"  Timeframe: {config.timeframe}")
    print(f"  Mode:      {config.mode.value}")
    print(f"  Pipeline:  {config.pipeline_json}")
    print(f"  Candidate: {config.candidate_index}")
    print(f"  Risk:      {config.risk_pct}%")
    print(f"  State dir: {config.state_dir}")
    print()

    trader = LiveTrader(config)
    trader.start()


if __name__ == "__main__":
    main()
