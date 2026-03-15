"""CLI entry point for backtest-to-live trade verification.

Usage:
    # Verify using MT5 (must be connected):
    uv run python scripts/verify_trades.py \
        --strategy ema_crossover --pair EURUSD --timeframe H1 \
        --checkpoint results/ema_eur_usd_h1/checkpoint.json

    # Verify using audit.jsonl (offline):
    uv run python scripts/verify_trades.py \
        --strategy ema_crossover --pair EURUSD --timeframe H1 \
        --checkpoint results/ema_eur_usd_h1/checkpoint.json \
        --audit state/ema_crossover_EURUSD_H1/audit.jsonl

    # Verify all deployed strategies:
    uv run python scripts/verify_trades.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def verify_single(
    strategy: str,
    pair: str,
    timeframe: str,
    checkpoint: str,
    candidate: int = 0,
    from_date: str | None = None,
    magic: int | None = None,
    audit: str | None = None,
    no_refresh: bool = False,
) -> None:
    """Run verification for a single strategy."""
    from backtester.verification.comparator import (
        print_report,
        run_verification,
        save_report,
    )

    from_dt = None
    if from_date:
        from_dt = datetime.fromisoformat(from_date).replace(tzinfo=timezone.utc)

    report = run_verification(
        strategy_name=strategy,
        pair=pair,
        timeframe=timeframe,
        checkpoint_path=checkpoint,
        candidate_index=candidate,
        from_date=from_dt,
        magic=magic,
        refresh_data=not no_refresh,
        use_audit=audit,
    )

    print_report(report)

    # Save report
    pair_file = pair.replace("/", "_").replace(" ", "")
    output_dir = Path("results") / f"{strategy}_{pair_file.lower()}_{timeframe.lower()}"
    filepath = save_report(report, output_dir)
    print(f"\nReport saved to: {filepath}")


def verify_all(
    from_date: str | None = None,
    no_refresh: bool = False,
) -> None:
    """Verify all deployed strategies by scanning results/ and state/ directories."""
    from backtester.verification.comparator import (
        print_report,
        run_verification,
        save_report,
    )

    results_dir = Path("results")
    state_dir = Path("state")

    if not results_dir.exists():
        print("No results/ directory found")
        return

    # Find all checkpoints
    checkpoints = list(results_dir.glob("*/checkpoint.json"))
    if not checkpoints:
        print("No checkpoint.json files found in results/")
        return

    from_dt = None
    if from_date:
        from_dt = datetime.fromisoformat(from_date).replace(tzinfo=timezone.utc)

    print(f"Found {len(checkpoints)} strategy checkpoints\n")

    for cp in sorted(checkpoints):
        # Parse strategy info from directory name: {strategy}_{pair}_{timeframe}
        dir_name = cp.parent.name
        parts = dir_name.rsplit("_", 2)
        if len(parts) < 3:
            logger.warning("Cannot parse directory name: %s", dir_name)
            continue

        timeframe = parts[-1].upper()
        pair_part = parts[-2].upper()
        strategy = "_".join(parts[:-2])

        # Reconstruct pair with slash for Dukascopy
        if len(pair_part) == 6:
            pair_display = pair_part[:3] + "/" + pair_part[3:]
        else:
            pair_display = pair_part

        # Look for audit file
        instance_id = f"{strategy}_{pair_part}_{timeframe}"
        audit_path = state_dir / instance_id / "audit.jsonl"
        use_audit = str(audit_path) if audit_path.exists() else None

        print(f"\n{'='*60}")
        print(f"  Verifying: {strategy} {pair_display} {timeframe}")
        print(f"{'='*60}")

        try:
            report = run_verification(
                strategy_name=strategy,
                pair=pair_display,
                timeframe=timeframe,
                checkpoint_path=str(cp),
                candidate_index=0,
                from_date=from_dt,
                refresh_data=not no_refresh,
                use_audit=use_audit,
            )
            print_report(report)

            output_dir = cp.parent
            save_report(report, output_dir)

        except Exception as e:
            logger.error("Failed to verify %s: %s", dir_name, e, exc_info=True)
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Verify backtest-to-live trade parity"
    )
    parser.add_argument("--strategy", help="Strategy name (e.g., ema_crossover)")
    parser.add_argument("--pair", help="Trading pair (e.g., EURUSD)")
    parser.add_argument("--timeframe", default="H1", help="Timeframe (default: H1)")
    parser.add_argument("--checkpoint", help="Path to checkpoint.json")
    parser.add_argument("--candidate", type=int, default=0, help="Candidate index (default: 0)")
    parser.add_argument("--from-date", help="Only check trades after this date (ISO format)")
    parser.add_argument("--magic", type=int, help="MT5 magic number (auto-computed if omitted)")
    parser.add_argument("--audit", help="Path to audit.jsonl (use instead of MT5)")
    parser.add_argument("--no-refresh", action="store_true", help="Skip data download/refresh")
    parser.add_argument("--all", action="store_true", help="Verify all deployed strategies")

    args = parser.parse_args()

    if args.all:
        verify_all(args.from_date, args.no_refresh)
    elif args.strategy and args.checkpoint:
        verify_single(
            strategy=args.strategy,
            pair=args.pair or "EURUSD",
            timeframe=args.timeframe,
            checkpoint=args.checkpoint,
            candidate=args.candidate,
            from_date=args.from_date,
            magic=args.magic,
            audit=args.audit,
            no_refresh=args.no_refresh,
        )
    else:
        parser.error("Either --all or (--strategy + --checkpoint) is required")


if __name__ == "__main__":
    main()
