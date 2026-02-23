"""Show status of all running traders."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    state_base = root / "state"

    # Check which python processes are running live_trade.py
    result = subprocess.run(
        ["wmic", "process", "where",
         "commandline like '%live_trade.py%' and name='python.exe'",
         "get", "processid", "/format:list"],
        capture_output=True, text=True,
    )
    running_pids = set()
    for line in result.stdout.split("\n"):
        line = line.strip()
        if line.startswith("ProcessId="):
            running_pids.add(line.split("=")[1])

    print(f"\n{'='*65}")
    print(f"  TRADER STATUS  |  {len(running_pids)} process(es) running")
    print(f"{'='*65}")

    if not state_base.exists():
        print("\n  No state directory found. Nothing has been deployed yet.")
        return

    for state_dir in sorted(state_base.iterdir()):
        if not state_dir.is_dir():
            continue

        name = state_dir.name
        hb_path = state_dir / "heartbeat.json"
        log_path = state_dir / "trader.log"

        print(f"\n  --- {name} ---")

        # Heartbeat
        if hb_path.exists():
            try:
                with open(hb_path) as f:
                    hb = json.load(f)
                ts = hb.get("timestamp", "?")
                positions = hb.get("positions", 0)
                trades = hb.get("daily_trades", 0)
                cb = hb.get("circuit_breaker", False)
                errors = hb.get("errors", 0)
                mode = hb.get("mode", "?")

                # How old is the heartbeat?
                try:
                    hb_time = datetime.fromisoformat(ts)
                    age_secs = (datetime.now(tz=timezone.utc) - hb_time).total_seconds()
                    if age_secs < 120:
                        age_str = "FRESH"
                    elif age_secs < 7200:
                        age_str = f"{int(age_secs/60)}min ago"
                    else:
                        age_str = f"{int(age_secs/3600)}h ago"
                except Exception:
                    age_str = "?"

                status = "OK" if not cb else "CIRCUIT BREAKER"
                print(f"    Status:    {status}  ({age_str})")
                print(f"    Mode:      {mode}")
                print(f"    Positions: {positions}  |  Daily trades: {trades}  |  Errors: {errors}")
            except (json.JSONDecodeError, KeyError):
                print(f"    Heartbeat: corrupt")
        else:
            print(f"    Heartbeat: none (not started yet?)")

        # Last log lines
        if log_path.exists():
            try:
                with open(log_path) as f:
                    lines = f.readlines()
                last_lines = lines[-3:] if len(lines) >= 3 else lines
                print(f"    Last log:")
                for line in last_lines:
                    print(f"      {line.rstrip()}")
            except Exception:
                pass

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
