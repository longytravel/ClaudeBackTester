"""Show status of all running traders."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def count_running_traders() -> int:
    """Count python processes running live_trade.py."""
    try:
        ps_cmd = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -like '*live_trade.py*' -and $_.Name -eq 'python.exe' } | "
            "Measure-Object | Select-Object -ExpandProperty Count"
        )
        result = subprocess.run(
            ["powershell", "-Command", ps_cmd],
            capture_output=True, text=True,
        )
        return int(result.stdout.strip()) if result.stdout.strip() else 0
    except Exception:
        return -1  # unknown


def main():
    root = Path(__file__).parent.parent
    state_base = root / "state"

    running = count_running_traders()
    running_str = str(running) if running >= 0 else "?"

    print(f"\n{'='*65}")
    print(f"  TRADER STATUS  |  {running_str} process(es) running")
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
