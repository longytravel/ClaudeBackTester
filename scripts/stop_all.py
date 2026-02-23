"""Stop all running live traders."""

from __future__ import annotations

import subprocess
import sys


def main():
    print("\nStopping all live traders...")

    # Find and kill all python processes running live_trade.py
    result = subprocess.run(
        ["wmic", "process", "where",
         "commandline like '%live_trade.py%' and name='python.exe'",
         "get", "processid,commandline", "/format:list"],
        capture_output=True, text=True,
    )

    killed = 0
    for line in result.stdout.split("\n"):
        line = line.strip()
        if line.startswith("ProcessId="):
            pid = line.split("=")[1]
            try:
                subprocess.run(["taskkill", "/F", "/PID", pid],
                               capture_output=True)
                print(f"  Killed PID {pid}")
                killed += 1
            except Exception:
                pass

    if killed == 0:
        print("  No traders were running.")
    else:
        print(f"\n  Stopped {killed} trader(s).")


if __name__ == "__main__":
    main()
