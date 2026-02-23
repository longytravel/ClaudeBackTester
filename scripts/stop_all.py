"""Stop all running live traders."""

from __future__ import annotations

import subprocess
import sys


def main():
    print("\nStopping all live traders...")

    # Use PowerShell to find python processes with live_trade.py in command line
    ps_cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*live_trade.py*' -and $_.Name -eq 'python.exe' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    result = subprocess.run(
        ["powershell", "-Command", ps_cmd],
        capture_output=True, text=True,
    )

    pids = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    killed = 0

    for pid in pids:
        try:
            subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
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
