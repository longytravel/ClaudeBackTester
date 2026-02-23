"""Auto-discover and launch all validated strategies.

Scans results/*/checkpoint.json for pipeline results.
Starts one live_trade.py process per strategy in the background.
SAFE: skips strategies that are already running (no interruption).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def find_strategies(results_dir: str) -> list[dict]:
    """Find all checkpoint.json files and extract strategy info."""
    strategies = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return strategies

    for checkpoint in results_path.glob("*/checkpoint.json"):
        try:
            with open(checkpoint) as f:
                data = json.load(f)

            candidates = data.get("candidates", [])
            if not candidates:
                continue

            # Skip eliminated candidates
            first = candidates[0]
            if first.get("eliminated", False):
                continue

            strategy_name = data.get("strategy_name", "")
            pair = data.get("pair", "").replace("/", "")  # EUR/USD -> EURUSD
            timeframe = data.get("timeframe", "H1")

            if not strategy_name or not pair:
                continue

            strategies.append({
                "strategy": strategy_name,
                "pair": pair,
                "timeframe": timeframe,
                "checkpoint": str(checkpoint),
                "quality": first.get("back_quality", 0),
                "rating": "",  # will check report if exists
                "dir_name": checkpoint.parent.name,
            })

            # Check report for rating
            report = checkpoint.parent / "report.json"
            if report.exists():
                with open(report) as f:
                    rdata = json.load(f)
                rcands = rdata.get("candidates", [])
                if rcands:
                    strategies[-1]["rating"] = rcands[0].get("rating", "")

        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    return strategies


def get_running_strategies() -> set[str]:
    """Get dir_names of strategies that already have a running process."""
    try:
        ps_cmd = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -like '*live_trade.py*' -and $_.Name -eq 'python.exe' } | "
            "Select-Object -ExpandProperty CommandLine"
        )
        result = subprocess.run(
            ["powershell", "-Command", ps_cmd],
            capture_output=True, text=True, timeout=10,
        )
        running = set()
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Extract --state-dir value to identify which strategy
            if "--state-dir" in line:
                parts = line.split("--state-dir")
                if len(parts) > 1:
                    state_dir = parts[1].strip().split()[0]
                    # dir_name is the last folder in the state path
                    running.add(Path(state_dir).name)
        return running
    except Exception:
        return set()


def main():
    mode = "practice"  # Default to demo account
    if len(sys.argv) > 1:
        mode = sys.argv[1]

    if mode == "live":
        print("\n" + "=" * 60)
        print("  WARNING: LIVE MODE — REAL MONEY")
        print("=" * 60)
        confirm = input("Type LIVE to confirm: ")
        if confirm.strip() != "LIVE":
            print("Aborted.")
            sys.exit(1)

    root = Path(__file__).parent.parent
    results_dir = root / "results"
    state_base = root / "state"
    state_base.mkdir(exist_ok=True)

    strategies = find_strategies(str(results_dir))

    if not strategies:
        print("\nNo validated strategies found in results/")
        print("Run optimization + validation first.")
        sys.exit(1)

    # Find which strategies are already running
    already_running = get_running_strategies()

    print(f"\n{'='*60}")
    print(f"  Found {len(strategies)} strategy(ies)")
    print(f"  Already running: {len(already_running)}")
    print(f"  Mode: {mode.upper()}")
    print(f"{'='*60}\n")

    launched = 0
    skipped_rating = 0
    skipped_running = 0

    for s in strategies:
        name = s["strategy"]
        pair = s["pair"]
        tf = s["timeframe"]
        rating = s["rating"]
        dir_name = s["dir_name"]
        state_dir = str(state_base / dir_name)

        # Only launch GREEN or AMBER rated strategies
        if rating and rating not in ("GREEN", "AMBER"):
            print(f"  SKIP:    {name} / {pair} / {tf}  [{rating}] — not GREEN/AMBER")
            skipped_rating += 1
            continue

        # Skip if already running (don't interrupt open trades!)
        if dir_name in already_running:
            print(f"  RUNNING: {name} / {pair} / {tf}  — already active, not touching it")
            skipped_running += 1
            continue

        tag = f"  [{rating}]" if rating else ""
        print(f"  START:   {name} / {pair} / {tf}{tag}")

        log_file = os.path.join(state_dir, "trader.log")
        os.makedirs(state_dir, exist_ok=True)

        cmd = [
            sys.executable,
            str(root / "scripts" / "live_trade.py"),
            "--strategy", name,
            "--pair", pair,
            "--timeframe", tf,
            "--pipeline", s["checkpoint"],
            "--mode", mode,
            "--state-dir", state_dir,
        ]

        with open(log_file, "a") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
            )

        print(f"           PID: {proc.pid} | Log: {log_file}")
        launched += 1

    print(f"\n{'='*60}")
    print(f"  {launched} new trader(s) launched")
    if skipped_running:
        print(f"  {skipped_running} already running (left alone)")
    if skipped_rating:
        print(f"  {skipped_rating} skipped (not GREEN/AMBER)")
    print(f"{'='*60}")
    print(f"\n  You can close this window now.")
    print(f"  Run STATUS.bat to check on them.")
    print(f"  Run STOP.bat to shut them all down.\n")


if __name__ == "__main__":
    main()
