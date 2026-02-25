"""Batch runner — run all strategy/pair/timeframe combinations.

Executes full_run.py for each combination, logging output per run.
Continues on errors, prints summary at end.

Usage:
    uv run python scripts/batch_run.py
    uv run python scripts/batch_run.py --preset standard
    uv run python scripts/batch_run.py --no-m1
    uv run python scripts/batch_run.py --strategies macd_crossover donchian_breakout
    uv run python scripts/batch_run.py --pairs EUR/USD GBP/USD
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

STRATEGIES = [
    "macd_crossover",
    "bollinger_reversion",
    "stochastic_crossover",
    "donchian_breakout",
    "adx_trend",
    "ema_crossover",
    "rsi_mean_reversion",
]

PAIRS = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "AUD/USD",
    "NZD/USD",
    "USD/CHF",
    "EUR/GBP",
    "EUR/JPY",
    "GBP/JPY",
    "AUD/JPY",
]

TIMEFRAME = "M15"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch pipeline runner")
    parser.add_argument("--preset", default="turbo", help="Optimization preset")
    parser.add_argument("--timeframe", default=TIMEFRAME, help="Timeframe")
    parser.add_argument("--no-m1", action="store_true", help="Disable M1 sub-bars")
    parser.add_argument(
        "--strategies", nargs="+", default=None,
        help="Only run these strategies (default: all 7)",
    )
    parser.add_argument(
        "--pairs", nargs="+", default=None,
        help="Only run these pairs (default: all 10)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip runs that already have a report.json",
    )
    return parser.parse_args()


def run_dir_name(strategy: str, pair: str, timeframe: str) -> str:
    return f"{strategy}_{pair.replace('/', '_').lower()}_{timeframe.lower()}"


def main():
    args = parse_args()
    strategies = args.strategies or STRATEGIES
    pairs = args.pairs or PAIRS
    timeframe = args.timeframe
    preset = args.preset

    combos = [(s, p) for s in strategies for p in pairs]
    n_total = len(combos)

    print("=" * 80)
    print("BATCH PIPELINE RUNNER")
    print("=" * 80)
    print(f"  Strategies:  {len(strategies)}")
    print(f"  Pairs:       {len(pairs)}")
    print(f"  Timeframe:   {timeframe}")
    print(f"  Preset:      {preset}")
    print(f"  M1 sub-bars: {'DISABLED' if args.no_m1 else 'ENABLED'}")
    print(f"  Total runs:  {n_total}")
    print(f"  Skip exist:  {args.skip_existing}")
    print(f"  Started:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = []
    t_batch_start = time.time()

    for idx, (strategy, pair) in enumerate(combos, 1):
        run_name = run_dir_name(strategy, pair, timeframe)
        output_dir = Path("results") / run_name
        log_file = output_dir / "run.log"

        print(f"\n{'─' * 70}")
        print(f"[{idx}/{n_total}] {strategy} × {pair} × {timeframe}")
        print(f"  Output: {output_dir}")

        # Skip if report already exists
        if args.skip_existing and (output_dir / "report.json").exists():
            print(f"  SKIPPED — report.json already exists")
            results.append((run_name, "SKIPPED", 0))
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "scripts/full_run.py",
            "--strategy", strategy,
            "--pair", pair,
            "--timeframe", timeframe,
            "--preset", preset,
            "--output", str(output_dir),
        ]
        if args.no_m1:
            cmd.append("--no-m1")

        t_start = time.time()
        try:
            with open(log_file, "w") as log_f:
                proc = subprocess.run(
                    cmd,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=7200,  # 2 hour max per run
                    cwd=str(Path.cwd()),
                )
            elapsed = time.time() - t_start

            if proc.returncode == 0:
                status = "OK"
                print(f"  DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
            else:
                status = f"FAIL(rc={proc.returncode})"
                print(f"  FAILED (exit code {proc.returncode}) after {elapsed:.0f}s")
                print(f"  Check log: {log_file}")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - t_start
            status = "TIMEOUT"
            print(f"  TIMEOUT after {elapsed:.0f}s")

        except Exception as e:
            elapsed = time.time() - t_start
            status = f"ERROR({type(e).__name__})"
            print(f"  ERROR: {e}")

        results.append((run_name, status, elapsed))

    # ============================================================
    # SUMMARY
    # ============================================================
    total_elapsed = time.time() - t_batch_start
    print(f"\n{'=' * 80}")
    print("BATCH RUN SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f} hours)")
    print(f"  Finished:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    ok_count = sum(1 for _, s, _ in results if s == "OK")
    skip_count = sum(1 for _, s, _ in results if s == "SKIPPED")
    fail_count = sum(1 for _, s, _ in results if s not in ("OK", "SKIPPED"))

    print(f"  OK:       {ok_count}")
    print(f"  Skipped:  {skip_count}")
    print(f"  Failed:   {fail_count}")
    print()

    if fail_count > 0:
        print("  FAILURES:")
        for name, status, elapsed in results:
            if status not in ("OK", "SKIPPED"):
                print(f"    {name:50s}  {status:20s}  {elapsed:.0f}s")

    # Write summary to file
    summary_path = Path("results") / "batch_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Batch run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total: {n_total}, OK: {ok_count}, Skipped: {skip_count}, "
                f"Failed: {fail_count}\n")
        f.write(f"Time: {total_elapsed:.0f}s ({total_elapsed/3600:.1f}h)\n\n")
        for name, status, elapsed in results:
            f.write(f"{name:50s}  {status:20s}  {elapsed:.0f}s\n")
    print(f"\n  Summary saved to {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
