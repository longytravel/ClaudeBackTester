"""Download all 25 pairs with bid+ask spread data.

Run with: uv run python scripts/download_all.py
Progress is logged to console and to logs/download.log
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

# Set up file logging alongside console
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "download.log"

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)
log = structlog.get_logger()


def main():
    from backtester.data.downloader import ALL_PAIRS, DEFAULT_DATA_DIR, download_pair

    # Write progress to a simple status file for monitoring
    status_file = Path("logs/download_status.txt")

    total = len(ALL_PAIRS)
    completed = []
    failed = []

    log.info("download_all_start", pairs=total, data_dir=DEFAULT_DATA_DIR)
    t0 = time.time()

    for i, pair in enumerate(ALL_PAIRS):
        pair_start = time.time()
        status = f"[{i+1}/{total}] Downloading {pair}... (completed: {len(completed)}, failed: {len(failed)})"
        log.info(status)
        status_file.write_text(
            f"{status}\nStarted: {datetime.now(timezone.utc).isoformat()}\n"
            f"Completed: {', '.join(completed) or 'none'}\n"
            f"Failed: {', '.join(failed) or 'none'}\n"
        )

        try:
            path = download_pair(pair, DEFAULT_DATA_DIR, start_year=2005, force=True)
            elapsed = time.time() - pair_start
            completed.append(pair)
            log.info("pair_done", pair=pair, path=str(path), elapsed_min=round(elapsed / 60, 1))
        except Exception as e:
            failed.append(pair)
            log.exception("pair_failed", pair=pair, error=str(e))
            continue

    total_elapsed = time.time() - t0
    summary = (
        f"Download complete: {len(completed)}/{total} pairs in {total_elapsed/3600:.1f} hours\n"
        f"Completed: {', '.join(completed)}\n"
        f"Failed: {', '.join(failed) or 'none'}\n"
    )
    log.info(summary)
    status_file.write_text(summary)


if __name__ == "__main__":
    main()
