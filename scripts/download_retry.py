"""Retry failed pairs from the main download batch.

Uses the fixed code that cleans old chunks before force re-downloading.
Run with: uv run python scripts/download_retry.py
"""

import time
from pathlib import Path

import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)
log = structlog.get_logger()

# Pairs that failed due to old corrupt chunks
RETRY_PAIRS = ["EUR/USD", "GBP/USD", "AUD/USD"]


def main():
    from backtester.data.downloader import DEFAULT_DATA_DIR, download_pair

    status_file = Path("logs/download_retry_status.txt")

    completed = []
    failed = []

    for i, pair in enumerate(RETRY_PAIRS):
        log.info(f"[{i+1}/{len(RETRY_PAIRS)}] Retrying {pair}...")
        status_file.write_text(f"Retrying {pair} ({i+1}/{len(RETRY_PAIRS)})\n")

        try:
            path = download_pair(pair, DEFAULT_DATA_DIR, start_year=2005, force=True)
            completed.append(pair)
            log.info("retry_done", pair=pair, path=str(path))
        except Exception as e:
            failed.append(pair)
            log.exception("retry_failed", pair=pair, error=str(e))

    summary = f"Retry complete: {len(completed)}/{len(RETRY_PAIRS)}\nCompleted: {completed}\nFailed: {failed}\n"
    log.info(summary)
    status_file.write_text(summary)


if __name__ == "__main__":
    main()
