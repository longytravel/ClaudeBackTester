"""Overnight download: re-download all pairs missing spread data.

Downloads with --force to clean old chunks and get fresh bid+ask data.
Retries failed pairs up to 3 times with exponential backoff.

Run with: uv run python scripts/download_overnight.py
"""

import time
from datetime import datetime, timezone
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

# Pairs that need re-downloading (no spread column or corrupt)
PAIRS_NEEDED = [
    # Corrupt files (Google Drive placeholders)
    "EUR/USD", "GBP/USD", "AUD/USD",
    # Old format (no spread column)
    "NZD/USD", "EUR/JPY", "AUD/JPY", "EUR/AUD", "EUR/CAD",
    "EUR/CHF", "GBP/AUD", "GBP/CAD", "GBP/CHF", "AUD/CAD",
    "AUD/NZD", "NZD/JPY", "CAD/JPY", "CHF/JPY", "EUR/NZD",
    "GBP/NZD", "XAU/USD",
]

MAX_RETRIES = 3
STATUS_FILE = Path("logs/download_overnight_status.txt")


def _update_status(msg: str) -> None:
    STATUS_FILE.parent.mkdir(exist_ok=True)
    STATUS_FILE.write_text(f"{msg}\nUpdated: {datetime.now(timezone.utc).isoformat()}\n")


def main():
    from backtester.data.downloader import DEFAULT_DATA_DIR, download_pair

    total = len(PAIRS_NEEDED)
    completed = []
    failed = []

    log.info("overnight_download_start", pairs=total, data_dir=DEFAULT_DATA_DIR)
    t0 = time.time()

    for i, pair in enumerate(PAIRS_NEEDED):
        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            pair_start = time.time()
            status = (
                f"[{i+1}/{total}] {pair} (attempt {attempt}/{MAX_RETRIES})\n"
                f"Completed: {len(completed)}/{total} — {', '.join(completed) or 'none yet'}\n"
                f"Failed: {', '.join(failed) or 'none'}"
            )
            log.info(status)
            _update_status(status)

            try:
                path = download_pair(pair, DEFAULT_DATA_DIR, start_year=2005, force=True)
                elapsed = time.time() - pair_start
                completed.append(pair)
                log.info("pair_done", pair=pair, path=str(path), elapsed_min=round(elapsed / 60, 1))
                success = True
                break
            except Exception as e:
                log.exception("pair_attempt_failed", pair=pair, attempt=attempt, error=str(e))
                if attempt < MAX_RETRIES:
                    backoff = 30 * attempt  # 30s, 60s, 90s
                    log.info(f"Retrying {pair} in {backoff}s...")
                    time.sleep(backoff)

        if not success:
            failed.append(pair)
            log.error("pair_exhausted_retries", pair=pair)

    total_elapsed = time.time() - t0
    summary = (
        f"Overnight download complete: {len(completed)}/{total} pairs in {total_elapsed/3600:.1f} hours\n"
        f"Completed ({len(completed)}): {', '.join(completed)}\n"
        f"Failed ({len(failed)}): {', '.join(failed) or 'none'}\n"
    )
    log.info(summary)
    _update_status(summary)


if __name__ == "__main__":
    main()
