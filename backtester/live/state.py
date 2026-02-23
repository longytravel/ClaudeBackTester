"""State persistence for the live trading engine.

Follows the atomic-write pattern from pipeline/checkpoint.py:
write to temp file, then rename.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from backtester.live.types import TraderState

logger = logging.getLogger(__name__)


def save_state(state: TraderState, state_dir: str) -> None:
    """Save trader state atomically."""
    os.makedirs(state_dir, exist_ok=True)
    filepath = os.path.join(state_dir, "state.json")
    tmp_path = filepath + ".tmp"

    data = state.to_dict()
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(tmp_path, filepath)
    logger.debug(f"State saved to {filepath}")


def load_state(state_dir: str) -> TraderState | None:
    """Load trader state from disk. Returns None if no state file."""
    filepath = os.path.join(state_dir, "state.json")
    if not os.path.exists(filepath):
        return None

    with open(filepath) as f:
        data = json.load(f)

    state = TraderState.from_dict(data)
    logger.info(
        f"State loaded: {len(state.positions)} positions, "
        f"last_bar={state.last_bar_time}"
    )
    return state


def save_heartbeat(state_dir: str, status: dict) -> None:
    """Write heartbeat file (REQ-L30) for external monitoring."""
    os.makedirs(state_dir, exist_ok=True)
    filepath = os.path.join(state_dir, "heartbeat.json")
    tmp_path = filepath + ".tmp"

    status["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    with open(tmp_path, "w") as f:
        json.dump(status, f, indent=2)

    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(tmp_path, filepath)


def append_audit(state_dir: str, entry: dict) -> None:
    """Append to audit trail (REQ-L14). Append-only JSONL file."""
    os.makedirs(state_dir, exist_ok=True)
    filepath = os.path.join(state_dir, "audit.jsonl")

    entry["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    with open(filepath, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
