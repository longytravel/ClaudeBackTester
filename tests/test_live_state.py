"""Tests for live/state.py — save/load round-trip and atomic writes."""

import json
import os
import tempfile
from datetime import datetime, timezone

import pytest

from backtester.live.state import append_audit, load_state, save_heartbeat, save_state
from backtester.live.types import DailyStats, LivePosition, TraderState


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_state():
    state = TraderState(
        instance_id="test_instance_123",
        circuit_breaker_active=False,
        last_bar_time="2026-02-23T12:00:00+00:00",
        peak_balance=10500.0,
        total_errors=2,
        last_error="timeout",
    )
    state.daily_stats = DailyStats(
        date="2026-02-23",
        starting_balance=10000.0,
        trades_opened=3,
        trades_closed=2,
        wins=1,
        losses=1,
        gross_pnl=50.0,
        net_pnl=43.0,
    )
    pos = LivePosition(
        ticket=99999,
        symbol="EURUSD",
        direction=1,
        volume=0.1,
        original_volume=0.1,
        entry_price=1.10500,
        entry_time=datetime(2026, 2, 23, 10, 0, tzinfo=timezone.utc),
        sl_price=1.10000,
        tp_price=1.11500,
        pip_value=0.0001,
        trailing_active=True,
        be_locked=True,
        bars_held=15,
        atr_pips=18.5,
        params={"trailing_mode": "fixed_pip", "trail_distance_pips": 10},
    )
    state.positions[99999] = pos
    return state


class TestSaveLoadRoundTrip:
    def test_round_trip(self, tmp_dir, sample_state):
        save_state(sample_state, tmp_dir)
        loaded = load_state(tmp_dir)

        assert loaded is not None
        assert loaded.instance_id == "test_instance_123"
        assert loaded.peak_balance == 10500.0
        assert loaded.total_errors == 2
        assert loaded.last_bar_time == "2026-02-23T12:00:00+00:00"

    def test_positions_round_trip(self, tmp_dir, sample_state):
        save_state(sample_state, tmp_dir)
        loaded = load_state(tmp_dir)

        assert 99999 in loaded.positions
        pos = loaded.positions[99999]
        assert pos.symbol == "EURUSD"
        assert pos.direction == 1
        assert pos.volume == 0.1
        assert pos.entry_price == 1.10500
        assert pos.trailing_active is True
        assert pos.be_locked is True
        assert pos.bars_held == 15
        assert pos.atr_pips == 18.5
        assert pos.params["trailing_mode"] == "fixed_pip"

    def test_daily_stats_round_trip(self, tmp_dir, sample_state):
        save_state(sample_state, tmp_dir)
        loaded = load_state(tmp_dir)

        ds = loaded.daily_stats
        assert ds.date == "2026-02-23"
        assert ds.starting_balance == 10000.0
        assert ds.trades_opened == 3
        assert ds.wins == 1

    def test_load_nonexistent_returns_none(self, tmp_dir):
        result = load_state(os.path.join(tmp_dir, "nope"))
        assert result is None

    def test_empty_state_round_trip(self, tmp_dir):
        state = TraderState(instance_id="empty")
        save_state(state, tmp_dir)
        loaded = load_state(tmp_dir)
        assert loaded.instance_id == "empty"
        assert len(loaded.positions) == 0

    def test_atomic_write_no_tmp_left(self, tmp_dir, sample_state):
        save_state(sample_state, tmp_dir)
        files = os.listdir(tmp_dir)
        assert "state.json" in files
        assert "state.json.tmp" not in files


class TestHeartbeat:
    def test_heartbeat_written(self, tmp_dir):
        save_heartbeat(tmp_dir, {"status": "ok", "positions": 2})
        path = os.path.join(tmp_dir, "heartbeat.json")
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)
        assert data["status"] == "ok"
        assert "timestamp" in data


class TestAuditTrail:
    def test_audit_append(self, tmp_dir):
        append_audit(tmp_dir, {"event": "order_placed", "ticket": 123})
        append_audit(tmp_dir, {"event": "position_closed", "ticket": 123})

        path = os.path.join(tmp_dir, "audit.jsonl")
        with open(path) as f:
            lines = f.readlines()

        assert len(lines) == 2
        entry1 = json.loads(lines[0])
        assert entry1["event"] == "order_placed"
        assert "timestamp" in entry1

        entry2 = json.loads(lines[1])
        assert entry2["event"] == "position_closed"
