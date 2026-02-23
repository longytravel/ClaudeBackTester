"""Tests for live/position_manager.py — mirrors telemetry.py check order."""

import pytest
from collections import deque
from datetime import datetime, timezone

from backtester.live.position_manager import manage_position
from backtester.live.types import LivePosition, ManagementActionType


def make_pos(direction=1, entry_price=1.10000, sl=1.09500, tp=1.11000, **kwargs):
    """Helper to create a LivePosition with sensible defaults."""
    defaults = dict(
        ticket=12345,
        symbol="EURUSD",
        direction=direction,
        volume=0.1,
        original_volume=0.1,
        entry_price=entry_price,
        entry_time=datetime.now(tz=timezone.utc),
        sl_price=sl,
        tp_price=tp,
        pip_value=0.0001,
        atr_pips=20.0,
        params={
            "max_bars": 0,
            "stale_exit_enabled": False,
            "breakeven_enabled": False,
            "trailing_mode": "off",
            "partial_close_enabled": False,
        },
    )
    defaults.update(kwargs)
    return LivePosition(**defaults)


class TestMaxBarsExit:
    def test_max_bars_triggers_close(self):
        pos = make_pos()
        pos.params["max_bars"] = 10
        pos.bars_held = 9  # will become 10 after manage_position increments

        actions = manage_position(pos, 1.1050, 1.0980, 1.1020)
        assert len(actions) == 1
        assert actions[0].action_type == ManagementActionType.CLOSE
        assert actions[0].reason == "max_bars"

    def test_max_bars_zero_disabled(self):
        pos = make_pos()
        pos.params["max_bars"] = 0
        pos.bars_held = 999

        actions = manage_position(pos, 1.1050, 1.0980, 1.1020)
        assert not any(a.action_type == ManagementActionType.CLOSE for a in actions)

    def test_max_bars_not_yet(self):
        pos = make_pos()
        pos.params["max_bars"] = 10
        pos.bars_held = 5  # will become 6

        actions = manage_position(pos, 1.1050, 1.0980, 1.1020)
        assert not any(a.action_type == ManagementActionType.CLOSE for a in actions)


class TestStaleExit:
    def test_stale_exit_triggers(self):
        pos = make_pos()
        pos.params["stale_exit_enabled"] = True
        pos.params["stale_exit_bars"] = 3
        pos.params["stale_exit_atr_threshold"] = 0.5
        pos.atr_pips = 20.0
        pos.bars_held = 2  # will become 3

        # Recent bars with tiny range (< 0.5 * 20 = 10 pips)
        recent = deque(maxlen=200)
        for _ in range(5):
            recent.append((1.1001, 1.0999))  # 0.2 pip range

        actions = manage_position(pos, 1.1001, 1.0999, 1.1000, recent)
        assert len(actions) == 1
        assert actions[0].action_type == ManagementActionType.CLOSE
        assert actions[0].reason == "stale_exit"

    def test_stale_exit_not_enough_bars(self):
        pos = make_pos()
        pos.params["stale_exit_enabled"] = True
        pos.params["stale_exit_bars"] = 50
        pos.bars_held = 5  # will become 6, < 50

        recent = deque(maxlen=200)
        for _ in range(5):
            recent.append((1.1001, 1.0999))

        actions = manage_position(pos, 1.1001, 1.0999, 1.1000, recent)
        assert not any(a.action_type == ManagementActionType.CLOSE for a in actions)


class TestBreakeven:
    def test_breakeven_locks_buy(self):
        pos = make_pos(direction=1, entry_price=1.10000, sl=1.09500, tp=1.11000)
        pos.params["breakeven_enabled"] = True
        pos.params["breakeven_trigger_pips"] = 20
        pos.params["breakeven_offset_pips"] = 2

        # bar_high = 1.1020 = 20 pips above entry
        actions = manage_position(pos, 1.10200, 1.10100, 1.10150)

        assert pos.be_locked
        expected_sl = 1.10000 + 2 * 0.0001  # entry + 2 pips = 1.10020
        assert pos.current_sl == pytest.approx(expected_sl, abs=1e-6)
        assert any(a.reason == "breakeven" for a in actions)

    def test_breakeven_locks_sell(self):
        pos = make_pos(direction=-1, entry_price=1.10000, sl=1.10500, tp=1.09000)
        pos.params["breakeven_enabled"] = True
        pos.params["breakeven_trigger_pips"] = 20
        pos.params["breakeven_offset_pips"] = 2

        # bar_low = 1.0980 = 20 pips below entry
        actions = manage_position(pos, 1.10000, 1.09800, 1.09900)

        assert pos.be_locked
        expected_sl = 1.10000 - 2 * 0.0001  # entry - 2 pips = 1.09980
        assert pos.current_sl == pytest.approx(expected_sl, abs=1e-6)

    def test_breakeven_one_shot(self):
        pos = make_pos()
        pos.params["breakeven_enabled"] = True
        pos.params["breakeven_trigger_pips"] = 20
        pos.params["breakeven_offset_pips"] = 2

        # First bar: trigger breakeven
        manage_position(pos, 1.10200, 1.10100, 1.10150)
        first_sl = pos.current_sl
        assert pos.be_locked

        # Second bar: shouldn't change BE
        manage_position(pos, 1.10300, 1.10200, 1.10250)
        assert pos.current_sl == first_sl  # BE doesn't move (unless trailing overrides)

    def test_breakeven_no_downgrade(self):
        """BE price below current SL should not move SL down (BUY)."""
        pos = make_pos(direction=1, entry_price=1.10000, sl=1.10010)
        pos.params["breakeven_enabled"] = True
        pos.params["breakeven_trigger_pips"] = 20
        pos.params["breakeven_offset_pips"] = 0

        # BE price = 1.10000 which is below current_sl = 1.10010
        manage_position(pos, 1.10200, 1.10100, 1.10150)
        assert pos.current_sl == 1.10010  # unchanged


class TestTrailingStop:
    def test_trailing_fixed_pip_buy(self):
        pos = make_pos(direction=1, entry_price=1.10000, sl=1.09500, tp=1.11000)
        pos.params["trailing_mode"] = "fixed_pip"
        pos.params["trail_activate_pips"] = 10
        pos.params["trail_distance_pips"] = 15

        # First bar: activate trailing (float_pnl = 15 pips >= 10)
        actions = manage_position(pos, 1.10150, 1.10050, 1.10100)
        assert pos.trailing_active

        # SL should be high - 15 pips = 1.10150 - 0.00150 = 1.10000
        expected_sl = 1.10150 - 15 * 0.0001
        assert pos.current_sl == pytest.approx(expected_sl, abs=1e-6)
        assert any(a.reason == "trailing_stop" for a in actions)

    def test_trailing_only_tightens_buy(self):
        pos = make_pos(direction=1, entry_price=1.10000, sl=1.09500, tp=1.11000)
        pos.params["trailing_mode"] = "fixed_pip"
        pos.params["trail_activate_pips"] = 0
        pos.params["trail_distance_pips"] = 10

        # Bar 1: high = 1.1020 -> SL = 1.1020 - 10 pips = 1.1010
        manage_position(pos, 1.10200, 1.10100, 1.10150)
        sl_after_bar1 = pos.current_sl

        # Bar 2: lower high = 1.1015 -> new_sl = 1.1005 < current (1.1010), no change
        manage_position(pos, 1.10150, 1.10050, 1.10100)
        assert pos.current_sl == sl_after_bar1

    def test_trailing_sell(self):
        pos = make_pos(direction=-1, entry_price=1.10000, sl=1.10500, tp=1.09000)
        pos.params["trailing_mode"] = "fixed_pip"
        pos.params["trail_activate_pips"] = 0
        pos.params["trail_distance_pips"] = 10

        # bar_low = 1.0985 -> float_pnl = 15 pips. new_sl = 1.0985 + 10 pips = 1.0995
        actions = manage_position(pos, 1.10000, 1.09850, 1.09900)
        assert pos.trailing_active
        expected_sl = 1.09850 + 10 * 0.0001
        assert pos.current_sl == pytest.approx(expected_sl, abs=1e-6)

    def test_trailing_atr_chandelier(self):
        pos = make_pos(direction=1, entry_price=1.10000, sl=1.09500, tp=1.11000)
        pos.params["trailing_mode"] = "atr_chandelier"
        pos.params["trail_activate_pips"] = 0
        pos.params["trail_atr_mult"] = 2.0
        pos.atr_pips = 15.0

        # trail distance = 2.0 * 15 pips = 30 pips = 0.0030
        # high = 1.1050 -> new_sl = 1.1050 - 0.0030 = 1.1020
        manage_position(pos, 1.10500, 1.10200, 1.10400)
        expected_sl = 1.10500 - 2.0 * 15.0 * 0.0001
        assert pos.current_sl == pytest.approx(expected_sl, abs=1e-6)


class TestPartialClose:
    def test_partial_close_buy(self):
        pos = make_pos(direction=1, entry_price=1.10000, sl=1.09500, tp=1.11000)
        pos.params["partial_close_enabled"] = True
        pos.params["partial_close_pct"] = 50
        pos.params["partial_close_trigger_pips"] = 20

        # float_pnl = 25 pips >= 20
        actions = manage_position(pos, 1.10250, 1.10100, 1.10200)

        assert pos.partial_done
        assert pos.position_pct == pytest.approx(0.5)
        assert len(actions) == 1
        assert actions[0].action_type == ManagementActionType.PARTIAL_CLOSE

        # Realized PnL: (1.10200 - 1.10000) / 0.0001 * 0.5 = 20 * 0.5 = 10 pips
        assert pos.realized_pnl_pips == pytest.approx(10.0, abs=0.1)

    def test_partial_close_one_shot(self):
        pos = make_pos()
        pos.params["partial_close_enabled"] = True
        pos.params["partial_close_pct"] = 50
        pos.params["partial_close_trigger_pips"] = 20

        # First trigger
        manage_position(pos, 1.10250, 1.10100, 1.10200)
        assert pos.partial_done

        # Second time: should NOT trigger again
        actions = manage_position(pos, 1.10350, 1.10200, 1.10300)
        assert not any(a.action_type == ManagementActionType.PARTIAL_CLOSE for a in actions)


class TestCombinedManagement:
    def test_breakeven_then_trailing(self):
        """Breakeven locks first, then trailing takes over."""
        pos = make_pos(direction=1, entry_price=1.10000, sl=1.09500, tp=1.11000)
        pos.params["breakeven_enabled"] = True
        pos.params["breakeven_trigger_pips"] = 10
        pos.params["breakeven_offset_pips"] = 2
        pos.params["trailing_mode"] = "fixed_pip"
        pos.params["trail_activate_pips"] = 20
        pos.params["trail_distance_pips"] = 10

        # Bar 1: trigger BE (float_pnl = 15 >= 10), but not trailing (15 < 20)
        manage_position(pos, 1.10150, 1.10050, 1.10100)
        assert pos.be_locked
        assert not pos.trailing_active
        be_sl = pos.current_sl  # should be entry + 2 pips = 1.10020

        # Bar 2: trigger trailing (float_pnl = 25 >= 20)
        manage_position(pos, 1.10250, 1.10150, 1.10200)
        assert pos.trailing_active
        # trailing SL = high - 10 pips = 1.10250 - 0.0010 = 1.10150
        assert pos.current_sl > be_sl

    def test_bars_held_increments(self):
        pos = make_pos()
        assert pos.bars_held == 0
        manage_position(pos, 1.1010, 1.0990, 1.1000)
        assert pos.bars_held == 1
        manage_position(pos, 1.1010, 1.0990, 1.1000)
        assert pos.bars_held == 2
