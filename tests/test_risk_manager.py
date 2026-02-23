"""Tests for risk/manager.py — pre-trade checks and position sizing."""

import pytest

from backtester.live.config import LiveConfig
from backtester.live.types import DailyStats, LivePosition, TradingMode
from backtester.risk.manager import RiskManager

from datetime import datetime, timezone


@pytest.fixture
def config():
    return LiveConfig(
        strategy_name="test",
        pair="EURUSD",
        mode=TradingMode.DRY_RUN,
        risk_pct=1.0,
        max_daily_trades=5,
        max_daily_loss_pct=3.0,
        max_drawdown_pct=10.0,
        max_open_positions=3,
        max_spread_pips=3.0,
    )


@pytest.fixture
def rm(config):
    return RiskManager(config)


@pytest.fixture
def daily_stats():
    return DailyStats(date="2026-02-23", starting_balance=10000.0)


@pytest.fixture
def symbol_info():
    return {
        "point": 0.00001,
        "digits": 5,
        "trade_contract_size": 100000,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01,
    }


class TestPreTradeChecks:
    def test_all_clear(self, rm, daily_stats):
        result = rm.check_pre_trade(daily_stats, {}, 10000, 10000, 1.0, "EURUSD")
        assert result.allowed

    def test_circuit_breaker_blocks(self, rm, daily_stats):
        rm.trip_circuit_breaker("test reason")
        result = rm.check_pre_trade(daily_stats, {}, 10000, 10000, 1.0, "EURUSD")
        assert not result.allowed
        assert "Circuit breaker" in result.reason

    def test_daily_trade_limit(self, rm, daily_stats):
        daily_stats.trades_opened = 5  # max is 5
        result = rm.check_pre_trade(daily_stats, {}, 10000, 10000, 1.0, "EURUSD")
        assert not result.allowed
        assert "Daily trade limit" in result.reason

    def test_daily_loss_limit_trips_breaker(self, rm, daily_stats):
        daily_stats.net_pnl = -350.0  # 3.5% of 10000
        result = rm.check_pre_trade(daily_stats, {}, 10000, 10000, 1.0, "EURUSD")
        assert not result.allowed
        assert "Daily loss" in result.reason
        assert rm.circuit_breaker_active

    def test_drawdown_limit(self, rm, daily_stats):
        # Peak 10000, current 8900 = 11% drawdown > 10% limit
        result = rm.check_pre_trade(daily_stats, {}, 8900, 10000, 1.0, "EURUSD")
        assert not result.allowed
        assert "drawdown" in result.reason.lower()

    def test_max_open_positions(self, rm, daily_stats):
        positions = {}
        for i in range(3):
            positions[i] = LivePosition(
                ticket=i, symbol=f"PAIR{i}", direction=1, volume=0.1,
                original_volume=0.1, entry_price=1.0,
                entry_time=datetime.now(tz=timezone.utc),
                sl_price=0.99, tp_price=1.02, pip_value=0.0001,
            )
        result = rm.check_pre_trade(daily_stats, positions, 10000, 10000, 1.0, "EURUSD")
        assert not result.allowed
        assert "Max open" in result.reason

    def test_pair_occupancy(self, rm, daily_stats):
        positions = {
            1: LivePosition(
                ticket=1, symbol="EURUSD", direction=1, volume=0.1,
                original_volume=0.1, entry_price=1.0,
                entry_time=datetime.now(tz=timezone.utc),
                sl_price=0.99, tp_price=1.02, pip_value=0.0001,
            )
        }
        result = rm.check_pre_trade(daily_stats, positions, 10000, 10000, 1.0, "EURUSD")
        assert not result.allowed
        assert "Already have" in result.reason

    def test_spread_too_wide(self, rm, daily_stats):
        result = rm.check_pre_trade(daily_stats, {}, 10000, 10000, 5.0, "EURUSD")
        assert not result.allowed
        assert "Spread too wide" in result.reason

    def test_spread_check_disabled(self, config):
        config.max_spread_pips = 0.0
        rm = RiskManager(config)
        ds = DailyStats(date="2026-02-23", starting_balance=10000.0)
        result = rm.check_pre_trade(ds, {}, 10000, 10000, 99.0, "EURUSD")
        assert result.allowed


class TestPositionSizing:
    def test_basic_sizing(self, rm, symbol_info):
        # 1% of $10,000 = $100 risk. 30 pip SL. pip value = 0.0001 * 100000 = $10/pip
        # lots = 100 / (30 * 10) = 0.33
        lots = rm.calculate_position_size(10000, 30, symbol_info)
        assert lots == pytest.approx(0.33, abs=0.01)

    def test_small_sl_large_position(self, rm, symbol_info):
        # 10 pip SL: lots = 100 / (10 * 10) = 1.0
        lots = rm.calculate_position_size(10000, 10, symbol_info)
        assert lots == pytest.approx(1.0, abs=0.01)

    def test_zero_sl_returns_zero(self, rm, symbol_info):
        lots = rm.calculate_position_size(10000, 0, symbol_info)
        assert lots == 0.0

    def test_clamp_to_min(self, rm, symbol_info):
        # Tiny balance -> very small lots, clamped to volume_min
        lots = rm.calculate_position_size(10, 100, symbol_info)
        assert lots == 0.01  # volume_min

    def test_jpy_pair(self, rm):
        info = {
            "point": 0.001,
            "digits": 3,
            "trade_contract_size": 100000,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
        }
        # pip = 0.001 * 10 = 0.01. pip_value_per_lot = 0.01 * 100000 = 1000
        # risk = 100. sl = 30 pips. lots = 100 / (30 * 1000) = 0.0033 -> clamped to 0.01
        lots = rm.calculate_position_size(10000, 30, info)
        assert lots >= 0.01


class TestCircuitBreaker:
    def test_trip_and_reset(self, rm):
        assert not rm.circuit_breaker_active
        rm.trip_circuit_breaker("test")
        assert rm.circuit_breaker_active
        assert rm.circuit_breaker_reason == "test"
        rm.reset_circuit_breaker()
        assert not rm.circuit_breaker_active
        assert rm.circuit_breaker_reason == ""

    def test_get_status(self, rm):
        status = rm.get_status()
        assert "circuit_breaker_active" in status
        assert "risk_pct" in status
        assert status["risk_pct"] == 1.0
