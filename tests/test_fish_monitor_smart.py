"""Tests for portfolio.fish_monitor_smart — signal-aware position monitoring."""

import time
from unittest.mock import MagicMock, patch

import pytest

from portfolio.fish_monitor_smart import SmartFishMonitor


@pytest.fixture
def monitor():
    """Create a basic monitor for testing."""
    return SmartFishMonitor(
        ticker="XAG-USD",
        entry_price=75.00,
        direction="SHORT",
        entry_conviction=65,
        cert_entry_price=7.20,
        cert_units=180,
        cert_leverage=10.0,
    )


class TestMonitorInit:
    def test_basic_init(self, monitor):
        assert monitor.ticker == "XAG-USD"
        assert monitor.entry_price == 75.00
        assert monitor.direction == "SHORT"
        assert monitor.entry_conviction == 65
        assert monitor.binance_symbol == "XAGUSDT"

    def test_long_direction(self):
        m = SmartFishMonitor("XAU-USD", entry_price=4600, direction="LONG")
        assert m.direction == "LONG"
        assert m.binance_symbol == "XAUUSDT"

    def test_default_state(self, monitor):
        assert monitor.current_price == 75.00
        assert monitor.session_high == 75.00
        assert monitor.session_low == 75.00
        assert monitor.check_count == 0


class TestExitSignals:
    def test_tp0_triggers_at_5pct_short(self, monitor):
        # For SHORT: entry 75, price drops to 71.25 = +5% in our favor
        monitor.current_price = 71.25
        exits = monitor.compute_exit_signals()
        triggers = [e["trigger"] for e in exits]
        assert "TP0" in triggers

    def test_tp0_triggers_at_5pct_long(self):
        m = SmartFishMonitor("XAG-USD", entry_price=75.00, direction="LONG")
        m.current_price = 78.75  # +5%
        exits = m.compute_exit_signals()
        triggers = [e["trigger"] for e in exits]
        assert "TP0" in triggers

    def test_tp_partial_at_2_5pct(self, monitor):
        monitor.current_price = 73.125  # SHORT entry 75, 2.5% drop
        exits = monitor.compute_exit_signals()
        triggers = [e["trigger"] for e in exits]
        assert "TP_PARTIAL" in triggers

    def test_no_exit_at_small_move(self, monitor):
        monitor.current_price = 74.50  # only 0.67% move
        exits = monitor.compute_exit_signals()
        assert len(exits) == 0

    def test_conviction_drop_alert(self, monitor):
        monitor.current_conviction = 40  # dropped from 65 to 40 = 25 pts
        exits = monitor.compute_exit_signals()
        triggers = [e["trigger"] for e in exits]
        assert "CONVICTION_DROP" in triggers

    def test_no_conviction_alert_small_drop(self, monitor):
        monitor.current_conviction = 55  # only 10 pt drop
        exits = monitor.compute_exit_signals()
        triggers = [e["trigger"] for e in exits]
        assert "CONVICTION_DROP" not in triggers

    def test_time_decay_3h(self, monitor):
        monitor.start_time = time.time() - 3.5 * 3600  # 3.5 hours ago
        exits = monitor.compute_exit_signals()
        triggers = [e["trigger"] for e in exits]
        assert "TIME_DECAY_3H" in triggers

    def test_time_decay_5h(self, monitor):
        monitor.start_time = time.time() - 5.5 * 3600
        exits = monitor.compute_exit_signals()
        triggers = [e["trigger"] for e in exits]
        assert "TIME_DECAY_5H" in triggers

    def test_adverse_move_alert(self, monitor):
        # SHORT at 75, price goes UP to 77.25 = -3% adverse
        monitor.current_price = 77.25
        exits = monitor.compute_exit_signals()
        triggers = [e["trigger"] for e in exits]
        assert "ADVERSE_MOVE" in triggers

    def test_alerts_not_duplicated(self, monitor):
        monitor.current_price = 71.25  # TP0
        exits1 = monitor.compute_exit_signals()
        assert len(exits1) > 0

        # Simulate what run() does: mark alerts as sent
        for ex in exits1:
            monitor.alerts_sent.add(ex["trigger"])

        # Second call should not re-trigger same alerts
        exits2 = monitor.compute_exit_signals()
        assert "TP0" not in [e["trigger"] for e in exits2]

    def test_cross_asset_divergence_short(self, monitor):
        # Set up: gold rallied (adverse for short silver since gold leads silver)
        monitor.cross_asset_baselines = {"gold": 4600.0}
        monitor.cross_asset_prices = {"gold": 4690.0}  # +1.96% (above threshold 0.5%)
        exits = monitor.compute_exit_signals()
        triggers = [e["trigger"] for e in exits]
        assert "CROSS_ASSET_GOLD" in triggers


class TestFormatStatus:
    def test_basic_format(self, monitor):
        monitor.current_price = 74.50
        status = monitor._format_status()
        assert "XAG-USD" in status
        assert "$74.50" in status

    def test_format_with_signal_data(self, monitor):
        signal_data = {
            "rsi": 65.0,
            "regime": "trending-up",
            "action": "BUY",
            "buy_count": 8,
            "sell_count": 2,
            "focus_3h": {"direction": "up", "probability": 0.6},
            "focus_1d": {"direction": "down", "probability": 0.55},
        }
        status = monitor._format_status(signal_data)
        assert "RSI 65.0" in status
        assert "trending-up" in status
        assert "8B/2S" in status

    def test_format_with_cert_pnl(self, monitor):
        monitor.current_price = 73.50  # SHORT entry 75, +2% in favor
        status = monitor._format_status()
        assert "Cert P&L" in status


class TestMonitorRun:
    @patch.object(SmartFishMonitor, "_fetch_price", return_value=74.80)
    @patch.object(SmartFishMonitor, "_fetch_cross_asset_prices", return_value={})
    @patch("portfolio.fish_monitor_smart.atomic_append_jsonl")
    @patch("portfolio.fish_monitor_smart.atomic_write_json")
    def test_run_limited_checks(self, mock_write, mock_append, mock_ca, mock_price):
        m = SmartFishMonitor("XAG-USD", entry_price=75.00, direction="SHORT")
        exits = m.run(max_checks=2, quiet=True)
        assert m.check_count == 2
        assert mock_append.call_count == 2  # one log per check
        assert mock_write.call_count == 1   # final state
