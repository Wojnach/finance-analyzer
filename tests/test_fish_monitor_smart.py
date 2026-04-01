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
    """Test exit signal detection with signal_data parameter."""

    def _sig(self, **overrides):
        """Create default signal data dict with optional overrides."""
        defaults = {
            "rsi": 50, "mc_p_up": 0.5, "buy_count": 5, "sell_count": 5,
            "action": "HOLD", "metals_action": "HOLD", "news_action": "HOLD",
        }
        defaults.update(overrides)
        return defaults

    def test_combined_exit_long_rsi62_mc35(self):
        """Lesson 45: RSI>62 + MC<35% triggers COMBINED_EXIT for LONG."""
        m = SmartFishMonitor("XAG-USD", entry_price=75.00, direction="LONG")
        m.current_price = 76.00
        exits = m.compute_exit_signals(self._sig(rsi=65, mc_p_up=0.28))
        triggers = [e["trigger"] for e in exits]
        assert "COMBINED_EXIT" in triggers

    def test_combined_exit_does_not_fire_for_short(self):
        """Backtest shows combined exit doesn't work for shorts."""
        m = SmartFishMonitor("XAG-USD", entry_price=75.00, direction="SHORT")
        m.current_price = 74.00
        exits = m.compute_exit_signals(self._sig(rsi=35, mc_p_up=0.70))
        triggers = [e["trigger"] for e in exits]
        assert "COMBINED_EXIT" not in triggers

    def test_short_exit_rsi30_solo(self):
        """SHORT exits on RSI<30 alone (backtest validated)."""
        m = SmartFishMonitor("XAG-USD", entry_price=75.00, direction="SHORT")
        m.current_price = 73.00
        exits = m.compute_exit_signals(self._sig(rsi=28))
        triggers = [e["trigger"] for e in exits]
        assert "RSI_EXIT" in triggers

    def test_no_combined_exit_below_threshold(self):
        """RSI 60 + MC 40% should NOT trigger combined exit."""
        m = SmartFishMonitor("XAG-USD", entry_price=75.00, direction="LONG")
        m.current_price = 76.00
        exits = m.compute_exit_signals(self._sig(rsi=60, mc_p_up=0.40))
        triggers = [e["trigger"] for e in exits]
        assert "COMBINED_EXIT" not in triggers

    def test_signal_flip_4_margin(self):
        """Need 4+ vote margin for signal flip exit."""
        m = SmartFishMonitor("XAG-USD", entry_price=75.00, direction="LONG")
        m.current_price = 75.50
        # 3 margin: should NOT trigger
        exits = m.compute_exit_signals(self._sig(buy_count=2, sell_count=5))
        assert "SIGNAL_FLIP" not in [e["trigger"] for e in exits]
        # 5 margin: SHOULD trigger
        exits = m.compute_exit_signals(self._sig(buy_count=1, sell_count=6))
        assert "SIGNAL_FLIP" in [e["trigger"] for e in exits]

    def test_tp_at_2pct(self, monitor):
        # SHORT at 75, price drops to 73.5 = +2%
        monitor.current_price = 73.50
        exits = monitor.compute_exit_signals(self._sig())
        triggers = [e["trigger"] for e in exits]
        assert "TP" in triggers

    def test_sl_at_minus_2pct(self, monitor):
        # SHORT at 75, price goes UP to 76.50 = -2%
        monitor.current_price = 76.50
        exits = monitor.compute_exit_signals(self._sig())
        triggers = [e["trigger"] for e in exits]
        assert "SL" in triggers

    def test_time_decay_3h(self, monitor):
        monitor.start_time = time.time() - 3.5 * 3600
        exits = monitor.compute_exit_signals(self._sig())
        triggers = [e["trigger"] for e in exits]
        assert "TIME_DECAY_3H" in triggers

    def test_adverse_move_alert(self, monitor):
        monitor.current_price = 77.25  # SHORT at 75, -3%
        exits = monitor.compute_exit_signals(self._sig())
        triggers = [e["trigger"] for e in exits]
        assert "ADVERSE_MOVE" in triggers

    def test_alerts_not_duplicated(self, monitor):
        monitor.current_price = 73.50  # TP
        exits1 = monitor.compute_exit_signals(self._sig())
        for ex in exits1:
            monitor.alerts_sent.add(ex["trigger"])
        exits2 = monitor.compute_exit_signals(self._sig())
        assert "TP" not in [e["trigger"] for e in exits2]

    def test_news_adverse_long(self):
        """Lesson 50: news SELL while holding LONG triggers warning."""
        m = SmartFishMonitor("XAG-USD", entry_price=75.00, direction="LONG")
        m.current_price = 75.50
        exits = m.compute_exit_signals(self._sig(news_action="SELL"))
        triggers = [e["trigger"] for e in exits]
        assert "NEWS_ADVERSE" in triggers

    def test_news_not_adverse_when_aligned(self):
        """news BUY while LONG should NOT trigger."""
        m = SmartFishMonitor("XAG-USD", entry_price=75.00, direction="LONG")
        m.current_price = 75.50
        exits = m.compute_exit_signals(self._sig(news_action="BUY"))
        triggers = [e["trigger"] for e in exits]
        assert "NEWS_ADVERSE" not in triggers


class TestMetalsDisagreement:
    """Lesson 53: metals loop disagreement detection."""

    def test_disagreement_counter_increments(self, monitor):
        # SHORT position, metals says BUY = disagrees
        monitor.update_signal_state({"mc_p_up": 0.5, "metals_action": "BUY"})
        assert monitor.metals_disagree_count == 1
        monitor.update_signal_state({"mc_p_up": 0.5, "metals_action": "BUY"})
        assert monitor.metals_disagree_count == 2

    def test_disagreement_resets_on_agreement(self, monitor):
        # SHORT position: BUY disagrees, then SELL agrees
        monitor.update_signal_state({"mc_p_up": 0.5, "metals_action": "BUY"})
        assert monitor.metals_disagree_count == 1
        monitor.update_signal_state({"mc_p_up": 0.5, "metals_action": "SELL"})
        assert monitor.metals_disagree_count == 0

    def test_disagreement_exit_at_2(self, monitor):
        monitor.metals_disagree_count = 2
        monitor.current_price = 74.50
        exits = monitor.compute_exit_signals({"rsi": 50, "mc_p_up": 0.5, "buy_count": 3, "sell_count": 3, "news_action": "HOLD"})
        triggers = [e["trigger"] for e in exits]
        assert "METALS_DISAGREE" in triggers


class TestMCStability:
    """Lesson 39: MC stability tracking."""

    def test_mc_history_tracks(self, monitor):
        monitor.update_signal_state({"mc_p_up": 0.80, "metals_action": "HOLD"})
        monitor.update_signal_state({"mc_p_up": 0.75, "metals_action": "HOLD"})
        assert len(monitor.mc_history) == 2
        assert monitor.mc_stable_bullish()

    def test_mc_not_stable_with_one_check(self, monitor):
        monitor.update_signal_state({"mc_p_up": 0.80, "metals_action": "HOLD"})
        assert not monitor.mc_stable_bullish()  # need 2

    def test_mc_stable_bearish(self, monitor):
        monitor.update_signal_state({"mc_p_up": 0.20, "metals_action": "HOLD"})
        monitor.update_signal_state({"mc_p_up": 0.25, "metals_action": "HOLD"})
        assert monitor.mc_stable_bearish()

    def test_mc_not_stable_mixed(self, monitor):
        monitor.update_signal_state({"mc_p_up": 0.80, "metals_action": "HOLD"})
        monitor.update_signal_state({"mc_p_up": 0.20, "metals_action": "HOLD"})
        assert not monitor.mc_stable_bullish()
        assert not monitor.mc_stable_bearish()


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
            "w_confidence": 0.75,
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
