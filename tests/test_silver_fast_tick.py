"""Tests for silver fast-tick monitor (merged from silver_monitor.py into metals_loop.py).

Tests the 10-second price check, threshold alerts, velocity detection,
and session tracking that were ported from the standalone silver_monitor.py.
"""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))


@pytest.fixture(autouse=True)
def _isolate_files(tmp_path, monkeypatch):
    """Redirect file paths to tmp_path and reset silver state."""
    import metals_loop as mod

    monkeypatch.setattr(mod, "POSITIONS_STATE_FILE", str(tmp_path / "positions.json"))
    if hasattr(mod, "STOP_ORDER_FILE"):
        monkeypatch.setattr(mod, "STOP_ORDER_FILE", str(tmp_path / "stop_orders.json"))
    if hasattr(mod, "TRADE_QUEUE_FILE"):
        monkeypatch.setattr(mod, "TRADE_QUEUE_FILE", str(tmp_path / "trade_queue.json"))

    # Reset silver fast-tick state between tests
    mod._silver_fast_prices.clear()
    mod._silver_alerted_levels.clear()
    mod._silver_session_low = None
    mod._silver_session_high = None
    mod._silver_consecutive_down = 0
    mod._silver_prev_price = None
    mod._silver_underlying_ref = None

    yield


@pytest.fixture
def active_silver_positions(monkeypatch):
    """Set up POSITIONS with an active silver position."""
    import metals_loop as mod

    positions = {
        "gold": {"name": "BULL GULD", "ob_id": "1", "api_type": "cert",
                 "active": False, "units": 0, "entry": 0, "stop": 0},
        "silver301": {"name": "MINI L SILVER AVA 301", "ob_id": "2334960",
                      "api_type": "warrant", "active": True, "units": 521,
                      "entry": 12.86, "stop": 12.22, "leverage": 4.76},
    }
    monkeypatch.setattr(mod, "POSITIONS", positions)
    return positions


@pytest.fixture
def no_silver_positions(monkeypatch):
    """Set up POSITIONS with no active silver."""
    import metals_loop as mod

    positions = {
        "gold": {"name": "BULL GULD", "ob_id": "1", "api_type": "cert",
                 "active": True, "units": 10, "entry": 500, "stop": 450},
    }
    monkeypatch.setattr(mod, "POSITIONS", positions)
    return positions


# ---------------------------------------------------------------------------
# _has_active_silver / _get_active_silver
# ---------------------------------------------------------------------------

class TestActivesilverDetection:
    def test_has_active_silver(self, active_silver_positions):
        import metals_loop as mod
        assert mod._has_active_silver() is True

    def test_no_active_silver(self, no_silver_positions):
        import metals_loop as mod
        assert mod._has_active_silver() is False

    def test_get_active_silver_returns_key_pos(self, active_silver_positions):
        import metals_loop as mod
        key, pos = mod._get_active_silver()
        assert key == "silver301"
        assert pos["active"] is True
        assert pos["units"] == 521

    def test_get_active_silver_returns_none(self, no_silver_positions):
        import metals_loop as mod
        key, pos = mod._get_active_silver()
        assert key is None
        assert pos is None

    def test_inactive_silver_not_detected(self, monkeypatch):
        import metals_loop as mod
        positions = {
            "silver301": {"name": "MINI", "ob_id": "1", "api_type": "warrant",
                          "active": False, "units": 0, "entry": 0, "stop": 0},
        }
        monkeypatch.setattr(mod, "POSITIONS", positions)
        assert mod._has_active_silver() is False


# ---------------------------------------------------------------------------
# _silver_init_ref
# ---------------------------------------------------------------------------

class TestSilverInitRef:
    def test_init_from_underlying_entry_field(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        # Add underlying_entry to the position
        active_silver_positions["silver301"]["underlying_entry"] = 88.50
        mod._silver_init_ref()
        assert mod._silver_underlying_ref == 88.50

    def test_init_from_current_price(self, active_silver_positions, monkeypatch, tmp_path):
        import metals_loop as mod
        # No underlying_entry in position — fallback to current price
        monkeypatch.setattr(mod, "_underlying_prices", {"XAG-USD": 84.00})
        monkeypatch.setattr(mod, "POSITIONS_STATE_FILE", str(tmp_path / "pos.json"))
        # Write initial state so persist works
        (tmp_path / "pos.json").write_text(json.dumps(active_silver_positions))
        mod._silver_init_ref()
        assert mod._silver_underlying_ref == 84.00

    def test_init_skips_if_already_set(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.00
        mod._silver_init_ref()
        assert mod._silver_underlying_ref == 90.00  # Unchanged

    def test_init_no_silver_position(self, no_silver_positions):
        import metals_loop as mod
        mod._silver_init_ref()
        assert mod._silver_underlying_ref is None


# ---------------------------------------------------------------------------
# _silver_reset_session
# ---------------------------------------------------------------------------

class TestSilverResetSession:
    def test_reset_clears_all_state(self, active_silver_positions):
        import metals_loop as mod
        # Set some state
        mod._silver_underlying_ref = 90.0
        mod._silver_session_low = 85.0
        mod._silver_session_high = 95.0
        mod._silver_consecutive_down = 5
        mod._silver_prev_price = 87.0
        mod._silver_fast_prices.append(88.0)
        mod._silver_alerted_levels.add(-3.0)

        mod._silver_reset_session()

        assert mod._silver_underlying_ref is None
        assert mod._silver_session_low is None
        assert mod._silver_session_high is None
        assert mod._silver_consecutive_down == 0
        assert mod._silver_prev_price is None
        assert len(mod._silver_fast_prices) == 0
        assert len(mod._silver_alerted_levels) == 0


# ---------------------------------------------------------------------------
# _silver_fast_tick
# ---------------------------------------------------------------------------

class TestSilverFastTick:
    def test_noop_without_silver_position(self, no_silver_positions, monkeypatch):
        import metals_loop as mod
        # Should not raise
        mod._silver_fast_tick()
        assert mod._silver_session_low is None

    def test_updates_session_low_high(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        monkeypatch.setattr(mod, "_underlying_prices", {"XAG-USD": 88.0})
        # Mock _silver_fetch_xag to return from _underlying_prices
        monkeypatch.setattr(mod, "_silver_fetch_xag", lambda: 88.0)

        mod._silver_fast_tick()
        assert mod._silver_session_low == 88.0
        assert mod._silver_session_high == 88.0

        monkeypatch.setattr(mod, "_silver_fetch_xag", lambda: 87.0)
        mod._silver_fast_tick()
        assert mod._silver_session_low == 87.0
        assert mod._silver_session_high == 88.0

        monkeypatch.setattr(mod, "_silver_fetch_xag", lambda: 89.0)
        mod._silver_fast_tick()
        assert mod._silver_session_low == 87.0
        assert mod._silver_session_high == 89.0

    def test_consecutive_down_tracking(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        prices = [88.0, 87.5, 87.0, 86.5, 87.0]

        for p in prices:
            monkeypatch.setattr(mod, "_silver_fetch_xag", lambda _p=p: _p)
            mod._silver_fast_tick()

        # 87.5 < 88.0 (down), 87.0 < 87.5 (down), 86.5 < 87.0 (down), 87.0 > 86.5 (reset)
        assert mod._silver_consecutive_down == 0

    def test_consecutive_down_counts_correctly(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        prices = [88.0, 87.5, 87.0, 86.5]

        for p in prices:
            monkeypatch.setattr(mod, "_silver_fetch_xag", lambda _p=p: _p)
            mod._silver_fast_tick()

        # 3 consecutive drops after the first tick
        assert mod._silver_consecutive_down == 3

    def test_threshold_alert_fires(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        telegrams = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: telegrams.append(msg))

        # -3.5% drop from 90.0 = 86.85
        monkeypatch.setattr(mod, "_silver_fetch_xag", lambda: 86.85)
        mod._silver_fast_tick()

        assert len(telegrams) == 1
        assert "WARNING" in telegrams[0]
        assert -3.0 in mod._silver_alerted_levels

    def test_threshold_alert_fires_once(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        telegrams = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: telegrams.append(msg))

        # Trigger -3% twice
        monkeypatch.setattr(mod, "_silver_fetch_xag", lambda: 86.85)
        mod._silver_fast_tick()
        mod._silver_fast_tick()

        assert len(telegrams) == 1  # Only one alert

    def test_multiple_thresholds(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        telegrams = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: telegrams.append(msg))

        # -6% drop from 90.0 = 84.6 — should trigger WARNING (-3%) and DANGER (-5%)
        monkeypatch.setattr(mod, "_silver_fetch_xag", lambda: 84.6)
        mod._silver_fast_tick()

        assert len(telegrams) == 2
        assert -3.0 in mod._silver_alerted_levels
        assert -5.0 in mod._silver_alerted_levels

    def test_no_alert_on_small_drop(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        telegrams = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: telegrams.append(msg))

        # -1% drop = 89.1 — no threshold triggered
        monkeypatch.setattr(mod, "_silver_fetch_xag", lambda: 89.1)
        mod._silver_fast_tick()

        assert len(telegrams) == 0

    def test_velocity_alert(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        telegrams = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: telegrams.append(msg))

        # Fill velocity window with gradually declining prices
        # 18 ticks, start at 88.0, end at 87.2 — that's ~0.9% drop
        start_price = 88.0
        end_price = start_price * (1 + mod.SILVER_VELOCITY_ALERT_PCT / 100) - 0.01
        step = (end_price - start_price) / 17

        for i in range(18):
            p = start_price + step * i
            monkeypatch.setattr(mod, "_silver_fetch_xag", lambda _p=p: _p)
            mod._silver_fast_tick()

        # Should have triggered a velocity alert
        velocity_alerts = [m for m in telegrams if "RAPID DROP" in m]
        assert len(velocity_alerts) == 1

    def test_velocity_no_alert_on_stable(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        telegrams = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: telegrams.append(msg))

        # Fill window with stable price
        for _ in range(18):
            monkeypatch.setattr(mod, "_silver_fetch_xag", lambda: 88.0)
            mod._silver_fast_tick()

        velocity_alerts = [m for m in telegrams if "RAPID DROP" in m]
        assert len(velocity_alerts) == 0

    def test_null_price_handled(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        monkeypatch.setattr(mod, "_silver_fetch_xag", lambda: None)
        # Should not raise
        mod._silver_fast_tick()
        assert mod._silver_session_low is None

    def test_velocity_window_deque_bounded(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0

        # Add way more than window size
        for i in range(50):
            monkeypatch.setattr(mod, "_silver_fetch_xag", lambda _i=i: 85.0 + _i * 0.01)
            mod._silver_fast_tick()

        assert len(mod._silver_fast_prices) == mod.SILVER_VELOCITY_WINDOW


# ---------------------------------------------------------------------------
# _sleep_for_cycle with silver ticks
# ---------------------------------------------------------------------------

class TestSleepWithSilverTicks:
    def test_sleep_without_silver_does_not_tick(self, no_silver_positions, monkeypatch):
        import metals_loop as mod
        tick_count = [0]
        original_tick = mod._silver_fast_tick
        monkeypatch.setattr(mod, "_silver_fast_tick", lambda: tick_count.__setitem__(0, tick_count[0] + 1))

        # Sleep for 0.05s with no silver — should not tick
        start = time.monotonic()
        mod._sleep_for_cycle(start - 0.05, 0.1, "test")
        assert tick_count[0] == 0

    def test_sleep_with_silver_ticks(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        tick_count = [0]

        def mock_tick():
            tick_count[0] += 1

        monkeypatch.setattr(mod, "_silver_fast_tick", mock_tick)
        monkeypatch.setattr(mod, "_has_active_silver", lambda: True)
        # Use a very short interval so test runs fast
        monkeypatch.setattr(mod, "SILVER_FAST_TICK_INTERVAL", 0.05)

        start = time.monotonic()
        mod._sleep_for_cycle(start, 0.3, "test")

        # Should have ticked multiple times during 0.3s at 0.05s intervals
        assert tick_count[0] >= 2


# ---------------------------------------------------------------------------
# _silver_persist_ref
# ---------------------------------------------------------------------------

class TestSilverPersistRef:
    def test_persists_to_state_file(self, active_silver_positions, tmp_path, monkeypatch):
        import metals_loop as mod
        state_file = str(tmp_path / "pos.json")
        monkeypatch.setattr(mod, "POSITIONS_STATE_FILE", state_file)

        # Write initial state
        with open(state_file, "w") as f:
            json.dump(active_silver_positions, f)

        mod._silver_persist_ref("silver301", 88.50)

        with open(state_file) as f:
            saved = json.load(f)
        assert saved["silver301"]["underlying_entry"] == 88.50

    def test_persist_no_crash_on_missing_file(self, monkeypatch, tmp_path):
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS_STATE_FILE", str(tmp_path / "nonexistent.json"))
        # Should not raise
        mod._silver_persist_ref("silver301", 88.50)


# ---------------------------------------------------------------------------
# Alert message formatting
# ---------------------------------------------------------------------------

class TestAlertFormatting:
    def test_warning_includes_position_info(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        telegrams = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: telegrams.append(msg))

        monkeypatch.setattr(mod, "_silver_fetch_xag", lambda: 86.5)
        mod._silver_fast_tick()

        assert len(telegrams) >= 1
        msg = telegrams[0]
        assert "WARNING" in msg
        assert "XAG" in msg
        assert "Warrant" in msg
        assert "SEK" in msg
        assert "silver301" in msg

    def test_emergency_message_content(self, active_silver_positions, monkeypatch):
        import metals_loop as mod
        mod._silver_underlying_ref = 90.0
        telegrams = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: telegrams.append(msg))

        # -13% drop triggers all 5 levels
        monkeypatch.setattr(mod, "_silver_fetch_xag", lambda: 78.3)
        mod._silver_fast_tick()

        assert len(telegrams) == 5
        levels_seen = set()
        for msg in telegrams:
            for _, level in mod.SILVER_ALERT_LEVELS:
                if level in msg:
                    levels_seen.add(level)
        assert levels_seen == {"WARNING", "DANGER", "HIGH RISK", "CRITICAL", "EMERGENCY"}
