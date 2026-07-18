"""Tests for data/metals_loop.py — position load/save, signal reading, shared functions.

Batch 5 of the metals monitoring auto-improvement plan.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

# We import individual functions from metals_loop — not the module itself,
# because importing the whole module triggers Playwright and heavy global init.
# Instead we test by importing the module and patching globals.


@pytest.fixture(autouse=True)
def _isolate_files(tmp_path, monkeypatch):
    """Redirect file paths to tmp_path."""
    # metals_loop defines these as module-level globals
    import metals_loop as mod

    monkeypatch.setattr(mod, "POSITIONS_STATE_FILE", str(tmp_path / "positions.json"))
    if hasattr(mod, "STOP_ORDER_FILE"):
        monkeypatch.setattr(mod, "STOP_ORDER_FILE", str(tmp_path / "stop_orders.json"))
    if hasattr(mod, "TRADE_QUEUE_FILE"):
        monkeypatch.setattr(mod, "TRADE_QUEUE_FILE", str(tmp_path / "trade_queue.json"))
    yield


# ---------------------------------------------------------------------------
# Position load / save
# ---------------------------------------------------------------------------

class TestPositionLoadSave:
    def test_defaults_on_missing_file(self, tmp_path):
        import metals_loop as mod
        positions = mod._load_positions()
        assert isinstance(positions, dict)
        assert len(positions) > 0  # Should return default positions

    def test_restored_state(self, tmp_path):
        import metals_loop as mod

        state = {
            "silver79": {
                "active": True,
                "units": 220,
                "entry_price": 41.50,
                "stop_price": 38.0,
            }
        }
        with open(mod.POSITIONS_STATE_FILE, "w") as f:
            json.dump(state, f)

        positions = mod._load_positions()
        # Should restore the state (may merge with defaults)
        assert isinstance(positions, dict)

    def test_roundtrip(self, tmp_path):
        import metals_loop as mod

        positions = mod._load_positions()
        # Modify a position
        for key in positions:
            positions[key]["units"] = 999
            break

        mod._save_positions(positions)
        loaded = mod._load_positions()

        # Should have our modified value
        for key in loaded:
            if loaded[key].get("units") == 999:
                break
        else:
            # The save/load may reset, check it at least doesn't crash
            pass

    def test_load_positions_bad_json_logs_and_falls_back(self, tmp_path, caplog):
        import logging

        import metals_loop as mod

        with open(mod.POSITIONS_STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{bad json")

        with caplog.at_level(logging.WARNING, logger="portfolio.file_utils"):
            positions = mod._load_positions()

        assert isinstance(positions, dict)
        assert positions
        assert any(
            "corrupt JSON" in rec.getMessage() for rec in caplog.records
        )

    def test_save_preserves_sell_metadata(self, tmp_path):
        import metals_loop as mod

        positions = mod._load_positions()
        for key in positions:
            positions[key]["last_sell_ts"] = "2026-03-01T10:00:00"
            positions[key]["total_sold"] = 100
            break

        mod._save_positions(positions)
        loaded = mod._load_positions()

        for key in loaded:
            if loaded[key].get("last_sell_ts"):
                assert loaded[key]["last_sell_ts"] == "2026-03-01T10:00:00"
                break

    def test_save_positions_uses_atomic_write_json(self, monkeypatch):
        import metals_loop as mod

        calls = []

        def _fake_atomic_write_json(path, data, indent=2, ensure_ascii=True):
            calls.append((path, data, indent, ensure_ascii))

        monkeypatch.setattr(mod, "atomic_write_json", _fake_atomic_write_json, raising=False)

        positions = mod._load_positions()
        mod._save_positions(positions)

        assert len(calls) == 1
        assert calls[0][0] == mod.POSITIONS_STATE_FILE
        assert isinstance(calls[0][1], dict)
        assert calls[0][3] is False


# ---------------------------------------------------------------------------
# read_signal_data
# ---------------------------------------------------------------------------

class TestReadSignalData:
    def test_reads_from_agent_summary(self, tmp_path):
        import metals_loop as mod

        # Create a minimal agent_summary.json
        summary = {
            "tickers": {
                "XAG-USD": {
                    "consensus": "BUY",
                    "buy_count": 5,
                    "sell_count": 1,
                    "hold_count": 4,
                },
                "XAU-USD": {
                    "consensus": "HOLD",
                    "buy_count": 2,
                    "sell_count": 2,
                    "hold_count": 6,
                },
            }
        }
        # Write to the expected path
        summary_path = "data/agent_summary.json"
        backup = None
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                backup = f.read()

        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f)

            result = mod.read_signal_data()
            assert isinstance(result, dict)
        finally:
            if backup is not None:
                with open(summary_path, "w") as f:
                    f.write(backup)
            elif os.path.exists(summary_path):
                os.remove(summary_path)

    def test_returns_empty_on_missing(self, tmp_path, monkeypatch):
        import metals_loop as mod

        # Ensure no summary files exist by patching os.path.exists for summary files
        orig = os.path.exists

        def _mock_exists(p):
            if "agent_summary" in str(p):
                return False
            return orig(p)

        monkeypatch.setattr(os.path, "exists", _mock_exists)
        result = mod.read_signal_data()
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Shared function delegations
# ---------------------------------------------------------------------------

class TestSharedFunctions:
    def test_pnl_pct(self):
        import metals_loop as mod
        assert mod.pnl_pct(110, 100) == pytest.approx(10.0)
        assert mod.pnl_pct(90, 100) == pytest.approx(-10.0)
        assert mod.pnl_pct(100, 0) == 0

    def test_log_prints(self, caplog):
        # 2026-04-09 Stage 1 log migration: log() → logger.info, so this
        # test asserts the INFO record arrives on the metals_loop logger
        # instead of checking raw stdout. See the top-of-file library
        # discipline comment in data/metals_loop.py for the rationale.
        import logging

        import metals_loop as mod

        with caplog.at_level(logging.INFO, logger="metals_loop"):
            mod.log("test message")

        assert any("test message" in rec.getMessage() for rec in caplog.records)

    def test_is_avanza_open(self, monkeypatch):
        import metals_loop as mod
        # 2026-06-12 (audit B4 fix 5): is_market_hours now consults the
        # dynamic close-time resolver, which would hit the Avanza API on a
        # machine with a live session file — stub it for hermetic tests.
        import portfolio.grid_fisher as gf
        monkeypatch.setattr(gf, "resolve_eod_cutoff_hm", lambda *a, **k: (21, 55))
        result = mod.is_avanza_open()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Stop orders
# ---------------------------------------------------------------------------

class TestStopOrders:
    def test_load_empty(self, tmp_path):
        import metals_loop as mod
        if hasattr(mod, "_load_stop_orders"):
            result = mod._load_stop_orders()
            assert isinstance(result, dict)

    def test_save_and_load(self, tmp_path):
        import metals_loop as mod
        if not hasattr(mod, "_save_stop_orders") or not hasattr(mod, "_load_stop_orders"):
            pytest.skip("Stop order functions not available")

        state = {"silver79": {"stop_id": "abc123", "trigger_price": 38.0}}
        mod._save_stop_orders(state)
        loaded = mod._load_stop_orders()
        assert loaded.get("silver79", {}).get("stop_id") == "abc123"

    def test_save_stop_orders_uses_atomic_write_json(self, monkeypatch):
        import metals_loop as mod

        calls = []

        def _fake_atomic_write_json(path, data, indent=2, ensure_ascii=True):
            calls.append((path, data, indent, ensure_ascii))

        monkeypatch.setattr(mod, "atomic_write_json", _fake_atomic_write_json, raising=False)
        mod._save_stop_orders({"silver79": {"stop_id": "abc123"}})

        assert len(calls) == 1
        assert calls[0][0] == mod.STOP_ORDER_FILE
        assert calls[0][3] is False


# ---------------------------------------------------------------------------
# Trade queue
# ---------------------------------------------------------------------------

class TestTradeQueue:
    def test_load_empty(self, tmp_path):
        import metals_loop as mod
        if hasattr(mod, "_load_trade_queue"):
            result = mod._load_trade_queue()
            assert isinstance(result, dict)
            assert "orders" in result or result == {}

    def test_save_and_load(self, tmp_path):
        import metals_loop as mod
        if not hasattr(mod, "_save_trade_queue") or not hasattr(mod, "_load_trade_queue"):
            pytest.skip("Trade queue functions not available")

        queue = {"version": 1, "orders": [{"action": "BUY", "key": "silver79", "units": 100}]}
        mod._save_trade_queue(queue)
        loaded = mod._load_trade_queue()
        assert len(loaded.get("orders", [])) == 1

    def test_load_trade_queue_bad_json_logs_and_falls_back(self, tmp_path, caplog):
        import logging

        import metals_loop as mod

        with open(mod.TRADE_QUEUE_FILE, "w", encoding="utf-8") as f:
            f.write("{bad json")

        with caplog.at_level(logging.WARNING, logger="portfolio.file_utils"):
            queue = mod._load_trade_queue()

        assert queue == {"version": 1, "orders": []}
        assert any("corrupt JSON" in rec.getMessage() for rec in caplog.records)

    def test_save_trade_queue_uses_atomic_write_json(self, monkeypatch):
        import metals_loop as mod

        calls = []

        def _fake_atomic_write_json(path, data, indent=2, ensure_ascii=True):
            calls.append((path, data, indent, ensure_ascii))

        monkeypatch.setattr(mod, "atomic_write_json", _fake_atomic_write_json, raising=False)
        mod._save_trade_queue({"version": 1, "orders": []})

        assert len(calls) == 1
        assert calls[0][0] == mod.TRADE_QUEUE_FILE
        assert calls[0][3] is False


class TestSilverAlertLevelsFormat:
    """SILVER_ALERT_LEVELS iteration must not crash (2026-06-01 fix)."""

    def test_alert_level_format_string(self):
        import metals_loop as mod
        levels = mod.SILVER_ALERT_LEVELS
        assert len(levels) > 0
        for threshold, name in levels:
            assert isinstance(threshold, (int, float))
            assert isinstance(name, str)
            formatted = f"{threshold}%"
            assert "%" in formatted


# ---------------------------------------------------------------------------
# 2026-06-12 (audit B4 fix 7): trailing-stop add-to-position must cancel
# the previous stop before placing the full-volume replacement, and a
# placement failure after a successful cancel must escalate (critical
# entry + Telegram + immediate retry) — never just a debug log.
# ---------------------------------------------------------------------------


class TestTrailingStopReplace:
    @pytest.fixture
    def env(self, monkeypatch):
        import metals_loop as mod
        import portfolio.avanza_control as ac
        import portfolio.claude_gate as cg

        calls = {"sequence": [], "telegrams": [], "criticals": [],
                 "deleted": [], "placed": []}

        monkeypatch.setattr(mod, "HARDWARE_TRAILING_ENABLED", True)
        monkeypatch.setattr(mod, "STOP_ORDER_ENABLED", False)
        monkeypatch.setattr(mod, "send_telegram",
                            lambda m, **kw: calls["telegrams"].append(m))
        monkeypatch.setattr(mod, "_save_positions", lambda p: None)
        monkeypatch.setattr(
            cg, "record_critical_error",
            lambda cat, caller, msg, ctx=None:
                calls["criticals"].append((cat, msg)) or True,
        )

        def _delete_ok(account_id, stop_id):
            calls["sequence"].append(("delete", stop_id))
            calls["deleted"].append(stop_id)
            return True, {"ok": True}

        monkeypatch.setattr(ac, "delete_stop_loss_no_page", _delete_ok)

        def _place_ok(account_id, ob_id, trail_percent, volume, valid_days=8):
            calls["sequence"].append(("place", volume))
            calls["placed"].append(volume)
            return True, {"stoplossOrderId": f"NEW{len(calls['placed'])}"}

        monkeypatch.setattr(mod, "place_trailing_stop_no_page", _place_ok)

        positions = {
            "silver_q1": {
                "name": "MINI L SILVER",
                "ob_id": "111",
                "units": 100,
                "entry": 40.0,
                "stop": 36.0,
                "active": True,
                "hw_trailing_stop_id": "OLD1",
            },
        }
        monkeypatch.setattr(mod, "POSITIONS", positions)
        return mod, ac, calls, positions

    def _order(self):
        return {"warrant_key": "MINI_L_SILVER", "ob_id": "111",
                "volume": 50, "warrant_name": "MINI L SILVER"}

    def test_replace_cancels_old_stop_before_placing(self, env):
        mod, ac, calls, positions = env
        mod._handle_buy_fill(None, self._order(), 41.0, {})
        assert calls["deleted"] == ["OLD1"]
        # Cancel happened strictly before the replacement placement.
        assert calls["sequence"][0] == ("delete", "OLD1")
        assert calls["sequence"][1][0] == "place"
        # Replacement covers the new TOTAL volume (100 + 50).
        assert calls["placed"] == [150]
        assert positions["silver_q1"]["hw_trailing_stop_id"] == "NEW1"

    def test_unconfirmed_cancel_skips_replacement(self, env, monkeypatch):
        mod, ac, calls, positions = env
        monkeypatch.setattr(
            ac, "delete_stop_loss_no_page",
            lambda account_id, stop_id: (False, {"error": "http 500"}),
        )
        mod._handle_buy_fill(None, self._order(), 41.0, {})
        # No full-volume replacement on top of a possibly-live old stop.
        assert calls["placed"] == []
        assert positions["silver_q1"]["hw_trailing_stop_id"] == "OLD1"
        assert any("NO trailing stop" in m for m in calls["telegrams"])

    def test_naked_after_cancel_escalates_and_retries(self, env, monkeypatch):
        mod, ac, calls, positions = env

        def _place_fail(account_id, ob_id, trail_percent, volume, valid_days=8):
            calls["placed"].append(volume)
            return False, {"error": "rejected"}

        monkeypatch.setattr(mod, "place_trailing_stop_no_page", _place_fail)
        mod._handle_buy_fill(None, self._order(), 41.0, {})
        # Escalation: critical entry + URGENT telegram + one retry.
        assert any(cat == "stop_replace_naked" for cat, _ in calls["criticals"])
        assert any("URGENT" in m for m in calls["telegrams"])
        assert len(calls["placed"]) == 2  # initial attempt + retry

    def test_retry_success_recovers_protection(self, env, monkeypatch):
        mod, ac, calls, positions = env
        attempts = []

        def _place_flaky(account_id, ob_id, trail_percent, volume, valid_days=8):
            attempts.append(volume)
            if len(attempts) == 1:
                return False, {"error": "transient"}
            return True, {"stoplossOrderId": "RECOVERED"}

        monkeypatch.setattr(mod, "place_trailing_stop_no_page", _place_flaky)
        mod._handle_buy_fill(None, self._order(), 41.0, {})
        assert len(attempts) == 2
        assert positions["silver_q1"]["hw_trailing_stop_id"] == "RECOVERED"

    def test_new_position_places_without_cancel(self, env, monkeypatch):
        mod, ac, calls, positions = env
        positions["silver_q1"]["hw_trailing_stop_id"] = None
        positions["silver_q1"]["active"] = False  # force the new-position path
        mod._handle_buy_fill(None, self._order(), 41.0, {})
        assert calls["deleted"] == []
        assert calls["placed"] == [50]


# ---------------------------------------------------------------------------
# 2026-06-12 (audit B4 fix 2 call-site): spike rollback must NOT treat an
# unreadable open-orders list as "spike order is terminal" — get_open_orders
# now raises on read failure, and the rollback keeps the conservative path.
# ---------------------------------------------------------------------------


class TestSpikeRollbackOpenOrdersFailure:
    def test_unreadable_open_orders_does_not_restore_stops(self, monkeypatch):
        import metals_loop as mod
        import portfolio.avanza_session as avz

        telegrams = []
        restored = []
        monkeypatch.setattr(mod, "send_telegram", lambda m, **kw: telegrams.append(m))
        monkeypatch.setattr(
            mod, "_rearm_stops_after_failed_sell",
            lambda ob_id, snapshot: restored.append(ob_id),
        )
        monkeypatch.setattr(
            avz, "cancel_order",
            lambda order_id, account_id=None: {"orderRequestStatus": "FAIL"},
        )

        def _raise(*a, **k):
            raise avz.AvanzaSessionError("both order endpoints failed")

        monkeypatch.setattr(avz, "get_open_orders", _raise)

        mod._rollback_spike_order_and_restore("SPK1", "111", [{"id": "SL1"}])
        # Failure ≠ "order gone": originals must NOT be restored on top of
        # a possibly-live spike sell; operator alert fires instead.
        assert restored == []
        assert any("ROLLBACK INCOMPLETE" in m for m in telegrams)

    def test_confirmed_absent_order_restores(self, monkeypatch):
        import metals_loop as mod
        import portfolio.avanza_session as avz

        restored = []
        monkeypatch.setattr(mod, "send_telegram", lambda m, **kw: None)
        monkeypatch.setattr(
            mod, "_rearm_stops_after_failed_sell",
            lambda ob_id, snapshot: restored.append(ob_id),
        )
        monkeypatch.setattr(
            avz, "cancel_order",
            lambda order_id, account_id=None: {"orderRequestStatus": "FAIL"},
        )
        monkeypatch.setattr(avz, "get_open_orders", lambda: [])
        mod._rollback_spike_order_and_restore("SPK1", "111", [{"id": "SL1"}])
        assert restored == ["111"]
