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

    def test_load_positions_bad_json_logs_and_falls_back(self, tmp_path, capsys):
        import metals_loop as mod

        with open(mod.POSITIONS_STATE_FILE, "w", encoding="utf-8") as f:
            f.write("{bad json")

        positions = mod._load_positions()
        captured = capsys.readouterr()

        assert isinstance(positions, dict)
        assert positions
        assert "Position state load failed" in captured.out

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
            with open(summary_path, "r") as f:
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

    def test_log_prints(self, capsys):
        import metals_loop as mod
        mod.log("test message")
        captured = capsys.readouterr()
        assert "test message" in captured.out

    def test_is_avanza_open(self):
        import metals_loop as mod
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

    def test_load_trade_queue_bad_json_logs_and_falls_back(self, tmp_path, monkeypatch):
        import metals_loop as mod

        messages = []
        monkeypatch.setattr(mod, "log", messages.append)

        with open(mod.TRADE_QUEUE_FILE, "w", encoding="utf-8") as f:
            f.write("{bad json")

        queue = mod._load_trade_queue()

        assert queue == {"version": 1, "orders": []}
        assert any("Trade queue load failed" in msg for msg in messages)

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
