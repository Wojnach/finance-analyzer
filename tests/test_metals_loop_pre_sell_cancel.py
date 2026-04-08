"""Tests for _ensure_stops_cancelled_before_sell in data/metals_loop.py.

Pins the pre-sell stop-cancel safety net that prevents the
short.sell.not.allowed bug. The helper has two layers:

  1. Local cascade cancel via the existing Playwright path
  2. Server-side cancel + verify poll via portfolio.avanza_session

Both run on every call. Layer 2 is the authoritative gate — its return
value decides whether the dependent SELL is allowed to proceed.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))


@pytest.fixture(autouse=True)
def _isolate_files(tmp_path, monkeypatch):
    """Redirect state files to tmp_path."""
    import metals_loop as mod
    monkeypatch.setattr(mod, "POSITIONS_STATE_FILE", str(tmp_path / "positions.json"))
    monkeypatch.setattr(mod, "STOP_ORDER_FILE", str(tmp_path / "stop_orders.json"))
    yield


@pytest.fixture
def silence_telegram(monkeypatch):
    """Block any send_telegram calls in tests."""
    import metals_loop as mod
    monkeypatch.setattr(mod, "send_telegram", lambda *a, **kw: None)


def _ok(cancelled=None):
    """Build a SUCCESS server-side result."""
    return {
        "status": "SUCCESS",
        "cancelled": cancelled or [],
        "remaining": [],
        "elapsed_seconds": 0.1,
    }


def _failed(remaining=None):
    return {
        "status": "FAILED",
        "cancelled": [],
        "remaining": remaining or ["S1"],
        "elapsed_seconds": 3.0,
    }


def _partial(cancelled, remaining):
    return {
        "status": "PARTIAL",
        "cancelled": cancelled,
        "remaining": remaining,
        "elapsed_seconds": 3.0,
    }


# ---------------------------------------------------------------------------


class TestEnsureStopsCancelledBeforeSell:
    def test_empty_ob_id_returns_true_immediately(self, monkeypatch, silence_telegram):
        import metals_loop as mod
        # Use a sentinel: server cancel must NOT be called
        called = {"server": False}
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: (called.__setitem__("server", True), _ok())[1],
        )
        assert mod._ensure_stops_cancelled_before_sell(None, "") is True
        assert mod._ensure_stops_cancelled_before_sell(None, None) is True
        assert called["server"] is False

    def test_no_local_position_still_runs_server_cancel(self, monkeypatch, silence_telegram):
        """No matching POSITIONS key → skip local layer, still run server layer."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})
        called = {"server": False}

        def fake_server(ob_id, account_id=None, max_wait=3.0):
            called["server"] = True
            assert ob_id == "OB1"
            return _ok(cancelled=["S1"])

        monkeypatch.setattr("portfolio.avanza_session.cancel_all_stop_losses_for", fake_server)
        assert mod._ensure_stops_cancelled_before_sell(None, "OB1") is True
        assert called["server"] is True

    def test_matching_position_runs_both_layers_and_keeps_local_state(
        self, monkeypatch, silence_telegram, tmp_path
    ):
        """Matching key + cascade present → local cancel runs but does NOT delete
        the state entry (so the post-sell housekeeping still finds it)."""
        import metals_loop as mod

        # Position state with a matching ob_id
        monkeypatch.setattr(mod, "POSITIONS", {
            "silver_q0": {"ob_id": "OB1", "active": True, "name": "TEST"}
        })

        # Cascade stop state with one entry
        cascade_before = {
            "silver_q0": {
                "date": "2026-04-08",
                "stop_base": 1.0,
                "orders": [
                    {"order_id": "LOCAL_S1", "trigger": 0.95, "sell": 0.94, "units": 100, "status": "placed"}
                ],
            }
        }
        monkeypatch.setattr(mod, "_load_stop_orders", lambda: cascade_before)

        local_called = {"n": 0}
        def fake_local_cancel(page, key, state, csrf):
            local_called["n"] += 1
            assert key == "silver_q0"
        monkeypatch.setattr(mod, "_cancel_stop_orders", fake_local_cancel)
        monkeypatch.setattr(mod, "get_csrf", lambda page: "TOKEN_X")

        save_called = {"n": 0}
        monkeypatch.setattr(mod, "_save_stop_orders", lambda s: save_called.__setitem__("n", save_called["n"] + 1))

        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: _ok(cancelled=["SERVER_S1"]),
        )

        assert mod._ensure_stops_cancelled_before_sell(object(), "OB1") is True
        assert local_called["n"] == 1
        # CRITICAL: must NOT save (which would delete) — _cleanup_stop_orders_for
        # handles state removal AFTER the sell fills
        assert save_called["n"] == 0

    def test_local_cascade_skipped_when_no_orders(self, monkeypatch, silence_telegram):
        """Matching key but the local state has empty orders → skip local cancel."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {
            "silver_q0": {"ob_id": "OB1", "active": True}
        })
        monkeypatch.setattr(mod, "_load_stop_orders", lambda: {"silver_q0": {"orders": []}})

        local_called = {"n": 0}
        monkeypatch.setattr(mod, "_cancel_stop_orders", lambda *a, **kw: local_called.__setitem__("n", 1))
        monkeypatch.setattr(mod, "get_csrf", lambda page: "TOKEN")

        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: _ok(),
        )
        assert mod._ensure_stops_cancelled_before_sell(object(), "OB1") is True
        assert local_called["n"] == 0  # local layer never invoked

    def test_local_cascade_error_does_not_block_sell(self, monkeypatch, silence_telegram):
        """If the local cascade cancel raises, the server layer is still the
        authoritative gate. A SUCCESS server response still allows the sell."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {
            "silver_q0": {"ob_id": "OB1", "active": True}
        })

        def boom(*a, **kw):
            raise RuntimeError("simulated cascade load failure")
        monkeypatch.setattr(mod, "_load_stop_orders", boom)

        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: _ok(),
        )
        assert mod._ensure_stops_cancelled_before_sell(object(), "OB1") is True

    def test_server_failed_blocks_sell_and_alerts(self, monkeypatch):
        """Server FAILED → return False AND send a Telegram alert."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})

        alerts = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: alerts.append(msg))
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: _failed(remaining=["S1", "S2"]),
        )
        assert mod._ensure_stops_cancelled_before_sell(object(), "OB1") is False
        assert any("STOP CANCEL FAILED" in m for m in alerts)
        assert any("OB1" in m for m in alerts)

    def test_server_partial_blocks_sell(self, monkeypatch, silence_telegram):
        """PARTIAL is also a hard block — some stops still encumber the volume."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: _partial(cancelled=["S1"], remaining=["S2"]),
        )
        assert mod._ensure_stops_cancelled_before_sell(object(), "OB1") is False

    def test_server_exception_blocks_sell(self, monkeypatch, silence_telegram):
        """If the server cancel call itself raises, treat as FAILED."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})

        def raises(*a, **kw):
            raise RuntimeError("connection refused")
        monkeypatch.setattr("portfolio.avanza_session.cancel_all_stop_losses_for", raises)
        assert mod._ensure_stops_cancelled_before_sell(object(), "OB1") is False

    def test_passes_account_id_to_server_call(self, monkeypatch, silence_telegram):
        """ACCOUNT_ID must be propagated so we don't accidentally cancel
        another account's stops."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})
        captured = {}

        def fake_server(ob_id, account_id=None, max_wait=3.0):
            captured["ob_id"] = ob_id
            captured["account_id"] = account_id
            captured["max_wait"] = max_wait
            return _ok()

        monkeypatch.setattr("portfolio.avanza_session.cancel_all_stop_losses_for", fake_server)
        mod._ensure_stops_cancelled_before_sell(object(), "OB1", max_wait=5.5)
        assert captured["ob_id"] == "OB1"
        assert captured["account_id"] == mod.ACCOUNT_ID
        assert captured["max_wait"] == 5.5
