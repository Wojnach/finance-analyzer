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


def _stub_snapshot(monkeypatch, snapshot):
    """Patch get_stop_losses_strict (used by _capture_stop_snapshot) to
    return the given fixture data without hitting Avanza."""
    monkeypatch.setattr(
        "portfolio.avanza_session.get_stop_losses_strict",
        lambda: snapshot,
    )


def _ok(cancelled=None, snapshot=None):
    """Build a SUCCESS server-side result."""
    return {
        "status": "SUCCESS",
        "cancelled": cancelled or [],
        "remaining": [],
        "snapshot": snapshot or [],
        "elapsed_seconds": 0.1,
    }


def _failed(remaining=None, snapshot=None):
    return {
        "status": "FAILED",
        "cancelled": [],
        "remaining": remaining or ["S1"],
        "snapshot": snapshot or [],
        "elapsed_seconds": 3.0,
    }


def _partial(cancelled, remaining, snapshot=None):
    return {
        "status": "PARTIAL",
        "cancelled": cancelled,
        "remaining": remaining,
        "snapshot": snapshot or [],
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
        assert mod._ensure_stops_cancelled_before_sell(None, "") == (True, [])
        assert mod._ensure_stops_cancelled_before_sell(None, None) == (True, [])
        assert called["server"] is False

    def test_no_local_position_still_runs_server_cancel(self, monkeypatch, silence_telegram):
        """No matching POSITIONS key → skip local layer, still run server layer."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})
        # Snapshot returns one matching stop so server cancel actually runs
        _stub_snapshot(monkeypatch, [
            {"id": "S1", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
             "trigger": {"value": 0.95, "type": "LESS_OR_EQUAL"}, "order": {"price": 0.94, "volume": 100}},
        ])
        called = {"server": False}

        def fake_server(ob_id, account_id=None, max_wait=3.0):
            called["server"] = True
            assert ob_id == "OB1"
            return _ok(cancelled=["S1"])

        monkeypatch.setattr("portfolio.avanza_session.cancel_all_stop_losses_for", fake_server)
        ok, snap = mod._ensure_stops_cancelled_before_sell(None, "OB1")
        assert ok is True
        assert called["server"] is True
        # Rollback snapshot now contains only stops actually cancelled (S1)
        assert len(snap) == 1
        assert snap[0]["id"] == "S1"

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

        # Pre-cancel snapshot returns one matching stop
        _stub_snapshot(monkeypatch, [
            {"id": "SERVER_S1", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
             "trigger": {"value": 0.95, "type": "LESS_OR_EQUAL"}, "order": {"price": 0.94, "volume": 100}},
        ])

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
            lambda *a, **kw: _ok(cancelled=["SERVER_S1"], snapshot=[{"id": "SERVER_S1"}]),
        )

        ok, snap = mod._ensure_stops_cancelled_before_sell(object(), "OB1")
        assert ok is True
        # Rollback snapshot is filtered to confirmed-cancelled stops
        assert len(snap) == 1
        assert snap[0]["id"] == "SERVER_S1"
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
        # Pre-cancel snapshot has one stop so server cancel runs
        _stub_snapshot(monkeypatch, [
            {"id": "S1", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
             "trigger": {"value": 0.95}, "order": {"price": 0.94, "volume": 100}},
        ])
        monkeypatch.setattr(mod, "_load_stop_orders", lambda: {"silver_q0": {"orders": []}})

        local_called = {"n": 0}
        monkeypatch.setattr(mod, "_cancel_stop_orders", lambda *a, **kw: local_called.__setitem__("n", 1))
        monkeypatch.setattr(mod, "get_csrf", lambda page: "TOKEN")

        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: _ok(cancelled=["S1"]),
        )
        ok, _ = mod._ensure_stops_cancelled_before_sell(object(), "OB1")
        assert ok is True
        assert local_called["n"] == 0  # local layer never invoked

    def test_local_cascade_error_does_not_block_sell(self, monkeypatch, silence_telegram):
        """If the local cascade cancel raises, the server layer is still the
        authoritative gate. A SUCCESS server response still allows the sell."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {
            "silver_q0": {"ob_id": "OB1", "active": True}
        })
        _stub_snapshot(monkeypatch, [
            {"id": "S1", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
             "trigger": {"value": 0.95}, "order": {"price": 0.94, "volume": 100}},
        ])

        def boom(*a, **kw):
            raise RuntimeError("simulated cascade load failure")
        monkeypatch.setattr(mod, "_load_stop_orders", boom)

        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: _ok(cancelled=["S1"]),
        )
        ok, _ = mod._ensure_stops_cancelled_before_sell(object(), "OB1")
        assert ok is True

    def test_server_failed_blocks_sell_and_alerts(self, monkeypatch):
        """Server FAILED → return False AND send a Telegram alert."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})

        alerts = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: alerts.append(msg))
        # Pre-cancel snapshot has 2 stops; server cancels neither
        _stub_snapshot(monkeypatch, [
            {"id": "S1", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
             "trigger": {"value": 0.95}, "order": {"price": 0.94, "volume": 100}},
            {"id": "S2", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
             "trigger": {"value": 0.90}, "order": {"price": 0.89, "volume": 100}},
        ])
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: _failed(remaining=["S1", "S2"]),
        )
        ok, snap = mod._ensure_stops_cancelled_before_sell(object(), "OB1")
        assert ok is False
        # FAILED → no stops were cancelled → rollback snapshot is empty
        # (we don't re-arm stops that are still alive)
        assert snap == []
        assert any("STOP CANCEL FAILED" in m for m in alerts)
        assert any("OB1" in m for m in alerts)

    def test_server_partial_blocks_sell(self, monkeypatch, silence_telegram):
        """PARTIAL is also a hard block — some stops still encumber the volume."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})
        _stub_snapshot(monkeypatch, [
            {"id": "S1", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
             "trigger": {"value": 0.95}, "order": {"price": 0.94, "volume": 100}},
            {"id": "S2", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
             "trigger": {"value": 0.90}, "order": {"price": 0.89, "volume": 100}},
        ])
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: _partial(cancelled=["S1"], remaining=["S2"]),
        )
        ok, snap = mod._ensure_stops_cancelled_before_sell(object(), "OB1")
        assert ok is False
        # CRITICAL: rollback snapshot must contain ONLY the cancelled stop (S1),
        # NOT the still-alive S2. Re-arming S2 would create a duplicate and
        # inflate encumbered volume.
        assert len(snap) == 1
        assert snap[0]["id"] == "S1"

    def test_server_exception_blocks_sell(self, monkeypatch, silence_telegram):
        """If the server cancel call itself raises, treat as FAILED."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})

        def raises(*a, **kw):
            raise RuntimeError("connection refused")
        monkeypatch.setattr("portfolio.avanza_session.cancel_all_stop_losses_for", raises)
        ok, snap = mod._ensure_stops_cancelled_before_sell(object(), "OB1")
        assert ok is False
        assert snap == []

    def test_passes_account_id_to_server_call(self, monkeypatch, silence_telegram):
        """ACCOUNT_ID must be propagated so we don't accidentally cancel
        another account's stops."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})
        _stub_snapshot(monkeypatch, [
            {"id": "S1", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
             "trigger": {"value": 0.95}, "order": {"price": 0.94, "volume": 100}},
        ])
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

    def test_pre_cancel_snapshot_read_failure_blocks_sell(self, monkeypatch):
        """If we cannot read the inventory before cancelling, refuse to
        cancel — fail closed. Otherwise we'd cancel without a rollback path."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})
        alerts = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: alerts.append(msg))

        def boom():
            raise RuntimeError("session expired")
        monkeypatch.setattr("portfolio.avanza_session.get_stop_losses_strict", boom)

        # cancel_all_stop_losses_for must NOT be called — we never get past
        # the snapshot step
        called = {"cancel": False}
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: (called.__setitem__("cancel", True), _ok())[1],
        )

        ok, snap = mod._ensure_stops_cancelled_before_sell(object(), "OB1")
        assert ok is False
        assert snap == []
        assert called["cancel"] is False
        assert any("STOP SNAPSHOT FAILED" in m for m in alerts)

    def test_no_stops_short_circuits_to_success(self, monkeypatch, silence_telegram):
        """Empty inventory → no need to cancel anything → SUCCESS without
        invoking cancel_all_stop_losses_for."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})
        _stub_snapshot(monkeypatch, [])

        called = {"n": 0}
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: (called.__setitem__("n", called["n"] + 1), _ok())[1],
        )
        ok, snap = mod._ensure_stops_cancelled_before_sell(object(), "OB1")
        assert ok is True
        assert snap == []
        assert called["n"] == 0  # short-circuited

    def test_partial_returns_only_confirmed_cancelled(self, monkeypatch, silence_telegram):
        """Codex finding 2: PARTIAL must filter the rollback snapshot to
        only the stops the server confirmed it cancelled. Re-arming the
        full inventory would duplicate the still-active stops and
        re-trigger the same volume-constraint bug."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "POSITIONS", {})

        # Pre-cancel: 3 stops on the orderbook
        _stub_snapshot(monkeypatch, [
            {"id": "A", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
             "trigger": {"value": 1.0}, "order": {"price": 1.0, "volume": 100}},
            {"id": "B", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
             "trigger": {"value": 0.9}, "order": {"price": 0.9, "volume": 100}},
            {"id": "C", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
             "trigger": {"value": 0.8}, "order": {"price": 0.8, "volume": 100}},
        ])

        # Server confirms only A cancelled, B and C remain
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: _partial(cancelled=["A"], remaining=["B", "C"]),
        )
        ok, snap = mod._ensure_stops_cancelled_before_sell(object(), "OB1")
        assert ok is False
        assert len(snap) == 1
        assert snap[0]["id"] == "A"
        # B and C must NOT be in the rollback set — they're still alive
        assert all(s["id"] != "B" for s in snap)
        assert all(s["id"] != "C" for s in snap)


class TestRearmAfterFailedSell:
    """Tests for _rearm_stops_after_failed_sell — the rollback safety net."""

    def test_empty_snapshot_is_noop(self, monkeypatch):
        import metals_loop as mod
        called = {"n": 0}
        monkeypatch.setattr(
            "portfolio.avanza_session.rearm_stop_losses_from_snapshot",
            lambda *a, **kw: (called.__setitem__("n", 1), {"status": "SUCCESS"})[1],
        )
        mod._rearm_stops_after_failed_sell("OB1", [])
        assert called["n"] == 0  # never called for empty snapshot

    def test_success_logs_only(self, monkeypatch):
        import metals_loop as mod
        alerts = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: alerts.append(msg))
        monkeypatch.setattr(
            "portfolio.avanza_session.rearm_stop_losses_from_snapshot",
            lambda snap: {"status": "SUCCESS", "rearmed": ["NEW1"], "failed": []},
        )
        mod._rearm_stops_after_failed_sell("OB1", [{"id": "OLD"}])
        assert alerts == []  # no alert on clean re-arm

    def test_partial_alerts(self, monkeypatch):
        import metals_loop as mod
        alerts = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: alerts.append(msg))
        monkeypatch.setattr(
            "portfolio.avanza_session.rearm_stop_losses_from_snapshot",
            lambda snap: {"status": "PARTIAL", "rearmed": ["NEW1"], "failed": ["OLD2"]},
        )
        mod._rearm_stops_after_failed_sell("OB1", [{"id": "OLD1"}, {"id": "OLD2"}])
        assert any("RE-ARM PARTIAL" in m for m in alerts)
        assert any("OB1" in m for m in alerts)

    def test_total_failure_alerts(self, monkeypatch):
        import metals_loop as mod
        alerts = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: alerts.append(msg))
        monkeypatch.setattr(
            "portfolio.avanza_session.rearm_stop_losses_from_snapshot",
            lambda snap: {"status": "FAILED", "rearmed": [], "failed": ["OLD"]},
        )
        mod._rearm_stops_after_failed_sell("OB1", [{"id": "OLD"}])
        assert any("RE-ARM FAILED" in m for m in alerts)

    def test_exception_alerts_naked(self, monkeypatch):
        import metals_loop as mod
        alerts = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: alerts.append(msg))
        def raises(*a, **kw):
            raise RuntimeError("API down")
        monkeypatch.setattr(
            "portfolio.avanza_session.rearm_stop_losses_from_snapshot",
            raises,
        )
        # Must NOT raise — best effort with operator alert
        mod._rearm_stops_after_failed_sell("OB1", [{"id": "OLD"}])
        assert any("RE-ARM FAILED" in m for m in alerts)
        assert any("naked" in m.lower() for m in alerts)


class TestRestoreFullStopProtection:
    """Tests for _restore_full_stop_protection — used by cancel_spike_orders
    to roll back the partial protection that place_spike_orders left.
    """

    def test_empty_inputs_return_false(self, monkeypatch):
        import metals_loop as mod
        called = {"n": 0}
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: (called.__setitem__("n", called["n"] + 1), _ok())[1],
        )
        assert mod._restore_full_stop_protection("", [{"id": "X"}]) is False
        assert mod._restore_full_stop_protection("OB1", []) is False
        assert mod._restore_full_stop_protection(None, None) is False
        assert called["n"] == 0

    def test_clears_then_rearms_returns_true(self, monkeypatch, silence_telegram):
        import metals_loop as mod
        order = []
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: (order.append("clear"), {"status": "SUCCESS", "cancelled": [], "remaining": [], "snapshot": [], "elapsed_seconds": 0.0})[1],
        )
        monkeypatch.setattr(
            "portfolio.avanza_session.rearm_stop_losses_from_snapshot",
            lambda snap: (order.append("rearm"), {"status": "SUCCESS", "rearmed": ["NEW1"], "failed": []})[1],
        )
        assert mod._restore_full_stop_protection("OB1", [{"id": "OLD"}]) is True
        assert order == ["clear", "rearm"]

    def test_clear_failure_still_rearms_with_warning(self, monkeypatch):
        """Codex-flagged tradeoff: if we can't cleanly clear the resized
        stops, re-arm anyway and warn. Better over-coverage than naked."""
        import metals_loop as mod
        alerts = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: alerts.append(msg))

        rearm_called = {"n": 0}
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: {"status": "FAILED", "cancelled": [], "remaining": ["X"], "snapshot": [], "elapsed_seconds": 0.0},
        )
        monkeypatch.setattr(
            "portfolio.avanza_session.rearm_stop_losses_from_snapshot",
            lambda snap: (rearm_called.__setitem__("n", 1), {"status": "SUCCESS", "rearmed": ["NEW"], "failed": []})[1],
        )
        ok = mod._restore_full_stop_protection("OB1", [{"id": "OLD"}])
        assert ok is True
        assert rearm_called["n"] == 1
        assert any("RESTORE WARNING" in m for m in alerts)

    def test_rearm_partial_returns_true_with_alert(self, monkeypatch):
        """PARTIAL means SOME re-arms succeeded — return True so the caller
        drops the snapshot. The Telegram alert covers the gap."""
        import metals_loop as mod
        alerts = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: alerts.append(msg))
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: {"status": "SUCCESS", "cancelled": [], "remaining": [], "snapshot": [], "elapsed_seconds": 0.0},
        )
        monkeypatch.setattr(
            "portfolio.avanza_session.rearm_stop_losses_from_snapshot",
            lambda snap: {"status": "PARTIAL", "rearmed": ["A"], "failed": ["B"]},
        )
        ok = mod._restore_full_stop_protection("OB1", [{"id": "A"}, {"id": "B"}])
        assert ok is True
        assert any("SPIKE RESTORE PARTIAL" in m for m in alerts)

    def test_rearm_total_failure_returns_false(self, monkeypatch):
        """FAILED with no rearmed stops → return False so the caller keeps
        the snapshot for retry."""
        import metals_loop as mod
        alerts = []
        monkeypatch.setattr(mod, "send_telegram", lambda msg: alerts.append(msg))
        monkeypatch.setattr(
            "portfolio.avanza_session.cancel_all_stop_losses_for",
            lambda *a, **kw: {"status": "SUCCESS", "cancelled": [], "remaining": [], "snapshot": [], "elapsed_seconds": 0.0},
        )
        monkeypatch.setattr(
            "portfolio.avanza_session.rearm_stop_losses_from_snapshot",
            lambda snap: {"status": "FAILED", "rearmed": [], "failed": ["A"]},
        )
        ok = mod._restore_full_stop_protection("OB1", [{"id": "A"}])
        assert ok is False
        assert any("SPIKE RESTORE FAILED" in m for m in alerts)


class TestFetchLivePositionVolume:
    def test_none_ob_id_returns_none(self, monkeypatch):
        import metals_loop as mod
        assert mod._fetch_live_position_volume(None) is None
        assert mod._fetch_live_position_volume("") is None

    def test_returns_volume_for_held_position(self, monkeypatch):
        import metals_loop as mod
        monkeypatch.setattr(
            "portfolio.avanza_session.get_positions",
            lambda: [
                {"orderbook_id": "OTHER", "volume": 999},
                {"orderbook_id": "OB1", "volume": 60},
            ],
        )
        assert mod._fetch_live_position_volume("OB1") == 60

    def test_returns_zero_when_not_held(self, monkeypatch):
        """Position not in holdings → 0 (sold/never held)."""
        import metals_loop as mod
        monkeypatch.setattr(
            "portfolio.avanza_session.get_positions",
            lambda: [{"orderbook_id": "OTHER", "volume": 100}],
        )
        assert mod._fetch_live_position_volume("OB1") == 0

    def test_returns_none_on_api_error(self, monkeypatch):
        """Critical: distinguish read-failure (None) from no-holding (0)."""
        import metals_loop as mod
        def boom():
            raise RuntimeError("session expired")
        monkeypatch.setattr("portfolio.avanza_session.get_positions", boom)
        assert mod._fetch_live_position_volume("OB1") is None


class TestCancelSpikeOrdersRestore:
    """Tests for cancel_spike_orders' restore-and-retain behavior."""

    def _make_state(self, orders=None, snapshots=None):
        return {
            "orders": orders or {"silver_q0": "ORDER1"},
            "stop_snapshots": snapshots if snapshots is not None else {
                "silver_q0": [
                    {"id": "OLD1", "orderbook": {"id": "OB1"}, "account": {"id": "1625505"},
                     "trigger": {"value": 1.0}, "order": {"price": 1.0, "volume": 100}}
                ],
            },
            "targets": {},
            "date": "2026-04-08",
            "placed": True,
            "cancelled": False,
        }

    def _fake_page(self, delete_status=200):
        class FakePage:
            def evaluate(self, _js, _args):
                return {"status": delete_status, "body": ""}
        return FakePage()

    def test_failed_delete_keeps_snapshot(self, monkeypatch, silence_telegram):
        """Codex finding 3 part 1: if the spike DELETE returns non-2xx,
        the order is still open. Restoring stops would over-encumber."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "get_csrf", lambda page: "TOKEN")
        spike_state = self._make_state()
        positions = {"silver_q0": {"ob_id": "OB1", "units": 100}}

        restore_called = {"n": 0}
        monkeypatch.setattr(mod, "_restore_full_stop_protection",
                            lambda *a, **kw: (restore_called.__setitem__("n", 1), True)[1])
        monkeypatch.setattr(mod, "_fetch_live_position_volume", lambda ob_id: 100)

        mod.cancel_spike_orders(self._fake_page(delete_status=500), spike_state, positions)
        # Restore must NOT have been called — order is still open
        assert restore_called["n"] == 0
        # Snapshot must be retained for retry
        assert "silver_q0" in spike_state["stop_snapshots"]

    def test_zero_live_volume_drops_snapshot_no_restore(self, monkeypatch, silence_telegram):
        """If position is fully gone (0 live volume), no restore needed.
        Drop the snapshot since nothing to retry."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "get_csrf", lambda page: "TOKEN")
        spike_state = self._make_state()
        positions = {"silver_q0": {"ob_id": "OB1", "units": 100}}

        restore_called = {"n": 0}
        monkeypatch.setattr(mod, "_restore_full_stop_protection",
                            lambda *a, **kw: (restore_called.__setitem__("n", 1), True)[1])
        monkeypatch.setattr(mod, "_fetch_live_position_volume", lambda ob_id: 0)

        mod.cancel_spike_orders(self._fake_page(delete_status=200), spike_state, positions)
        assert restore_called["n"] == 0
        assert "silver_q0" not in spike_state["stop_snapshots"]

    def test_partial_fill_resizes_snapshot_to_live_volume(self, monkeypatch, silence_telegram):
        """Codex finding 3 part 2: live position is smaller than original
        (partial spike fill). Resize snapshot before restoring."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "get_csrf", lambda page: "TOKEN")
        spike_state = self._make_state()
        positions = {"silver_q0": {"ob_id": "OB1", "units": 100}}

        captured = {}
        def fake_restore(ob_id, snap):
            captured["ob_id"] = ob_id
            captured["volumes"] = [s["order"]["volume"] for s in snap]
            return True
        monkeypatch.setattr(mod, "_restore_full_stop_protection", fake_restore)
        monkeypatch.setattr(mod, "_fetch_live_position_volume", lambda ob_id: 60)

        mod.cancel_spike_orders(self._fake_page(delete_status=200), spike_state, positions)
        # Snapshot was capped at the live 60u volume, not the original 100u
        assert captured["volumes"] == [60]
        # Successful restore → snapshot dropped
        assert "silver_q0" not in spike_state["stop_snapshots"]

    def test_restore_failure_keeps_snapshot(self, monkeypatch, silence_telegram):
        """Codex finding 3 part 3: failed restore must NOT drop the
        snapshot — operator needs the rollback record."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "get_csrf", lambda page: "TOKEN")
        spike_state = self._make_state()
        positions = {"silver_q0": {"ob_id": "OB1", "units": 100}}

        monkeypatch.setattr(mod, "_restore_full_stop_protection", lambda *a, **kw: False)
        monkeypatch.setattr(mod, "_fetch_live_position_volume", lambda ob_id: 100)

        mod.cancel_spike_orders(self._fake_page(delete_status=200), spike_state, positions)
        # Snapshot is RETAINED for next-iteration retry
        assert "silver_q0" in spike_state["stop_snapshots"]

    def test_volume_read_failure_keeps_snapshot(self, monkeypatch, silence_telegram):
        """If we cannot read live volume, we don't know how to size the
        restore. Keep the snapshot for retry."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "get_csrf", lambda page: "TOKEN")
        spike_state = self._make_state()
        positions = {"silver_q0": {"ob_id": "OB1", "units": 100}}

        restore_called = {"n": 0}
        monkeypatch.setattr(mod, "_restore_full_stop_protection",
                            lambda *a, **kw: (restore_called.__setitem__("n", 1), True)[1])
        monkeypatch.setattr(mod, "_fetch_live_position_volume", lambda ob_id: None)

        mod.cancel_spike_orders(self._fake_page(delete_status=200), spike_state, positions)
        assert restore_called["n"] == 0
        assert "silver_q0" in spike_state["stop_snapshots"]

    def test_no_positions_arg_keeps_snapshot(self, monkeypatch, silence_telegram):
        """Backward-compat: legacy callers without positions arg → keep
        snapshot, log it, never restore."""
        import metals_loop as mod
        monkeypatch.setattr(mod, "get_csrf", lambda page: "TOKEN")
        spike_state = self._make_state()

        restore_called = {"n": 0}
        monkeypatch.setattr(mod, "_restore_full_stop_protection",
                            lambda *a, **kw: (restore_called.__setitem__("n", 1), True)[1])

        mod.cancel_spike_orders(self._fake_page(delete_status=200), spike_state, None)
        assert restore_called["n"] == 0
        assert "silver_q0" in spike_state["stop_snapshots"]


class TestResizeSnapshotVolume:
    def test_caps_volume_at_new_max(self):
        import metals_loop as mod
        snapshot = [
            {"id": "S1", "order": {"volume": 1000, "price": 1.0}, "trigger": {"value": 0.9}, "orderbook": {"id": "OB1"}},
            {"id": "S2", "order": {"volume": 500, "price": 1.0}, "trigger": {"value": 0.8}, "orderbook": {"id": "OB1"}},
        ]
        resized = mod._resize_snapshot_volume(snapshot, 750)
        # S1 (1000) capped to 750, S2 (500) unchanged
        assert resized[0]["order"]["volume"] == 750
        assert resized[1]["order"]["volume"] == 500
        # Originals must NOT be mutated
        assert snapshot[0]["order"]["volume"] == 1000

    def test_preserves_non_dict_filtering(self):
        import metals_loop as mod
        snapshot = [
            "garbage",
            None,
            {"id": "GOOD", "order": {"volume": 100, "price": 1.0}, "trigger": {"value": 0.9}, "orderbook": {"id": "OB1"}},
        ]
        resized = mod._resize_snapshot_volume(snapshot, 50)
        assert len(resized) == 1
        assert resized[0]["order"]["volume"] == 50
