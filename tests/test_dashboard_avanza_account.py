"""Tests for /api/avanza_account — live broker-state mirror endpoint.

The route makes live calls into `portfolio.avanza_session` (the
Playwright BankID auth path used by the live metals_loop / golddigger).
We mock those imports so the dashboard test suite never touches the
network or expects a valid Avanza session.

avanza_session returns plain dicts, so the success-path stubs below are
dicts (not dataclasses).
"""

from unittest.mock import patch

import pytest

from dashboard.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    # Reset the per-process TTL cache so each test sees a fresh fetch.
    from dashboard.app import _AVANZA_CACHE
    _AVANZA_CACHE.update({"at": 0.0, "value": None})
    with app.test_client() as c:
        yield c


def _no_auth():
    return patch("dashboard.auth._get_dashboard_token", return_value=None)


# ---------------------------------------------------------------------------
# Stub dicts that match what `portfolio.avanza_session.*` actually returns.
# ---------------------------------------------------------------------------


def _stub_cash():
    return {"buying_power": 12_345.6, "total_value": 98_765.4, "own_capital": 80_000.0}


def _stub_position():
    return {
        "name": "MINI L SILVER",
        "orderbook_id": "123456",
        "instrument_id": "999",
        "type": "WARRANT",
        "volume": 10.0,
        "value": 1500.0,
        "acquired_value": 1400.0,
        "profit": 100.0,
        "profit_percent": 7.14,
        "currency": "SEK",
        "last_price": 150.0,
        "change_percent": 1.25,
        "account_id": "1625505",
        "account_type": "ISK",
    }


def _stub_raw_order():
    """Avanza orders endpoint returns camelCase dicts which the dashboard
    normalizes via _norm_order."""
    return {
        "orderId": "ord-1",
        "orderBookId": "123456",
        "orderType": "BUY",
        "price": 149.5,
        "volume": 5,
        "status": "ACTIVE",
        "accountId": "1625505",
    }


def _stub_raw_stop():
    return {
        "id": "sl-1",
        "orderbook": {"id": "123456"},
        "trigger": {"value": 140.0, "type": "LAST_PRICE"},
        "orderEvent": {"price": 139.0, "volume": 10},
        "status": "ACTIVE",
        "accountId": "1625505",
    }


def _patch_avanza_success():
    """Patch all four import sites with successful stubs."""
    return [
        patch("portfolio.avanza_session.get_buying_power", return_value=_stub_cash()),
        patch("portfolio.avanza_session.get_positions", return_value=[_stub_position()]),
        patch("portfolio.avanza_session.get_open_orders", return_value=[_stub_raw_order()]),
        patch("portfolio.avanza_session.get_stop_losses", return_value=[_stub_raw_stop()]),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAvanzaAccountEndpoint:
    def test_success_path(self, client):
        patches = _patch_avanza_success()
        for p in patches:
            p.start()
        try:
            with _no_auth():
                resp = client.get("/api/avanza_account")
        finally:
            for p in patches:
                p.stop()

        assert resp.status_code == 200
        data = resp.get_json()
        assert data["errors"] == []
        assert data["cash"]["buying_power"] == 12_345.6
        assert data["cash"]["total_value"] == 98_765.4
        assert len(data["positions"]) == 1
        assert data["positions"][0]["name"] == "MINI L SILVER"
        assert data["positions"][0]["account_id"] == "1625505"
        assert len(data["orders"]) == 1
        assert data["orders"][0]["side"] == "BUY"
        assert len(data["stop_losses"]) == 1
        assert data["stop_losses"][0]["trigger_type"] == "LAST_PRICE"
        assert "ts" in data and data["ts"]

    def test_partial_failure_isolates_per_section(self, client):
        """One failing subsection (positions) should not blank the others."""
        cash = {"buying_power": 1.0, "total_value": 2.0, "own_capital": 3.0}
        with patch("portfolio.avanza_session.get_buying_power", return_value=cash), \
             patch("portfolio.avanza_session.get_positions", side_effect=RuntimeError("avanza 503")), \
             patch("portfolio.avanza_session.get_open_orders", return_value=[]), \
             patch("portfolio.avanza_session.get_stop_losses", return_value=[]), \
             _no_auth():
            resp = client.get("/api/avanza_account")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["cash"]["buying_power"] == 1.0
        assert data["positions"] == []
        assert any("positions: RuntimeError" in e for e in data["errors"])
        # Other sections still empty arrays, not None
        assert data["orders"] == []
        assert data["stop_losses"] == []

    def test_cash_returning_none_records_error(self, client):
        """avanza_session.get_buying_power returns None on read failure
        (intentional sentinel — see the docstring there). The endpoint
        should surface that as an error rather than misreading 'no cash'."""
        with patch("portfolio.avanza_session.get_buying_power", return_value=None), \
             patch("portfolio.avanza_session.get_positions", return_value=[]), \
             patch("portfolio.avanza_session.get_open_orders", return_value=[]), \
             patch("portfolio.avanza_session.get_stop_losses", return_value=[]), \
             _no_auth():
            resp = client.get("/api/avanza_account")
        data = resp.get_json()
        assert data["cash"] is None
        assert any("cash:" in e for e in data["errors"])

    def test_all_failures_return_200_with_errors(self, client):
        """Even total auth failure should return 200 + errors[], not 500."""
        with patch("portfolio.avanza_session.get_buying_power", side_effect=RuntimeError("no session")), \
             patch("portfolio.avanza_session.get_positions", side_effect=RuntimeError("no session")), \
             patch("portfolio.avanza_session.get_open_orders", side_effect=RuntimeError("no session")), \
             patch("portfolio.avanza_session.get_stop_losses", side_effect=RuntimeError("no session")), \
             _no_auth():
            resp = client.get("/api/avanza_account")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["cash"] is None
        assert data["positions"] == []
        assert data["orders"] == []
        assert data["stop_losses"] == []
        assert len(data["errors"]) == 4

    def test_ttl_cache_returns_same_payload_within_window(self, client):
        """Two consecutive hits inside the 30s TTL must share the same payload."""
        call_count = {"n": 0}

        def _count_cash(*_args, **_kwargs):
            # Accepts account_id kwarg etc. — the endpoint passes it through
            # since the codex P2 fix on 2026-05-04.
            call_count["n"] += 1
            return {"buying_power": float(call_count["n"]), "total_value": 0.0, "own_capital": 0.0}

        with patch("portfolio.avanza_session.get_buying_power", side_effect=_count_cash), \
             patch("portfolio.avanza_session.get_positions", return_value=[]), \
             patch("portfolio.avanza_session.get_open_orders", return_value=[]), \
             patch("portfolio.avanza_session.get_stop_losses", return_value=[]), \
             _no_auth():
            r1 = client.get("/api/avanza_account").get_json()
            r2 = client.get("/api/avanza_account").get_json()
        # Exactly one upstream call across two requests.
        assert call_count["n"] == 1
        assert r1["cash"]["buying_power"] == 1.0
        assert r2["cash"]["buying_power"] == 1.0

    def test_force_query_bypasses_cache(self, client):
        """`?force=1` must re-fetch from Avanza even within the TTL window
        (codex P2 fix 2026-05-04). Without this the manual Refresh button
        in the view would silently serve stale broker state."""
        call_count = {"n": 0}

        def _count_cash(*_args, **_kwargs):
            call_count["n"] += 1
            return {"buying_power": float(call_count["n"]), "total_value": 0.0, "own_capital": 0.0}

        with patch("portfolio.avanza_session.get_buying_power", side_effect=_count_cash), \
             patch("portfolio.avanza_session.get_positions", return_value=[]), \
             patch("portfolio.avanza_session.get_open_orders", return_value=[]), \
             patch("portfolio.avanza_session.get_stop_losses", return_value=[]), \
             _no_auth():
            r1 = client.get("/api/avanza_account").get_json()
            r2 = client.get("/api/avanza_account?force=1").get_json()
        # Both requests fired upstream.
        assert call_count["n"] == 2
        assert r1["cash"]["buying_power"] == 1.0
        assert r2["cash"]["buying_power"] == 2.0

    def test_requires_auth_when_token_configured(self, client):
        with patch("dashboard.auth._get_dashboard_token", return_value="secret"):
            resp = client.get("/api/avanza_account")
        assert resp.status_code == 401

    def test_bearer_auth_works(self, client):
        patches = _patch_avanza_success()
        for p in patches:
            p.start()
        try:
            with patch("dashboard.auth._get_dashboard_token", return_value="secret"):
                resp = client.get(
                    "/api/avanza_account",
                    headers={"Authorization": "Bearer secret"},
                )
        finally:
            for p in patches:
                p.stop()
        assert resp.status_code == 200
        assert resp.get_json()["cash"]["buying_power"] == 12_345.6


class TestOpenOrdersFailurePropagation:
    """2026-06-12 (audit B4 fix 2): avanza_session.get_open_orders now
    RAISES on read failure instead of returning []. The dashboard must
    surface that as an error — never render 'no open orders' for an
    unreadable book."""

    def test_open_orders_read_failure_recorded_not_empty_success(self, client):
        from portfolio.avanza_session import AvanzaSessionError
        with patch("portfolio.avanza_session.get_buying_power", return_value=_stub_cash()), \
             patch("portfolio.avanza_session.get_positions", return_value=[]), \
             patch("portfolio.avanza_session.get_open_orders",
                   side_effect=AvanzaSessionError("Could not fetch open orders")), \
             patch("portfolio.avanza_session.get_stop_losses", return_value=[]), \
             _no_auth():
            resp = client.get("/api/avanza_account?force=1")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["orders"] == []
        assert any("orders: AvanzaSessionError" in e for e in data["errors"])
