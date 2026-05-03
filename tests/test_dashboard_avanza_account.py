"""Tests for /api/avanza_account — live broker-state mirror endpoint.

The route makes live calls into `portfolio.avanza.{account, trading}`. We
mock those imports so the dashboard test suite never touches the network
or expects valid Avanza credentials.
"""

from dataclasses import dataclass
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
# Stub dataclasses so dataclasses.asdict succeeds without touching the
# real avanza package types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cash:
    buying_power: float
    total_value: float
    own_capital: float


@dataclass(frozen=True)
class _Position:
    name: str
    orderbook_id: str
    instrument_type: str
    volume: float
    value: float
    acquired_value: float
    profit: float
    profit_percent: float
    last_price: float
    change_percent: float
    account_id: str
    currency: str


@dataclass(frozen=True)
class _Order:
    order_id: str
    orderbook_id: str
    side: str
    price: float
    volume: int
    status: str
    account_id: str


@dataclass(frozen=True)
class _StopLoss:
    stop_id: str
    orderbook_id: str
    trigger_price: float
    trigger_type: str
    sell_price: float
    volume: int
    status: str
    account_id: str


def _patch_avanza_success():
    """Patch all four import sites with successful stubs."""
    cash = _Cash(buying_power=12_345.6, total_value=98_765.4, own_capital=80_000.0)
    pos = _Position(
        name="MINI L SILVER",
        orderbook_id="123456",
        instrument_type="WARRANT",
        volume=10.0,
        value=1500.0,
        acquired_value=1400.0,
        profit=100.0,
        profit_percent=7.14,
        last_price=150.0,
        change_percent=1.25,
        account_id="1625505",
        currency="SEK",
    )
    order = _Order(
        order_id="ord-1", orderbook_id="123456", side="BUY",
        price=149.5, volume=5, status="ACTIVE", account_id="1625505",
    )
    stop = _StopLoss(
        stop_id="sl-1", orderbook_id="123456", trigger_price=140.0,
        trigger_type="LAST_PRICE", sell_price=139.0, volume=10,
        status="ACTIVE", account_id="1625505",
    )

    return [
        patch("portfolio.avanza.get_buying_power", return_value=cash),
        patch("portfolio.avanza.get_positions", return_value=[pos]),
        patch("portfolio.avanza.get_orders", return_value=[order]),
        patch("portfolio.avanza.get_stop_losses", return_value=[stop]),
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
        cash = _Cash(buying_power=1.0, total_value=2.0, own_capital=3.0)
        with patch("portfolio.avanza.get_buying_power", return_value=cash), \
             patch("portfolio.avanza.get_positions", side_effect=RuntimeError("avanza 503")), \
             patch("portfolio.avanza.get_orders", return_value=[]), \
             patch("portfolio.avanza.get_stop_losses", return_value=[]), \
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

    def test_all_failures_return_200_with_errors(self, client):
        """Even total auth failure should return 200 + errors[], not 500."""
        with patch("portfolio.avanza.get_buying_power", side_effect=RuntimeError("no session")), \
             patch("portfolio.avanza.get_positions", side_effect=RuntimeError("no session")), \
             patch("portfolio.avanza.get_orders", side_effect=RuntimeError("no session")), \
             patch("portfolio.avanza.get_stop_losses", side_effect=RuntimeError("no session")), \
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

        def _count_cash():
            call_count["n"] += 1
            return _Cash(buying_power=float(call_count["n"]), total_value=0.0, own_capital=0.0)

        with patch("portfolio.avanza.get_buying_power", side_effect=_count_cash), \
             patch("portfolio.avanza.get_positions", return_value=[]), \
             patch("portfolio.avanza.get_orders", return_value=[]), \
             patch("portfolio.avanza.get_stop_losses", return_value=[]), \
             _no_auth():
            r1 = client.get("/api/avanza_account").get_json()
            r2 = client.get("/api/avanza_account").get_json()
        # Exactly one upstream call across two requests.
        assert call_count["n"] == 1
        assert r1["cash"]["buying_power"] == 1.0
        assert r2["cash"]["buying_power"] == 1.0

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
