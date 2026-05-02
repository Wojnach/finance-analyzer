"""Smoke tests for the new /api/btc /api/eth /api/mstr /api/crypto endpoints."""
from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def client():
    """Flask test client. Bypasses dashboard auth via token=None stub.

    Matches the pattern in tests/test_dashboard.py — when a token is
    configured (config.json present in CI / dev machines) the routes return
    401 unless we explicitly tell `_get_dashboard_token` to act as if no
    token is configured for this test.
    """
    from dashboard.app import app
    app.config["TESTING"] = True
    with patch("dashboard.auth._get_dashboard_token", return_value=None), app.test_client() as c:
        yield c


class TestCryptoEndpoint:
    def test_crypto_endpoint_returns_200(self, client):
        r = client.get("/api/crypto")
        assert r.status_code == 200
        d = r.get_json()
        for k in ("state", "context", "warrant_catalog",
                  "risk", "decisions", "trades"):
            assert k in d, f"missing {k}"

    def test_btc_endpoint_returns_btc_ticker(self, client):
        r = client.get("/api/btc")
        assert r.status_code == 200
        d = r.get_json()
        assert d["ticker"] == "BTC-USD"
        assert "instrument" in d
        assert "deep_context" in d
        assert "shared_context" in d
        assert isinstance(d["decisions"], list)

    def test_eth_endpoint_returns_eth_ticker(self, client):
        r = client.get("/api/eth")
        assert r.status_code == 200
        d = r.get_json()
        assert d["ticker"] == "ETH-USD"
        assert "instrument" in d


class TestMstrEndpoint:
    def test_mstr_endpoint_returns_200(self, client):
        r = client.get("/api/mstr")
        assert r.status_code == 200
        d = r.get_json()
        assert d["ticker"] == "MSTR"
        assert "deep_context" in d
        assert "loop_state" in d
        assert "scorecard" in d


class TestCryptoSliceFunctions:
    """Unit-level tests on the helper functions."""

    def test_per_instrument_filters_by_ticker(self):
        from dashboard.app import _crypto_per_instrument
        state = {
            "positions": {
                "p1": {"ticker": "BTC-USD"},
                "p2": {"ticker": "ETH-USD"},
                "p3": {"ticker": "BTC-USD"},
            },
            "last_buy_ts": {"BTC-USD": "ts-btc", "ETH-USD": "ts-eth"},
        }
        btc_slice = _crypto_per_instrument(state, "BTC-USD")
        assert btc_slice["n_positions"] == 2
        assert btc_slice["last_buy_ts"] == "ts-btc"
        eth_slice = _crypto_per_instrument(state, "ETH-USD")
        assert eth_slice["n_positions"] == 1

    def test_per_instrument_handles_empty_state(self):
        from dashboard.app import _crypto_per_instrument
        result = _crypto_per_instrument({}, "BTC-USD")
        assert result["n_positions"] == 0
        assert result["last_buy_ts"] is None

    def test_decisions_filter_matches_ticker(self):
        from dashboard.app import _crypto_decisions_for
        decisions = [
            {"pos": {"ticker": "BTC-USD"}, "action": "BUY"},
            {"pos": {"ticker": "ETH-USD"}, "action": "BUY"},
            {"ticker": "BTC-USD", "action": "SELL"},
            {"action": "noise"},
        ]
        btc = _crypto_decisions_for(decisions, "BTC-USD")
        assert len(btc) == 2  # 1 with pos.ticker, 1 with ticker top-level
        eth = _crypto_decisions_for(decisions, "ETH-USD")
        assert len(eth) == 1
