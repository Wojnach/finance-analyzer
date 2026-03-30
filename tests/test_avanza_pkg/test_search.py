"""Tests for portfolio.avanza.search — instrument discovery."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.search import find_certificates, find_warrants, search
from portfolio.avanza.types import SearchHit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset singletons before and after every test."""
    AvanzaClient.reset()
    AvanzaAuth.reset()
    yield
    AvanzaClient.reset()
    AvanzaAuth.reset()


def _make_mock_client():
    client = MagicMock()
    client._push_subscription_id = "push-123"
    client._security_token = "csrf-abc"
    client._authentication_session = "auth-xyz"
    client._customer_id = "cust-42"
    client._session = MagicMock()
    return client


def _make_config():
    return {
        "avanza": {
            "username": "testuser",
            "password": "testpass",
            "totp_secret": "TESTSECRET",
        }
    }


@pytest.fixture()
def mock_avanza():
    """Set up AvanzaClient singleton with a mocked underlying avanza lib."""
    with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
        mock_client = _make_mock_client()
        mock_create.return_value = mock_client
        AvanzaClient.get_instance(_make_config())
        yield mock_client


# ---------------------------------------------------------------------------
# Shared raw data
# ---------------------------------------------------------------------------

RAW_HITS = [
    {
        "id": "2213050",
        "name": "MINI S SILVER AVA 26",
        "instrumentType": "CERTIFICATE",
        "tradable": True,
        "lastPrice": {"value": 5.80},
        "changePercent": {"value": -0.5},
    },
    {
        "id": "1234567",
        "name": "BULL SILVER X5 AVA 3",
        "instrumentType": "CERTIFICATE",
        "tradable": True,
        "lastPrice": {"value": 12.50},
        "changePercent": {"value": 2.3},
    },
]


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_returns_hits(self, mock_avanza):
        mock_avanza.search_for_instrument.return_value = RAW_HITS
        results = search("silver", limit=10)
        assert len(results) == 2
        assert all(isinstance(h, SearchHit) for h in results)
        assert results[0].orderbook_id == "2213050"
        assert results[0].name == "MINI S SILVER AVA 26"
        assert results[1].last_price == 12.50

    def test_with_instrument_type(self, mock_avanza):
        mock_avanza.search_for_instrument.return_value = []
        search("gold", instrument_type="certificate")
        call_args = mock_avanza.search_for_instrument.call_args[0]
        # Compare by .name to avoid xdist mock contamination of avanza.constants
        assert call_args[0].name == "CERTIFICATE"
        assert call_args[1] == "gold"

    def test_default_any_type(self, mock_avanza):
        mock_avanza.search_for_instrument.return_value = []
        search("SAAB")
        call_args = mock_avanza.search_for_instrument.call_args[0]
        # Compare by .name to avoid xdist mock contamination of avanza.constants
        assert call_args[0].name == "ANY"

    def test_custom_limit(self, mock_avanza):
        mock_avanza.search_for_instrument.return_value = []
        search("test", limit=50)
        call_args = mock_avanza.search_for_instrument.call_args[0]
        assert call_args[2] == 50

    def test_empty_results(self, mock_avanza):
        mock_avanza.search_for_instrument.return_value = []
        results = search("nonexistent")
        assert results == []

    def test_dict_response(self, mock_avanza):
        mock_avanza.search_for_instrument.return_value = {"hits": RAW_HITS}
        results = search("silver")
        assert len(results) == 2


# ---------------------------------------------------------------------------
# find_warrants
# ---------------------------------------------------------------------------

class TestFindWarrants:
    def test_delegates_with_warrant_type(self, mock_avanza):
        mock_avanza.search_for_instrument.return_value = [
            {"id": "999", "name": "WARRANT GOLD", "instrumentType": "WARRANT", "tradable": True},
        ]
        results = find_warrants("gold")
        assert len(results) == 1
        call_args = mock_avanza.search_for_instrument.call_args[0]
        # Compare by .name to avoid xdist mock contamination of avanza.constants
        assert call_args[0].name == "WARRANT"

    def test_default_limit(self, mock_avanza):
        mock_avanza.search_for_instrument.return_value = []
        find_warrants()
        call_args = mock_avanza.search_for_instrument.call_args[0]
        assert call_args[2] == 20  # default limit


# ---------------------------------------------------------------------------
# find_certificates
# ---------------------------------------------------------------------------

class TestFindCertificates:
    def test_delegates_with_certificate_type(self, mock_avanza):
        mock_avanza.search_for_instrument.return_value = RAW_HITS
        results = find_certificates("silver")
        assert len(results) == 2
        call_args = mock_avanza.search_for_instrument.call_args[0]
        # Compare by .name to avoid xdist mock contamination of avanza.constants
        assert call_args[0].name == "CERTIFICATE"

    def test_default_limit(self, mock_avanza):
        mock_avanza.search_for_instrument.return_value = []
        find_certificates()
        call_args = mock_avanza.search_for_instrument.call_args[0]
        assert call_args[2] == 20
