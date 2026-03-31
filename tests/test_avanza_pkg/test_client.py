"""Tests for portfolio.avanza.client — singleton HTTP wrapper."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import DEFAULT_ACCOUNT_ID, AvanzaClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset both singletons before and after every test."""
    AvanzaClient.reset()
    AvanzaAuth.reset()
    yield
    AvanzaClient.reset()
    AvanzaAuth.reset()


def _make_mock_client():
    """Create a mock mimicking an authenticated avanza.Avanza instance."""
    client = MagicMock()
    client._push_subscription_id = "push-123"
    client._security_token = "csrf-abc"
    client._authentication_session = "auth-xyz"
    client._customer_id = "cust-42"
    client._session = MagicMock()  # requests.Session mock
    return client


def _make_config(account_id=None):
    """Create a minimal config dict for AvanzaClient."""
    cfg = {
        "avanza": {
            "username": "testuser",
            "password": "testpass",
            "totp_secret": "TESTSECRET",
        }
    }
    if account_id is not None:
        cfg["avanza"]["account_id"] = account_id
    return cfg


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    @patch("portfolio.avanza.auth._create_avanza_client")
    def test_creates_instance(self, mock_create):
        mock_create.return_value = _make_mock_client()

        client = AvanzaClient.get_instance(_make_config())

        assert client is not None
        assert client.account_id == DEFAULT_ACCOUNT_ID

    @patch("portfolio.avanza.auth._create_avanza_client")
    def test_returns_same_instance(self, mock_create):
        mock_create.return_value = _make_mock_client()

        c1 = AvanzaClient.get_instance(_make_config())
        c2 = AvanzaClient.get_instance()  # No config needed on second call

        assert c1 is c2
        assert mock_create.call_count == 1

    @patch("portfolio.avanza.auth._create_avanza_client")
    def test_reset_clears_singleton(self, mock_create):
        mock_create.return_value = _make_mock_client()

        c1 = AvanzaClient.get_instance(_make_config())
        AvanzaClient.reset()
        AvanzaAuth.reset()  # Also reset auth since client depends on it

        mock_create.return_value = _make_mock_client()
        c2 = AvanzaClient.get_instance(_make_config())

        assert c1 is not c2

    def test_first_call_without_config_raises(self):
        with pytest.raises(ValueError, match="requires config"):
            AvanzaClient.get_instance()

    @patch("portfolio.avanza.auth._create_avanza_client")
    def test_thread_safety(self, mock_create):
        mock_create.return_value = _make_mock_client()
        results: list[AvanzaClient] = []
        barrier = threading.Barrier(5)

        def get_client():
            barrier.wait()
            c = AvanzaClient.get_instance(_make_config())
            results.append(c)

        threads = [threading.Thread(target=get_client) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == 5
        assert all(r is results[0] for r in results)


# ---------------------------------------------------------------------------
# Account ID from config
# ---------------------------------------------------------------------------

class TestAccountId:
    @patch("portfolio.avanza.auth._create_avanza_client")
    def test_default_account_id(self, mock_create):
        mock_create.return_value = _make_mock_client()

        client = AvanzaClient.get_instance(_make_config())
        assert client.account_id == "1625505"

    @patch("portfolio.avanza.auth._create_avanza_client")
    def test_custom_account_id(self, mock_create):
        mock_create.return_value = _make_mock_client()

        client = AvanzaClient.get_instance(_make_config(account_id="9999999"))
        assert client.account_id == "9999999"


# ---------------------------------------------------------------------------
# Delegation to avanza lib
# ---------------------------------------------------------------------------

class TestDelegation:
    @pytest.fixture()
    def client(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_client = _make_mock_client()
            mock_create.return_value = mock_client
            c = AvanzaClient.get_instance(_make_config())
            yield c

    def test_avanza_property(self, client):
        # The avanza property should return the underlying library instance
        assert client.avanza is not None

    def test_session_property(self, client):
        # Session should be the requests.Session from the underlying client
        assert client.session is not None

    def test_push_subscription_id(self, client):
        assert client.push_subscription_id == "push-123"

    def test_csrf_token(self, client):
        assert client.csrf_token == "csrf-abc"

    def test_get_positions_raw(self, client):
        client.avanza.get_accounts_positions.return_value = {"positions": []}
        result = client.get_positions_raw()
        client.avanza.get_accounts_positions.assert_called_once()
        assert result == {"positions": []}

    def test_get_overview_raw(self, client):
        client.avanza.get_overview.return_value = {"accounts": []}
        result = client.get_overview_raw()
        client.avanza.get_overview.assert_called_once()
        assert result == {"accounts": []}

    def test_get_market_data_raw(self, client):
        client.avanza.get_market_data.return_value = {"quote": {}}
        result = client.get_market_data_raw("12345")
        client.avanza.get_market_data.assert_called_once_with("12345")
        assert result == {"quote": {}}

    def test_get_order_book_raw(self, client):
        client.avanza.get_order_book.return_value = {"id": "12345"}
        result = client.get_order_book_raw("12345")
        client.avanza.get_order_book.assert_called_once_with("12345")

    def test_get_deals_raw(self, client):
        client.avanza.get_deals.return_value = []
        result = client.get_deals_raw()
        client.avanza.get_deals.assert_called_once()

    def test_get_orders_raw(self, client):
        client.avanza.get_orders.return_value = []
        result = client.get_orders_raw()
        client.avanza.get_orders.assert_called_once()

    def test_get_all_stop_losses_raw(self, client):
        client.avanza.get_all_stop_losses.return_value = []
        result = client.get_all_stop_losses_raw()
        client.avanza.get_all_stop_losses.assert_called_once()

    def test_get_news_raw(self, client):
        client.avanza.get_news.return_value = {"articles": []}
        result = client.get_news_raw("12345")
        client.avanza.get_news.assert_called_once_with("12345")
