"""Tests for portfolio.avanza.account — positions, buying power, transactions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.account import get_buying_power, get_positions, get_transactions
from portfolio.avanza.types import AccountCash, Position, Transaction


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

POSITION_A = {
    "instrument": {
        "type": "CERTIFICATE",
        "name": "MINI S SILVER AVA 26",
        "orderbook": {
            "id": "2213050",
            "name": "MINI S SILVER AVA 26",
            "type": "CERTIFICATE",
            "quote": {"latest": {"value": 5.80}, "changePercent": {"value": -1.5}},
        },
        "currency": "SEK",
    },
    "account": {"id": "1625505"},
    "volume": {"value": 500},
    "value": {"value": 2900.0},
    "acquiredValue": {"value": 3000.0},
    "lastTradingDayPerformance": {"absolute": {"value": -100.0}, "relative": {"value": -3.3}},
}

POSITION_B = {
    "instrument": {
        "type": "STOCK",
        "name": "SAAB-B",
        "orderbook": {
            "id": "5555",
            "name": "SAAB-B",
            "type": "STOCK",
            "quote": {"latest": {"value": 300.0}, "changePercent": {"value": 2.0}},
        },
        "currency": "SEK",
    },
    "account": {"id": "9999999"},
    "volume": {"value": 100},
    "value": {"value": 30000.0},
    "acquiredValue": {"value": 25000.0},
    "lastTradingDayPerformance": {"absolute": {"value": 5000.0}, "relative": {"value": 20.0}},
}


# ---------------------------------------------------------------------------
# get_positions
# ---------------------------------------------------------------------------

class TestGetPositions:
    def test_returns_all_positions(self, mock_avanza):
        mock_avanza.get_accounts_positions.return_value = {
            "withOrderbook": [POSITION_A, POSITION_B],
        }
        positions = get_positions()
        assert len(positions) == 2
        assert all(isinstance(p, Position) for p in positions)
        assert positions[0].name == "MINI S SILVER AVA 26"
        assert positions[1].name == "SAAB-B"

    def test_filter_by_account_id(self, mock_avanza):
        mock_avanza.get_accounts_positions.return_value = {
            "withOrderbook": [POSITION_A, POSITION_B],
        }
        positions = get_positions(account_id="1625505")
        assert len(positions) == 1
        assert positions[0].account_id == "1625505"

    def test_filter_no_match(self, mock_avanza):
        mock_avanza.get_accounts_positions.return_value = {
            "withOrderbook": [POSITION_A],
        }
        positions = get_positions(account_id="0000000")
        assert positions == []

    def test_list_response(self, mock_avanza):
        mock_avanza.get_accounts_positions.return_value = [POSITION_A]
        positions = get_positions()
        assert len(positions) == 1

    def test_empty(self, mock_avanza):
        mock_avanza.get_accounts_positions.return_value = {"withOrderbook": []}
        assert get_positions() == []

    def test_fallback_to_positions_key(self, mock_avanza):
        mock_avanza.get_accounts_positions.return_value = {
            "positions": [POSITION_A],
        }
        positions = get_positions()
        assert len(positions) == 1


# ---------------------------------------------------------------------------
# get_buying_power
# ---------------------------------------------------------------------------

class TestGetBuyingPower:
    def test_finds_account(self, mock_avanza):
        mock_avanza.get_overview.return_value = {
            "accounts": [
                {
                    "accountId": "1625505",
                    "buyingPower": {"value": 50000.0},
                    "totalValue": {"value": 120000.0},
                    "ownCapital": {"value": 100000.0},
                },
                {
                    "accountId": "9999999",
                    "buyingPower": {"value": 10000.0},
                    "totalValue": {"value": 30000.0},
                    "ownCapital": {"value": 25000.0},
                },
            ]
        }
        cash = get_buying_power()
        assert isinstance(cash, AccountCash)
        assert cash.buying_power == 50000.0
        assert cash.total_value == 120000.0
        assert cash.own_capital == 100000.0

    def test_specific_account(self, mock_avanza):
        mock_avanza.get_overview.return_value = {
            "accounts": [
                {"accountId": "1625505", "buyingPower": {"value": 50000.0}, "totalValue": {"value": 120000.0}, "ownCapital": {"value": 100000.0}},
                {"accountId": "9999999", "buyingPower": {"value": 10000.0}, "totalValue": {"value": 30000.0}, "ownCapital": {"value": 25000.0}},
            ]
        }
        cash = get_buying_power(account_id="9999999")
        assert cash.buying_power == 10000.0

    def test_account_not_found(self, mock_avanza):
        mock_avanza.get_overview.return_value = {
            "accounts": [
                {"accountId": "1625505", "buyingPower": {"value": 50000.0}, "totalValue": {"value": 120000.0}, "ownCapital": {"value": 100000.0}},
            ]
        }
        cash = get_buying_power(account_id="0000000")
        assert cash.buying_power == 0.0
        assert cash.total_value == 0.0

    def test_alternative_id_key(self, mock_avanza):
        mock_avanza.get_overview.return_value = {
            "accounts": [
                {"id": "1625505", "buyingPower": 80000.0, "totalValue": 200000.0, "ownCapital": 150000.0},
            ]
        }
        cash = get_buying_power()
        assert cash.buying_power == 80000.0


# ---------------------------------------------------------------------------
# get_transactions
# ---------------------------------------------------------------------------

class TestGetTransactions:
    def test_returns_transactions(self, mock_avanza):
        mock_avanza.get_transactions_details.return_value = {
            "transactions": [
                {
                    "id": "TX-1",
                    "type": "BUY",
                    "instrumentName": "MINI S SILVER",
                    "amount": {"value": -2900.0},
                    "priceInTradedCurrency": {"value": 5.80},
                    "volume": {"value": 500},
                    "date": "2026-03-15",
                    "account": {"id": "1625505"},
                },
                {
                    "id": "TX-2",
                    "type": "SELL",
                    "instrumentName": "MINI S SILVER",
                    "amount": {"value": 3100.0},
                    "priceInTradedCurrency": {"value": 6.20},
                    "volume": {"value": 500},
                    "date": "2026-03-20",
                    "account": {"id": "1625505"},
                },
            ]
        }
        txs = get_transactions("2026-03-01", "2026-03-31")
        assert len(txs) == 2
        assert all(isinstance(t, Transaction) for t in txs)
        assert txs[0].transaction_id == "TX-1"
        assert txs[0].transaction_type == "BUY"
        assert txs[0].amount == -2900.0
        assert txs[1].price == 6.20

    def test_with_type_filter(self, mock_avanza):
        mock_avanza.get_transactions_details.return_value = {"transactions": []}
        get_transactions("2026-03-01", "2026-03-31", types=["BUY", "SELL"])

        call_kwargs = mock_avanza.get_transactions_details.call_args[1]
        from avanza.constants import TransactionsDetailsType
        assert TransactionsDetailsType.BUY in call_kwargs["transaction_details_types"]
        assert TransactionsDetailsType.SELL in call_kwargs["transaction_details_types"]

    def test_filter_by_account_id(self, mock_avanza):
        mock_avanza.get_transactions_details.return_value = {
            "transactions": [
                {"id": "TX-1", "type": "BUY", "account": {"id": "1625505"}, "date": "2026-03-15"},
                {"id": "TX-2", "type": "BUY", "account": {"id": "9999999"}, "date": "2026-03-15"},
            ]
        }
        txs = get_transactions("2026-03-01", "2026-03-31", account_id="1625505")
        assert len(txs) == 1
        assert txs[0].transaction_id == "TX-1"

    def test_list_response(self, mock_avanza):
        mock_avanza.get_transactions_details.return_value = [
            {"id": "TX-1", "type": "SELL", "account": {"id": "1625505"}, "date": "2026-03-20"},
        ]
        txs = get_transactions("2026-03-01", "2026-03-31")
        assert len(txs) == 1

    def test_empty(self, mock_avanza):
        mock_avanza.get_transactions_details.return_value = {"transactions": []}
        assert get_transactions("2026-03-01", "2026-03-31") == []
