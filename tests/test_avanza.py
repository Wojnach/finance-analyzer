"""Tests for Avanza API client -- mock-based, no live API calls."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock the avanza package before importing avanza_client
_mock_avanza_module = MagicMock()
_MockAvanzaClass = MagicMock()
_mock_avanza_module.Avanza = _MockAvanzaClass
sys.modules["avanza"] = _mock_avanza_module

import portfolio.avanza_client as mod  # noqa: E402


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the client singleton and mock state before each test."""
    mod._client = None
    _MockAvanzaClass.reset_mock()
    yield
    mod._client = None


@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config.json with Avanza credentials."""
    config = {
        "telegram": {"token": "fake", "chat_id": "123"},
        "avanza": {
            "username": "test_user",
            "password": "test_pass",
            "totp_secret": "JBSWY3DPEHPK3PXP",
        },
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    return cfg_path


@pytest.fixture
def config_file_missing_avanza(tmp_path):
    """Create a config.json without the avanza section."""
    config = {"telegram": {"token": "fake", "chat_id": "123"}}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config), encoding="utf-8")
    return cfg_path


class TestLoadCredentials:
    """Tests for _load_credentials()."""

    def test_loads_credentials_from_config(self, config_file):
        """Credentials are correctly read from config.json."""
        with patch.object(mod, "CONFIG_FILE", config_file):
            creds = mod._load_credentials()

        assert creds["username"] == "test_user"
        assert creds["password"] == "test_pass"
        assert creds["totp_secret"] == "JBSWY3DPEHPK3PXP"

    def test_raises_on_missing_config_file(self, tmp_path):
        """FileNotFoundError when config.json does not exist."""
        with patch.object(mod, "CONFIG_FILE", tmp_path / "nonexistent.json"):
            with pytest.raises(FileNotFoundError):
                mod._load_credentials()

    def test_raises_on_missing_avanza_section(self, config_file_missing_avanza):
        """KeyError when 'avanza' key is not in config."""
        with patch.object(mod, "CONFIG_FILE", config_file_missing_avanza):
            with pytest.raises(KeyError, match="avanza"):
                mod._load_credentials()

    def test_raises_on_empty_credential_field(self, tmp_path):
        """KeyError when a required field is empty."""
        config = {
            "avanza": {
                "username": "user",
                "password": "",
                "totp_secret": "secret",
            }
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(config), encoding="utf-8")

        with patch.object(mod, "CONFIG_FILE", cfg_path):
            with pytest.raises(KeyError, match="avanza.password"):
                mod._load_credentials()


class TestGetClient:
    """Tests for get_client() singleton."""

    def test_creates_client_with_correct_args(self, config_file):
        """Avanza is instantiated with the right credential dict."""
        mock_instance = MagicMock()
        _MockAvanzaClass.return_value = mock_instance

        with patch.object(mod, "CONFIG_FILE", config_file):
            client = mod.get_client()

        _MockAvanzaClass.assert_called_once_with({
            "username": "test_user",
            "password": "test_pass",
            "totpSecret": "JBSWY3DPEHPK3PXP",
        })
        assert client is mock_instance

    def test_singleton_returns_same_instance(self, config_file):
        """Calling get_client() twice returns the same object."""
        _MockAvanzaClass.return_value = MagicMock()

        with patch.object(mod, "CONFIG_FILE", config_file):
            client1 = mod.get_client()
            client2 = mod.get_client()

        assert client1 is client2
        assert _MockAvanzaClass.call_count == 1

    def test_reset_client_clears_singleton(self, config_file):
        """reset_client() allows creating a fresh instance."""
        _MockAvanzaClass.return_value = MagicMock()

        with patch.object(mod, "CONFIG_FILE", config_file):
            client1 = mod.get_client()
            mod.reset_client()
            _MockAvanzaClass.return_value = MagicMock()
            client2 = mod.get_client()

        assert client1 is not client2
        assert _MockAvanzaClass.call_count == 2


class TestFindInstrument:
    """Tests for find_instrument()."""

    def test_calls_search_with_query(self, config_file):
        """find_instrument passes the query to search_for_stock."""
        mock_client = MagicMock()
        mock_client.search_for_stock.return_value = [
            {"id": "12345", "name": "Bitcoin Tracker One"},
            {"id": "67890", "name": "Bitcoin XBT"},
        ]
        _MockAvanzaClass.return_value = mock_client

        with patch.object(mod, "CONFIG_FILE", config_file):
            results = mod.find_instrument("Bitcoin")

        mock_client.search_for_stock.assert_called_once_with("Bitcoin")
        assert len(results) == 2
        assert results[0]["name"] == "Bitcoin Tracker One"


class TestGetPrice:
    """Tests for get_price()."""

    def test_returns_stock_info(self, config_file):
        """get_price returns the full stock info dict."""
        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {
            "lastPrice": 1234.56,
            "change": 12.34,
            "changePercent": 1.01,
            "name": "Test Stock",
        }
        _MockAvanzaClass.return_value = mock_client

        with patch.object(mod, "CONFIG_FILE", config_file):
            info = mod.get_price("12345")

        mock_client.get_stock_info.assert_called_once_with("12345")
        assert info["lastPrice"] == 1234.56
        assert info["name"] == "Test Stock"


class TestGetPositions:
    """Tests for get_positions()."""

    def test_extracts_positions_from_overview(self, config_file):
        """Positions are correctly extracted from the overview response."""
        mock_client = MagicMock()
        mock_client.get_overview.return_value = {
            "accounts": [
                {
                    "name": "ISK",
                    "accountId": "111",
                    "totalValue": 100000,
                    "positions": [
                        {
                            "name": "Bitcoin Tracker",
                            "orderbookId": "12345",
                            "volume": 10,
                            "value": 50000,
                            "profit": 5000,
                            "profitPercent": 11.1,
                            "currency": "SEK",
                        }
                    ],
                },
                {
                    "name": "AF",
                    "accountId": "222",
                    "totalValue": 50000,
                    "positions": [],
                },
            ]
        }
        _MockAvanzaClass.return_value = mock_client

        with patch.object(mod, "CONFIG_FILE", config_file):
            positions = mod.get_positions()

        assert len(positions) == 1
        assert positions[0]["name"] == "Bitcoin Tracker"
        assert positions[0]["account"] == "ISK"
        assert positions[0]["volume"] == 10
        assert positions[0]["value"] == 50000
        assert positions[0]["profit"] == 5000

    def test_empty_positions(self, config_file):
        """Returns empty list when no positions exist."""
        mock_client = MagicMock()
        mock_client.get_overview.return_value = {
            "accounts": [
                {"name": "ISK", "accountId": "111", "totalValue": 0, "positions": []}
            ]
        }
        _MockAvanzaClass.return_value = mock_client

        with patch.object(mod, "CONFIG_FILE", config_file):
            positions = mod.get_positions()

        assert positions == []


class TestGetPortfolioValue:
    """Tests for get_portfolio_value()."""

    def test_sums_all_account_values(self, config_file):
        """Total value is the sum of all account totalValue fields."""
        mock_client = MagicMock()
        mock_client.get_overview.return_value = {
            "accounts": [
                {"name": "ISK", "accountId": "111", "totalValue": 100000},
                {"name": "AF", "accountId": "222", "totalValue": 50000},
                {"name": "KF", "accountId": "333", "totalValue": 25000},
            ]
        }
        _MockAvanzaClass.return_value = mock_client

        with patch.object(mod, "CONFIG_FILE", config_file):
            value = mod.get_portfolio_value()

        assert value == 175000.0
