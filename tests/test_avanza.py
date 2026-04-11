"""Tests for Avanza API client -- mock-based, no live API calls."""

import json
import sys
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

# Mock the avanza package before importing avanza_client
_mock_avanza_module = MagicMock()
_MockAvanzaClass = MagicMock()
_mock_avanza_module.Avanza = _MockAvanzaClass

# Mock the constants sub-module with real-looking enums
import enum


class _MockOrderType(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"

_mock_constants = MagicMock()
_mock_constants.OrderType = _MockOrderType
_mock_avanza_module.constants = _mock_constants
sys.modules["avanza"] = _mock_avanza_module
sys.modules["avanza.constants"] = _mock_constants

import portfolio.avanza_client as mod  # noqa: E402


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the client singleton and mock state before each test."""
    mod._client = None
    mod._account_id = None
    mod._session_client = None
    _MockAvanzaClass.reset_mock()
    with patch.object(mod, "_try_session_auth", return_value=False):
        yield
    mod._client = None
    mod._account_id = None
    mod._session_client = None


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
        with patch.object(mod, "CONFIG_FILE", tmp_path / "nonexistent.json"), pytest.raises(FileNotFoundError):
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

        with patch.object(mod, "CONFIG_FILE", cfg_path), pytest.raises(KeyError, match="avanza.password"):
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

    def test_uses_session_price_when_available(self):
        """BankID session path should use portfolio.avanza_session directly."""
        with patch.object(mod, "_try_session_auth", return_value=True), \
             patch("portfolio.avanza_session.get_instrument_price",
                   return_value={"lastPrice": 42.0, "name": "Session Price"}) as mock_session:
            info = mod.get_price("12345")

        mock_session.assert_called_once_with("12345")
        assert info["lastPrice"] == 42.0

    def test_session_price_failure_falls_back_to_totp(self, config_file):
        """Session lookup errors should fall back to the avanza-api client."""
        mock_client = MagicMock()
        mock_client.get_stock_info.return_value = {"lastPrice": 55.0, "name": "Fallback"}
        _MockAvanzaClass.return_value = mock_client

        with patch.object(mod, "_try_session_auth", return_value=True), \
             patch("portfolio.avanza_session.get_instrument_price",
                   side_effect=RuntimeError("session failed")), \
             patch.object(mod, "CONFIG_FILE", config_file):
            info = mod.get_price("12345")

        mock_client.get_stock_info.assert_called_once_with("12345")
        assert info["lastPrice"] == 55.0


class TestGetPositions:
    """Tests for get_positions()."""

    def test_extracts_positions_from_overview(self, config_file):
        """Positions are correctly extracted from the overview response.

        A-AV-2: Uses the whitelisted account 1625505 — non-whitelisted
        accounts in the same overview are filtered out.
        """
        mock_client = MagicMock()
        mock_client.get_overview.return_value = {
            "accounts": [
                {
                    "name": "ISK",
                    "accountId": "1625505",  # whitelisted
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
                    "accountId": "222",  # not whitelisted — must be ignored
                    "totalValue": 50000,
                    "positions": [
                        {"name": "Should Not Appear", "orderbookId": "X", "volume": 1,
                         "value": 1, "profit": 0, "profitPercent": 0, "currency": "SEK"}
                    ],
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
                {"name": "ISK", "accountId": "1625505", "totalValue": 0, "positions": []}
            ]
        }
        _MockAvanzaClass.return_value = mock_client

        with patch.object(mod, "CONFIG_FILE", config_file):
            positions = mod.get_positions()

        assert positions == []

    def test_filters_pension_account(self, config_file):
        """A-AV-2: Pension account 2674244 must NEVER appear in positions
        even if Avanza returns it as ISK-typed."""
        mock_client = MagicMock()
        mock_client.get_overview.return_value = {
            "accounts": [
                {
                    "name": "Pension",
                    "accountId": "2674244",
                    "accountType": "Investeringssparkonto_ISK",  # ISK-shaped!
                    "totalValue": 999999,
                    "positions": [
                        {"name": "Pension Holding", "orderbookId": "P1", "volume": 1,
                         "value": 999999, "profit": 0, "profitPercent": 0, "currency": "SEK"}
                    ],
                },
                {
                    "name": "Trading ISK",
                    "accountId": "1625505",
                    "totalValue": 100000,
                    "positions": [],
                },
            ]
        }
        _MockAvanzaClass.return_value = mock_client

        with patch.object(mod, "CONFIG_FILE", config_file):
            positions = mod.get_positions()

        # No pension positions leaked through
        assert all(p["account_id"] == "1625505" for p in positions)
        assert not any("Pension" in p.get("name", "") for p in positions)

    def test_uses_session_positions_when_available(self):
        """BankID session path should use portfolio.avanza_session directly."""
        expected = [{"name": "Session Position", "orderbook_id": "12345"}]
        with patch.object(mod, "_try_session_auth", return_value=True), \
             patch("portfolio.avanza_session.get_positions", return_value=expected) as mock_session:
            positions = mod.get_positions()

        mock_session.assert_called_once_with()
        assert positions == expected


class TestGetPortfolioValue:
    """Tests for get_portfolio_value()."""

    def test_sums_only_whitelisted_accounts(self, config_file):
        """A-AV-2: Only whitelisted accounts contribute to the total.
        Non-whitelisted accounts (AF, KF, pension) must be excluded."""
        mock_client = MagicMock()
        mock_client.get_overview.return_value = {
            "accounts": [
                {"name": "ISK", "accountId": "1625505", "totalValue": 100000},
                {"name": "AF", "accountId": "222", "totalValue": 50000},  # excluded
                {"name": "KF", "accountId": "333", "totalValue": 25000},  # excluded
                {"name": "Pension", "accountId": "2674244", "totalValue": 999999},  # excluded
            ]
        }
        _MockAvanzaClass.return_value = mock_client

        with patch.object(mod, "CONFIG_FILE", config_file):
            value = mod.get_portfolio_value()

        # Only the 100000 from 1625505, NOT 1175000
        assert value == 100000.0


# --- Helper to set up a client with ISK account ---
# A-AV-2: defaults to the whitelisted account "1625505". Tests that need a
# non-whitelisted scenario should pass account_id explicitly.

def _make_client_with_isk(config_file, account_id="1625505", account_type="ISK"):
    """Create a mock client that returns an overview with an ISK account."""
    mock_client = MagicMock()
    mock_client.get_overview.return_value = {
        "accounts": [
            {
                "name": "ISK",
                "accountId": account_id,
                "accountType": account_type,
                "totalValue": 100000,
                "positions": [],
            }
        ]
    }
    _MockAvanzaClass.return_value = mock_client
    return mock_client


class TestGetAccountId:
    """Tests for get_account_id() — A-AV-2 whitelist enforcement."""

    def test_returns_whitelisted_isk_account_id(self, config_file):
        _make_client_with_isk(config_file, account_id="1625505")
        with patch.object(mod, "CONFIG_FILE", config_file):
            aid = mod.get_account_id()
        assert aid == "1625505"

    def test_caches_after_first_call(self, config_file):
        mock_client = _make_client_with_isk(config_file)
        with patch.object(mod, "CONFIG_FILE", config_file):
            mod.get_account_id()
            mod.get_account_id()
        # get_overview called only once thanks to caching
        assert mock_client.get_overview.call_count == 1

    def test_raises_when_no_isk(self, config_file):
        mock_client = MagicMock()
        mock_client.get_overview.return_value = {
            "accounts": [
                {"name": "AF", "accountId": "111", "accountType": "Aktie_Fondkonto",
                 "totalValue": 0, "positions": []},
            ]
        }
        _MockAvanzaClass.return_value = mock_client
        with patch.object(mod, "CONFIG_FILE", config_file), pytest.raises(RuntimeError, match="No whitelisted ISK"):
            mod.get_account_id()

    def test_finds_whitelisted_among_multiple_accounts(self, config_file):
        """If multiple ISK accounts exist, only the whitelisted one is returned."""
        mock_client = MagicMock()
        mock_client.get_overview.return_value = {
            "accounts": [
                {"name": "AF", "accountId": "111", "accountType": "Aktie_Fondkonto",
                 "totalValue": 0, "positions": []},
                {"name": "ISK", "accountId": "1625505", "accountType": "Investeringssparkonto_ISK",
                 "totalValue": 50000, "positions": []},
            ]
        }
        _MockAvanzaClass.return_value = mock_client
        with patch.object(mod, "CONFIG_FILE", config_file):
            assert mod.get_account_id() == "1625505"

    def test_rejects_non_whitelisted_isk_account(self, config_file):
        """A-AV-2: An ISK-typed account NOT in ALLOWED_ACCOUNT_IDS must be
        rejected, even if it's the only ISK in the response. This prevents
        a future-added account (or pension reclassified as ISK) from being
        traded on."""
        mock_client = MagicMock()
        mock_client.get_overview.return_value = {
            "accounts": [
                {"name": "Stranger ISK", "accountId": "9999999",
                 "accountType": "Investeringssparkonto_ISK",
                 "totalValue": 50000, "positions": []},
            ]
        }
        _MockAvanzaClass.return_value = mock_client
        with patch.object(mod, "CONFIG_FILE", config_file), \
             pytest.raises(RuntimeError, match="No whitelisted ISK"):
            mod.get_account_id()

    def test_rejects_pension_2674244_even_if_iskish(self, config_file):
        """A-AV-2: The actual pension account ID 2674244 must be rejected
        even if Avanza tags it as ISK-typed."""
        mock_client = MagicMock()
        mock_client.get_overview.return_value = {
            "accounts": [
                {"name": "Pension", "accountId": "2674244",
                 "accountType": "Investeringssparkonto_ISK",
                 "totalValue": 999999, "positions": []},
            ]
        }
        _MockAvanzaClass.return_value = mock_client
        with patch.object(mod, "CONFIG_FILE", config_file), \
             pytest.raises(RuntimeError, match="No whitelisted ISK"):
            mod.get_account_id()

    def test_picks_whitelisted_when_pension_listed_first(self, config_file):
        """A-AV-2: If Avanza re-orders the response and pension comes first,
        we still pick the correct whitelisted account."""
        mock_client = MagicMock()
        mock_client.get_overview.return_value = {
            "accounts": [
                {"name": "Pension", "accountId": "2674244",
                 "accountType": "Investeringssparkonto_ISK",
                 "totalValue": 999999, "positions": []},
                {"name": "Trading ISK", "accountId": "1625505",
                 "accountType": "Investeringssparkonto_ISK",
                 "totalValue": 100000, "positions": []},
            ]
        }
        _MockAvanzaClass.return_value = mock_client
        with patch.object(mod, "CONFIG_FILE", config_file):
            assert mod.get_account_id() == "1625505"

    def test_allowed_account_ids_constant_exists(self):
        """A-AV-2: The constant must be a frozen set-like object listing
        only the trading account."""
        assert hasattr(mod, "ALLOWED_ACCOUNT_IDS")
        assert "1625505" in mod.ALLOWED_ACCOUNT_IDS
        assert "2674244" not in mod.ALLOWED_ACCOUNT_IDS


class TestPlaceBuyOrder:
    """Tests for place_buy_order()."""

    def test_places_buy_via_client(self, config_file):
        mock_client = _make_client_with_isk(config_file)
        mock_client.place_order.return_value = {
            "orderId": "ORD-1",
            "orderRequestStatus": "SUCCESS",
            "message": "",
        }
        with patch.object(mod, "CONFIG_FILE", config_file):
            result = mod.place_buy_order("5533", price=245.0, volume=50)

        assert result["orderRequestStatus"] == "SUCCESS"
        assert result["orderId"] == "ORD-1"
        mock_client.place_order.assert_called_once()
        call_kwargs = mock_client.place_order.call_args
        assert call_kwargs.kwargs["order_type"] == _MockOrderType.BUY
        assert call_kwargs.kwargs["price"] == 245.0
        assert call_kwargs.kwargs["volume"] == 50

    def test_buy_defaults_valid_until_today(self, config_file):
        mock_client = _make_client_with_isk(config_file)
        mock_client.place_order.return_value = {"orderId": "1", "orderRequestStatus": "SUCCESS", "message": ""}
        with patch.object(mod, "CONFIG_FILE", config_file):
            mod.place_buy_order("5533", price=100.0, volume=10)
        call_kwargs = mock_client.place_order.call_args.kwargs
        assert call_kwargs["valid_until"] == date.today()

    def test_buy_custom_valid_until(self, config_file):
        mock_client = _make_client_with_isk(config_file)
        mock_client.place_order.return_value = {"orderId": "1", "orderRequestStatus": "SUCCESS", "message": ""}
        custom_date = date(2026, 3, 15)
        with patch.object(mod, "CONFIG_FILE", config_file):
            mod.place_buy_order("5533", price=100.0, volume=10, valid_until=custom_date)
        assert mock_client.place_order.call_args.kwargs["valid_until"] == custom_date

    def test_buy_rejects_zero_volume(self, config_file):
        _make_client_with_isk(config_file)
        with patch.object(mod, "CONFIG_FILE", config_file), pytest.raises(ValueError, match="Volume"):
            mod.place_buy_order("5533", price=100.0, volume=0)

    def test_buy_rejects_negative_price(self, config_file):
        _make_client_with_isk(config_file)
        with patch.object(mod, "CONFIG_FILE", config_file), pytest.raises(ValueError, match="Price"):
            mod.place_buy_order("5533", price=-5.0, volume=10)


class TestPlaceSellOrder:
    """Tests for place_sell_order()."""

    def test_places_sell_via_client(self, config_file):
        mock_client = _make_client_with_isk(config_file)
        mock_client.place_order.return_value = {
            "orderId": "ORD-2",
            "orderRequestStatus": "SUCCESS",
            "message": "",
        }
        with patch.object(mod, "CONFIG_FILE", config_file):
            result = mod.place_sell_order("5533", price=260.0, volume=25)

        assert result["orderId"] == "ORD-2"
        call_kwargs = mock_client.place_order.call_args.kwargs
        assert call_kwargs["order_type"] == _MockOrderType.SELL
        assert call_kwargs["price"] == 260.0
        assert call_kwargs["volume"] == 25


class TestGetOrderStatus:
    """Tests for get_order_status()."""

    def test_returns_order_dict(self, config_file):
        mock_client = _make_client_with_isk(config_file)
        mock_client.get_order.return_value = {
            "orderId": "ORD-1",
            "state": "FILLED",
            "price": 245.0,
            "volume": 50,
        }
        with patch.object(mod, "CONFIG_FILE", config_file):
            status = mod.get_order_status("ORD-1")
        assert status["state"] == "FILLED"
        mock_client.get_order.assert_called_once_with("1625505", "ORD-1")


class TestDeleteOrder:
    """Tests for delete_order()."""

    def test_cancels_order(self, config_file):
        mock_client = _make_client_with_isk(config_file)
        mock_client.delete_order.return_value = {
            "orderId": "ORD-1",
            "orderRequestStatus": "SUCCESS",
            "messages": "",
        }
        with patch.object(mod, "CONFIG_FILE", config_file):
            result = mod.delete_order("ORD-1")
        assert result["orderRequestStatus"] == "SUCCESS"
        mock_client.delete_order.assert_called_once_with("1625505", "ORD-1")
