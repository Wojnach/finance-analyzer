"""Tests for portfolio.avanza_session — session load, save, expiry, API wrapper."""

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza_session import (
    API_BASE,
    AvanzaSessionError,
    api_get,
    get_instrument_price,
    get_positions,
    is_session_expiring_soon,
    load_session,
    session_remaining_minutes,
    verify_session,
)

# --- Fixtures ---


def _make_session_data(
    expires_in_minutes=120,
    cookies=None,
    security_token="test-token-abc",
    authentication_session="test-auth-session",
    customer_id="12345",
):
    """Create a valid session data dict."""
    now = datetime.now(UTC)
    return {
        "cookies": cookies or [
            {
                "name": "ava_cookie",
                "value": "cookie_value",
                "domain": ".avanza.se",
                "path": "/",
                "secure": True,
                "httpOnly": False,
                "sameSite": "None",
            },
            {
                "name": "session_id",
                "value": "sess_123",
                "domain": ".avanza.se",
                "path": "/",
                "secure": True,
                "httpOnly": True,
                "sameSite": "Lax",
            },
        ],
        "security_token": security_token,
        "authentication_session": authentication_session,
        "customer_id": customer_id,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(minutes=expires_in_minutes)).isoformat(),
        "max_inactive_minutes": expires_in_minutes,
    }


@pytest.fixture
def session_file():
    """Create a temp session file and patch SESSION_FILE + STORAGE_STATE_FILE."""
    with tempfile.TemporaryDirectory() as td:
        sf = Path(td) / "avanza_session.json"
        ssf = Path(td) / "avanza_storage_state.json"
        # Create a minimal storage state file so load_session doesn't fail
        ssf.write_text("{}", encoding="utf-8")
        with patch("portfolio.avanza_session.SESSION_FILE", sf), \
             patch("portfolio.avanza_session.STORAGE_STATE_FILE", ssf):
            yield sf


def _make_mock_pw_context(status=200, json_data=None, ok=True):
    """Create a mock Playwright context for testing API functions."""
    mock_ctx = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.ok = ok
    mock_resp.json.return_value = json_data or {}
    mock_resp.text.return_value = ""
    mock_ctx.request.get.return_value = mock_resp
    return mock_ctx, mock_resp


# --- load_session tests ---


class TestLoadSession:
    def test_load_valid_session(self, session_file):
        data = _make_session_data(expires_in_minutes=120)
        session_file.write_text(json.dumps(data), encoding="utf-8")
        result = load_session()
        assert result["security_token"] == "test-token-abc"
        assert result["customer_id"] == "12345"
        assert len(result["cookies"]) == 2

    def test_missing_file_raises(self, session_file):
        # session_file fixture patches the path but doesn't create the file
        with pytest.raises(AvanzaSessionError, match="No session file"):
            load_session()

    def test_expired_session_raises(self, session_file):
        data = _make_session_data(expires_in_minutes=-10)  # Already expired
        session_file.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(AvanzaSessionError, match="expired"):
            load_session()

    def test_corrupt_json_raises(self, session_file):
        session_file.write_text("{bad json", encoding="utf-8")
        with pytest.raises(AvanzaSessionError, match="Failed to read"):
            load_session()

    def test_no_storage_state_raises(self, session_file):
        """load_session requires STORAGE_STATE_FILE to exist (Playwright auth)."""
        data = _make_session_data()
        session_file.write_text(json.dumps(data), encoding="utf-8")
        # Remove the storage state file created by the fixture
        import portfolio.avanza_session as mod
        mod.STORAGE_STATE_FILE.unlink(missing_ok=True)
        with pytest.raises(AvanzaSessionError, match="storage state"):
            load_session()

    def test_no_expiry_still_loads(self, session_file):
        data = _make_session_data()
        del data["expires_at"]
        session_file.write_text(json.dumps(data), encoding="utf-8")
        result = load_session()
        assert result["security_token"] == "test-token-abc"

    def test_invalid_expiry_format_still_loads(self, session_file):
        data = _make_session_data()
        data["expires_at"] = "not-a-date"
        session_file.write_text(json.dumps(data), encoding="utf-8")
        result = load_session()
        assert result["security_token"] == "test-token-abc"


# --- session_remaining_minutes tests ---


class TestSessionRemainingMinutes:
    def test_valid_session_returns_minutes(self, session_file):
        data = _make_session_data(expires_in_minutes=120)
        session_file.write_text(json.dumps(data), encoding="utf-8")
        remaining = session_remaining_minutes()
        assert remaining is not None
        assert 118 < remaining <= 120  # Allow small timing drift

    def test_expired_session_returns_negative(self, session_file):
        data = _make_session_data(expires_in_minutes=-60)
        session_file.write_text(json.dumps(data), encoding="utf-8")
        remaining = session_remaining_minutes()
        assert remaining is not None
        assert remaining < 0

    def test_no_file_returns_none(self, session_file):
        assert session_remaining_minutes() is None

    def test_no_expiry_returns_none(self, session_file):
        data = _make_session_data()
        del data["expires_at"]
        session_file.write_text(json.dumps(data), encoding="utf-8")
        assert session_remaining_minutes() is None


# --- is_session_expiring_soon tests ---


class TestIsSessionExpiringSoon:
    def test_fresh_session_not_expiring(self, session_file):
        data = _make_session_data(expires_in_minutes=120)
        session_file.write_text(json.dumps(data), encoding="utf-8")
        assert is_session_expiring_soon(threshold_minutes=60.0) is False

    def test_almost_expired_is_expiring(self, session_file):
        data = _make_session_data(expires_in_minutes=30)
        session_file.write_text(json.dumps(data), encoding="utf-8")
        assert is_session_expiring_soon(threshold_minutes=60.0) is True

    def test_no_session_is_expiring(self, session_file):
        assert is_session_expiring_soon() is True

    def test_custom_threshold(self, session_file):
        data = _make_session_data(expires_in_minutes=45)
        session_file.write_text(json.dumps(data), encoding="utf-8")
        assert is_session_expiring_soon(threshold_minutes=30.0) is False
        assert is_session_expiring_soon(threshold_minutes=60.0) is True


# --- verify_session tests (Playwright-based) ---


class TestVerifySession:
    @patch("portfolio.avanza_session._get_playwright_context")
    def test_valid_session_returns_true(self, mock_get_ctx, session_file):
        mock_ctx, mock_resp = _make_mock_pw_context(status=200, ok=True)
        mock_get_ctx.return_value = mock_ctx
        assert verify_session() is True
        mock_ctx.request.get.assert_called_once()

    @patch("portfolio.avanza_session._get_playwright_context")
    def test_401_returns_false(self, mock_get_ctx, session_file):
        mock_ctx, mock_resp = _make_mock_pw_context(status=401, ok=False)
        mock_get_ctx.return_value = mock_ctx
        assert verify_session() is False

    @patch("portfolio.avanza_session._get_playwright_context")
    def test_network_error_returns_false(self, mock_get_ctx, session_file):
        mock_get_ctx.side_effect = ConnectionError("Network down")
        assert verify_session() is False


# --- api_get tests (Playwright-based) ---


class TestApiGet:
    @patch("portfolio.avanza_session._get_playwright_context")
    def test_successful_get(self, mock_get_ctx, session_file):
        mock_ctx, mock_resp = _make_mock_pw_context(
            status=200, json_data={"data": "test"}, ok=True
        )
        mock_get_ctx.return_value = mock_ctx

        result = api_get("/_api/test")
        assert result == {"data": "test"}
        mock_ctx.request.get.assert_called_once_with(f"{API_BASE}/_api/test")

    @patch("portfolio.avanza_session._get_playwright_context")
    def test_401_raises_session_error(self, mock_get_ctx, session_file):
        mock_ctx, mock_resp = _make_mock_pw_context(status=401, ok=False)
        mock_get_ctx.return_value = mock_ctx

        with pytest.raises(AvanzaSessionError, match="401"):
            api_get("/_api/test")

    @patch("portfolio.avanza_session._get_playwright_context")
    def test_full_url_not_prepended(self, mock_get_ctx, session_file):
        mock_ctx, mock_resp = _make_mock_pw_context(
            status=200, json_data={}, ok=True
        )
        mock_get_ctx.return_value = mock_ctx

        api_get("https://custom.url/api/test")
        mock_ctx.request.get.assert_called_once_with("https://custom.url/api/test")

    @patch("portfolio.avanza_session._get_playwright_context")
    def test_non_ok_raises_runtime_error(self, mock_get_ctx, session_file):
        mock_ctx, mock_resp = _make_mock_pw_context(status=500, ok=False)
        mock_get_ctx.return_value = mock_ctx

        with pytest.raises(RuntimeError, match="Avanza API error 500"):
            api_get("/_api/test")


# --- get_positions tests (mock api_get) ---


class TestGetPositions:
    @patch("portfolio.avanza_session.api_get")
    def test_parses_positions(self, mock_api_get, session_file):
        mock_api_get.return_value = {
            "withOrderbook": [
                {
                    "instrument": {
                        "name": "SAAB B",
                        "id": "inst-1",
                        "orderbook": {
                            "id": "5533",
                            "name": "SAAB B",
                            "type": "STOCK",
                            "quote": {
                                "latest": {"value": 735.0},
                                "changePercent": {"value": 1.2},
                            },
                        },
                        "type": "STOCK",
                        "currency": "SEK",
                    },
                    "volume": {"value": 100},
                    "value": {"value": 73500.0},
                    "acquiredValue": {"value": 68500.0},
                    "lastTradingDayPerformance": {},
                    "account": {"id": "acc-1", "type": "ISK"},
                }
            ]
        }

        positions = get_positions()
        assert len(positions) == 1
        assert positions[0]["name"] == "SAAB B"
        assert positions[0]["orderbook_id"] == "5533"
        assert positions[0]["volume"] == 100
        assert positions[0]["value"] == 73500.0

    @patch("portfolio.avanza_session.api_get")
    def test_empty_positions(self, mock_api_get, session_file):
        mock_api_get.return_value = {"withOrderbook": []}
        positions = get_positions()
        assert positions == []


# --- get_instrument_price tests (mock api_get) ---


class TestGetInstrumentPrice:
    @patch("portfolio.avanza_session.api_get")
    def test_stock_lookup(self, mock_api_get, session_file):
        mock_api_get.return_value = {"lastPrice": 735.0, "changePercent": 1.2}
        result = get_instrument_price("5533")
        assert result["lastPrice"] == 735.0
        # First call should try stock endpoint
        mock_api_get.assert_called_once_with(
            "/_api/market-guide/stock/5533",
        )

    @patch("portfolio.avanza_session.api_get")
    def test_fallback_to_certificate(self, mock_api_get, session_file):
        call_count = 0

        def side_effect(path, **kwargs):
            nonlocal call_count
            call_count += 1
            if "stock" in path:
                raise Exception("Not found")
            return {"lastPrice": 42.0}

        mock_api_get.side_effect = side_effect
        result = get_instrument_price("9999")
        assert result["lastPrice"] == 42.0
        assert call_count == 2  # stock failed, certificate succeeded


# --- Login script helpers ---


class TestLoginHelpers:
    def test_no_tokens_not_logged_in(self):
        from scripts.avanza_login import _is_logged_in
        assert _is_logged_in({}) is False

    def test_customer_id_means_logged_in(self):
        from scripts.avanza_login import _is_logged_in
        assert _is_logged_in({"customer_id": "12345"}) is True

    def test_csid_cstoken_cookies_means_logged_in(self):
        from scripts.avanza_login import _is_logged_in
        cookies = [
            {"name": "csid", "value": "abc"},
            {"name": "cstoken", "value": "def"},
        ]
        assert _is_logged_in({}, cookies=cookies) is True

    def test_partial_cookies_not_logged_in(self):
        from scripts.avanza_login import _is_logged_in
        cookies = [{"name": "csid", "value": "abc"}]
        assert _is_logged_in({}, cookies=cookies) is False

    def test_irrelevant_cookies_not_logged_in(self):
        from scripts.avanza_login import _is_logged_in
        cookies = [{"name": "session", "value": "xyz"}]
        assert _is_logged_in({}, cookies=cookies) is False


# --- place_stop_loss tests ---


class TestPlaceStopLoss:
    @patch("portfolio.avanza_session.api_post")
    def test_standard_stop_loss(self, mock_post, session_file):
        from portfolio.avanza_session import place_stop_loss

        mock_post.return_value = {"status": "SUCCESS", "stoplossOrderId": "SL-123"}

        result = place_stop_loss("856394", trigger_price=23.0, sell_price=22.5, volume=100)
        assert result["status"] == "SUCCESS"
        assert result["stoplossOrderId"] == "SL-123"

        # Verify correct endpoint and payload structure
        call_args = mock_post.call_args
        assert call_args[0][0] == "/_api/trading/stoploss/new"
        payload = call_args[0][1]
        assert payload["accountId"] == "1625505"
        assert payload["orderBookId"] == "856394"
        assert payload["stopLossTrigger"]["type"] == "LESS_OR_EQUAL"
        assert payload["stopLossTrigger"]["value"] == 23.0
        assert payload["stopLossTrigger"]["valueType"] == "MONETARY"
        assert payload["stopLossOrderEvent"]["type"] == "SELL"
        assert payload["stopLossOrderEvent"]["price"] == 22.5
        assert payload["stopLossOrderEvent"]["volume"] == 100

    @patch("portfolio.avanza_session.api_post")
    def test_custom_account_id(self, mock_post, session_file):
        """Whitelisted account ID is accepted and propagated to the payload."""
        from portfolio.avanza_session import ALLOWED_ACCOUNT_IDS, place_stop_loss

        mock_post.return_value = {"status": "SUCCESS", "stoplossOrderId": "SL-456"}

        acct = next(iter(ALLOWED_ACCOUNT_IDS))
        place_stop_loss("856394", 23.0, 22.5, 100, account_id=acct)
        payload = mock_post.call_args[0][1]
        assert payload["accountId"] == acct

    @patch("portfolio.avanza_session.api_post")
    def test_non_whitelisted_account_raises(self, mock_post, session_file):
        """Non-whitelisted account must be rejected before any API call."""
        from portfolio.avanza_session import place_stop_loss

        with pytest.raises(ValueError, match="non-whitelisted"):
            place_stop_loss("856394", 23.0, 22.5, 100, account_id="9999")
        mock_post.assert_not_called()

    @patch("portfolio.avanza_session.api_post")
    def test_follow_downwards_trigger_type(self, mock_post, session_file):
        from portfolio.avanza_session import place_stop_loss

        mock_post.return_value = {"status": "SUCCESS", "stoplossOrderId": "SL-TRAIL"}

        place_stop_loss(
            "856394", trigger_price=5.0, sell_price=0, volume=50,
            trigger_type="FOLLOW_DOWNWARDS", value_type="PERCENTAGE",
        )
        payload = mock_post.call_args[0][1]
        assert payload["stopLossTrigger"]["type"] == "FOLLOW_DOWNWARDS"
        assert payload["stopLossTrigger"]["valueType"] == "PERCENTAGE"
        assert payload["stopLossTrigger"]["value"] == 5.0

    @patch("portfolio.avanza_session.api_post")
    def test_stop_loss_rejects_zero_sell_price_monetary(self, mock_post, session_file):
        """BUG-223: Non-trailing MONETARY stop with sell_price=0 must raise."""
        from portfolio.avanza_session import place_stop_loss

        with pytest.raises(ValueError, match="sell_price > 0"):
            place_stop_loss("856394", trigger_price=23.0, sell_price=0, volume=100)
        mock_post.assert_not_called()

    @patch("portfolio.avanza_session.api_post")
    def test_stop_loss_allows_zero_sell_price_trailing(self, mock_post, session_file):
        """BUG-223: Trailing stop (FOLLOW_DOWNWARDS) with sell_price=0 is valid."""
        from portfolio.avanza_session import place_stop_loss

        mock_post.return_value = {"status": "SUCCESS", "stoplossOrderId": "SL-T0"}

        result = place_stop_loss(
            "856394", trigger_price=5.0, sell_price=0, volume=50,
            trigger_type="FOLLOW_DOWNWARDS", value_type="MONETARY",
        )
        assert result["status"] == "SUCCESS"
        mock_post.assert_called_once()


class TestPlaceTrailingStop:
    @patch("portfolio.avanza_session.api_post")
    def test_trailing_stop_delegates_correctly(self, mock_post, session_file):
        from portfolio.avanza_session import place_trailing_stop

        mock_post.return_value = {"status": "SUCCESS", "stoplossOrderId": "SL-HW"}

        result = place_trailing_stop("856394", trail_percent=5.0, volume=100)
        assert result["status"] == "SUCCESS"

        payload = mock_post.call_args[0][1]
        assert payload["stopLossTrigger"]["type"] == "FOLLOW_DOWNWARDS"
        assert payload["stopLossTrigger"]["valueType"] == "PERCENTAGE"
        assert payload["stopLossTrigger"]["value"] == 5.0
        assert payload["stopLossOrderEvent"]["volume"] == 100

    @patch("portfolio.avanza_session.api_post")
    def test_trailing_stop_custom_percent(self, mock_post, session_file):
        from portfolio.avanza_session import place_trailing_stop

        mock_post.return_value = {"status": "SUCCESS", "stoplossOrderId": "SL-HW2"}

        place_trailing_stop("856394", trail_percent=3.0, volume=50, valid_days=14)
        payload = mock_post.call_args[0][1]
        assert payload["stopLossTrigger"]["value"] == 3.0
        assert payload["stopLossOrderEvent"]["validDays"] == 14

    @patch("portfolio.avanza_session.api_post")
    def test_trailing_stop_failure(self, mock_post, session_file):
        from portfolio.avanza_session import place_trailing_stop

        mock_post.return_value = {"status": "ERROR", "message": "Invalid instrument"}

        result = place_trailing_stop("000000", trail_percent=5.0, volume=1)
        assert result["status"] == "ERROR"


class TestGetStopLosses:
    @patch("portfolio.avanza_session.api_get")
    def test_returns_list(self, mock_get, session_file):
        from portfolio.avanza_session import get_stop_losses

        mock_get.return_value = [
            {"id": "SL-1", "orderbookId": "856394", "status": "ACTIVE"},
            {"id": "SL-2", "orderbookId": "2334960", "status": "ACTIVE"},
        ]

        result = get_stop_losses()
        assert len(result) == 2
        assert result[0]["id"] == "SL-1"

    @patch("portfolio.avanza_session.api_get")
    def test_empty_list(self, mock_get, session_file):
        from portfolio.avanza_session import get_stop_losses

        mock_get.return_value = []
        assert get_stop_losses() == []

    @patch("portfolio.avanza_session.api_get")
    def test_error_returns_empty(self, mock_get, session_file):
        from portfolio.avanza_session import get_stop_losses

        mock_get.side_effect = RuntimeError("API error")
        assert get_stop_losses() == []


class TestGetBuyingPower:
    """Tests for get_buying_power — multi-shape response parsing (Bug C7 fix).

    As of 2026-04-09 Avanza's ``/_api/account-overview/overview/categorizedAccounts``
    endpoint may return any of three shapes (legacy categorized, new flat, new
    categorized) and may use any of four ID field names. These tests exercise
    each shape plus failure modes.
    """

    _ACCT = "1625505"

    def _make_acc(self, acc_id_field="accountId", acc_id="1625505"):
        """Build a minimal account record with the given ID field."""
        return {
            acc_id_field: acc_id,
            "buyingPower": {"value": 12345.0},
            "totalValue": {"value": 54321.0},
            "ownCapital": {"value": 42000.0},
        }

    @patch("portfolio.avanza_session.api_get")
    def test_legacy_categorized_shape(self, mock_api_get, session_file):
        """Path A: legacy data.categorizedAccounts[].accounts[] with accountId."""
        from portfolio.avanza_session import get_buying_power

        mock_api_get.return_value = {
            "categorizedAccounts": [
                {"accounts": [self._make_acc("accountId", self._ACCT)]},
            ],
        }
        result = get_buying_power(self._ACCT)
        assert result is not None
        assert result["buying_power"] == 12345.0
        assert result["total_value"] == 54321.0
        assert result["own_capital"] == 42000.0

    @patch("portfolio.avanza_session.api_get")
    def test_new_flat_shape(self, mock_api_get, session_file):
        """Path B: new data.accounts[] flat shape with id field."""
        from portfolio.avanza_session import get_buying_power

        mock_api_get.return_value = {
            "categories": [],
            "accounts": [self._make_acc("id", self._ACCT)],
            "loans": [],
        }
        result = get_buying_power(self._ACCT)
        assert result is not None
        assert result["buying_power"] == 12345.0
        assert result["total_value"] == 54321.0
        assert result["own_capital"] == 42000.0

    @patch("portfolio.avanza_session.api_get")
    def test_new_categorized_shape(self, mock_api_get, session_file):
        """Path C: new data.categories[].accounts[] shape with id field."""
        from portfolio.avanza_session import get_buying_power

        mock_api_get.return_value = {
            "categories": [
                {"accounts": [self._make_acc("id", self._ACCT)]},
            ],
            "accounts": [],
            "loans": [],
        }
        result = get_buying_power(self._ACCT)
        assert result is not None
        assert result["buying_power"] == 12345.0

    @patch("portfolio.avanza_session.api_get")
    def test_alternate_id_field_accountnumber(self, mock_api_get, session_file):
        """Multi-field ID fallback: accountNumber works when accountId/id missing."""
        from portfolio.avanza_session import get_buying_power

        mock_api_get.return_value = {
            "accounts": [self._make_acc("accountNumber", self._ACCT)],
        }
        result = get_buying_power(self._ACCT)
        assert result is not None
        assert result["buying_power"] == 12345.0

    @patch("portfolio.avanza_session.api_get")
    def test_alternate_balance_field(self, mock_api_get, session_file):
        """Multi-field balance fallback: buyingPowerAvailable works when buyingPower missing."""
        from portfolio.avanza_session import get_buying_power

        mock_api_get.return_value = {
            "accounts": [
                {
                    "id": self._ACCT,
                    "buyingPowerAvailable": {"value": 9999.0},
                    "accountTotalValue": {"value": 50000.0},
                    "netDeposit": {"value": 40000.0},
                },
            ],
        }
        result = get_buying_power(self._ACCT)
        assert result is not None
        assert result["buying_power"] == 9999.0
        assert result["total_value"] == 50000.0
        assert result["own_capital"] == 40000.0

    @patch("portfolio.avanza_session.api_get")
    def test_account_not_found_returns_none(self, mock_api_get, session_file):
        """Target account not present in any shape → None (not silent zero)."""
        from portfolio.avanza_session import get_buying_power

        mock_api_get.return_value = {
            "accounts": [self._make_acc("id", "9999")],  # wrong ID
        }
        assert get_buying_power(self._ACCT) is None

    @patch("portfolio.avanza_session.api_get")
    def test_api_get_raises_returns_none(self, mock_api_get, session_file):
        """HTTP / session errors from api_get → None, never raise."""
        from portfolio.avanza_session import get_buying_power

        mock_api_get.side_effect = RuntimeError("Avanza API error 500: boom")
        assert get_buying_power(self._ACCT) is None

    @patch("portfolio.avanza_session.api_get")
    def test_unexpected_response_type_returns_none(self, mock_api_get, session_file):
        """Non-dict top-level response → None."""
        from portfolio.avanza_session import get_buying_power

        mock_api_get.return_value = ["not", "a", "dict"]
        assert get_buying_power(self._ACCT) is None

    @patch("portfolio.avanza_session.api_get")
    def test_empty_response_returns_none(self, mock_api_get, session_file):
        """Empty dict (no shapes populated) → None."""
        from portfolio.avanza_session import get_buying_power

        mock_api_get.return_value = {}
        assert get_buying_power(self._ACCT) is None

    @patch("portfolio.avanza_session.api_get")
    def test_unwrapped_balance_value(self, mock_api_get, session_file):
        """If Avanza returns raw number instead of {value: N}, unwrap still works."""
        from portfolio.avanza_session import get_buying_power

        mock_api_get.return_value = {
            "accounts": [
                {
                    "id": self._ACCT,
                    "buyingPower": 12345.0,  # raw number, not wrapped
                    "totalValue": 54321.0,
                    "ownCapital": 42000.0,
                },
            ],
        }
        result = get_buying_power(self._ACCT)
        assert result is not None
        assert result["buying_power"] == 12345.0


class TestApiDelete:
    @patch("portfolio.avanza_session._get_csrf", return_value="csrf-token")
    @patch("portfolio.avanza_session._get_playwright_context")
    def test_successful_delete(self, mock_get_ctx, mock_csrf, session_file):
        from portfolio.avanza_session import api_delete

        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.ok = True
        mock_resp.text.return_value = ""
        mock_ctx.request.delete.return_value = mock_resp
        mock_get_ctx.return_value = mock_ctx

        result = api_delete("/_api/trading/stoploss/1625505/SL-1")
        assert result["http_status"] == 200
        assert result["ok"] is True

    @patch("portfolio.avanza_session._get_csrf", return_value="csrf-token")
    @patch("portfolio.avanza_session._get_playwright_context")
    def test_401_raises(self, mock_get_ctx, mock_csrf, session_file):
        from portfolio.avanza_session import api_delete

        mock_ctx = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status = 401
        mock_ctx.request.delete.return_value = mock_resp
        mock_get_ctx.return_value = mock_ctx

        with pytest.raises(AvanzaSessionError, match="401"):
            api_delete("/_api/test")


class TestPlaywrightLockSerialization:
    """A-AV-1 (2026-04-11): api_get/api_post/api_delete must serialize all
    Playwright access through _pw_lock. Without it, the metals 10s fast-tick
    thread + main loop's 8-worker pool race on ctx.request.* and corrupt
    trade responses.

    These tests don't try to reproduce the race directly (it would be flaky)
    — instead they assert the *invariant* that the lock is held during the
    critical section by counting concurrent overlap. With the lock, max
    concurrency observed must be 1.
    """

    def test_lock_is_reentrant_rlock(self):
        """The lock must be an RLock so api_get can acquire it and then
        call _get_playwright_context which also acquires it."""
        import threading
        from portfolio.avanza_session import _pw_lock
        # threading.RLock is a factory, not a class — check via behavior:
        # RLock allows the same thread to acquire twice without blocking.
        acquired_twice = [False]
        def double_acquire():
            with _pw_lock:
                with _pw_lock:
                    acquired_twice[0] = True
        t = threading.Thread(target=double_acquire)
        t.start()
        t.join(timeout=1.0)
        assert acquired_twice[0], "Lock is not reentrant — would deadlock api_get"

    @patch("portfolio.avanza_session._get_playwright_context")
    def test_concurrent_api_get_serialized(self, mock_get_ctx, session_file):
        """Run 10 api_get calls concurrently and verify they never overlap."""
        import threading
        import time
        from portfolio.avanza_session import api_get

        active = [0]
        max_active = [0]
        active_lock = threading.Lock()

        def fake_get(url):
            # Simulate Playwright doing internal state mutation
            with active_lock:
                active[0] += 1
                if active[0] > max_active[0]:
                    max_active[0] = active[0]
            time.sleep(0.01)  # hold the "request" briefly
            with active_lock:
                active[0] -= 1
            resp = MagicMock()
            resp.status = 200
            resp.ok = True
            resp.json.return_value = {"ok": True}
            return resp

        mock_ctx = MagicMock()
        mock_ctx.request.get.side_effect = fake_get
        mock_get_ctx.return_value = mock_ctx

        threads = [threading.Thread(target=api_get, args=("/_api/test",)) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        # If _pw_lock serializes correctly, max concurrent ctx.request.get
        # calls should be exactly 1. Without the lock it would be ~10.
        assert max_active[0] == 1, (
            f"Expected serialized access (max=1), got max={max_active[0]} "
            "concurrent ctx.request.get calls — _pw_lock not held during request"
        )

    @patch("portfolio.avanza_session._get_csrf", return_value="csrf-token")
    @patch("portfolio.avanza_session._get_playwright_context")
    def test_concurrent_api_post_serialized(self, mock_get_ctx, mock_csrf, session_file):
        """Same invariant for api_post."""
        import threading
        import time
        from portfolio.avanza_session import api_post

        active = [0]
        max_active = [0]
        active_lock = threading.Lock()

        def fake_post(url, **kwargs):
            with active_lock:
                active[0] += 1
                if active[0] > max_active[0]:
                    max_active[0] = active[0]
            time.sleep(0.01)
            with active_lock:
                active[0] -= 1
            resp = MagicMock()
            resp.status = 200
            resp.ok = True
            resp.text.return_value = '{"ok": true}'
            return resp

        mock_ctx = MagicMock()
        mock_ctx.request.post.side_effect = fake_post
        mock_get_ctx.return_value = mock_ctx

        threads = [threading.Thread(target=api_post, args=("/_api/test", {"a": 1})) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert max_active[0] == 1, (
            f"api_post not serialized: max concurrent={max_active[0]}"
        )

    @patch("portfolio.avanza_session._get_csrf", return_value="csrf-token")
    @patch("portfolio.avanza_session._get_playwright_context")
    def test_concurrent_mixed_get_post_delete_serialized(self, mock_get_ctx, mock_csrf, session_file):
        """Mixed api_get/api_post/api_delete must all serialize through the
        same lock — that's the actual race condition in production
        (one ticker reads positions while another places an order)."""
        import threading
        import time
        from portfolio.avanza_session import api_get, api_post, api_delete

        active = [0]
        max_active = [0]
        active_lock = threading.Lock()

        def make_fake(method_name):
            def fake(url, **kwargs):
                with active_lock:
                    active[0] += 1
                    if active[0] > max_active[0]:
                        max_active[0] = active[0]
                time.sleep(0.005)
                with active_lock:
                    active[0] -= 1
                resp = MagicMock()
                resp.status = 200
                resp.ok = True
                resp.json.return_value = {}
                resp.text.return_value = "{}"
                return resp
            return fake

        mock_ctx = MagicMock()
        mock_ctx.request.get.side_effect = make_fake("get")
        mock_ctx.request.post.side_effect = make_fake("post")
        mock_ctx.request.delete.side_effect = make_fake("delete")
        mock_get_ctx.return_value = mock_ctx

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=api_get, args=("/_api/positions",)))
            threads.append(threading.Thread(target=api_post, args=("/_api/order/new", {"x": 1})))
            threads.append(threading.Thread(target=api_delete, args=("/_api/order/123",)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert max_active[0] == 1, (
            f"Mixed api_*/api_*/api_* not serialized: max concurrent={max_active[0]}. "
            "All three methods must share _pw_lock or trades will corrupt."
        )
