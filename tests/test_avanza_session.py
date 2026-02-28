"""Tests for portfolio.avanza_session — session load, save, expiry, API wrapper."""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza_session import (
    API_BASE,
    EXPIRY_BUFFER_MINUTES,
    AvanzaSessionError,
    is_session_expiring_soon,
    load_session,
    session_remaining_minutes,
    verify_session,
    api_get,
    get_positions,
    get_instrument_price,
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
    now = datetime.now(timezone.utc)
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
