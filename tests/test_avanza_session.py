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
    create_requests_session,
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
    """Create a temp session file and patch SESSION_FILE to point to it."""
    with tempfile.TemporaryDirectory() as td:
        sf = Path(td) / "avanza_session.json"
        with patch("portfolio.avanza_session.SESSION_FILE", sf):
            yield sf


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

    def test_no_cookies_raises(self, session_file):
        data = _make_session_data()
        data["cookies"] = []
        session_file.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(AvanzaSessionError, match="no cookies"):
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


# --- create_requests_session tests ---


class TestCreateRequestsSession:
    def test_creates_session_with_cookies(self, session_file):
        data = _make_session_data()
        session_file.write_text(json.dumps(data), encoding="utf-8")
        s = create_requests_session()
        # Check cookies loaded
        assert s.cookies.get("ava_cookie") == "cookie_value"
        assert s.cookies.get("session_id") == "sess_123"

    def test_sets_security_token_header(self, session_file):
        data = _make_session_data(security_token="my-sec-token")
        session_file.write_text(json.dumps(data), encoding="utf-8")
        s = create_requests_session()
        assert s.headers.get("X-SecurityToken") == "my-sec-token"

    def test_sets_auth_session_header(self, session_file):
        data = _make_session_data(authentication_session="my-auth-sess")
        session_file.write_text(json.dumps(data), encoding="utf-8")
        s = create_requests_session()
        assert s.headers.get("X-AuthenticationSession") == "my-auth-sess"

    def test_no_token_skips_header(self, session_file):
        data = _make_session_data(security_token=None)
        session_file.write_text(json.dumps(data), encoding="utf-8")
        s = create_requests_session()
        assert "X-SecurityToken" not in s.headers

    def test_accepts_explicit_session_data(self, session_file):
        data = _make_session_data(security_token="explicit-token")
        # Don't write to file — pass directly
        s = create_requests_session(session_data=data)
        assert s.headers.get("X-SecurityToken") == "explicit-token"

    def test_has_user_agent(self, session_file):
        data = _make_session_data()
        session_file.write_text(json.dumps(data), encoding="utf-8")
        s = create_requests_session()
        assert "Mozilla" in s.headers.get("User-Agent", "")

    def test_has_json_accept(self, session_file):
        data = _make_session_data()
        session_file.write_text(json.dumps(data), encoding="utf-8")
        s = create_requests_session()
        assert s.headers.get("Accept") == "application/json"


# --- verify_session tests ---


class TestVerifySession:
    @patch("portfolio.avanza_session.create_requests_session")
    def test_valid_session_returns_true(self, mock_create, session_file):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session.get.return_value = mock_resp
        mock_create.return_value = mock_session
        assert verify_session() is True
        mock_session.get.assert_called_once()

    @patch("portfolio.avanza_session.create_requests_session")
    def test_401_returns_false(self, mock_create, session_file):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_session.get.return_value = mock_resp
        mock_create.return_value = mock_session
        assert verify_session() is False

    def test_accepts_existing_session(self, session_file):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_session.get.return_value = mock_resp
        assert verify_session(session=mock_session) is True

    @patch("portfolio.avanza_session.create_requests_session")
    def test_network_error_returns_false(self, mock_create, session_file):
        mock_session = MagicMock()
        mock_session.get.side_effect = ConnectionError("Network down")
        mock_create.return_value = mock_session
        assert verify_session() is False


# --- api_get tests ---


class TestApiGet:
    def test_successful_get(self, session_file):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": "test"}
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        result = api_get("/_api/test", session=mock_session)
        assert result == {"data": "test"}
        mock_session.get.assert_called_once_with(
            f"{API_BASE}/_api/test", timeout=15
        )

    def test_401_raises_session_error(self, session_file):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_session.get.return_value = mock_resp

        with pytest.raises(AvanzaSessionError, match="401"):
            api_get("/_api/test", session=mock_session)

    def test_full_url_not_prepended(self, session_file):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        api_get("https://custom.url/api/test", session=mock_session)
        mock_session.get.assert_called_once_with(
            "https://custom.url/api/test", timeout=15
        )

    def test_custom_timeout(self, session_file):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        api_get("/_api/test", session=mock_session, timeout=30)
        mock_session.get.assert_called_once_with(
            f"{API_BASE}/_api/test", timeout=30
        )


# --- get_positions tests ---


class TestGetPositions:
    def test_parses_positions(self, session_file):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "withOrderbook": [
                {
                    "instruments": [
                        {
                            "name": "SAAB B",
                            "orderbookId": "5533",
                            "volume": 100,
                            "value": 73500.0,
                            "profit": 5000.0,
                            "profitPercent": 7.3,
                            "currency": "SEK",
                            "lastPrice": 735.0,
                            "changePercent": 1.2,
                        }
                    ]
                }
            ]
        }
        mock_session.get.return_value = mock_resp

        positions = get_positions(session=mock_session)
        assert len(positions) == 1
        assert positions[0]["name"] == "SAAB B"
        assert positions[0]["orderbook_id"] == "5533"
        assert positions[0]["volume"] == 100
        assert positions[0]["value"] == 73500.0

    def test_empty_positions(self, session_file):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"withOrderbook": []}
        mock_session.get.return_value = mock_resp

        positions = get_positions(session=mock_session)
        assert positions == []


# --- get_instrument_price tests ---


class TestGetInstrumentPrice:
    def test_stock_lookup(self, session_file):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"lastPrice": 735.0, "changePercent": 1.2}
        mock_session.get.return_value = mock_resp

        result = get_instrument_price("5533", session=mock_session)
        assert result["lastPrice"] == 735.0

    def test_fallback_to_certificate(self, session_file):
        mock_session = MagicMock()
        call_count = 0

        def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            if "stock" in url:
                resp.status_code = 404
                resp.raise_for_status.side_effect = Exception("Not found")
                resp.json.side_effect = Exception("Not found")
                raise Exception("Not found")
            else:
                resp.status_code = 200
                resp.raise_for_status = MagicMock()
                resp.json.return_value = {"lastPrice": 42.0}
                return resp

        mock_session.get = mock_get

        result = get_instrument_price("9999", session=mock_session)
        assert result["lastPrice"] == 42.0
        assert call_count == 2  # stock failed, certificate succeeded


# --- Login script helpers ---


class TestLoginHelpers:
    def test_is_logged_in_login_page(self):
        from scripts.avanza_login import _is_logged_in
        assert _is_logged_in("https://www.avanza.se/logga-in") is False

    def test_is_logged_in_start_page(self):
        from scripts.avanza_login import _is_logged_in
        assert _is_logged_in("https://www.avanza.se/start") is True

    def test_is_logged_in_mina_sidor(self):
        from scripts.avanza_login import _is_logged_in
        assert _is_logged_in("https://www.avanza.se/mina-sidor/overview") is True

    def test_is_logged_in_root(self):
        from scripts.avanza_login import _is_logged_in
        assert _is_logged_in("https://www.avanza.se/") is True

    def test_is_logged_in_hem(self):
        from scripts.avanza_login import _is_logged_in
        assert _is_logged_in("https://www.avanza.se/hem/dashboard") is True
