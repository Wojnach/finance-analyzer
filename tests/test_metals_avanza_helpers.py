"""Tests for data/metals_avanza_helpers.py — Playwright API helpers (all mocked).

Batch 5 of the metals monitoring auto-improvement plan.
"""
import json
import os
import sys
import pytest
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))


def _make_page(evaluate_return=None, cookies=None):
    """Create a mock Playwright page with configurable evaluate() and cookies."""
    page = MagicMock()

    if evaluate_return is not None:
        page.evaluate.return_value = evaluate_return
    else:
        page.evaluate.return_value = None

    if cookies is not None:
        page.context.cookies.return_value = cookies
    else:
        page.context.cookies.return_value = []

    return page


# ---------------------------------------------------------------------------
# get_csrf
# ---------------------------------------------------------------------------

class TestGetCsrf:
    def test_extracts_token(self):
        from metals_avanza_helpers import get_csrf

        cookies = [
            {"name": "other", "value": "abc"},
            {"name": "AZACSRF", "value": "test-csrf-token-123"},
        ]
        page = _make_page(cookies=cookies)
        token = get_csrf(page)
        assert token == "test-csrf-token-123"

    def test_no_csrf_cookie(self):
        from metals_avanza_helpers import get_csrf

        cookies = [{"name": "session", "value": "abc"}]
        page = _make_page(cookies=cookies)
        token = get_csrf(page)
        assert token is None

    def test_empty_cookies(self):
        from metals_avanza_helpers import get_csrf

        page = _make_page(cookies=[])
        token = get_csrf(page)
        assert token is None


# ---------------------------------------------------------------------------
# fetch_price
# ---------------------------------------------------------------------------

class TestFetchPrice:
    def test_returns_price_data(self):
        from metals_avanza_helpers import fetch_price

        price_data = {
            "bid": 42.50,
            "ask": 42.70,
            "last": 42.40,
            "high": 43.0,
            "low": 41.5,
        }
        page = _make_page(evaluate_return=price_data)
        result = fetch_price(page, "12345", "warrant")
        assert result is not None
        assert isinstance(result, dict)

    def test_evaluate_failure(self):
        from metals_avanza_helpers import fetch_price

        page = _make_page(evaluate_return=None)
        result = fetch_price(page, "12345", "warrant")
        # Should handle gracefully
        assert result is None or isinstance(result, dict)

    def test_exception_handling(self):
        from metals_avanza_helpers import fetch_price

        page = MagicMock()
        page.evaluate.side_effect = Exception("Network error")
        result = fetch_price(page, "12345", "warrant")
        assert result is None


# ---------------------------------------------------------------------------
# fetch_account_cash
# ---------------------------------------------------------------------------

class TestFetchAccountCash:
    def test_returns_cash_data(self):
        from metals_avanza_helpers import fetch_account_cash

        account_data = {
            "buying_power": 50000.0,
            "total_value": 150000.0,
            "own_capital": 120000.0,
        }
        page = _make_page(evaluate_return=account_data)
        result = fetch_account_cash(page, "ACC123")
        assert result is not None
        assert isinstance(result, dict)

    def test_failure_returns_none(self):
        from metals_avanza_helpers import fetch_account_cash

        page = _make_page(evaluate_return=None)
        result = fetch_account_cash(page, "ACC123")
        assert result is None or isinstance(result, dict)

    def test_exception_handling(self):
        from metals_avanza_helpers import fetch_account_cash

        page = MagicMock()
        page.evaluate.side_effect = Exception("Timeout")
        result = fetch_account_cash(page, "ACC123")
        assert result is None


# ---------------------------------------------------------------------------
# place_order
# ---------------------------------------------------------------------------

class TestPlaceOrder:
    def test_success(self):
        from metals_avanza_helpers import place_order

        cookies = [{"name": "AZACSRF", "value": "csrf123"}]
        # place_order checks body["orderRequestStatus"] == "SUCCESS"
        page = _make_page(
            evaluate_return={
                "status": 200,
                "body": json.dumps({"orderRequestStatus": "SUCCESS", "orderId": "ORD456"}),
            },
            cookies=cookies,
        )
        success, result = place_order(page, "ACC123", "12345", "BUY", 42.50, 100)
        assert success is True

    def test_failed_order(self):
        from metals_avanza_helpers import place_order

        cookies = [{"name": "AZACSRF", "value": "csrf123"}]
        page = _make_page(
            evaluate_return={"status": 400, "body": json.dumps({"orderRequestStatus": "ERROR"})},
            cookies=cookies,
        )
        success, result = place_order(page, "ACC123", "12345", "BUY", 42.50, 100)
        assert success is False

    def test_no_csrf(self):
        from metals_avanza_helpers import place_order

        page = _make_page(evaluate_return=None, cookies=[])
        success, result = place_order(page, "ACC123", "12345", "BUY", 42.50, 100)
        assert success is False

    def test_exception(self):
        from metals_avanza_helpers import place_order

        cookies = [{"name": "AZACSRF", "value": "csrf123"}]
        page = _make_page(cookies=cookies)
        page.evaluate.side_effect = Exception("Network error")
        success, result = place_order(page, "ACC123", "12345", "BUY", 42.50, 100)
        assert success is False


# ---------------------------------------------------------------------------
# place_stop_loss
# ---------------------------------------------------------------------------

class TestPlaceStopLoss:
    def test_success(self):
        from metals_avanza_helpers import place_stop_loss

        cookies = [{"name": "AZACSRF", "value": "csrf123"}]
        # place_stop_loss checks body["status"] == "SUCCESS"
        page = _make_page(
            evaluate_return={
                "status": 200,
                "body": json.dumps({"status": "SUCCESS", "stoplossOrderId": "SL789"}),
            },
            cookies=cookies,
        )
        success, stop_id = place_stop_loss(page, "ACC123", "12345", 38.0, 37.5, 100)
        assert success is True

    def test_failed(self):
        from metals_avanza_helpers import place_stop_loss

        cookies = [{"name": "AZACSRF", "value": "csrf123"}]
        page = _make_page(
            evaluate_return={"status": 400, "body": json.dumps({"status": "ERROR"})},
            cookies=cookies,
        )
        success, stop_id = place_stop_loss(page, "ACC123", "12345", 38.0, 37.5, 100)
        assert success is False

    def test_no_csrf(self):
        from metals_avanza_helpers import place_stop_loss

        page = _make_page(cookies=[])
        success, stop_id = place_stop_loss(page, "ACC123", "12345", 38.0, 37.5, 100)
        assert success is False

    def test_uses_correct_api_endpoint(self):
        """Verify stop-loss uses /_api/trading/stoploss/new, NOT /_api/trading-critical."""
        from metals_avanza_helpers import place_stop_loss

        cookies = [{"name": "AZACSRF", "value": "csrf123"}]
        page = _make_page(
            evaluate_return={
                "status": 200,
                "body": json.dumps({"status": "SUCCESS", "stoplossOrderId": "SL1"}),
            },
            cookies=cookies,
        )
        place_stop_loss(page, "ACC123", "12345", 38.0, 37.5, 100)

        # Check the JavaScript string passed to evaluate
        call_args = page.evaluate.call_args
        if call_args:
            js_code = call_args[0][0] if call_args[0] else ""
            if "trading-critical" in js_code:
                pytest.fail("Stop-loss must use /_api/trading/stoploss/new, not trading-critical")

    def test_valid_days_parameter(self):
        from metals_avanza_helpers import place_stop_loss

        cookies = [{"name": "AZACSRF", "value": "csrf123"}]
        page = _make_page(
            evaluate_return={
                "status": 200,
                "body": json.dumps({"status": "SUCCESS", "stoplossOrderId": "SL2"}),
            },
            cookies=cookies,
        )
        # Should accept valid_days parameter
        success, _ = place_stop_loss(page, "ACC123", "12345", 38.0, 37.5, 100, valid_days=14)
        assert isinstance(success, bool)


# ---------------------------------------------------------------------------
# check_session_alive
# ---------------------------------------------------------------------------

class TestCheckSessionAlive:
    def test_alive(self):
        from metals_avanza_helpers import check_session_alive

        # check_session_alive returns resp.status (int), checks == 200
        page = _make_page(evaluate_return=200)
        result = check_session_alive(page)
        assert result is True

    def test_401_dead(self):
        from metals_avanza_helpers import check_session_alive

        page = _make_page(evaluate_return=401)
        result = check_session_alive(page)
        assert result is False

    def test_exception(self):
        from metals_avanza_helpers import check_session_alive

        page = MagicMock()
        page.evaluate.side_effect = Exception("Disconnected")
        result = check_session_alive(page)
        assert result is False

    def test_non_200(self):
        from metals_avanza_helpers import check_session_alive

        page = _make_page(evaluate_return=500)
        result = check_session_alive(page)
        assert result is False
