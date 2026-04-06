"""Tests for portfolio.avanza_control."""

from unittest.mock import MagicMock, patch

import pytest

import portfolio.avanza_control as mod


class TestNormalizeApiType:
    def test_aliases_certificate(self):
        assert mod.normalize_api_type("cert") == "certificate"

    def test_aliases_etf(self):
        assert mod.normalize_api_type("etf") == "exchange_traded_fund"

    def test_empty_uses_default(self):
        assert mod.normalize_api_type("") == "certificate"


class TestFetchPriceWithFallback:
    @patch("portfolio.avanza_control.fetch_price")
    def test_returns_first_viable_quote_and_tags_type(self, mock_fetch):
        mock_fetch.side_effect = [
            None,
            {"bid": 9.48, "ask": 9.52, "last": 9.14},
        ]

        result = mod.fetch_price_with_fallback(object(), "2308943", api_type="warrant")

        assert result["api_type"] == "certificate"
        assert result["bid"] == 9.48
        assert [call.args[2] for call in mock_fetch.call_args_list] == ["warrant", "certificate"]


class TestPlaceOrder:
    @patch("portfolio.avanza_control._place_page_order", return_value=(True, {"order_id": "123"}))
    @patch("portfolio.avanza_control.get_account_id", return_value="1625505")
    def test_resolves_missing_account_id(self, mock_account_id, mock_place):
        page = object()
        success, result = mod.place_order(page, None, "856394", "buy", 900.0, 4)

        assert success is True
        assert result["order_id"] == "123"
        mock_place.assert_called_once_with(page, "1625505", "856394", "BUY", 900.0, 4)


class TestDeleteStopLoss:
    @patch("portfolio.avanza_control.get_csrf", return_value="csrf-token")
    def test_parses_success_response(self, mock_csrf):
        page = MagicMock()
        page.evaluate.return_value = {"status": 200, "body": "{\"status\":\"SUCCESS\"}"}

        success, result = mod.delete_stop_loss(page, "1625505", "SL123")

        assert success is True
        assert result["http_status"] == 200
        assert result["parsed"]["status"] == "SUCCESS"


class TestDeleteOrderLive:
    @patch("portfolio.avanza_control.get_csrf", return_value="csrf-token")
    @patch("portfolio.avanza_control.get_account_id", return_value="1625505")
    def test_parses_success_response(self, mock_account_id, mock_csrf):
        page = MagicMock()
        page.evaluate.return_value = {"status": 200, "body": '{"orderRequestStatus":"SUCCESS"}'}

        success, result = mod.delete_order_live(page, None, "12345")

        assert success is True
        assert result["http_status"] == 200


# --- Page-free function tests ---


class TestFetchPriceNoPage:
    @patch("portfolio.avanza_control._api_get")
    def test_returns_parsed_quote(self, mock_get):
        mock_get.return_value = {
            "quote": {
                "buy": {"value": 24.50},
                "sell": {"value": 24.55},
                "last": {"value": 24.52},
                "changePercent": {"value": 1.5},
                "highest": {"value": 25.0},
                "lowest": {"value": 24.0},
            },
            "keyIndicators": {"leverage": {"value": 5.0}, "barrierLevel": {"value": 35.0}},
            "underlying": {"name": "Silver", "quote": {"last": {"value": 33.5}}},
        }

        result = mod.fetch_price_no_page("856394", api_type="certificate")
        assert result is not None
        assert result["bid"] == 24.50
        assert result["ask"] == 24.55
        assert result["leverage"] == 5.0
        assert result["barrier"] == 35.0
        assert result["underlying"] == 33.5

    @patch("portfolio.avanza_control._api_get")
    def test_returns_none_on_error(self, mock_get):
        mock_get.side_effect = RuntimeError("API error")
        result = mod.fetch_price_no_page("000000")
        assert result is None


class TestPlaceOrderNoPage:
    @patch("portfolio.avanza_control._place_buy_order")
    def test_buy_order(self, mock_buy):
        mock_buy.return_value = {"orderRequestStatus": "SUCCESS", "orderId": "123"}
        ok, result = mod.place_order_no_page("1625505", "856394", "BUY", 24.50, 100)
        assert ok is True
        assert result["orderRequestStatus"] == "SUCCESS"
        mock_buy.assert_called_once_with("856394", 24.50, 100, "1625505")

    @patch("portfolio.avanza_control._place_sell_order")
    def test_sell_order(self, mock_sell):
        mock_sell.return_value = {"orderRequestStatus": "SUCCESS", "orderId": "456"}
        ok, result = mod.place_order_no_page("1625505", "856394", "SELL", 25.00, 50)
        assert ok is True
        assert result["orderRequestStatus"] == "SUCCESS"
        mock_sell.assert_called_once_with("856394", 25.00, 50, "1625505")

    @patch("portfolio.avanza_control._place_buy_order")
    def test_failed_order(self, mock_buy):
        mock_buy.return_value = {"orderRequestStatus": "ERROR", "message": "Insufficient funds"}
        ok, result = mod.place_order_no_page("1625505", "856394", "BUY", 24.50, 100)
        assert ok is False
        assert result["message"] == "Insufficient funds"

    def test_invalid_side_raises(self):
        """C2: Non-BUY/SELL side must raise ValueError, not fall through to SELL."""
        with pytest.raises(ValueError, match="Invalid order side"):
            mod.place_order_no_page("1625505", "856394", "HOLD", 24.50, 100)

    def test_none_side_raises(self):
        """C2: None side must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid order side"):
            mod.place_order_no_page("1625505", "856394", None, 24.50, 100)

    def test_empty_side_raises(self):
        """C2: Empty string side must raise ValueError."""
        with pytest.raises(ValueError, match="Invalid order side"):
            mod.place_order_no_page("1625505", "856394", "", 24.50, 100)

    @patch("portfolio.avanza_control._place_sell_order")
    def test_lowercase_sell_works(self, mock_sell):
        """C2: Lowercase 'sell' should still work (normalized to uppercase)."""
        mock_sell.return_value = {"orderRequestStatus": "SUCCESS"}
        ok, _ = mod.place_order_no_page("1625505", "856394", "sell", 25.00, 50)
        assert ok is True


class TestPlaceStopLossNoPage:
    @patch("portfolio.avanza_control._place_stop_loss_session")
    def test_delegates_to_session(self, mock_stop):
        mock_stop.return_value = {"status": "SUCCESS", "stoplossOrderId": "SL-1"}
        ok, result = mod.place_stop_loss_no_page("1625505", "856394", 23.0, 22.5, 100)
        assert ok is True
        assert result["stoplossOrderId"] == "SL-1"
        mock_stop.assert_called_once_with("856394", 23.0, 22.5, 100, "1625505", 8)


class TestPlaceTrailingStopNoPage:
    @patch("portfolio.avanza_control._place_trailing_stop_session")
    def test_delegates_to_session(self, mock_trail):
        mock_trail.return_value = {"status": "SUCCESS", "stoplossOrderId": "SL-TRAIL"}
        ok, result = mod.place_trailing_stop_no_page("1625505", "856394", 5.0, 100)
        assert ok is True
        assert result["stoplossOrderId"] == "SL-TRAIL"
        mock_trail.assert_called_once_with("856394", 5.0, 100, "1625505", 8)


class TestDeleteOrderNoPage:
    @patch("portfolio.avanza_control._cancel_order")
    def test_delegates_to_session(self, mock_cancel):
        mock_cancel.return_value = {"orderRequestStatus": "SUCCESS"}
        ok, result = mod.delete_order_no_page("1625505", "12345")
        assert ok is True
        mock_cancel.assert_called_once_with("12345", "1625505")

    @patch("portfolio.avanza_control._cancel_order")
    def test_failed_cancel(self, mock_cancel):
        mock_cancel.return_value = {"orderRequestStatus": "ERROR"}
        ok, result = mod.delete_order_no_page("1625505", "12345")
        assert ok is False


class TestDeleteStopLossNoPage:
    @patch("portfolio.avanza_control._api_delete")
    def test_delegates_to_session(self, mock_delete):
        mock_delete.return_value = {}
        ok, result = mod.delete_stop_loss_no_page("1625505", "SL-123")
        assert ok is True
        mock_delete.assert_called_once_with("/_api/trading/stoploss/1625505/SL-123")

    @patch("portfolio.avanza_control._api_delete")
    def test_error_returns_false(self, mock_delete):
        mock_delete.side_effect = RuntimeError("API error")
        ok, result = mod.delete_stop_loss_no_page("1625505", "SL-123")
        assert ok is False
        assert "error" in result

    @patch("portfolio.avanza_control._api_delete")
    def test_api_error_code_returns_false(self, mock_delete):
        """H18: API returning errorCode should report failure, not success."""
        mock_delete.return_value = {"errorCode": "NOT_FOUND", "message": "Stop not found"}
        ok, result = mod.delete_stop_loss_no_page("1625505", "SL-123")
        assert ok is False
        assert result.get("errorCode") == "NOT_FOUND"
