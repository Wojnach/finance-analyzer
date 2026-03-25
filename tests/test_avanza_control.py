"""Tests for portfolio.avanza_control."""

from unittest.mock import MagicMock, patch

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
        page.evaluate.return_value = {"status": 200, "body": "{\"status\":\"SUCCESS\"}"}

        success, result = mod.delete_order_live(page, None, "12345")

        assert success is True
        assert result["http_status"] == 200
        assert result["parsed"]["status"] == "SUCCESS"
