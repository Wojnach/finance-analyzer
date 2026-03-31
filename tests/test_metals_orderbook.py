"""Tests for metals order book + trades fetcher."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

MOCK_DEPTH_RESPONSE = {
    "lastUpdateId": 123456789,
    "E": 1711900000000,
    "T": 1711900000000,
    "bids": [
        ["3100.50", "2.5"],
        ["3100.00", "1.8"],
        ["3099.50", "3.2"],
    ],
    "asks": [
        ["3101.00", "1.2"],
        ["3101.50", "2.0"],
        ["3102.00", "4.1"],
    ],
}

MOCK_TRADES_RESPONSE = [
    {"id": 1, "price": "3100.80", "qty": "0.5", "quoteQty": "1550.40",
     "time": 1711900001000, "isBuyerMaker": False},
    {"id": 2, "price": "3100.50", "qty": "1.0", "quoteQty": "3100.50",
     "time": 1711900002000, "isBuyerMaker": True},
    {"id": 3, "price": "3101.00", "qty": "0.3", "quoteQty": "930.30",
     "time": 1711900003000, "isBuyerMaker": False},
    {"id": 4, "price": "3100.20", "qty": "0.8", "quoteQty": "2480.16",
     "time": 1711900004000, "isBuyerMaker": True},
]


class TestGetOrderbookDepth:
    @patch("portfolio.metals_orderbook._fetch_fapi_json")
    def test_returns_parsed_depth(self, mock_fetch):
        from portfolio.metals_orderbook import get_orderbook_depth
        mock_fetch.return_value = MOCK_DEPTH_RESPONSE

        result = get_orderbook_depth.__wrapped__("XAU-USD", limit=20)

        assert result is not None
        assert len(result["bids"]) == 3
        assert len(result["asks"]) == 3
        assert result["bids"][0] == [3100.50, 2.5]
        assert result["asks"][0] == [3101.00, 1.2]
        assert result["best_bid"] == 3100.50
        assert result["best_ask"] == 3101.00
        assert result["mid_price"] == pytest.approx(3100.75)
        assert result["spread"] == pytest.approx(0.50)
        assert result["spread_bps"] == pytest.approx(0.50 / 3100.75 * 10000, rel=1e-3)

    @patch("portfolio.metals_orderbook._fetch_fapi_json")
    def test_returns_none_for_unknown_ticker(self, mock_fetch):
        from portfolio.metals_orderbook import get_orderbook_depth
        result = get_orderbook_depth.__wrapped__("UNKNOWN", limit=20)
        assert result is None
        mock_fetch.assert_not_called()

    @patch("portfolio.metals_orderbook._fetch_fapi_json")
    def test_returns_none_on_api_failure(self, mock_fetch):
        from portfolio.metals_orderbook import get_orderbook_depth
        mock_fetch.return_value = None
        result = get_orderbook_depth.__wrapped__("XAG-USD", limit=20)
        assert result is None


class TestGetRecentTrades:
    @patch("portfolio.metals_orderbook._fetch_fapi_json")
    def test_returns_parsed_trades(self, mock_fetch):
        from portfolio.metals_orderbook import get_recent_trades
        mock_fetch.return_value = MOCK_TRADES_RESPONSE

        result = get_recent_trades.__wrapped__("XAU-USD", limit=50)

        assert result is not None
        assert len(result) == 4
        assert result[0]["price"] == 3100.80
        assert result[0]["qty"] == 0.5
        assert result[0]["is_buyer_maker"] is False
        assert result[1]["is_buyer_maker"] is True

    @patch("portfolio.metals_orderbook._fetch_fapi_json")
    def test_returns_none_for_unknown_ticker(self, mock_fetch):
        from portfolio.metals_orderbook import get_recent_trades
        result = get_recent_trades.__wrapped__("UNKNOWN", limit=50)
        assert result is None
        mock_fetch.assert_not_called()


class TestComputeTradeSign:
    @patch("portfolio.metals_orderbook._fetch_fapi_json")
    def test_trade_sign_from_buyer_maker(self, mock_fetch):
        """isBuyerMaker=True means seller initiated (hit the bid) = -1."""
        from portfolio.metals_orderbook import get_recent_trades
        mock_fetch.return_value = MOCK_TRADES_RESPONSE

        result = get_recent_trades.__wrapped__("XAU-USD", limit=50)

        assert result[0]["sign"] == 1
        assert result[1]["sign"] == -1
