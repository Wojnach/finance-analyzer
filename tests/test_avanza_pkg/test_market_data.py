"""Tests for portfolio.avanza.market_data — quotes, depth, OHLC, info, news."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.market_data import (
    get_instrument_info,
    get_market_data,
    get_news,
    get_ohlc,
    get_quote,
)
from portfolio.avanza.types import InstrumentInfo, MarketData, NewsArticle, OHLC, Quote


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
    """Create a mock mimicking an authenticated avanza.Avanza instance."""
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
        client = AvanzaClient.get_instance(_make_config())
        yield mock_client


# ---------------------------------------------------------------------------
# get_quote
# ---------------------------------------------------------------------------

class TestGetQuote:
    def test_returns_quote(self, mock_avanza):
        mock_avanza.get_instrument.return_value = {
            "buy": 5.80,
            "sell": 5.85,
            "last": 5.82,
            "changePercent": -1.2,
            "highest": 5.90,
            "lowest": 5.75,
            "totalVolumeTraded": 10000,
            "updated": "2026-03-30T14:00:00",
        }
        q = get_quote("2213050")
        assert isinstance(q, Quote)
        assert q.bid == 5.80
        assert q.ask == 5.85
        assert q.last == 5.82
        assert q.change_percent == -1.2
        mock_avanza.get_instrument.assert_called_once_with("certificate", "2213050")

    def test_custom_instrument_type(self, mock_avanza):
        mock_avanza.get_instrument.return_value = {
            "buy": 100.0,
            "sell": 101.0,
            "last": 100.5,
        }
        q = get_quote("12345", instrument_type="stock")
        mock_avanza.get_instrument.assert_called_once_with("stock", "12345")
        assert q.last == 100.5


# ---------------------------------------------------------------------------
# get_market_data
# ---------------------------------------------------------------------------

class TestGetMarketData:
    def test_returns_market_data(self, mock_avanza):
        mock_avanza.get_market_data.return_value = {
            "quote": {
                "buy": 100.0,
                "sell": 101.0,
                "last": 100.5,
                "changePercent": 0.2,
                "highest": 102.0,
                "lowest": 99.0,
                "totalVolumeTraded": 5000,
                "updated": 1609459200000,
            },
            "orderDepth": {
                "levels": [
                    {
                        "buySide": {"price": 100.0, "volume": 200},
                        "sellSide": {"price": 101.0, "volume": 150},
                    },
                ],
                "marketMakerExpected": True,
            },
            "trades": [
                {"price": 100.5, "volume": 50, "buyer": "A", "seller": "B", "dealTime": "10:30"},
            ],
        }
        md = get_market_data("2213050")
        assert isinstance(md, MarketData)
        assert md.quote.bid == 100.0
        assert len(md.bid_levels) == 1
        assert len(md.ask_levels) == 1
        assert len(md.recent_trades) == 1
        assert md.market_maker_expected is True


# ---------------------------------------------------------------------------
# get_ohlc
# ---------------------------------------------------------------------------

class TestGetOhlc:
    def test_returns_list_of_ohlc(self, mock_avanza):
        mock_avanza.get_chart_data.return_value = [
            {"timestamp": 1609459200000, "open": 100, "high": 105, "low": 98, "close": 103, "totalVolumeTraded": 25000},
            {"timestamp": 1609545600000, "open": 103, "high": 107, "low": 102, "close": 106, "totalVolumeTraded": 18000},
        ]
        candles = get_ohlc("2213050", period="ONE_MONTH")
        assert len(candles) == 2
        assert all(isinstance(c, OHLC) for c in candles)
        assert candles[0].open == 100.0
        assert candles[1].close == 106.0

    def test_dict_response_with_ohlc_key(self, mock_avanza):
        mock_avanza.get_chart_data.return_value = {
            "ohlc": [
                {"timestamp": 1609459200000, "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100},
            ]
        }
        candles = get_ohlc("123", period="ONE_WEEK")
        assert len(candles) == 1
        assert candles[0].close == 10.5

    def test_dict_response_with_datapoints_key(self, mock_avanza):
        mock_avanza.get_chart_data.return_value = {
            "dataPoints": [
                {"timestamp": 1609459200000, "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 100},
            ]
        }
        candles = get_ohlc("123", period="THREE_MONTHS")
        assert len(candles) == 1

    def test_custom_resolution(self, mock_avanza):
        mock_avanza.get_chart_data.return_value = []
        candles = get_ohlc("123", period="ONE_MONTH", resolution="WEEK")
        assert candles == []
        # Verify correct enum values were passed
        from avanza.constants import Resolution, TimePeriod
        mock_avanza.get_chart_data.assert_called_once_with(
            "123", TimePeriod.ONE_MONTH, Resolution.WEEK,
        )

    def test_empty_result(self, mock_avanza):
        mock_avanza.get_chart_data.return_value = []
        candles = get_ohlc("123")
        assert candles == []


# ---------------------------------------------------------------------------
# get_instrument_info
# ---------------------------------------------------------------------------

class TestGetInstrumentInfo:
    def test_returns_instrument_info(self, mock_avanza):
        mock_avanza.get_instrument.return_value = {
            "id": "2213050",
            "name": "MINI S SILVER AVA 26",
            "instrumentType": "CERTIFICATE",
            "currency": "SEK",
            "leverage": 5.0,
            "barrier": 25.50,
            "underlyingName": "Silver",
            "underlyingPrice": 30.50,
        }
        info = get_instrument_info("2213050")
        assert isinstance(info, InstrumentInfo)
        assert info.orderbook_id == "2213050"
        assert info.name == "MINI S SILVER AVA 26"
        assert info.leverage == 5.0
        assert info.barrier == 25.50
        assert info.underlying_name == "Silver"
        assert info.underlying_price == 30.50
        mock_avanza.get_instrument.assert_called_once_with("certificate", "2213050")

    def test_custom_instrument_type(self, mock_avanza):
        mock_avanza.get_instrument.return_value = {"id": "999", "name": "Test"}
        get_instrument_info("999", instrument_type="warrant")
        mock_avanza.get_instrument.assert_called_once_with("warrant", "999")


# ---------------------------------------------------------------------------
# get_news
# ---------------------------------------------------------------------------

class TestGetNews:
    def test_returns_articles(self, mock_avanza):
        mock_avanza.get_news.return_value = [
            {"id": "n1", "headline": "Silver surges", "timePublishedMillis": 1609459200000, "newsSource": "Reuters"},
            {"id": "n2", "headline": "Metals update", "timePublishedMillis": 1609545600000, "newsSource": "Bloomberg"},
        ]
        articles = get_news("2213050")
        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert articles[0].headline == "Silver surges"
        assert articles[0].source == "Reuters"

    def test_dict_response_with_articles_key(self, mock_avanza):
        mock_avanza.get_news.return_value = {
            "articles": [
                {"id": "n1", "headline": "Test", "newsSource": "AFP"},
            ]
        }
        articles = get_news("123")
        assert len(articles) == 1
        assert articles[0].headline == "Test"

    def test_empty_news(self, mock_avanza):
        mock_avanza.get_news.return_value = []
        articles = get_news("123")
        assert articles == []
