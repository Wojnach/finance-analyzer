"""Tests for portfolio.avanza.scanner — instrument search + ranking."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.scanner import format_scan_results, scan_instruments


@pytest.fixture(autouse=True)
def reset_singletons():
    AvanzaAuth._instance = None
    AvanzaAuth._lock = threading.Lock()
    AvanzaClient._instance = None
    AvanzaClient._lock = threading.Lock()
    yield
    AvanzaAuth._instance = None
    AvanzaClient._instance = None


def _setup():
    mock_avanza = MagicMock()
    mock_avanza._push_subscription_id = "push"
    mock_avanza._security_token = "csrf"
    mock_avanza._authentication_session = "auth"
    mock_avanza._customer_id = "cust"
    return mock_avanza


SEARCH_HITS = {
    "hits": [
        {"orderBookId": "111", "title": "BULL OLJA X5 AVA 3", "type": "CERTIFICATE", "tradeable": True},
        {"orderBookId": "222", "title": "BULL OLJA X10 AVA 7", "type": "CERTIFICATE", "tradeable": True},
        {"orderBookId": "333", "title": "BEAR OLJA X5 AVA 2", "type": "CERTIFICATE", "tradeable": True},
    ],
}

INSTRUMENT_DATA = {
    "111": {
        "name": "BULL OLJA X5 AVA 3",
        "instrumentType": "CERTIFICATE",
        "currency": "SEK",
        "quote": {
            "buy": {"value": 50.0}, "sell": {"value": 50.30}, "last": {"value": 50.15},
            "totalVolumeTraded": 1200, "totalValueTraded": 60000,
        },
        "keyIndicators": {"leverage": {"value": 5.0}, "barrierLevel": {"value": 35.0}},
        "underlying": {"name": "Brent Crude", "quote": {"last": {"value": 73.5}}},
    },
    "222": {
        "name": "BULL OLJA X10 AVA 7",
        "instrumentType": "CERTIFICATE",
        "currency": "SEK",
        "quote": {
            "buy": {"value": 12.0}, "sell": {"value": 12.20}, "last": {"value": 12.10},
            "totalVolumeTraded": 500, "totalValueTraded": 6000,
        },
        "keyIndicators": {"leverage": {"value": 10.0}, "barrierLevel": {"value": 66.0}},
        "underlying": {"name": "Brent Crude", "quote": {"last": {"value": 73.5}}},
    },
    "333": {
        "name": "BEAR OLJA X5 AVA 2",
        "instrumentType": "CERTIFICATE",
        "currency": "SEK",
        "quote": {
            "buy": {"value": 30.0}, "sell": {"value": 30.60}, "last": {"value": 30.30},
            "totalVolumeTraded": 800, "totalValueTraded": 24000,
        },
        "keyIndicators": {"leverage": {"value": 5.0}, "barrierLevel": {"value": 95.0}},
        "underlying": {"name": "Brent Crude", "quote": {"last": {"value": 73.5}}},
    },
}


def _mock_get_instrument(itype, ob_id):
    return INSTRUMENT_DATA.get(ob_id, {})


def _mock_get_market_data(ob_id):
    return {
        "orderDepth": {
            "levels": [
                {"buySide": {"price": 50.0, "volume": 10000}, "sellSide": {"price": 50.3, "volume": 10000}},
            ],
        },
        "marketMakerExpected": True,
    }


class TestScanInstruments:
    def test_finds_bull_instruments(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.search_for_instrument.return_value = SEARCH_HITS
            mock_avanza.get_instrument.side_effect = _mock_get_instrument
            mock_avanza.get_market_data.side_effect = _mock_get_market_data

            results = scan_instruments("OLJA", direction="BULL", instrument_type="certificate")
            assert len(results) == 2  # Only BULL, not BEAR
            assert all("BULL" in r.name for r in results)

    def test_sorts_by_spread(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.search_for_instrument.return_value = SEARCH_HITS
            mock_avanza.get_instrument.side_effect = _mock_get_instrument
            mock_avanza.get_market_data.side_effect = _mock_get_market_data

            results = scan_instruments("OLJA", sort_by="spread", instrument_type="certificate")
            assert len(results) >= 2
            # X5 has spread 0.6% (0.30/50), X10 has ~1.67% (0.20/12) — X5 should be first
            spreads = [r.spread_pct for r in results if r.spread_pct is not None]
            assert spreads == sorted(spreads)

    def test_sorts_by_leverage(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.search_for_instrument.return_value = SEARCH_HITS
            mock_avanza.get_instrument.side_effect = _mock_get_instrument
            mock_avanza.get_market_data.side_effect = _mock_get_market_data

            results = scan_instruments("OLJA", sort_by="leverage", instrument_type="certificate")
            leverages = [r.leverage for r in results if r.leverage]
            # Sorted descending (highest leverage first)
            assert leverages == sorted(leverages, reverse=True)

    def test_min_leverage_filter(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.search_for_instrument.return_value = SEARCH_HITS
            mock_avanza.get_instrument.side_effect = _mock_get_instrument
            mock_avanza.get_market_data.side_effect = _mock_get_market_data

            results = scan_instruments("OLJA", min_leverage=8, instrument_type="certificate")
            assert len(results) == 1
            assert results[0].leverage == 10.0

    def test_extracts_details(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.search_for_instrument.return_value = SEARCH_HITS
            mock_avanza.get_instrument.side_effect = _mock_get_instrument
            mock_avanza.get_market_data.side_effect = _mock_get_market_data

            results = scan_instruments("OLJA", instrument_type="certificate")
            r = next(r for r in results if r.orderbook_id == "111")
            assert r.leverage == 5.0
            assert r.barrier == 35.0
            assert r.underlying_name == "Brent Crude"
            assert r.bid == 50.0
            assert r.ask == 50.30
            assert r.barrier_distance_pct is not None


class TestFormatScanResults:
    def test_empty(self):
        assert format_scan_results([]) == "No instruments found."

    def test_formats_table(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.search_for_instrument.return_value = SEARCH_HITS
            mock_avanza.get_instrument.side_effect = _mock_get_instrument
            mock_avanza.get_market_data.side_effect = _mock_get_market_data

            results = scan_instruments("OLJA", instrument_type="certificate")
            table = format_scan_results(results)
            assert "BULL OLJA X5" in table
            assert "Lev" in table  # Header
            assert "Spread" in table
