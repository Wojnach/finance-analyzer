"""Tests for portfolio.avanza.types — response dataclasses and from_api parsing."""

from __future__ import annotations

import pytest

from portfolio.avanza.types import (
    OHLC,
    MarketData,
    OrderResult,
    Position,
    Quote,
    SearchHit,
    StopLossResult,
    TickEntry,
    _ts,
    _val,
)

# ---------------------------------------------------------------------------
# _val helper
# ---------------------------------------------------------------------------

class TestVal:
    def test_plain_scalar(self):
        assert _val(42) == 42

    def test_none_returns_default(self):
        assert _val(None, 99) == 99

    def test_dict_with_value_key(self):
        assert _val({"value": 1.23, "unit": "SEK"}) == 1.23

    def test_dict_without_value_key(self):
        assert _val({"unit": "SEK"}, "missing") == "missing"

    def test_string_passthrough(self):
        assert _val("hello") == "hello"

    def test_zero_value(self):
        assert _val({"value": 0}) == 0

    def test_none_default(self):
        assert _val(None) is None


# ---------------------------------------------------------------------------
# _ts helper
# ---------------------------------------------------------------------------

class TestTs:
    def test_millis(self):
        result = _ts(1609459200000)
        assert "2021-01-01" in result

    def test_string_passthrough(self):
        assert _ts("2021-01-01T00:00:00") == "2021-01-01T00:00:00"

    def test_none(self):
        assert _ts(None) == ""


# ---------------------------------------------------------------------------
# Quote
# ---------------------------------------------------------------------------

class TestQuote:
    def test_from_api_basic(self):
        raw = {
            "buy": 100.5,
            "sell": 101.0,
            "last": 100.75,
            "changePercent": -0.5,
            "highest": 102.0,
            "lowest": 99.0,
            "totalVolumeTraded": 15000,
            "updated": 1609459200000,
        }
        q = Quote.from_api(raw)
        assert q.bid == 100.5
        assert q.ask == 101.0
        assert q.last == 100.75
        assert q.spread == 0.5  # computed: ask - bid
        assert q.change_percent == -0.5
        assert q.high == 102.0
        assert q.low == 99.0
        assert q.volume == 15000.0
        assert "2021" in q.updated

    def test_from_api_with_wrapped_values(self):
        raw = {
            "buy": {"value": 50.0, "unit": "SEK"},
            "sell": {"value": 50.5, "unit": "SEK"},
            "last": {"value": 50.25, "unit": "SEK"},
            "spread": {"value": 0.5},
            "changePercent": {"value": 1.2},
            "highest": {"value": 51.0},
            "lowest": {"value": 49.5},
            "totalVolumeTraded": {"value": 8000},
            "updated": "2025-01-01T10:00:00",
        }
        q = Quote.from_api(raw)
        assert q.bid == 50.0
        assert q.ask == 50.5
        assert q.spread == 0.5
        assert q.change_percent == 1.2
        assert q.volume == 8000.0

    def test_from_api_missing_fields(self):
        q = Quote.from_api({})
        assert q.bid == 0.0
        assert q.ask == 0.0
        assert q.last == 0.0
        assert q.spread == 0.0

    def test_immutable(self):
        q = Quote.from_api({"buy": 1, "sell": 2, "last": 1.5})
        with pytest.raises(AttributeError):
            q.bid = 999  # type: ignore[misc]

    def test_alternative_key_names(self):
        """Some endpoints use 'bid'/'ask'/'latest' instead of 'buy'/'sell'."""
        raw = {
            "bid": 200.0,
            "ask": 201.0,
            "latest": {"value": 200.5},
            "change_percent": 0.3,
            "high": 202.0,
            "low": 199.0,
            "volume": 5000,
            "updated": "",
        }
        q = Quote.from_api(raw)
        assert q.bid == 200.0
        assert q.ask == 201.0
        assert q.last == 200.5


# ---------------------------------------------------------------------------
# MarketData
# ---------------------------------------------------------------------------

class TestMarketData:
    @pytest.fixture()
    def raw_market_data(self):
        return {
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
                        "buySide": {"price": 100.0, "volume": 200, "priceString": "100.00"},
                        "sellSide": {"price": 101.0, "volume": 150, "priceString": "101.00"},
                    },
                    {
                        "buySide": {"price": 99.5, "volume": 300, "priceString": "99.50"},
                        "sellSide": {"price": 101.5, "volume": 100, "priceString": "101.50"},
                    },
                ],
                "marketMakerExpected": True,
            },
            "trades": [
                {"price": 100.5, "volume": 50, "buyer": "Nordnet", "seller": "SEB", "dealTime": "10:30:00"},
                {"price": 100.4, "volume": 30, "buyer": "Avanza", "seller": "Nordea", "dealTime": "10:29:55"},
            ],
        }

    def test_from_api(self, raw_market_data):
        md = MarketData.from_api(raw_market_data)
        assert md.quote.bid == 100.0
        assert md.quote.ask == 101.0
        assert len(md.bid_levels) == 2
        assert len(md.ask_levels) == 2
        assert md.bid_levels[0].price == 100.0
        assert md.bid_levels[0].volume == 200
        assert md.ask_levels[0].price == 101.0
        assert len(md.recent_trades) == 2
        assert md.recent_trades[0].buyer == "Nordnet"
        assert md.market_maker_expected is True

    def test_empty_depth(self):
        md = MarketData.from_api({"quote": {}, "orderDepth": {}, "trades": []})
        assert len(md.bid_levels) == 0
        assert len(md.ask_levels) == 0
        assert len(md.recent_trades) == 0

    def test_tuples_are_immutable(self, raw_market_data):
        md = MarketData.from_api(raw_market_data)
        assert isinstance(md.bid_levels, tuple)
        assert isinstance(md.ask_levels, tuple)
        assert isinstance(md.recent_trades, tuple)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class TestPosition:
    def test_from_api(self):
        raw = {
            "instrument": {
                "type": "CERTIFICATE",
                "name": "MINI S SILVER AVA 26",
                "orderbook": {
                    "id": "2213050",
                    "name": "MINI S SILVER AVA 26",
                    "type": "CERTIFICATE",
                    "quote": {
                        "latest": {"value": 5.80},
                        "changePercent": {"value": -1.5},
                    },
                },
                "currency": "SEK",
            },
            "account": {"id": "1625505"},
            "volume": {"value": 500},
            "value": {"value": 2900.0},
            "acquiredValue": {"value": 3000.0},
            "lastTradingDayPerformance": {
                "absolute": {"value": -100.0},
                "relative": {"value": -3.3},
            },
            "id": "pos-123",
        }
        p = Position.from_api(raw)
        assert p.name == "MINI S SILVER AVA 26"
        assert p.orderbook_id == "2213050"
        assert p.instrument_type == "CERTIFICATE"
        assert p.volume == 500.0
        assert p.value == 2900.0
        assert p.acquired_value == 3000.0
        assert p.profit == -100.0
        assert p.profit_percent == -3.3
        assert p.last_price == 5.80
        assert p.change_percent == -1.5
        assert p.account_id == "1625505"
        assert p.currency == "SEK"

    def test_minimal_position(self):
        p = Position.from_api({})
        assert p.name == ""
        assert p.volume == 0.0


# ---------------------------------------------------------------------------
# OrderResult
# ---------------------------------------------------------------------------

class TestOrderResult:
    def test_success(self):
        raw = {"orderRequestStatus": "SUCCESS", "orderId": "12345", "message": ""}
        r = OrderResult.from_api(raw)
        assert r.success is True
        assert r.order_id == "12345"
        assert r.status == "SUCCESS"

    def test_failure(self):
        raw = {"orderRequestStatus": "ERROR", "orderId": "", "message": "Insufficient funds"}
        r = OrderResult.from_api(raw)
        assert r.success is False
        assert r.message == "Insufficient funds"

    def test_message_list(self):
        raw = {"orderRequestStatus": "ERROR", "messages": ["err1", "err2"]}
        r = OrderResult.from_api(raw)
        assert "err1" in r.message
        assert "err2" in r.message


# ---------------------------------------------------------------------------
# StopLossResult
# ---------------------------------------------------------------------------

class TestStopLossResult:
    def test_success(self):
        raw = {"status": "ACTIVE", "stopLossId": "sl-999"}
        r = StopLossResult.from_api(raw)
        assert r.success is True
        assert r.stop_id == "sl-999"

    def test_failure(self):
        raw = {"status": "REJECTED", "id": "sl-000"}
        r = StopLossResult.from_api(raw)
        assert r.success is False
        assert r.stop_id == "sl-000"


# ---------------------------------------------------------------------------
# SearchHit
# ---------------------------------------------------------------------------

class TestSearchHit:
    def test_from_api(self):
        raw = {
            "id": "2213050",
            "name": "MINI S SILVER AVA 26",
            "instrumentType": "CERTIFICATE",
            "tradable": True,
            "lastPrice": {"value": 5.80},
            "changePercent": {"value": -0.5},
        }
        s = SearchHit.from_api(raw)
        assert s.orderbook_id == "2213050"
        assert s.name == "MINI S SILVER AVA 26"
        assert s.instrument_type == "CERTIFICATE"
        assert s.tradeable is True
        assert s.last_price == 5.80
        assert s.change_percent == -0.5

    def test_alternative_keys(self):
        raw = {
            "orderbookId": "123",
            "name": "Test",
            "type": "STOCK",
            "tradeable": False,
            "last_price": 10.0,
            "change_percent": 0.0,
        }
        s = SearchHit.from_api(raw)
        assert s.orderbook_id == "123"
        assert s.instrument_type == "STOCK"
        assert s.tradeable is False


# ---------------------------------------------------------------------------
# TickEntry
# ---------------------------------------------------------------------------

class TestTickEntry:
    def test_from_api(self):
        raw = {"min": 0.0, "max": 0.999, "tick": 0.001}
        t = TickEntry.from_api(raw)
        assert t.min_price == 0.0
        assert t.max_price == 0.999
        assert t.tick_size == 0.001

    def test_alternative_keys(self):
        raw = {"minPrice": 1.0, "maxPrice": 9.999, "tickSize": 0.01}
        t = TickEntry.from_api(raw)
        assert t.min_price == 1.0
        assert t.max_price == 9.999
        assert t.tick_size == 0.01


# ---------------------------------------------------------------------------
# OHLC
# ---------------------------------------------------------------------------

class TestOHLC:
    def test_from_api(self):
        raw = {
            "timestamp": 1609459200000,
            "open": 100.0,
            "high": 105.0,
            "low": 98.0,
            "close": 103.0,
            "totalVolumeTraded": 25000,
        }
        c = OHLC.from_api(raw)
        assert "2021" in c.timestamp
        assert c.open == 100.0
        assert c.high == 105.0
        assert c.low == 98.0
        assert c.close == 103.0
        assert c.volume == 25000

    def test_volume_fallback(self):
        raw = {"timestamp": 0, "open": 1, "high": 2, "low": 0, "close": 1, "volume": 999}
        c = OHLC.from_api(raw)
        assert c.volume == 999
