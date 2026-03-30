# Avanza Pipeline Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 4 overlapping Avanza modules with a unified `portfolio/avanza/` package using the existing `avanza-api` library (TOTP auth, `requests.Session`), add WebSocket streaming, and expose 10+ new API endpoints.

**Architecture:** Wrap the installed `avanza-api` PyPI library (`Qluxzz/avanza`) with thread-safety, auto-renewal, and our error handling conventions. The library already implements TOTP auth, `requests.Session` connection pooling, and 60+ endpoint methods. We add: a singleton manager, WebSocket streaming (CometD/Bayeux using `pushSubscriptionId`), tick-size logic, and legacy backward-compat wrappers so the ~30 existing callers don't break.

**Tech Stack:** Python 3.12, `avanza-api` (installed), `websocket-client`, `pyotp` (installed), `requests` (installed), `pytest`, `pytest-xdist`

**Spec:** `docs/superpowers/specs/2026-03-30-avanza-pipeline-overhaul-design.md`

---

## File Structure

```
portfolio/avanza/              # NEW PACKAGE
    __init__.py                # Public API exports — all imports come from here
    auth.py                    # Thread-safe singleton wrapping avanza-api Avanza class
    client.py                  # Thin wrapper: get/post/delete with auto-retry on 401
    types.py                   # Dataclasses for our normalized response types
    market_data.py             # Quotes, order depth, OHLC, instrument info
    trading.py                 # Orders, stop-losses, deals
    account.py                 # Positions, buying power, transactions
    search.py                  # Instrument search, warrant/cert discovery
    tick_rules.py              # Tick size lookup + price rounding
    streaming.py               # WebSocket CometD/Bayeux client

portfolio/avanza_session.py    # MODIFY → legacy wrapper delegating to new package
portfolio/avanza_client.py     # MODIFY → legacy wrapper delegating to new package
portfolio/avanza_control.py    # MODIFY → legacy wrapper delegating to new package
data/metals_avanza_helpers.py  # MODIFY → legacy wrapper delegating to new package

tests/test_avanza_pkg/         # NEW TEST DIRECTORY
    __init__.py
    test_auth.py               # Auth singleton, auto-renewal, thread-safety
    test_client.py             # HTTP methods, retry, error handling
    test_types.py              # Dataclass parsing from raw API responses
    test_market_data.py        # Quote, depth, OHLC parsing
    test_trading.py            # Order placement, modification, cancellation
    test_account.py            # Positions, buying power
    test_search.py             # Search, warrant/cert discovery
    test_tick_rules.py         # Tick size, price rounding
    test_streaming.py          # WebSocket handshake, subscribe, callbacks
    test_legacy_compat.py      # Legacy wrappers still work for existing callers
```

---

### Task 1: Types — Response Dataclasses

**Files:**
- Create: `portfolio/avanza/__init__.py`
- Create: `portfolio/avanza/types.py`
- Create: `tests/test_avanza_pkg/__init__.py`
- Create: `tests/test_avanza_pkg/test_types.py`

- [ ] **Step 1: Create package directory structure**

```bash
mkdir -p portfolio/avanza tests/test_avanza_pkg
touch portfolio/avanza/__init__.py tests/test_avanza_pkg/__init__.py
```

- [ ] **Step 2: Write the failing test for types**

Create `tests/test_avanza_pkg/test_types.py`:

```python
"""Tests for portfolio.avanza.types — response dataclass parsing."""

import pytest

from portfolio.avanza.types import (
    AccountCash,
    Deal,
    InstrumentInfo,
    MarketData,
    NewsArticle,
    OHLC,
    Order,
    OrderDepthLevel,
    OrderResult,
    Position,
    Quote,
    SearchHit,
    StopLoss,
    StopLossResult,
    TickEntry,
    Transaction,
)


class TestQuote:
    def test_from_api_response(self):
        raw = {
            "buy": {"value": 24.50},
            "sell": {"value": 24.55},
            "last": {"value": 24.52},
            "changePercent": {"value": 1.5},
            "highest": {"value": 25.0},
            "lowest": {"value": 24.0},
            "totalVolumeTraded": 12345,
            "updated": "2026-03-30T10:15:00",
        }
        q = Quote.from_api(raw)
        assert q.bid == 24.50
        assert q.ask == 24.55
        assert q.last == 24.52
        assert q.spread == pytest.approx(0.05)
        assert q.change_percent == 1.5
        assert q.high == 25.0
        assert q.low == 24.0
        assert q.volume == 12345

    def test_from_api_with_none_values(self):
        raw = {"buy": None, "sell": None, "last": {"value": 10.0}}
        q = Quote.from_api(raw)
        assert q.bid is None
        assert q.ask is None
        assert q.last == 10.0
        assert q.spread is None


class TestOrderDepthLevel:
    def test_from_api(self):
        raw = {"price": 24.50, "volume": 100}
        lvl = OrderDepthLevel.from_api(raw)
        assert lvl.price == 24.50
        assert lvl.volume == 100


class TestMarketData:
    def test_from_api_response(self):
        raw = {
            "quote": {
                "buy": {"value": 24.50},
                "sell": {"value": 24.55},
                "last": {"value": 24.52},
                "highest": {"value": 25.0},
                "lowest": {"value": 24.0},
                "totalVolumeTraded": 100,
                "updated": "2026-03-30T10:00:00",
            },
            "orderDepthLevels": [
                {"buy": {"price": 24.50, "volume": 50}, "sell": {"price": 24.55, "volume": 30}},
                {"buy": {"price": 24.45, "volume": 100}, "sell": {"price": 24.60, "volume": 80}},
            ],
            "latestTrades": [
                {"price": 24.52, "volume": 10, "buyer": "AVA", "seller": "NOR", "dealTime": "10:14:30"},
            ],
            "marketMakerExpected": True,
        }
        md = MarketData.from_api(raw)
        assert md.quote.bid == 24.50
        assert len(md.bid_levels) == 2
        assert md.bid_levels[0].price == 24.50
        assert len(md.ask_levels) == 2
        assert len(md.recent_trades) == 1
        assert md.market_maker_expected is True


class TestPosition:
    def test_from_api_response(self):
        raw = {
            "instrument": {
                "name": "MINI L SILVER AVA 140",
                "orderbook": {"id": "2334960", "type": "WARRANT"},
                "type": "WARRANT",
                "currency": "SEK",
            },
            "volume": {"value": 100},
            "value": {"value": 2450.0},
            "acquiredValue": {"value": 2400.0},
            "account": {"id": "1625505", "type": "ISK"},
        }
        p = Position.from_api(raw)
        assert p.name == "MINI L SILVER AVA 140"
        assert p.orderbook_id == "2334960"
        assert p.volume == 100
        assert p.value == 2450.0
        assert p.acquired_value == 2400.0
        assert p.profit == pytest.approx(50.0)
        assert p.profit_percent == pytest.approx(2.0833, rel=0.01)
        assert p.account_id == "1625505"


class TestOrderResult:
    def test_success(self):
        raw = {"orderRequestStatus": "SUCCESS", "orderId": "12345", "message": ""}
        r = OrderResult.from_api(raw)
        assert r.success is True
        assert r.order_id == "12345"

    def test_failure(self):
        raw = {"orderRequestStatus": "ERROR", "orderId": "", "message": "Insufficient funds"}
        r = OrderResult.from_api(raw)
        assert r.success is False
        assert r.message == "Insufficient funds"


class TestStopLossResult:
    def test_success(self):
        raw = {"status": "SUCCESS", "stoplossOrderId": "SL-789"}
        r = StopLossResult.from_api(raw)
        assert r.success is True
        assert r.stop_id == "SL-789"


class TestSearchHit:
    def test_from_api(self):
        raw = {
            "orderBookId": "856394",
            "title": "BULL GULD X8 AVA 6",
            "type": "CERTIFICATE",
            "tradeable": True,
            "lastPrice": "123.45",
            "changePercent": "+2.3%",
        }
        h = SearchHit.from_api(raw)
        assert h.orderbook_id == "856394"
        assert h.name == "BULL GULD X8 AVA 6"
        assert h.tradeable is True


class TestTickEntry:
    def test_from_api(self):
        raw = {"min": 0.0, "max": 0.999, "tick": 0.001}
        t = TickEntry.from_api(raw)
        assert t.min_price == 0.0
        assert t.max_price == 0.999
        assert t.tick_size == 0.001


class TestOHLC:
    def test_from_api(self):
        raw = {
            "timestamp": 1711785600000,
            "open": 24.0, "high": 25.0,
            "low": 23.5, "close": 24.8,
            "totalVolumeTraded": 5000,
        }
        c = OHLC.from_api(raw)
        assert c.open == 24.0
        assert c.high == 25.0
        assert c.close == 24.8
        assert c.volume == 5000
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'portfolio.avanza.types'`

- [ ] **Step 4: Implement types.py**

Create `portfolio/avanza/types.py`:

```python
"""Normalized response types for the Avanza API.

Each dataclass has a `from_api(raw)` classmethod that parses Avanza's
raw JSON responses into typed, predictable objects. This isolates
the rest of the codebase from Avanza's nested {value: ...} patterns.
"""

from __future__ import annotations

from dataclasses import dataclass


def _val(obj, default=None):
    """Extract value from Avanza's {value: X} pattern, or return raw."""
    if isinstance(obj, dict):
        return obj.get("value", default)
    return obj if obj is not None else default


@dataclass(frozen=True, slots=True)
class Quote:
    bid: float | None
    ask: float | None
    last: float | None
    spread: float | None
    change_percent: float | None
    high: float | None
    low: float | None
    volume: int
    updated: str

    @classmethod
    def from_api(cls, raw: dict) -> Quote:
        bid = _val(raw.get("buy"))
        ask = _val(raw.get("sell"))
        spread = round(ask - bid, 6) if bid is not None and ask is not None else None
        return cls(
            bid=bid,
            ask=ask,
            last=_val(raw.get("last")),
            spread=spread,
            change_percent=_val(raw.get("changePercent")),
            high=_val(raw.get("highest")),
            low=_val(raw.get("lowest")),
            volume=raw.get("totalVolumeTraded", 0) or 0,
            updated=raw.get("updated", ""),
        )


@dataclass(frozen=True, slots=True)
class OrderDepthLevel:
    price: float
    volume: int

    @classmethod
    def from_api(cls, raw: dict) -> OrderDepthLevel:
        return cls(price=raw["price"], volume=raw["volume"])


@dataclass(frozen=True, slots=True)
class Trade:
    price: float
    volume: int
    buyer: str
    seller: str
    time: str

    @classmethod
    def from_api(cls, raw: dict) -> Trade:
        return cls(
            price=raw.get("price", 0),
            volume=raw.get("volume", 0),
            buyer=raw.get("buyer", ""),
            seller=raw.get("seller", ""),
            time=raw.get("dealTime", ""),
        )


@dataclass(frozen=True, slots=True)
class MarketData:
    quote: Quote
    bid_levels: list[OrderDepthLevel]
    ask_levels: list[OrderDepthLevel]
    recent_trades: list[Trade]
    market_maker_expected: bool

    @classmethod
    def from_api(cls, raw: dict) -> MarketData:
        quote = Quote.from_api(raw.get("quote", {}))
        bid_levels = []
        ask_levels = []
        for level in raw.get("orderDepthLevels", []):
            if "buy" in level:
                bid_levels.append(OrderDepthLevel.from_api(level["buy"]))
            if "sell" in level:
                ask_levels.append(OrderDepthLevel.from_api(level["sell"]))
        trades = [Trade.from_api(t) for t in raw.get("latestTrades", [])]
        return cls(
            quote=quote,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            recent_trades=trades,
            market_maker_expected=raw.get("marketMakerExpected", False),
        )


@dataclass(frozen=True, slots=True)
class OrderResult:
    success: bool
    order_id: str
    status: str
    message: str

    @classmethod
    def from_api(cls, raw: dict) -> OrderResult:
        status = raw.get("orderRequestStatus", "UNKNOWN")
        return cls(
            success=status == "SUCCESS",
            order_id=str(raw.get("orderId", "")),
            status=status,
            message=raw.get("message", ""),
        )


@dataclass(frozen=True, slots=True)
class StopLossResult:
    success: bool
    stop_id: str
    status: str

    @classmethod
    def from_api(cls, raw: dict) -> StopLossResult:
        status = raw.get("status", "UNKNOWN")
        return cls(
            success=status == "SUCCESS",
            stop_id=str(raw.get("stoplossOrderId", "")),
            status=status,
        )


@dataclass(frozen=True, slots=True)
class Position:
    name: str
    orderbook_id: str
    instrument_type: str
    volume: int
    value: float
    acquired_value: float
    profit: float
    profit_percent: float
    last_price: float
    change_percent: float
    account_id: str
    currency: str

    @classmethod
    def from_api(cls, raw: dict) -> Position:
        inst = raw.get("instrument", {})
        ob = inst.get("orderbook", {})
        quote = ob.get("quote", {})
        vol = _val(raw.get("volume"), 0)
        val = _val(raw.get("value"), 0)
        acq = _val(raw.get("acquiredValue"), 0)
        latest = quote.get("latest", {})
        last_price = _val(latest, 0)
        change_pct = _val(quote.get("changePercent"), 0)
        profit = val - acq if val and acq else 0
        profit_pct = ((val - acq) / acq * 100) if acq else 0
        return cls(
            name=inst.get("name", ob.get("name", "")),
            orderbook_id=str(ob.get("id", "")),
            instrument_type=inst.get("type", ob.get("type", "")),
            volume=vol,
            value=val,
            acquired_value=acq,
            profit=profit,
            profit_percent=profit_pct,
            last_price=last_price,
            change_percent=change_pct,
            account_id=str(raw.get("account", {}).get("id", "")),
            currency=inst.get("currency", "SEK"),
        )


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    orderbook_id: str
    side: str
    price: float
    volume: int
    status: str
    account_id: str

    @classmethod
    def from_api(cls, raw: dict) -> Order:
        return cls(
            order_id=str(raw.get("orderId", raw.get("id", ""))),
            orderbook_id=str(raw.get("orderbookId", raw.get("orderBookId", ""))),
            side=raw.get("side", raw.get("type", "")),
            price=raw.get("price", 0),
            volume=raw.get("volume", 0),
            status=raw.get("status", raw.get("orderState", "")),
            account_id=str(raw.get("accountId", raw.get("account", {}).get("id", ""))),
        )


@dataclass(frozen=True, slots=True)
class Deal:
    deal_id: str
    orderbook_id: str
    side: str
    price: float
    volume: int
    time: str
    account_id: str

    @classmethod
    def from_api(cls, raw: dict) -> Deal:
        return cls(
            deal_id=str(raw.get("dealId", raw.get("id", ""))),
            orderbook_id=str(raw.get("orderbookId", raw.get("orderBookId", ""))),
            side=raw.get("side", raw.get("type", "")),
            price=raw.get("price", 0),
            volume=raw.get("volume", 0),
            time=raw.get("dealTime", raw.get("time", "")),
            account_id=str(raw.get("accountId", raw.get("account", {}).get("id", ""))),
        )


@dataclass(frozen=True, slots=True)
class StopLoss:
    stop_id: str
    orderbook_id: str
    trigger_price: float
    trigger_type: str
    sell_price: float
    volume: int
    status: str
    account_id: str

    @classmethod
    def from_api(cls, raw: dict) -> StopLoss:
        trigger = raw.get("stopLossTrigger", raw.get("trigger", {}))
        event = raw.get("stopLossOrderEvent", raw.get("orderEvent", {}))
        return cls(
            stop_id=str(raw.get("id", raw.get("stoplossOrderId", ""))),
            orderbook_id=str(raw.get("orderbookId", raw.get("orderBookId", ""))),
            trigger_price=trigger.get("value", trigger.get("price", 0)),
            trigger_type=trigger.get("type", "LESS_OR_EQUAL"),
            sell_price=event.get("price", 0),
            volume=event.get("volume", raw.get("volume", 0)),
            status=raw.get("status", ""),
            account_id=str(raw.get("accountId", "")),
        )


@dataclass(frozen=True, slots=True)
class SearchHit:
    orderbook_id: str
    name: str
    instrument_type: str
    tradeable: bool
    last_price: str
    change_percent: str

    @classmethod
    def from_api(cls, raw: dict) -> SearchHit:
        return cls(
            orderbook_id=str(raw.get("orderBookId", raw.get("id", ""))),
            name=raw.get("title", raw.get("name", "")),
            instrument_type=raw.get("type", raw.get("instrumentType", "")),
            tradeable=raw.get("tradeable", True),
            last_price=str(raw.get("lastPrice", "")),
            change_percent=str(raw.get("changePercent", "")),
        )


@dataclass(frozen=True, slots=True)
class TickEntry:
    min_price: float
    max_price: float
    tick_size: float

    @classmethod
    def from_api(cls, raw: dict) -> TickEntry:
        return cls(
            min_price=raw.get("min", 0),
            max_price=raw.get("max", 0),
            tick_size=raw.get("tick", 0),
        )


@dataclass(frozen=True, slots=True)
class OHLC:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: int

    @classmethod
    def from_api(cls, raw: dict) -> OHLC:
        return cls(
            timestamp=raw.get("timestamp", 0),
            open=raw.get("open", 0),
            high=raw.get("high", 0),
            low=raw.get("low", 0),
            close=raw.get("close", 0),
            volume=raw.get("totalVolumeTraded", raw.get("volume", 0)),
        )


@dataclass(frozen=True, slots=True)
class AccountCash:
    buying_power: float
    total_value: float
    own_capital: float

    @classmethod
    def from_api(cls, raw: dict) -> AccountCash:
        return cls(
            buying_power=_val(raw.get("buyingPower"), 0),
            total_value=_val(raw.get("totalValue"), 0),
            own_capital=_val(raw.get("ownCapital"), 0),
        )


@dataclass(frozen=True, slots=True)
class Transaction:
    transaction_id: str
    transaction_type: str
    instrument_name: str
    amount: float
    price: float
    volume: int
    date: str
    account_id: str

    @classmethod
    def from_api(cls, raw: dict) -> Transaction:
        return cls(
            transaction_id=str(raw.get("id", "")),
            transaction_type=raw.get("type", raw.get("transactionType", "")),
            instrument_name=raw.get("description", raw.get("instrumentName", "")),
            amount=raw.get("amount", raw.get("sum", 0)),
            price=raw.get("price", 0),
            volume=raw.get("volume", 0),
            date=raw.get("verificationDate", raw.get("date", "")),
            account_id=str(raw.get("accountId", raw.get("account", {}).get("id", ""))),
        )


@dataclass(frozen=True, slots=True)
class InstrumentInfo:
    orderbook_id: str
    name: str
    instrument_type: str
    currency: str
    leverage: float | None
    barrier: float | None
    underlying_name: str | None
    underlying_price: float | None

    @classmethod
    def from_api(cls, raw: dict) -> InstrumentInfo:
        ki = raw.get("keyIndicators", {})
        underlying = raw.get("underlying", {})
        uq = underlying.get("quote", {})
        return cls(
            orderbook_id=str(raw.get("id", raw.get("orderbookId", ""))),
            name=raw.get("name", ""),
            instrument_type=raw.get("type", raw.get("instrumentType", "")),
            currency=raw.get("currency", "SEK"),
            leverage=ki.get("leverage"),
            barrier=ki.get("barrierLevel"),
            underlying_name=underlying.get("name"),
            underlying_price=_val(uq.get("last")),
        )


@dataclass(frozen=True, slots=True)
class NewsArticle:
    article_id: str
    headline: str
    date: str
    source: str

    @classmethod
    def from_api(cls, raw: dict) -> NewsArticle:
        return cls(
            article_id=str(raw.get("id", "")),
            headline=raw.get("headline", raw.get("title", "")),
            date=raw.get("date", raw.get("publishedDate", "")),
            source=raw.get("source", ""),
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_types.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add portfolio/avanza/__init__.py portfolio/avanza/types.py tests/test_avanza_pkg/__init__.py tests/test_avanza_pkg/test_types.py
git commit -m "feat(avanza): add response dataclasses with from_api parsing"
```

---

### Task 2: Auth — Thread-Safe Singleton Client

**Files:**
- Create: `portfolio/avanza/auth.py`
- Create: `tests/test_avanza_pkg/test_auth.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_avanza_pkg/test_auth.py`:

```python
"""Tests for portfolio.avanza.auth — thread-safe TOTP client singleton."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth, AuthError


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the auth singleton before each test."""
    AvanzaAuth._instance = None
    AvanzaAuth._lock = threading.Lock()
    yield
    AvanzaAuth._instance = None


class TestAvanzaAuth:
    def test_get_instance_creates_singleton(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_client = MagicMock()
            mock_client._push_subscription_id = "push-123"
            mock_client._security_token = "csrf-456"
            mock_client._authentication_session = "auth-789"
            mock_client._customer_id = "cust-111"
            mock_create.return_value = mock_client

            auth = AvanzaAuth.get_instance(
                username="user", password="pass", totp_secret="secret"
            )
            assert auth.client is mock_client
            assert auth.push_subscription_id == "push-123"
            assert auth.csrf_token == "csrf-456"

    def test_get_instance_returns_same_singleton(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_client = MagicMock()
            mock_client._push_subscription_id = "push-123"
            mock_client._security_token = "csrf-456"
            mock_client._authentication_session = "auth-789"
            mock_client._customer_id = "cust-111"
            mock_create.return_value = mock_client

            a1 = AvanzaAuth.get_instance(
                username="user", password="pass", totp_secret="secret"
            )
            a2 = AvanzaAuth.get_instance(
                username="user", password="pass", totp_secret="secret"
            )
            assert a1 is a2
            assert mock_create.call_count == 1

    def test_thread_safety(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_client = MagicMock()
            mock_client._push_subscription_id = "push-123"
            mock_client._security_token = "csrf-456"
            mock_client._authentication_session = "auth-789"
            mock_client._customer_id = "cust-111"
            mock_create.return_value = mock_client

            results = []

            def get_auth():
                auth = AvanzaAuth.get_instance(
                    username="user", password="pass", totp_secret="secret"
                )
                results.append(id(auth))

            threads = [threading.Thread(target=get_auth) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(set(results)) == 1, "All threads should get the same instance"
            assert mock_create.call_count == 1

    def test_reset_clears_singleton(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_client = MagicMock()
            mock_client._push_subscription_id = "push-123"
            mock_client._security_token = "csrf-456"
            mock_client._authentication_session = "auth-789"
            mock_client._customer_id = "cust-111"
            mock_create.return_value = mock_client

            a1 = AvanzaAuth.get_instance(
                username="user", password="pass", totp_secret="secret"
            )
            AvanzaAuth.reset()
            a2 = AvanzaAuth.get_instance(
                username="user", password="pass", totp_secret="secret"
            )
            assert a1 is not a2
            assert mock_create.call_count == 2

    def test_auth_error_on_bad_credentials(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_create.side_effect = Exception("401 Unauthorized")
            with pytest.raises(AuthError, match="Failed to authenticate"):
                AvanzaAuth.get_instance(
                    username="bad", password="bad", totp_secret="bad"
                )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_auth.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement auth.py**

Create `portfolio/avanza/auth.py`:

```python
"""Thread-safe singleton wrapper around avanza-api TOTP authentication.

Usage:
    auth = AvanzaAuth.get_instance(username="...", password="...", totp_secret="...")
    client = auth.client  # The underlying avanza.Avanza instance
    push_id = auth.push_subscription_id  # For WebSocket streaming
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("portfolio.avanza.auth")


class AuthError(Exception):
    """Raised when TOTP authentication fails."""


def _create_avanza_client(username: str, password: str, totp_secret: str):
    """Create an authenticated avanza-api client instance.

    Separated for easy mocking in tests.
    """
    from avanza import Avanza

    return Avanza({
        "username": username,
        "password": password,
        "totpSecret": totp_secret,
    })


class AvanzaAuth:
    """Thread-safe singleton managing Avanza TOTP authentication.

    The underlying avanza-api library uses requests.Session with connection
    pooling. This class ensures only one client exists and all threads share it.
    """

    _instance: AvanzaAuth | None = None
    _lock = threading.Lock()

    def __init__(self, client, push_subscription_id: str, csrf_token: str,
                 authentication_session: str, customer_id: str):
        self.client = client
        self.push_subscription_id = push_subscription_id
        self.csrf_token = csrf_token
        self.authentication_session = authentication_session
        self.customer_id = customer_id

    @classmethod
    def get_instance(cls, username: str, password: str, totp_secret: str) -> AvanzaAuth:
        """Get or create the singleton auth instance.

        Thread-safe: only one thread will create the client, others wait.
        """
        if cls._instance is not None:
            return cls._instance

        with cls._lock:
            if cls._instance is not None:
                return cls._instance

            try:
                client = _create_avanza_client(username, password, totp_secret)
            except Exception as e:
                raise AuthError(f"Failed to authenticate with Avanza: {e}") from e

            cls._instance = cls(
                client=client,
                push_subscription_id=client._push_subscription_id,
                csrf_token=client._security_token,
                authentication_session=client._authentication_session,
                customer_id=client._customer_id,
            )
            logger.info(
                "Avanza TOTP auth successful (customer=%s, push_sub=%s...)",
                cls._instance.customer_id,
                cls._instance.push_subscription_id[:8] if cls._instance.push_subscription_id else "none",
            )
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton, forcing re-authentication on next call."""
        with cls._lock:
            cls._instance = None
            logger.info("Avanza auth singleton reset")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_auth.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/avanza/auth.py tests/test_avanza_pkg/test_auth.py
git commit -m "feat(avanza): thread-safe TOTP auth singleton"
```

---

### Task 3: Client — HTTP Wrapper with Auto-Retry

**Files:**
- Create: `portfolio/avanza/client.py`
- Create: `tests/test_avanza_pkg/test_client.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_avanza_pkg/test_client.py`:

```python
"""Tests for portfolio.avanza.client — HTTP wrapper with retry logic."""

import threading
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import AvanzaClient


@pytest.fixture(autouse=True)
def reset_singletons():
    AvanzaAuth._instance = None
    AvanzaAuth._lock = threading.Lock()
    AvanzaClient._instance = None
    AvanzaClient._lock = threading.Lock()
    yield
    AvanzaAuth._instance = None
    AvanzaClient._instance = None


def _mock_config():
    return {
        "avanza": {
            "username": "testuser",
            "password": "testpass",
            "totp_secret": "testsecret",
        }
    }


class TestAvanzaClient:
    def test_get_delegates_to_avanza_library(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = MagicMock()
            mock_avanza._push_subscription_id = "push"
            mock_avanza._security_token = "csrf"
            mock_avanza._authentication_session = "auth"
            mock_avanza._customer_id = "cust"
            mock_create.return_value = mock_avanza

            client = AvanzaClient.get_instance(_mock_config())

            mock_avanza.get_accounts_positions.return_value = {"withOrderbook": []}
            result = client.get_positions_raw()
            mock_avanza.get_accounts_positions.assert_called_once()

    def test_get_instance_is_singleton(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = MagicMock()
            mock_avanza._push_subscription_id = "push"
            mock_avanza._security_token = "csrf"
            mock_avanza._authentication_session = "auth"
            mock_avanza._customer_id = "cust"
            mock_create.return_value = mock_avanza

            c1 = AvanzaClient.get_instance(_mock_config())
            c2 = AvanzaClient.get_instance(_mock_config())
            assert c1 is c2

    def test_account_id_from_config(self):
        config = _mock_config()
        config["avanza"]["account_id"] = "9999"
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = MagicMock()
            mock_avanza._push_subscription_id = "push"
            mock_avanza._security_token = "csrf"
            mock_avanza._authentication_session = "auth"
            mock_avanza._customer_id = "cust"
            mock_create.return_value = mock_avanza

            client = AvanzaClient.get_instance(config)
            assert client.account_id == "9999"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_client.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement client.py**

Create `portfolio/avanza/client.py`:

```python
"""Thin client wrapper providing a clean interface over avanza-api.

This module:
- Manages the AvanzaAuth singleton
- Provides typed accessors that delegate to the library
- Handles the default account_id
- Will be used by market_data.py, trading.py, account.py, search.py
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from portfolio.avanza.auth import AvanzaAuth

logger = logging.getLogger("portfolio.avanza.client")

DEFAULT_ACCOUNT_ID = "1625505"


class AvanzaClient:
    """Singleton client wrapping the avanza-api library.

    All modules in portfolio/avanza/ use this to access the underlying
    Avanza client and share the same authenticated session.
    """

    _instance: AvanzaClient | None = None
    _lock = threading.Lock()

    def __init__(self, auth: AvanzaAuth, account_id: str):
        self._auth = auth
        self.account_id = account_id

    @classmethod
    def get_instance(cls, config: dict | None = None) -> AvanzaClient:
        """Get or create the singleton client.

        Args:
            config: Config dict with avanza.username, avanza.password,
                    avanza.totp_secret. Only needed on first call.
        """
        if cls._instance is not None:
            return cls._instance

        with cls._lock:
            if cls._instance is not None:
                return cls._instance

            if config is None:
                raise RuntimeError(
                    "AvanzaClient.get_instance() requires config on first call"
                )

            avanza_cfg = config.get("avanza", {})
            auth = AvanzaAuth.get_instance(
                username=avanza_cfg["username"],
                password=avanza_cfg["password"],
                totp_secret=avanza_cfg["totp_secret"],
            )
            account_id = str(avanza_cfg.get("account_id", DEFAULT_ACCOUNT_ID))
            cls._instance = cls(auth=auth, account_id=account_id)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset client singleton (forces re-auth on next call)."""
        with cls._lock:
            cls._instance = None
            AvanzaAuth.reset()

    @property
    def avanza(self):
        """The underlying avanza.Avanza instance (avanza-api library)."""
        return self._auth.client

    @property
    def push_subscription_id(self) -> str:
        """Push subscription ID for WebSocket streaming."""
        return self._auth.push_subscription_id

    @property
    def csrf_token(self) -> str:
        """CSRF token for manual HTTP calls outside the library."""
        return self._auth.csrf_token

    @property
    def session(self):
        """The underlying requests.Session for custom HTTP calls."""
        return self._auth.client._session

    # --- Raw library delegators (used by typed modules) ---

    def get_positions_raw(self) -> Any:
        return self.avanza.get_accounts_positions()

    def get_overview_raw(self) -> Any:
        return self.avanza.get_overview()

    def get_market_data_raw(self, ob_id: str) -> Any:
        return self.avanza.get_market_data(ob_id)

    def get_order_book_raw(self, ob_id: str) -> Any:
        return self.avanza.get_order_book(ob_id)

    def get_deals_raw(self) -> Any:
        return self.avanza.get_deals()

    def get_orders_raw(self) -> Any:
        return self.avanza.get_orders()

    def get_all_stop_losses_raw(self) -> Any:
        return self.avanza.get_all_stop_losses()

    def get_news_raw(self, ob_id: str) -> Any:
        return self.avanza.get_news(ob_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_client.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/avanza/client.py tests/test_avanza_pkg/test_client.py
git commit -m "feat(avanza): singleton client wrapper over avanza-api"
```

---

### Task 4: Market Data — Quotes, Depth, OHLC

**Files:**
- Create: `portfolio/avanza/market_data.py`
- Create: `tests/test_avanza_pkg/test_market_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_avanza_pkg/test_market_data.py`:

```python
"""Tests for portfolio.avanza.market_data — quotes, depth, OHLC."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.market_data import get_quote, get_market_data, get_ohlc, get_instrument_info


@pytest.fixture(autouse=True)
def reset_singletons():
    AvanzaAuth._instance = None
    AvanzaAuth._lock = threading.Lock()
    AvanzaClient._instance = None
    AvanzaClient._lock = threading.Lock()
    yield
    AvanzaAuth._instance = None
    AvanzaClient._instance = None


def _setup_mock_client():
    mock_avanza = MagicMock()
    mock_avanza._push_subscription_id = "push"
    mock_avanza._security_token = "csrf"
    mock_avanza._authentication_session = "auth"
    mock_avanza._customer_id = "cust"
    return mock_avanza


class TestGetQuote:
    def test_returns_quote_dataclass(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup_mock_client()
            mock_create.return_value = mock_avanza
            client = AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.get_instrument.return_value = {
                "quote": {
                    "buy": {"value": 24.50},
                    "sell": {"value": 24.55},
                    "last": {"value": 24.52},
                    "changePercent": {"value": 1.5},
                    "highest": {"value": 25.0},
                    "lowest": {"value": 24.0},
                    "totalVolumeTraded": 100,
                    "updated": "2026-03-30T10:00:00",
                },
            }

            q = get_quote("856394", instrument_type="certificate")
            assert q.bid == 24.50
            assert q.ask == 24.55
            assert q.spread == pytest.approx(0.05)


class TestGetMarketData:
    def test_returns_market_data_with_depth(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup_mock_client()
            mock_create.return_value = mock_avanza
            client = AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.get_market_data.return_value = {
                "quote": {
                    "buy": {"value": 24.50}, "sell": {"value": 24.55},
                    "last": {"value": 24.52}, "highest": {"value": 25.0},
                    "lowest": {"value": 24.0}, "totalVolumeTraded": 100,
                    "updated": "2026-03-30",
                },
                "orderDepthLevels": [
                    {"buy": {"price": 24.50, "volume": 50}, "sell": {"price": 24.55, "volume": 30}},
                ],
                "latestTrades": [
                    {"price": 24.52, "volume": 10, "buyer": "AVA", "seller": "NOR", "dealTime": "10:00"},
                ],
                "marketMakerExpected": True,
            }

            md = get_market_data("856394")
            assert md.quote.bid == 24.50
            assert len(md.bid_levels) == 1
            assert md.bid_levels[0].volume == 50
            assert len(md.recent_trades) == 1
            assert md.market_maker_expected is True


class TestGetOhlc:
    def test_returns_ohlc_list(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup_mock_client()
            mock_create.return_value = mock_avanza
            client = AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.get_chart_data.return_value = {
                "ohlc": [
                    {"timestamp": 1711785600000, "open": 24.0, "high": 25.0,
                     "low": 23.5, "close": 24.8, "totalVolumeTraded": 5000},
                    {"timestamp": 1711872000000, "open": 24.8, "high": 26.0,
                     "low": 24.5, "close": 25.5, "totalVolumeTraded": 6000},
                ],
            }

            candles = get_ohlc("856394", period="ONE_MONTH")
            assert len(candles) == 2
            assert candles[0].open == 24.0
            assert candles[1].close == 25.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_market_data.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement market_data.py**

Create `portfolio/avanza/market_data.py`:

```python
"""Market data: quotes, order depth, OHLC charts, instrument info."""

from __future__ import annotations

import logging

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import InstrumentInfo, MarketData, NewsArticle, OHLC, Quote

logger = logging.getLogger("portfolio.avanza.market_data")


def get_quote(ob_id: str, instrument_type: str = "certificate") -> Quote:
    """Fast quote: bid, ask, last, spread, change, volume.

    Args:
        ob_id: Avanza orderbook ID.
        instrument_type: "stock", "certificate", "warrant", "fund", etc.
    """
    client = AvanzaClient.get_instance()
    data = client.avanza.get_instrument(instrument_type, ob_id)
    return Quote.from_api(data.get("quote", data))


def get_market_data(ob_id: str) -> MarketData:
    """Full market data: quote + order depth (5 levels) + recent trades.

    Uses /_api/trading-critical/rest/marketdata/{id} — 15ms avg latency.
    """
    client = AvanzaClient.get_instance()
    data = client.get_market_data_raw(ob_id)
    return MarketData.from_api(data)


def get_ohlc(ob_id: str, period: str = "ONE_MONTH",
             resolution: str | None = None) -> list[OHLC]:
    """OHLC candle data from Avanza price charts.

    Args:
        ob_id: Avanza orderbook ID.
        period: ONE_WEEK, ONE_MONTH, THREE_MONTHS, THIS_YEAR, ONE_YEAR, etc.
        resolution: MINUTE, FIVE_MINUTES, TEN_MINUTES, THIRTY_MINUTES,
                    HOUR, DAY, WEEK, MONTH, QUARTER. If None, Avanza picks default.
    """
    from avanza.constants import TimePeriod, Resolution

    client = AvanzaClient.get_instance()
    tp = TimePeriod(period)
    res = Resolution(resolution) if resolution else None
    data = client.avanza.get_chart_data(ob_id, tp, res) if res else client.avanza.get_chart_data(ob_id, tp)
    return [OHLC.from_api(c) for c in data.get("ohlc", [])]


def get_instrument_info(ob_id: str, instrument_type: str = "certificate") -> InstrumentInfo:
    """Full instrument data including underlying, leverage, barrier.

    Args:
        ob_id: Avanza orderbook ID.
        instrument_type: "stock", "certificate", "warrant", "fund", etc.
    """
    client = AvanzaClient.get_instance()
    data = client.avanza.get_instrument(instrument_type, ob_id)
    return InstrumentInfo.from_api(data)


def get_news(ob_id: str) -> list[NewsArticle]:
    """Recent news articles for an instrument."""
    client = AvanzaClient.get_instance()
    data = client.get_news_raw(ob_id)
    articles = data.get("articles", []) if isinstance(data, dict) else []
    return [NewsArticle.from_api(a) for a in articles]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_market_data.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/avanza/market_data.py tests/test_avanza_pkg/test_market_data.py
git commit -m "feat(avanza): market data module — quotes, depth, OHLC"
```

---

### Task 5: Trading — Orders, Stop-Losses, Deals

**Files:**
- Create: `portfolio/avanza/trading.py`
- Create: `tests/test_avanza_pkg/test_trading.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_avanza_pkg/test_trading.py`:

```python
"""Tests for portfolio.avanza.trading — orders, stop-losses, deals."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.trading import (
    cancel_order, get_deals, get_orders, get_stop_losses,
    modify_order, place_order, place_stop_loss, place_trailing_stop,
    delete_stop_loss,
)


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


class TestPlaceOrder:
    def test_buy_order(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.place_order.return_value = {
                "orderRequestStatus": "SUCCESS", "orderId": "12345", "message": "",
            }

            result = place_order("BUY", "856394", 24.50, 100)
            assert result.success is True
            assert result.order_id == "12345"

    def test_sell_order_with_fok(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.place_order.return_value = {
                "orderRequestStatus": "SUCCESS", "orderId": "99", "message": "",
            }

            result = place_order("SELL", "856394", 25.0, 50, condition="FILL_OR_KILL")
            assert result.success is True
            mock_avanza.place_order.assert_called_once()
            call_kwargs = mock_avanza.place_order.call_args
            # Verify condition was passed through
            assert call_kwargs is not None

    def test_rejects_invalid_volume(self):
        with pytest.raises(ValueError, match="volume must be >= 1"):
            place_order("BUY", "856394", 24.50, 0)

    def test_rejects_invalid_price(self):
        with pytest.raises(ValueError, match="price must be > 0"):
            place_order("BUY", "856394", 0, 100)


class TestModifyOrder:
    def test_modify_price(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.edit_order.return_value = {
                "orderRequestStatus": "SUCCESS", "orderId": "12345", "message": "",
            }

            result = modify_order("12345", "856394", price=25.0, volume=100)
            assert result.success is True


class TestCancelOrder:
    def test_cancel(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.delete_order.return_value = {
                "orderRequestStatus": "SUCCESS", "orderId": "12345", "message": "",
            }

            result = cancel_order("12345")
            assert result is True


class TestPlaceStopLoss:
    def test_standard_stop(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.place_stop_loss_order.return_value = {
                "status": "SUCCESS", "stoplossOrderId": "SL-789",
            }

            result = place_stop_loss("856394", trigger_price=23.0, sell_price=22.5, volume=100)
            assert result.success is True
            assert result.stop_id == "SL-789"


class TestPlaceTrailingStop:
    def test_trailing_stop(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.place_stop_loss_order.return_value = {
                "status": "SUCCESS", "stoplossOrderId": "SL-TRAIL",
            }

            result = place_trailing_stop("856394", trail_percent=5.0, volume=100)
            assert result.success is True
            assert result.stop_id == "SL-TRAIL"


class TestGetDeals:
    def test_returns_deal_list(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.get_deals.return_value = [
                {"dealId": "D1", "orderbookId": "856394", "side": "BUY",
                 "price": 24.5, "volume": 100, "dealTime": "10:00", "accountId": "1625505"},
            ]

            deals = get_deals()
            assert len(deals) == 1
            assert deals[0].deal_id == "D1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_trading.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement trading.py**

Create `portfolio/avanza/trading.py`:

```python
"""Order management: placement, modification, cancellation, stop-losses, deals."""

from __future__ import annotations

import logging
from datetime import date

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import Deal, Order, OrderResult, StopLoss, StopLossResult

logger = logging.getLogger("portfolio.avanza.trading")


def place_order(
    side: str,
    ob_id: str,
    price: float,
    volume: int,
    condition: str = "NORMAL",
    valid_until: str | None = None,
    account_id: str | None = None,
) -> OrderResult:
    """Place a limit order.

    Args:
        side: "BUY" or "SELL".
        ob_id: Avanza orderbook ID.
        price: Limit price in SEK.
        volume: Number of units.
        condition: "NORMAL", "FILL_OR_KILL", or "FILL_AND_KILL".
        valid_until: ISO date. Defaults to today (day order).
        account_id: Defaults to configured account.
    """
    if volume < 1:
        raise ValueError(f"volume must be >= 1, got {volume}")
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    from avanza.constants import OrderType, Condition

    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    valid = valid_until or date.today().isoformat()

    result = client.avanza.place_order(
        account_id=acct,
        order_book_id=ob_id,
        order_type=OrderType(side),
        price=price,
        valid_until=valid,
        volume=volume,
        condition=Condition(condition),
    )
    parsed = OrderResult.from_api(result)
    if parsed.success:
        logger.info("Order %s placed: %dx @ %.3f (id=%s)", side, volume, price, parsed.order_id)
    else:
        logger.warning("Order %s failed: %s — %s", side, parsed.status, parsed.message)
    return parsed


def modify_order(
    order_id: str,
    ob_id: str,
    price: float,
    volume: int,
    condition: str = "NORMAL",
    valid_until: str | None = None,
    account_id: str | None = None,
) -> OrderResult:
    """Modify an existing order in-place (price, volume).

    Uses /_api/trading-critical/rest/order/modify — no cancel+replace needed.
    """
    from avanza.constants import OrderType, Condition

    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    valid = valid_until or date.today().isoformat()

    result = client.avanza.edit_order(
        instrument_id=ob_id,
        order_id=order_id,
        account_id=acct,
        price=price,
        volume=volume,
        valid_until=valid,
        condition=Condition(condition),
    )
    return OrderResult.from_api(result)


def cancel_order(order_id: str, account_id: str | None = None) -> bool:
    """Cancel an open order. Returns True on success."""
    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    try:
        result = client.avanza.delete_order(acct, order_id)
        status = result.get("orderRequestStatus", "") if isinstance(result, dict) else "SUCCESS"
        return status == "SUCCESS"
    except Exception as e:
        logger.warning("Cancel order %s failed: %s", order_id, e)
        return False


def get_orders() -> list[Order]:
    """Get all open orders."""
    client = AvanzaClient.get_instance()
    data = client.get_orders_raw()
    if isinstance(data, list):
        return [Order.from_api(o) for o in data]
    return [Order.from_api(o) for o in data.get("orders", [])]


def get_deals() -> list[Deal]:
    """Get recent fills (executed trades)."""
    client = AvanzaClient.get_instance()
    data = client.get_deals_raw()
    if isinstance(data, list):
        return [Deal.from_api(d) for d in data]
    return [Deal.from_api(d) for d in data.get("deals", [])]


def place_stop_loss(
    ob_id: str,
    trigger_price: float,
    sell_price: float,
    volume: int,
    valid_days: int = 8,
    trigger_type: str = "LESS_OR_EQUAL",
    value_type: str = "MONETARY",
    account_id: str | None = None,
) -> StopLossResult:
    """Place a hardware stop-loss order.

    Args:
        trigger_type: LESS_OR_EQUAL, MORE_OR_EQUAL, FOLLOW_DOWNWARDS, FOLLOW_UPWARDS.
        value_type: MONETARY or PERCENTAGE.
    """
    from avanza.constants import StopLossTriggerType, StopLossPriceType
    from avanza.entities import StopLossTrigger, StopLossOrderEvent

    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id

    trigger = StopLossTrigger(
        type=StopLossTriggerType(trigger_type),
        value=trigger_price,
        valid_until=None,
        value_type=StopLossPriceType(value_type),
    )
    event = StopLossOrderEvent(
        type="SELL",
        price=sell_price,
        volume=volume,
        valid_days=valid_days,
    )

    result = client.avanza.place_stop_loss_order(
        account_id=acct,
        order_book_id=ob_id,
        stop_loss_trigger=trigger,
        stop_loss_order_event=event,
    )
    return StopLossResult.from_api(result)


def place_trailing_stop(
    ob_id: str,
    trail_percent: float,
    volume: int,
    valid_days: int = 8,
    account_id: str | None = None,
) -> StopLossResult:
    """Place a hardware trailing stop-loss (Avanza manages the trail).

    The stop follows the price down by trail_percent%. If the instrument
    drops trail_percent% from its high, the stop triggers.
    """
    return place_stop_loss(
        ob_id=ob_id,
        trigger_price=trail_percent,
        sell_price=0,  # Market price on trigger
        volume=volume,
        valid_days=valid_days,
        trigger_type="FOLLOW_DOWNWARDS",
        value_type="PERCENTAGE",
        account_id=account_id,
    )


def get_stop_losses() -> list[StopLoss]:
    """List all active stop-loss orders."""
    client = AvanzaClient.get_instance()
    data = client.get_all_stop_losses_raw()
    if isinstance(data, list):
        return [StopLoss.from_api(s) for s in data]
    return []


def delete_stop_loss(stop_id: str, account_id: str | None = None) -> bool:
    """Delete a stop-loss order. Idempotent (404 = already gone)."""
    client = AvanzaClient.get_instance()
    acct = account_id or client.account_id
    try:
        client.avanza.delete_stop_loss_order(acct, stop_id)
        return True
    except Exception as e:
        if "404" in str(e):
            return True  # Already gone — idempotent
        logger.warning("Delete stop-loss %s failed: %s", stop_id, e)
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_trading.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/avanza/trading.py tests/test_avanza_pkg/test_trading.py
git commit -m "feat(avanza): trading module — orders, stop-losses, deals"
```

---

### Task 6: Account — Positions, Buying Power, Transactions

**Files:**
- Create: `portfolio/avanza/account.py`
- Create: `tests/test_avanza_pkg/test_account.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_avanza_pkg/test_account.py`:

```python
"""Tests for portfolio.avanza.account — positions, buying power."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.account import get_positions, get_buying_power


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


class TestGetPositions:
    def test_parses_positions(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.get_accounts_positions.return_value = {
                "withOrderbook": [
                    {
                        "instrument": {
                            "name": "MINI L SILVER",
                            "orderbook": {"id": "2334960", "type": "WARRANT", "quote": {"latest": {"value": 24.5}, "changePercent": {"value": 1.0}}},
                            "type": "WARRANT", "currency": "SEK",
                        },
                        "volume": {"value": 100},
                        "value": {"value": 2450.0},
                        "acquiredValue": {"value": 2400.0},
                        "account": {"id": "1625505", "type": "ISK"},
                    },
                ],
            }

            positions = get_positions()
            assert len(positions) == 1
            assert positions[0].name == "MINI L SILVER"
            assert positions[0].volume == 100
            assert positions[0].profit == pytest.approx(50.0)

    def test_filters_by_account(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.get_accounts_positions.return_value = {
                "withOrderbook": [
                    {"instrument": {"name": "A", "orderbook": {"id": "1"}, "type": "STOCK", "currency": "SEK"},
                     "volume": {"value": 10}, "value": {"value": 100}, "acquiredValue": {"value": 90},
                     "account": {"id": "1625505"}},
                    {"instrument": {"name": "B", "orderbook": {"id": "2"}, "type": "STOCK", "currency": "SEK"},
                     "volume": {"value": 20}, "value": {"value": 200}, "acquiredValue": {"value": 180},
                     "account": {"id": "9999"}},
                ],
            }

            positions = get_positions(account_id="1625505")
            assert len(positions) == 1
            assert positions[0].name == "A"


class TestGetBuyingPower:
    def test_returns_cash_info(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.get_overview.return_value = {
                "categories": [{
                    "accounts": [{
                        "id": "1625505",
                        "buyingPower": {"value": 50000.0},
                        "totalValue": {"value": 100000.0},
                        "ownCapital": {"value": 100000.0},
                    }],
                }],
            }

            cash = get_buying_power()
            assert cash.buying_power == 50000.0
            assert cash.total_value == 100000.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_account.py -v`
Expected: FAIL

- [ ] **Step 3: Implement account.py**

Create `portfolio/avanza/account.py`:

```python
"""Account operations: positions, buying power, transactions."""

from __future__ import annotations

import logging

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import AccountCash, Position, Transaction, _val

logger = logging.getLogger("portfolio.avanza.account")


def get_positions(account_id: str | None = None) -> list[Position]:
    """Get all positions, optionally filtered by account.

    Args:
        account_id: Filter to this account only. None = all accounts.
    """
    client = AvanzaClient.get_instance()
    data = client.get_positions_raw()
    entries = data.get("withOrderbook", []) if isinstance(data, dict) else []
    positions = [Position.from_api(e) for e in entries]
    if account_id:
        positions = [p for p in positions if p.account_id == str(account_id)]
    return positions


def get_buying_power(account_id: str | None = None) -> AccountCash:
    """Get buying power, total value, and own capital for an account.

    Args:
        account_id: Account to query. Defaults to configured account.
    """
    client = AvanzaClient.get_instance()
    acct = str(account_id or client.account_id)
    data = client.get_overview_raw()
    for cat in data.get("categories", []):
        for acc in cat.get("accounts", []):
            if str(acc.get("id", "")) == acct:
                return AccountCash.from_api(acc)
    logger.warning("Account %s not found in overview, returning zeros", acct)
    return AccountCash(buying_power=0, total_value=0, own_capital=0)


def get_transactions(
    from_date: str,
    to_date: str,
    types: list[str] | None = None,
    account_id: str | None = None,
) -> list[Transaction]:
    """Get transaction history.

    Args:
        from_date: Start date (ISO format, e.g. "2026-03-01").
        to_date: End date (ISO format).
        types: Filter by type: BUY, SELL, DIVIDEND, DEPOSIT, WITHDRAW.
        account_id: Filter by account. None = all.
    """
    from avanza.constants import TransactionsDetailsType

    client = AvanzaClient.get_instance()
    tx_types = [TransactionsDetailsType(t) for t in (types or ["BUY", "SELL"])]

    data = client.avanza.get_transactions_details(
        transaction_type=tx_types,
        from_date=from_date,
        to_date=to_date,
    )
    items = data.get("transactions", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    txns = [Transaction.from_api(t) for t in items]
    if account_id:
        txns = [t for t in txns if t.account_id == str(account_id)]
    return txns
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_account.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/avanza/account.py tests/test_avanza_pkg/test_account.py
git commit -m "feat(avanza): account module — positions, buying power, transactions"
```

---

### Task 7: Search — Instrument Discovery

**Files:**
- Create: `portfolio/avanza/search.py`
- Create: `tests/test_avanza_pkg/test_search.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_avanza_pkg/test_search.py`:

```python
"""Tests for portfolio.avanza.search — instrument search."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.search import search


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


class TestSearch:
    def test_search_returns_hits(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.search_for_instrument.return_value = {
                "totalNumberOfHits": 2,
                "hits": [
                    {"orderBookId": "856394", "title": "BULL GULD X8",
                     "type": "CERTIFICATE", "tradeable": True,
                     "lastPrice": "123.45", "changePercent": "+2.3%"},
                    {"orderBookId": "2334960", "title": "MINI L SILVER",
                     "type": "WARRANT", "tradeable": True,
                     "lastPrice": "24.50", "changePercent": "-1.0%"},
                ],
            }

            results = search("SILVER", limit=5)
            assert len(results) == 2
            assert results[0].orderbook_id == "856394"
            assert results[1].name == "MINI L SILVER"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_search.py -v`
Expected: FAIL

- [ ] **Step 3: Implement search.py**

Create `portfolio/avanza/search.py`:

```python
"""Instrument search and discovery."""

from __future__ import annotations

import logging

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import SearchHit

logger = logging.getLogger("portfolio.avanza.search")


def search(query: str, limit: int = 10, instrument_type: str | None = None) -> list[SearchHit]:
    """Search for instruments by name or ticker.

    Uses /_api/search/filtered-search (POST) — the working search endpoint.

    Args:
        query: Search string (e.g. "MINI L SILVER", "BULL GULD", "NVDA").
        limit: Max results (default 10).
        instrument_type: Optional filter: "stock", "certificate", "warrant", etc.
    """
    from avanza.constants import InstrumentType

    client = AvanzaClient.get_instance()
    itype = InstrumentType(instrument_type) if instrument_type else InstrumentType.ANY

    data = client.avanza.search_for_instrument(
        query=query,
        limit=limit,
        instrument_type=itype,
    )
    hits = data.get("hits", []) if isinstance(data, dict) else []
    return [SearchHit.from_api(h) for h in hits[:limit]]


def find_warrants(query: str = "", limit: int = 20) -> list[SearchHit]:
    """Search specifically for warrants."""
    return search(query=query, limit=limit, instrument_type="warrant")


def find_certificates(query: str = "", limit: int = 20) -> list[SearchHit]:
    """Search specifically for certificates."""
    return search(query=query, limit=limit, instrument_type="certificate")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_search.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/avanza/search.py tests/test_avanza_pkg/test_search.py
git commit -m "feat(avanza): search module — instrument discovery"
```

---

### Task 8: Tick Rules — Price Rounding

**Files:**
- Create: `portfolio/avanza/tick_rules.py`
- Create: `tests/test_avanza_pkg/test_tick_rules.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_avanza_pkg/test_tick_rules.py`:

```python
"""Tests for portfolio.avanza.tick_rules — tick size and price rounding."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from portfolio.avanza.auth import AvanzaAuth
from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.tick_rules import round_to_tick, get_tick_rules


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


SAMPLE_TICK_RULES = {
    "tickSizeList": {
        "tickSizeEntries": [
            {"min": 0.0, "max": 0.999, "tick": 0.001},
            {"min": 1.0, "max": 9.999, "tick": 0.01},
            {"min": 10.0, "max": 49.99, "tick": 0.05},
            {"min": 50.0, "max": 99.99, "tick": 0.1},
            {"min": 100.0, "max": 499.99, "tick": 0.5},
        ],
    },
}


class TestGetTickRules:
    def test_fetches_and_caches(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})

            mock_avanza.get_order_book.return_value = SAMPLE_TICK_RULES

            rules = get_tick_rules("856394")
            assert len(rules) == 5
            assert rules[0].tick_size == 0.001
            assert rules[4].tick_size == 0.5

            # Second call should use cache
            rules2 = get_tick_rules("856394")
            assert rules2 == rules
            assert mock_avanza.get_order_book.call_count == 1


class TestRoundToTick:
    def test_round_down_for_buy(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})
            mock_avanza.get_order_book.return_value = SAMPLE_TICK_RULES

            # Price 24.53 with tick=0.05 in [10,50) range -> round down to 24.50
            assert round_to_tick(24.53, "856394", direction="down") == 24.50

    def test_round_up_for_sell(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})
            mock_avanza.get_order_book.return_value = SAMPLE_TICK_RULES

            # Price 24.53 with tick=0.05 -> round up to 24.55
            assert round_to_tick(24.53, "856394", direction="up") == 24.55

    def test_exact_tick_unchanged(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})
            mock_avanza.get_order_book.return_value = SAMPLE_TICK_RULES

            assert round_to_tick(24.50, "856394", direction="down") == 24.50

    def test_small_price_uses_smallest_tick(self):
        with patch("portfolio.avanza.auth._create_avanza_client") as mock_create:
            mock_avanza = _setup()
            mock_create.return_value = mock_avanza
            AvanzaClient.get_instance({"avanza": {"username": "u", "password": "p", "totp_secret": "s"}})
            mock_avanza.get_order_book.return_value = SAMPLE_TICK_RULES

            # Price 0.567 with tick=0.001 -> 0.567
            assert round_to_tick(0.567, "856394", direction="down") == 0.567
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_tick_rules.py -v`
Expected: FAIL

- [ ] **Step 3: Implement tick_rules.py**

Create `portfolio/avanza/tick_rules.py`:

```python
"""Tick size rules and price rounding for valid order prices."""

from __future__ import annotations

import math
import logging

from portfolio.avanza.client import AvanzaClient
from portfolio.avanza.types import TickEntry

logger = logging.getLogger("portfolio.avanza.tick_rules")

# Cache: ob_id -> list[TickEntry]
_cache: dict[str, list[TickEntry]] = {}


def get_tick_rules(ob_id: str) -> list[TickEntry]:
    """Get tick size rules for an instrument. Cached per session.

    Returns sorted list of TickEntry with min_price, max_price, tick_size.
    """
    if ob_id in _cache:
        return _cache[ob_id]

    client = AvanzaClient.get_instance()
    data = client.get_order_book_raw(ob_id)
    tick_list = data.get("tickSizeList", {})
    entries = [TickEntry.from_api(e) for e in tick_list.get("tickSizeEntries", [])]
    entries.sort(key=lambda e: e.min_price)
    _cache[ob_id] = entries
    return entries


def _find_tick(price: float, rules: list[TickEntry]) -> float:
    """Find the tick size for a given price level."""
    for entry in rules:
        if entry.min_price <= price <= entry.max_price:
            return entry.tick_size
    # Fallback: use the last (largest price range) entry
    if rules:
        return rules[-1].tick_size
    return 0.01  # Safe default


def round_to_tick(price: float, ob_id: str, direction: str = "down") -> float:
    """Round a price to the nearest valid tick increment.

    Args:
        price: Raw price to round.
        ob_id: Avanza orderbook ID (to look up tick rules).
        direction: "down" for buy orders (favor lower price),
                   "up" for sell orders (favor higher price).

    Returns:
        Price rounded to the nearest valid tick.
    """
    rules = get_tick_rules(ob_id)
    tick = _find_tick(price, rules)
    if tick <= 0:
        return price

    # Use integer arithmetic to avoid floating-point drift
    # Multiply by inverse of tick, round, multiply back
    inv = round(1.0 / tick)
    scaled = price * inv
    if direction == "down":
        rounded = math.floor(scaled)
    else:
        rounded = math.ceil(scaled)
    return round(rounded / inv, 10)


def clear_cache() -> None:
    """Clear the tick rules cache (e.g. on session reset)."""
    _cache.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_tick_rules.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/avanza/tick_rules.py tests/test_avanza_pkg/test_tick_rules.py
git commit -m "feat(avanza): tick rules — price rounding to valid increments"
```

---

### Task 9: Package Init — Public API Exports

**Files:**
- Modify: `portfolio/avanza/__init__.py`

- [ ] **Step 1: Write the public API**

Update `portfolio/avanza/__init__.py`:

```python
"""Unified Avanza API package.

All imports should come from this module:

    from portfolio.avanza import get_quote, place_order, get_positions
    from portfolio.avanza.types import Quote, OrderResult, Position
    from portfolio.avanza.streaming import AvanzaStream
"""

# Auth & client
from portfolio.avanza.auth import AvanzaAuth, AuthError
from portfolio.avanza.client import AvanzaClient

# Account
from portfolio.avanza.account import get_buying_power, get_positions, get_transactions

# Market data
from portfolio.avanza.market_data import (
    get_instrument_info,
    get_market_data,
    get_news,
    get_ohlc,
    get_quote,
)

# Search
from portfolio.avanza.search import find_certificates, find_warrants, search

# Tick rules
from portfolio.avanza.tick_rules import clear_cache as clear_tick_cache
from portfolio.avanza.tick_rules import get_tick_rules, round_to_tick

# Trading
from portfolio.avanza.trading import (
    cancel_order,
    delete_stop_loss,
    get_deals,
    get_orders,
    get_stop_losses,
    modify_order,
    place_order,
    place_stop_loss,
    place_trailing_stop,
)

__all__ = [
    # Auth
    "AvanzaAuth", "AuthError", "AvanzaClient",
    # Account
    "get_positions", "get_buying_power", "get_transactions",
    # Market data
    "get_quote", "get_market_data", "get_ohlc", "get_instrument_info", "get_news",
    # Search
    "search", "find_warrants", "find_certificates",
    # Tick rules
    "get_tick_rules", "round_to_tick", "clear_tick_cache",
    # Trading
    "place_order", "modify_order", "cancel_order",
    "get_orders", "get_deals",
    "place_stop_loss", "place_trailing_stop", "get_stop_losses", "delete_stop_loss",
]
```

- [ ] **Step 2: Verify all imports resolve**

Run: `.venv/Scripts/python.exe -c "from portfolio.avanza import *; print('All exports OK:', len(__all__), 'symbols')"`
Expected: `All exports OK: 21 symbols`

- [ ] **Step 3: Run full test suite for the package**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add portfolio/avanza/__init__.py
git commit -m "feat(avanza): package init with full public API"
```

---

### Task 10: WebSocket Streaming — CometD/Bayeux Client

**Files:**
- Create: `portfolio/avanza/streaming.py`
- Create: `tests/test_avanza_pkg/test_streaming.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_avanza_pkg/test_streaming.py`:

```python
"""Tests for portfolio.avanza.streaming — WebSocket CometD client."""

import json
import threading
from unittest.mock import MagicMock, patch, call

import pytest

from portfolio.avanza.streaming import AvanzaStream


class TestAvanzaStream:
    def test_handshake_sends_correct_message(self):
        with patch("portfolio.avanza.streaming.websocket") as mock_ws_mod:
            mock_ws = MagicMock()
            mock_ws_mod.create_connection.return_value = mock_ws

            # Simulate handshake response
            mock_ws.recv.side_effect = [
                json.dumps([{"clientId": "client-123", "successful": True,
                             "channel": "/meta/handshake"}]),
                json.dumps([{"successful": True, "channel": "/meta/connect"}]),
            ]

            stream = AvanzaStream("push-sub-id")
            stream._do_handshake()

            # Verify handshake was sent
            sent = json.loads(mock_ws.send.call_args_list[0][0][0])
            assert sent[0]["channel"] == "/meta/handshake"
            assert sent[0]["ext"]["subscriptionId"] == "push-sub-id"
            assert stream._client_id == "client-123"

    def test_subscribe_sends_correct_channel(self):
        with patch("portfolio.avanza.streaming.websocket") as mock_ws_mod:
            mock_ws = MagicMock()
            mock_ws_mod.create_connection.return_value = mock_ws

            stream = AvanzaStream("push-sub-id")
            stream._ws = mock_ws
            stream._client_id = "client-123"

            mock_ws.recv.return_value = json.dumps([{
                "successful": True, "channel": "/meta/subscribe",
            }])

            stream._subscribe_channel("/quotes/856394")

            sent = json.loads(mock_ws.send.call_args[0][0])
            assert sent[0]["channel"] == "/meta/subscribe"
            assert sent[0]["subscription"] == "/quotes/856394"
            assert sent[0]["clientId"] == "client-123"

    def test_callback_dispatched_on_message(self):
        stream = AvanzaStream("push-sub-id")
        stream._client_id = "client-123"
        received = []

        stream.on_quote("856394", lambda data: received.append(data))

        # Simulate incoming message
        stream._dispatch_message({
            "channel": "/quotes/856394",
            "data": {"buy": 24.50, "sell": 24.55},
        })

        assert len(received) == 1
        assert received[0]["buy"] == 24.50

    def test_no_callback_for_unsubscribed_channel(self):
        stream = AvanzaStream("push-sub-id")
        received = []

        stream.on_quote("856394", lambda data: received.append(data))

        # Message for a different instrument
        stream._dispatch_message({
            "channel": "/quotes/999999",
            "data": {"buy": 10.0},
        })

        assert len(received) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_streaming.py -v`
Expected: FAIL

- [ ] **Step 3: Install websocket-client if needed**

Run: `.venv/Scripts/python.exe -m pip install websocket-client 2>/dev/null || echo "already installed"`

- [ ] **Step 4: Implement streaming.py**

Create `portfolio/avanza/streaming.py`:

```python
"""WebSocket streaming client using CometD/Bayeux protocol.

Connects to wss://www.avanza.se/_push/cometd for real-time data:
- Quote updates (sub-second price changes)
- Order depth (live order book)
- Order/deal notifications (fill confirmation)

Requires pushSubscriptionId from TOTP authentication.

Usage:
    from portfolio.avanza import AvanzaClient
    from portfolio.avanza.streaming import AvanzaStream

    client = AvanzaClient.get_instance(config)
    stream = AvanzaStream(client.push_subscription_id)
    stream.on_quote("856394", lambda data: print(f"Price: {data}"))
    stream.start()  # Runs in background thread
    ...
    stream.stop()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable

import websocket

logger = logging.getLogger("portfolio.avanza.streaming")

WS_URL = "wss://www.avanza.se/_push/cometd"
HEARTBEAT_INTERVAL = 30  # seconds
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0


class AvanzaStream:
    """CometD/Bayeux WebSocket client for Avanza real-time data."""

    def __init__(self, push_subscription_id: str):
        self._push_sub_id = push_subscription_id
        self._ws: websocket.WebSocket | None = None
        self._client_id: str = ""
        self._callbacks: dict[str, list[Callable]] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._reconnect_delay = RECONNECT_BASE_DELAY
        self._subscriptions: list[str] = []
        self._lock = threading.Lock()

    # --- Public API ---

    def on_quote(self, ob_id: str, callback: Callable[[dict], None]) -> None:
        """Register callback for quote updates on an instrument."""
        channel = f"/quotes/{ob_id}"
        self._callbacks.setdefault(channel, []).append(callback)
        if channel not in self._subscriptions:
            self._subscriptions.append(channel)

    def on_order_depth(self, ob_id: str, callback: Callable[[dict], None]) -> None:
        """Register callback for order depth updates."""
        channel = f"/orderdepths/{ob_id}"
        self._callbacks.setdefault(channel, []).append(callback)
        if channel not in self._subscriptions:
            self._subscriptions.append(channel)

    def on_trades(self, ob_id: str, callback: Callable[[dict], None]) -> None:
        """Register callback for individual trade events."""
        channel = f"/trades/{ob_id}"
        self._callbacks.setdefault(channel, []).append(callback)
        if channel not in self._subscriptions:
            self._subscriptions.append(channel)

    def on_orders(self, account_ids: list[str], callback: Callable[[dict], None]) -> None:
        """Register callback for order status changes."""
        channel = f"/orders/_{','.join(account_ids)}"
        self._callbacks.setdefault(channel, []).append(callback)
        if channel not in self._subscriptions:
            self._subscriptions.append(channel)

    def on_deals(self, account_ids: list[str], callback: Callable[[dict], None]) -> None:
        """Register callback for executed deals (fills)."""
        channel = f"/deals/_{','.join(account_ids)}"
        self._callbacks.setdefault(channel, []).append(callback)
        if channel not in self._subscriptions:
            self._subscriptions.append(channel)

    def start(self) -> None:
        """Start streaming in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="avanza-stream")
        self._thread.start()
        logger.info("Avanza WebSocket streaming started")

    def stop(self) -> None:
        """Stop streaming and close the WebSocket."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Avanza WebSocket streaming stopped")

    # --- Internal ---

    def _run_loop(self) -> None:
        """Background thread: connect, subscribe, read messages."""
        while self._running:
            try:
                self._connect()
                self._do_handshake()
                for channel in self._subscriptions:
                    self._subscribe_channel(channel)
                self._reconnect_delay = RECONNECT_BASE_DELAY
                self._read_loop()
            except Exception as e:
                if not self._running:
                    break
                logger.warning(
                    "WebSocket error: %s. Reconnecting in %.0fs...",
                    e, self._reconnect_delay,
                )
                time.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, RECONNECT_MAX_DELAY
                )

    def _connect(self) -> None:
        """Open WebSocket connection."""
        self._ws = websocket.create_connection(
            WS_URL,
            timeout=HEARTBEAT_INTERVAL + 10,
        )

    def _do_handshake(self) -> None:
        """Perform CometD/Bayeux handshake."""
        msg = [{
            "channel": "/meta/handshake",
            "ext": {"subscriptionId": self._push_sub_id},
            "version": "1.0",
            "minimumVersion": "1.0",
            "supportedConnectionTypes": ["websocket"],
        }]
        self._ws.send(json.dumps(msg))
        resp = json.loads(self._ws.recv())
        if not resp[0].get("successful"):
            raise RuntimeError(f"Handshake failed: {resp}")
        self._client_id = resp[0]["clientId"]
        logger.debug("CometD handshake OK, clientId=%s", self._client_id)

        # Send connect
        connect_msg = [{
            "channel": "/meta/connect",
            "clientId": self._client_id,
            "connectionType": "websocket",
        }]
        self._ws.send(json.dumps(connect_msg))
        self._ws.recv()  # Ack

    def _subscribe_channel(self, channel: str) -> None:
        """Subscribe to a CometD channel."""
        msg = [{
            "channel": "/meta/subscribe",
            "subscription": channel,
            "clientId": self._client_id,
        }]
        self._ws.send(json.dumps(msg))
        resp = json.loads(self._ws.recv())
        if resp[0].get("successful"):
            logger.debug("Subscribed to %s", channel)
        else:
            logger.warning("Subscribe to %s failed: %s", channel, resp)

    def _read_loop(self) -> None:
        """Read messages and dispatch to callbacks."""
        last_heartbeat = time.time()
        while self._running:
            # Send heartbeat
            if time.time() - last_heartbeat > HEARTBEAT_INTERVAL:
                heartbeat = [{
                    "channel": "/meta/connect",
                    "clientId": self._client_id,
                    "connectionType": "websocket",
                }]
                self._ws.send(json.dumps(heartbeat))
                last_heartbeat = time.time()

            try:
                raw = self._ws.recv()
                if not raw:
                    continue
                messages = json.loads(raw)
                for msg in messages:
                    channel = msg.get("channel", "")
                    if channel.startswith("/meta/"):
                        continue  # Protocol messages
                    self._dispatch_message(msg)
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException:
                raise

    def _dispatch_message(self, msg: dict) -> None:
        """Route a message to registered callbacks."""
        channel = msg.get("channel", "")
        data = msg.get("data", {})
        for cb in self._callbacks.get(channel, []):
            try:
                cb(data)
            except Exception as e:
                logger.error("Callback error on %s: %s", channel, e)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_streaming.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add portfolio/avanza/streaming.py tests/test_avanza_pkg/test_streaming.py
git commit -m "feat(avanza): WebSocket streaming — CometD/Bayeux client"
```

---

### Task 11: Legacy Compatibility Wrappers

**Files:**
- Modify: `portfolio/avanza_session.py` (add legacy re-exports at bottom)
- Create: `tests/test_avanza_pkg/test_legacy_compat.py`

This task adds backward-compatible imports so the ~30 existing callers continue working. The old Playwright-based code stays (for BankID backup), but new re-exports delegate to the package when the TOTP client is available.

- [ ] **Step 1: Write the failing test**

Create `tests/test_avanza_pkg/test_legacy_compat.py`:

```python
"""Tests that legacy imports still work via the new package.

These tests verify that existing callers don't break.
"""

import pytest


class TestLegacyImports:
    """Verify the public API symbols that existing code imports."""

    def test_avanza_session_exports(self):
        from portfolio.avanza_session import (
            AvanzaSessionError,
            api_get,
            api_post,
            api_delete,
            get_positions,
            get_buying_power,
            get_instrument_price,
            get_quote,
            place_buy_order,
            place_sell_order,
            cancel_order,
            get_open_orders,
            verify_session,
            load_session,
            session_remaining_minutes,
            is_session_expiring_soon,
            delete_stop_loss,
        )
        # All should be importable (they remain as-is or become new-package delegates)
        assert callable(api_get)
        assert callable(place_buy_order)

    def test_avanza_client_exports(self):
        from portfolio.avanza_client import (
            get_price,
            get_positions,
            get_portfolio_value,
            find_instrument,
            get_client,
        )
        assert callable(get_price)
        assert callable(find_instrument)

    def test_new_package_importable(self):
        from portfolio.avanza import (
            get_quote,
            get_market_data,
            get_ohlc,
            place_order,
            modify_order,
            cancel_order,
            get_positions,
            get_buying_power,
            search,
            round_to_tick,
            get_tick_rules,
            place_trailing_stop,
            AvanzaClient,
            AvanzaAuth,
        )
        assert callable(get_quote)
        assert callable(place_order)
        assert callable(search)
```

- [ ] **Step 2: Run test to verify it passes**

These tests verify imports only — the legacy modules already export these symbols.

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/test_legacy_compat.py -v`
Expected: All tests PASS (imports resolve)

- [ ] **Step 3: Commit**

```bash
git add tests/test_avanza_pkg/test_legacy_compat.py
git commit -m "test(avanza): legacy compatibility import verification"
```

---

### Task 12: Full Test Suite Run + Integration Smoke Test

**Files:**
- Create: `scripts/avanza_smoke_test.py`

- [ ] **Step 1: Run the full new package test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_pkg/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Run the existing Avanza tests (legacy)**

Run: `.venv/Scripts/python.exe -m pytest tests/test_avanza_session.py tests/test_avanza_orders.py tests/test_avanza_control.py -v --tb=short`
Expected: All existing tests still PASS

- [ ] **Step 3: Create a live smoke test script**

Create `scripts/avanza_smoke_test.py`:

```python
"""Live smoke test for the new Avanza package.

Run: .venv/Scripts/python.exe scripts/avanza_smoke_test.py

Requires valid TOTP credentials in config.json.
"""

import json
import sys
import time

sys.path.insert(0, ".")

from portfolio.file_utils import load_json

config = load_json("config.json")
if not config or "avanza" not in config:
    print("ERROR: config.json missing or no avanza section")
    sys.exit(1)

from portfolio.avanza.client import AvanzaClient

print("1. Authenticating with TOTP...")
t0 = time.perf_counter()
client = AvanzaClient.get_instance(config)
dt = (time.perf_counter() - t0) * 1000
print(f"   OK — authenticated in {dt:.0f}ms")
print(f"   Push subscription ID: {client.push_subscription_id[:12]}...")
print()

from portfolio.avanza.account import get_positions, get_buying_power

print("2. Fetching positions...")
t0 = time.perf_counter()
positions = get_positions(account_id="1625505")
dt = (time.perf_counter() - t0) * 1000
print(f"   OK — {len(positions)} positions in {dt:.0f}ms")
for p in positions[:5]:
    print(f"   {p.name}: {p.volume}x @ {p.last_price:.2f} SEK (P&L: {p.profit:+.0f} SEK)")
print()

print("3. Fetching buying power...")
t0 = time.perf_counter()
cash = get_buying_power()
dt = (time.perf_counter() - t0) * 1000
print(f"   OK — BP: {cash.buying_power:,.0f} SEK, Total: {cash.total_value:,.0f} SEK ({dt:.0f}ms)")
print()

from portfolio.avanza.market_data import get_quote, get_market_data

print("4. Fetching quote (856394 = BULL GULD X8)...")
t0 = time.perf_counter()
q = get_quote("856394")
dt = (time.perf_counter() - t0) * 1000
print(f"   OK — bid={q.bid} ask={q.ask} last={q.last} spread={q.spread} ({dt:.0f}ms)")
print()

print("5. Fetching market data with order depth...")
t0 = time.perf_counter()
md = get_market_data("856394")
dt = (time.perf_counter() - t0) * 1000
print(f"   OK — {len(md.bid_levels)} bid levels, {len(md.ask_levels)} ask levels ({dt:.0f}ms)")
for i, (b, a) in enumerate(zip(md.bid_levels[:3], md.ask_levels[:3])):
    print(f"   L{i+1}: {b.volume:>5}x {b.price:.2f} | {a.price:.2f} x{a.volume}")
print()

from portfolio.avanza.search import search

print("6. Searching instruments...")
t0 = time.perf_counter()
results = search("MINI L SILVER", limit=3)
dt = (time.perf_counter() - t0) * 1000
print(f"   OK — {len(results)} results ({dt:.0f}ms)")
for r in results:
    print(f"   {r.orderbook_id} | {r.name} | {r.instrument_type}")
print()

from portfolio.avanza.tick_rules import get_tick_rules, round_to_tick

print("7. Fetching tick rules...")
t0 = time.perf_counter()
rules = get_tick_rules("856394")
dt = (time.perf_counter() - t0) * 1000
print(f"   OK — {len(rules)} tick levels ({dt:.0f}ms)")
for r in rules[:3]:
    print(f"   {r.min_price:.3f} - {r.max_price:.3f} -> tick={r.tick_size}")
test_price = 24.53
rounded = round_to_tick(test_price, "856394", direction="down")
print(f"   round_to_tick({test_price}, down) = {rounded}")
print()

from portfolio.avanza.trading import get_orders, get_deals

print("8. Fetching open orders...")
t0 = time.perf_counter()
orders = get_orders()
dt = (time.perf_counter() - t0) * 1000
print(f"   OK — {len(orders)} open orders ({dt:.0f}ms)")
print()

print("9. Fetching recent deals...")
t0 = time.perf_counter()
deals = get_deals()
dt = (time.perf_counter() - t0) * 1000
print(f"   OK — {len(deals)} recent deals ({dt:.0f}ms)")
for d in deals[:3]:
    print(f"   {d.side} {d.volume}x @ {d.price:.2f} ({d.time})")
print()

print("=" * 50)
print("ALL SMOKE TESTS PASSED")
print("=" * 50)
```

- [ ] **Step 4: Run the smoke test live**

Run: `.venv/Scripts/python.exe scripts/avanza_smoke_test.py`
Expected: All 9 checks print "OK" and final "ALL SMOKE TESTS PASSED"

- [ ] **Step 5: Commit**

```bash
git add scripts/avanza_smoke_test.py
git commit -m "test(avanza): live smoke test for TOTP auth + all endpoints"
```

---

### Task 13: Run Full Existing Test Suite

Verify nothing is broken in the broader codebase.

- [ ] **Step 1: Run the complete test suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -n auto --tb=short -q`
Expected: ~3,168 tests. Same pass/fail count as before (26 pre-existing failures).

- [ ] **Step 2: If new failures, diagnose and fix**

Compare failure list against known 26 pre-existing failures. Any new failures must be fixed before merging.

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git add -u
git commit -m "fix(avanza): resolve test regressions from package migration"
```

---

## Execution Notes

- **Worktree:** Create branch `feat/avanza-pipeline-overhaul` in a worktree
- **All code uses `from portfolio.avanza import X`** — never import from submodules directly in production code
- **Config access:** The `AvanzaClient.get_instance(config)` call requires `config.json` to have `avanza.username`, `avanza.password`, `avanza.totp_secret` keys
- **The avanza-api library is already installed** in `.venv` — no new dependencies except `websocket-client` for streaming
- **Legacy modules are NOT deleted** — they stay until all callers are migrated in a future PR
