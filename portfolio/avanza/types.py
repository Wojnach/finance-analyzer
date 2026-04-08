"""Typed response dataclasses for Avanza API data.

Avanza wraps many numeric values in ``{"value": X, "unit": "SEK", ...}``
objects.  The ``_val`` helper unwraps these transparently so callers always
get plain Python scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val(obj: Any, default: Any = None) -> Any:
    """Unwrap Avanza ``{value: X}`` wrappers, or return *obj* as-is.

    Handles:
      - ``{"value": 1.23, "unit": "SEK", ...}`` -> ``1.23``
      - plain scalars -> passed through
      - ``None`` / missing -> *default*
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        if "value" in obj:
            return obj["value"]
        return default
    return obj


def _ts(millis: Any) -> str:
    """Convert a millisecond Unix timestamp to an ISO-8601 string."""
    if millis is None:
        return ""
    if isinstance(millis, str):
        return millis
    try:
        return datetime.fromtimestamp(int(millis) / 1000, tz=UTC).isoformat()
    except (ValueError, TypeError, OSError):
        return str(millis)


# ---------------------------------------------------------------------------
# Quote & Market Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Quote:
    """Parsed quote snapshot."""

    bid: float
    ask: float
    last: float
    spread: float
    change_percent: float
    high: float
    low: float
    volume: float
    updated: str

    @classmethod
    def from_api(cls, raw: dict) -> Quote:
        bid = _val(raw.get("buy"), _val(raw.get("bid"), 0.0))
        ask = _val(raw.get("sell"), _val(raw.get("ask"), 0.0))
        last = _val(raw.get("last"), _val(raw.get("latest"), 0.0))
        spread = _val(raw.get("spread"))
        if spread is None:
            spread = round(ask - bid, 6) if (ask and bid) else 0.0
        change_percent = _val(raw.get("changePercent"), _val(raw.get("change_percent"), 0.0))
        high = _val(raw.get("highest"), _val(raw.get("high"), 0.0))
        low = _val(raw.get("lowest"), _val(raw.get("low"), 0.0))
        volume = _val(raw.get("totalVolumeTraded"), _val(raw.get("volume"), 0.0))
        updated = _ts(raw.get("updated", ""))
        return cls(
            bid=float(bid),
            ask=float(ask),
            last=float(last),
            spread=float(spread),
            change_percent=float(change_percent),
            high=float(high),
            low=float(low),
            volume=float(volume),
            updated=updated,
        )


@dataclass(frozen=True, slots=True)
class OrderDepthLevel:
    """One price level in the order book."""

    price: float
    volume: int

    @classmethod
    def from_api(cls, raw: dict) -> OrderDepthLevel:
        return cls(
            price=float(_val(raw.get("price"), 0.0)),
            volume=int(_val(raw.get("volume"), 0)),
        )


@dataclass(frozen=True, slots=True)
class Trade:
    """A single executed trade from the market-data feed."""

    price: float
    volume: int
    buyer: str
    seller: str
    time: str

    @classmethod
    def from_api(cls, raw: dict) -> Trade:
        return cls(
            price=float(_val(raw.get("price"), 0.0)),
            volume=int(_val(raw.get("volume"), 0)),
            buyer=str(raw.get("buyer", "")),
            seller=str(raw.get("seller", "")),
            time=str(raw.get("dealTime", raw.get("time", ""))),
        )


@dataclass(frozen=True, slots=True)
class MarketData:
    """Aggregated market data: quote + depth + recent trades."""

    quote: Quote
    bid_levels: tuple[OrderDepthLevel, ...]
    ask_levels: tuple[OrderDepthLevel, ...]
    recent_trades: tuple[Trade, ...]
    market_maker_expected: bool

    @classmethod
    def from_api(cls, raw: dict) -> MarketData:
        # Quote
        quote_raw = raw.get("quote", {})
        quote = Quote.from_api(quote_raw)

        # Order depth
        depth = raw.get("orderDepth", {})
        levels = depth.get("levels", [])
        bid_levels: list[OrderDepthLevel] = []
        ask_levels: list[OrderDepthLevel] = []
        for lvl in levels:
            buy_side = lvl.get("buySide", {})
            sell_side = lvl.get("sellSide", {})
            if buy_side:
                bid_levels.append(OrderDepthLevel.from_api(buy_side))
            if sell_side:
                ask_levels.append(OrderDepthLevel.from_api(sell_side))

        # Trades
        trades_raw = raw.get("trades", [])
        trades = tuple(Trade.from_api(t) for t in trades_raw)

        mm = depth.get("marketMakerExpected", False)

        return cls(
            quote=quote,
            bid_levels=tuple(bid_levels),
            ask_levels=tuple(ask_levels),
            recent_trades=trades,
            market_maker_expected=bool(mm),
        )


# ---------------------------------------------------------------------------
# Order / StopLoss results
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OrderResult:
    """Result of placing or deleting an order."""

    success: bool
    order_id: str
    status: str
    message: str

    @classmethod
    def from_api(cls, raw: dict) -> OrderResult:
        status = raw.get("orderRequestStatus", raw.get("status", ""))
        success = str(status).upper() == "SUCCESS"
        order_id = str(raw.get("orderId", raw.get("order_id", "")))
        message = raw.get("message", raw.get("messages", ""))
        if isinstance(message, list):
            message = "; ".join(str(m) for m in message)
        return cls(
            success=success,
            order_id=order_id,
            status=str(status),
            message=str(message),
        )


@dataclass(frozen=True, slots=True)
class StopLossResult:
    """Result of placing or modifying a stop-loss."""

    success: bool
    stop_id: str
    status: str

    @classmethod
    def from_api(cls, raw: dict) -> StopLossResult:
        status = raw.get("status", raw.get("orderRequestStatus", ""))
        success = str(status).upper() in ("SUCCESS", "OK", "ACTIVE")
        stop_id = str(raw.get("stoplossOrderId", raw.get("stopLossId", raw.get("stop_id", raw.get("id", "")))))
        return cls(
            success=success,
            stop_id=stop_id,
            status=str(status),
        )


# ---------------------------------------------------------------------------
# Account & Portfolio
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Position:
    """A single instrument position within an account."""

    name: str
    orderbook_id: str
    instrument_type: str
    volume: float
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
        instrument = raw.get("instrument", {})
        orderbook = instrument.get("orderbook", {})
        account = raw.get("account", {})
        perf = raw.get("lastTradingDayPerformance", {})

        # Quote values from the orderbook sub-object
        ob_quote = orderbook.get("quote", {})
        latest = _val(ob_quote.get("latest"), _val(ob_quote.get("last"), 0.0))
        change_pct = _val(ob_quote.get("changePercent"), _val(ob_quote.get("change_percent"), 0.0))

        return cls(
            name=orderbook.get("name", instrument.get("name", "")),
            orderbook_id=str(orderbook.get("id", raw.get("id", ""))),
            instrument_type=instrument.get("type", orderbook.get("type", "")),
            volume=float(_val(raw.get("volume"), 0.0)),
            value=float(_val(raw.get("value"), 0.0)),
            acquired_value=float(_val(raw.get("acquiredValue"), 0.0)),
            profit=float(_val(perf.get("absolute"), 0.0)),
            profit_percent=float(_val(perf.get("relative"), 0.0)),
            last_price=float(latest),
            change_percent=float(change_pct),
            account_id=str(account.get("id", "")),
            currency=instrument.get("currency", ""),
        )


@dataclass(frozen=True, slots=True)
class Order:
    """An open or filled order."""

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
            orderbook_id=str(raw.get("orderBookId", raw.get("orderbookId", raw.get("orderbook_id", "")))),
            side=str(raw.get("orderType", raw.get("side", ""))),
            price=float(_val(raw.get("price"), 0.0)),
            volume=int(_val(raw.get("volume"), 0)),
            status=str(raw.get("status", raw.get("statusDescription", ""))),
            account_id=str(raw.get("accountId", raw.get("account_id", ""))),
        )


@dataclass(frozen=True, slots=True)
class Deal:
    """A completed deal (execution)."""

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
            orderbook_id=str(raw.get("orderBookId", raw.get("orderbookId", ""))),
            side=str(raw.get("orderType", raw.get("side", ""))),
            price=float(_val(raw.get("price"), 0.0)),
            volume=int(_val(raw.get("volume"), 0)),
            time=str(raw.get("dealTime", raw.get("time", ""))),
            account_id=str(raw.get("accountId", raw.get("account_id", ""))),
        )


@dataclass(frozen=True, slots=True)
class StopLoss:
    """An active stop-loss order."""

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
        trigger = raw.get("trigger", {})
        order_event = raw.get("orderEvent", raw.get("order", {}))
        return cls(
            stop_id=str(raw.get("id", raw.get("stopLossId", ""))),
            orderbook_id=str(raw.get("orderBookId", raw.get("orderbookId", ""))),
            trigger_price=float(_val(trigger.get("value"), _val(raw.get("triggerPrice"), 0.0))),
            trigger_type=str(trigger.get("type", raw.get("triggerType", "LAST_PRICE"))),
            sell_price=float(_val(order_event.get("price"), _val(raw.get("sellPrice"), 0.0))),
            volume=int(_val(order_event.get("volume"), _val(raw.get("volume"), 0))),
            status=str(raw.get("status", "")),
            account_id=str(raw.get("accountId", raw.get("account_id", ""))),
        )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SearchHit:
    """Instrument search result."""

    orderbook_id: str
    name: str
    instrument_type: str
    tradeable: bool
    last_price: float
    change_percent: float

    @classmethod
    def from_api(cls, raw: dict) -> SearchHit:
        return cls(
            orderbook_id=str(raw.get("id", raw.get("orderbookId", ""))),
            name=str(raw.get("name", "")),
            instrument_type=str(raw.get("instrumentType", raw.get("type", ""))),
            tradeable=bool(raw.get("tradable", raw.get("tradeable", False))),
            last_price=float(_val(raw.get("lastPrice"), _val(raw.get("last_price"), 0.0))),
            change_percent=float(_val(raw.get("changePercent"), _val(raw.get("change_percent"), 0.0))),
        )


# ---------------------------------------------------------------------------
# Tick Table
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TickEntry:
    """One row from the tick-size table."""

    min_price: float
    max_price: float
    tick_size: float

    @classmethod
    def from_api(cls, raw: dict) -> TickEntry:
        return cls(
            min_price=float(raw.get("min", raw.get("minPrice", 0.0))),
            max_price=float(raw.get("max", raw.get("maxPrice", 0.0))),
            tick_size=float(raw.get("tick", raw.get("tickSize", raw.get("tick_size", 0.0)))),
        )


# ---------------------------------------------------------------------------
# OHLC
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class OHLC:
    """A single OHLCV candle."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    @classmethod
    def from_api(cls, raw: dict) -> OHLC:
        return cls(
            timestamp=_ts(raw.get("timestamp")),
            open=float(raw.get("open", 0.0)),
            high=float(raw.get("high", 0.0)),
            low=float(raw.get("low", 0.0)),
            close=float(raw.get("close", 0.0)),
            volume=int(raw.get("totalVolumeTraded", raw.get("volume", 0))),
        )


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AccountCash:
    """Account-level cash info."""

    buying_power: float
    total_value: float
    own_capital: float

    @classmethod
    def from_api(cls, raw: dict) -> AccountCash:
        return cls(
            buying_power=float(_val(raw.get("buyingPower"), 0.0)),
            total_value=float(_val(raw.get("totalValue"), 0.0)),
            own_capital=float(_val(raw.get("ownCapital"), _val(raw.get("buyingPowerWithoutCredit"), 0.0))),
        )


# ---------------------------------------------------------------------------
# Transaction
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Transaction:
    """A historical account transaction."""

    transaction_id: str
    transaction_type: str
    instrument_name: str
    amount: float
    price: float
    volume: float
    date: str
    account_id: str

    @classmethod
    def from_api(cls, raw: dict) -> Transaction:
        account = raw.get("account", {})
        return cls(
            transaction_id=str(raw.get("id", "")),
            transaction_type=str(raw.get("type", "")),
            instrument_name=str(raw.get("instrumentName", raw.get("description", ""))),
            amount=float(_val(raw.get("amount"), 0.0)),
            price=float(_val(raw.get("priceInTradedCurrency"), _val(raw.get("price"), 0.0))),
            volume=float(_val(raw.get("volume"), 0.0)),
            date=str(raw.get("date", raw.get("tradeDate", ""))),
            account_id=str(account.get("id", raw.get("accountId", ""))),
        )


# ---------------------------------------------------------------------------
# Instrument Info
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class InstrumentInfo:
    """Core instrument metadata (works for certificates, warrants, stocks)."""

    orderbook_id: str
    name: str
    instrument_type: str
    currency: str
    leverage: float
    barrier: float
    underlying_name: str
    underlying_price: float

    @classmethod
    def from_api(cls, raw: dict) -> InstrumentInfo:
        return cls(
            orderbook_id=str(raw.get("id", raw.get("orderbookId", ""))),
            name=str(raw.get("name", "")),
            instrument_type=str(raw.get("instrumentType", raw.get("type", ""))),
            currency=str(raw.get("currency", "")),
            leverage=float(_val(raw.get("leverage"), 0.0)),
            barrier=float(_val(raw.get("barrier"), _val(raw.get("barrierLevel"), 0.0))),
            underlying_name=str(raw.get("underlyingName", raw.get("underlying", {}).get("name", ""))),
            underlying_price=float(_val(
                raw.get("underlyingPrice"),
                _val(raw.get("underlying", {}).get("price"), 0.0),
            )),
        )


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NewsArticle:
    """A single news article linked to an instrument."""

    article_id: str
    headline: str
    date: str
    source: str

    @classmethod
    def from_api(cls, raw: dict) -> NewsArticle:
        return cls(
            article_id=str(raw.get("id", raw.get("articleId", ""))),
            headline=str(raw.get("headline", "")),
            date=_ts(raw.get("timePublishedMillis", raw.get("date", raw.get("timePublished", "")))),
            source=str(raw.get("newsSource", raw.get("source", ""))),
        )
