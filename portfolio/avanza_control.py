"""Canonical Avanza control facade for reads, quotes, and browser-session trades.

Use this module as the shared import path for Avanza operations in strategy code.
It keeps the currently working Playwright-page execution path for metals/gold
while exposing the broader account/session helpers from ``portfolio.avanza_*``.
"""

from __future__ import annotations

import json
import logging

from portfolio.avanza_order_lock import avanza_order_lock

logger = logging.getLogger("portfolio.avanza_control")

from data.metals_avanza_helpers import (
    check_session_alive,
    get_csrf,
)
from data.metals_avanza_helpers import (
    fetch_account_cash as _fetch_account_cash,
)
from data.metals_avanza_helpers import (
    fetch_positions as _fetch_page_positions,
)
from data.metals_avanza_helpers import (
    fetch_price as _fetch_page_price,
)
from data.metals_avanza_helpers import (
    place_order as _place_page_order,
)
from data.metals_avanza_helpers import (
    place_stop_loss as _place_page_stop_loss,
)
from portfolio.avanza_client import (
    delete_order,
    find_instrument,
    get_account_id,
    get_open_orders,
    get_portfolio_value,
    get_positions,
    place_buy_order,
    place_sell_order,
)
from portfolio.avanza_client import (
    get_price as get_price_info,
)

_TYPE_ALIASES = {
    "cert": "certificate",
    "certifikat": "certificate",
    "certificate": "certificate",
    "warrant": "warrant",
    "mini": "warrant",
    "mini-future": "warrant",
    "mini_future": "warrant",
    "stock": "stock",
    "share": "stock",
    "fund": "fund",
    "etf": "exchange_traded_fund",
    "exchange_traded_fund": "exchange_traded_fund",
}

_PRICE_FALLBACK_TYPES = (
    "certificate",
    "warrant",
    "stock",
    "exchange_traded_fund",
    "fund",
)


def normalize_api_type(api_type: str | None, default: str = "certificate") -> str:
    """Normalize Avanza instrument type names for market-guide lookups."""
    normalized = (api_type or "").strip().lower()
    if not normalized:
        return default
    return _TYPE_ALIASES.get(normalized, normalized)


def fetch_price(page, orderbook_id: str, api_type: str = "certificate"):
    """Fetch a quote from the market-guide API using an authenticated page."""
    return _fetch_page_price(page, orderbook_id, normalize_api_type(api_type))


def fetch_price_with_fallback(page, orderbook_id: str, api_type: str | None = None):
    """Try the preferred market-guide type and then the common fallback types."""
    if not orderbook_id:
        return None

    candidates: list[str] = []
    preferred = normalize_api_type(api_type) if api_type else ""
    if preferred:
        candidates.append(preferred)
    for fallback in _PRICE_FALLBACK_TYPES:
        if fallback not in candidates:
            candidates.append(fallback)

    for candidate in candidates:
        data = fetch_price(page, orderbook_id, candidate)
        if not data:
            continue
        if data.get("bid") is None and data.get("ask") is None and data.get("last") is None:
            continue
        payload = dict(data)
        payload["api_type"] = candidate
        return payload
    return None


def fetch_account_cash(page, account_id: str | None = None):
    """Fetch buying power for an account via the authenticated browser session."""
    resolved_account_id = str(account_id or get_account_id())
    return _fetch_account_cash(page, resolved_account_id)


def fetch_page_positions(page, account_id: str | None = None):
    """Fetch current positions keyed by orderbook id via the page session.

    Returns dict[ob_id -> {name, units, value, avg_price, api_type}] on
    success, or None on transient failure. An empty dict `{}` is a valid
    response meaning the account is flat — callers should distinguish it
    from None.
    """
    resolved_account_id = str(account_id or get_account_id())
    return _fetch_page_positions(page, resolved_account_id)


def place_order(page, account_id: str | None, ob_id: str, side: str, price: float, volume: int):
    """Place a BUY/SELL order via the authenticated browser session."""
    resolved_account_id = str(account_id or get_account_id())
    normalized_side = (side or "").strip().upper()
    return _place_page_order(page, resolved_account_id, ob_id, normalized_side, price, volume)


def place_stop_loss(
    page,
    account_id: str | None,
    ob_id: str,
    trigger_price: float,
    sell_price: float,
    volume: int,
    valid_days: int = 8,
):
    """Place a hardware stop-loss order via the authenticated browser session."""
    resolved_account_id = str(account_id or get_account_id())
    return _place_page_stop_loss(
        page,
        resolved_account_id,
        ob_id,
        trigger_price,
        sell_price,
        volume,
        valid_days=valid_days,
    )


def delete_order_live(page, account_id: str | None, order_id: str):
    """Cancel an open order via the authenticated page session.

    IMPORTANT: Uses POST to /_api/trading-critical/rest/order/delete with
    JSON body {accountId, orderId}. The DELETE HTTP verb to
    /_api/trading-critical/rest/order/{accountId}/{orderId} returns 404
    (Avanza API change discovered 2026-03-24).
    """
    csrf = get_csrf(page)
    if not csrf:
        return False, {"error": "no CSRF token"}

    resolved_account_id = str(account_id or get_account_id())
    try:
        # 2026-04-13: cross-process order lock (see metals_avanza_helpers.place_order).
        with avanza_order_lock(op=f"delete_order_live/{order_id}"):
            result = page.evaluate(
                """async (args) => {
                    const [accountId, orderId, token] = args;
                    const resp = await fetch(
                        'https://www.avanza.se/_api/trading-critical/rest/order/delete',
                        {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                                'X-SecurityToken': token,
                            },
                            credentials: 'include',
                            body: JSON.stringify({accountId: accountId, orderId: orderId}),
                        }
                    );
                    return {status: resp.status, body: await resp.text()};
                }""",
                [resolved_account_id, order_id, csrf],
            )
        http_status = int(result.get("status") or 0)
        body_text = result.get("body", "")
        parsed = {}
        try:
            if body_text:
                parsed = json.loads(body_text)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        success = parsed.get("orderRequestStatus") == "SUCCESS"
        return success, {
            "http_status": http_status,
            "parsed": parsed,
            "body": body_text,
        }
    except Exception as exc:
        logger.error("Delete order failed for order %s: %s", order_id, exc, exc_info=True)
        return False, {"error": str(exc)}


def delete_stop_loss(page, account_id: str | None, stop_id: str):
    """Delete an existing Avanza stop-loss order via the authenticated page."""
    csrf = get_csrf(page)
    if not csrf:
        return False, {"error": "no CSRF token"}

    resolved_account_id = str(account_id or get_account_id())
    try:
        # 2026-04-13: cross-process order lock. SL delete is mutating.
        with avanza_order_lock(op=f"delete_stop_loss/{stop_id}"):
            result = page.evaluate(
                """async (args) => {
                    const [accountId, stopId, token] = args;
                    const resp = await fetch(
                        'https://www.avanza.se/_api/trading/stoploss/' + accountId + '/' + stopId,
                        {
                            method: 'DELETE',
                            headers: {'X-SecurityToken': token},
                            credentials: 'include',
                        }
                    );
                    return {status: resp.status, body: await resp.text()};
                }""",
                [resolved_account_id, stop_id, csrf],
            )
        http_status = int(result.get("status") or 0)
        # 2xx = deleted successfully.  404 = stop already gone (triggered/expired/cancelled).
        # Both mean the stop no longer exists, which is the goal of a cancel.
        success = (200 <= http_status < 300) or http_status == 404
        body_text = result.get("body", "")
        parsed = {}
        try:
            if body_text:
                parsed = json.loads(body_text)
        except (TypeError, json.JSONDecodeError):
            parsed = {}
        return success, {
            "http_status": http_status,
            "parsed": parsed,
            "body": body_text,
        }
    except Exception as exc:
        logger.error("Delete stop-loss failed for stop %s: %s", stop_id, exc, exc_info=True)
        return False, {"error": str(exc)}



# --- Page-free API (uses BankID session, no Playwright page needed) ---

from portfolio.avanza_session import (
    api_delete as _api_delete,
)
from portfolio.avanza_session import (
    api_get as _api_get,
)
from portfolio.avanza_session import (
    cancel_order as _cancel_order,
)
from portfolio.avanza_session import (
    place_buy_order as _place_buy_order,
)
from portfolio.avanza_session import (
    place_sell_order as _place_sell_order,
)
from portfolio.avanza_session import (
    place_stop_loss as _place_stop_loss_session,
)
from portfolio.avanza_session import (
    place_trailing_stop as _place_trailing_stop_session,
)
from portfolio.avanza_session import (
    verify_session,
)


def fetch_price_no_page(orderbook_id: str, api_type: str = "certificate"):
    """Fetch a quote without a Playwright page — uses BankID session API."""
    normalized = normalize_api_type(api_type)
    try:
        data = _api_get(f"/_api/market-guide/{normalized}/{orderbook_id}")
        quote = data.get("quote", {})
        ki = data.get("keyIndicators", {})
        underlying = data.get("underlying", {})
        def _v(obj):
            return obj.get("value") if isinstance(obj, dict) else obj
        return {
            "bid": _v(quote.get("buy")),
            "ask": _v(quote.get("sell")),
            "last": _v(quote.get("last")),
            "change_pct": _v(quote.get("changePercent")),
            "high": _v(quote.get("highest")),
            "low": _v(quote.get("lowest")),
            "underlying": _v(underlying.get("quote", {}).get("last")),
            "underlying_name": underlying.get("name"),
            "leverage": _v(ki.get("leverage")),
            "barrier": _v(ki.get("barrierLevel")),
            "api_type": normalized,
        }
    except Exception as e:
        logger.error("Warrant price fetch failed for orderbook %s: %s", orderbook_id, e, exc_info=True)
        return None


def fetch_price_no_page_with_fallback(orderbook_id: str, api_type: str | None = None):
    """Try preferred type then fallback chain — no Playwright page needed."""
    if not orderbook_id:
        return None
    candidates = []
    preferred = normalize_api_type(api_type) if api_type else ""
    if preferred:
        candidates.append(preferred)
    for fb in _PRICE_FALLBACK_TYPES:
        if fb not in candidates:
            candidates.append(fb)
    for candidate in candidates:
        data = fetch_price_no_page(orderbook_id, candidate)
        if data and (data.get("bid") is not None or data.get("ask") is not None or data.get("last") is not None):
            return data
    return None


def place_order_no_page(account_id, ob_id, side, price, volume):
    """Place BUY/SELL via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.

    Raises:
        ValueError: If *side* is not "BUY" or "SELL" (C2 fail-safe).
    """
    normalized_side = (side or "").strip().upper()
    if normalized_side not in ("BUY", "SELL"):
        raise ValueError(
            f"Invalid order side: {side!r} (must be 'BUY' or 'SELL')"
        )
    if normalized_side == "BUY":
        result = _place_buy_order(ob_id, price, volume, account_id)
    else:
        result = _place_sell_order(ob_id, price, volume, account_id)
    ok = result.get("orderRequestStatus") == "SUCCESS"
    return ok, result


def place_stop_loss_no_page(account_id, ob_id, trigger_price, sell_price, volume, valid_days=8):
    """Hardware stop-loss via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.
    """
    result = _place_stop_loss_session(ob_id, trigger_price, sell_price, volume, account_id, valid_days)
    ok = result.get("status") == "SUCCESS"
    return ok, result


def place_trailing_stop_no_page(account_id, ob_id, trail_percent, volume, valid_days=8):
    """Hardware trailing stop via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.
    """
    result = _place_trailing_stop_session(ob_id, trail_percent, volume, account_id, valid_days)
    ok = result.get("status") == "SUCCESS"
    return ok, result


def delete_order_no_page(account_id, order_id):
    """Cancel order via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.
    """
    result = _cancel_order(order_id, account_id)
    ok = result.get("orderRequestStatus") == "SUCCESS"
    return ok, result


def delete_stop_loss_no_page(account_id, stop_id):
    """Delete stop-loss via BankID session — no Playwright page needed.

    Returns:
        Tuple (ok: bool, result: dict) matching the page-based interface.
    """
    resolved_account_id = str(account_id or get_account_id())
    try:
        result = _api_delete(f"/_api/trading/stoploss/{resolved_account_id}/{stop_id}")
        # H18: Check for error indicators in the response.
        # API returns {} on success (200 with empty body).
        # A non-empty response with error keys indicates failure.
        if isinstance(result, dict) and result.get("errorCode"):
            logger.warning("Delete stop-loss returned error for stop %s: %s", stop_id, result)
            return False, result
        return True, result
    except Exception as e:
        logger.error("Delete stop-loss (no page) failed for stop %s: %s", stop_id, e, exc_info=True)
        return False, {"error": str(e)}


__all__ = [
    "check_session_alive",
    "delete_order",
    "delete_order_live",
    "delete_order_no_page",
    "delete_stop_loss",
    "delete_stop_loss_no_page",
    "fetch_account_cash",
    "fetch_page_positions",
    "fetch_price",
    "fetch_price_no_page",
    "fetch_price_no_page_with_fallback",
    "fetch_price_with_fallback",
    "find_instrument",
    "get_account_id",
    "get_csrf",
    "get_open_orders",
    "get_portfolio_value",
    "get_positions",
    "get_price_info",
    "normalize_api_type",
    "place_buy_order",
    "place_order",
    "place_order_no_page",
    "place_sell_order",
    "place_stop_loss",
    "place_stop_loss_no_page",
    "place_trailing_stop_no_page",
    "verify_session",
]
