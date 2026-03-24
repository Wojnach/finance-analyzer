"""Canonical Avanza control facade for reads, quotes, and browser-session trades.

Use this module as the shared import path for Avanza operations in strategy code.
It keeps the currently working Playwright-page execution path for metals/gold
while exposing the broader account/session helpers from ``portfolio.avanza_*``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from data.metals_avanza_helpers import (
    check_session_alive,
    fetch_account_cash as _fetch_account_cash,
    fetch_price as _fetch_page_price,
    get_csrf,
    place_order as _place_page_order,
    place_stop_loss as _place_page_stop_loss,
)
from portfolio.avanza_client import (
    delete_order,
    find_instrument,
    get_account_id,
    get_open_orders,
    get_portfolio_value,
    get_positions,
    get_price as get_price_info,
    place_buy_order,
    place_sell_order,
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


def normalize_api_type(api_type: Optional[str], default: str = "certificate") -> str:
    """Normalize Avanza instrument type names for market-guide lookups."""
    normalized = (api_type or "").strip().lower()
    if not normalized:
        return default
    return _TYPE_ALIASES.get(normalized, normalized)


def fetch_price(page, orderbook_id: str, api_type: str = "certificate"):
    """Fetch a quote from the market-guide API using an authenticated page."""
    return _fetch_page_price(page, orderbook_id, normalize_api_type(api_type))


def fetch_price_with_fallback(page, orderbook_id: str, api_type: Optional[str] = None):
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


def fetch_account_cash(page, account_id: Optional[str] = None):
    """Fetch buying power for an account via the authenticated browser session."""
    resolved_account_id = str(account_id or get_account_id())
    return _fetch_account_cash(page, resolved_account_id)


def place_order(page, account_id: Optional[str], ob_id: str, side: str, price: float, volume: int):
    """Place a BUY/SELL order via the authenticated browser session."""
    resolved_account_id = str(account_id or get_account_id())
    normalized_side = (side or "").strip().upper()
    return _place_page_order(page, resolved_account_id, ob_id, normalized_side, price, volume)


def place_stop_loss(
    page,
    account_id: Optional[str],
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


def delete_order_live(page, account_id: Optional[str], order_id: str):
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
        return False, {"error": str(exc)}


def delete_stop_loss(page, account_id: Optional[str], stop_id: str):
    """Delete an existing Avanza stop-loss order via the authenticated page."""
    csrf = get_csrf(page)
    if not csrf:
        return False, {"error": "no CSRF token"}

    resolved_account_id = str(account_id or get_account_id())
    try:
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
        return False, {"error": str(exc)}


__all__ = [
    "check_session_alive",
    "delete_order",
    "delete_order_live",
    "delete_stop_loss",
    "fetch_account_cash",
    "fetch_price",
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
    "place_sell_order",
    "place_stop_loss",
]
