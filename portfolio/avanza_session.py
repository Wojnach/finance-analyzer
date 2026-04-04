"""Avanza session management — load, validate, and use BankID-captured sessions.

Uses Playwright's saved storage state to make authenticated API calls via a
headless browser context. This ensures cookies and TLS session match what
Avanza expects (replaying cookies via requests library causes 401s).

This is the preferred auth method until TOTP credentials are configured.
"""

import json
import logging
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from portfolio.file_utils import load_json

logger = logging.getLogger("portfolio.avanza_session")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SESSION_FILE = DATA_DIR / "avanza_session.json"
STORAGE_STATE_FILE = DATA_DIR / "avanza_storage_state.json"
API_BASE = "https://www.avanza.se"

# Minimum remaining session life before we consider it expired (minutes)
EXPIRY_BUFFER_MINUTES = 30

# Default trading account
DEFAULT_ACCOUNT_ID = "1625505"

# Module-level Playwright context (lazy-initialized, reused across calls)
# BUG-129: Protected by _pw_lock to prevent concurrent access corruption
_pw_lock = threading.Lock()
_pw_instance = None
_pw_browser = None
_pw_context = None


class AvanzaSessionError(Exception):
    """Raised when session is missing, expired, or invalid."""


def load_session() -> dict:
    """Load saved BankID session metadata from disk.

    Returns:
        Session dict with expiry info, customer_id, etc.

    Raises:
        AvanzaSessionError: if file missing, unreadable, or expired.
    """
    if not SESSION_FILE.exists():
        raise AvanzaSessionError(
            f"No session file found at {SESSION_FILE}. "
            "Run: python scripts/avanza_login.py"
        )

    data = load_json(SESSION_FILE)
    if data is None:
        raise AvanzaSessionError(f"Failed to read session file: {SESSION_FILE}")

    # Check expiry
    expires_at = data.get("expires_at")
    if expires_at:
        try:
            exp = datetime.fromisoformat(expires_at)
            now = datetime.now(UTC)
            if exp <= now:
                raise AvanzaSessionError(
                    f"Session expired at {expires_at}. "
                    "Run: python scripts/avanza_login.py"
                )
        except ValueError:
            pass  # Can't parse expiry, proceed anyway

    if not STORAGE_STATE_FILE.exists():
        raise AvanzaSessionError(
            f"No storage state file at {STORAGE_STATE_FILE}. "
            "Run: python scripts/avanza_login.py"
        )

    return data


def session_remaining_minutes() -> float | None:
    """Get minutes remaining on the current session, or None if no session."""
    try:
        data = load_json(SESSION_FILE)
        if data is None:
            return None
        expires_at = data.get("expires_at")
        if not expires_at:
            return None
        exp = datetime.fromisoformat(expires_at)
        now = datetime.now(UTC)
        return (exp - now).total_seconds() / 60.0
    except Exception as e:
        logger.debug("Failed to compute session minutes remaining: %s", e)
        return None


def is_session_expiring_soon(threshold_minutes: float = 60.0) -> bool:
    """Check if session will expire within the given threshold.

    Returns True if session is expired, expiring soon, or doesn't exist.
    """
    remaining = session_remaining_minutes()
    if remaining is None:
        return True
    return remaining < threshold_minutes


def _get_playwright_context():
    """Get or create a headless Playwright browser context with saved auth state."""
    global _pw_instance, _pw_browser, _pw_context

    with _pw_lock:
        if _pw_context is not None:
            return _pw_context

        # Validate session first
        load_session()

        from playwright.sync_api import sync_playwright

        _pw_instance = sync_playwright().start()
        _pw_browser = _pw_instance.chromium.launch(headless=True)
        _pw_context = _pw_browser.new_context(
            storage_state=str(STORAGE_STATE_FILE),
            locale="sv-SE",
        )
        return _pw_context


def close_playwright():
    """Clean up Playwright resources."""
    global _pw_instance, _pw_browser, _pw_context
    with _pw_lock:
        if _pw_context:
            try:
                _pw_context.close()
            except Exception as e:
                logger.debug("Context close failed: %s", e)
            _pw_context = None
        if _pw_browser:
            try:
                _pw_browser.close()
            except Exception as e:
                logger.debug("Browser close failed: %s", e)
            _pw_browser = None
        if _pw_instance:
            try:
                _pw_instance.stop()
            except Exception as e:
                logger.debug("Playwright stop failed: %s", e)
            _pw_instance = None


def verify_session() -> bool:
    """Verify that the session is valid by making a lightweight API call.

    Returns:
        True if session is valid, False otherwise.
    """
    try:
        ctx = _get_playwright_context()
        resp = ctx.request.get(f"{API_BASE}/_api/position-data/positions")
        return resp.ok
    except Exception as e:
        logger.warning("Session verification failed: %s", e)
        close_playwright()
        return False


# --- API convenience functions ---


def api_get(path: str, **kwargs) -> Any:
    """Make an authenticated GET request to Avanza API.

    Args:
        path: API path (e.g., "/_api/position-data/positions")

    Returns:
        Parsed JSON response.

    Raises:
        AvanzaSessionError: if session is invalid.
    """
    ctx = _get_playwright_context()
    url = f"{API_BASE}{path}" if path.startswith("/") else path
    resp = ctx.request.get(url)
    if resp.status == 401:
        close_playwright()
        raise AvanzaSessionError(
            "Session returned 401 Unauthorized. "
            "Run: python scripts/avanza_login.py"
        )
    if not resp.ok:
        raise RuntimeError(f"Avanza API error {resp.status}: {resp.text()[:500]}")
    return resp.json()


def _get_csrf() -> str:
    """Extract CSRF token from Playwright context cookies."""
    ctx = _get_playwright_context()
    for c in ctx.cookies():
        if c["name"] == "AZACSRF":
            return c["value"]
    raise AvanzaSessionError("No AZACSRF cookie found — session may be invalid")


def api_post(path: str, payload: dict) -> Any:
    """Make an authenticated POST request to Avanza API.

    Automatically includes the X-SecurityToken (CSRF) header.

    Args:
        path: API path (e.g., "/_api/trading-critical/rest/order/new")
        payload: Request body dict.

    Returns:
        Parsed JSON response.
    """
    ctx = _get_playwright_context()
    csrf = _get_csrf()
    url = f"{API_BASE}{path}" if path.startswith("/") else path
    resp = ctx.request.post(
        url,
        data=json.dumps(payload),
        headers={
            "Content-Type": "application/json",
            "X-SecurityToken": csrf,
        },
    )
    if resp.status == 401:
        close_playwright()
        raise AvanzaSessionError(
            "Session returned 401 Unauthorized. "
            "Run: python scripts/avanza_login.py"
        )
    if resp.status == 403:
        close_playwright()
        raise AvanzaSessionError(
            "Session returned 403 Forbidden — CSRF token may be stale. "
            "Run: python scripts/avanza_login.py"
        )
    body = resp.text()
    try:
        return json.loads(body)
    except (json.JSONDecodeError, TypeError):
        if not resp.ok:
            raise RuntimeError(f"Avanza API error {resp.status}: {body[:500]}") from None
        return {"raw": body}


def api_delete(path: str) -> Any:
    """Make an authenticated DELETE request to Avanza API.

    Automatically includes the X-SecurityToken (CSRF) header.

    Args:
        path: API path (e.g., "/_api/trading/stoploss/{stop_id}")

    Returns:
        Dict with ``http_status`` and ``ok`` keys.
    """
    ctx = _get_playwright_context()
    csrf = _get_csrf()
    url = f"{API_BASE}{path}" if path.startswith("/") else path
    resp = ctx.request.delete(
        url,
        headers={
            "Content-Type": "application/json",
            "X-SecurityToken": csrf,
        },
    )
    if resp.status == 401:
        close_playwright()
        raise AvanzaSessionError(
            "Session returned 401 Unauthorized. "
            "Run: python scripts/avanza_login.py"
        )
    return {"http_status": resp.status, "ok": 200 <= resp.status < 300 or resp.status == 404}


# --- Trading convenience functions ---


def get_buying_power(account_id: str | None = None) -> dict:
    """Get buying power and account value for an account.

    Returns:
        Dict with buying_power, total_value, own_capital (all in SEK).
        Returns empty dict on failure.
    """
    aid = str(account_id or DEFAULT_ACCOUNT_ID)
    data = api_get("/_api/account-overview/overview/categorizedAccounts")
    for cat in data.get("categories", []):
        for acc in cat.get("accounts", []):
            if str(acc.get("id", "")) == aid:
                def _v(obj):
                    return obj.get("value", 0) if isinstance(obj, dict) else (obj or 0)
                return {
                    "buying_power": _v(acc.get("buyingPower", {})),
                    "total_value": _v(acc.get("totalValue", {})),
                    "own_capital": _v(acc.get("ownCapital", {})),
                }
    # Account not found by id — try matching with categorizedAccounts
    # (structure may nest accounts differently across Avanza updates)
    total = data.get("categories", [{}])[0].get("totalValue", {})
    total_val = total.get("value", 0) if isinstance(total, dict) else 0
    positions = get_positions()
    pos_val = sum(p.get("value", 0) for p in positions if str(p.get("account_id")) == aid)
    return {
        "buying_power": round(total_val - pos_val, 2),
        "total_value": total_val,
        "own_capital": total_val,
    }


def place_buy_order(
    orderbook_id: str,
    price: float,
    volume: int,
    account_id: str | None = None,
    valid_until: str | None = None,
) -> dict:
    """Place a limit BUY order on Avanza.

    Args:
        orderbook_id: Avanza orderbook ID.
        price: Limit price in SEK.
        volume: Number of units (int >= 1).
        account_id: Defaults to DEFAULT_ACCOUNT_ID.
        valid_until: ISO date string. Defaults to today (day order).

    Returns:
        Dict with orderRequestStatus, orderId, message.
    """
    return _place_order("BUY", orderbook_id, price, volume, account_id, valid_until)


def place_sell_order(
    orderbook_id: str,
    price: float,
    volume: int,
    account_id: str | None = None,
    valid_until: str | None = None,
) -> dict:
    """Place a limit SELL order on Avanza."""
    return _place_order("SELL", orderbook_id, price, volume, account_id, valid_until)


def _place_order(
    side: str,
    orderbook_id: str,
    price: float,
    volume: int,
    account_id: str | None = None,
    valid_until: str | None = None,
) -> dict:
    """Internal: place a BUY or SELL limit order."""
    if volume < 1:
        raise ValueError(f"volume must be >= 1, got {volume}")
    if price <= 0:
        raise ValueError(f"price must be > 0, got {price}")

    payload = {
        "accountId": str(account_id or DEFAULT_ACCOUNT_ID),
        "orderbookId": str(orderbook_id),
        "side": side,
        "condition": "NORMAL",
        "price": price,
        "validUntil": valid_until or date.today().isoformat(),
        "volume": volume,
    }
    result = api_post("/_api/trading-critical/rest/order/new", payload)
    status = result.get("orderRequestStatus", "UNKNOWN")
    if status != "SUCCESS":
        logger.warning("Order %s failed: %s — %s", side, status, result.get("message", ""))
    else:
        logger.info(
            "Order %s placed: %dx @ %.3f SEK (id=%s)",
            side, volume, price, result.get("orderId", "?"),
        )
    return result


def cancel_order(order_id: str, account_id: str | None = None) -> dict:
    """Cancel an open order.

    IMPORTANT: Uses POST (not DELETE verb) — Avanza API change 2026-03-24.
    """
    payload = {
        "accountId": str(account_id or DEFAULT_ACCOUNT_ID),
        "orderId": str(order_id),
    }
    return api_post("/_api/trading-critical/rest/order/delete", payload)


def get_open_orders(account_id: str | None = None) -> list[dict]:
    """Get all open (unfilled) orders for an account."""
    aid = str(account_id or DEFAULT_ACCOUNT_ID)
    try:
        data = api_get(f"/_api/trading/rest/order/account/{aid}")
        if isinstance(data, list):
            return data
        return data.get("orders", data.get("openOrders", []))
    except RuntimeError:
        # Endpoint may vary — fallback to deal endpoint
        try:
            data = api_get("/_api/trading/rest/deals-and-orders")
            orders = data.get("orders", [])
            return [o for o in orders if str(o.get("accountId", "")) == aid]
        except RuntimeError:
            logger.debug("Could not fetch open orders")
            return []


def get_quote(orderbook_id: str) -> dict:
    """Get bid/ask/last quote for an instrument. Fast single-endpoint call.

    Returns:
        Dict with buy, sell, last, changePercent, highest, lowest.
    """
    return api_get(f"/_api/market-guide/stock/{orderbook_id}/quote")


def get_positions() -> list[dict]:
    """Get all positions via session-based auth.

    Returns:
        List of position dicts with name, value, profit, etc.
    """
    data = api_get("/_api/position-data/positions")
    positions = []
    for entry in data.get("withOrderbook", []):
        inst = entry.get("instrument", {})
        orderbook = inst.get("orderbook", {})
        quote = orderbook.get("quote", {})
        volume_obj = entry.get("volume", {})
        value_obj = entry.get("value", {})
        acquired_obj = entry.get("acquiredValue", {})
        account = entry.get("account", {})

        vol = volume_obj.get("value", 0) if isinstance(volume_obj, dict) else volume_obj
        val = value_obj.get("value", 0) if isinstance(value_obj, dict) else value_obj
        acq = acquired_obj.get("value", 0) if isinstance(acquired_obj, dict) else acquired_obj
        latest = quote.get("latest", {})
        last_price = latest.get("value", 0) if isinstance(latest, dict) else latest
        change_pct_obj = quote.get("changePercent", {})
        change_pct = change_pct_obj.get("value", 0) if isinstance(change_pct_obj, dict) else change_pct_obj

        positions.append({
            "name": inst.get("name", orderbook.get("name", "")),
            "orderbook_id": str(orderbook.get("id", "")),
            "instrument_id": str(inst.get("id", "")),
            "type": inst.get("type", orderbook.get("type", "")),
            "volume": vol,
            "value": val,
            "acquired_value": acq,
            "profit": val - acq if val and acq else 0,
            "profit_percent": ((val - acq) / acq * 100) if acq else 0,
            "currency": inst.get("currency", "SEK"),
            "last_price": last_price,
            "change_percent": change_pct,
            "account_id": account.get("id", ""),
            "account_type": account.get("type", ""),
        })
    return positions


def place_stop_loss(
    orderbook_id: str,
    trigger_price: float,
    sell_price: float,
    volume: int,
    account_id: str | None = None,
    valid_days: int = 8,
    trigger_type: str = "LESS_OR_EQUAL",
    value_type: str = "MONETARY",
) -> dict:
    """Place a hardware stop-loss order on Avanza.

    IMPORTANT: Uses /_api/trading/stoploss/new, NOT the regular order API.

    Args:
        orderbook_id: Avanza orderbook ID.
        trigger_price: Price at which to trigger the stop-loss.
            For FOLLOW_DOWNWARDS with PERCENTAGE, this is the trail %.
        sell_price: Price to sell at when triggered.
            For trailing stops (FOLLOW_DOWNWARDS), set to 0 (market).
        volume: Number of units to sell.
        account_id: Defaults to DEFAULT_ACCOUNT_ID.
        valid_days: Days until the stop-loss expires (default 8).
        trigger_type: LESS_OR_EQUAL, MORE_OR_EQUAL, FOLLOW_DOWNWARDS, FOLLOW_UPWARDS.
        value_type: MONETARY (absolute price) or PERCENTAGE.

    Returns:
        Dict with status, stoplossOrderId.
    """
    acct = str(account_id or DEFAULT_ACCOUNT_ID)
    valid_until = (date.today() + timedelta(days=valid_days)).isoformat()

    payload = {
        "parentStopLossId": "0",
        "accountId": acct,
        "orderBookId": str(orderbook_id),
        "stopLossTrigger": {
            "type": trigger_type,
            "value": trigger_price,
            "validUntil": valid_until,
            "valueType": value_type,
            "triggerOnMarketMakerQuote": True,
        },
        "stopLossOrderEvent": {
            "type": "SELL",
            "price": sell_price,
            "volume": volume,
            "validDays": valid_days,
            "priceType": value_type,
            "shortSellingAllowed": False,
        },
    }
    result = api_post("/_api/trading/stoploss/new", payload)
    status = result.get("status", "UNKNOWN")
    if status == "SUCCESS":
        logger.info(
            "Stop-loss placed: %s trigger=%.3f sell=%.3f vol=%d (id=%s)",
            trigger_type, trigger_price, sell_price, volume,
            result.get("stoplossOrderId", "?"),
        )
    else:
        logger.warning("Stop-loss failed: %s — %s", status, result)
    return result


def place_trailing_stop(
    orderbook_id: str,
    trail_percent: float,
    volume: int,
    account_id: str | None = None,
    valid_days: int = 8,
) -> dict:
    """Place a hardware trailing stop-loss that Avanza manages automatically.

    The stop follows the price downward by trail_percent%. If the instrument
    drops trail_percent% from its peak since placement, the stop triggers a
    market sell.

    Args:
        orderbook_id: Avanza orderbook ID.
        trail_percent: Trailing distance as percentage (e.g. 5.0 for 5%).
        volume: Number of units to sell.
        account_id: Defaults to DEFAULT_ACCOUNT_ID.
        valid_days: Days until the stop expires (default 8).

    Returns:
        Dict with status, stoplossOrderId.
    """
    return place_stop_loss(
        orderbook_id=orderbook_id,
        trigger_price=trail_percent,
        sell_price=0,
        volume=volume,
        account_id=account_id,
        valid_days=valid_days,
        trigger_type="FOLLOW_DOWNWARDS",
        value_type="PERCENTAGE",
    )


def get_stop_losses() -> list[dict]:
    """Get all active stop-loss orders."""
    try:
        data = api_get("/_api/trading/stoploss")
        return data if isinstance(data, list) else []
    except RuntimeError:
        logger.debug("Could not fetch stop-losses")
        return []


def get_instrument_price(orderbook_id: str) -> dict[str, Any]:
    """Get price info for a specific instrument.

    Args:
        orderbook_id: Avanza orderbook ID (numeric string)

    Returns:
        Dict with lastPrice, changePercent, etc.
    """
    # Try stock first, then fund, then certificate/warrant
    for instrument_type in ("stock", "certificate", "fund", "exchange_traded_fund"):
        try:
            data = api_get(
                f"/_api/market-guide/{instrument_type}/{orderbook_id}",
            )
            return data
        except Exception as e:
            logger.debug("Market guide lookup failed for %s/%s: %s", instrument_type, orderbook_id, e)
            continue

    # Fallback: generic orderbook endpoint
    return api_get(f"/_api/orderbook/{orderbook_id}")
