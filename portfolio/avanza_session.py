"""Avanza session management — load, validate, and use BankID-captured sessions.

Uses Playwright's saved storage state to make authenticated API calls via a
headless browser context. This ensures cookies and TLS session match what
Avanza expects (replaying cookies via requests library causes 401s).

This is the preferred auth method until TOTP credentials are configured.
"""

import json
import logging
import threading
import time
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
        logger.warning("Failed to compute session minutes remaining: %s", e)
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
            logger.warning("Could not fetch open orders")
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
    """Get all active stop-loss orders.

    Returns ``[]`` on read failure for backward compatibility with
    callers that treat empty as "nothing to monitor". Code that needs
    to distinguish "no stops" from "could not read stops" must use
    :func:`get_stop_losses_strict` instead — or a False return from
    that function will leave the caller unable to make safety
    decisions like cancel-before-sell.
    """
    try:
        data = api_get("/_api/trading/stoploss")
        return data if isinstance(data, list) else []
    except RuntimeError:
        logger.warning("Could not fetch stop-losses")
        return []


def get_stop_losses_strict() -> list[dict]:
    """Get all active stop-loss orders, raising on any read failure.

    Use this in safety-critical paths (e.g., before a sell) where
    "could not read" must NOT be silently treated as "no stops exist".
    A swallowed read error there would let the dependent sell proceed
    against still-encumbered volume, producing the very
    ``short.sell.not.allowed`` error this module exists to prevent.

    Raises:
        RuntimeError: if the underlying ``api_get`` call fails or
            returns a non-list shape.
    """
    data = api_get("/_api/trading/stoploss")
    if not isinstance(data, list):
        raise RuntimeError(
            f"Unexpected stop-loss response shape: {type(data).__name__}"
        )
    return data


def cancel_stop_loss(stop_id: str, account_id: str | None = None) -> dict:
    """Cancel a single stop-loss order by ID.

    Idempotent: HTTP 404 (already gone) is treated as success since the
    end-state is identical from the caller's perspective.

    Uses DELETE /_api/trading/stoploss/{accountId}/{stopId}, which is the
    correct endpoint per portfolio/avanza_control.py:206. Do NOT use the
    regular order cancel API — it returns "crossing prices" errors for
    stop-losses (March 3 incident).

    Args:
        stop_id: Avanza stop-loss ID (e.g. "A2^1773297348702^1346781").
        account_id: Avanza account ID. Defaults to ``DEFAULT_ACCOUNT_ID``.

    Returns:
        Dict with keys ``status`` ("SUCCESS"/"FAILED"), ``http_status`` (int),
        and ``stop_id`` (str). Errors that prevent the call from running
        (network, missing CSRF, etc.) yield ``status="FAILED"`` with
        ``http_status=0`` and an ``error`` key describing the cause.
    """
    if not stop_id:
        return {"status": "FAILED", "http_status": 0, "stop_id": "", "error": "empty stop_id"}
    acct = str(account_id or DEFAULT_ACCOUNT_ID)
    try:
        result = api_delete(f"/_api/trading/stoploss/{acct}/{stop_id}")
    except Exception as exc:  # noqa: BLE001 — propagate as structured failure
        logger.error("cancel_stop_loss(%s) raised: %s", stop_id, exc, exc_info=True)
        return {"status": "FAILED", "http_status": 0, "stop_id": stop_id, "error": str(exc)}
    http_status = int(result.get("http_status", 0)) if isinstance(result, dict) else 0
    # 2xx = deleted; 404 = already gone (triggered/expired/cancelled). Both succeed.
    ok = (200 <= http_status < 300) or http_status == 404
    if ok:
        logger.info("cancel_stop_loss(%s) -> %s", stop_id, http_status)
    else:
        logger.warning("cancel_stop_loss(%s) failed: http=%s result=%s", stop_id, http_status, result)
    return {
        "status": "SUCCESS" if ok else "FAILED",
        "http_status": http_status,
        "stop_id": stop_id,
    }


def cancel_all_stop_losses_for(
    orderbook_id: str,
    account_id: str | None = None,
    max_wait: float = 3.0,
    poll_interval: float = 0.5,
) -> dict:
    """Cancel every active stop-loss for ``orderbook_id`` and verify clearance.

    The "verify" step is the critical part: Avanza's DELETE returns 200 OK
    immediately, but the encumbered volume on the position is not released
    until the SL actually disappears from the position view. Without polling,
    a follow-up SELL still gets ``short.sell.not.allowed``. We therefore
    re-query ``get_stop_losses_strict()`` every ``poll_interval`` seconds
    until none remain for the target orderbook (or ``max_wait`` is exceeded).

    **Fail-closed semantics**: if the stop-loss list cannot be read (network
    error, 5xx, malformed response), the function returns ``status="FAILED"``
    rather than silently treating "could not read" as "no stops exist".
    A safety-critical caller deciding whether to proceed with a sell MUST
    NOT be misled into believing the path is clear when reality is unknown.

    The function is idempotent and safe to call when no SLs exist — it
    short-circuits to ``status="SUCCESS"`` without any DELETE calls.

    Args:
        orderbook_id: Avanza orderbook ID to clear.
        account_id: Account filter. ``None`` means accept any account.
        max_wait: Maximum total wall-clock seconds to wait for clearance.
        poll_interval: Seconds between re-query attempts.

    Returns:
        Dict with:
            - ``status``: "SUCCESS" (cleared), "PARTIAL" (some cancelled, some
              still showing after timeout), or "FAILED" (no cancels succeeded
              and stops still present, OR the SL list could not be read).
            - ``cancelled``: list of stop_ids the DELETE call accepted.
            - ``remaining``: list of stop_ids still present after the wait.
            - ``snapshot``: list of full stop-loss dicts that were present at
              the start of the cancel sequence. Callers can use this to
              **re-arm** identical stops if the dependent sell fails — the
              cancel/sell sequence is otherwise rollbackable but leaves the
              position naked on partial-completion failure.
            - ``elapsed_seconds``: float, total time spent in this call.
            - ``error``: optional, present only when ``status="FAILED"`` due
              to a read error rather than cancel failures.
    """
    started = time.monotonic()
    target_ob = str(orderbook_id)
    aid_filter = str(account_id) if account_id is not None else None

    def _filter_for_ob(stops: list[dict]) -> list[dict]:
        out = []
        for sl in stops:
            if not isinstance(sl, dict):
                continue
            ob = (sl.get("orderbook") or {}).get("id")
            if str(ob) != target_ob:
                continue
            if aid_filter is not None:
                acct = (sl.get("account") or {}).get("id")
                if str(acct) != aid_filter:
                    continue
            out.append(sl)
        return out

    # Initial fetch — fail closed on read errors. A safety-critical caller
    # cannot tell "no stops" apart from "API down" without this distinction.
    try:
        all_stops = get_stop_losses_strict()
    except Exception as exc:  # noqa: BLE001 — convert to structured failure
        elapsed = time.monotonic() - started
        logger.error(
            "cancel_all_stop_losses_for(%s): cannot read stop-loss list: %s",
            target_ob, exc,
        )
        return {
            "status": "FAILED",
            "cancelled": [],
            "remaining": [],
            "snapshot": [],
            "elapsed_seconds": elapsed,
            "error": f"read_error: {exc}",
        }

    initial = _filter_for_ob(all_stops)
    if not initial:
        return {
            "status": "SUCCESS",
            "cancelled": [],
            "remaining": [],
            "snapshot": [],
            "elapsed_seconds": time.monotonic() - started,
        }

    # Snapshot full dicts before cancelling so a caller can re-arm if the
    # dependent sell fails downstream. We deep-copy to insulate against any
    # downstream mutation of the returned structure.
    import copy as _copy
    snapshot = [_copy.deepcopy(sl) for sl in initial]

    # Issue cancels for every matching stop. Use the SL's own account id when
    # available — Avanza's DELETE endpoint requires the account that owns the
    # stop, which may differ from DEFAULT_ACCOUNT_ID for multi-account users.
    cancelled: list[str] = []
    for sl in initial:
        sid = sl.get("id") or ""
        if not sid:
            continue
        sl_acct = (sl.get("account") or {}).get("id") or account_id
        result = cancel_stop_loss(sid, account_id=sl_acct)
        if result.get("status") == "SUCCESS":
            cancelled.append(sid)

    # Poll until cleared or timeout. Re-query is also fail-closed — if the
    # API stops responding mid-poll, treat the orderbook as "may still have
    # stops" rather than declaring victory.
    remaining: list[str] = []
    poll_read_failed = False
    while True:
        try:
            poll_stops = get_stop_losses_strict()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cancel_all_stop_losses_for(%s): poll read failed: %s",
                target_ob, exc,
            )
            poll_read_failed = True
            # We don't know if the stops are gone. Fail closed.
            remaining = [sl.get("id", "") for sl in initial if sl.get("id") and sl.get("id") not in cancelled]
            break
        still = _filter_for_ob(poll_stops)
        remaining = [s.get("id", "") for s in still if s.get("id")]
        if not remaining:
            break
        if (time.monotonic() - started) >= max_wait:
            break
        time.sleep(poll_interval)

    elapsed = time.monotonic() - started

    # CODEX-7 finding 1: critical filter — a DELETE-accepted id can still
    # be in `remaining` if the verification poll observed it alive
    # (broker rejected the cancel late, or the DELETE was acknowledged
    # but never propagated). The set we expose to callers as the
    # rollback set MUST be the VERIFIED-cleared set:
    #     verified = cancelled - remaining
    # Re-arming a stop that is still alive would create a duplicate
    # at the broker, recreating the exact over-encumbered failure mode
    # this whole module exists to prevent.
    remaining_set = set(remaining)
    cancelled = [c for c in cancelled if c not in remaining_set]

    if not remaining and not poll_read_failed:
        status = "SUCCESS"
        logger.info(
            "cancel_all_stop_losses_for(%s): cleared %d stops in %.2fs",
            target_ob, len(cancelled), elapsed,
        )
    elif cancelled and not poll_read_failed:
        status = "PARTIAL"
        logger.warning(
            "cancel_all_stop_losses_for(%s): PARTIAL — verified_cancelled=%s remaining=%s elapsed=%.2fs",
            target_ob, cancelled, remaining, elapsed,
        )
    else:
        status = "FAILED"
        logger.error(
            "cancel_all_stop_losses_for(%s): FAILED — cancelled=%s remaining=%s read_failed=%s",
            target_ob, cancelled, remaining, poll_read_failed,
        )
        # When the verification poll failed, we don't actually know which
        # DELETEs took effect. The list of DELETE-accepted ids is
        # broker-acknowledged but NOT verified-cleared. Drop them all to
        # be safe on the rollback side.
        if poll_read_failed:
            cancelled = []
    return {
        "status": status,
        "cancelled": cancelled,
        "remaining": remaining,
        "snapshot": snapshot,
        "elapsed_seconds": elapsed,
    }


def rearm_stop_losses_from_snapshot(snapshot: list[dict]) -> dict:
    """Re-place stop-losses from the snapshot returned by
    :func:`cancel_all_stop_losses_for`.

    Used to roll back a cancel-then-sell sequence when the sell fails:
    we cancelled the stops to clear the volume, the sell didn't go through,
    and the position is now naked. Re-arming restores the original
    protection so we are no worse off than before the attempt.

    Notes on best-effort behavior:

    - Each re-arm is independent. If one fails, the others still try.
    - The new stop-loss IDs differ from the originals — Avanza issues
      fresh IDs on every place. Callers tracking IDs in local state must
      replace, not deduplicate.
    - ``valid_days`` is computed from the snapshot's ``trigger.validUntil``
      field where present, falling back to 8 days. The trigger semantics
      and price/volume are preserved exactly.

    Args:
        snapshot: List of stop-loss dicts as returned in
            ``cancel_all_stop_losses_for(...)["snapshot"]``.

    Returns:
        Dict with:
            - ``status``: "SUCCESS" (all re-armed), "PARTIAL" (some failed),
              "FAILED" (none succeeded), or "SUCCESS" (snapshot was empty).
            - ``rearmed``: list of new stop_ids placed.
            - ``failed``: list of original stop_ids that could not be re-armed.
    """
    if not snapshot:
        return {"status": "SUCCESS", "rearmed": [], "failed": []}

    rearmed: list[str] = []
    failed: list[str] = []
    today_iso = date.today()

    for sl in snapshot:
        if not isinstance(sl, dict):
            continue
        original_id = sl.get("id", "")
        try:
            ob_id = (sl.get("orderbook") or {}).get("id")
            account = (sl.get("account") or {}).get("id")
            trigger = sl.get("trigger") or {}
            order = sl.get("order") or {}
            trigger_value = trigger.get("value")
            trigger_type = trigger.get("type", "LESS_OR_EQUAL")
            value_type = trigger.get("valueType", "MONETARY")
            sell_price = order.get("price")
            volume = order.get("volume")

            # Compute valid_days from validUntil if present, else default 8.
            valid_days = 8
            valid_until = trigger.get("validUntil")
            if valid_until:
                try:
                    parsed = datetime.strptime(valid_until, "%Y-%m-%d").date()
                    delta = (parsed - today_iso).days
                    if delta > 0:
                        valid_days = delta
                except (ValueError, TypeError):
                    pass

            if not (ob_id and trigger_value is not None and sell_price is not None and volume):
                logger.warning("rearm_stop_losses: snapshot entry missing fields: %s", sl)
                failed.append(original_id)
                continue

            result = place_stop_loss(
                orderbook_id=str(ob_id),
                trigger_price=float(trigger_value),
                sell_price=float(sell_price),
                volume=int(volume),
                account_id=account,
                valid_days=valid_days,
                trigger_type=str(trigger_type),
                value_type=str(value_type),
            )
            if result.get("status") == "SUCCESS":
                new_id = result.get("stoplossOrderId", "")
                rearmed.append(new_id)
                logger.info(
                    "rearm_stop_losses: replaced %s -> %s (ob=%s vol=%s)",
                    original_id, new_id, ob_id, volume,
                )
            else:
                logger.warning(
                    "rearm_stop_losses: place_stop_loss failed for original %s: %s",
                    original_id, result,
                )
                failed.append(original_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "rearm_stop_losses: exception for original %s: %s",
                original_id, exc, exc_info=True,
            )
            failed.append(original_id)

    if not failed:
        status = "SUCCESS"
    elif rearmed:
        status = "PARTIAL"
    else:
        status = "FAILED"
    return {"status": status, "rearmed": rearmed, "failed": failed}


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
            logger.warning("Market guide lookup failed for %s/%s: %s", instrument_type, orderbook_id, e)
            continue

    # Fallback: generic orderbook endpoint
    return api_get(f"/_api/orderbook/{orderbook_id}")
