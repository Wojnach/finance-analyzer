"""Avanza session management — load, validate, and use BankID-captured sessions.

Uses Playwright's saved storage state to make authenticated API calls via a
headless browser context. This ensures cookies and TLS session match what
Avanza expects (replaying cookies via requests library causes 401s).

This is the preferred auth method until TOTP credentials are configured.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("portfolio.avanza_session")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SESSION_FILE = DATA_DIR / "avanza_session.json"
STORAGE_STATE_FILE = DATA_DIR / "avanza_storage_state.json"
API_BASE = "https://www.avanza.se"

# Minimum remaining session life before we consider it expired (minutes)
EXPIRY_BUFFER_MINUTES = 30

# Module-level Playwright context (lazy-initialized, reused across calls)
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

    try:
        data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise AvanzaSessionError(f"Failed to read session file: {e}")

    # Check expiry
    expires_at = data.get("expires_at")
    if expires_at:
        try:
            exp = datetime.fromisoformat(expires_at)
            now = datetime.now(timezone.utc)
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


def session_remaining_minutes() -> Optional[float]:
    """Get minutes remaining on the current session, or None if no session."""
    try:
        data = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
        expires_at = data.get("expires_at")
        if not expires_at:
            return None
        exp = datetime.fromisoformat(expires_at)
        now = datetime.now(timezone.utc)
        return (exp - now).total_seconds() / 60.0
    except Exception:
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
    if _pw_context:
        try:
            _pw_context.close()
        except Exception:
            pass
        _pw_context = None
    if _pw_browser:
        try:
            _pw_browser.close()
        except Exception:
            pass
        _pw_browser = None
    if _pw_instance:
        try:
            _pw_instance.stop()
        except Exception:
            pass
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
        perf = entry.get("lastTradingDayPerformance", {})
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
        except Exception:
            continue

    # Fallback: generic orderbook endpoint
    return api_get(f"/_api/orderbook/{orderbook_id}")
