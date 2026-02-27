"""Avanza session management — load, validate, and use BankID-captured sessions.

Provides a lightweight requests.Session wrapper that uses cookies + security
token from a BankID browser login (saved by scripts/avanza_login.py).

This is the preferred auth method until TOTP credentials are configured.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger("portfolio.avanza_session")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SESSION_FILE = DATA_DIR / "avanza_session.json"
API_BASE = "https://www.avanza.se"

# Minimum remaining session life before we consider it expired (minutes)
EXPIRY_BUFFER_MINUTES = 30


class AvanzaSessionError(Exception):
    """Raised when session is missing, expired, or invalid."""


def load_session() -> dict:
    """Load saved BankID session from disk.

    Returns:
        Session dict with cookies, security_token, etc.

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

    if not data.get("cookies"):
        raise AvanzaSessionError("Session file has no cookies.")

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


def create_requests_session(session_data: Optional[dict] = None) -> requests.Session:
    """Create a requests.Session pre-loaded with Avanza cookies and headers.

    Args:
        session_data: Pre-loaded session dict. If None, loads from file.

    Returns:
        Configured requests.Session ready for API calls.

    Raises:
        AvanzaSessionError: if session can't be loaded or is expired.
    """
    if session_data is None:
        session_data = load_session()

    s = requests.Session()

    # Load cookies
    for cookie in session_data.get("cookies", []):
        s.cookies.set(
            cookie["name"],
            cookie["value"],
            domain=cookie.get("domain", ".avanza.se"),
            path=cookie.get("path", "/"),
        )

    # Set security token header if available
    security_token = session_data.get("security_token")
    if security_token:
        s.headers["X-SecurityToken"] = security_token

    # Set auth session header if available
    auth_session = session_data.get("authentication_session")
    if auth_session:
        s.headers["X-AuthenticationSession"] = auth_session

    # Common headers
    s.headers["Accept"] = "application/json"
    s.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    return s


def verify_session(session: Optional[requests.Session] = None) -> bool:
    """Verify that the session is valid by making a lightweight API call.

    Args:
        session: Existing session to verify. If None, creates one from file.

    Returns:
        True if session is valid, False otherwise.
    """
    try:
        if session is None:
            session = create_requests_session()
        resp = session.get(
            f"{API_BASE}/_api/position-data/positions",
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        logger.warning("Session verification failed: %s", e)
        return False


# --- API convenience functions ---


def api_get(path: str, session: Optional[requests.Session] = None, **kwargs) -> Any:
    """Make an authenticated GET request to Avanza API.

    Args:
        path: API path (e.g., "/_api/position-data/positions")
        session: Pre-created session. If None, creates from file.
        **kwargs: Additional kwargs passed to requests.get

    Returns:
        Parsed JSON response.

    Raises:
        AvanzaSessionError: if session is invalid.
        requests.HTTPError: on non-2xx response.
    """
    if session is None:
        session = create_requests_session()
    kwargs.setdefault("timeout", 15)
    url = f"{API_BASE}{path}" if path.startswith("/") else path
    resp = session.get(url, **kwargs)
    if resp.status_code == 401:
        raise AvanzaSessionError(
            "Session returned 401 Unauthorized. "
            "Run: python scripts/avanza_login.py"
        )
    resp.raise_for_status()
    return resp.json()


def get_positions(session: Optional[requests.Session] = None) -> list[dict]:
    """Get all positions via session-based auth.

    Returns:
        List of position dicts with name, value, profit, etc.
    """
    data = api_get("/_api/position-data/positions", session=session)
    positions = []
    for category in data.get("withOrderbook", []):
        for instrument in category.get("instruments", [category]):
            positions.append({
                "name": instrument.get("name", ""),
                "orderbook_id": str(instrument.get("orderbookId", "")),
                "volume": instrument.get("volume", 0),
                "value": instrument.get("value", 0),
                "profit": instrument.get("profit", 0),
                "profit_percent": instrument.get("profitPercent", 0),
                "currency": instrument.get("currency", "SEK"),
                "last_price": instrument.get("lastPrice", 0),
                "change_percent": instrument.get("changePercent", 0),
            })
    return positions


def get_instrument_price(
    orderbook_id: str, session: Optional[requests.Session] = None
) -> dict[str, Any]:
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
                session=session,
            )
            return data
        except Exception:
            continue

    # Fallback: generic orderbook endpoint
    return api_get(f"/_api/orderbook/{orderbook_id}", session=session)
