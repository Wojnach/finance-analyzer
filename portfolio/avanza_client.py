"""Avanza API client for portfolio monitoring and trading.

Supports two authentication methods:
1. BankID session (preferred) — captured by scripts/avanza_login.py, stored in
   data/avanza_session.json. No credentials needed, valid ~24h.
2. TOTP credentials (fallback) — uses avanza-api library with username/password/TOTP
   from config.json.

The client transparently tries BankID session first, then falls back to TOTP.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Any

from portfolio.file_utils import load_json

logger = logging.getLogger("portfolio.avanza_client")

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.json"

# Singleton client instance (avanza-api library)
_client = None
# Cached signal that a BankID Playwright session has already been verified.
_session_client = None


def _load_credentials() -> dict:
    """Load Avanza credentials from config.json.

    Returns:
        dict with keys: username, password, totp_secret

    Raises:
        FileNotFoundError: if config.json does not exist
        KeyError: if 'avanza' section is missing or credentials incomplete
    """
    config = load_json(CONFIG_FILE)
    if config is None:
        raise FileNotFoundError(f"Config file not found or unreadable: {CONFIG_FILE}")
    if "avanza" not in config:
        raise KeyError(
            "Missing 'avanza' section in config.json. "
            "Add: {\"avanza\": {\"username\": \"...\", \"password\": \"...\", \"totp_secret\": \"...\"}}"
        )
    creds = config["avanza"]
    for key in ("username", "password", "totp_secret"):
        if key not in creds or not creds[key]:
            raise KeyError(f"Missing or empty 'avanza.{key}' in config.json")
    return creds


def _try_session_auth() -> bool:
    """Return True when a BankID-backed Playwright session is available."""
    global _session_client
    if _session_client is True:
        return True
    try:
        from portfolio.avanza_session import verify_session
        if verify_session():
            _session_client = True
            logger.info("Using BankID session for Avanza API")
            return True
        logger.info("BankID session exists but verification failed")
    except Exception as e:
        logger.debug("BankID session not available: %s", e)
    return False


def get_client():
    """Get or create a singleton Avanza client.

    Tries BankID session first, then falls back to TOTP credentials.

    Returns:
        Authenticated Avanza client instance (avanza-api library)

    Raises:
        Exception: if neither auth method works
    """
    global _client
    if _client is not None:
        return _client
    try:
        from avanza import Avanza
    except ImportError:
        raise ImportError(
            "avanza-api package not installed. Run: pip install avanza-api"
        )
    creds = _load_credentials()
    _client = Avanza({
        "username": creds["username"],
        "password": creds["password"],
        "totpSecret": creds["totp_secret"],
    })
    return _client


def reset_client() -> None:
    """Reset the singleton TOTP client (useful for re-authentication)."""
    global _client
    _client = None


def reset_session() -> None:
    """Reset the cached BankID session verification flag."""
    global _session_client
    _session_client = None


def find_instrument(query: str) -> list[dict]:
    """Search for instruments by name or ticker.

    Args:
        query: Search string (e.g., 'Bitcoin', 'NVDA', 'Silver')

    Returns:
        List of matching instruments with id, name, and type
    """
    client = get_client()
    results = client.search_for_stock(query)
    return results


def get_price(orderbook_id: str) -> dict[str, Any]:
    """Get current price and info for an instrument.

    Tries BankID session first, then falls back to TOTP client.

    Args:
        orderbook_id: Avanza orderbook ID (numeric string)

    Returns:
        Dict with price info including lastPrice, change, changePercent, etc.
    """
    # Try session-based auth first
    if _try_session_auth():
        try:
            from portfolio.avanza_session import get_instrument_price
            return get_instrument_price(orderbook_id)
        except Exception as e:
            logger.warning("Session-based price fetch failed, trying TOTP: %s", e)
            reset_session()

    client = get_client()
    info = client.get_stock_info(orderbook_id)
    return info


def get_positions() -> list[dict]:
    """Get all current positions from the Avanza account.

    Tries BankID session first, then falls back to TOTP client.

    Returns:
        List of position dicts, each with name, value, profit, etc.
        Returns empty list if no positions or on error.
    """
    # Try session-based auth first
    if _try_session_auth():
        try:
            from portfolio.avanza_session import get_positions as session_get_positions
            return session_get_positions()
        except Exception as e:
            logger.warning("Session-based positions fetch failed, trying TOTP: %s", e)
            reset_session()

    client = get_client()
    overview = client.get_overview()
    positions = []
    for account in overview.get("accounts", []):
        for pos in account.get("positions", []):
            positions.append({
                "account": account.get("name", ""),
                "account_id": account.get("accountId", ""),
                "name": pos.get("name", ""),
                "ticker": pos.get("orderbookId", ""),
                "volume": pos.get("volume", 0),
                "value": pos.get("value", 0),
                "profit": pos.get("profit", 0),
                "profit_percent": pos.get("profitPercent", 0),
                "currency": pos.get("currency", "SEK"),
            })
    return positions


def get_portfolio_value() -> float:
    """Get total portfolio value in SEK across all Avanza accounts.

    Returns:
        Total portfolio value in SEK
    """
    client = get_client()
    overview = client.get_overview()
    total = 0.0
    for account in overview.get("accounts", []):
        total += account.get("totalValue", 0)
    return total


def get_open_orders() -> list:
    """Return open orders for the ISK account (read-only).

    Uses the authenticated Avanza client; does not place or cancel orders.
    """
    client = get_client()
    account_id = get_account_id()
    try:
        orders = client.get_orders(account_id)
    except Exception as e:
        logger.error("Failed to fetch open orders: %s", e)
        raise
    return orders or []


# --- Account ID ---

_account_id: str | None = None


def get_account_id() -> str:
    """Get ISK account ID from Avanza overview (cached after first call).

    Scans all accounts and returns the first one whose type contains 'ISK'.

    Returns:
        Account ID string

    Raises:
        RuntimeError: if no ISK account is found
    """
    global _account_id
    if _account_id is not None:
        return _account_id
    client = get_client()
    overview = client.get_overview()
    for account in overview.get("accounts", []):
        atype = account.get("accountType", "")
        if "ISK" in atype.upper():
            _account_id = str(account["accountId"])
            logger.info("Found ISK account: %s", _account_id)
            return _account_id
    raise RuntimeError(
        "No ISK account found in Avanza overview. "
        f"Account types: {[a.get('accountType') for a in overview.get('accounts', [])]}"
    )


# --- Trading functions ---


def place_buy_order(
    orderbook_id: str,
    price: float,
    volume: int,
    valid_until: date | None = None,
) -> dict:
    """Place a limit BUY order on Avanza.

    Args:
        orderbook_id: Avanza orderbook ID for the instrument
        price: Limit price in SEK
        volume: Number of shares (must be int >= 1)
        valid_until: Order expiry date. Defaults to today (day order).

    Returns:
        Dict with orderId, orderRequestStatus, message
    """
    from avanza.constants import OrderType
    return _place_order(orderbook_id, OrderType.BUY, price, volume, valid_until)


def place_sell_order(
    orderbook_id: str,
    price: float,
    volume: int,
    valid_until: date | None = None,
) -> dict:
    """Place a limit SELL order on Avanza.

    Args:
        orderbook_id: Avanza orderbook ID for the instrument
        price: Limit price in SEK
        volume: Number of shares (must be int >= 1)
        valid_until: Order expiry date. Defaults to today (day order).

    Returns:
        Dict with orderId, orderRequestStatus, message
    """
    from avanza.constants import OrderType
    return _place_order(orderbook_id, OrderType.SELL, price, volume, valid_until)


def _place_order(orderbook_id, order_type, price, volume, valid_until):
    """Internal: place an order via the Avanza API."""
    if volume < 1:
        raise ValueError(f"Volume must be >= 1, got {volume}")
    if price <= 0:
        raise ValueError(f"Price must be > 0, got {price}")

    client = get_client()
    account_id = get_account_id()
    expiry = valid_until or date.today()

    logger.info(
        "Placing %s order: orderbook=%s price=%.2f vol=%d until=%s account=%s",
        order_type.value, orderbook_id, price, volume, expiry, account_id,
    )
    result = client.place_order(
        account_id=account_id,
        order_book_id=orderbook_id,
        order_type=order_type,
        price=price,
        valid_until=expiry,
        volume=volume,
    )
    logger.info("Order result: %s", result)
    return result


def get_order_status(order_id: str) -> dict:
    """Check the status of an order by ID.

    Returns:
        Order dict with state, price, volume, etc.
    """
    client = get_client()
    account_id = get_account_id()
    return client.get_order(account_id, order_id)


def delete_order(order_id: str) -> dict:
    """Cancel a pending order.

    Returns:
        Dict with orderId, orderRequestStatus, messages
    """
    client = get_client()
    account_id = get_account_id()
    logger.info("Deleting order %s on account %s", order_id, account_id)
    return client.delete_order(account_id, order_id)
