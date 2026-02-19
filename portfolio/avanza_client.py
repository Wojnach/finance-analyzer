"""Avanza API client for portfolio monitoring.

Uses the avanza-api library to authenticate and interact with Avanza's
trading platform. Credentials are read from config.json under the 'avanza' key.

Required config.json entry:
    "avanza": {
        "username": "YOUR_AVANZA_USERNAME",
        "password": "YOUR_AVANZA_PASSWORD",
        "totp_secret": "YOUR_TOTP_SECRET"
    }
"""

import json
from pathlib import Path
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.json"

# Singleton client instance
_client = None


def _load_credentials() -> dict:
    """Load Avanza credentials from config.json.

    Returns:
        dict with keys: username, password, totp_secret

    Raises:
        FileNotFoundError: if config.json does not exist
        KeyError: if 'avanza' section is missing from config
    """
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
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


def get_client():
    """Get or create a singleton Avanza client with TOTP authentication.

    Returns:
        Authenticated Avanza client instance

    Raises:
        Exception: if authentication fails or avanza-api not installed
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
    """Reset the singleton client (useful for re-authentication)."""
    global _client
    _client = None


def find_instrument(query: str) -> list[dict]:
    """Search for instruments by name or ticker.

    Args:
        query: Search string (e.g., 'Bitcoin', 'MSTR', 'Nvidia')

    Returns:
        List of matching instruments with id, name, and type
    """
    client = get_client()
    results = client.search_for_stock(query)
    return results


def get_price(orderbook_id: str) -> dict[str, Any]:
    """Get current price and info for an instrument.

    Args:
        orderbook_id: Avanza orderbook ID (numeric string)

    Returns:
        Dict with price info including lastPrice, change, changePercent, etc.
    """
    client = get_client()
    info = client.get_stock_info(orderbook_id)
    return info


def get_positions() -> list[dict]:
    """Get all current positions from the Avanza account.

    Returns:
        List of position dicts, each with name, value, profit, etc.
        Returns empty list if no positions or on error.
    """
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
