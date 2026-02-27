"""Avanza-tracked instruments: Nordic stocks (price-only) and warrants (underlying signals).

Tier 2 (Nordic equities): Price + P&L only via Avanza API. No technical signals.
Tier 3 (Warrants): Warrant price via Avanza + underlying ticker's signals for decisions.

Configuration lives in config.json under "avanza.instruments":
    {
        "avanza": {
            "instruments": {
                "SAAB-B": {"orderbook_id": "5533", "type": "equity", "name": "SAAB B"},
                "BULL-NDX3X": {"orderbook_id": "1234", "type": "warrant", "name": "BULL NASDAQ X3", "underlying": "QQQ"}
            }
        }
    }
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("portfolio.avanza_tracker")

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.json"


def load_avanza_instruments() -> dict[str, dict]:
    """Load Avanza instrument config from config.json.

    Returns:
        Dict of {config_key: instrument_config} or empty dict if not configured.
    """
    if not CONFIG_FILE.exists():
        return {}
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return config.get("avanza", {}).get("instruments", {})
    except Exception:
        return {}


def fetch_avanza_prices() -> dict[str, dict[str, Any]]:
    """Fetch current prices for all configured Avanza instruments.

    Returns:
        Dict of {config_key: {"name": str, "price_sek": float, "change_pct": float, "type": str}}
        Skips instruments with missing or empty orderbook_id.
    """
    instruments = load_avanza_instruments()
    if not instruments:
        return {}

    try:
        from portfolio.avanza_client import get_price
    except Exception:
        return {}

    results = {}
    for key, cfg in instruments.items():
        ob_id = cfg.get("orderbook_id", "")
        if not ob_id:
            continue
        try:
            info = get_price(ob_id)
            results[key] = {
                "name": cfg.get("name", key),
                "price_sek": float(info.get("lastPrice", 0)),
                "change_pct": float(info.get("changePercent", 0)),
                "type": cfg.get("type", "equity"),
                "underlying": cfg.get("underlying"),
            }
        except Exception as e:
            logger.warning("Price fetch failed for %s: %s", key, e)
    return results


def get_warrant_underlying(config_key: str) -> str | None:
    """Get the underlying ticker for a warrant instrument.

    Args:
        config_key: The config key (e.g., "BULL-NDX3X")

    Returns:
        Underlying ticker (e.g., "QQQ") or None if not a warrant.
    """
    instruments = load_avanza_instruments()
    cfg = instruments.get(config_key, {})
    if cfg.get("type") != "warrant":
        return None
    return cfg.get("underlying")


def get_all_underlyings() -> dict[str, str]:
    """Get mapping of all warrant config keys to their underlying tickers.

    Returns:
        Dict of {config_key: underlying_ticker} for all warrants.
    """
    instruments = load_avanza_instruments()
    return {
        key: cfg["underlying"]
        for key, cfg in instruments.items()
        if cfg.get("type") == "warrant" and cfg.get("underlying")
    }


def check_session_expiry() -> str | None:
    """Check if Avanza BankID session is expired or expiring soon.

    Returns:
        Warning message string if session needs refresh, None if OK.
    """
    try:
        from portfolio.avanza_session import (
            is_session_expiring_soon,
            session_remaining_minutes,
        )
    except ImportError:
        return None

    remaining = session_remaining_minutes()
    if remaining is None:
        return "Avanza session not found. Run: python scripts/avanza_login.py"
    if remaining <= 0:
        return "Avanza session expired. Run: python scripts/avanza_login.py"
    if is_session_expiring_soon(threshold_minutes=60.0):
        mins = int(remaining)
        return (
            f"Avanza session expires in {mins}min. "
            "Run: python scripts/avanza_login.py"
        )
    return None
