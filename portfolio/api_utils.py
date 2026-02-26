"""Shared API utilities for finance-analyzer."""

import json
import pathlib
import threading
import time

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# Canonical API endpoint URLs — import these instead of hardcoding
BINANCE_BASE = "https://api.binance.com/api/v3"
BINANCE_FAPI_BASE = "https://fapi.binance.com/fapi/v1"
BINANCE_FUTURES_DATA = "https://fapi.binance.com/futures/data"
ALPACA_BASE = "https://data.alpaca.markets/v2"

# Config cache (thread-safe)
_config_lock = threading.Lock()
_config_cache = None
_config_mtime = 0.0


def load_config():
    """Load config.json with mtime-based caching."""
    global _config_cache, _config_mtime
    config_path = BASE_DIR / "config.json"

    with _config_lock:
        try:
            mtime = config_path.stat().st_mtime
            if _config_cache is None or mtime != _config_mtime:
                with open(config_path, "r", encoding="utf-8") as f:
                    _config_cache = json.load(f)
                _config_mtime = mtime
        except Exception:
            if _config_cache is None:
                raise
    return _config_cache


def get_alpaca_headers():
    """Get Alpaca API authentication headers."""
    config = load_config()
    alpaca = config.get("alpaca", {})
    return {
        "APCA-API-KEY-ID": alpaca.get("key", ""),
        "APCA-API-SECRET-KEY": alpaca.get("secret", ""),
    }


def get_telegram_config():
    """Get Telegram bot token and chat ID."""
    config = load_config()
    tg = config.get("telegram", {})
    return tg.get("token", ""), tg.get("chat_id", "")


def get_binance_config():
    """Get Binance API credentials."""
    config = load_config()
    ex = config.get("exchange", {})
    return ex.get("apiKey", ""), ex.get("secret", "")
