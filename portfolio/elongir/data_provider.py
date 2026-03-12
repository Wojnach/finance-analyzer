"""Market data collection for the Elongir silver dip-trading bot.

Fetches silver spot from Binance FAPI, USD/SEK from frankfurter.app,
and XAG-USD signals from agent_summary_compact.json.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from portfolio.circuit_breaker import CircuitBreaker

logger = logging.getLogger("portfolio.elongir.data_provider")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

BINANCE_FAPI_BASE = "https://fapi.binance.com"
BINANCE_SYMBOL = "XAGUSDT"

# Circuit breakers for each data source
_cb_binance = CircuitBreaker("elongir-binance", failure_threshold=5, recovery_timeout=60)
_cb_fx = CircuitBreaker("elongir-fx", failure_threshold=3, recovery_timeout=120)


# ---------------------------------------------------------------------------
# Individual data fetchers
# ---------------------------------------------------------------------------

def fetch_silver_price() -> Optional[float]:
    """Fetch current silver spot price from Binance FAPI (XAGUSDT)."""
    if not _cb_binance.allow_request():
        logger.warning("Binance circuit breaker OPEN -- skipping")
        return None
    try:
        r = requests.get(
            f"{BINANCE_FAPI_BASE}/fapi/v1/ticker/price",
            params={"symbol": BINANCE_SYMBOL},
            timeout=5,
        )
        r.raise_for_status()
        price = float(r.json()["price"])
        _cb_binance.record_success()
        return price
    except Exception as e:
        logger.warning("Silver price fetch failed: %s", e)
        _cb_binance.record_failure()
        return None


def fetch_klines(
    interval: str = "1m",
    limit: int = 100,
) -> Optional[list]:
    """Fetch klines from Binance FAPI for XAGUSDT.

    Returns list of raw kline arrays, or None on failure.
    Each kline: [open_time, open, high, low, close, volume, ...]
    """
    if not _cb_binance.allow_request():
        return None
    try:
        r = requests.get(
            f"{BINANCE_FAPI_BASE}/fapi/v1/klines",
            params={"symbol": BINANCE_SYMBOL, "interval": interval, "limit": limit},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        _cb_binance.record_success()
        return data
    except Exception as e:
        logger.warning("Klines fetch failed (%s): %s", interval, e)
        _cb_binance.record_failure()
        return None


def fetch_usdsek() -> Optional[float]:
    """Fetch USD/SEK exchange rate using portfolio.fx_rates."""
    if not _cb_fx.allow_request():
        logger.warning("FX circuit breaker OPEN -- skipping")
        return None
    try:
        from portfolio.fx_rates import fetch_usd_sek
        rate = fetch_usd_sek()
        _cb_fx.record_success()
        return rate
    except Exception as e:
        logger.warning("FX rate fetch failed: %s", e)
        _cb_fx.record_failure()
        return None


def read_xag_signals() -> Optional[dict]:
    """Read XAG-USD signals from agent_summary_compact.json.

    Returns the XAG-USD ticker section, or None if unavailable.
    """
    compact_path = DATA_DIR / "agent_summary_compact.json"
    try:
        with open(compact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tickers = data.get("tickers", {})
        return tickers.get("XAG-USD")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        logger.debug("XAG signals not available: %s", e)
        return None


# ---------------------------------------------------------------------------
# MarketSnapshot
# ---------------------------------------------------------------------------

@dataclass
class MarketSnapshot:
    """A point-in-time snapshot of all market data needed by the bot."""
    silver_usd: float = 0.0
    fx_rate: float = 0.0
    klines_1m: Optional[list] = None
    klines_5m: Optional[list] = None
    klines_15m: Optional[list] = None
    xag_signals: Optional[dict] = None
    timestamp: str = ""

    def is_complete(self) -> bool:
        """Check if the snapshot has the minimum required data."""
        return self.silver_usd > 0 and self.fx_rate > 0


def collect_snapshot() -> MarketSnapshot:
    """Collect a complete market snapshot.

    Fetches silver price, FX rate, klines at 1m/5m/15m, and XAG signals.
    Returns a MarketSnapshot even if some data is missing.
    """
    silver = fetch_silver_price()
    fx = fetch_usdsek()
    klines_1m = fetch_klines("1m", 100)
    klines_5m = fetch_klines("5m", 60)
    klines_15m = fetch_klines("15m", 40)
    xag_signals = read_xag_signals()

    return MarketSnapshot(
        silver_usd=silver or 0.0,
        fx_rate=fx or 0.0,
        klines_1m=klines_1m,
        klines_5m=klines_5m,
        klines_15m=klines_15m,
        xag_signals=xag_signals,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
