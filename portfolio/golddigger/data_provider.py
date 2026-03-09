"""Data providers for GoldDigger — gold price, USD/SEK, US10Y yield, certificate quotes.

Uses Binance FAPI for gold (XAUUSDT), existing fx_rates for USD/SEK,
FRED for US10Y yield, and Avanza Playwright session for certificate bid/ask.
"""

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from portfolio.circuit_breaker import CircuitBreaker
from portfolio.http_retry import fetch_with_retry

logger = logging.getLogger("portfolio.golddigger.data")

BINANCE_FAPI_BASE = "https://fapi.binance.com/fapi/v1"

_gold_cb = CircuitBreaker("golddigger_gold", failure_threshold=5, recovery_timeout=60)
_fred_cb = CircuitBreaker("golddigger_fred", failure_threshold=3, recovery_timeout=300)

# FRED yield cache (yields don't change intraday often)
_yield_cache: dict = {"value": None, "time": 0}
_YIELD_CACHE_TTL = 3600  # 1 hour


@dataclass
class MarketSnapshot:
    """A single point-in-time market data sample."""
    ts_utc: datetime
    gold: float           # XAUUSD price
    usdsek: float         # USD/SEK exchange rate
    us10y: float          # US 10Y yield as decimal (e.g., 0.0425)
    cert_bid: Optional[float] = None
    cert_ask: Optional[float] = None
    cert_last: Optional[float] = None
    cert_spread_pct: Optional[float] = None
    data_quality: str = "ok"  # "ok", "partial", "stale"

    def is_complete(self) -> bool:
        return (
            self.gold > 0
            and self.usdsek > 0
            and self.us10y > 0
        )


def fetch_gold_price() -> Optional[float]:
    """Fetch XAUUSDT from Binance FAPI (futures). Returns price or None."""
    if not _gold_cb.allow_request():
        logger.warning("Gold circuit breaker OPEN")
        return None
    try:
        r = fetch_with_retry(
            f"{BINANCE_FAPI_BASE}/ticker/price",
            params={"symbol": "XAUUSDT"},
            timeout=10,
        )
        if r is None:
            _gold_cb.record_failure()
            return None
        r.raise_for_status()
        price = float(r.json()["price"])
        _gold_cb.record_success()
        return price
    except Exception as e:
        logger.warning("Gold price fetch failed: %s", e)
        _gold_cb.record_failure()
        return None


def fetch_usdsek() -> Optional[float]:
    """Fetch USD/SEK via existing fx_rates module."""
    try:
        from portfolio.fx_rates import fetch_usd_sek
        return fetch_usd_sek()
    except Exception as e:
        logger.warning("USD/SEK fetch failed: %s", e)
        return None


def fetch_us10y(fred_api_key: str = "") -> Optional[float]:
    """Fetch US 10Y yield from FRED API. Returns yield as decimal (e.g., 0.0425).

    Cached for 1 hour since Treasury yields are based on daily closes.
    Falls back to cached value if API is unavailable.
    """
    now = time.time()
    if _yield_cache["value"] is not None and now - _yield_cache["time"] < _YIELD_CACHE_TTL:
        return _yield_cache["value"]

    if not fred_api_key:
        logger.debug("No FRED API key — using cached yield or None")
        return _yield_cache.get("value")

    if not _fred_cb.allow_request():
        logger.warning("FRED circuit breaker OPEN")
        return _yield_cache.get("value")

    try:
        r = fetch_with_retry(
            "https://api.stlouisfed.org/fred/series/observations",
            params={
                "series_id": "DGS10",
                "api_key": fred_api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 5,
            },
            timeout=15,
        )
        if r is None:
            _fred_cb.record_failure()
            return _yield_cache.get("value")
        r.raise_for_status()
        observations = r.json().get("observations", [])
        for obs in observations:
            val = obs.get("value", ".")
            if val != ".":
                yield_pct = float(val)
                yield_decimal = yield_pct / 100.0
                _yield_cache["value"] = yield_decimal
                _yield_cache["time"] = now
                _fred_cb.record_success()
                return yield_decimal
        logger.warning("FRED returned no valid yield observations")
        _fred_cb.record_failure()
        return _yield_cache.get("value")
    except Exception as e:
        logger.warning("FRED US10Y fetch failed: %s", e)
        _fred_cb.record_failure()
        return _yield_cache.get("value")


def fetch_certificate_price(page, orderbook_id: str, api_type: str = "warrant") -> Optional[dict]:
    """Fetch certificate bid/ask/last from Avanza via Playwright page.

    Returns dict: {bid, ask, last, spread_pct} or None.
    """
    if not orderbook_id:
        return None
    try:
        from data.metals_avanza_helpers import fetch_price
        data = fetch_price(page, orderbook_id, api_type)
        if data is None:
            return None
        bid = data.get("bid")
        ask = data.get("ask")
        last = data.get("last")
        spread_pct = None
        if bid and ask and bid > 0:
            spread_pct = (ask - bid) / bid
        return {
            "bid": bid,
            "ask": ask,
            "last": last,
            "spread_pct": spread_pct,
        }
    except Exception as e:
        logger.warning("Certificate price fetch failed: %s", e)
        return None


def collect_snapshot(
    fred_api_key: str = "",
    page=None,
    orderbook_id: str = "",
    api_type: str = "warrant",
) -> MarketSnapshot:
    """Collect a complete market snapshot from all data sources.

    Fetches gold, FX, yield, and optionally certificate quotes.
    Returns a MarketSnapshot even if some fields are missing (data_quality
    will reflect the completeness).
    """
    ts = datetime.now(timezone.utc)
    gold = fetch_gold_price()
    usdsek = fetch_usdsek()
    us10y = fetch_us10y(fred_api_key)

    cert_data = None
    if page and orderbook_id:
        cert_data = fetch_certificate_price(page, orderbook_id, api_type)

    # Determine data quality
    missing = []
    if gold is None:
        missing.append("gold")
        gold = 0.0
    if usdsek is None:
        missing.append("usdsek")
        usdsek = 0.0
    if us10y is None:
        missing.append("us10y")
        us10y = 0.0

    quality = "ok"
    if missing:
        quality = "partial" if len(missing) < 3 else "stale"
        logger.warning("Missing data sources: %s", ", ".join(missing))

    snap = MarketSnapshot(
        ts_utc=ts,
        gold=gold,
        usdsek=usdsek,
        us10y=us10y,
        data_quality=quality,
    )
    if cert_data:
        snap.cert_bid = cert_data.get("bid")
        snap.cert_ask = cert_data.get("ask")
        snap.cert_last = cert_data.get("last")
        snap.cert_spread_pct = cert_data.get("spread_pct")

    return snap
