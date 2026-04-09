"""Crypto market data module for the unified monitoring loop.

Provides Fear & Greed index, crypto news, MSTR price + BTC NAV premium,
and on-chain metrics. All with TTL caching to avoid API spam.

Usage from metals_loop.py:
    from crypto_data import (
        get_fear_greed, get_crypto_news, fetch_mstr_price,
        compute_mstr_btc_nav, get_onchain_summary,
    )
"""

import datetime
import logging
import os
import sys
import time

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests

logger = logging.getLogger(__name__)

# --- Cache with TTL ---
_cache = {}  # key -> {"ts": float, "data": any}

FEAR_GREED_TTL = 300       # 5 min
NEWS_TTL = 300             # 5 min
MSTR_PRICE_TTL = 60        # 1 min (only during US hours)
ONCHAIN_TTL = 3600         # 1 hour

# Log-once guard for CryptoCompare news failures — prevents log spam when the
# endpoint is broken for extended periods. First failure logs with a
# "(suppressing further)" marker; subsequent failures are silenced until the
# process restarts. Added 2026-04-09 after slice-TypeError incident spammed
# warnings every news-fetch cycle.
_WARNED_CRYPTO_NEWS: bool = False


def _cached(key, ttl):
    """Return cached data if still fresh, else None."""
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < ttl:
        return entry["data"]
    return None


def _set_cache(key, data):
    """Store data in cache with current timestamp."""
    _cache[key] = {"ts": time.time(), "data": data}


# ---------------------------------------------------------------------------
# Fear & Greed Index (Alternative.me — crypto)
# ---------------------------------------------------------------------------

def get_fear_greed():
    """Fetch crypto Fear & Greed index. Returns dict or None.

    Returns: {"value": int, "classification": str, "timestamp": str}
    """
    cached = _cached("fear_greed", FEAR_GREED_TTL)
    if cached is not None:
        return cached

    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", [{}])[0]
            result = {
                "value": int(data.get("value", 50)),
                "classification": data.get("value_classification", "Neutral"),
                "timestamp": data.get("timestamp", ""),
            }
            _set_cache("fear_greed", result)
            return result
    except Exception as e:
        logger.warning(f"Fear & Greed fetch error: {e}")
    return None


# ---------------------------------------------------------------------------
# Crypto News (CryptoCompare)
# ---------------------------------------------------------------------------

def get_crypto_news(limit=10):
    """Fetch latest crypto news headlines from CryptoCompare.

    Returns list of {"title": str, "source": str, "categories": str, "published_on": int}
    """
    cached = _cached("crypto_news", NEWS_TTL)
    if cached is not None:
        return cached

    try:
        r = requests.get(
            "https://min-api.cryptocompare.com/data/v2/news/",
            params={"lang": "EN", "sortOrder": "latest"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json().get("Data", [])
            if not isinstance(data, list):
                logger.warning(
                    "CryptoCompare returned non-list Data (type=%s); treating as empty",
                    type(data).__name__,
                )
                data = []
            articles = data[:limit]
            result = []
            for a in articles:
                result.append({
                    "title": a.get("title", ""),
                    "source": a.get("source_info", {}).get("name", a.get("source", "")),
                    "categories": a.get("categories", ""),
                    "published_on": a.get("published_on", 0),
                })
            _set_cache("crypto_news", result)
            return result
    except Exception as e:
        global _WARNED_CRYPTO_NEWS
        if not _WARNED_CRYPTO_NEWS:
            logger.warning("CryptoCompare news error (suppressing further): %s", e)
            _WARNED_CRYPTO_NEWS = True
    return []


# ---------------------------------------------------------------------------
# MSTR Price (Yahoo Finance v8 API)
# ---------------------------------------------------------------------------

def fetch_mstr_price():
    """Fetch MSTR price from Yahoo Finance. Returns dict or None.

    Returns: {"price": float, "change_pct": float, "market_state": str}
    """
    cached = _cached("mstr_price", MSTR_PRICE_TTL)
    if cached is not None:
        return cached

    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/MSTR",
            params={"interval": "1d", "range": "2d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
            price = meta.get("regularMarketPrice", 0)
            prev_close = meta.get("previousClose", 0) or meta.get("chartPreviousClose", 0)
            market_state = meta.get("marketState", "CLOSED")

            change_pct = 0
            if prev_close and prev_close > 0:
                change_pct = ((price - prev_close) / prev_close) * 100

            result = {
                "price": float(price),
                "change_pct": round(change_pct, 2),
                "market_state": market_state,
                "prev_close": float(prev_close) if prev_close else 0,
                "currency": meta.get("currency", "USD"),
            }
            _set_cache("mstr_price", result)
            return result
    except Exception as e:
        logger.warning(f"MSTR Yahoo fetch error: {e}")
    return None


# ---------------------------------------------------------------------------
# MSTR-BTC NAV Premium/Discount
# ---------------------------------------------------------------------------

# MSTR BTC holdings (updated periodically — approximate)
MSTR_BTC_HOLDINGS = 499_096  # as of early 2026
MSTR_SHARES_OUTSTANDING = 229_000_000  # approximate

def compute_mstr_btc_nav(mstr_price, btc_price):
    """Compute MSTR premium/discount to BTC NAV.

    Returns: {"nav_per_share": float, "premium_pct": float} or None
    """
    if not mstr_price or mstr_price <= 0 or not btc_price or btc_price <= 0:
        return None

    nav_total = MSTR_BTC_HOLDINGS * btc_price
    nav_per_share = nav_total / MSTR_SHARES_OUTSTANDING
    premium_pct = ((mstr_price - nav_per_share) / nav_per_share) * 100

    return {
        "nav_per_share": round(nav_per_share, 2),
        "premium_pct": round(premium_pct, 1),
        "btc_holdings": MSTR_BTC_HOLDINGS,
    }


# ---------------------------------------------------------------------------
# On-chain Summary (wraps portfolio.onchain_data)
# ---------------------------------------------------------------------------

def get_onchain_summary():
    """Get on-chain metrics summary (MVRV, SOPR, NUPL, etc.).

    Returns dict with zone classification or None.
    """
    cached = _cached("onchain", ONCHAIN_TTL)
    if cached is not None:
        return cached

    try:
        from portfolio.onchain_data import get_onchain_data, interpret_onchain
        raw = get_onchain_data()
        if not raw:
            return None
        interpretation = interpret_onchain(raw)
        result = {
            "mvrv": raw.get("mvrv"),
            "sopr": raw.get("sopr"),
            "nupl": raw.get("nupl"),
            "zone": interpretation.get("zone", "neutral"),
            "bias": interpretation.get("bias", "neutral"),
            "summary": interpretation.get("summary", ""),
        }
        _set_cache("onchain", result)
        return result
    except ImportError:
        logger.warning("portfolio.onchain_data not available")
        return None
    except Exception as e:
        logger.warning(f"On-chain data error: {e}")
        return None


# ---------------------------------------------------------------------------
# US Market Hours Check
# ---------------------------------------------------------------------------

def is_us_market_hours(now=None):
    """Check if the US regular session is open (09:30-16:00 New York time)."""
    now = now or datetime.datetime.now(datetime.UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=datetime.UTC)

    if ZoneInfo is not None:
        ny = now.astimezone(ZoneInfo("America/New_York"))
        minutes = ny.hour * 60 + ny.minute
        return ny.weekday() < 5 and (9 * 60 + 30) <= minutes <= (16 * 60)

    # Fallback: UTC-based approximation when zoneinfo is unavailable.
    return now.weekday() < 5 and 14 <= now.hour <= 21
