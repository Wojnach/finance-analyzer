"""BGeometrics on-chain data integration for Bitcoin.

Fetches MVRV Z-Score, SOPR, NUPL, realized price, exchange netflow,
and liquidation data from the free BGeometrics API (bitcoin-data.com).

Free tier: 8 requests/hour, 15 requests/day.
Budget: 6 metrics x 2 refreshes/day = 12 requests.
Cache: 12 hours per metric (on-chain data doesn't change fast).

Usage:
    from portfolio.onchain_data import get_onchain_data, interpret_onchain
    data = get_onchain_data()  # returns dict or None
    interp = interpret_onchain(data)  # returns interpretation dict
"""

import json
import logging
import time
from pathlib import Path

from portfolio.http_retry import fetch_with_retry
from portfolio.shared_state import _cached

logger = logging.getLogger("portfolio.onchain_data")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "onchain_cache.json"

API_BASE = "https://bitcoin-data.com"
ONCHAIN_TTL = 43200  # 12 hours


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config_token():
    """Load BGeometrics API token from config.json."""
    try:
        config = json.loads((BASE_DIR / "config.json").read_text(encoding="utf-8"))
        token = config.get("bgeometrics", {}).get("api_token", "")
        return token if token else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Persistent cache (survives restarts)
# ---------------------------------------------------------------------------

def _save_onchain_cache(data):
    """Save on-chain data to persistent cache file."""
    try:
        CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        logger.warning("Failed to write onchain cache", exc_info=True)


def _load_onchain_cache(max_age_seconds=ONCHAIN_TTL):
    """Load on-chain data from persistent cache if fresh enough."""
    try:
        if not CACHE_FILE.exists():
            return None
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        ts = data.get("ts", 0)
        if time.time() - ts > max_age_seconds:
            return None
        return data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual metric fetchers
# ---------------------------------------------------------------------------

def _api_get(endpoint, token, params=None):
    """Make authenticated GET request to BGeometrics API.

    Uses fetch_with_retry but skips retries on 429 (rate limit) since
    retrying just burns more of the 8 req/hour free tier budget.
    """
    url = f"{API_BASE}{endpoint}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = fetch_with_retry(url, headers=headers, params=params, timeout=15,
                            retries=0)  # No retries — rate limit is tight
    if resp is None or resp.status_code != 200:
        status = resp.status_code if resp else "no response"
        logger.warning("BGeometrics %s returned %s", endpoint, status)
        return None
    try:
        return resp.json()
    except Exception:
        logger.warning("BGeometrics %s invalid JSON", endpoint)
        return None


def _fetch_mvrv(token):
    """Fetch latest MVRV and MVRV Z-Score."""
    data = _api_get("/v1/mvrv/last", token)
    if not data or not isinstance(data, dict):
        return None
    return {
        "mvrv": data.get("mvrv"),
        "mvrv_zscore": data.get("mvrvZScore"),
    }


def _fetch_sopr(token):
    """Fetch latest SOPR (Spent Output Profit Ratio)."""
    data = _api_get("/v1/sopr/last", token)
    if not data or not isinstance(data, dict):
        return None
    return {"sopr": data.get("sopr")}


def _fetch_nupl(token):
    """Fetch latest NUPL (Net Unrealized Profit/Loss)."""
    data = _api_get("/v1/nupl/last", token)
    if not data or not isinstance(data, dict):
        return None
    return {"nupl": data.get("nupl")}


def _fetch_realized_price(token):
    """Fetch latest realized price."""
    data = _api_get("/v1/realized-price/last", token)
    if not data or not isinstance(data, dict):
        return None
    return {"realized_price": data.get("realizedPrice")}


def _fetch_exchange_netflow(token):
    """Fetch latest exchange netflow (negative = accumulation)."""
    data = _api_get("/v1/exchange-netflow", token, params={"size": 1})
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    latest = data[0] if isinstance(data[0], dict) else data[-1]
    return {"netflow": latest.get("netflow")}


def _fetch_liquidations(token):
    """Fetch latest BTC liquidation data."""
    data = _api_get("/v1/btc-liquidations", token, params={"size": 1})
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    latest = data[0] if isinstance(data[0], dict) else data[-1]
    return {
        "long_liquidations": latest.get("longLiquidations"),
        "short_liquidations": latest.get("shortLiquidations"),
    }


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------

def _safe_float(val):
    """Convert API value to float, handling strings and None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _fetch_all_onchain(token):
    """Fetch all 6 on-chain metrics and aggregate into a single dict.

    Adds 1s delay between requests to respect free tier rate limits (8 req/hr).
    """
    result = {"ts": time.time()}

    fetchers = [
        ("mvrv", _fetch_mvrv),
        ("sopr", _fetch_sopr),
        ("nupl", _fetch_nupl),
        ("realized_price", _fetch_realized_price),
        ("exchange_netflow", _fetch_exchange_netflow),
        ("liquidations", _fetch_liquidations),
    ]

    any_success = False
    for i, (name, fetcher) in enumerate(fetchers):
        if i > 0:
            time.sleep(1)  # Rate limit: space out requests
        try:
            data = fetcher(token)
            if data:
                # Convert string values to float
                result.update({k: _safe_float(v) if k != "ts" else v
                              for k, v in data.items()})
                any_success = True
        except Exception:
            logger.warning("BGeometrics %s fetch failed", name, exc_info=True)

    if not any_success:
        return None

    # Save to persistent cache
    _save_onchain_cache(result)
    return result


def get_onchain_data():
    """Get on-chain data for BTC, using in-memory + persistent cache.

    Returns dict with all available metrics, or None if unavailable.
    """
    token = _load_config_token()
    if not token:
        # Try persistent cache even without token
        cached = _load_onchain_cache(max_age_seconds=ONCHAIN_TTL * 2)
        if cached:
            logger.debug("No BGeometrics token, using stale cache")
            return cached
        return None

    return _cached("onchain_btc", ONCHAIN_TTL, _fetch_all_onchain, token)


# ---------------------------------------------------------------------------
# Interpretation helpers (for Layer 2 context)
# ---------------------------------------------------------------------------

def interpret_onchain(data):
    """Interpret on-chain metrics into human-readable zones.

    Returns dict with zone classifications for each available metric.
    """
    if not data:
        return {}

    interp = {}

    # MVRV Z-Score zones
    zscore = _safe_float(data.get("mvrv_zscore"))
    if zscore is not None:
        if zscore < 1:
            interp["mvrv_zone"] = "undervalued"
        elif zscore > 7:
            interp["mvrv_zone"] = "overheated"
        else:
            interp["mvrv_zone"] = "neutral"

    # SOPR zones
    sopr = _safe_float(data.get("sopr"))
    if sopr is not None:
        if sopr < 0.97:
            interp["sopr_zone"] = "capitulation"
        elif sopr > 1.05:
            interp["sopr_zone"] = "profit_taking"
        else:
            interp["sopr_zone"] = "neutral"

    # NUPL zones
    nupl = _safe_float(data.get("nupl"))
    if nupl is not None:
        if nupl < 0:
            interp["nupl_zone"] = "capitulation"
        elif nupl > 0.75:
            interp["nupl_zone"] = "euphoria"
        elif nupl > 0.5:
            interp["nupl_zone"] = "greed"
        elif nupl > 0.25:
            interp["nupl_zone"] = "optimism"
        else:
            interp["nupl_zone"] = "hope"

    # Exchange netflow
    netflow = _safe_float(data.get("netflow"))
    if netflow is not None:
        if netflow < 0:
            interp["netflow_signal"] = "accumulation"
        else:
            interp["netflow_signal"] = "distribution"

    return interp
