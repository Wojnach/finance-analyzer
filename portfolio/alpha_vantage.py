"""Alpha Vantage fundamentals — daily-refresh stock fundamentals cache.

Fetches OVERVIEW endpoint data for stock tickers and caches persistently.
Free tier: 25 requests/day, 5 requests/minute.

Not used for crypto or metals (no OVERVIEW data available).
"""

import json
import logging
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

from portfolio.http_retry import fetch_with_retry
from portfolio.shared_state import _alpha_vantage_limiter
from portfolio.tickers import STOCK_SYMBOLS

logger = logging.getLogger("portfolio.alpha_vantage")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_FILE = DATA_DIR / "fundamentals_cache.json"

AV_BASE_URL = "https://www.alphavantage.co/query"

# Module-level state
_cache = {}  # ticker -> normalized fundamentals dict
_cache_lock = threading.Lock()
_daily_budget_used = 0
_budget_reset_date = ""  # ISO date string for budget tracking
_circuit_breaker_failures = 0
_circuit_breaker_paused_until = 0.0


def load_persistent_cache():
    """Load fundamentals cache from disk on startup."""
    global _cache
    if not CACHE_FILE.exists():
        logger.info("No fundamentals cache found at %s", CACHE_FILE)
        return
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            with _cache_lock:
                _cache = data
            logger.info("Loaded fundamentals for %d tickers from cache", len(data))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load fundamentals cache: %s", e)


def _save_persistent_cache():
    """Write current cache to disk atomically."""
    try:
        from portfolio.portfolio_mgr import _atomic_write_json
        with _cache_lock:
            snapshot = dict(_cache)
        _atomic_write_json(CACHE_FILE, snapshot)
    except Exception as e:
        logger.warning("Failed to save fundamentals cache: %s", e)


def get_fundamentals(ticker):
    """Return cached fundamentals for a ticker, or None if not available."""
    with _cache_lock:
        return _cache.get(ticker)


def get_all_fundamentals():
    """Return all cached fundamentals (for reporting)."""
    with _cache_lock:
        return dict(_cache)


def _normalize_overview(raw):
    """Convert Alpha Vantage OVERVIEW response to clean format.

    AV returns "None" strings for missing values and all values as strings.
    """
    def _float(val, default=None):
        if val is None or val == "None" or val == "-" or val == "":
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _int(val, default=None):
        if val is None or val == "None" or val == "-" or val == "":
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    # Check for error responses
    if "Error Message" in raw or "Note" in raw:
        return None

    # Require at least Symbol to be present
    if not raw.get("Symbol"):
        return None

    result = {
        "pe_ratio": _float(raw.get("PERatio")),
        "forward_pe": _float(raw.get("ForwardPE")),
        "peg_ratio": _float(raw.get("PEGRatio")),
        "eps": _float(raw.get("EPS")),
        "revenue_growth_yoy": _float(raw.get("QuarterlyRevenueGrowthYOY")),
        "earnings_growth_yoy": _float(raw.get("QuarterlyEarningsGrowthYOY")),
        "profit_margin": _float(raw.get("ProfitMargin")),
        "market_cap": _int(raw.get("MarketCapitalization")),
        "sector": raw.get("Sector") if raw.get("Sector") != "None" else None,
        "industry": raw.get("Industry") if raw.get("Industry") != "None" else None,
        "dividend_yield": _float(raw.get("DividendYield")),
        "analyst_target": _float(raw.get("AnalystTargetPrice")),
        "analyst_ratings": {
            "strong_buy": _int(raw.get("AnalystRatingStrongBuy"), 0),
            "buy": _int(raw.get("AnalystRatingBuy"), 0),
            "hold": _int(raw.get("AnalystRatingHold"), 0),
            "sell": _int(raw.get("AnalystRatingSell"), 0),
            "strong_sell": _int(raw.get("AnalystRatingStrongSell"), 0),
        },
        "beta": _float(raw.get("Beta")),
        "w52_high": _float(raw.get("52WeekHigh")),
        "w52_low": _float(raw.get("52WeekLow")),
        "_fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


def _fetch_overview(ticker, api_key):
    """Fetch OVERVIEW data for a single ticker from Alpha Vantage."""
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": api_key,
    }
    resp = fetch_with_retry(
        AV_BASE_URL,
        params=params,
        timeout=15,
        retries=2,
    )
    if resp is None:
        return None
    try:
        data = resp.json()
    except (ValueError, AttributeError):
        return None

    # Rate limit info check
    if isinstance(data, dict) and "Note" in data:
        logger.warning("Alpha Vantage rate limit hit: %s", data["Note"][:100])
        return None

    return data


def _check_budget():
    """Check and reset daily budget counter. Returns True if budget available."""
    global _daily_budget_used, _budget_reset_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _budget_reset_date != today:
        _daily_budget_used = 0
        _budget_reset_date = today
    return _daily_budget_used


def _is_stale(ticker, max_stale_days=5):
    """Check if cached data for ticker is older than max_stale_days."""
    with _cache_lock:
        entry = _cache.get(ticker)
    if not entry:
        return True
    fetched_at = entry.get("_fetched_at")
    if not fetched_at:
        return True
    try:
        fetched_time = datetime.fromisoformat(fetched_at)
        age_seconds = (datetime.now(timezone.utc) - fetched_time).total_seconds()
        return age_seconds > max_stale_days * 86400
    except (ValueError, TypeError):
        return True


def _cache_age_hours(ticker):
    """Return age of cached data in hours, or None if not cached."""
    with _cache_lock:
        entry = _cache.get(ticker)
    if not entry:
        return None
    fetched_at = entry.get("_fetched_at")
    if not fetched_at:
        return None
    try:
        fetched_time = datetime.fromisoformat(fetched_at)
        return (datetime.now(timezone.utc) - fetched_time).total_seconds() / 3600
    except (ValueError, TypeError):
        return None


def refresh_fundamentals_batch(config):
    """Batch-refresh fundamentals for all stock tickers.

    Respects rate limits (5/min) and daily budget (25/day).
    Returns number of successfully refreshed tickers.
    """
    global _daily_budget_used, _circuit_breaker_failures, _circuit_breaker_paused_until

    av_config = config.get("alpha_vantage", {})
    if not av_config.get("enabled", True):
        logger.info("Alpha Vantage disabled in config")
        return 0

    api_key = av_config.get("api_key", "")
    if not api_key:
        logger.warning("Alpha Vantage API key not configured")
        return 0

    daily_budget = av_config.get("daily_budget", 25)
    max_stale_days = av_config.get("max_stale_days", 5)
    skip_tickers = set(av_config.get("skip_tickers", ["QQQ"]))
    cache_ttl_hours = av_config.get("cache_ttl_hours", 24)

    # Check circuit breaker
    if time.time() < _circuit_breaker_paused_until:
        remaining = int(_circuit_breaker_paused_until - time.time())
        logger.info("Alpha Vantage circuit breaker active, %ds remaining", remaining)
        return 0

    budget_used = _check_budget()
    if budget_used >= daily_budget:
        logger.info("Alpha Vantage daily budget exhausted (%d/%d)", budget_used, daily_budget)
        return 0

    # Build refresh list: stock tickers not recently cached, not skipped
    tickers_to_refresh = []
    for ticker in sorted(STOCK_SYMBOLS):
        if ticker in skip_tickers:
            continue
        age = _cache_age_hours(ticker)
        if age is None or age > cache_ttl_hours:
            tickers_to_refresh.append(ticker)

    if not tickers_to_refresh:
        logger.info("All fundamentals fresh, nothing to refresh")
        return 0

    # Cap by remaining budget
    remaining_budget = daily_budget - budget_used
    tickers_to_refresh = tickers_to_refresh[:remaining_budget]

    logger.info(
        "Refreshing fundamentals for %d tickers (budget: %d/%d used)",
        len(tickers_to_refresh), budget_used, daily_budget,
    )

    success_count = 0
    for ticker in tickers_to_refresh:
        # Rate limit
        _alpha_vantage_limiter.wait()

        try:
            raw = _fetch_overview(ticker, api_key)
            if raw is None:
                _circuit_breaker_failures += 1
                if _circuit_breaker_failures >= 3:
                    _circuit_breaker_paused_until = time.time() + 300  # 5 min pause
                    logger.warning("Alpha Vantage circuit breaker tripped after %d failures", _circuit_breaker_failures)
                    break
                continue

            normalized = _normalize_overview(raw)
            if normalized is None:
                logger.warning("Alpha Vantage: empty/error response for %s", ticker)
                _circuit_breaker_failures += 1
                if _circuit_breaker_failures >= 3:
                    _circuit_breaker_paused_until = time.time() + 300
                    logger.warning("Alpha Vantage circuit breaker tripped after %d failures", _circuit_breaker_failures)
                    break
                continue

            with _cache_lock:
                _cache[ticker] = normalized
            _daily_budget_used += 1
            _circuit_breaker_failures = 0  # reset on success
            success_count += 1
            logger.info("Refreshed fundamentals for %s (PE=%.1f, sector=%s)",
                        ticker,
                        normalized.get("pe_ratio") or 0,
                        normalized.get("sector", "?"))

        except Exception as e:
            logger.warning("Alpha Vantage fetch failed for %s: %s", ticker, e)
            _circuit_breaker_failures += 1
            if _circuit_breaker_failures >= 3:
                _circuit_breaker_paused_until = time.time() + 300
                logger.warning("Alpha Vantage circuit breaker tripped")
                break

    if success_count > 0:
        _save_persistent_cache()
        logger.info("Fundamentals refresh complete: %d/%d succeeded", success_count, len(tickers_to_refresh))

    return success_count


def should_batch_refresh(config):
    """Check if a batch refresh should run now.

    Only runs when cache is >24h old. Designed to be called from the main loop.
    """
    av_config = config.get("alpha_vantage", {})
    if not av_config.get("enabled", True) or not av_config.get("api_key", ""):
        return False

    cache_ttl_hours = av_config.get("cache_ttl_hours", 24)

    # Check if any stock ticker needs refresh
    for ticker in STOCK_SYMBOLS:
        if ticker in set(av_config.get("skip_tickers", ["QQQ"])):
            continue
        age = _cache_age_hours(ticker)
        if age is None or age > cache_ttl_hours:
            return True

    return False
