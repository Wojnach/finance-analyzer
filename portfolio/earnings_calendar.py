"""Earnings calendar — proximity gate to prevent trading near earnings.

Fetches upcoming earnings dates from yfinance for US stock tickers.
When a ticker has earnings within GATE_DAYS, BUY signals are gated to HOLD
to avoid unpredictable binary events.

Cache: per-ticker, 24h TTL (earnings dates don't change intraday).
"""

import logging
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime

from portfolio.tickers import STOCK_SYMBOLS

logger = logging.getLogger("portfolio.earnings_calendar")

# Gate window: force HOLD if earnings within this many calendar days
GATE_DAYS = 2

# Cache TTL: 24 hours (earnings dates are stable)
EARNINGS_CACHE_TTL = 86400

# Per-ticker cache: {ticker: {"data": {...}, "time": epoch}}
_earnings_cache: dict[str, dict] = {}
_earnings_lock = threading.Lock()


def _fetch_earnings_alpha_vantage(ticker: str) -> dict | None:
    """Fetch next earnings date from Alpha Vantage EARNINGS endpoint.

    Uses the already-configured AV API key and rate limiter.
    """
    try:
        from portfolio.api_utils import load_config
        from portfolio.http_retry import fetch_with_retry
        from portfolio.shared_state import _alpha_vantage_limiter

        config = load_config()
        # C9/DC-R3-1: key lives under config["alpha_vantage"]["api_key"], not
        # the flat "alpha_vantage_key" key that doesn't exist.
        api_key = config.get("alpha_vantage", {}).get("api_key", "")
        if not api_key:
            return None

        _alpha_vantage_limiter.wait()
        # NOTE: earnings calls bypass alpha_vantage.py's _daily_budget_used counter
        # because there is no public increment function exported from that module.
        # Known limitation — earnings fetches consume 1 AV call each but are not
        # reflected in the budget tracker.  Each ticker only fetches once per 24h.
        r = fetch_with_retry(
            "https://www.alphavantage.co/query",
            params={
                "function": "EARNINGS",
                "symbol": ticker,
                "apikey": api_key,
            },
            timeout=10,
        )
        if r is None:
            return None
        data = r.json()

        # AV EARNINGS returns quarterlyEarnings and annualEarnings
        quarterly = data.get("quarterlyEarnings", [])
        if not quarterly:
            return None

        today = datetime.now(UTC).date()
        # Find the next upcoming earnings (reportedDate in the future or very recent)
        for q in quarterly:
            rd = q.get("reportedDate")
            if not rd or rd == "None":
                continue
            try:
                from datetime import date as _date
                ed = _date.fromisoformat(rd)
                days_until = (ed - today).days
                if days_until >= -1:
                    return {
                        "earnings_date": ed.isoformat(),
                        "days_until": days_until,
                        "gate_active": 0 <= days_until <= GATE_DAYS,
                        "timing": "unknown",
                    }
            except (ValueError, TypeError):
                continue

        return None
    except Exception:
        logger.debug("Alpha Vantage earnings fetch failed for %s", ticker, exc_info=True)
        return None


def _fetch_earnings_yfinance(ticker: str) -> dict | None:
    """Fallback: fetch next earnings date from yfinance."""
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        with suppress(Exception):
            cal = t.calendar
            if cal is not None and not (hasattr(cal, 'empty') and cal.empty):
                if isinstance(cal, dict):
                    earnings_date = cal.get("Earnings Date")
                    if isinstance(earnings_date, list) and earnings_date:
                        earnings_date = earnings_date[0]
                else:
                    if "Earnings Date" in cal.index:
                        earnings_date = cal.loc["Earnings Date"].iloc[0]
                    else:
                        earnings_date = None

                if earnings_date is not None:
                    if hasattr(earnings_date, "date"):
                        ed = earnings_date.date()
                    elif isinstance(earnings_date, str):
                        ed = datetime.fromisoformat(earnings_date).date()
                    else:
                        ed = None

                    if ed:
                        today = datetime.now(UTC).date()
                        days_until = (ed - today).days
                        if days_until < -5:
                            return None
                        return {
                            "earnings_date": ed.isoformat(),
                            "days_until": days_until,
                            "gate_active": 0 <= days_until <= GATE_DAYS,
                            "timing": "unknown",
                        }
        return None
    except Exception:
        logger.debug("yfinance earnings fetch failed for %s", ticker, exc_info=True)
        return None


def _fetch_earnings_date(ticker: str) -> dict | None:
    """Fetch next earnings date — Alpha Vantage primary, yfinance fallback.

    Returns dict with earnings_date, days_until, timing, or None.
    """
    # Primary: Alpha Vantage (already have API key + rate limiter)
    result = _fetch_earnings_alpha_vantage(ticker)
    if result:
        return result

    # Fallback: yfinance
    return _fetch_earnings_yfinance(ticker)


def get_earnings_proximity(ticker: str) -> dict | None:
    """Get cached earnings proximity for a ticker.

    Only works for STOCK_SYMBOLS — returns None for crypto/metals.

    Returns:
        dict with earnings_date, days_until, gate_active, timing
        or None if no upcoming earnings found.
    """
    if ticker not in STOCK_SYMBOLS:
        return None

    now = time.time()
    with _earnings_lock:
        cached = _earnings_cache.get(ticker)
        if cached and now - cached["time"] < EARNINGS_CACHE_TTL:
            return cached["data"]

    # Fetch fresh
    data = _fetch_earnings_date(ticker)

    with _earnings_lock:
        _earnings_cache[ticker] = {"data": data, "time": now}

    return data


def should_gate_earnings(ticker: str) -> bool:
    """Check if ticker should be gated to HOLD due to nearby earnings.

    Returns True if:
    - Ticker is a US stock (not crypto/metals)
    - Earnings are within GATE_DAYS calendar days
    """
    if ticker not in STOCK_SYMBOLS:
        return False

    prox = get_earnings_proximity(ticker)
    if prox is None:
        return False

    return prox.get("gate_active", False)


def get_all_earnings_proximity() -> dict:
    """Get earnings proximity for all stock tickers.

    Returns dict keyed by ticker with proximity info.
    Used by reporting.py to enrich agent_summary.
    """
    result = {}
    for ticker in STOCK_SYMBOLS:
        prox = get_earnings_proximity(ticker)
        if prox:
            result[ticker] = prox
    return result


def clear_cache() -> None:
    """Clear the earnings cache (for testing)."""
    with _earnings_lock:
        _earnings_cache.clear()
