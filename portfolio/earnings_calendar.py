"""Earnings calendar — proximity gate to prevent trading near earnings.

Fetches upcoming earnings dates from yfinance for US stock tickers.
When a ticker has earnings within GATE_DAYS, BUY signals are gated to HOLD
to avoid unpredictable binary events.

Cache: per-ticker, 24h TTL (earnings dates don't change intraday).
"""

import logging
import time
import threading
from datetime import UTC, datetime, timedelta

from portfolio.tickers import STOCK_SYMBOLS

logger = logging.getLogger("portfolio.earnings_calendar")

# Gate window: force HOLD if earnings within this many calendar days
GATE_DAYS = 2

# Cache TTL: 24 hours (earnings dates are stable)
EARNINGS_CACHE_TTL = 86400

# Per-ticker cache: {ticker: {"data": {...}, "time": epoch}}
_earnings_cache: dict[str, dict] = {}
_earnings_lock = threading.Lock()


def _fetch_earnings_date(ticker: str) -> dict | None:
    """Fetch next earnings date from yfinance.

    Returns dict with earnings_date, days_until, timing, or None.
    """
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)

        # Try .calendar first (has earnings date + timing)
        try:
            cal = t.calendar
            if cal is not None and not (hasattr(cal, 'empty') and cal.empty):
                # yfinance returns calendar as dict or DataFrame depending on version
                if isinstance(cal, dict):
                    earnings_date = cal.get("Earnings Date")
                    if isinstance(earnings_date, list) and earnings_date:
                        earnings_date = earnings_date[0]
                else:
                    # DataFrame — look for Earnings Date row
                    if "Earnings Date" in cal.index:
                        earnings_date = cal.loc["Earnings Date"].iloc[0]
                    else:
                        earnings_date = None

                if earnings_date is not None:
                    # Convert to date
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
                            # Past earnings, not useful
                            return None
                        return {
                            "earnings_date": ed.isoformat(),
                            "days_until": days_until,
                            "gate_active": 0 <= days_until <= GATE_DAYS,
                            "timing": "unknown",
                        }
        except Exception:
            pass  # calendar not available, try earnings_dates

        # Fallback: earnings_dates property
        try:
            dates = t.earnings_dates
            if dates is not None and not dates.empty:
                today = datetime.now(UTC).date()
                for idx in dates.index:
                    if hasattr(idx, "date"):
                        ed = idx.date()
                    else:
                        continue
                    days_until = (ed - today).days
                    if days_until >= -1:  # future or very recent
                        return {
                            "earnings_date": ed.isoformat(),
                            "days_until": days_until,
                            "gate_active": 0 <= days_until <= GATE_DAYS,
                            "timing": "unknown",
                        }
        except Exception:
            pass

        return None

    except Exception:
        logger.debug("Failed to fetch earnings for %s", ticker, exc_info=True)
        return None


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
