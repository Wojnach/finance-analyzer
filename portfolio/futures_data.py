"""Binance FAPI futures data — open interest, long/short ratios, funding history.

Fetches public endpoints for crypto tickers (BTC-USD, ETH-USD only).
Uses existing infrastructure: fetch_with_retry, _cached, _binance_limiter.
"""

import logging
import time

from portfolio.api_utils import BINANCE_FAPI_BASE, BINANCE_FUTURES_DATA
from portfolio.http_retry import fetch_with_retry
from portfolio.shared_state import _cached, _binance_limiter

logger = logging.getLogger("portfolio.futures_data")

SYMBOL_MAP = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}

# Cache TTLs
_OI_TTL = 300        # 5 min
_LS_TTL = 300        # 5 min
_FUNDING_TTL = 900   # 15 min


def _fetch_json(url, params=None, timeout=10):
    """Fetch JSON from Binance FAPI with rate limiting and retry."""
    _binance_limiter.wait()
    r = fetch_with_retry(url, params=params, timeout=timeout)
    if r is None:
        return None
    if r.status_code != 200:
        logger.warning("Binance FAPI %s returned %d", url, r.status_code)
        return None
    return r.json()


def get_open_interest(ticker):
    """Current open interest for a crypto ticker.

    Returns: {oi, oi_usdt, symbol, time} or None.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_json(
            f"{BINANCE_FAPI_BASE}/openInterest",
            params={"symbol": symbol},
        )
        if data is None:
            return None
        return {
            "oi": float(data["openInterest"]),
            "symbol": data["symbol"],
            "time": data.get("time", int(time.time() * 1000)),
        }

    return _cached(f"futures_oi_{ticker}", _OI_TTL, _fetch)


def get_open_interest_history(ticker, period="5m", limit=30):
    """Historical open interest snapshots.

    Returns: list of {oi, oi_usdt, timestamp} or None.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_json(
            f"{BINANCE_FUTURES_DATA}/openInterestHist",
            params={"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "oi": float(d["sumOpenInterest"]),
                "oi_usdt": float(d["sumOpenInterestValue"]),
                "timestamp": d["timestamp"],
            }
            for d in data
        ]

    return _cached(f"futures_oi_hist_{ticker}_{period}", _OI_TTL, _fetch)


def get_long_short_ratio(ticker, period="5m", limit=30):
    """Global long/short account ratio.

    Returns: list of {longShortRatio, longAccount, shortAccount, timestamp} or None.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_json(
            f"{BINANCE_FUTURES_DATA}/globalLongShortAccountRatio",
            params={"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "longShortRatio": float(d["longShortRatio"]),
                "longAccount": float(d["longAccount"]),
                "shortAccount": float(d["shortAccount"]),
                "timestamp": d["timestamp"],
            }
            for d in data
        ]

    return _cached(f"futures_ls_{ticker}_{period}", _LS_TTL, _fetch)


def get_top_trader_position_ratio(ticker, period="5m", limit=30):
    """Top trader long/short position ratio.

    Returns: list of {longShortRatio, longAccount, shortAccount, timestamp} or None.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_json(
            f"{BINANCE_FUTURES_DATA}/topLongShortPositionRatio",
            params={"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "longShortRatio": float(d["longShortRatio"]),
                "longAccount": float(d["longAccount"]),
                "shortAccount": float(d["shortAccount"]),
                "timestamp": d["timestamp"],
            }
            for d in data
        ]

    return _cached(f"futures_top_pos_{ticker}_{period}", _LS_TTL, _fetch)


def get_top_trader_account_ratio(ticker, period="5m", limit=30):
    """Top trader long/short account ratio.

    Returns: list of {longShortRatio, longAccount, shortAccount, timestamp} or None.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_json(
            f"{BINANCE_FUTURES_DATA}/topLongShortAccountRatio",
            params={"symbol": symbol, "period": period, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "longShortRatio": float(d["longShortRatio"]),
                "longAccount": float(d["longAccount"]),
                "shortAccount": float(d["shortAccount"]),
                "timestamp": d["timestamp"],
            }
            for d in data
        ]

    return _cached(f"futures_top_acct_{ticker}_{period}", _LS_TTL, _fetch)


def get_funding_rate_history(ticker, limit=100):
    """Historical funding rates.

    Returns: list of {fundingRate, fundingTime, symbol} or None.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_json(
            f"{BINANCE_FAPI_BASE}/fundingRate",
            params={"symbol": symbol, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "fundingRate": float(d["fundingRate"]),
                "fundingTime": d["fundingTime"],
                "symbol": d.get("symbol", symbol),
            }
            for d in data
        ]

    return _cached(f"futures_funding_hist_{ticker}", _FUNDING_TTL, _fetch)


def get_all_futures_data(ticker):
    """Fetch all futures data for a ticker. Each sub-key can be None on failure.

    Returns: dict with keys: open_interest, oi_history, ls_ratio,
             top_position_ratio, top_account_ratio, funding_history.
    """
    if ticker not in SYMBOL_MAP:
        return None

    return {
        "open_interest": get_open_interest(ticker),
        "oi_history": get_open_interest_history(ticker),
        "ls_ratio": get_long_short_ratio(ticker),
        "top_position_ratio": get_top_trader_position_ratio(ticker),
        "top_account_ratio": get_top_trader_account_ratio(ticker),
        "funding_history": get_funding_rate_history(ticker),
    }


if __name__ == "__main__":
    import json
    for t in ["BTC-USD", "ETH-USD", "NVDA"]:
        print(f"\n=== {t} ===")
        result = get_all_futures_data(t)
        if result is None:
            print("  Not a crypto ticker — skipped")
        else:
            for k, v in result.items():
                if v is None:
                    print(f"  {k}: None (fetch failed)")
                elif isinstance(v, list):
                    print(f"  {k}: {len(v)} entries, latest={v[-1] if v else 'empty'}")
                else:
                    print(f"  {k}: {json.dumps(v, indent=2)}")
