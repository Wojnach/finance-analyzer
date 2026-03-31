"""Binance FAPI order book depth and recent trades for metals.

Fetches L2 snapshots and trade ticks for XAUUSDT / XAGUSDT.
Data feeds into microstructure feature computations (OFI, depth
imbalance, VPIN, spread metrics).

Uses the same rate-limiting and caching patterns as futures_data.py.
"""
from __future__ import annotations

import logging
import time
from functools import wraps

from portfolio.api_utils import BINANCE_FAPI_BASE
from portfolio.http_retry import fetch_json
from portfolio.shared_state import _binance_limiter, _cached

logger = logging.getLogger("portfolio.metals_orderbook")

SYMBOL_MAP = {
    "XAU-USD": "XAUUSDT",
    "XAG-USD": "XAGUSDT",
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
}

_DEPTH_TTL = 10
_TRADES_TTL = 10


def _fetch_fapi_json(url, params=None, timeout=10):
    """Fetch JSON from Binance FAPI with rate limiting and retry."""
    _binance_limiter.wait()
    return fetch_json(url, params=params, timeout=timeout, label="metals_orderbook")


def _nocache(func):
    """Mark a function so tests can call func.__wrapped__ to bypass _cached."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__wrapped__ = func
    return wrapper


@_nocache
def get_orderbook_depth(ticker: str, limit: int = 20) -> dict | None:
    """Fetch order book depth snapshot from Binance FAPI.

    Returns dict with bids, asks (as [[price, qty], ...] floats), best_bid, best_ask,
    mid_price, spread, spread_bps. None on failure.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_fapi_json(
            f"{BINANCE_FAPI_BASE}/depth",
            params={"symbol": symbol, "limit": limit},
        )
        if data is None or "bids" not in data or "asks" not in data:
            return None
        bids = [[float(p), float(q)] for p, q in data["bids"]]
        asks = [[float(p), float(q)] for p, q in data["asks"]]
        if not bids or not asks:
            return None
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread": spread,
            "spread_bps": (spread / mid) * 10000 if mid > 0 else 0.0,
            "bid_depth_total": sum(q for _, q in bids),
            "ask_depth_total": sum(q for _, q in asks),
            "ts": int(time.time() * 1000),
        }

    return _cached(f"depth_{ticker}_{limit}", _DEPTH_TTL, _fetch)


@_nocache
def get_recent_trades(ticker: str, limit: int = 100) -> list[dict] | None:
    """Fetch recent trades from Binance FAPI.

    Each trade includes a sign: +1 for buyer-initiated (taker buy),
    -1 for seller-initiated (taker sell). isBuyerMaker=True means the maker
    was the buyer, so the taker (aggressor) was the seller -> sign = -1.
    """
    if ticker not in SYMBOL_MAP:
        return None
    symbol = SYMBOL_MAP[ticker]

    def _fetch():
        data = _fetch_fapi_json(
            f"{BINANCE_FAPI_BASE}/trades",
            params={"symbol": symbol, "limit": limit},
        )
        if not data:
            return None
        return [
            {
                "id": d["id"],
                "price": float(d["price"]),
                "qty": float(d["qty"]),
                "time": d["time"],
                "is_buyer_maker": d.get("isBuyerMaker", False),
                "sign": -1 if d.get("isBuyerMaker", False) else 1,
            }
            for d in data
        ]

    return _cached(f"trades_{ticker}_{limit}", _TRADES_TTL, _fetch)
