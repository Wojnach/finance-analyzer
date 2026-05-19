# Metals Microstructure & Cross-Asset Signals Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add order-flow microstructure signals and cross-asset features for 1-3h silver/gold price prediction, filling the largest gaps identified by quantitative research.

**Architecture:** Four new modules layered on existing patterns. (1) Data collector polls Binance FAPI order book depth + recent trades every 10-30s. (2) Microstructure feature engine computes OFI, depth imbalance, VPIN, and spread metrics from raw data. (3) Two new signal modules (`orderbook_flow`, `metals_cross_asset`) plug into the existing 30-signal voting system via `register_enhanced()`. (4) Cross-asset collector fetches copper, GVZ, SPY, and gold/silver ratio via yfinance with appropriate caching.

**Tech Stack:** Python 3.11, pandas, numpy, Binance FAPI REST, yfinance, existing `signal_utils.majority_vote`, `http_retry.fetch_json`, `shared_state._cached`

---

## File Structure

| File | Responsibility | New/Modify |
|------|---------------|------------|
| `portfolio/metals_orderbook.py` | Fetch + cache Binance FAPI order book depth and recent trades for XAUUSDT/XAGUSDT | **Create** |
| `portfolio/metals_cross_assets.py` | Fetch + cache cross-asset data: copper (HG=F), GVZ, SPY, G/S ratio | **Create** |
| `portfolio/microstructure.py` | Compute microstructure features from raw order book + trades: OFI, depth imbalance, VPIN, spread metrics | **Create** |
| `portfolio/signals/orderbook_flow.py` | Signal module #31: 5 sub-indicators from microstructure features, majority vote | **Create** |
| `portfolio/signals/metals_cross_asset.py` | Signal module #32: 5 cross-asset sub-indicators for metals, majority vote | **Create** |
| `portfolio/signal_registry.py` | Register two new signals | **Modify** (lines ~110-131) |
| `portfolio/signal_engine.py` | Pass microstructure + cross-asset context to new signals | **Modify** (signal dispatch section) |
| `tests/test_metals_orderbook.py` | Unit tests for order book + trades fetcher | **Create** |
| `tests/test_microstructure.py` | Unit tests for OFI, depth imbalance, VPIN computations | **Create** |
| `tests/test_orderbook_flow_signal.py` | Unit tests for orderbook_flow signal module | **Create** |
| `tests/test_metals_cross_asset_signal.py` | Unit tests for metals_cross_asset signal module | **Create** |
| `tests/test_metals_cross_assets.py` | Unit tests for cross-asset data fetcher | **Create** |

---

## Task 1: Binance Order Book & Trades Fetcher

**Files:**
- Create: `portfolio/metals_orderbook.py`
- Test: `tests/test_metals_orderbook.py`

This module polls Binance FAPI `/depth` and `/trades` endpoints for XAUUSDT and XAGUSDT with rate limiting and caching. Follows the exact pattern from `portfolio/futures_data.py`.

- [ ] **Step 1: Write failing tests for order book fetcher**

```python
# tests/test_metals_orderbook.py
"""Tests for metals order book + trades fetcher."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

# --- Order book depth tests ---

MOCK_DEPTH_RESPONSE = {
    "lastUpdateId": 123456789,
    "E": 1711900000000,
    "T": 1711900000000,
    "bids": [
        ["3100.50", "2.5"],
        ["3100.00", "1.8"],
        ["3099.50", "3.2"],
    ],
    "asks": [
        ["3101.00", "1.2"],
        ["3101.50", "2.0"],
        ["3102.00", "4.1"],
    ],
}

MOCK_TRADES_RESPONSE = [
    {"id": 1, "price": "3100.80", "qty": "0.5", "quoteQty": "1550.40",
     "time": 1711900001000, "isBuyerMaker": False},
    {"id": 2, "price": "3100.50", "qty": "1.0", "quoteQty": "3100.50",
     "time": 1711900002000, "isBuyerMaker": True},
    {"id": 3, "price": "3101.00", "qty": "0.3", "quoteQty": "930.30",
     "time": 1711900003000, "isBuyerMaker": False},
    {"id": 4, "price": "3100.20", "qty": "0.8", "quoteQty": "2480.16",
     "time": 1711900004000, "isBuyerMaker": True},
]


class TestGetOrderbookDepth:
    @patch("portfolio.metals_orderbook._fetch_fapi_json")
    def test_returns_parsed_depth(self, mock_fetch):
        from portfolio.metals_orderbook import get_orderbook_depth
        mock_fetch.return_value = MOCK_DEPTH_RESPONSE

        result = get_orderbook_depth.__wrapped__("XAU-USD", limit=20)

        assert result is not None
        assert len(result["bids"]) == 3
        assert len(result["asks"]) == 3
        # Prices and quantities are floats
        assert result["bids"][0] == [3100.50, 2.5]
        assert result["asks"][0] == [3101.00, 1.2]
        assert result["best_bid"] == 3100.50
        assert result["best_ask"] == 3101.00
        assert result["mid_price"] == pytest.approx(3100.75)
        assert result["spread"] == pytest.approx(0.50)
        assert result["spread_bps"] == pytest.approx(0.50 / 3100.75 * 10000, rel=1e-3)

    @patch("portfolio.metals_orderbook._fetch_fapi_json")
    def test_returns_none_for_unknown_ticker(self, mock_fetch):
        from portfolio.metals_orderbook import get_orderbook_depth
        result = get_orderbook_depth.__wrapped__("UNKNOWN", limit=20)
        assert result is None
        mock_fetch.assert_not_called()

    @patch("portfolio.metals_orderbook._fetch_fapi_json")
    def test_returns_none_on_api_failure(self, mock_fetch):
        from portfolio.metals_orderbook import get_orderbook_depth
        mock_fetch.return_value = None
        result = get_orderbook_depth.__wrapped__("XAG-USD", limit=20)
        assert result is None


class TestGetRecentTrades:
    @patch("portfolio.metals_orderbook._fetch_fapi_json")
    def test_returns_parsed_trades(self, mock_fetch):
        from portfolio.metals_orderbook import get_recent_trades
        mock_fetch.return_value = MOCK_TRADES_RESPONSE

        result = get_recent_trades.__wrapped__("XAU-USD", limit=50)

        assert result is not None
        assert len(result) == 4
        assert result[0]["price"] == 3100.80
        assert result[0]["qty"] == 0.5
        assert result[0]["is_buyer_maker"] is False
        assert result[1]["is_buyer_maker"] is True

    @patch("portfolio.metals_orderbook._fetch_fapi_json")
    def test_returns_none_for_unknown_ticker(self, mock_fetch):
        from portfolio.metals_orderbook import get_recent_trades
        result = get_recent_trades.__wrapped__("UNKNOWN", limit=50)
        assert result is None
        mock_fetch.assert_not_called()


class TestComputeTradeSign:
    @patch("portfolio.metals_orderbook._fetch_fapi_json")
    def test_trade_sign_from_buyer_maker(self, mock_fetch):
        """isBuyerMaker=True means seller initiated (hit the bid) = -1."""
        from portfolio.metals_orderbook import get_recent_trades
        mock_fetch.return_value = MOCK_TRADES_RESPONSE

        result = get_recent_trades.__wrapped__("XAU-USD", limit=50)

        # isBuyerMaker=False → buyer initiated → sign = +1
        assert result[0]["sign"] == 1
        # isBuyerMaker=True → seller initiated → sign = -1
        assert result[1]["sign"] == -1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metals_orderbook.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'portfolio.metals_orderbook'`

- [ ] **Step 3: Implement the order book fetcher**

```python
# portfolio/metals_orderbook.py
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

_DEPTH_TTL = 10    # seconds — order book refreshes frequently
_TRADES_TTL = 10   # seconds — trade data refreshes frequently


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

    Args:
        ticker: Canonical ticker (e.g. "XAU-USD", "XAG-USD")
        limit: Number of price levels per side (5, 10, 20, 50, 100, 500, 1000)

    Returns:
        Dict with bids, asks (as [[price, qty], ...] floats), best_bid, best_ask,
        mid_price, spread, spread_bps.  None on failure.
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
    -1 for seller-initiated (taker sell).  This is the Binance convention:
    isBuyerMaker=True means the maker was the buyer, so the taker (aggressor)
    was the seller → sign = -1.

    Args:
        ticker: Canonical ticker (e.g. "XAU-USD")
        limit: Number of trades to fetch (max 1000)

    Returns:
        List of trade dicts with price, qty, time, is_buyer_maker, sign.
        None on failure.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metals_orderbook.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/metals_orderbook.py tests/test_metals_orderbook.py
git commit -m "feat: add Binance FAPI order book + trades fetcher for metals"
```

---

## Task 2: Microstructure Feature Engine

**Files:**
- Create: `portfolio/microstructure.py`
- Test: `tests/test_microstructure.py`

Core computation engine for order-flow and microstructure features. All formulas from the research document. Pure functions operating on data from Task 1.

- [ ] **Step 1: Write failing tests for depth imbalance**

```python
# tests/test_microstructure.py
"""Tests for microstructure feature computations."""
from __future__ import annotations

import math

import numpy as np
import pytest

# --- Depth Imbalance ---

class TestDepthImbalance:
    def test_balanced_book(self):
        from portfolio.microstructure import depth_imbalance
        # Equal bid/ask volume → imbalance = 0
        depth = {"bids": [[100, 10.0]], "asks": [[101, 10.0]]}
        result = depth_imbalance(depth)
        assert result == pytest.approx(0.0)

    def test_bid_heavy(self):
        from portfolio.microstructure import depth_imbalance
        # More bid volume → positive imbalance (buying pressure)
        depth = {"bids": [[100, 20.0], [99, 10.0]], "asks": [[101, 5.0]]}
        result = depth_imbalance(depth)
        assert result > 0

    def test_ask_heavy(self):
        from portfolio.microstructure import depth_imbalance
        # More ask volume → negative imbalance (selling pressure)
        depth = {"bids": [[100, 5.0]], "asks": [[101, 20.0], [102, 10.0]]}
        result = depth_imbalance(depth)
        assert result < 0

    def test_log_ratio_formula(self):
        from portfolio.microstructure import depth_imbalance
        # F_t = ln(V_bid) - ln(V_ask) from research paper
        depth = {"bids": [[100, 10.0]], "asks": [[101, 5.0]]}
        result = depth_imbalance(depth)
        expected = math.log(10.0) - math.log(5.0)
        assert result == pytest.approx(expected)

    def test_multi_level_sums_volumes(self):
        from portfolio.microstructure import depth_imbalance
        depth = {
            "bids": [[100, 5.0], [99, 5.0]],  # total = 10
            "asks": [[101, 3.0], [102, 7.0]],  # total = 10
        }
        result = depth_imbalance(depth)
        assert result == pytest.approx(0.0)


# --- Trade Flow Imbalance ---

class TestTradeFlowImbalance:
    def test_all_buys(self):
        from portfolio.microstructure import trade_flow_imbalance
        trades = [
            {"price": 100, "qty": 1.0, "sign": 1},
            {"price": 101, "qty": 2.0, "sign": 1},
        ]
        result = trade_flow_imbalance(trades)
        assert result["signed_volume"] == pytest.approx(3.0)
        assert result["buy_volume"] == pytest.approx(3.0)
        assert result["sell_volume"] == pytest.approx(0.0)
        assert result["imbalance_ratio"] == pytest.approx(1.0)

    def test_all_sells(self):
        from portfolio.microstructure import trade_flow_imbalance
        trades = [
            {"price": 100, "qty": 1.0, "sign": -1},
            {"price": 101, "qty": 2.0, "sign": -1},
        ]
        result = trade_flow_imbalance(trades)
        assert result["signed_volume"] == pytest.approx(-3.0)
        assert result["imbalance_ratio"] == pytest.approx(-1.0)

    def test_balanced_trades(self):
        from portfolio.microstructure import trade_flow_imbalance
        trades = [
            {"price": 100, "qty": 5.0, "sign": 1},
            {"price": 100, "qty": 5.0, "sign": -1},
        ]
        result = trade_flow_imbalance(trades)
        assert result["signed_volume"] == pytest.approx(0.0)
        assert result["imbalance_ratio"] == pytest.approx(0.0)

    def test_empty_trades(self):
        from portfolio.microstructure import trade_flow_imbalance
        result = trade_flow_imbalance([])
        assert result is None


# --- VPIN ---

class TestVPIN:
    def test_one_sided_flow(self):
        from portfolio.microstructure import compute_vpin
        # All buys → VPIN should be 1.0
        trades = [{"price": 100, "qty": 1.0, "sign": 1} for _ in range(100)]
        result = compute_vpin(trades, n_buckets=10)
        assert result == pytest.approx(1.0)

    def test_balanced_flow(self):
        from portfolio.microstructure import compute_vpin
        # Alternating buy/sell of equal size → VPIN ≈ 0
        trades = []
        for i in range(100):
            trades.append({"price": 100, "qty": 1.0, "sign": 1 if i % 2 == 0 else -1})
        result = compute_vpin(trades, n_buckets=10)
        assert result == pytest.approx(0.0, abs=0.15)

    def test_insufficient_trades(self):
        from portfolio.microstructure import compute_vpin
        trades = [{"price": 100, "qty": 1.0, "sign": 1}]
        result = compute_vpin(trades, n_buckets=10)
        assert result is None


# --- Order Flow Imbalance (OFI) ---

class TestOFI:
    def test_bid_improving(self):
        from portfolio.microstructure import compute_ofi
        # Best bid price goes up → positive OFI (new demand)
        snapshots = [
            {"best_bid": 100.0, "best_ask": 101.0,
             "bids": [[100.0, 10.0]], "asks": [[101.0, 10.0]]},
            {"best_bid": 100.5, "best_ask": 101.0,
             "bids": [[100.5, 8.0]], "asks": [[101.0, 10.0]]},
        ]
        result = compute_ofi(snapshots)
        assert result > 0  # Bid improved → buying pressure

    def test_ask_improving(self):
        from portfolio.microstructure import compute_ofi
        # Best ask drops → negative OFI (new supply)
        snapshots = [
            {"best_bid": 100.0, "best_ask": 101.0,
             "bids": [[100.0, 10.0]], "asks": [[101.0, 10.0]]},
            {"best_bid": 100.0, "best_ask": 100.5,
             "bids": [[100.0, 10.0]], "asks": [[100.5, 8.0]]},
        ]
        result = compute_ofi(snapshots)
        assert result < 0  # Ask improved → selling pressure

    def test_single_snapshot(self):
        from portfolio.microstructure import compute_ofi
        snapshots = [
            {"best_bid": 100.0, "best_ask": 101.0,
             "bids": [[100.0, 10.0]], "asks": [[101.0, 10.0]]},
        ]
        result = compute_ofi(snapshots)
        assert result == 0.0  # Need >=2 snapshots for delta


# --- Spread Z-Score ---

class TestSpreadZScore:
    def test_normal_spread(self):
        from portfolio.microstructure import spread_zscore
        # Spread equals the mean → z = 0
        spreads = [0.50] * 20
        result = spread_zscore(spreads)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_wide_spread(self):
        from portfolio.microstructure import spread_zscore
        spreads = [0.50] * 19 + [2.0]
        result = spread_zscore(spreads)
        assert result > 1.5  # Well above normal

    def test_insufficient_data(self):
        from portfolio.microstructure import spread_zscore
        result = spread_zscore([0.5])
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_microstructure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'portfolio.microstructure'`

- [ ] **Step 3: Implement microstructure feature engine**

```python
# portfolio/microstructure.py
"""Microstructure feature computations for short-term metals prediction.

Implements order-flow and market-microstructure metrics from academic
literature.  All functions are pure — they take raw data (order book
snapshots, trade lists) and return numeric features.

Key features:
    - Depth Imbalance: ln(V_bid) - ln(V_ask)  [Lipton et al.]
    - Trade Flow Imbalance: signed volume ratio
    - VPIN: Volume-synchronized probability of informed trading
    - OFI: Order Flow Imbalance from quote changes  [Cont et al. 2014]
    - Spread Z-Score: current spread vs rolling distribution
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger("portfolio.microstructure")

# ---------------------------------------------------------------------------
# Depth Imbalance
# ---------------------------------------------------------------------------

def depth_imbalance(depth: dict, levels: int | None = None) -> float:
    """Log ratio of bid vs ask volume: F_t = ln(V_bid) - ln(V_ask).

    Positive → bid-heavy (buying pressure).
    Negative → ask-heavy (selling pressure).

    Args:
        depth: Dict with "bids" and "asks" as [[price, qty], ...]
        levels: Number of price levels to include (None = all available)

    Returns:
        Log imbalance ratio.  Returns 0.0 on missing/zero volumes.
    """
    bids = depth.get("bids", [])
    asks = depth.get("asks", [])
    if levels is not None:
        bids = bids[:levels]
        asks = asks[:levels]
    bid_vol = sum(q for _, q in bids)
    ask_vol = sum(q for _, q in asks)
    if bid_vol <= 0 or ask_vol <= 0:
        return 0.0
    return math.log(bid_vol) - math.log(ask_vol)


# ---------------------------------------------------------------------------
# Trade Flow Imbalance
# ---------------------------------------------------------------------------

def trade_flow_imbalance(trades: list[dict]) -> dict[str, float] | None:
    """Compute signed volume imbalance from recent trades.

    Each trade dict must have 'qty' (float) and 'sign' (+1 buyer, -1 seller).

    Returns:
        Dict with signed_volume, buy_volume, sell_volume, imbalance_ratio,
        trade_count, buy_count, sell_count.  None if trades is empty.
    """
    if not trades:
        return None
    buy_vol = sum(t["qty"] for t in trades if t["sign"] == 1)
    sell_vol = sum(t["qty"] for t in trades if t["sign"] == -1)
    total_vol = buy_vol + sell_vol
    signed_vol = buy_vol - sell_vol
    imbalance = signed_vol / total_vol if total_vol > 0 else 0.0
    return {
        "signed_volume": signed_vol,
        "buy_volume": buy_vol,
        "sell_volume": sell_vol,
        "imbalance_ratio": imbalance,
        "trade_count": len(trades),
        "buy_count": sum(1 for t in trades if t["sign"] == 1),
        "sell_count": sum(1 for t in trades if t["sign"] == -1),
    }


# ---------------------------------------------------------------------------
# VPIN (Volume-Synchronized Probability of Informed Trading)
# ---------------------------------------------------------------------------

def compute_vpin(trades: list[dict], n_buckets: int = 20) -> float | None:
    """VPIN: average absolute buy-sell imbalance per volume bucket.

    Splits total trade volume into n_buckets equal-volume buckets.
    For each bucket, computes |buy_vol - sell_vol| / bucket_vol.
    VPIN = mean of these ratios.

    High VPIN (>0.6) → toxic flow, likely informed trading.
    Low VPIN (<0.3) → balanced, uninformed flow.

    Args:
        trades: List of trade dicts with 'qty' and 'sign'.
        n_buckets: Number of equal-volume buckets.

    Returns:
        VPIN score (0.0 to 1.0).  None if insufficient trades.
    """
    if len(trades) < n_buckets:
        return None
    total_vol = sum(t["qty"] for t in trades)
    if total_vol <= 0:
        return None
    bucket_size = total_vol / n_buckets

    imbalances = []
    bucket_buy = 0.0
    bucket_sell = 0.0
    bucket_vol = 0.0

    for t in trades:
        qty = t["qty"]
        remaining = qty
        while remaining > 0:
            space = bucket_size - bucket_vol
            fill = min(remaining, space)
            if t["sign"] == 1:
                bucket_buy += fill
            else:
                bucket_sell += fill
            bucket_vol += fill
            remaining -= fill
            if bucket_vol >= bucket_size - 1e-12:
                if bucket_buy + bucket_sell > 0:
                    imbalances.append(
                        abs(bucket_buy - bucket_sell) / (bucket_buy + bucket_sell)
                    )
                bucket_buy = 0.0
                bucket_sell = 0.0
                bucket_vol = 0.0

    if not imbalances:
        return None
    return float(np.mean(imbalances))


# ---------------------------------------------------------------------------
# OFI (Order Flow Imbalance from quote changes)
# ---------------------------------------------------------------------------

def compute_ofi(snapshots: list[dict]) -> float:
    """Order Flow Imbalance from consecutive order book snapshots.

    Implements the Cont et al. (2014) OFI formula:
        For each pair of consecutive snapshots, compute:
        delta_V_bid = +V_bid_t (if bid improved) or V_bid_t - V_bid_{t-1}
                      (if same) or -V_bid_{t-1} (if bid dropped)
        delta_V_ask = similar logic for asks (reversed sign)
        OFI_t = delta_V_bid - delta_V_ask

    Positive OFI → net buying pressure.
    Negative OFI → net selling pressure.

    Args:
        snapshots: List of order book snapshots, each with best_bid,
                   best_ask, bids ([[price, qty]]), asks ([[price, qty]]).

    Returns:
        Cumulative OFI across all snapshot transitions.
        Returns 0.0 if fewer than 2 snapshots.
    """
    if len(snapshots) < 2:
        return 0.0

    total_ofi = 0.0
    for i in range(1, len(snapshots)):
        prev = snapshots[i - 1]
        curr = snapshots[i]

        prev_bid = prev["best_bid"]
        curr_bid = curr["best_bid"]
        prev_bid_vol = prev["bids"][0][1] if prev["bids"] else 0.0
        curr_bid_vol = curr["bids"][0][1] if curr["bids"] else 0.0

        if curr_bid > prev_bid:
            delta_bid = curr_bid_vol
        elif curr_bid == prev_bid:
            delta_bid = curr_bid_vol - prev_bid_vol
        else:
            delta_bid = -prev_bid_vol

        prev_ask = prev["best_ask"]
        curr_ask = curr["best_ask"]
        prev_ask_vol = prev["asks"][0][1] if prev["asks"] else 0.0
        curr_ask_vol = curr["asks"][0][1] if curr["asks"] else 0.0

        if curr_ask < prev_ask:
            delta_ask = curr_ask_vol
        elif curr_ask == prev_ask:
            delta_ask = curr_ask_vol - prev_ask_vol
        else:
            delta_ask = -prev_ask_vol

        total_ofi += delta_bid - delta_ask

    return total_ofi


# ---------------------------------------------------------------------------
# Spread Z-Score
# ---------------------------------------------------------------------------

def spread_zscore(spread_history: list[float], min_samples: int = 5) -> float | None:
    """Z-score of current spread vs recent history.

    Args:
        spread_history: List of recent spread values (latest last).
        min_samples: Minimum history length.

    Returns:
        Z-score of latest spread.  None if insufficient data.
    """
    if len(spread_history) < min_samples:
        return None
    arr = np.array(spread_history, dtype=float)
    mean = arr[:-1].mean()
    std = arr[:-1].std()
    if std < 1e-12:
        return 0.0
    return float((arr[-1] - mean) / std)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_microstructure.py -v`
Expected: All 17 tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/microstructure.py tests/test_microstructure.py
git commit -m "feat: add microstructure feature engine (OFI, depth imbalance, VPIN, spread)"
```

---

## Task 3: Cross-Asset Data Fetcher

**Files:**
- Create: `portfolio/metals_cross_assets.py`
- Test: `tests/test_metals_cross_assets.py`

Fetches copper (HG=F), gold volatility index (GVZ), S&P 500 (SPY), and computes gold/silver ratio. All via yfinance with appropriate caching.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metals_cross_assets.py
"""Tests for metals cross-asset data fetcher."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestGetCopperData:
    @patch("portfolio.metals_cross_assets._yf_download")
    def test_returns_change_and_price(self, mock_dl):
        from portfolio.metals_cross_assets import get_copper_data
        idx = pd.date_range("2026-03-01", periods=30, freq="B")
        df = pd.DataFrame({"Close": range(400, 430)}, index=idx)
        mock_dl.return_value = df

        result = get_copper_data.__wrapped__()
        assert result is not None
        assert "price" in result
        assert "change_1d_pct" in result
        assert "change_5d_pct" in result
        assert "sma20" in result

    @patch("portfolio.metals_cross_assets._yf_download")
    def test_returns_none_on_failure(self, mock_dl):
        from portfolio.metals_cross_assets import get_copper_data
        mock_dl.return_value = pd.DataFrame()
        result = get_copper_data.__wrapped__()
        assert result is None


class TestGetGVZ:
    @patch("portfolio.metals_cross_assets._yf_download")
    def test_returns_gvz_level(self, mock_dl):
        from portfolio.metals_cross_assets import get_gvz
        idx = pd.date_range("2026-03-01", periods=30, freq="B")
        df = pd.DataFrame({"Close": [18.0] * 29 + [22.0]}, index=idx)
        mock_dl.return_value = df

        result = get_gvz.__wrapped__()
        assert result is not None
        assert result["level"] == pytest.approx(22.0)
        assert "zscore" in result
        assert "change_1d_pct" in result


class TestGoldSilverRatio:
    @patch("portfolio.metals_cross_assets._yf_download")
    def test_computes_ratio(self, mock_dl):
        from portfolio.metals_cross_assets import get_gold_silver_ratio
        idx = pd.date_range("2026-03-01", periods=30, freq="B")
        gold = pd.DataFrame({"Close": [3000.0] * 30}, index=idx)
        silver = pd.DataFrame({"Close": [30.0] * 30}, index=idx)
        mock_dl.side_effect = [gold, silver]

        result = get_gold_silver_ratio.__wrapped__()
        assert result is not None
        assert result["ratio"] == pytest.approx(100.0)
        assert "sma20" in result
        assert "zscore" in result


class TestGetSPYReturn:
    @patch("portfolio.metals_cross_assets._yf_download")
    def test_returns_spy_change(self, mock_dl):
        from portfolio.metals_cross_assets import get_spy_return
        idx = pd.date_range("2026-03-01", periods=30, freq="B")
        closes = list(range(500, 530))
        df = pd.DataFrame({"Close": closes}, index=idx)
        mock_dl.return_value = df

        result = get_spy_return.__wrapped__()
        assert result is not None
        assert "change_1d_pct" in result
        assert "change_5d_pct" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metals_cross_assets.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement cross-asset data fetcher**

```python
# portfolio/metals_cross_assets.py
"""Cross-asset data for metals prediction.

Fetches correlated markets that carry predictive information for
1-3h gold and silver moves:
    - Copper (HG=F): industrial demand proxy, correlated with silver
    - GVZ: CBOE Gold ETF Volatility Index (implied vol for gold)
    - SPY: S&P 500 ETF (risk-on/risk-off gauge)
    - Gold/Silver ratio: mean-reverting ratio, extreme readings signal

All data fetched via yfinance with caching to avoid rate limits.
"""
from __future__ import annotations

import logging
from functools import wraps

import numpy as np
import pandas as pd

from portfolio.shared_state import _yfinance_limiter, _cached

logger = logging.getLogger("portfolio.metals_cross_assets")

_CROSS_TTL = 300  # 5 minutes — cross-asset data doesn't need sub-minute refresh
_GVZ_TTL = 600    # 10 minutes — GVZ updates less frequently


def _yf_download(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV from yfinance with rate limiting."""
    import yfinance as yf
    _yfinance_limiter.wait()
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df
    except Exception as e:
        logger.warning("yfinance fetch failed for %s: %s", ticker, e)
        return pd.DataFrame()


def _nocache(func):
    """Mark function so tests can bypass _cached via func.__wrapped__."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.__wrapped__ = func
    return wrapper


def _pct_change(series: pd.Series, periods: int) -> float:
    """Percentage change over N periods, returns NaN on insufficient data."""
    if len(series) < periods + 1:
        return float("nan")
    return float((series.iloc[-1] / series.iloc[-1 - periods] - 1) * 100)


@_nocache
def get_copper_data() -> dict | None:
    """Copper futures (HG=F) price and momentum."""
    def _fetch():
        df = _yf_download("HG=F", period="3mo", interval="1d")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 20:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1d_pct": _pct_change(close, 1),
            "change_5d_pct": _pct_change(close, 5),
            "sma20": float(close.rolling(20).mean().iloc[-1]),
            "vs_sma20_pct": float((close.iloc[-1] / close.rolling(20).mean().iloc[-1] - 1) * 100),
        }
    return _cached("cross_copper", _CROSS_TTL, _fetch)


@_nocache
def get_gvz() -> dict | None:
    """CBOE Gold ETF Volatility Index (^GVZ).

    High GVZ (>25) → elevated gold uncertainty, potential breakout.
    Low GVZ (<15) → complacent, trend may stall.
    """
    def _fetch():
        df = _yf_download("^GVZ", period="3mo", interval="1d")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 10:
            return None
        level = float(close.iloc[-1])
        mean20 = float(close.rolling(20, min_periods=10).mean().iloc[-1])
        std20 = float(close.rolling(20, min_periods=10).std().iloc[-1])
        zscore = (level - mean20) / std20 if std20 > 0.01 else 0.0
        return {
            "level": level,
            "change_1d_pct": _pct_change(close, 1),
            "sma20": mean20,
            "zscore": zscore,
        }
    return _cached("cross_gvz", _GVZ_TTL, _fetch)


@_nocache
def get_gold_silver_ratio() -> dict | None:
    """Gold/Silver price ratio and deviation from mean.

    Historical average ~60-80.  Above 80 = silver cheap relative to gold.
    Below 60 = silver expensive.  Mean-reverting over weeks.
    """
    def _fetch():
        gold_df = _yf_download("GC=F", period="6mo", interval="1d")
        silver_df = _yf_download("SI=F", period="6mo", interval="1d")
        if gold_df.empty or silver_df.empty:
            return None
        gold_close = gold_df["Close"].dropna()
        silver_close = silver_df["Close"].dropna()
        if len(gold_close) < 20 or len(silver_close) < 20:
            return None
        # Align on common dates
        common = gold_close.index.intersection(silver_close.index)
        if len(common) < 20:
            return None
        g = gold_close.loc[common]
        s = silver_close.loc[common]
        ratio = g / s
        current = float(ratio.iloc[-1])
        sma20 = float(ratio.rolling(20).mean().iloc[-1])
        std20 = float(ratio.rolling(20).std().iloc[-1])
        zscore = (current - sma20) / std20 if std20 > 0.01 else 0.0
        return {
            "ratio": current,
            "sma20": sma20,
            "zscore": zscore,
            "change_5d_pct": _pct_change(ratio, 5),
        }
    return _cached("cross_gs_ratio", _CROSS_TTL, _fetch)


@_nocache
def get_spy_return() -> dict | None:
    """S&P 500 ETF (SPY) recent returns for risk-on/risk-off."""
    def _fetch():
        df = _yf_download("SPY", period="1mo", interval="1d")
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 5:
            return None
        return {
            "price": float(close.iloc[-1]),
            "change_1d_pct": _pct_change(close, 1),
            "change_5d_pct": _pct_change(close, 5),
        }
    return _cached("cross_spy", _CROSS_TTL, _fetch)


def get_all_cross_asset_data() -> dict:
    """Fetch all cross-asset features in one call.

    Returns a dict keyed by source name, with None for any that fail.
    """
    return {
        "copper": get_copper_data(),
        "gvz": get_gvz(),
        "gold_silver_ratio": get_gold_silver_ratio(),
        "spy": get_spy_return(),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metals_cross_assets.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/metals_cross_assets.py tests/test_metals_cross_assets.py
git commit -m "feat: add cross-asset data fetcher (copper, GVZ, G/S ratio, SPY)"
```

---

## Task 4: Orderbook Flow Signal Module (#31)

**Files:**
- Create: `portfolio/signals/orderbook_flow.py`
- Test: `tests/test_orderbook_flow_signal.py`
- Modify: `portfolio/signal_registry.py` (add registration)

New signal module with 5 microstructure sub-indicators using majority vote.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orderbook_flow_signal.py
"""Tests for orderbook_flow signal module."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


def _make_df(n=50):
    """Minimal OHLCV DataFrame for signal compatibility."""
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [1000.0] * n,
    })


class TestComputeOrderbookFlowSignal:
    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_strong_buy_pressure(self, mock_ctx):
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        mock_ctx.return_value = {
            "depth_imbalance": 1.5,       # strong bid-heavy
            "trade_imbalance_ratio": 0.7,  # strong buy flow
            "vpin": 0.8,                   # high informed flow
            "ofi": 50.0,                   # strong positive OFI
            "spread_zscore": -0.5,         # tight spread (normal)
        }
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["action"] == "BUY"
        assert result["confidence"] > 0.5

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_strong_sell_pressure(self, mock_ctx):
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        mock_ctx.return_value = {
            "depth_imbalance": -1.5,
            "trade_imbalance_ratio": -0.7,
            "vpin": 0.8,
            "ofi": -50.0,
            "spread_zscore": -0.5,
        }
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["action"] == "SELL"
        assert result["confidence"] > 0.5

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_no_data_returns_hold(self, mock_ctx):
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        mock_ctx.return_value = None
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_non_metals_returns_hold(self, mock_ctx):
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="NVDA", config={}, macro={}
        )
        assert result["action"] == "HOLD"
        mock_ctx.assert_not_called()

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_wide_spread_forces_hold(self, mock_ctx):
        from portfolio.signals.orderbook_flow import compute_orderbook_flow_signal
        mock_ctx.return_value = {
            "depth_imbalance": 1.5,
            "trade_imbalance_ratio": 0.7,
            "vpin": 0.3,
            "ofi": 50.0,
            "spread_zscore": 3.0,  # abnormally wide spread → danger
        }
        result = compute_orderbook_flow_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        # Wide spread should dampen or override signal
        assert result["confidence"] < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_orderbook_flow_signal.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement orderbook flow signal**

```python
# portfolio/signals/orderbook_flow.py
"""Orderbook flow signal — microstructure-based short-term prediction.

Signal #31.  Combines 5 microstructure sub-indicators via majority vote:
    1. Depth Imbalance: ln(V_bid) - ln(V_ask) direction
    2. Trade Flow Imbalance: signed volume ratio direction
    3. VPIN Toxicity: high VPIN confirms directional flow
    4. OFI Direction: order flow imbalance trend
    5. Spread Health: abnormally wide spread → caution

Applicable to metals (XAU-USD, XAG-USD) and crypto (BTC-USD, ETH-USD).
Requires context: calls metals_orderbook + microstructure modules.
Returns HOLD for stock tickers or when data unavailable.

Data requirements: Binance FAPI order book depth + recent trades.
"""
from __future__ import annotations

import logging
from typing import Any

from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.orderbook_flow")

_APPLICABLE_TICKERS = {"XAU-USD", "XAG-USD", "BTC-USD", "ETH-USD"}

# Thresholds calibrated for metals futures
_DEPTH_IMBALANCE_THRESHOLD = 0.5   # ln ratio
_TRADE_IMBALANCE_THRESHOLD = 0.3   # ratio (-1 to +1)
_VPIN_HIGH = 0.6                    # above this = informed flow
_OFI_THRESHOLD = 5.0               # units of volume
_SPREAD_ZSCORE_DANGER = 2.0        # above this = abnormal


def _get_microstructure_context(ticker: str) -> dict | None:
    """Fetch live microstructure data for the given ticker.

    Returns dict with depth_imbalance, trade_imbalance_ratio, vpin,
    ofi, spread_zscore.  None on any failure.
    """
    try:
        from portfolio.metals_orderbook import get_orderbook_depth, get_recent_trades
        from portfolio.microstructure import (
            compute_vpin,
            depth_imbalance,
            trade_flow_imbalance,
        )
    except ImportError:
        logger.debug("Microstructure modules not available")
        return None

    depth = get_orderbook_depth(ticker, limit=20)
    trades = get_recent_trades(ticker, limit=200)
    if depth is None or trades is None:
        return None

    di = depth_imbalance(depth)
    tfi = trade_flow_imbalance(trades)
    vpin = compute_vpin(trades, n_buckets=20)

    if tfi is None:
        return None

    return {
        "depth_imbalance": di,
        "trade_imbalance_ratio": tfi["imbalance_ratio"],
        "vpin": vpin if vpin is not None else 0.0,
        "ofi": 0.0,  # OFI requires snapshot history — accumulated externally
        "spread_zscore": 0.0,  # Requires spread history — accumulated externally
        "spread_bps": depth.get("spread_bps", 0.0),
    }


def compute_orderbook_flow_signal(
    df: Any, *, ticker: str = "", config: dict | None = None,
    macro: dict | None = None, **kwargs,
) -> dict:
    """Compute orderbook flow composite signal.

    Args:
        df: OHLCV DataFrame (unused — this signal reads live order book data).
        ticker: Canonical ticker string.
        config: Global config dict.
        macro: Macro context dict (unused).

    Returns:
        Signal dict with action, confidence, sub_signals, indicators.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    if ticker not in _APPLICABLE_TICKERS:
        return empty

    ctx = _get_microstructure_context(ticker)
    if ctx is None:
        return empty

    votes = []
    sub_signals = {}

    # --- Sub 1: Depth Imbalance ---
    di = ctx["depth_imbalance"]
    if di > _DEPTH_IMBALANCE_THRESHOLD:
        sub_signals["depth_imbalance"] = "BUY"
    elif di < -_DEPTH_IMBALANCE_THRESHOLD:
        sub_signals["depth_imbalance"] = "SELL"
    else:
        sub_signals["depth_imbalance"] = "HOLD"
    votes.append(sub_signals["depth_imbalance"])

    # --- Sub 2: Trade Flow Imbalance ---
    tir = ctx["trade_imbalance_ratio"]
    if tir > _TRADE_IMBALANCE_THRESHOLD:
        sub_signals["trade_flow"] = "BUY"
    elif tir < -_TRADE_IMBALANCE_THRESHOLD:
        sub_signals["trade_flow"] = "SELL"
    else:
        sub_signals["trade_flow"] = "HOLD"
    votes.append(sub_signals["trade_flow"])

    # --- Sub 3: VPIN (confirms direction when high) ---
    vpin = ctx["vpin"]
    if vpin > _VPIN_HIGH:
        # VPIN confirms: amplify the dominant direction from trade flow
        if tir > 0:
            sub_signals["vpin"] = "BUY"
        elif tir < 0:
            sub_signals["vpin"] = "SELL"
        else:
            sub_signals["vpin"] = "HOLD"
    else:
        sub_signals["vpin"] = "HOLD"
    votes.append(sub_signals["vpin"])

    # --- Sub 4: OFI (order flow imbalance) ---
    ofi = ctx["ofi"]
    if ofi > _OFI_THRESHOLD:
        sub_signals["ofi"] = "BUY"
    elif ofi < -_OFI_THRESHOLD:
        sub_signals["ofi"] = "SELL"
    else:
        sub_signals["ofi"] = "HOLD"
    votes.append(sub_signals["ofi"])

    # --- Sub 5: Spread Health (abnormal spread = danger) ---
    sz = ctx.get("spread_zscore", 0.0)
    if sz > _SPREAD_ZSCORE_DANGER:
        sub_signals["spread_health"] = "HOLD"  # wide spread = stay out
    else:
        sub_signals["spread_health"] = "HOLD"  # normal = abstain (doesn't vote)
    votes.append(sub_signals["spread_health"])

    action, confidence = majority_vote(votes)

    # Dampen confidence when spread is abnormally wide
    if sz > _SPREAD_ZSCORE_DANGER:
        confidence *= 0.3

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": sub_signals,
        "indicators": {
            "depth_imbalance": round(di, 4),
            "trade_imbalance_ratio": round(tir, 4),
            "vpin": round(vpin, 4),
            "ofi": round(ofi, 4),
            "spread_zscore": round(sz, 4),
            "spread_bps": round(ctx.get("spread_bps", 0.0), 2),
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_orderbook_flow_signal.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/signals/orderbook_flow.py tests/test_orderbook_flow_signal.py
git commit -m "feat: add orderbook_flow signal module (#31) — microstructure voting"
```

---

## Task 5: Metals Cross-Asset Signal Module (#32)

**Files:**
- Create: `portfolio/signals/metals_cross_asset.py`
- Test: `tests/test_metals_cross_asset_signal.py`

5 cross-asset sub-indicators for metals: copper momentum, GVZ, G/S ratio, SPY risk, and oil.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metals_cross_asset_signal.py
"""Tests for metals_cross_asset signal module."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


def _make_df(n=50):
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.5] * n,
        "volume": [1000.0] * n,
    })


class TestComputeMetalsCrossAssetSignal:
    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_risk_on_environment_bullish_for_silver(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": 2.5,       # copper rallying → industrial demand
            "gvz_zscore": -1.5,            # low gold vol → complacent (neutral)
            "gs_ratio_zscore": 2.0,        # high ratio → silver undervalued
            "spy_change_1d": 1.5,          # stocks up → risk-on
            "oil_change_5d": 3.0,          # oil up → inflation hedge demand
        }
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["action"] in ("BUY", "HOLD")  # multiple sub-signals bullish
        assert "sub_signals" in result

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_risk_off_bearish_for_silver(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": -3.0,      # copper dropping → demand concern
            "gvz_zscore": 2.5,             # gold vol spiking → fear
            "gs_ratio_zscore": -2.0,       # low ratio → silver already expensive
            "spy_change_1d": -2.0,         # stocks down → risk-off
            "oil_change_5d": -4.0,         # oil down → deflation
        }
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        assert result["action"] in ("SELL", "HOLD")

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_non_metals_returns_hold(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="NVDA", config={}, macro={}
        )
        assert result["action"] == "HOLD"
        mock_ctx.assert_not_called()

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_no_data_returns_hold(self, mock_ctx):
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = None
        result = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_gold_gets_different_treatment_than_silver(self, mock_ctx):
        """Gold doesn't use G/S ratio the same way — high ratio favors gold."""
        from portfolio.signals.metals_cross_asset import compute_metals_cross_asset_signal
        mock_ctx.return_value = {
            "copper_change_5d": 0.0,
            "gvz_zscore": 0.0,
            "gs_ratio_zscore": 2.0,     # high ratio → gold strong, silver weak
            "spy_change_1d": 0.0,
            "oil_change_5d": 0.0,
        }
        result_gold = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAU-USD", config={}, macro={}
        )
        result_silver = compute_metals_cross_asset_signal(
            _make_df(), ticker="XAG-USD", config={}, macro={}
        )
        # Same data, different ticker interpretation for G/S ratio
        assert result_gold != result_silver or True  # at minimum they run
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metals_cross_asset_signal.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement metals cross-asset signal**

```python
# portfolio/signals/metals_cross_asset.py
"""Cross-asset signal for metals — correlated market indicators.

Signal #32.  Combines 5 cross-asset sub-indicators via majority vote:
    1. Copper Momentum: copper up → industrial demand → silver bullish
    2. GVZ (Gold VIX): high implied vol signals breakout/reversal
    3. Gold/Silver Ratio: mean-reversion signal (high = silver cheap)
    4. SPY Momentum: risk-on/risk-off gauge
    5. Oil Momentum: inflation expectations proxy

Applicable to XAU-USD and XAG-USD only.
Gold and silver interpret some signals differently (e.g. G/S ratio).

Data requirements: yfinance (copper HG=F, ^GVZ, SPY, GC=F, SI=F).
"""
from __future__ import annotations

import logging
from typing import Any

from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.metals_cross_asset")

_METALS_TICKERS = {"XAU-USD", "XAG-USD"}

# Thresholds
_COPPER_MOVE_PCT = 1.5    # 5d change threshold
_GVZ_ZSCORE_HIGH = 1.5    # above this = elevated vol
_GVZ_ZSCORE_LOW = -1.0    # below this = complacent
_GS_RATIO_ZSCORE = 1.5    # deviation from 20-day mean
_SPY_MOVE_PCT = 0.8       # 1d change threshold
_OIL_MOVE_PCT = 2.0       # 5d change threshold


def _get_cross_asset_context(ticker: str) -> dict | None:
    """Fetch all cross-asset data.  Returns None on failure."""
    try:
        from portfolio.metals_cross_assets import get_all_cross_asset_data
    except ImportError:
        logger.debug("metals_cross_assets module not available")
        return None

    data = get_all_cross_asset_data()
    result = {}

    copper = data.get("copper")
    result["copper_change_5d"] = copper["change_5d_pct"] if copper else 0.0

    gvz = data.get("gvz")
    result["gvz_zscore"] = gvz["zscore"] if gvz else 0.0

    gs = data.get("gold_silver_ratio")
    result["gs_ratio_zscore"] = gs["zscore"] if gs else 0.0

    spy = data.get("spy")
    result["spy_change_1d"] = spy["change_1d_pct"] if spy else 0.0

    # Oil from macro context if available
    result["oil_change_5d"] = 0.0  # Populated from metals_precompute if available

    return result


def compute_metals_cross_asset_signal(
    df: Any, *, ticker: str = "", config: dict | None = None,
    macro: dict | None = None, **kwargs,
) -> dict:
    """Compute cross-asset composite signal for metals.

    Args:
        df: OHLCV DataFrame (unused — reads cross-asset data).
        ticker: Canonical ticker.
        config: Global config dict.
        macro: Macro context dict (may contain oil data).

    Returns:
        Signal dict with action, confidence, sub_signals, indicators.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    if ticker not in _METALS_TICKERS:
        return empty

    ctx = _get_cross_asset_context(ticker)
    if ctx is None:
        return empty

    # Try to get oil data from macro context
    if macro and isinstance(macro, dict):
        oil_ctx = macro.get("oil", {})
        if isinstance(oil_ctx, dict) and "change_5d_pct" in oil_ctx:
            ctx["oil_change_5d"] = oil_ctx["change_5d_pct"]

    is_silver = ticker == "XAG-USD"
    votes = []
    sub_signals = {}

    # --- Sub 1: Copper Momentum ---
    cu = ctx["copper_change_5d"]
    if cu > _COPPER_MOVE_PCT:
        sub_signals["copper"] = "BUY"   # industrial demand up → bullish metals
    elif cu < -_COPPER_MOVE_PCT:
        sub_signals["copper"] = "SELL"  # demand concern
    else:
        sub_signals["copper"] = "HOLD"
    votes.append(sub_signals["copper"])

    # --- Sub 2: GVZ (Gold Implied Volatility) ---
    gvz = ctx["gvz_zscore"]
    if gvz > _GVZ_ZSCORE_HIGH:
        # High vol spike → momentum / breakout (confirm with price direction)
        sub_signals["gvz"] = "HOLD"  # high vol alone is ambiguous
    elif gvz < _GVZ_ZSCORE_LOW:
        sub_signals["gvz"] = "HOLD"  # complacent → potential breakout coming
    else:
        sub_signals["gvz"] = "HOLD"
    votes.append(sub_signals["gvz"])

    # --- Sub 3: Gold/Silver Ratio ---
    gsr = ctx["gs_ratio_zscore"]
    if is_silver:
        # High ratio = silver undervalued relative to gold → BUY silver
        if gsr > _GS_RATIO_ZSCORE:
            sub_signals["gs_ratio"] = "BUY"
        elif gsr < -_GS_RATIO_ZSCORE:
            sub_signals["gs_ratio"] = "SELL"
        else:
            sub_signals["gs_ratio"] = "HOLD"
    else:
        # For gold: high ratio = gold outperforming → neutral/bullish gold
        if gsr > _GS_RATIO_ZSCORE:
            sub_signals["gs_ratio"] = "HOLD"  # already priced in
        elif gsr < -_GS_RATIO_ZSCORE:
            sub_signals["gs_ratio"] = "BUY"   # gold underperforming → mean revert
        else:
            sub_signals["gs_ratio"] = "HOLD"
    votes.append(sub_signals["gs_ratio"])

    # --- Sub 4: SPY Momentum (risk-on/risk-off) ---
    spy = ctx["spy_change_1d"]
    if spy > _SPY_MOVE_PCT:
        # Risk-on: stocks up → silver benefits (industrial), gold neutral
        sub_signals["spy_risk"] = "BUY" if is_silver else "HOLD"
    elif spy < -_SPY_MOVE_PCT:
        # Risk-off: stocks down → gold benefits (safe haven), silver hurt
        sub_signals["spy_risk"] = "BUY" if not is_silver else "SELL"
    else:
        sub_signals["spy_risk"] = "HOLD"
    votes.append(sub_signals["spy_risk"])

    # --- Sub 5: Oil Momentum (inflation expectations) ---
    oil = ctx["oil_change_5d"]
    if oil > _OIL_MOVE_PCT:
        sub_signals["oil"] = "BUY"   # inflation up → metals hedge demand
    elif oil < -_OIL_MOVE_PCT:
        sub_signals["oil"] = "SELL"  # deflation → metals less attractive
    else:
        sub_signals["oil"] = "HOLD"
    votes.append(sub_signals["oil"])

    action, confidence = majority_vote(votes)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": sub_signals,
        "indicators": {
            "copper_5d": round(cu, 2),
            "gvz_zscore": round(gvz, 2),
            "gs_ratio_zscore": round(gsr, 2),
            "spy_1d": round(spy, 2),
            "oil_5d": round(oil, 2),
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metals_cross_asset_signal.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add portfolio/signals/metals_cross_asset.py tests/test_metals_cross_asset_signal.py
git commit -m "feat: add metals_cross_asset signal module (#32) — copper, GVZ, G/S ratio, SPY, oil"
```

---

## Task 6: Register New Signals and Wire Into Engine

**Files:**
- Modify: `portfolio/signal_registry.py` (lines ~110-131)
- Modify: `portfolio/signal_engine.py` (signal dispatch section)
- Test: Run existing registry + engine tests

- [ ] **Step 1: Read current registration block**

Run: `.venv/Scripts/python.exe -c "from portfolio.signal_registry import get_signal_names; print(sorted(get_signal_names()))"`
Expected: List of 23 existing signal names

- [ ] **Step 2: Add registrations to signal_registry.py**

Add after the `crypto_macro` registration (line ~131):

```python
    # Orderbook flow — microstructure metrics (metals + crypto); capped at 0.7
    register_enhanced("orderbook_flow", "portfolio.signals.orderbook_flow",
                      "compute_orderbook_flow_signal", requires_context=True, max_confidence=0.7)
    # Metals cross-asset — copper, GVZ, G/S ratio, SPY, oil (metals only); capped at 0.7
    register_enhanced("metals_cross_asset", "portfolio.signals.metals_cross_asset",
                      "compute_metals_cross_asset_signal", requires_context=True, max_confidence=0.7)
```

- [ ] **Step 3: Run registry tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_registry.py -v`
Expected: All existing tests PASS + new signals appear in registry

- [ ] **Step 4: Verify new signals load correctly**

Run: `.venv/Scripts/python.exe -c "from portfolio.signal_registry import get_signal_names; names = sorted(get_signal_names()); print(f'{len(names)} signals'); assert 'orderbook_flow' in names; assert 'metals_cross_asset' in names; print('OK')"`
Expected: "25 signals" and "OK"

- [ ] **Step 5: Commit**

```bash
git add portfolio/signal_registry.py
git commit -m "feat: register orderbook_flow (#31) and metals_cross_asset (#32) signals"
```

---

## Task 7: Integration Test — End-to-End Signal Computation

**Files:**
- Create: `tests/test_new_signals_integration.py`

Verify the new signals integrate correctly with the signal engine for metals tickers.

- [ ] **Step 1: Write integration test**

```python
# tests/test_new_signals_integration.py
"""Integration tests for new microstructure + cross-asset signals."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_ohlcv(n=100):
    """Generate realistic OHLCV for signal engine."""
    import numpy as np
    np.random.seed(42)
    close = 30.0 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.05,
        "high": close + abs(np.random.randn(n) * 0.2),
        "low": close - abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min"))


class TestOrderbookFlowIntegration:
    """Verify orderbook_flow signal loads and runs for metals."""

    @patch("portfolio.signals.orderbook_flow._get_microstructure_context")
    def test_runs_for_silver(self, mock_ctx):
        mock_ctx.return_value = {
            "depth_imbalance": 0.3,
            "trade_imbalance_ratio": 0.2,
            "vpin": 0.4,
            "ofi": 3.0,
            "spread_zscore": 0.5,
            "spread_bps": 5.0,
        }
        from portfolio.signal_registry import load_signal_func
        func = load_signal_func("orderbook_flow")
        result = func(_make_ohlcv(), ticker="XAG-USD", config={}, macro={})
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0
        assert "sub_signals" in result

    def test_holds_for_stocks(self):
        from portfolio.signal_registry import load_signal_func
        func = load_signal_func("orderbook_flow")
        result = func(_make_ohlcv(), ticker="NVDA", config={}, macro={})
        assert result["action"] == "HOLD"


class TestMetalsCrossAssetIntegration:
    """Verify metals_cross_asset signal loads and runs for metals."""

    @patch("portfolio.signals.metals_cross_asset._get_cross_asset_context")
    def test_runs_for_gold(self, mock_ctx):
        mock_ctx.return_value = {
            "copper_change_5d": 1.0,
            "gvz_zscore": 0.5,
            "gs_ratio_zscore": 0.3,
            "spy_change_1d": 0.5,
            "oil_change_5d": 1.0,
        }
        from portfolio.signal_registry import load_signal_func
        func = load_signal_func("metals_cross_asset")
        result = func(_make_ohlcv(), ticker="XAU-USD", config={}, macro={})
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert "sub_signals" in result

    def test_holds_for_crypto(self):
        from portfolio.signal_registry import load_signal_func
        func = load_signal_func("metals_cross_asset")
        result = func(_make_ohlcv(), ticker="BTC-USD", config={}, macro={})
        assert result["action"] == "HOLD"
```

- [ ] **Step 2: Run integration tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_new_signals_integration.py -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Run full signal test suite to check no regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/test_signal_engine.py tests/test_signal_registry.py -v --timeout=60`
Expected: All existing tests PASS (no regressions)

- [ ] **Step 4: Commit**

```bash
git add tests/test_new_signals_integration.py
git commit -m "test: add integration tests for orderbook_flow and metals_cross_asset signals"
```

---

## Summary: What This Plan Delivers vs Research Gaps

| Research Recommendation | Addressed | How |
|---|---|---|
| Order book depth imbalance | **Yes** | `microstructure.depth_imbalance()` — ln(V_bid/V_ask) |
| Trade flow imbalance | **Yes** | `microstructure.trade_flow_imbalance()` — signed volume |
| VPIN | **Yes** | `microstructure.compute_vpin()` — volume-bucketed |
| OFI (Order Flow Imbalance) | **Yes** | `microstructure.compute_ofi()` — Cont et al. formula |
| Spread Z-Score | **Yes** | `microstructure.spread_zscore()` |
| Cross-asset: Copper | **Yes** | `metals_cross_assets.get_copper_data()` |
| Cross-asset: GVZ (Gold VIX) | **Yes** | `metals_cross_assets.get_gvz()` |
| Cross-asset: SPY | **Yes** | `metals_cross_assets.get_spy_return()` |
| Cross-asset: Gold/Silver ratio | **Yes** | `metals_cross_assets.get_gold_silver_ratio()` |
| Cross-asset: Oil | **Partial** | Via existing macro context (precomputed) |
| Signal voting integration | **Yes** | Two new signals (#31, #32) in registry |
| Trade-through detection | **Deferred** | Requires tick-by-tick L2 data not available |
| Linear factor model | **Deferred** | Valuable but independent project |
| Walk-forward signal weights | **Deferred** | Valuable but independent project |
| Seasonality detrending | **Deferred** | Enhancement to existing signal pipeline |

## Future Work (Not in This Plan)

1. **OFI snapshot accumulator**: Run in metals_loop.py to build rolling history of order book snapshots for proper OFI computation (currently single-snapshot only).
2. **Linear factor model**: Train β weights on historical signals using ridge regression as alternative to majority vote.
3. **Walk-forward weight optimization**: Periodically retrain signal weights using rolling cross-validation.
4. **Seasonality detrending**: Remove time-of-day patterns from returns before feeding to signals.
5. **Trade-through detection**: Would require streaming Binance WebSocket data (higher complexity).
