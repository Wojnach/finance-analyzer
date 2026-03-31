"""Microstructure feature computations for short-term metals prediction.

Implements order-flow and market-microstructure metrics from academic literature.
All functions are pure — they take raw data and return numeric features.

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


def depth_imbalance(depth: dict, levels: int | None = None) -> float:
    """Log ratio of bid vs ask volume: F_t = ln(V_bid) - ln(V_ask).

    Positive → bid-heavy (buying pressure).
    Negative → ask-heavy (selling pressure).
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


def trade_flow_imbalance(trades: list[dict]) -> dict[str, float] | None:
    """Compute signed volume imbalance from recent trades.
    Each trade dict must have 'qty' (float) and 'sign' (+1 buyer, -1 seller).
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


def compute_vpin(trades: list[dict], n_buckets: int = 20) -> float | None:
    """VPIN: average absolute buy-sell imbalance per volume bucket.

    High VPIN (>0.6) → toxic flow, likely informed trading.
    Low VPIN (<0.3) → balanced, uninformed flow.
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


def compute_ofi(snapshots: list[dict]) -> float:
    """Order Flow Imbalance from consecutive order book snapshots.

    Implements the Cont et al. (2014) OFI formula.
    Positive OFI → net buying pressure.
    Negative OFI → net selling pressure.
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


def spread_zscore(spread_history: list[float], min_samples: int = 5) -> float | None:
    """Z-score of current spread vs recent history."""
    if len(spread_history) < min_samples:
        return None
    arr = np.array(spread_history, dtype=float)
    mean = arr[:-1].mean()
    std = arr[:-1].std()
    if std < 1e-12:
        # Zero variance: if current matches mean it's normal (0.0),
        # otherwise it's an extreme outlier — return large signed value.
        diff = arr[-1] - mean
        if abs(diff) < 1e-12:
            return 0.0
        return float(np.sign(diff) * 10.0)
    return float((arr[-1] - mean) / std)
