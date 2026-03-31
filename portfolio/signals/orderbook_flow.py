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
"""
from __future__ import annotations

import logging
from typing import Any

from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.orderbook_flow")

_APPLICABLE_TICKERS = {"XAU-USD", "XAG-USD", "BTC-USD", "ETH-USD"}

_DEPTH_IMBALANCE_THRESHOLD = 0.5
_TRADE_IMBALANCE_THRESHOLD = 0.3
_VPIN_HIGH = 0.6
_OFI_THRESHOLD = 5.0
_SPREAD_ZSCORE_DANGER = 2.0


def _get_microstructure_context(ticker: str) -> dict | None:
    """Fetch live microstructure data for the given ticker."""
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

    # Read accumulated OFI and spread z-score from microstructure state
    ofi = 0.0
    sz = 0.0
    try:
        from portfolio.microstructure_state import load_persisted_state
        ms_state = load_persisted_state(ticker)
        if ms_state:
            ofi = ms_state.get("ofi", 0.0)
            sz = ms_state.get("spread_zscore", 0.0)
    except ImportError:
        pass

    return {
        "depth_imbalance": di,
        "trade_imbalance_ratio": tfi["imbalance_ratio"],
        "vpin": vpin if vpin is not None else 0.0,
        "ofi": ofi,
        "spread_zscore": sz,
        "spread_bps": depth.get("spread_bps", 0.0),
    }


def compute_orderbook_flow_signal(
    df: Any, *, ticker: str = "", config: dict | None = None,
    macro: dict | None = None, **kwargs,
) -> dict:
    """Compute orderbook flow composite signal."""
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

    # Sub 1: Depth Imbalance
    di = ctx["depth_imbalance"]
    if di > _DEPTH_IMBALANCE_THRESHOLD:
        sub_signals["depth_imbalance"] = "BUY"
    elif di < -_DEPTH_IMBALANCE_THRESHOLD:
        sub_signals["depth_imbalance"] = "SELL"
    else:
        sub_signals["depth_imbalance"] = "HOLD"
    votes.append(sub_signals["depth_imbalance"])

    # Sub 2: Trade Flow Imbalance
    tir = ctx["trade_imbalance_ratio"]
    if tir > _TRADE_IMBALANCE_THRESHOLD:
        sub_signals["trade_flow"] = "BUY"
    elif tir < -_TRADE_IMBALANCE_THRESHOLD:
        sub_signals["trade_flow"] = "SELL"
    else:
        sub_signals["trade_flow"] = "HOLD"
    votes.append(sub_signals["trade_flow"])

    # Sub 3: VPIN (confirms direction when high)
    vpin = ctx["vpin"]
    if vpin > _VPIN_HIGH:
        if tir > 0:
            sub_signals["vpin"] = "BUY"
        elif tir < 0:
            sub_signals["vpin"] = "SELL"
        else:
            sub_signals["vpin"] = "HOLD"
    else:
        sub_signals["vpin"] = "HOLD"
    votes.append(sub_signals["vpin"])

    # Sub 4: OFI
    ofi = ctx["ofi"]
    if ofi > _OFI_THRESHOLD:
        sub_signals["ofi"] = "BUY"
    elif ofi < -_OFI_THRESHOLD:
        sub_signals["ofi"] = "SELL"
    else:
        sub_signals["ofi"] = "HOLD"
    votes.append(sub_signals["ofi"])

    # Sub 5: Spread Health — always abstains (HOLD) because spread width
    # is non-directional.  The actual effect is the 0.3x confidence penalty
    # applied below when spread_zscore > threshold.
    sz = ctx.get("spread_zscore", 0.0)
    sub_signals["spread_health"] = "HOLD"
    votes.append(sub_signals["spread_health"])

    action, confidence = majority_vote(votes)

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
