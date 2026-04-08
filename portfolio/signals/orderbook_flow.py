"""Orderbook flow signal — microstructure-based short-term prediction.

Signal #31.  Combines 6 microstructure sub-indicators via majority vote:
    1. Depth Imbalance: ln(V_bid) - ln(V_ask) direction
    2. Trade Flow Imbalance: signed volume ratio direction
    3. VPIN Toxicity: high VPIN confirms directional flow
    4. OFI Direction: order flow imbalance trend
    5. Spread Health: abnormally wide spread → caution
    6. Trade Pressure: trade-through imbalance (buy vs sell throughs)

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
_VPIN_TOXICITY = 0.7  # threshold for flagging vol expansion risk
_OFI_ZSCORE_THRESHOLD = 1.5  # z-score threshold (was absolute 5.0)
_OFI_THRESHOLD = 5.0  # fallback when z-score unavailable
_SPREAD_ZSCORE_DANGER = 2.0
_TRADE_THROUGH_THRESHOLD = 2


def _get_microstructure_context(ticker: str) -> dict | None:
    """Fetch live microstructure data for the given ticker."""
    try:
        from portfolio.metals_orderbook import get_orderbook_depth, get_recent_trades
        from portfolio.microstructure import (
            compute_vpin,
            depth_imbalance,
            detect_trade_throughs,
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
    tt = detect_trade_throughs(trades)

    if tfi is None:
        return None

    # Read accumulated OFI, OFI z-score, multiscale, and spread z-score
    ofi = 0.0
    ofi_zscore = 0.0
    sz = 0.0
    flow_acceleration = 0.0
    try:
        from portfolio.microstructure_state import load_persisted_state
        ms_state = load_persisted_state(ticker)
        if ms_state:
            ofi = ms_state.get("ofi", 0.0)
            ofi_zscore = ms_state.get("ofi_zscore", 0.0)
            sz = ms_state.get("spread_zscore", 0.0)
            flow_acceleration = ms_state.get("flow_acceleration", 0.0)
    except ImportError:
        pass

    return {
        "depth_imbalance": di,
        "trade_imbalance_ratio": tfi["imbalance_ratio"],
        "vpin": vpin if vpin is not None else 0.0,
        "ofi": ofi,
        "ofi_zscore": ofi_zscore,
        "spread_zscore": sz,
        "spread_bps": depth.get("spread_bps", 0.0),
        "trade_throughs": tt,
        "flow_acceleration": flow_acceleration,
    }


def compute_orderbook_flow_signal(
    df: Any, context: dict | None = None, **kwargs,
) -> dict:
    """Compute orderbook flow composite signal.

    Args:
        df: OHLCV DataFrame (unused — microstructure data fetched separately).
        context: dict with keys {ticker, config, macro, regime}.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    context = context or {}
    ticker = context.get("ticker", kwargs.get("ticker", ""))

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

    # Sub 4: OFI — prefer z-score (asset-normalized), fallback to absolute
    ofi_z = ctx.get("ofi_zscore", 0.0)
    ofi = ctx["ofi"]
    if abs(ofi_z) > 0.01:  # z-score available (non-zero)
        if ofi_z > _OFI_ZSCORE_THRESHOLD:
            sub_signals["ofi"] = "BUY"
        elif ofi_z < -_OFI_ZSCORE_THRESHOLD:
            sub_signals["ofi"] = "SELL"
        else:
            sub_signals["ofi"] = "HOLD"
    else:  # cold start fallback
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

    # Sub 6: Trade Pressure — trade-through imbalance
    tt = ctx.get("trade_throughs") or {}
    buy_tt = tt.get("buy_throughs", 0)
    sell_tt = tt.get("sell_throughs", 0)
    if buy_tt > sell_tt + _TRADE_THROUGH_THRESHOLD:
        sub_signals["trade_pressure"] = "BUY"
    elif sell_tt > buy_tt + _TRADE_THROUGH_THRESHOLD:
        sub_signals["trade_pressure"] = "SELL"
    else:
        sub_signals["trade_pressure"] = "HOLD"
    votes.append(sub_signals["trade_pressure"])

    action, confidence = majority_vote(votes)

    if sz > _SPREAD_ZSCORE_DANGER:
        confidence *= 0.3

    # VPIN toxicity flag: VPIN > 0.7 predicts volatility expansion
    # (independent of direction — used by risk management for stop widening)
    high_toxicity = vpin > _VPIN_TOXICITY

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": sub_signals,
        "indicators": {
            "depth_imbalance": round(di, 4),
            "trade_imbalance_ratio": round(tir, 4),
            "vpin": round(vpin, 4),
            "high_toxicity": high_toxicity,
            "ofi": round(ofi, 4),
            "ofi_zscore": round(ctx.get("ofi_zscore", 0.0), 4),
            "flow_acceleration": round(ctx.get("flow_acceleration", 0.0), 4),
            "spread_zscore": round(sz, 4),
            "spread_bps": round(ctx.get("spread_bps", 0.0), 2),
            "buy_throughs": buy_tt,
            "sell_throughs": sell_tt,
            "through_volume": tt.get("through_volume", 0.0),
            "max_gap_bps": tt.get("max_gap_bps", 0.0),
        },
    }
