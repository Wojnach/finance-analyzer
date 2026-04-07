"""Crypto macro signal — options gravity, gold rotation, exchange reserves.

Composite signal with 5 sub-indicators, majority vote, confidence capped at 0.7.
Only applicable to crypto tickers (BTC-USD, ETH-USD). Non-crypto -> immediate HOLD.

Sub-indicators:
  1. options_gravity   — Price vs max pain: gravitational pull toward max pain
  2. put_call_ratio    — Contrarian: extreme puts = BUY, extreme calls = SELL
  3. gold_btc_rotation — Declining gold/BTC ratio = capital rotating to crypto
  4. exchange_netflow  — Sustained negative netflow = supply squeeze = BUY
  5. options_expiry    — Quarterly expiry proximity = volatility warning

The ``context`` parameter is a dict with keys: ticker, config, macro.
"""

from __future__ import annotations

import logging

import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote

logger = logging.getLogger("portfolio.signals.crypto_macro")

_MAX_CONFIDENCE = 0.7

# Thresholds
_MAX_PAIN_PULL_PCT = 0.03     # 3% — price within 3% of max pain = neutral zone
_PCR_HIGH = 1.2               # put/call > 1.2 = contrarian BUY (too many puts)
_PCR_LOW = 0.6                # put/call < 0.6 = contrarian SELL (too many calls)
_GOLD_ROTATION_THRESHOLD = 0.02  # 2% ratio change = meaningful rotation
_NETFLOW_ACCUM_DAYS = 5       # 5+ consecutive negative netflow = strong signal
_EXPIRY_WARNING_DAYS = 3      # warn within 3 days of expiry


def _options_gravity(options, current_price):
    """Sub-1: Price vs max pain — gravitational pull.

    Price well below max pain: market makers will push price up -> BUY
    Price well above max pain: market makers will push price down -> SELL
    Within 3% of max pain: neutral zone, gravity balanced
    """
    if not options or not current_price:
        return "HOLD", {}

    max_pain = options.get("max_pain")
    if not max_pain or max_pain <= 0:
        return "HOLD", {}

    distance_pct = (current_price - max_pain) / max_pain
    indicators = {
        "max_pain": max_pain,
        "distance_pct": round(distance_pct * 100, 2),
        "days_to_expiry": options.get("days_to_expiry"),
    }

    # Gravity weakens further from expiry
    days = options.get("days_to_expiry")
    if days is not None and days > 7:
        # Too far from expiry — gravity effect is weak
        return "HOLD", indicators

    if distance_pct < -_MAX_PAIN_PULL_PCT:
        return "BUY", indicators    # price below max pain, pulled up
    elif distance_pct > _MAX_PAIN_PULL_PCT:
        return "SELL", indicators   # price above max pain, pulled down
    return "HOLD", indicators


def _put_call_sentiment(options):
    """Sub-2: Put/call ratio — contrarian sentiment.

    High PCR (>1.2) = lots of puts = fear = contrarian BUY
    Low PCR (<0.6) = lots of calls = greed = contrarian SELL
    """
    if not options:
        return "HOLD", {}

    # Use nearest expiry PCR for short-term signal
    pcr = options.get("nearest_pcr")
    total_pcr = options.get("total_pcr")

    # Prefer nearest expiry if available, fall back to total
    active_pcr = pcr if pcr is not None else total_pcr
    if active_pcr is None:
        return "HOLD", {}

    indicators = {
        "nearest_pcr": pcr,
        "total_pcr": total_pcr,
    }

    if active_pcr > _PCR_HIGH:
        return "BUY", indicators    # too many puts, contrarian bullish
    elif active_pcr < _PCR_LOW:
        return "SELL", indicators   # too many calls, contrarian bearish
    return "HOLD", indicators


def _gold_rotation(gold_btc_data):
    """Sub-3: Gold-BTC rotation signal.

    Declining gold/BTC ratio = capital rotating from gold to BTC = BUY
    Rising gold/BTC ratio = capital flowing to gold, away from BTC = SELL

    Based on the thesis that gold leads BTC by ~3 months.
    When gold peaks and ratio starts declining, BTC is about to rally.
    """
    if not gold_btc_data:
        return "HOLD", {}

    trend = gold_btc_data.get("trend", "flat")
    ratio = gold_btc_data.get("gold_btc_ratio")
    ratio_7d = gold_btc_data.get("ratio_7d_ago")

    indicators = {
        "gold_btc_ratio": ratio,
        "ratio_7d_ago": ratio_7d,
        "trend": trend,
    }

    if trend == "btc_outperforming":
        return "BUY", indicators   # BTC gaining vs gold — rotation happening
    elif trend == "gold_outperforming":
        return "SELL", indicators  # gold still leading — rotation not started
    return "HOLD", indicators


def _exchange_netflow_signal(netflow_data):
    """Sub-4: Exchange netflow trend — supply squeeze detection.

    Sustained negative netflow = coins leaving exchanges = supply squeeze = BUY
    Sustained positive netflow = coins entering exchanges = selling pressure = SELL
    """
    if not netflow_data:
        return "HOLD", {}

    trend = netflow_data.get("trend", "neutral")
    consecutive_neg = netflow_data.get("consecutive_negative", 0)
    sum_7d = netflow_data.get("sum_7d")

    indicators = {
        "netflow_trend": trend,
        "consecutive_negative": consecutive_neg,
        "sum_7d": sum_7d,
    }

    if (
        trend in ("strong_accumulation",)
        or consecutive_neg >= _NETFLOW_ACCUM_DAYS
        or (trend in ("accumulation",) and consecutive_neg >= 3)
    ):
        return "BUY", indicators
    elif trend in ("strong_distribution",) or (trend in ("distribution",) and consecutive_neg == 0):
        return "SELL", indicators
    return "HOLD", indicators


def _expiry_proximity(options):
    """Sub-5: Options expiry proximity — volatility warning.

    Within 3 days of quarterly expiry: expect increased volatility.
    This is informational — votes HOLD but flags the risk.
    Very close to expiry (0-1 days): slight BUY bias (post-expiry relief rally).
    """
    if not options:
        return "HOLD", {}

    days = options.get("days_to_expiry")
    expiry = options.get("nearest_expiry", "")

    indicators = {
        "days_to_expiry": days,
        "expiry_date": expiry,
    }

    if days is None:
        return "HOLD", indicators

    # Check if it's a quarterly expiry (MAR, JUN, SEP, DEC)
    quarterly_months = {"MAR", "JUN", "SEP", "DEC"}
    is_quarterly = any(m in expiry.upper() for m in quarterly_months)
    indicators["is_quarterly"] = is_quarterly

    if days <= 1 and is_quarterly:
        # Day of/after quarterly expiry — post-expiry relief often bullish
        return "BUY", indicators
    elif days <= 1:
        # Day of regular expiry — slight bullish bias
        return "BUY", indicators

    return "HOLD", indicators


def compute_crypto_macro_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute the crypto macro composite signal.

    Args:
        df: OHLCV DataFrame for the ticker.
        context: dict with keys {ticker, config, macro}.

    Returns:
        dict with action, confidence, sub_signals, indicators.
    """
    ticker = context.get("ticker", "") if context else ""

    # Non-crypto -> immediate HOLD
    from portfolio.tickers import CRYPTO_SYMBOLS
    if ticker not in CRYPTO_SYMBOLS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {"skip_reason": "non_crypto"},
        }

    # Get current price from DataFrame
    current_price = None
    if df is not None and len(df) > 0 and "close" in df.columns:
        current_price = float(df["close"].iloc[-1])

    # Fetch crypto macro data
    from portfolio.crypto_macro_data import get_crypto_macro_data
    macro_data = _cached(
        f"crypto_macro_{ticker}",
        OPTIONS_TTL,
        get_crypto_macro_data,
        ticker,
    )

    if not macro_data:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {"error": "fetch_failed"},
        }

    options = macro_data.get("options")
    gold_btc = macro_data.get("gold_btc_ratio")
    netflow = macro_data.get("netflow_trend")

    # Compute sub-signals
    gravity_vote, gravity_ind = _options_gravity(options, current_price)
    pcr_vote, pcr_ind = _put_call_sentiment(options)
    rotation_vote, rotation_ind = _gold_rotation(gold_btc)
    netflow_vote, netflow_ind = _exchange_netflow_signal(netflow)
    expiry_vote, expiry_ind = _expiry_proximity(options)

    sub = {
        "options_gravity": gravity_vote,
        "put_call_ratio": pcr_vote,
        "gold_btc_rotation": rotation_vote,
        "exchange_netflow": netflow_vote,
        "expiry_proximity": expiry_vote,
    }

    # Majority vote
    action, confidence = majority_vote(list(sub.values()))
    confidence = min(confidence, _MAX_CONFIDENCE)

    # Merge all indicators
    indicators = {}
    for label, ind in [("gravity", gravity_ind), ("pcr", pcr_ind),
                       ("rotation", rotation_ind), ("netflow", netflow_ind),
                       ("expiry", expiry_ind)]:
        for k, v in ind.items():
            indicators[f"{label}_{k}"] = v

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub,
        "indicators": indicators,
    }


# Cache TTL imported from data module
OPTIONS_TTL = 900
