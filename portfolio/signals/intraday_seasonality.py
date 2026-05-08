"""Intraday seasonality gate signal.

Exploits empirical hour-of-day and day-of-week return patterns to modulate
signal confidence. Three sub-indicators vote via majority:

  1. Hour Alpha        — UTC hour maps to empirical return multiplier per asset class
  2. Day-of-Week Bias  — Monday crypto boost, Wednesday FOMC caution
  3. Trend Context     — EMA(9)/EMA(21) crossover for directional context

During high-alpha hours, confidence is boosted. During low-alpha hours,
signal outputs HOLD to suppress noise. Direction comes from trend context,
not from time alone.

Academic backing:
  - BTC 22:00-23:00 UTC: 33% annualized (ScienceDirect 2024)
  - Monday Asia Open: Sharpe 1.6 (Concretum Group 2025)
  - Gold London-NY overlap: 60-70% of daily range (CME Group 2026)
  - Crypto tea-time peak 16:00-17:00 UTC (Springer RQFA 2024)

Requires DataFrame with OHLCV and at least 21 rows.
"""
from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd

from portfolio.signal_utils import ema, majority_vote, safe_float

logger = logging.getLogger(__name__)

MIN_ROWS = 21
_MAX_CONFIDENCE = 0.7

# Hour multipliers: >1.0 = high-alpha, <1.0 = low-alpha, ~0.0 = suppress
# Based on empirical intraday return patterns from academic literature.

_CRYPTO_HOUR_MULT = {
    0: 1.2, 1: 1.1, 2: 0.9, 3: 0.9, 4: 0.8, 5: 0.8,
    6: 0.9, 7: 0.9, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0,
    12: 0.9, 13: 0.9, 14: 0.7, 15: 0.7, 16: 0.8, 17: 0.7,
    18: 0.7, 19: 0.7, 20: 0.8, 21: 1.3, 22: 1.4, 23: 1.3,
}

_METALS_HOUR_MULT = {
    0: 0.7, 1: 0.7, 2: 0.5, 3: 0.5, 4: 0.5, 5: 0.5,
    6: 0.6, 7: 0.7, 8: 0.9, 9: 1.0, 10: 1.0, 11: 1.0,
    12: 1.1, 13: 1.3, 14: 1.3, 15: 1.3, 16: 1.2, 17: 1.1,
    18: 0.9, 19: 0.8, 20: 0.7, 21: 0.6, 22: 0.6, 23: 0.6,
}

_STOCKS_HOUR_MULT = {
    0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3, 5: 0.3,
    6: 0.3, 7: 0.3, 8: 0.3, 9: 0.5, 10: 0.6, 11: 0.7,
    12: 0.8, 13: 0.9, 14: 1.3, 15: 1.2, 16: 1.0, 17: 0.7,
    18: 0.8, 19: 1.2, 20: 1.3, 21: 0.5, 22: 0.4, 23: 0.3,
}

_ASSET_HOUR_MAP = {
    "crypto": _CRYPTO_HOUR_MULT,
    "metals": _METALS_HOUR_MULT,
    "stocks": _STOCKS_HOUR_MULT,
}

# Day-of-week multipliers (0=Monday..6=Sunday)
_CRYPTO_DOW_MULT = {0: 1.15, 1: 1.0, 2: 0.9, 3: 1.0, 4: 1.0, 5: 0.95, 6: 1.1}
_METALS_DOW_MULT = {0: 1.0, 1: 1.05, 2: 0.9, 3: 1.0, 4: 0.95, 5: 0.5, 6: 0.5}
_STOCKS_DOW_MULT = {0: 0.95, 1: 1.05, 2: 0.9, 3: 1.0, 4: 1.05, 5: 0.3, 6: 0.3}

_ASSET_DOW_MAP = {
    "crypto": _CRYPTO_DOW_MULT,
    "metals": _METALS_DOW_MULT,
    "stocks": _STOCKS_DOW_MULT,
}


def _get_utc_hour_and_dow(df: pd.DataFrame) -> tuple[int, int]:
    """Extract UTC hour and day-of-week from DataFrame's last timestamp."""
    if hasattr(df.index, "hour"):
        try:
            last_ts = df.index[-1]
            if hasattr(last_ts, "hour"):
                return last_ts.hour, last_ts.weekday()
        except Exception:
            pass
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.hour, now.weekday()


def _classify_asset(context: dict | None) -> str:
    """Determine asset class from context."""
    if not context:
        return "crypto"
    ac = context.get("asset_class", "")
    if ac in ("crypto", "metals", "stocks"):
        return ac
    ticker = context.get("ticker", "")
    if "XAU" in ticker or "XAG" in ticker:
        return "metals"
    if "MSTR" in ticker:
        return "stocks"
    return "crypto"


def _hour_alpha_vote(hour: int, asset_class: str) -> tuple[str, float]:
    """Return (vote, raw_multiplier) based on hour and asset class."""
    mult_map = _ASSET_HOUR_MAP.get(asset_class, _CRYPTO_HOUR_MULT)
    mult = mult_map.get(hour, 1.0)
    if mult >= 1.2:
        return "BUY", mult
    if mult <= 0.5:
        return "HOLD", mult
    return "HOLD", mult


def _dow_vote(dow: int, asset_class: str) -> tuple[str, float]:
    """Return (vote, multiplier) based on day of week."""
    dow_map = _ASSET_DOW_MAP.get(asset_class, _CRYPTO_DOW_MULT)
    mult = dow_map.get(dow, 1.0)
    if mult >= 1.1:
        return "BUY", mult
    if mult <= 0.5:
        return "HOLD", mult
    return "HOLD", mult


def _trend_direction(close: pd.Series) -> tuple[str, float]:
    """EMA(9)/EMA(21) crossover for direction."""
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    if ema9 is None or ema21 is None:
        return "HOLD", 0.0
    last_fast = safe_float(ema9.iloc[-1])
    last_slow = safe_float(ema21.iloc[-1])
    if last_fast is None or last_slow is None:
        return "HOLD", 0.0
    pct_diff = (last_fast - last_slow) / last_slow if last_slow != 0 else 0.0
    if pct_diff > 0.002:
        return "BUY", abs(pct_diff)
    if pct_diff < -0.002:
        return "SELL", abs(pct_diff)
    return "HOLD", abs(pct_diff)


def compute_intraday_seasonality_signal(
    df: pd.DataFrame, context: dict | None = None
) -> dict:
    """Compute intraday seasonality gate signal."""
    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    close = df["close"].dropna()
    if len(close) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    asset_class = _classify_asset(context)
    utc_hour, dow = _get_utc_hour_and_dow(df)

    # Sub-signal 1: Hour alpha
    hour_vote, hour_mult = _hour_alpha_vote(utc_hour, asset_class)

    # Sub-signal 2: Day-of-week
    dow_vote_str, dow_mult = _dow_vote(dow, asset_class)

    # Sub-signal 3: Trend direction
    trend_vote, trend_strength = _trend_direction(close)

    # Combine: hour/dow determine WHEN to act, trend determines direction
    combined_mult = hour_mult * dow_mult

    if combined_mult < 0.6:
        action = "HOLD"
        confidence = 0.0
    elif combined_mult >= 1.1 and trend_vote != "HOLD":
        action = trend_vote
        base_conf = min(0.3 + trend_strength * 5.0, 0.6)
        confidence = min(base_conf * (combined_mult / 1.0), _MAX_CONFIDENCE)
    elif trend_vote != "HOLD":
        action = trend_vote
        confidence = min(0.2 + trend_strength * 3.0, 0.4) * combined_mult
    else:
        action = "HOLD"
        confidence = 0.0

    confidence = round(min(max(confidence, 0.0), _MAX_CONFIDENCE), 4)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "hour_alpha": hour_vote,
            "day_of_week": dow_vote_str,
            "trend_context": trend_vote,
        },
        "indicators": {
            "utc_hour": utc_hour,
            "day_of_week": dow,
            "hour_multiplier": round(hour_mult, 3),
            "dow_multiplier": round(dow_mult, 3),
            "combined_multiplier": round(combined_mult, 3),
            "trend_strength": round(trend_strength, 6),
            "asset_class": asset_class,
        },
    }
