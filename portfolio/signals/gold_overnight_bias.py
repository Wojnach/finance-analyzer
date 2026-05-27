"""Gold overnight bias signal module.

Exploits the systematic directional drift in gold prices between LBMA
fix times:

  - Overnight (PM fix 15:00 UTC to AM fix 10:30 UTC): persistent positive
    drift.  Over 54 years, $100 invested only overnight became $112,274.
  - London PM session (AM fix 10:30 UTC to PM fix 15:00 UTC): persistent
    negative drift.  Same $100 invested intraday became $6.97.

Three sub-signals:
  1. Session Phase    -- BUY overnight, SELL London PM
  2. Trend Alignment  -- EMA(9)/EMA(21) confirms or contradicts session bias
  3. Fix Proximity    -- confidence boost within 90 min of fix times

Applies only to metals (XAU-USD primary, XAG-USD secondary with reduced
weight).  Non-metals tickers return HOLD.

Source: Sprott Money 2024, LBMA fix data 1970-2024.
"""
from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd

from portfolio.signal_utils import ema, majority_vote, safe_float

logger = logging.getLogger(__name__)

MIN_ROWS = 21

_AM_FIX_HOUR = 10
_AM_FIX_MIN = 30
_PM_FIX_HOUR = 15
_PM_FIX_MIN = 0

_AM_FIX_MINUTES = _AM_FIX_HOUR * 60 + _AM_FIX_MIN   # 630
_PM_FIX_MINUTES = _PM_FIX_HOUR * 60 + _PM_FIX_MIN    # 900


def _get_utc_time(df: pd.DataFrame) -> tuple[int, int]:
    """Return (utc_hour, utc_minute) from DataFrame index or wall clock."""
    if hasattr(df.index, "hour"):
        try:
            last_ts = df.index[-1]
            if hasattr(last_ts, "hour"):
                if last_ts.tzinfo is not None:
                    utc_ts = last_ts.astimezone(datetime.timezone.utc)
                    return utc_ts.hour, utc_ts.minute
                return last_ts.hour, last_ts.minute
        except Exception:
            logger.debug("gold_overnight_bias: data parse error", exc_info=True)
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.hour, now.minute


def _is_metals(context: dict | None) -> tuple[bool, str]:
    """Check if ticker is metals.  Returns (is_metals, ticker)."""
    if not context:
        return False, ""
    ticker = context.get("ticker", "")
    ac = context.get("asset_class", "")
    if ac == "metals" or "XAU" in ticker or "XAG" in ticker:
        return True, ticker
    return False, ticker


def _session_phase_vote(minutes_of_day: int) -> tuple[str, float]:
    """Core sub-signal: BUY overnight, SELL London PM.

    Overnight runs from PM fix (15:00 = 900 min) to AM fix (10:30 = 630 min
    next day).  London PM runs from AM fix to PM fix.

    Confidence is higher mid-session (established drift) and lower near
    session boundaries (transition uncertainty).
    """
    if _AM_FIX_MINUTES <= minutes_of_day < _PM_FIX_MINUTES:
        # London PM session -> SELL
        session_len = _PM_FIX_MINUTES - _AM_FIX_MINUTES  # 270 min
        elapsed = minutes_of_day - _AM_FIX_MINUTES
        depth = elapsed / session_len
        # Mid-session (depth ~0.5) has highest confidence
        conf = 0.4 + 0.3 * (1.0 - abs(depth - 0.5) * 2)
        return "SELL", conf
    else:
        # Overnight session -> BUY
        if minutes_of_day >= _PM_FIX_MINUTES:
            elapsed = minutes_of_day - _PM_FIX_MINUTES
        else:
            elapsed = (1440 - _PM_FIX_MINUTES) + minutes_of_day
        session_len = 1440 - (_PM_FIX_MINUTES - _AM_FIX_MINUTES)  # 1170 min
        depth = elapsed / session_len
        conf = 0.4 + 0.3 * (1.0 - abs(depth - 0.5) * 2)
        return "BUY", conf


def _trend_alignment_vote(close: pd.Series) -> tuple[str, float]:
    """EMA(9)/EMA(21) for directional confirmation."""
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    if ema9 is None or ema21 is None:
        return "HOLD", 0.0
    fast = safe_float(ema9.iloc[-1])
    slow = safe_float(ema21.iloc[-1])
    if fast is None or slow is None or np.isnan(fast) or np.isnan(slow) or slow == 0:
        return "HOLD", 0.0
    pct_diff = (fast - slow) / slow
    if pct_diff > 0.002:
        return "BUY", min(abs(pct_diff) * 10, 0.7)
    if pct_diff < -0.002:
        return "SELL", min(abs(pct_diff) * 10, 0.7)
    return "HOLD", 0.0


def _fix_proximity_vote(minutes_of_day: int) -> tuple[str, float]:
    """Boost confidence within 90 minutes of fix times.

    Near AM fix (10:30): overnight drift concluding -> BUY (take profit
    on overnight position, but signal still BUY for the drift).
    Near PM fix (15:00): London PM drift concluding -> setup for
    overnight entry -> BUY.
    """
    dist_am = abs(minutes_of_day - _AM_FIX_MINUTES)
    dist_pm = abs(minutes_of_day - _PM_FIX_MINUTES)
    min_dist = min(dist_am, dist_pm)

    if min_dist > 90:
        return "HOLD", 0.0

    proximity_strength = (90 - min_dist) / 90

    if min_dist == dist_pm:
        # Near PM fix -> overnight about to start -> BUY setup
        return "BUY", 0.3 * proximity_strength
    else:
        # Near AM fix -> overnight concluding, still in BUY drift
        return "BUY", 0.2 * proximity_strength


def compute_gold_overnight_bias_signal(
    df: pd.DataFrame, context: dict | None = None
) -> dict:
    """Compute gold overnight bias signal."""
    empty = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    if df is None or len(df) < MIN_ROWS:
        return empty

    metals, ticker = _is_metals(context)
    if not metals:
        return empty

    close = df["close"].dropna()
    if len(close) < MIN_ROWS:
        return empty

    utc_hour, utc_minute = _get_utc_time(df)
    minutes_of_day = utc_hour * 60 + utc_minute

    # Sub-signal 1: Session phase (core)
    session_vote, session_conf = _session_phase_vote(minutes_of_day)

    # Sub-signal 2: Trend alignment
    trend_vote, trend_conf = _trend_alignment_vote(close)

    # Sub-signal 3: Fix proximity
    prox_vote, prox_conf = _fix_proximity_vote(minutes_of_day)

    votes = [session_vote, trend_vote, prox_vote]
    action, confidence = majority_vote(votes, count_hold=False)

    # Silver discount: effect is weaker for XAG than XAU
    if "XAG" in ticker:
        confidence *= 0.7

    indicators = {
        "utc_hour": utc_hour,
        "utc_minute": utc_minute,
        "minutes_of_day": minutes_of_day,
        "session_phase": "overnight" if session_vote == "BUY" else "london_pm",
        "session_conf": float(session_conf),
        "trend_conf": float(trend_conf),
        "proximity_conf": float(prox_conf),
    }

    return {
        "action": action,
        "confidence": round(min(confidence, 0.7), 4),
        "sub_signals": {
            "session_phase": session_vote,
            "trend_alignment": trend_vote,
            "fix_proximity": prox_vote,
        },
        "indicators": indicators,
    }
