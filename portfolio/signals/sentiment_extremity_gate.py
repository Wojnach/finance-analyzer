"""Sentiment Extremity Gate signal module.

Uses Fear & Greed intensity (distance from neutral) as a regime gate rather
than using F&G directionally. Academic basis: Farzulla 2026
(arxiv:2602.07018) — extreme sentiment (BOTH fear and greed) causes wider
spreads and adverse selection. BUYs execute better in moderate sentiment
(F&G 31-69) than in extreme zones.

Three sub-signals:
    1. Intensity Zone    — moderate (|FG-50| < 20) vs extreme (>= 20)
    2. Price-in-Range    — position within recent high-low range
    3. Range Compression — ATR-based volatility filter

Requires context with 'ticker' for F&G fetch. Crypto-only (F&G index is
crypto-specific from alternative.me; stock F&G is VIX-derived and does
not exhibit the same extremity premium).
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from portfolio.signal_utils import safe_float

logger = logging.getLogger(__name__)

_CRYPTO_TICKERS = frozenset({"BTC-USD", "ETH-USD"})

MIN_ROWS = 20

# --- F&G fetch with caching (60s TTL, same pattern as signal_engine) ------

_fg_cache: dict = {"value": None, "ts": 0.0}
_FG_TTL = 60.0


def _get_fg_value(ticker: str | None = None) -> int | None:
    now = time.monotonic()
    if _fg_cache["value"] is not None and (now - _fg_cache["ts"]) < _FG_TTL:
        return _fg_cache["value"]
    try:
        from portfolio.fear_greed import get_fear_greed
        fg = get_fear_greed(ticker)
        if fg and isinstance(fg.get("value"), (int, float)):
            _fg_cache["value"] = int(fg["value"])
            _fg_cache["ts"] = now
            return _fg_cache["value"]
    except Exception:
        logger.debug("sentiment_extremity_gate: F&G fetch failed", exc_info=True)
    return _fg_cache.get("value")


# --- Sub-signal 1: Intensity Zone ------------------------------------------

def _intensity_zone(fg_value: int) -> tuple[float, str]:
    """Classify F&G into moderate vs extreme intensity zone.

    Returns (intensity, zone_vote).
    Moderate (|FG-50| < 20, i.e. FG 31-69): signal passes through.
    Extreme (|FG-50| >= 20, i.e. FG <=30 or >=70): force HOLD.
    """
    intensity = abs(fg_value - 50)
    if intensity >= 20:
        return float(intensity), "HOLD"
    return float(intensity), "PASS"


# --- Sub-signal 2: Price-in-Range Position ---------------------------------

def _price_in_range(close: pd.Series, lookback: int = 20) -> tuple[float, str]:
    """Where is current price within recent range?

    Bottom 20%: BUY (at support in range).
    Top 20%: SELL (at resistance in range).
    Middle: HOLD (no edge in mid-range).
    """
    if len(close) < lookback:
        return 0.5, "HOLD"

    window = close.iloc[-lookback:]
    range_high = float(window.max())
    range_low = float(window.min())
    range_width = range_high - range_low

    if range_width <= 0 or np.isnan(range_width):
        return 0.5, "HOLD"

    price_pct = (float(close.iloc[-1]) - range_low) / range_width

    if np.isnan(price_pct):
        return 0.5, "HOLD"

    price_pct = max(0.0, min(1.0, price_pct))

    if price_pct < 0.2:
        return price_pct, "BUY"
    if price_pct > 0.8:
        return price_pct, "SELL"
    return price_pct, "HOLD"


# --- Sub-signal 3: Range Compression (ATR-based) --------------------------

def _range_compression(high: pd.Series, low: pd.Series, close: pd.Series,
                       lookback: int = 14) -> tuple[float, str]:
    """ATR compression detector. Low ATR ratio = tight range = MR favorable.

    When current ATR < 0.7x of longer-term ATR, range is compressed.
    Compressed range + price at extremes = higher-confidence MR signal.
    """
    if len(close) < lookback * 2:
        return 1.0, "HOLD"

    tr_vals = []
    for i in range(1, len(close)):
        h = float(high.iloc[i])
        lo = float(low.iloc[i])
        pc = float(close.iloc[i - 1])
        tr = max(h - lo, abs(h - pc), abs(lo - pc))
        tr_vals.append(tr)

    if len(tr_vals) < lookback * 2:
        return 1.0, "HOLD"

    tr_arr = np.array(tr_vals)
    short_atr = np.mean(tr_arr[-lookback:])
    long_atr = np.mean(tr_arr[-(lookback * 2):])

    if long_atr <= 0 or np.isnan(long_atr) or np.isnan(short_atr):
        return 1.0, "HOLD"

    atr_ratio = short_atr / long_atr

    if atr_ratio < 0.7:
        return float(atr_ratio), "PASS"
    if atr_ratio > 1.5:
        return float(atr_ratio), "HOLD"
    return float(atr_ratio), "PASS"


# --- Public API ------------------------------------------------------------

def compute_sentiment_extremity_gate_signal(
    df: pd.DataFrame, context: dict | None = None
) -> dict:
    """Compute sentiment extremity gate signal.

    Generates BUY/SELL only in moderate F&G sentiment zones (30-70).
    Forces HOLD during extreme sentiment (adverse selection risk).
    Direction determined by price-in-range position, not F&G direction.
    """
    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD", "confidence": 0.0,
            "sub_signals": {}, "indicators": {},
        }

    ticker = (context or {}).get("ticker")
    if ticker and ticker not in _CRYPTO_TICKERS:
        return {
            "action": "HOLD", "confidence": 0.0,
            "sub_signals": {"reason": "non_crypto_ticker"},
            "indicators": {},
        }
    fg_value = _get_fg_value(ticker)

    if fg_value is None:
        return {
            "action": "HOLD", "confidence": 0.0,
            "sub_signals": {"reason": "no_fg_data"},
            "indicators": {},
        }

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    intensity, zone_vote = _intensity_zone(fg_value)
    price_pct, price_vote = _price_in_range(close, lookback=20)
    atr_ratio, compression_vote = _range_compression(high, low, close, lookback=14)

    if zone_vote == "HOLD":
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {
                "intensity_zone": "HOLD",
                "price_in_range": price_vote,
                "range_compression": compression_vote,
            },
            "indicators": {
                "fg_value": fg_value,
                "fg_intensity": float(intensity),
                "price_pct": float(price_pct),
                "atr_ratio": safe_float(atr_ratio),
            },
        }

    if compression_vote == "HOLD":
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {
                "intensity_zone": "PASS",
                "price_in_range": price_vote,
                "range_compression": "HOLD",
            },
            "indicators": {
                "fg_value": fg_value,
                "fg_intensity": float(intensity),
                "price_pct": float(price_pct),
                "atr_ratio": safe_float(atr_ratio),
            },
        }

    action = price_vote
    if action == "HOLD":
        confidence = 0.0
    else:
        base_conf = 0.5
        intensity_bonus = (20 - intensity) / 20 * 0.2
        if atr_ratio < 0.7:
            compression_bonus = (0.7 - atr_ratio) / 0.7 * 0.1
        else:
            compression_bonus = 0.0
        confidence = min(0.7, base_conf + intensity_bonus + compression_bonus)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "intensity_zone": "PASS",
            "price_in_range": price_vote,
            "range_compression": compression_vote,
        },
        "indicators": {
            "fg_value": fg_value,
            "fg_intensity": float(intensity),
            "price_pct": float(price_pct),
            "atr_ratio": safe_float(atr_ratio),
        },
    }
