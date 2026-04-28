"""Drift Regime Gate signal module.

Detects whether an asset is in a "drift" regime (persistent positive/negative
close-to-close moves) and uses this to emit directional signals based on
mean-reversion vs momentum logic.

Sub-indicators:
    1. Drift Fraction    — % of positive close-to-close days in trailing window
    2. Drift Velocity    — rate of change of drift fraction (accelerating/decelerating)
    3. Price vs SMA      — ATR-normalised distance from SMA for direction

Research basis:
    "Discovery of a 13-Sharpe OOS Factor: Drift Regimes Unlock Hidden
    Cross-Sectional Predictability" (arxiv:2511.12490, 2025).
    Activating value + reversal signals ONLY during drift regimes (>60% positive
    days in trailing 63-day windows) yields OOS Sharpe >13 on 20 years of
    walk-forward S&P 500 data, p<0.001 across 1000 randomisation trials.

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 65 rows of data (63-bar lookback + 2 for diff/shift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float, sma, true_range

MIN_ROWS = 65  # 63-bar lookback + 2 for shift/diff


# ---- sub-indicator 1: Drift Fraction ----------------------------------------

def _drift_fraction(close: pd.Series, lookback: int = 63) -> tuple[float, str]:
    """Fraction of positive close-to-close moves in the trailing window.

    >0.60 → asset is in an "up-drift" regime → overextended → SELL (mean reversion).
    <0.40 → asset is in a "down-drift" regime → oversold → BUY (contrarian).
    0.40–0.60 → no clear drift → HOLD.

    Returns (fraction_value, signal).
    """
    if len(close) < lookback + 1:
        return float("nan"), "HOLD"

    changes = close.diff().iloc[-lookback:]
    positive = (changes > 0).sum()
    frac = positive / lookback

    frac_val = safe_float(frac)
    if np.isnan(frac_val):
        return float("nan"), "HOLD"

    if frac_val > 0.60:
        return frac_val, "SELL"
    if frac_val < 0.40:
        return frac_val, "BUY"
    return frac_val, "HOLD"


# ---- sub-indicator 2: Drift Velocity ----------------------------------------

def _drift_velocity(close: pd.Series, lookback: int = 63,
                    velocity_period: int = 10) -> tuple[float, str]:
    """Rate of change of drift fraction over *velocity_period* bars.

    Rising drift velocity (>0.05) during up-drift → SELL (accelerating overextension).
    Falling drift velocity (<-0.05) during down-drift → BUY (capitulation exhaustion).
    Flat or moderate velocity → HOLD.

    Returns (velocity_value, signal).
    """
    if len(close) < lookback + velocity_period + 1:
        return float("nan"), "HOLD"

    # Current drift fraction
    changes_now = close.diff().iloc[-lookback:]
    frac_now = (changes_now > 0).sum() / lookback

    # Drift fraction *velocity_period* bars ago
    changes_prev = close.diff().iloc[-(lookback + velocity_period):-velocity_period]
    if len(changes_prev) < lookback:
        return float("nan"), "HOLD"
    frac_prev = (changes_prev > 0).sum() / lookback

    velocity = frac_now - frac_prev
    vel_val = safe_float(velocity)
    if np.isnan(vel_val):
        return float("nan"), "HOLD"

    # Accelerating up-drift → SELL, accelerating down-drift → BUY
    if vel_val > 0.05 and frac_now > 0.55:
        return vel_val, "SELL"
    if vel_val < -0.05 and frac_now < 0.45:
        return vel_val, "BUY"
    return vel_val, "HOLD"


# ---- sub-indicator 3: Price vs SMA ------------------------------------------

def _price_vs_sma(close: pd.Series, high: pd.Series, low: pd.Series,
                  sma_period: int = 63, atr_period: int = 14) -> tuple[float, str]:
    """ATR-normalised distance of price from SMA.

    When drift fraction indicates overextended (up-drift) AND price is far above SMA,
    SELL is confirmed. When down-drift AND price is far below SMA, BUY is confirmed.

    Returns (normalised_distance, signal).
    """
    if len(close) < max(sma_period, atr_period) + 1:
        return float("nan"), "HOLD"

    sma_val = sma(close, sma_period).iloc[-1]
    tr = true_range(high, low, close)
    atr_val = tr.rolling(window=atr_period, min_periods=atr_period).mean().iloc[-1]

    if np.isnan(sma_val) or np.isnan(atr_val) or atr_val <= 0:
        return float("nan"), "HOLD"

    dist = safe_float((close.iloc[-1] - sma_val) / atr_val)
    if np.isnan(dist):
        return float("nan"), "HOLD"

    if dist > 1.5:
        return dist, "SELL"
    if dist < -1.5:
        return dist, "BUY"
    return dist, "HOLD"


# ---- composite ---------------------------------------------------------------

def compute_drift_regime_gate_signal(
    df: pd.DataFrame, context: dict = None,
) -> dict:
    """Compute drift regime gate signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        context: Optional dict with keys: ticker, config, asset_class, regime

    Returns:
        dict with keys:
            action: "BUY" | "SELL" | "HOLD"
            confidence: float 0.0–1.0
            sub_signals: dict of sub-indicator votes
            indicators: dict of raw indicator values
    """
    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # Compute sub-signals
    frac_val, frac_vote = _drift_fraction(close)
    vel_val, vel_vote = _drift_velocity(close)
    dist_val, dist_vote = _price_vs_sma(close, high, low)

    votes = [frac_vote, vel_vote, dist_vote]
    action, confidence = majority_vote(votes, count_hold=False)

    # Cap confidence at 0.7 (this is a regime-based signal, not pure price)
    confidence = min(confidence, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "drift_fraction": frac_vote,
            "drift_velocity": vel_vote,
            "price_vs_sma": dist_vote,
        },
        "indicators": {
            "drift_fraction": safe_float(frac_val),
            "drift_velocity": safe_float(vel_val),
            "price_vs_sma_atr": safe_float(dist_val),
        },
    }
