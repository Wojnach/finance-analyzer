"""Shannon Entropy signal module.

Uses Shannon entropy of price returns to measure market noise vs predictability.
Low entropy = trending/predictable regime (follow the trend). High entropy =
noisy/random regime (HOLD — no edge). This is fundamentally different from
volatility: a market can be volatile but trending (low entropy) or calm but
choppy (high entropy).

Sub-signals:
    1. Entropy Regime     — normalized entropy classification (trending/noisy/neutral)
    2. Trend Direction    — EMA(10) vs EMA(30) crossover direction
    3. Entropy Momentum   — is entropy rising or falling? (regime transition detection)
    4. Trend Strength     — magnitude of EMA spread as percentage of price

Source: Richard Shu (2025), Shannon Entropy breakout filter (29.6% Sharpe improvement).
        Preprints.org 202502.1717, entropy + LVQ on BTC (14.3% net profit).

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 60 rows of data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import ema, majority_vote, safe_float

MIN_ROWS = 60


def _compute_entropy(returns: np.ndarray, n_bins: int = 10) -> float:
    """Compute normalized Shannon entropy of return distribution.

    Returns value in [0, 1] where 0 = perfectly predictable, 1 = maximum noise.
    """
    if len(returns) < 10 or np.all(np.isnan(returns)):
        return float("nan")

    clean = returns[~np.isnan(returns)]
    if len(clean) < 10:
        return float("nan")

    counts, _ = np.histogram(clean, bins=n_bins)
    total = counts.sum()
    if total == 0:
        return float("nan")

    probs = counts / total
    probs = probs[probs > 0]

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(n_bins)

    if max_entropy == 0:
        return float("nan")

    return float(entropy / max_entropy)


# ---- sub-signal 1: Entropy Regime -------------------------------------------

def _entropy_regime(close: pd.Series, lookback: int = 50, n_bins: int = 10) -> tuple[float, str]:
    """Classify current entropy regime.

    Returns (normalized_entropy, regime_label).
    regime_label: "trending" (<0.65), "neutral" (0.65-0.82), "noisy" (>0.82)
    """
    returns = close.pct_change().dropna().values
    if len(returns) < lookback:
        return float("nan"), "unknown"

    window = returns[-lookback:]
    ent = _compute_entropy(window, n_bins)

    if np.isnan(ent):
        return float("nan"), "unknown"

    if ent < 0.65:
        return ent, "trending"
    if ent > 0.82:
        return ent, "noisy"
    return ent, "neutral"


# ---- sub-signal 2: Trend Direction ------------------------------------------

def _trend_direction(close: pd.Series, fast: int = 10, slow: int = 30) -> tuple[float, str]:
    """EMA crossover trend direction.

    Returns (spread_pct, direction).
    """
    if len(close) < slow + 5:
        return 0.0, "HOLD"

    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)

    fast_val = safe_float(ema_fast.iloc[-1])
    slow_val = safe_float(ema_slow.iloc[-1])

    if np.isnan(fast_val) or np.isnan(slow_val) or slow_val == 0:
        return 0.0, "HOLD"

    spread_pct = (fast_val - slow_val) / slow_val

    # Require minimum 0.1% spread to avoid noise
    if spread_pct > 0.001:
        return float(spread_pct), "BUY"
    if spread_pct < -0.001:
        return float(spread_pct), "SELL"
    return float(spread_pct), "HOLD"


# ---- sub-signal 3: Entropy Momentum -----------------------------------------

def _entropy_momentum(close: pd.Series, lookback: int = 50, n_bins: int = 10) -> tuple[float, str]:
    """Is entropy rising (becoming noisier) or falling (becoming more predictable)?

    Compare current entropy to entropy 10 bars ago.
    Returns (delta, signal).
    """
    returns = close.pct_change().dropna().values
    if len(returns) < lookback + 10:
        return 0.0, "HOLD"

    ent_now = _compute_entropy(returns[-lookback:], n_bins)
    ent_prev = _compute_entropy(returns[-(lookback + 10):-10], n_bins)

    if np.isnan(ent_now) or np.isnan(ent_prev):
        return 0.0, "HOLD"

    delta = ent_now - ent_prev

    # Falling entropy = market becoming more predictable = follow trend
    if delta < -0.05:
        return float(delta), "trending_strengthening"
    # Rising entropy = market becoming noisier = reduce confidence
    if delta > 0.05:
        return float(delta), "noise_increasing"
    return float(delta), "stable"


# ---- sub-signal 4: Trend Strength ------------------------------------------

def _trend_strength(close: pd.Series) -> tuple[float, str]:
    """Magnitude of recent directional move using ROC(10).

    Returns (roc_pct, strength_label).
    """
    if len(close) < 11:
        return 0.0, "weak"

    current = safe_float(close.iloc[-1])
    past = safe_float(close.iloc[-11])

    if np.isnan(current) or np.isnan(past) or past == 0:
        return 0.0, "weak"

    roc = (current - past) / past

    if abs(roc) > 0.03:
        return float(roc), "strong"
    if abs(roc) > 0.01:
        return float(roc), "moderate"
    return float(roc), "weak"


# ---- Main composite signal --------------------------------------------------

def compute_shannon_entropy_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute Shannon entropy-based market regime signal.

    Low entropy + trend → follow trend (BUY/SELL).
    High entropy → HOLD (no edge in noisy markets).
    Entropy momentum adds confidence adjustment.

    Args:
        df: DataFrame with columns: open, high, low, close, volume (min 60 rows)
        context: Optional dict with keys: ticker, config, asset_class, regime

    Returns:
        dict with keys: action, confidence, sub_signals, indicators
    """
    if df is None or len(df) < MIN_ROWS:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    close = df["close"].copy()

    # Drop NaN rows from close
    if close.isna().sum() > len(close) * 0.3:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    close = close.ffill()

    # Compute sub-signals
    entropy_val, regime = _entropy_regime(close)
    spread_pct, trend_dir = _trend_direction(close)
    ent_delta, ent_momentum_label = _entropy_momentum(close)
    roc_val, strength_label = _trend_strength(close)

    # Build indicators dict
    indicators = {
        "normalized_entropy": safe_float(entropy_val),
        "entropy_regime": regime,
        "ema_spread_pct": safe_float(spread_pct),
        "entropy_delta": safe_float(ent_delta),
        "entropy_momentum": ent_momentum_label,
        "roc_10": safe_float(roc_val),
        "trend_strength": strength_label,
    }

    # Decision logic
    if regime == "unknown" or np.isnan(entropy_val):
        return {
            "action": "HOLD", "confidence": 0.0,
            "sub_signals": {"entropy_regime": "HOLD", "trend": "HOLD",
                            "entropy_momentum": "HOLD", "trend_strength": "HOLD"},
            "indicators": indicators,
        }

    # High entropy → HOLD (noisy market, no edge)
    if regime == "noisy":
        return {
            "action": "HOLD", "confidence": 0.0,
            "sub_signals": {"entropy_regime": "HOLD", "trend": trend_dir,
                            "entropy_momentum": "HOLD", "trend_strength": "HOLD"},
            "indicators": indicators,
        }

    # Low entropy → follow the trend
    if regime == "trending":
        base_confidence = 0.5

        # Stronger trend = higher confidence
        if strength_label == "strong":
            base_confidence += 0.15
        elif strength_label == "moderate":
            base_confidence += 0.08

        # Entropy falling (becoming more predictable) = boost
        if ent_momentum_label == "trending_strengthening":
            base_confidence += 0.10
        # Entropy rising even though still low = reduce
        elif ent_momentum_label == "noise_increasing":
            base_confidence -= 0.10

        # How far below the trending threshold? Lower = more predictable
        entropy_depth = max(0, 0.65 - entropy_val)  # 0 to ~0.65
        base_confidence += entropy_depth * 0.3  # up to +0.195

        base_confidence = max(0.1, min(0.7, base_confidence))

        if trend_dir == "BUY":
            action = "BUY"
        elif trend_dir == "SELL":
            action = "SELL"
        else:
            action = "HOLD"
            base_confidence = 0.0

        sub_signals = {
            "entropy_regime": action,
            "trend": trend_dir,
            "entropy_momentum": action if ent_momentum_label == "trending_strengthening" else "HOLD",
            "trend_strength": action if strength_label in ("strong", "moderate") else "HOLD",
        }

        return {
            "action": action,
            "confidence": round(base_confidence, 4),
            "sub_signals": sub_signals,
            "indicators": indicators,
        }

    # Neutral entropy → use trend but with reduced confidence
    if trend_dir in ("BUY", "SELL") and strength_label in ("strong", "moderate"):
        base_confidence = 0.3

        if ent_momentum_label == "trending_strengthening":
            base_confidence += 0.10
        elif ent_momentum_label == "noise_increasing":
            base_confidence -= 0.15

        base_confidence = max(0.1, min(0.5, base_confidence))

        sub_signals = {
            "entropy_regime": "HOLD",
            "trend": trend_dir,
            "entropy_momentum": trend_dir if ent_momentum_label == "trending_strengthening" else "HOLD",
            "trend_strength": trend_dir if strength_label == "strong" else "HOLD",
        }

        return {
            "action": trend_dir,
            "confidence": round(base_confidence, 4),
            "sub_signals": sub_signals,
            "indicators": indicators,
        }

    # Default: neutral entropy, weak/no trend → HOLD
    return {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {"entropy_regime": "HOLD", "trend": trend_dir,
                        "entropy_momentum": "HOLD", "trend_strength": "HOLD"},
        "indicators": indicators,
    }
