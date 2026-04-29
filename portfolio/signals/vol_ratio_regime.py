"""Volatility ratio regime detector + directional signal.

Combines three orthogonal regime measures to classify market state as
trending or ranging, then generates directional BUY/SELL/HOLD signals:

Sub-indicators:
    1. GK/CC Ratio   — Garman-Klass vs close-to-close volatility ratio.
                        High ratio = large intrabar swings, small net moves (ranging).
                        Low ratio = consistent close-to-close moves (trending).
    2. Variance Ratio — k-period VR test. VR < 1 = mean-reverting, VR > 1 = trending.
    3. Efficiency Ratio (Kaufman) — displacement / path distance. High = trending.

Regime rules:
    Trending: GK/CC ratio < 1.5 AND VR > 0.85 AND ER > 0.25
    Ranging:  GK/CC ratio > 2.0 AND VR < 0.75 AND ER < 0.20
    Uncertain: everything else → HOLD

Directional logic (per regime):
    Ranging + price < SMA(20) → BUY  (mean reversion: price below equilibrium)
    Ranging + price > SMA(20) → SELL (mean reversion: price above equilibrium)
    Trending + SMA(10) > SMA(20) → BUY  (momentum up)
    Trending + SMA(10) < SMA(20) → SELL (momentum down)

Source: Signal research backlog (2026-04-28). Garman-Klass estimator from
Garman & Klass (1980), VR test from Lo & MacKinlay (1988), ER from Kaufman.

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 40 rows of data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float, sma

MIN_ROWS = 40  # Need enough history for 20-bar rolling windows + VR calc


def _garman_klass_cc_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Compute rolling ratio of Garman-Klass to close-to-close variance.

    GK_var = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
    CC_var = ln(C[t] / C[t-1])^2

    Returns rolling SMA(GK_var) / SMA(CC_var).
    """
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])

    # Clamp to >= 0: anomalous ticks can produce negative GK variance
    gk_var = np.maximum(0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2, 0.0)
    cc_var = np.log(df["close"] / df["close"].shift(1)) ** 2

    gk_sma = gk_var.rolling(window=window, min_periods=window).mean()
    cc_sma = cc_var.rolling(window=window, min_periods=window).mean()

    # Avoid division by zero
    cc_sma_safe = cc_sma.replace(0, np.nan)
    return gk_sma / cc_sma_safe


def _variance_ratio(close: pd.Series, k: int = 5, window: int = 20) -> pd.Series:
    """Rolling variance ratio test.

    VR(k) = Var(k-period returns) / (k * Var(1-period returns))
    VR ~ 1.0: random walk
    VR < 1.0: mean-reverting
    VR > 1.0: trending / momentum
    """
    log_ret_1 = np.log(close / close.shift(1))
    log_ret_k = np.log(close / close.shift(k))

    var_1 = log_ret_1.rolling(window=window, min_periods=window).var()
    var_k = log_ret_k.rolling(window=window, min_periods=window).var()

    var_1_safe = var_1.replace(0, np.nan)
    return var_k / (k * var_1_safe)


def _efficiency_ratio(close: pd.Series, period: int = 20) -> pd.Series:
    """Kaufman Efficiency Ratio.

    ER = |displacement| / path_distance
    ER → 1.0: trending (straight-line price movement)
    ER → 0.0: noisy / choppy (lots of movement, no net progress)
    """
    displacement = (close - close.shift(period)).abs()
    path = close.diff().abs().rolling(window=period, min_periods=period).sum()

    path_safe = path.replace(0, np.nan)
    return displacement / path_safe


def _classify_regime(gk_cc: float, vr: float, er: float) -> str:
    """Classify market regime from the three indicators."""
    if np.isnan(gk_cc) or np.isnan(vr) or np.isnan(er):
        return "uncertain"

    trending_votes = 0
    ranging_votes = 0

    # GK/CC ratio
    if gk_cc < 1.5:
        trending_votes += 1
    elif gk_cc > 2.0:
        ranging_votes += 1

    # Variance ratio
    if vr > 0.85:
        trending_votes += 1
    elif vr < 0.75:
        ranging_votes += 1

    # Efficiency ratio
    if er > 0.25:
        trending_votes += 1
    elif er < 0.20:
        ranging_votes += 1

    if trending_votes >= 2:
        return "trending"
    if ranging_votes >= 2:
        return "ranging"
    return "uncertain"


def compute_vol_ratio_regime_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute volatility ratio regime signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume (min 40 rows)
        context: Optional dict with keys: ticker, config, asset_class, regime

    Returns:
        dict with keys: action, confidence, sub_signals, indicators
    """
    _HOLD = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
    if df is None or len(df) < MIN_ROWS:
        return _HOLD
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return _HOLD

    close = df["close"].astype(float)

    # Compute the three regime indicators
    gk_cc_series = _garman_klass_cc_ratio(df)
    vr_series = _variance_ratio(close)
    er_series = _efficiency_ratio(close)

    gk_cc = safe_float(gk_cc_series.iloc[-1])
    vr = safe_float(vr_series.iloc[-1])
    er = safe_float(er_series.iloc[-1])

    # Z-score the GK/CC ratio for normalized reporting
    gk_cc_mean = gk_cc_series.rolling(60, min_periods=20).mean()
    gk_cc_std = gk_cc_series.rolling(60, min_periods=20).std()
    std_val = gk_cc_std.iloc[-1]
    if std_val is not None and np.isfinite(std_val) and std_val > 0:
        gk_cc_z = safe_float(
            (gk_cc_series.iloc[-1] - gk_cc_mean.iloc[-1]) / std_val
        )
    else:
        gk_cc_z = 0.0

    # Classify regime
    regime = _classify_regime(gk_cc, vr, er)

    # Directional sub-signals
    sma_10 = sma(close, 10)
    sma_20 = sma(close, 20)
    current_price = safe_float(close.iloc[-1])
    sma_20_val = safe_float(sma_20.iloc[-1])
    sma_10_val = safe_float(sma_10.iloc[-1])

    votes = []

    if regime == "ranging":
        # Mean-reversion logic
        if current_price < sma_20_val and sma_20_val > 0:
            votes.append("BUY")
        elif current_price > sma_20_val and sma_20_val > 0:
            votes.append("SELL")
        else:
            votes.append("HOLD")

        # Additional ranging sub-signal: Bollinger %B mean reversion
        bb_mid = sma_20
        bb_std = close.rolling(20, min_periods=20).std()
        if bb_std.iloc[-1] and bb_std.iloc[-1] > 0:
            pct_b = (current_price - (bb_mid.iloc[-1] - 2 * bb_std.iloc[-1])) / (
                4 * bb_std.iloc[-1]
            )
            if pct_b < 0.2:
                votes.append("BUY")
            elif pct_b > 0.8:
                votes.append("SELL")
            else:
                votes.append("HOLD")
        else:
            votes.append("HOLD")

    elif regime == "trending":
        # Momentum logic
        if sma_10_val > sma_20_val:
            votes.append("BUY")
        elif sma_10_val < sma_20_val:
            votes.append("SELL")
        else:
            votes.append("HOLD")

        # Additional trending sub-signal: price vs SMA(10) distance
        if sma_10_val > 0:
            dist_pct = (current_price - sma_10_val) / sma_10_val
            if dist_pct > 0.005:
                votes.append("BUY")
            elif dist_pct < -0.005:
                votes.append("SELL")
            else:
                votes.append("HOLD")
        else:
            votes.append("HOLD")

    else:
        # Uncertain regime → HOLD
        votes.append("HOLD")
        votes.append("HOLD")

    # Regime confidence: how clearly the indicators agree
    regime_clarity = 0.0
    if regime == "trending":
        clarity_signals = [
            1 if gk_cc < 1.5 else 0,
            1 if vr > 0.85 else 0,
            1 if er > 0.25 else 0,
        ]
        regime_clarity = sum(clarity_signals) / 3.0
    elif regime == "ranging":
        clarity_signals = [
            1 if gk_cc > 2.0 else 0,
            1 if vr < 0.75 else 0,
            1 if er < 0.20 else 0,
        ]
        regime_clarity = sum(clarity_signals) / 3.0

    action, vote_confidence = majority_vote(votes, count_hold=False)

    # Scale confidence by regime clarity
    confidence = min(vote_confidence * regime_clarity, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "gk_cc_regime": "ranging" if gk_cc > 2.0 else ("trending" if gk_cc < 1.5 else "neutral"),
            "vr_regime": "ranging" if vr < 0.75 else ("trending" if vr > 0.85 else "neutral"),
            "er_regime": "ranging" if er < 0.20 else ("trending" if er > 0.25 else "neutral"),
            "composite_regime": regime,
            "directional_vote": action,
        },
        "indicators": {
            "gk_cc_ratio": round(gk_cc, 4) if not np.isnan(gk_cc) else None,
            "gk_cc_z": round(gk_cc_z, 4) if not np.isnan(gk_cc_z) else None,
            "variance_ratio": round(vr, 4) if not np.isnan(vr) else None,
            "efficiency_ratio": round(er, 4) if not np.isnan(er) else None,
            "regime": regime,
            "regime_clarity": round(regime_clarity, 4),
            "sma_10": round(sma_10_val, 4) if not np.isnan(sma_10_val) else None,
            "sma_20": round(sma_20_val, 4) if not np.isnan(sma_20_val) else None,
        },
    }
