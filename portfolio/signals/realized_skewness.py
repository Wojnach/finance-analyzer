"""Realized skewness directional signal module.

Computes 4 sub-indicators based on the 3rd moment of daily returns and returns
a majority-vote composite BUY/SELL/HOLD signal with confidence score.

Sub-indicators:
    1. Skewness Z-Score      — z-scored realized skewness vs rolling baseline
    2. Skewness Momentum     — 5-bar delta of skewness (acceleration)
    3. Kurtosis Confirmation — high kurtosis + negative skew = stronger signal
    4. Rolling Skew Regime   — short-window skew vs long-window skew divergence

Academic evidence:
    - Fernandez-Perez et al (2018): Sharpe 0.79, 8.01% annual on 27 commodity
      futures. Low-skew assets outperform high-skew assets.
    - ScienceDirect (2024): negative cross-sectional relationship between
      skewness risk and future crypto returns.

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 60 rows of data (for z-score normalization).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from portfolio.signal_utils import majority_vote, safe_float

# Minimum rows required.  We need at least ``SKEW_LOOKBACK`` returns for a
# reliable skewness estimate, but we fall back to ``MIN_ROWS`` when the full
# lookback is not yet available.
MIN_ROWS = 60
SKEW_LOOKBACK = 252       # ~1 year of daily data
NORM_WINDOW = 60          # z-score normalisation window
Z_BUY = -1.5             # negative skew = mean-reversion opportunity
Z_SELL = 1.5              # positive skew = momentum exhaustion
SKEW_MOM_PERIOD = 5       # bars for skewness momentum (acceleration)
SHORT_SKEW_WINDOW = 20    # short-window skew for regime divergence


def _compute_rolling_skewness(returns: pd.Series, window: int) -> pd.Series:
    """Compute rolling skewness using scipy.stats.skew (Fisher definition)."""
    return returns.rolling(window=window, min_periods=max(window // 2, 20)).apply(
        lambda x: stats.skew(x, nan_policy="omit") if len(x.dropna()) >= 20 else np.nan,
        raw=False,
    )


def _sub_skew_zscore(returns: pd.Series) -> tuple[float, str, dict]:
    """Sub-signal 1: z-scored realized skewness.

    Negative skew (fat left tail) historically precedes mean-reversion rallies.
    Positive skew (fat right tail) signals momentum exhaustion.
    """
    lookback = min(SKEW_LOOKBACK, len(returns))
    skew_val = stats.skew(returns.iloc[-lookback:].dropna())
    if np.isnan(skew_val):
        return 0.0, "HOLD", {"raw_skewness": np.nan, "skew_z": np.nan}

    # Compute rolling skewness for z-scoring
    rolling_skew = _compute_rolling_skewness(returns, lookback)
    recent = rolling_skew.iloc[-NORM_WINDOW:]
    mean_skew = recent.mean()
    std_skew = recent.std()

    if std_skew < 1e-8 or np.isnan(std_skew):
        return 0.0, "HOLD", {"raw_skewness": safe_float(skew_val), "skew_z": 0.0}

    z = (skew_val - mean_skew) / std_skew

    indicators = {"raw_skewness": safe_float(skew_val), "skew_z": safe_float(z)}

    if z < Z_BUY:
        confidence = min(abs(z) / 3.0, 1.0)
        return confidence, "BUY", indicators
    elif z > Z_SELL:
        confidence = min(abs(z) / 3.0, 1.0)
        return confidence, "SELL", indicators
    return 0.0, "HOLD", indicators


def _sub_skew_momentum(returns: pd.Series) -> tuple[float, str, dict]:
    """Sub-signal 2: skewness momentum (acceleration).

    If skewness is falling rapidly (becoming more negative), the asset is
    developing a fat left tail — contrarian BUY.  Rising skewness toward
    positive = SELL.
    """
    lookback = min(SKEW_LOOKBACK, len(returns))
    rolling_skew = _compute_rolling_skewness(returns, lookback)
    valid = rolling_skew.dropna()

    if len(valid) < SKEW_MOM_PERIOD + 1:
        return 0.0, "HOLD", {"skew_momentum": np.nan}

    current = valid.iloc[-1]
    past = valid.iloc[-(SKEW_MOM_PERIOD + 1)]
    delta = current - past

    indicators = {"skew_momentum": safe_float(delta)}

    # Falling skew (becoming more negative) = BUY
    if delta < -0.3:
        return min(abs(delta) / 1.0, 0.8), "BUY", indicators
    # Rising skew (becoming more positive) = SELL
    elif delta > 0.3:
        return min(abs(delta) / 1.0, 0.8), "SELL", indicators
    return 0.0, "HOLD", indicators


def _sub_kurtosis_confirm(returns: pd.Series) -> tuple[float, str, dict]:
    """Sub-signal 3: kurtosis confirmation.

    High kurtosis (fat tails) + negative skew = strong mean-reversion BUY.
    High kurtosis + positive skew = strong momentum exhaustion SELL.
    Low kurtosis = no extreme tail structure = HOLD.
    """
    lookback = min(SKEW_LOOKBACK, len(returns))
    window = returns.iloc[-lookback:].dropna()
    if len(window) < 20:
        return 0.0, "HOLD", {"kurtosis": np.nan}

    kurt = stats.kurtosis(window)  # excess kurtosis (0 for normal)
    skew_val = stats.skew(window)

    indicators = {
        "kurtosis": safe_float(kurt),
        "kurtosis_skew_product": safe_float(kurt * skew_val),
    }

    # High kurtosis threshold (excess kurtosis > 1.0 = leptokurtic)
    if kurt <= 1.0:
        return 0.0, "HOLD", indicators

    # Negative skew + high kurtosis = BUY (fat left tail, expected snap-back)
    if skew_val < -0.5:
        confidence = min(kurt / 5.0, 0.8)
        return confidence, "BUY", indicators
    # Positive skew + high kurtosis = SELL (fat right tail, expected pullback)
    elif skew_val > 0.5:
        confidence = min(kurt / 5.0, 0.8)
        return confidence, "SELL", indicators
    return 0.0, "HOLD", indicators


def _sub_skew_regime_divergence(returns: pd.Series) -> tuple[float, str, dict]:
    """Sub-signal 4: short-window vs long-window skewness divergence.

    When recent (20-bar) skewness diverges significantly from long-term
    (252-bar) skewness, it signals a regime change.
    """
    if len(returns) < SKEW_LOOKBACK:
        long_window = max(len(returns), MIN_ROWS)
    else:
        long_window = SKEW_LOOKBACK

    short_data = returns.iloc[-SHORT_SKEW_WINDOW:].dropna()
    long_data = returns.iloc[-long_window:].dropna()

    if len(short_data) < 15 or len(long_data) < 30:
        return 0.0, "HOLD", {"skew_divergence": np.nan}

    short_skew = stats.skew(short_data)
    long_skew = stats.skew(long_data)
    divergence = short_skew - long_skew

    indicators = {
        "short_skew": safe_float(short_skew),
        "long_skew": safe_float(long_skew),
        "skew_divergence": safe_float(divergence),
    }

    # Short-term skew much more negative than long-term = recent sell-off,
    # expect mean reversion = BUY
    if divergence < -0.8:
        return min(abs(divergence) / 2.0, 0.7), "BUY", indicators
    # Short-term skew much more positive = recent euphoria, expect pullback = SELL
    elif divergence > 0.8:
        return min(abs(divergence) / 2.0, 0.7), "SELL", indicators
    return 0.0, "HOLD", indicators


def compute_realized_skewness_signal(
    df: pd.DataFrame, context: dict = None
) -> dict:
    """Compute realized skewness directional signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        context: Optional dict with keys: ticker, config, asset_class, regime

    Returns:
        dict with keys: action, confidence, sub_signals, indicators
    """
    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    close = df["close"].copy()
    returns = close.pct_change(fill_method=None).dropna()

    if len(returns) < MIN_ROWS - 1:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    # Compute all sub-signals
    conf1, vote1, ind1 = _sub_skew_zscore(returns)
    conf2, vote2, ind2 = _sub_skew_momentum(returns)
    conf3, vote3, ind3 = _sub_kurtosis_confirm(returns)
    conf4, vote4, ind4 = _sub_skew_regime_divergence(returns)

    sub_signals = {
        "skew_zscore": vote1,
        "skew_momentum": vote2,
        "kurtosis_confirm": vote3,
        "skew_regime_divergence": vote4,
    }

    votes = [vote1, vote2, vote3, vote4]
    action, confidence = majority_vote(votes, count_hold=False)

    # Merge all indicators
    indicators = {}
    for d in [ind1, ind2, ind3, ind4]:
        indicators.update(d)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
