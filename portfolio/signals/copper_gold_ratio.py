"""Copper/Gold ratio intermarket signal module.

The copper/gold ratio is one of the most reliable cross-asset regime
indicators in finance.  Copper is the most cyclical industrial metal
(construction, electronics, EVs), while gold is the quintessential
safe haven.  A falling ratio signals economic weakness; a rising ratio
signals expansion.

Key properties:
    - 94% recession prediction accuracy (3-month sustained move)
    - 0.85 correlation with 10Y Treasury yields historically
    - Currently at 5-decade lows (strongest risk-off signal in 50 years)

Sub-signals:
    1. Ratio z-score       — distance from rolling mean (regime strength)
    2. Ratio trend         — SMA(50) vs SMA(200) (structural direction)
    3. Ratio momentum      — 20-day rate-of-change (recent velocity)
    4. Copper-gold spread  — difference in percentage returns (divergence)

Asset-class aware:
    - Metals (XAU/XAG): falling ratio = gold strength = BUY
    - Risk assets (BTC/ETH/MSTR): falling ratio = recession = SELL

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 60 rows of data (for 50-period z-score + lookback).

Data: copper futures (HG=F) and gold futures (GC=F) via yfinance/price_source.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float, sma

logger = logging.getLogger(__name__)

MIN_ROWS = 60  # Need 50 for z-score + buffer
_METALS_TICKERS = {"XAU-USD", "XAG-USD"}
_CACHE: dict = {}
_CACHE_TTL = 300  # 5-minute cache for external fetches


def _extract_close(df: pd.DataFrame) -> pd.Series | None:
    """Extract Close price series from a DataFrame with flat or MultiIndex columns."""
    if df is None or df.empty:
        return None

    # Flatten MultiIndex columns (yfinance quirk: ("Close", "HG=F"))
    if isinstance(df.columns, pd.MultiIndex):
        # Try to find a "Close" column at level 0
        close_cols = [c for c in df.columns if c[0].lower() == "close"]
        if close_cols:
            return df[close_cols[0]].dropna()
        return None

    # Flat columns: try case-insensitive match
    col_map = {c.lower(): c for c in df.columns}
    close_name = col_map.get("close")
    if close_name is not None:
        return df[close_name].dropna()
    return None


def _fetch_ratio_data() -> pd.DataFrame | None:
    """Fetch copper and gold futures daily data, return ratio Series."""
    now = time.time()
    cached = _CACHE.get("ratio_df")
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    try:
        # Use the price_source router (same as metals_cross_assets.py)
        from portfolio.price_source import download as _download
        copper_df = _download("HG=F", period="1y", interval="1d")
        gold_df = _download("GC=F", period="1y", interval="1d")
    except Exception:
        try:
            import yfinance as yf
            copper_df = yf.download("HG=F", period="1y", interval="1d",
                                    progress=False, auto_adjust=True)
            gold_df = yf.download("GC=F", period="1y", interval="1d",
                                  progress=False, auto_adjust=True)
        except Exception as exc:
            logger.warning("copper_gold_ratio: data fetch failed: %s", exc)
            return None

    if copper_df is None or gold_df is None:
        return None
    if copper_df.empty or gold_df.empty:
        return None

    # Extract close prices, handling both flat and MultiIndex columns
    copper_close = _extract_close(copper_df)
    gold_close = _extract_close(gold_df)

    if copper_close is None or gold_close is None:
        return None

    # Align on dates
    combined = pd.DataFrame({
        "copper": copper_close,
        "gold": gold_close,
    }).dropna()

    if len(combined) < MIN_ROWS:
        return None

    combined["ratio"] = combined["copper"] / combined["gold"]
    _CACHE["ratio_df"] = (now, combined)
    return combined


def _ratio_zscore(ratio: pd.Series, window: int = 50) -> float:
    """Z-score of copper/gold ratio relative to rolling window."""
    if len(ratio) < window:
        return 0.0
    rolling_mean = ratio.rolling(window).mean()
    rolling_std = ratio.rolling(window).std()
    std_val = rolling_std.iloc[-1]
    if np.isnan(std_val) or std_val < 1e-10:
        return 0.0
    return float((ratio.iloc[-1] - rolling_mean.iloc[-1]) / std_val)


def _ratio_trend(ratio: pd.Series) -> int:
    """Structural trend: +1 if SMA(50) > SMA(200), -1 otherwise."""
    if len(ratio) < 200:
        # Fall back to shorter window
        if len(ratio) < 50:
            return 0
        sma_short = ratio.rolling(20).mean().iloc[-1]
        sma_long = ratio.rolling(50).mean().iloc[-1]
    else:
        sma_short = ratio.rolling(50).mean().iloc[-1]
        sma_long = ratio.rolling(200).mean().iloc[-1]

    if np.isnan(sma_short) or np.isnan(sma_long):
        return 0
    return 1 if sma_short > sma_long else -1


def _ratio_momentum(ratio: pd.Series, periods: int = 20) -> float:
    """20-day rate-of-change of the ratio."""
    if len(ratio) < periods + 1:
        return 0.0
    old = ratio.iloc[-1 - periods]
    if np.isnan(old) or old < 1e-10:
        return 0.0
    return float(ratio.iloc[-1] / old - 1)


def _copper_gold_spread(combined: pd.DataFrame, periods: int = 20) -> float:
    """Difference in percentage returns (copper_ret - gold_ret)."""
    if len(combined) < periods + 1:
        return 0.0
    copper_ret = combined["copper"].iloc[-1] / combined["copper"].iloc[-1 - periods] - 1
    gold_ret = combined["gold"].iloc[-1] / combined["gold"].iloc[-1 - periods] - 1
    return float(copper_ret - gold_ret)


def compute_copper_gold_ratio_signal(
    df: pd.DataFrame, context: dict | None = None,
) -> dict:
    """Compute copper/gold ratio intermarket signal.

    Args:
        df: OHLCV DataFrame (used for row count check; actual data fetched
            externally from copper/gold futures).
        context: Optional dict with keys: ticker, config, asset_class, regime.

    Returns:
        dict with action, confidence, sub_signals, indicators.
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    if df is None or len(df) < 10:
        return empty

    context = context or {}
    ticker = context.get("ticker", "")
    asset_class = context.get("asset_class", "")

    # Determine if this is a metals ticker (inverted signal direction)
    is_metals = ticker in _METALS_TICKERS or asset_class == "metals"

    # Fetch copper/gold ratio data
    combined = _fetch_ratio_data()
    if combined is None or len(combined) < MIN_ROWS:
        return empty

    ratio = combined["ratio"]

    # --- Sub-signal 1: Ratio z-score ---
    zscore = _ratio_zscore(ratio, window=50)
    if zscore < -2.0:
        zscore_vote = "SELL"  # Strong risk-off
    elif zscore < -1.5:
        zscore_vote = "SELL"  # Risk-off
    elif zscore > 2.0:
        zscore_vote = "BUY"   # Strong expansion
    elif zscore > 1.5:
        zscore_vote = "BUY"   # Expansion
    else:
        zscore_vote = "HOLD"

    # --- Sub-signal 2: Ratio trend ---
    trend = _ratio_trend(ratio)
    if trend == 1:
        trend_vote = "BUY"    # Expansion regime
    elif trend == -1:
        trend_vote = "SELL"   # Contraction regime
    else:
        trend_vote = "HOLD"

    # --- Sub-signal 3: Ratio momentum ---
    momentum = _ratio_momentum(ratio, periods=20)
    if momentum > 0.05:
        momentum_vote = "BUY"   # Ratio rising fast (expansion)
    elif momentum < -0.05:
        momentum_vote = "SELL"  # Ratio falling fast (contraction)
    else:
        momentum_vote = "HOLD"

    # --- Sub-signal 4: Copper-gold spread ---
    spread = _copper_gold_spread(combined, periods=20)
    if spread > 0.03:
        spread_vote = "BUY"    # Copper outperforming gold (risk-on)
    elif spread < -0.03:
        spread_vote = "SELL"   # Gold outperforming copper (risk-off)
    else:
        spread_vote = "HOLD"

    # Aggregate votes
    votes = [zscore_vote, trend_vote, momentum_vote, spread_vote]
    action, confidence = majority_vote(votes, count_hold=False)

    # For metals tickers: INVERT the signal direction.
    # Falling copper/gold ratio = gold strength = BUY metals.
    # Rising copper/gold ratio = risk-on = SELL metals (rotation out of gold).
    if is_metals and action != "HOLD":
        action = "SELL" if action == "BUY" else "BUY"

    # Cap confidence at 0.7 for external data signals
    confidence = min(confidence, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "ratio_zscore": zscore_vote,
            "ratio_trend": trend_vote,
            "ratio_momentum": momentum_vote,
            "copper_gold_spread": spread_vote,
        },
        "indicators": {
            "ratio": safe_float(ratio.iloc[-1]),
            "ratio_zscore": round(zscore, 4),
            "ratio_trend": trend,
            "ratio_momentum_20d": round(momentum, 4),
            "copper_gold_spread_20d": round(spread, 4),
            "copper_price": safe_float(combined["copper"].iloc[-1]),
            "gold_price": safe_float(combined["gold"].iloc[-1]),
        },
    }
