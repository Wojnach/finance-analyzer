"""Gold/Platinum ratio cross-asset risk appetite signal.

The log gold/platinum price ratio captures tail risk and geopolitical fear.
Both are precious metals, but platinum has substantially more industrial
demand (~60% industrial vs gold's ~10%).  A rising GP ratio signals that
investors are paying a fear premium for gold relative to platinum.

Key properties (Huang & Kilic 2019, JFE 132(3), 50-75):
    - One std dev GP increase predicts 6.4% annual excess stock returns
    - Outperforms nearly all existing return predictors at 12-month horizon
    - Peaks during all NBER recessions 1975-2013, Oct 1987, LTCM 1998, 2011
    - Correlated with option-implied tail risk measures

Sub-signals:
    1. GP z-score        — distance from 252d rolling mean (fear level)
    2. GP trend          — SMA(50) vs SMA(200) (structural direction)
    3. GP momentum       — 21-day rate-of-change (fear velocity)
    4. Gold-platinum spread — difference in 20d percentage returns

Asset-class aware:
    - Risk assets (BTC/ETH/MSTR): high GP = contrarian BUY (fear = future recovery)
    - Metals (XAU/XAG): high GP = fear already priced in = SELL

Data: gold futures (GC=F) and platinum futures (PL=F) via price_source/yfinance.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger(__name__)

MIN_ROWS = 60
_METALS_TICKERS = {"XAU-USD", "XAG-USD"}
_CACHE: dict = {}
_CACHE_TTL = 300


def _extract_close(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        close_cols = [c for c in df.columns if c[0].lower() == "close"]
        if close_cols:
            return df[close_cols[0]].dropna()
        return None
    col_map = {c.lower(): c for c in df.columns}
    close_name = col_map.get("close")
    if close_name is not None:
        return df[close_name].dropna()
    return None


def _fetch_gp_data() -> pd.DataFrame | None:
    now = time.time()
    cached = _CACHE.get("gp_df")
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]

    try:
        from portfolio.price_source import download as _download
        gold_df = _download("GC=F", period="2y", interval="1d")
        plat_df = _download("PL=F", period="2y", interval="1d")
    except Exception:
        try:
            import yfinance as yf
            gold_df = yf.download("GC=F", period="2y", interval="1d",
                                  progress=False, auto_adjust=True)
            plat_df = yf.download("PL=F", period="2y", interval="1d",
                                  progress=False, auto_adjust=True)
        except Exception as exc:
            logger.warning("gold_platinum_ratio_risk: data fetch failed: %s", exc)
            return None

    if gold_df is None or plat_df is None or gold_df.empty or plat_df.empty:
        return None

    gold_close = _extract_close(gold_df)
    plat_close = _extract_close(plat_df)
    if gold_close is None or plat_close is None:
        return None

    combined = pd.DataFrame({
        "gold": gold_close,
        "platinum": plat_close,
    }).dropna()

    if len(combined) < MIN_ROWS:
        return None

    combined["gp_ratio"] = np.log(combined["gold"]) - np.log(combined["platinum"])
    _CACHE["gp_df"] = (now, combined)
    return combined


def _gp_zscore(ratio: pd.Series, window: int = 252) -> float:
    if len(ratio) < window:
        window = max(len(ratio) // 2, 20)
    rolling_mean = ratio.rolling(window).mean()
    rolling_std = ratio.rolling(window).std()
    std_val = rolling_std.iloc[-1]
    if np.isnan(std_val) or std_val < 1e-10:
        return 0.0
    return float((ratio.iloc[-1] - rolling_mean.iloc[-1]) / std_val)


def _gp_trend(ratio: pd.Series) -> int:
    if len(ratio) < 200:
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


def _gp_momentum(ratio: pd.Series, periods: int = 21) -> float:
    if len(ratio) < periods + 1:
        return 0.0
    old = ratio.iloc[-1 - periods]
    if np.isnan(old) or abs(old) < 1e-10:
        return 0.0
    return float(ratio.iloc[-1] - old)


def _gold_plat_spread(combined: pd.DataFrame, periods: int = 20) -> float:
    if len(combined) < periods + 1:
        return 0.0
    gold_base = combined["gold"].iloc[-1 - periods]
    plat_base = combined["platinum"].iloc[-1 - periods]
    if np.isnan(gold_base) or np.isnan(plat_base) or abs(gold_base) < 1e-10 or abs(plat_base) < 1e-10:
        return 0.0
    gold_ret = combined["gold"].iloc[-1] / gold_base - 1
    plat_ret = combined["platinum"].iloc[-1] / plat_base - 1
    return float(gold_ret - plat_ret)


def compute_gold_platinum_ratio_risk_signal(
    df: pd.DataFrame, context: dict | None = None,
) -> dict:
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    if df is None or len(df) < 10:
        return empty

    context = context or {}
    ticker = context.get("ticker", "")
    asset_class = context.get("asset_class", "")
    is_metals = ticker in _METALS_TICKERS or asset_class == "metals"

    combined = _fetch_gp_data()
    if combined is None or len(combined) < MIN_ROWS:
        return empty

    gp = combined["gp_ratio"]

    # Sub-signal 1: GP z-score — CONTRARIAN at extremes only
    # High GP = extreme fear = contrarian BUY for risk assets
    # This is a slow-moving signal (monthly horizon) — only fire at extremes
    zscore = _gp_zscore(gp, window=252)
    if zscore > 1.5:
        zscore_vote = "BUY"
    elif zscore < -1.5:
        zscore_vote = "SELL"
    else:
        zscore_vote = "HOLD"

    # Sub-signal 2: GP trend — structural regime (loose filter)
    trend = _gp_trend(gp)
    if trend == 1:
        trend_vote = "BUY"
    elif trend == -1:
        trend_vote = "SELL"
    else:
        trend_vote = "HOLD"

    # Sub-signal 3: GP momentum — rapid fear shift (wide threshold)
    momentum = _gp_momentum(gp, periods=21)
    if momentum > 0.04:
        momentum_vote = "BUY"
    elif momentum < -0.04:
        momentum_vote = "SELL"
    else:
        momentum_vote = "HOLD"

    # Sub-signal 4: Gold-platinum return spread (wide threshold)
    spread = _gold_plat_spread(combined, periods=20)
    if spread > 0.04:
        spread_vote = "BUY"
    elif spread < -0.04:
        spread_vote = "SELL"
    else:
        spread_vote = "HOLD"

    votes = [zscore_vote, trend_vote, momentum_vote, spread_vote]

    # Require 3+ sub-signals to agree — this is a slow-moving macro signal,
    # only fire at genuine extremes where multiple indicators confirm
    active_votes = [v for v in votes if v != "HOLD"]
    if len(active_votes) < 3:
        action, confidence = "HOLD", 0.0
    else:
        action, confidence = majority_vote(votes, count_hold=True)

    # Invert for metals: high GP = fear priced in = SELL metals
    if is_metals and action != "HOLD":
        action = "SELL" if action == "BUY" else "BUY"

    confidence = min(confidence, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "gp_zscore": zscore_vote,
            "gp_trend": trend_vote,
            "gp_momentum": momentum_vote,
            "gold_plat_spread": spread_vote,
        },
        "indicators": {
            "gp_ratio": safe_float(gp.iloc[-1]),
            "gp_zscore": round(zscore, 4),
            "gp_trend": trend,
            "gp_momentum_21d": round(momentum, 4),
            "gold_plat_spread_20d": round(spread, 4),
            "gold_price": safe_float(combined["gold"].iloc[-1]),
            "platinum_price": safe_float(combined["platinum"].iloc[-1]),
        },
    }
