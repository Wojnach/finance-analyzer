"""Stablecoin Supply Ratio signal module.

Measures crypto buying power via the ratio of crypto market cap to
stablecoin supply on Ethereum. Low SSR = high stablecoin liquidity
relative to crypto cap = potential buying pressure. High SSR = drained
liquidity = bearish.

Sub-indicators:
    1. SSR Level      -- z-score of current SSR vs 90-day distribution
    2. Supply Momentum -- 7-day rate-of-change of stablecoin supply
    3. Supply-Price Divergence -- supply growing while price falling = BUY

Data: DefiLlama stablecoin API (free, no auth, daily resolution).
Applicable: BTC-USD, ETH-USD only.

Source: CryptoQuant SSR metric, Glassnode SSR oscillator.
"""
from __future__ import annotations

import logging
import threading
import time

import numpy as np
import pandas as pd

from portfolio.http_retry import fetch_json
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.stablecoin_supply_ratio")

MIN_ROWS = 30
MAX_CONFIDENCE = 0.7
Z_BUY = -1.5
Z_SELL = 1.5
LOOKBACK = 90
MOMENTUM_WINDOW = 7
MOMENTUM_BUY = 0.01
MOMENTUM_SELL = -0.01

_APPLICABLE_TICKERS = {"BTC-USD", "BTCUSDT", "ETH-USD", "ETHUSDT"}

_DEFILLAMA_URL = "https://stablecoins.llama.fi/stablecoincharts/Ethereum"
_CACHE_TTL = 4 * 3600
_cache: dict = {}
_cache_lock = threading.Lock()


def _fetch_stablecoin_supply() -> pd.Series | None:
    """Fetch historical daily stablecoin supply on Ethereum from DefiLlama.

    Returns a pandas Series indexed by date with total stablecoin supply (USD).
    Cached for 4 hours.
    """
    now = time.time()
    with _cache_lock:
        if _cache.get("data") is not None and now - _cache.get("ts", 0) < _CACHE_TTL:
            return _cache["data"]

    try:
        data = fetch_json(_DEFILLAMA_URL, timeout=20)
        if not data or not isinstance(data, list):
            logger.warning("DefiLlama stablecoin API returned empty/invalid data")
            return None

        dates = []
        supplies = []
        for entry in data:
            ts = entry.get("date")
            circ = entry.get("totalCirculatingUSD", {}).get("peggedUSD")
            if ts is None or circ is None:
                continue
            dates.append(pd.Timestamp.utcfromtimestamp(int(ts)))
            supplies.append(float(circ))

        if len(dates) < LOOKBACK:
            logger.warning("Insufficient stablecoin history: %d points", len(dates))
            return None

        series = pd.Series(supplies, index=pd.DatetimeIndex(dates), name="stablecoin_supply")
        series = series.sort_index()

        with _cache_lock:
            _cache["data"] = series
            _cache["ts"] = now

        return series
    except Exception:
        logger.exception("Failed to fetch stablecoin supply from DefiLlama")
        return None


def _ssr_level(ssr_series: pd.Series, lookback: int = LOOKBACK) -> tuple[float, str]:
    """Z-score of current SSR vs trailing window. Low z = buying power = BUY."""
    if len(ssr_series) < lookback + 1:
        return float("nan"), "HOLD"

    window = ssr_series.iloc[-(lookback + 1) : -1]
    current = ssr_series.iloc[-1]
    mean = window.mean()
    std = window.std()
    if std == 0 or np.isnan(std):
        return float("nan"), "HOLD"

    z = (current - mean) / std
    z_val = safe_float(z)
    if np.isnan(z_val):
        return float("nan"), "HOLD"

    if z_val < Z_BUY:
        return z_val, "BUY"
    if z_val > Z_SELL:
        return z_val, "SELL"
    return z_val, "HOLD"


def _supply_momentum(supply_series: pd.Series, window: int = MOMENTUM_WINDOW) -> tuple[float, str]:
    """Rate-of-change of stablecoin supply. Growing supply = buying power building."""
    if len(supply_series) < window + 1:
        return float("nan"), "HOLD"

    current = supply_series.iloc[-1]
    past = supply_series.iloc[-(window + 1)]
    if past == 0 or np.isnan(past):
        return float("nan"), "HOLD"

    roc = (current - past) / past
    roc_val = safe_float(roc)
    if np.isnan(roc_val):
        return float("nan"), "HOLD"

    if roc_val > MOMENTUM_BUY:
        return roc_val, "BUY"
    if roc_val < MOMENTUM_SELL:
        return roc_val, "SELL"
    return roc_val, "HOLD"


def _supply_price_divergence(
    supply_series: pd.Series, price_series: pd.Series, window: int = MOMENTUM_WINDOW
) -> tuple[float, str]:
    """Detect divergence between stablecoin supply growth and price movement.

    Supply growing while price falling = accumulation (BUY).
    Supply shrinking while price rising = distribution (SELL).
    """
    if len(supply_series) < window + 1 or len(price_series) < window + 1:
        return float("nan"), "HOLD"

    supply_roc = (supply_series.iloc[-1] - supply_series.iloc[-(window + 1)]) / supply_series.iloc[
        -(window + 1)
    ]
    price_roc = (price_series.iloc[-1] - price_series.iloc[-(window + 1)]) / price_series.iloc[
        -(window + 1)
    ]

    supply_roc_val = safe_float(supply_roc)
    price_roc_val = safe_float(price_roc)
    if np.isnan(supply_roc_val) or np.isnan(price_roc_val):
        return float("nan"), "HOLD"

    divergence = supply_roc_val - price_roc_val

    if supply_roc_val > 0.005 and price_roc_val < -0.01:
        return divergence, "BUY"
    if supply_roc_val < -0.005 and price_roc_val > 0.01:
        return divergence, "SELL"
    return divergence, "HOLD"


def compute_stablecoin_supply_ratio_signal(
    df: pd.DataFrame, context: dict = None
) -> dict:
    """Compute stablecoin supply ratio signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        context: Optional dict with keys: ticker, config, asset_class, regime

    Returns:
        dict with keys: action, confidence, sub_signals, indicators
    """
    hold = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    if df is None or len(df) < MIN_ROWS:
        return hold

    ticker = (context or {}).get("ticker", "")
    if ticker and ticker not in _APPLICABLE_TICKERS:
        return hold

    supply_series = _fetch_stablecoin_supply()
    if supply_series is None or len(supply_series) < LOOKBACK + MOMENTUM_WINDOW:
        return hold

    close = df["close"].dropna()
    if len(close) < MIN_ROWS:
        return hold

    current_price = safe_float(close.iloc[-1])
    if np.isnan(current_price) or current_price <= 0:
        return hold

    supply_recent = supply_series.iloc[-(LOOKBACK + MOMENTUM_WINDOW + 10) :]
    if len(supply_recent) < LOOKBACK:
        return hold

    approx_circulating = 120_000_000 if "ETH" in ticker else 21_000_000
    ssr_values = []
    for i in range(len(supply_recent)):
        s = supply_recent.iloc[i]
        if s > 0:
            ssr_values.append(current_price * approx_circulating / s)
        else:
            ssr_values.append(float("nan"))
    ssr_series = pd.Series(ssr_values, index=supply_recent.index)

    ssr_z, ssr_vote = _ssr_level(ssr_series, LOOKBACK)
    mom_val, mom_vote = _supply_momentum(supply_recent, MOMENTUM_WINDOW)

    price_for_div = close.iloc[-MOMENTUM_WINDOW - 1 :] if len(close) > MOMENTUM_WINDOW else close
    supply_for_div = supply_recent.iloc[-MOMENTUM_WINDOW - 1 :]
    div_val, div_vote = _supply_price_divergence(supply_for_div, price_for_div, MOMENTUM_WINDOW)

    votes = [ssr_vote, mom_vote, div_vote]
    action, confidence = majority_vote(votes, count_hold=False)

    confidence = min(confidence, MAX_CONFIDENCE)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "ssr_level": ssr_vote,
            "supply_momentum": mom_vote,
            "price_supply_divergence": div_vote,
        },
        "indicators": {
            "ssr_zscore": safe_float(ssr_z),
            "supply_7d_roc": safe_float(mom_val),
            "divergence": safe_float(div_val),
            "current_supply_usd": safe_float(supply_recent.iloc[-1]) if len(supply_recent) > 0 else 0.0,
        },
    }
