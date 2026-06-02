"""Crypto overnight sentiment signal module.

BTC and ETH returns during US equity market closure (~16:00-09:30 ET)
serve as a cross-asset sentiment proxy. Positive overnight returns
predict lower next-day VIX (risk-on). Negative predict higher VIX
(risk-off).

Three sub-signals:
  1. Overnight Return Z-Score  -- z-score of avg BTC+ETH overnight return
  2. Overnight Momentum        -- direction persistence (3 consecutive days)
  3. Intraday Divergence       -- overnight vs prior intraday return sign

Applies to ALL tickers. For safe-haven (XAU/XAG): inverted direction.
For risk-on (BTC/ETH/MSTR): direct direction.

Source: Gu, Liu, Lin (2026) "Beyond Conventional Sentiment Indicators:
Cryptocurrency's Hidden Potential in VIX Forecasting", Economic
Modelling 161. DOI: 10.1016/j.econmod.2026.107648
"""
from __future__ import annotations

import datetime
import logging
import threading
import time

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger(__name__)

MIN_ROWS = 20

_SAFE_HAVEN = {"XAU-USD", "XAG-USD"}

_US_CLOSE_UTC_SUMMER = 20
_US_CLOSE_UTC_WINTER = 21
_US_OPEN_UTC_SUMMER = 13
_US_OPEN_UTC_WINTER = 14

_Z_LOOKBACK = 30
_Z_BUY = 1.5
_Z_SELL = -1.5
_MOMENTUM_DAYS = 3

_CACHE_TTL = 300
_cache: dict = {}
_cache_lock = threading.Lock()


def _is_us_dst(dt: datetime.datetime) -> bool:
    """Check if date falls in US daylight saving time (EDT)."""
    year = dt.year
    mar_second_sun = 8 + (6 - datetime.date(year, 3, 8).weekday()) % 7
    nov_first_sun = 1 + (6 - datetime.date(year, 11, 1).weekday()) % 7
    dst_start = datetime.date(year, 3, mar_second_sun)
    dst_end = datetime.date(year, 11, nov_first_sun)
    return dst_start <= dt.date() <= dst_end


def _get_market_hours_utc() -> tuple[int, int]:
    """Return (close_hour_utc, open_hour_utc) based on current DST."""
    now = datetime.datetime.now(datetime.timezone.utc)
    if _is_us_dst(now):
        return _US_CLOSE_UTC_SUMMER, _US_OPEN_UTC_SUMMER
    return _US_CLOSE_UTC_WINTER, _US_OPEN_UTC_WINTER


def _fetch_hourly_data(ticker: str, limit: int = 800) -> pd.DataFrame | None:
    """Fetch 1h candles for a crypto ticker."""
    cache_key = f"overnight_{ticker}"
    with _cache_lock:
        if cache_key in _cache:
            ts, data = _cache[cache_key]
            if time.time() - ts < _CACHE_TTL:
                return data

    try:
        from portfolio.price_source import fetch_klines
        df = fetch_klines(ticker, interval="1h", limit=limit)
        if df is not None and len(df) > 0:
            with _cache_lock:
                _cache[cache_key] = (time.time(), df)
            return df
    except Exception:
        logger.debug("crypto_overnight_sentiment: fetch failed for %s", ticker, exc_info=True)
    return None


def _compute_overnight_returns(df_1h: pd.DataFrame, close_hour: int,
                                open_hour: int, n_days: int = 35) -> list[float]:
    """Extract overnight returns from hourly data.

    For each trading day, find the candle at close_hour and the candle
    at open_hour the next morning. Overnight return = open/close - 1.
    """
    if df_1h is None or len(df_1h) < 24:
        return []

    close_col = df_1h["close"]
    if not hasattr(df_1h.index, "hour"):
        try:
            df_1h = df_1h.copy()
            df_1h.index = pd.to_datetime(df_1h.index, utc=True)
        except Exception:
            return []

    # Use ±1h window to handle DST transitions robustly
    close_hours = {close_hour, (close_hour + 1) % 24}
    open_hours = {open_hour, (open_hour + 1) % 24}
    close_prices = df_1h[df_1h.index.hour.isin(close_hours)]["close"]
    open_prices = df_1h[df_1h.index.hour.isin(open_hours)]["close"]

    if len(close_prices) < 3 or len(open_prices) < 3:
        return []

    returns = []
    for i in range(min(len(close_prices) - 1, n_days)):
        close_ts = close_prices.index[-(i + 2)]
        close_date = close_ts.date()
        next_date = close_date + datetime.timedelta(days=1)

        next_opens = open_prices[open_prices.index.date == next_date]
        if len(next_opens) == 0:
            next_opens = open_prices[
                (open_prices.index > close_ts)
                & (open_prices.index < close_ts + datetime.timedelta(hours=24))
            ]
        if len(next_opens) == 0:
            continue

        close_val = float(close_prices.iloc[-(i + 2)])
        open_val = float(next_opens.iloc[0])
        if close_val > 0:
            returns.append((open_val - close_val) / close_val)

    returns.reverse()
    return returns


def _compute_intraday_returns(df_1h: pd.DataFrame, open_hour: int,
                               close_hour: int, n_days: int = 5) -> list[float]:
    """Compute intraday returns (open to close) for divergence sub-signal."""
    if df_1h is None or len(df_1h) < 24:
        return []

    if not hasattr(df_1h.index, "hour"):
        try:
            df_1h = df_1h.copy()
            df_1h.index = pd.to_datetime(df_1h.index, utc=True)
        except Exception:
            return []

    open_prices = df_1h[df_1h.index.hour == open_hour]["close"]
    close_prices = df_1h[df_1h.index.hour == close_hour]["close"]

    if len(open_prices) < 3 or len(close_prices) < 3:
        return []

    returns = []
    for i in range(min(len(close_prices) - 1, n_days)):
        close_ts = close_prices.index[-(i + 1)]
        close_date = close_ts.date()
        day_opens = open_prices[open_prices.index.date == close_date]
        if len(day_opens) == 0:
            continue
        o = float(day_opens.iloc[0])
        c = float(close_prices.iloc[-(i + 1)])
        if o > 0:
            returns.append((c - o) / o)

    returns.reverse()
    return returns


def compute_crypto_overnight_sentiment_signal(
    df: pd.DataFrame, context: dict | None = None,
) -> dict:
    """Compute crypto overnight sentiment signal.

    Args:
        df: OHLCV DataFrame (used for row count gate).
        context: Optional dict with ticker, asset_class, regime.

    Returns:
        dict with action, confidence, sub_signals, indicators.
    """
    empty = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    if df is None or len(df) < MIN_ROWS:
        return empty

    context = context or {}
    ticker = context.get("ticker", "")
    asset_class = context.get("asset_class", "")
    is_safe_haven = ticker in _SAFE_HAVEN or asset_class == "metals"

    close_hour, open_hour = _get_market_hours_utc()

    btc_1h = _fetch_hourly_data("BTC-USD")
    eth_1h = _fetch_hourly_data("ETH-USD")

    if btc_1h is None and eth_1h is None:
        return empty

    btc_rets = _compute_overnight_returns(btc_1h, close_hour, open_hour)
    eth_rets = _compute_overnight_returns(eth_1h, close_hour, open_hour)

    if len(btc_rets) < 5 and len(eth_rets) < 5:
        return empty

    n = min(len(btc_rets), len(eth_rets)) if btc_rets and eth_rets else max(len(btc_rets), len(eth_rets))
    if btc_rets and eth_rets:
        n = min(len(btc_rets), len(eth_rets))
        avg_rets = [(btc_rets[-(n - i)] + eth_rets[-(n - i)]) / 2 for i in range(n)]
    elif btc_rets:
        avg_rets = btc_rets
    else:
        avg_rets = eth_rets

    if len(avg_rets) < 5:
        return empty

    arr = np.array(avg_rets)
    lookback = min(_Z_LOOKBACK, len(arr))
    recent = arr[-lookback:]
    mean_r = np.mean(recent)
    std_r = np.std(recent)
    if std_r < 1e-10:
        return empty

    latest = arr[-1]
    z = (latest - mean_r) / std_r

    # --- Sub-signal 1: Z-Score ---
    if z > _Z_BUY:
        zscore_vote = "BUY"
    elif z < _Z_SELL:
        zscore_vote = "SELL"
    else:
        zscore_vote = "HOLD"

    # --- Sub-signal 2: Momentum (consecutive direction) ---
    if len(arr) >= _MOMENTUM_DAYS:
        last_n = arr[-_MOMENTUM_DAYS:]
        if all(r > 0 for r in last_n):
            momentum_vote = "BUY"
        elif all(r < 0 for r in last_n):
            momentum_vote = "SELL"
        else:
            momentum_vote = "HOLD"
    else:
        momentum_vote = "HOLD"

    # --- Sub-signal 3: Intraday divergence (averaged BTC+ETH) ---
    btc_intra = _compute_intraday_returns(btc_1h, open_hour, close_hour)
    eth_intra = _compute_intraday_returns(eth_1h, open_hour, close_hour)
    divergence_vote = "HOLD"
    intra_vals = [v for v in [
        btc_intra[-1] if btc_intra else None,
        eth_intra[-1] if eth_intra else None,
    ] if v is not None]
    if intra_vals and avg_rets:
        last_overnight = avg_rets[-1]
        last_intraday = sum(intra_vals) / len(intra_vals)
        if last_overnight > 0 and last_intraday < 0:
            divergence_vote = "BUY"
        elif last_overnight < 0 and last_intraday > 0:
            divergence_vote = "SELL"

    votes = [zscore_vote, momentum_vote, divergence_vote]
    action, confidence = majority_vote(votes, count_hold=False)

    if is_safe_haven and action != "HOLD":
        action = "SELL" if action == "BUY" else "BUY"

    confidence = min(confidence, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "overnight_zscore": zscore_vote,
            "overnight_momentum": momentum_vote,
            "intraday_divergence": divergence_vote,
        },
        "indicators": {
            "latest_overnight_ret": safe_float(latest),
            "overnight_zscore": safe_float(z),
            "overnight_mean_30d": safe_float(mean_r),
            "overnight_std_30d": safe_float(std_r),
            "btc_overnight_rets": len(btc_rets),
            "eth_overnight_rets": len(eth_rets),
        },
    }
