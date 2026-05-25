"""BTC-Gold correlation regime signal — intermarket correlation z-score.

Computes 30-day rolling Pearson correlation between BTC and Gold daily
returns, then z-scores against 252-day history. Extreme negative
correlation (z < -2.0) historically precedes BTC rallies. High positive
correlation (z > 1.5) precedes mean-reversion.

For XAU-USD: signals are inverted (BUY when z > 1.5, SELL when z < -2.0).

Academic basis: Mudrex BTC-Gold correlation 2026; arxiv 2512.12815.
Composite score 8.05/10.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import safe_float

logger = logging.getLogger("portfolio.signals.btc_gold_correlation_regime")

_CACHE_TTL = 14400  # 4 hours

_CORR_WINDOW = 30
_Z_LOOKBACK = 252
_BUY_Z = -2.0
_SELL_Z = 1.5
_MIN_ROWS = _Z_LOOKBACK + _CORR_WINDOW

_STALE_RATIO_THRESHOLD = 0.25

_COUNTERPART = {
    "BTC-USD": ("binance_fapi", "XAUUSDT"),
    "XAU-USD": ("binance", "BTCUSDT"),
}

_INVERT_TICKERS = {"XAU-USD"}


def _fetch_counterpart(source: str, symbol: str) -> pd.DataFrame | None:
    """Fetch daily closes for the counterpart asset via Binance."""
    def _do_fetch():
        try:
            from portfolio.data_collector import (
                binance_fapi_klines,
                binance_klines,
            )
            fetcher = binance_fapi_klines if source == "binance_fapi" else binance_klines
            df = fetcher(symbol, interval="1d", limit=400)
            if df is None or df.empty:
                return None
            return df
        except Exception as exc:
            logger.warning("btc_gold_corr: fetch %s failed: %s", symbol, exc)
            return None

    return _cached(f"btc_gold_corr_{symbol}", _CACHE_TTL, _do_fetch)


def compute_btc_gold_correlation_regime_signal(
    df: pd.DataFrame, context: dict | None = None,
) -> dict:
    hold = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {},
        "indicators": {},
    }

    if df is None or len(df) < _CORR_WINDOW:
        return hold

    ticker = (context or {}).get("ticker", "")
    if ticker not in _COUNTERPART:
        return hold

    source, symbol = _COUNTERPART[ticker]
    counter_df = _fetch_counterpart(source, symbol)
    if counter_df is None or len(counter_df) < _MIN_ROWS:
        return hold

    target_close = df["close"].copy()
    counter_close = counter_df["close"].copy()

    if hasattr(target_close.index, "tz"):
        target_close.index = target_close.index.tz_localize(None) if target_close.index.tz is None else target_close.index.tz_convert(None)
    if hasattr(counter_close.index, "tz"):
        counter_close.index = counter_close.index.tz_localize(None) if counter_close.index.tz is None else counter_close.index.tz_convert(None)

    merged = pd.DataFrame({
        "target": target_close,
        "counter": counter_close,
    }).dropna()

    if len(merged) < _MIN_ROWS:
        return hold

    target_ret = merged["target"].pct_change().dropna()
    counter_ret = merged["counter"].pct_change().dropna()

    if len(target_ret) < _MIN_ROWS:
        return hold

    zero_count = (counter_ret.iloc[-_CORR_WINDOW:].abs() < 1e-10).sum()
    stale_ratio = zero_count / _CORR_WINDOW
    if stale_ratio > _STALE_RATIO_THRESHOLD:
        hold["indicators"] = {"stale_data_ratio": safe_float(stale_ratio)}
        return hold

    rolling_corr = target_ret.rolling(window=_CORR_WINDOW).corr(counter_ret)
    rolling_corr = rolling_corr.dropna()

    if len(rolling_corr) < _Z_LOOKBACK:
        return hold

    mean_corr = rolling_corr.rolling(window=_Z_LOOKBACK).mean()
    std_corr = rolling_corr.rolling(window=_Z_LOOKBACK).std()

    latest_corr = rolling_corr.iloc[-1]
    latest_mean = mean_corr.iloc[-1]
    latest_std = std_corr.iloc[-1]

    if not np.isfinite(latest_std) or latest_std < 1e-8:
        return hold

    z_score = (latest_corr - latest_mean) / latest_std

    if not np.isfinite(z_score):
        return hold

    invert = ticker in _INVERT_TICKERS

    if invert:
        buy_z, sell_z = _SELL_Z, _BUY_Z
    else:
        buy_z, sell_z = _BUY_Z, _SELL_Z

    if (not invert and z_score < buy_z) or (invert and z_score > buy_z):
        action = "BUY"
    elif (not invert and z_score > sell_z) or (invert and z_score < sell_z):
        action = "SELL"
    else:
        action = "HOLD"

    z_magnitude = abs(z_score)
    confidence = min(0.7, 0.3 + 0.1 * (z_magnitude - 1.5))
    confidence = max(0.0, confidence)

    if action == "HOLD":
        confidence = 0.0

    return {
        "action": action,
        "confidence": safe_float(confidence),
        "sub_signals": {
            "correlation_z": f"{z_score:.2f} ({'inverted' if invert else 'normal'})",
        },
        "indicators": {
            "corr_30d": safe_float(latest_corr),
            "corr_z_score": safe_float(z_score),
            "corr_mean_252d": safe_float(latest_mean),
            "corr_std_252d": safe_float(latest_std),
            "stale_data_ratio": safe_float(stale_ratio),
        },
    }
