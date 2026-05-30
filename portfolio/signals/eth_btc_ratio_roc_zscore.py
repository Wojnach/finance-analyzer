"""ETH/BTC ratio rate-of-change z-score signal.

Measures relative momentum between ETH and BTC. A rising ETH/BTC ratio
signals risk appetite broadening (altseason). Falling ratio signals
flight to BTC quality. Works on both ETH-USD and BTC-USD with inverted
logic.

Sub-signals:
  1. ratio_roc_zscore — z-score of 20d ROC of ETH/BTC ratio
  2. ratio_vs_sma     — ratio position relative to 50-day SMA
  3. roc_acceleration  — second derivative (ROC of ROC) for momentum confirmation

Crypto-only (BTC-USD, ETH-USD). Non-crypto -> immediate HOLD.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.eth_btc_ratio_roc_zscore")

MIN_ROWS = 60
_MAX_CONFIDENCE = 0.7

_ROC_PERIOD = 20
_SMA_PERIOD = 50
_Z_LOOKBACK = 120
_Z_BUY_THRESHOLD = 1.5
_Z_SELL_THRESHOLD = -1.5
_SMA_MARGIN = 0.02
_ACCEL_THRESHOLD = 0.5

_CRYPTO_TICKERS = {"BTC-USD", "ETH-USD"}
_COUNTERPARTY = {"ETH-USD": "ETHUSDT", "BTC-USD": "BTCUSDT"}


def _fetch_counterparty_close(ticker: str) -> pd.Series | None:
    """Fetch counterparty close prices from Binance (cached by main loop)."""
    try:
        from portfolio.data_collector import binance_klines
    except ImportError:
        return None

    counter_symbol = "BTCUSDT" if ticker == "ETH-USD" else "ETHUSDT"
    try:
        df = binance_klines(counter_symbol, interval="5m", limit=200)
        if df is not None and len(df) >= MIN_ROWS and "close" in df.columns:
            return df["close"].astype(float)
    except Exception:
        logger.debug("Failed to fetch counterparty %s", counter_symbol, exc_info=True)
    return None


def _compute_ratio(df: pd.DataFrame, context: dict) -> pd.Series | None:
    """Compute ETH/BTC ratio from OHLCV data + counterparty fetch."""
    ticker = context.get("ticker", "")
    if ticker not in _CRYPTO_TICKERS:
        return None

    # Try context first (test harness), then live fetch
    all_prices = context.get("all_prices", {})
    counter_key = "BTC-USD" if ticker == "ETH-USD" else "ETH-USD"
    counter_df = all_prices.get(counter_key)

    if isinstance(counter_df, pd.DataFrame) and len(counter_df) >= MIN_ROWS:
        counter_close = counter_df["close"].astype(float)
    else:
        counter_close = _fetch_counterparty_close(ticker)

    if counter_close is None or len(counter_close) < MIN_ROWS:
        return None

    my_close = df["close"].astype(float)

    if ticker == "ETH-USD":
        eth_c, btc_c = my_close, counter_close
    else:
        btc_c, eth_c = my_close, counter_close

    min_len = min(len(eth_c), len(btc_c))
    if min_len < MIN_ROWS:
        return None

    eth_c = eth_c.iloc[-min_len:].reset_index(drop=True)
    btc_c = btc_c.iloc[-min_len:].reset_index(drop=True)

    btc_c = btc_c.replace(0, np.nan)
    ratio = eth_c / btc_c
    ratio = ratio.dropna()

    if len(ratio) < MIN_ROWS:
        return None
    return ratio


def _ratio_roc_zscore_vote(ratio: pd.Series, ticker: str) -> tuple[str, dict]:
    roc_vals = ratio.pct_change(_ROC_PERIOD).dropna()
    if len(roc_vals) < 10:
        return "HOLD", {}

    recent = roc_vals.iloc[-min(len(roc_vals), _Z_LOOKBACK):]
    mean_val = float(recent.mean())
    std_val = float(recent.std(ddof=0))

    if math.isnan(std_val) or std_val < 1e-10:
        return "HOLD", {"ratio_roc": float(roc_vals.iloc[-1]), "ratio_roc_z": 0.0}

    latest_roc = float(roc_vals.iloc[-1])
    z = (latest_roc - mean_val) / std_val

    indicators = {"ratio_roc": latest_roc, "ratio_roc_z": z}

    if ticker == "ETH-USD":
        if z > _Z_BUY_THRESHOLD:
            return "BUY", indicators
        if z < _Z_SELL_THRESHOLD:
            return "SELL", indicators
    elif ticker == "BTC-USD":
        if z > _Z_BUY_THRESHOLD:
            return "SELL", indicators
        if z < _Z_SELL_THRESHOLD:
            return "BUY", indicators

    return "HOLD", indicators


def _ratio_vs_sma_vote(ratio: pd.Series, ticker: str) -> tuple[str, dict]:
    if len(ratio) < _SMA_PERIOD:
        return "HOLD", {}

    sma = float(ratio.iloc[-_SMA_PERIOD:].mean())
    current = float(ratio.iloc[-1])

    if sma <= 0:
        return "HOLD", {}

    pct_above = (current - sma) / sma
    indicators = {"ratio_vs_sma_pct": pct_above, "ratio_sma50": sma, "ratio_current": current}

    if ticker == "ETH-USD":
        if pct_above > _SMA_MARGIN:
            return "BUY", indicators
        if pct_above < -_SMA_MARGIN:
            return "SELL", indicators
    elif ticker == "BTC-USD":
        if pct_above > _SMA_MARGIN:
            return "SELL", indicators
        if pct_above < -_SMA_MARGIN:
            return "BUY", indicators

    return "HOLD", indicators


def _roc_acceleration_vote(ratio: pd.Series, ticker: str) -> tuple[str, dict]:
    if len(ratio) < _ROC_PERIOD * 2 + 5:
        return "HOLD", {}

    roc = ratio.pct_change(_ROC_PERIOD).dropna()
    if len(roc) < _ROC_PERIOD + 5:
        return "HOLD", {}

    roc_of_roc = roc.diff(_ROC_PERIOD).dropna()
    if len(roc_of_roc) < 5:
        return "HOLD", {}

    recent = roc_of_roc.iloc[-min(len(roc_of_roc), _Z_LOOKBACK):]
    mean_val = float(recent.mean())
    std_val = float(recent.std(ddof=0))

    if math.isnan(std_val) or std_val < 1e-10:
        return "HOLD", {"roc_accel": float(roc_of_roc.iloc[-1]), "roc_accel_z": 0.0}

    latest = float(roc_of_roc.iloc[-1])
    z = (latest - mean_val) / std_val

    indicators = {"roc_accel": latest, "roc_accel_z": z}

    if ticker == "ETH-USD":
        if z > _ACCEL_THRESHOLD:
            return "BUY", indicators
        if z < -_ACCEL_THRESHOLD:
            return "SELL", indicators
    elif ticker == "BTC-USD":
        if z > _ACCEL_THRESHOLD:
            return "SELL", indicators
        if z < -_ACCEL_THRESHOLD:
            return "BUY", indicators

    return "HOLD", indicators


def compute_eth_btc_ratio_roc_zscore_signal(
    df: pd.DataFrame, context: dict = None
) -> dict:
    empty = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    if context is None:
        return empty

    ticker = context.get("ticker", "")
    if ticker not in _CRYPTO_TICKERS:
        return empty

    if df is None or len(df) < MIN_ROWS:
        return empty

    try:
        ratio = _compute_ratio(df, context)
    except Exception:
        logger.debug("Failed to compute ETH/BTC ratio", exc_info=True)
        return empty

    if ratio is None or len(ratio) < MIN_ROWS:
        return empty

    sub_signals = {}
    indicators = {}

    for label, func in [
        ("ratio_roc_zscore", _ratio_roc_zscore_vote),
        ("ratio_vs_sma", _ratio_vs_sma_vote),
        ("roc_acceleration", _roc_acceleration_vote),
    ]:
        try:
            vote, ind = func(ratio, ticker)
            sub_signals[label] = vote
            indicators.update(ind)
        except Exception:
            logger.debug("%s sub-signal failed", label, exc_info=True)
            sub_signals[label] = "HOLD"

    votes = list(sub_signals.values())
    action, confidence = majority_vote(votes, count_hold=False)
    confidence = min(confidence, _MAX_CONFIDENCE)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": {k: safe_float(v) for k, v in indicators.items()},
    }
