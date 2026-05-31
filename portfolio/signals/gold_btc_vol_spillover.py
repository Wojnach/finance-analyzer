"""Gold-to-Bitcoin volatility spillover signal.

Detects one-way volatility transmission from gold to BTC. Academic research
shows gold volatility spikes precede BTC volatility expansion within 1-3
days. When gold vol z-scores above 2.0, signal direction based on BTC
trend context at time of spike.

Sub-signals:
    1. Gold Vol Z-Score: gold realized vol vs 60-day distribution
    2. Gold Vol Momentum: vol acceleration (rising vol = stronger spillover)
    3. BTC Trend Alignment: SMA crossover for directional bias

Applicable to BTC-USD only. Confidence capped at 0.7 per external-data
signal convention.

Source: Dynamic Stochastic Volatility Spillover Between Bitcoin and
Precious Metals (2025). Composite score 7.5/10.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import safe_float, sma

logger = logging.getLogger("portfolio.signals.gold_btc_vol_spillover")

_CACHE_TTL = 14400  # 4 hours — gold vol doesn't change rapidly
_VOL_LOOKBACK = 20
_Z_LOOKBACK = 60
_SPIKE_THRESHOLD = 2.0
_STRONG_SPIKE = 3.0
_VOL_MOMENTUM_WINDOW = 5
_TREND_FAST = 20
_TREND_SLOW = 50
_MAX_CONFIDENCE = 0.7
_MIN_ROWS = max(_TREND_SLOW, _Z_LOOKBACK + _VOL_LOOKBACK) + 10

_APPLICABLE_TICKERS = {"BTC-USD"}


def _fetch_gold_data() -> pd.DataFrame | None:
    """Fetch daily gold price data via Binance FAPI."""
    def _do_fetch():
        try:
            from portfolio.data_collector import binance_fapi_klines
            df = binance_fapi_klines("XAUUSDT", interval="1d", limit=200)
            if df is None or df.empty:
                return None
            return df
        except Exception as exc:
            logger.warning("gold_btc_vol_spillover: gold fetch failed: %s", exc)
            return None

    return _cached("gold_btc_vol_spillover_xau", _CACHE_TTL, _do_fetch)


def compute_gold_btc_vol_spillover_signal(
    df: pd.DataFrame, context: dict | None = None,
) -> dict:
    """Compute gold-to-BTC volatility spillover signal."""
    hold = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {},
        "indicators": {},
    }

    if df is None or len(df) < _MIN_ROWS:
        return hold

    ticker = (context or {}).get("ticker", "")
    if ticker and ticker not in _APPLICABLE_TICKERS:
        return hold

    gold_df = _fetch_gold_data()
    if gold_df is None or len(gold_df) < _Z_LOOKBACK + _VOL_LOOKBACK:
        return hold

    gold_close = gold_df["close"].astype(float)
    gold_returns = np.log(gold_close / gold_close.shift(1)).dropna()

    if len(gold_returns) < _Z_LOOKBACK + _VOL_LOOKBACK:
        return hold

    gold_vol = gold_returns.rolling(window=_VOL_LOOKBACK).std()
    gold_vol = gold_vol.dropna()

    if len(gold_vol) < _Z_LOOKBACK:
        return hold

    vol_mean = gold_vol.rolling(window=_Z_LOOKBACK).mean()
    vol_std = gold_vol.rolling(window=_Z_LOOKBACK).std()

    latest_vol = gold_vol.iloc[-1]
    latest_mean = vol_mean.iloc[-1]
    latest_std = vol_std.iloc[-1]

    if not np.isfinite(latest_std) or latest_std < 1e-10:
        return hold

    gold_vol_z = (latest_vol - latest_mean) / latest_std

    if not np.isfinite(gold_vol_z):
        return hold

    vol_momentum = 0.0
    if len(gold_vol) >= _VOL_MOMENTUM_WINDOW + 1:
        recent_vol = gold_vol.iloc[-_VOL_MOMENTUM_WINDOW:]
        prior_vol = gold_vol.iloc[-2 * _VOL_MOMENTUM_WINDOW:-_VOL_MOMENTUM_WINDOW]
        if len(prior_vol) >= _VOL_MOMENTUM_WINDOW:
            vol_momentum = (recent_vol.mean() - prior_vol.mean()) / (prior_vol.mean() + 1e-10)

    btc_close = df["close"].astype(float)
    if len(btc_close) < _TREND_SLOW:
        return hold

    btc_sma_fast = sma(btc_close, _TREND_FAST)
    btc_sma_slow = sma(btc_close, _TREND_SLOW)

    if btc_sma_fast is None or btc_sma_slow is None:
        return hold

    latest_fast = float(btc_sma_fast.iloc[-1]) if hasattr(btc_sma_fast, "iloc") else float(btc_sma_fast)
    latest_slow = float(btc_sma_slow.iloc[-1]) if hasattr(btc_sma_slow, "iloc") else float(btc_sma_slow)

    if not np.isfinite(latest_fast) or not np.isfinite(latest_slow) or latest_slow == 0:
        return hold

    btc_trend = 1 if latest_fast > latest_slow else -1
    trend_strength = abs(latest_fast - latest_slow) / latest_slow

    sub_gold_vol = "HOLD"
    if gold_vol_z > _SPIKE_THRESHOLD:
        sub_gold_vol = "ACTIVE_SPIKE"
    elif gold_vol_z > _SPIKE_THRESHOLD * 0.75:
        sub_gold_vol = "ELEVATED"

    sub_vol_momentum = "HOLD"
    if vol_momentum > 0.2:
        sub_vol_momentum = "ACCELERATING"
    elif vol_momentum < -0.2:
        sub_vol_momentum = "DECELERATING"
    else:
        sub_vol_momentum = "STABLE"

    sub_btc_trend = "BUY" if btc_trend > 0 else "SELL"

    if gold_vol_z < _SPIKE_THRESHOLD:
        action = "HOLD"
        confidence = 0.0
    else:
        action = "BUY" if btc_trend > 0 else "SELL"

        base_conf = 0.3 + 0.1 * (gold_vol_z - _SPIKE_THRESHOLD)
        if gold_vol_z > _STRONG_SPIKE:
            base_conf += 0.1

        if vol_momentum > 0.2:
            base_conf += 0.05
        elif vol_momentum < -0.2:
            base_conf -= 0.05

        if trend_strength > 0.02:
            base_conf += 0.05

        confidence = max(0.0, min(_MAX_CONFIDENCE, base_conf))

    return {
        "action": action,
        "confidence": safe_float(confidence),
        "sub_signals": {
            "gold_vol_spike": sub_gold_vol,
            "vol_momentum": sub_vol_momentum,
            "btc_trend": sub_btc_trend,
        },
        "indicators": {
            "gold_vol_zscore": safe_float(gold_vol_z),
            "gold_realized_vol": safe_float(latest_vol),
            "vol_momentum": safe_float(vol_momentum),
            "btc_trend_strength": safe_float(trend_strength),
            "btc_sma_fast": safe_float(latest_fast),
            "btc_sma_slow": safe_float(latest_slow),
        },
    }
