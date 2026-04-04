"""Technical indicator computation and regime detection."""

import logging

import numpy as np
import pandas as pd

import portfolio.shared_state as _ss

logger = logging.getLogger("portfolio.indicators")


def compute_indicators(df, horizon=None):
    if horizon == "3h":
        rsi_period = 7
        macd_fast, macd_slow, macd_signal_period = 8, 17, 9
        min_rows = macd_slow  # 17
    else:
        rsi_period = 14
        macd_fast, macd_slow, macd_signal_period = 12, 26, 9
        min_rows = macd_slow  # 26

    if len(df) < min_rows:
        logger.debug("compute_indicators: insufficient data (%d rows, need %d)", len(df), min_rows)
        return None
    close = df["close"].copy()

    # BUG-87: Guard against NaN in close series
    if close.iloc[-1] != close.iloc[-1]:  # NaN check (NaN != NaN is True)
        logger.warning("compute_indicators: last close is NaN, returning None")
        return None
    if close.isna().all():
        logger.warning("compute_indicators: all close values are NaN, returning None")
        return None
    # Forward-fill interior NaN gaps to prevent downstream NaN propagation
    if close.isna().any():
        logger.debug("compute_indicators: forward-filling %d NaN close values", close.isna().sum())
        close = close.ffill().bfill()
        df = df.copy()
        df["close"] = close

    # RSI(rsi_period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss_safe = avg_loss.replace(0, np.finfo(float).eps)
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))

    # MACD(macd_fast, macd_slow, macd_signal_period)
    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=macd_signal_period, adjust=False).mean()
    macd_hist = macd - macd_signal

    # EMA(9, 21)
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()

    # Bollinger Bands(20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # ATR(14)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - close.shift(1)).abs(),
            (df["low"] - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean().iloc[-1]

    # RSI rolling percentiles for adaptive thresholds
    rsi_series = rsi
    rsi_p20 = rsi_series.rolling(100, min_periods=20).quantile(0.2).iloc[-1]
    rsi_p80 = rsi_series.rolling(100, min_periods=20).quantile(0.8).iloc[-1]

    def _safe(val, default=0.0):
        """Return float(val) if finite, else default. Prevents NaN in JSON output."""
        v = float(val)
        if v != v or not np.isfinite(v):  # NaN or Inf
            return default
        return v

    close_val = _safe(close.iloc[-1])
    bb_upper_val = _safe(bb_upper.iloc[-1], close_val)
    bb_lower_val = _safe(bb_lower.iloc[-1], close_val)

    return {
        "close": close_val,
        "rsi": _safe(rsi.iloc[-1], 50.0),
        "macd_hist": _safe(macd_hist.iloc[-1]),
        "macd_hist_prev": _safe(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0.0,
        "ema9": _safe(ema9.iloc[-1], close_val),
        "ema21": _safe(ema21.iloc[-1], close_val),
        "bb_upper": bb_upper_val,
        "bb_lower": bb_lower_val,
        "bb_mid": _safe(bb_mid.iloc[-1], close_val),
        "price_vs_bb": (
            "below_lower"
            if close_val <= bb_lower_val
            else (
                "above_upper"
                if close_val >= bb_upper_val
                else "inside"
            )
        ),
        "atr": _safe(atr14),
        "atr_pct": _safe(atr14 / close.iloc[-1] * 100) if close_val != 0 else 0.0,
        "rsi_p20": _safe(rsi_p20, 30.0),
        "rsi_p80": _safe(rsi_p80, 70.0),
    }


def detect_regime(indicators, is_crypto=True):
    # BUG-169: Access regime cache under lock — 8 threads from ThreadPoolExecutor
    # can call this concurrently. The check-then-clear pattern must be atomic.
    cache_key = (
        round(indicators.get("close", 0), 4),
        round(indicators.get("atr_pct", 0), 4),
        round(indicators.get("ema9", 0), 4),
        round(indicators.get("ema21", 0), 4),
        round(indicators.get("rsi", 50), 4),
        is_crypto,
    )
    with _ss._regime_lock:
        if _ss._run_cycle_id != _ss._regime_cache_cycle:
            _ss._regime_cache = {}
            _ss._regime_cache_cycle = _ss._run_cycle_id
        if cache_key in _ss._regime_cache:
            return _ss._regime_cache[cache_key]

    # Compute outside lock (pure function, no shared state)
    atr_pct = indicators.get("atr_pct", 0)
    ema9 = indicators.get("ema9", 0)
    ema21 = indicators.get("ema21", 0)
    rsi = indicators.get("rsi", 50)

    close = indicators.get("close", 0)
    high_vol_threshold = 4.0 if is_crypto else 3.0
    if atr_pct > high_vol_threshold:
        result = "high-vol"
    elif ema21 != 0 and abs(ema9 - ema21) / ema21 * 100 >= 0.5:
        if ema9 > ema21 and rsi > 45:
            result = "trending-up"
        elif ema9 < ema21 and rsi < 55:
            result = "trending-down"
        else:
            result = "ranging"
    else:
        result = "ranging"

    # BUG-156: EMA crossover lags behind V-shaped recoveries.
    if result == "trending-down" and close > 0 and ema21 > 0 and close > ema21:
        result = "ranging"
    elif result == "trending-up" and close > 0 and ema21 > 0 and close < ema21:
        result = "ranging"

    with _ss._regime_lock:
        _ss._regime_cache[cache_key] = result
    return result


def technical_signal(ind):
    buy = 0
    sell = 0
    total = 0
    # RSI: BUY when < 30 (oversold), SELL when > 70 (overbought), else neutral
    if ind["rsi"] < 30:
        buy += 1
        total += 1
    elif ind["rsi"] > 70:
        sell += 1
        total += 1
    # MACD: histogram crossover (neg->pos = BUY, pos->neg = SELL)
    macd_hist = ind["macd_hist"]
    macd_hist_prev = ind.get("macd_hist_prev", 0.0)
    if macd_hist > 0 and macd_hist_prev <= 0:
        buy += 1
        total += 1
    elif macd_hist < 0 and macd_hist_prev >= 0:
        sell += 1
        total += 1
    # EMA: with deadband — only signal when gap > 0.5%
    ema9 = ind["ema9"]
    ema21 = ind["ema21"]
    ema_gap_pct = abs(ema9 - ema21) / ema21 * 100 if ema21 != 0 else 0
    if ema_gap_pct >= 0.5:
        if ema9 > ema21:
            buy += 1
        else:
            sell += 1
        total += 1
    # BB: below lower = BUY, above upper = SELL
    if ind["price_vs_bb"] == "below_lower":
        buy += 1
        total += 1
    elif ind["price_vs_bb"] == "above_upper":
        sell += 1
        total += 1
    if total == 0:
        return "HOLD", 0.0
    if buy > sell:
        return "BUY", buy / total
    elif sell > buy:
        return "SELL", sell / total
    return "HOLD", 0.5
