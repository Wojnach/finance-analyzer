"""Technical indicator computation and regime detection."""

import numpy as np
import pandas as pd

from portfolio.shared_state import _run_cycle_id, _regime_cache, _regime_cache_cycle
import portfolio.shared_state as _ss


def compute_indicators(df):
    if len(df) < 26:
        return None
    close = df["close"]

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss_safe = avg_loss.replace(0, np.finfo(float).eps)
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))

    # MACD(12,26,9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
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

    return {
        "close": float(close.iloc[-1]),
        "rsi": float(rsi.iloc[-1]),
        "macd_hist": float(macd_hist.iloc[-1]),
        "macd_hist_prev": float(macd_hist.iloc[-2]) if len(macd_hist) > 1 else 0.0,
        "ema9": float(ema9.iloc[-1]),
        "ema21": float(ema21.iloc[-1]),
        "bb_upper": float(bb_upper.iloc[-1]),
        "bb_lower": float(bb_lower.iloc[-1]),
        "bb_mid": float(bb_mid.iloc[-1]),
        "price_vs_bb": (
            "below_lower"
            if float(close.iloc[-1]) <= float(bb_lower.iloc[-1])
            else (
                "above_upper"
                if float(close.iloc[-1]) >= float(bb_upper.iloc[-1])
                else "inside"
            )
        ),
        "atr": float(atr14),
        "atr_pct": float(atr14 / close.iloc[-1]) * 100,
        "rsi_p20": float(rsi_p20) if not pd.isna(rsi_p20) else 30.0,
        "rsi_p80": float(rsi_p80) if not pd.isna(rsi_p80) else 70.0,
    }


def detect_regime(indicators, is_crypto=True):
    # Access mutable state from shared_state module
    if _ss._run_cycle_id != _ss._regime_cache_cycle:
        _ss._regime_cache = {}
        _ss._regime_cache_cycle = _ss._run_cycle_id

    cache_key = (
        round(indicators.get("close", 0), 4),
        round(indicators.get("atr_pct", 0), 4),
        round(indicators.get("ema9", 0), 4),
        round(indicators.get("ema21", 0), 4),
        round(indicators.get("rsi", 50), 4),
        is_crypto,
    )
    if cache_key in _ss._regime_cache:
        return _ss._regime_cache[cache_key]

    atr_pct = indicators.get("atr_pct", 0)
    ema9 = indicators.get("ema9", 0)
    ema21 = indicators.get("ema21", 0)
    rsi = indicators.get("rsi", 50)

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
