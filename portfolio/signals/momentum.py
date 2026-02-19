"""Composite momentum signal module.

Computes 8 momentum sub-indicators and returns a majority-vote composite
BUY/SELL/HOLD signal with confidence score.

Sub-indicators:
    1. RSI Divergence (14-bar lookback)
    2. Stochastic Oscillator (14, 3, 3)
    3. Stochastic RSI (14)
    4. CCI (20)
    5. Williams %R (14)
    6. Rate of Change (12)
    7. PPO (12, 26, 9)
    8. Bull/Bear Power (13)

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 50 rows of data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Minimum rows required for reliable computation.  The longest lookback chain
# is PPO (26-period EMA warm-up + 9-period signal line) which needs ~35 bars,
# but we ask for 50 to give every indicator a reasonable warm-up.
# ---------------------------------------------------------------------------
MIN_ROWS = 50


# ---- helper: exponential moving average ------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average using pandas ewm."""
    return series.ewm(span=span, adjust=False).mean()


# ---- helper: simple moving average -----------------------------------------

def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period).mean()


# ---- helper: RSI -----------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-smoothed RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


# ---- sub-indicator implementations ----------------------------------------

def _rsi_divergence(close: pd.Series, lookback: int = 14) -> str:
    """Detect bullish / bearish RSI divergence over *lookback* bars.

    Bullish divergence: price makes a lower low while RSI makes a higher low.
    Bearish divergence: price makes a higher high while RSI makes a lower high.

    We compare the first and second halves of the lookback window to identify
    two local swing points.
    """
    rsi = _rsi(close)
    if rsi.isna().all() or len(close) < lookback * 2:
        return "HOLD"

    # Last 2*lookback bars, split into two halves to find swing points.
    price_window = close.iloc[-(lookback * 2):]
    rsi_window = rsi.iloc[-(lookback * 2):]

    first_half_price = price_window.iloc[:lookback]
    second_half_price = price_window.iloc[lookback:]
    first_half_rsi = rsi_window.iloc[:lookback]
    second_half_rsi = rsi_window.iloc[lookback:]

    # Bullish divergence: price lower low + RSI higher low
    price_low1 = first_half_price.min()
    price_low2 = second_half_price.min()
    rsi_low1 = first_half_rsi.min()
    rsi_low2 = second_half_rsi.min()

    if price_low2 < price_low1 and rsi_low2 > rsi_low1:
        return "BUY"

    # Bearish divergence: price higher high + RSI lower high
    price_high1 = first_half_price.max()
    price_high2 = second_half_price.max()
    rsi_high1 = first_half_rsi.max()
    rsi_high2 = second_half_rsi.max()

    if price_high2 > price_high1 and rsi_high2 < rsi_high1:
        return "SELL"

    return "HOLD"


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                k_period: int = 14, d_period: int = 3, smooth_k: int = 3
                ) -> tuple[float, float, str]:
    """Stochastic Oscillator (%K, %D).

    Returns (stoch_k, stoch_d, signal).
    %K crosses above %D below 20 = BUY.
    %K crosses below %D above 80 = SELL.
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    denom = highest_high - lowest_low
    raw_k = 100.0 * (close - lowest_low) / denom.replace(0, np.nan)
    k = raw_k.rolling(window=smooth_k).mean()
    d = k.rolling(window=d_period).mean()

    k_val = k.iloc[-1]
    d_val = d.iloc[-1]

    if np.isnan(k_val) or np.isnan(d_val):
        return float("nan"), float("nan"), "HOLD"

    # Need at least two values to detect a cross
    k_prev = k.iloc[-2] if len(k) >= 2 else np.nan
    d_prev = d.iloc[-2] if len(d) >= 2 else np.nan

    if np.isnan(k_prev) or np.isnan(d_prev):
        return k_val, d_val, "HOLD"

    # Bullish cross: %K crosses above %D in oversold zone
    if k_prev <= d_prev and k_val > d_val and d_val < 20:
        return k_val, d_val, "BUY"

    # Bearish cross: %K crosses below %D in overbought zone
    if k_prev >= d_prev and k_val < d_val and d_val > 80:
        return k_val, d_val, "SELL"

    return k_val, d_val, "HOLD"


def _stochastic_rsi(close: pd.Series, period: int = 14) -> tuple[float, str]:
    """Stochastic RSI.

    Returns (stoch_rsi_value, signal).
    StochRSI > 0.8 = overbought (SELL).
    StochRSI < 0.2 = oversold (BUY).
    """
    rsi = _rsi(close, period)
    rsi_min = rsi.rolling(window=period).min()
    rsi_max = rsi.rolling(window=period).max()
    denom = rsi_max - rsi_min
    stoch_rsi = (rsi - rsi_min) / denom.replace(0, np.nan)

    val = stoch_rsi.iloc[-1]
    if np.isnan(val):
        return float("nan"), "HOLD"

    if val < 0.2:
        return val, "BUY"
    if val > 0.8:
        return val, "SELL"
    return val, "HOLD"


def _cci(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 20) -> tuple[float, str]:
    """Commodity Channel Index.

    Returns (cci_value, signal).
    CCI > 100 = overbought (SELL).
    CCI < -100 = oversold (BUY).
    """
    tp = (high + low + close) / 3.0
    tp_sma = _sma(tp, period)
    mean_dev = tp.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cci_series = (tp - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))

    val = cci_series.iloc[-1]
    if np.isnan(val):
        return float("nan"), "HOLD"

    if val < -100:
        return val, "BUY"
    if val > 100:
        return val, "SELL"
    return val, "HOLD"


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> tuple[float, str]:
    """Williams %R.

    Returns (williams_r_value, signal).
    %R > -20 = overbought (SELL).
    %R < -80 = oversold (BUY).
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    denom = highest_high - lowest_low
    wr = -100.0 * (highest_high - close) / denom.replace(0, np.nan)

    val = wr.iloc[-1]
    if np.isnan(val):
        return float("nan"), "HOLD"

    if val < -80:
        return val, "BUY"
    if val > -20:
        return val, "SELL"
    return val, "HOLD"


def _rate_of_change(close: pd.Series, period: int = 12) -> tuple[float, str]:
    """Rate of Change with acceleration check.

    Returns (roc_value, signal).
    ROC > 0 with acceleration (ROC increasing) = BUY.
    ROC < 0 with deceleration (ROC decreasing) = SELL.
    """
    roc = 100.0 * (close - close.shift(period)) / close.shift(period).replace(0, np.nan)

    val = roc.iloc[-1]
    if np.isnan(val) or len(roc.dropna()) < 2:
        return float("nan"), "HOLD"

    prev = roc.iloc[-2]
    if np.isnan(prev):
        return val, "HOLD"

    # ROC positive and accelerating
    if val > 0 and val > prev:
        return val, "BUY"

    # ROC negative and decelerating (becoming more negative)
    if val < 0 and val < prev:
        return val, "SELL"

    return val, "HOLD"


def _ppo(close: pd.Series, fast: int = 12, slow: int = 26,
         signal_period: int = 9) -> tuple[float, float, str]:
    """Percentage Price Oscillator with signal line.

    Returns (ppo_value, ppo_signal_value, signal).
    PPO crosses above signal = BUY.
    PPO crosses below signal = SELL.
    """
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    ppo_line = 100.0 * (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
    signal_line = _ema(ppo_line, signal_period)

    ppo_val = ppo_line.iloc[-1]
    sig_val = signal_line.iloc[-1]

    if np.isnan(ppo_val) or np.isnan(sig_val):
        return float("nan"), float("nan"), "HOLD"

    # Need prior values for crossover detection
    ppo_prev = ppo_line.iloc[-2] if len(ppo_line) >= 2 else np.nan
    sig_prev = signal_line.iloc[-2] if len(signal_line) >= 2 else np.nan

    if np.isnan(ppo_prev) or np.isnan(sig_prev):
        return ppo_val, sig_val, "HOLD"

    # Bullish cross: PPO crosses above signal
    if ppo_prev <= sig_prev and ppo_val > sig_val:
        return ppo_val, sig_val, "BUY"

    # Bearish cross: PPO crosses below signal
    if ppo_prev >= sig_prev and ppo_val < sig_val:
        return ppo_val, sig_val, "SELL"

    return ppo_val, sig_val, "HOLD"


def _bull_bear_power(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 13) -> tuple[float, float, str]:
    """Bull / Bear Power (Elder).

    Bull power = High - EMA(period).
    Bear power = Low  - EMA(period).

    Returns (bull_power, bear_power, signal).
    Both positive = BUY.  Both negative = SELL.
    """
    ema_val = _ema(close, period)
    bull = high - ema_val
    bear = low - ema_val

    bp = bull.iloc[-1]
    brp = bear.iloc[-1]

    if np.isnan(bp) or np.isnan(brp):
        return float("nan"), float("nan"), "HOLD"

    if bp > 0 and brp > 0:
        return bp, brp, "BUY"
    if bp < 0 and brp < 0:
        return bp, brp, "SELL"
    return bp, brp, "HOLD"


# ---- public API ------------------------------------------------------------

def compute_momentum_signal(df: pd.DataFrame) -> dict:
    """Compute composite momentum signal from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``open``, ``high``, ``low``, ``close``, ``volume``
        with at least 50 rows of numeric data.

    Returns
    -------
    dict
        ``action``       : ``'BUY'`` | ``'SELL'`` | ``'HOLD'``
        ``confidence``   : float 0.0-1.0 (proportion of sub-signals agreeing
                           with the majority action)
        ``sub_signals``  : per-indicator votes
        ``indicators``   : raw indicator values for downstream use
    """
    # -- Default / fallback result -----------------------------------------
    hold_result: dict = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "rsi_divergence": "HOLD",
            "stochastic": "HOLD",
            "stochastic_rsi": "HOLD",
            "cci": "HOLD",
            "williams_r": "HOLD",
            "roc": "HOLD",
            "ppo": "HOLD",
            "bull_bear_power": "HOLD",
        },
        "indicators": {
            "stoch_k": float("nan"),
            "stoch_d": float("nan"),
            "stoch_rsi": float("nan"),
            "cci": float("nan"),
            "williams_r": float("nan"),
            "roc": float("nan"),
            "ppo": float("nan"),
            "ppo_signal": float("nan"),
            "bull_power": float("nan"),
            "bear_power": float("nan"),
        },
    }

    # -- Input validation --------------------------------------------------
    if df is None or not isinstance(df, pd.DataFrame):
        return hold_result

    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(set(df.columns)):
        return hold_result

    if len(df) < MIN_ROWS:
        return hold_result

    # Work on a clean copy to avoid mutating the caller's data
    df = df.copy()
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # -- Compute each sub-indicator ----------------------------------------
    sub_signals: dict[str, str] = {}
    indicators: dict[str, float] = {}

    # 1. RSI Divergence
    try:
        sub_signals["rsi_divergence"] = _rsi_divergence(close)
    except Exception:
        sub_signals["rsi_divergence"] = "HOLD"

    # 2. Stochastic Oscillator
    try:
        k_val, d_val, stoch_sig = _stochastic(high, low, close)
        sub_signals["stochastic"] = stoch_sig
        indicators["stoch_k"] = round(k_val, 4) if not np.isnan(k_val) else float("nan")
        indicators["stoch_d"] = round(d_val, 4) if not np.isnan(d_val) else float("nan")
    except Exception:
        sub_signals["stochastic"] = "HOLD"
        indicators["stoch_k"] = float("nan")
        indicators["stoch_d"] = float("nan")

    # 3. Stochastic RSI
    try:
        srsi_val, srsi_sig = _stochastic_rsi(close)
        sub_signals["stochastic_rsi"] = srsi_sig
        indicators["stoch_rsi"] = round(srsi_val, 4) if not np.isnan(srsi_val) else float("nan")
    except Exception:
        sub_signals["stochastic_rsi"] = "HOLD"
        indicators["stoch_rsi"] = float("nan")

    # 4. CCI
    try:
        cci_val, cci_sig = _cci(high, low, close)
        sub_signals["cci"] = cci_sig
        indicators["cci"] = round(cci_val, 4) if not np.isnan(cci_val) else float("nan")
    except Exception:
        sub_signals["cci"] = "HOLD"
        indicators["cci"] = float("nan")

    # 5. Williams %R
    try:
        wr_val, wr_sig = _williams_r(high, low, close)
        sub_signals["williams_r"] = wr_sig
        indicators["williams_r"] = round(wr_val, 4) if not np.isnan(wr_val) else float("nan")
    except Exception:
        sub_signals["williams_r"] = "HOLD"
        indicators["williams_r"] = float("nan")

    # 6. Rate of Change
    try:
        roc_val, roc_sig = _rate_of_change(close)
        sub_signals["roc"] = roc_sig
        indicators["roc"] = round(roc_val, 4) if not np.isnan(roc_val) else float("nan")
    except Exception:
        sub_signals["roc"] = "HOLD"
        indicators["roc"] = float("nan")

    # 7. PPO
    try:
        ppo_val, ppo_sig_val, ppo_sig = _ppo(close)
        sub_signals["ppo"] = ppo_sig
        indicators["ppo"] = round(ppo_val, 4) if not np.isnan(ppo_val) else float("nan")
        indicators["ppo_signal"] = round(ppo_sig_val, 4) if not np.isnan(ppo_sig_val) else float("nan")
    except Exception:
        sub_signals["ppo"] = "HOLD"
        indicators["ppo"] = float("nan")
        indicators["ppo_signal"] = float("nan")

    # 8. Bull/Bear Power
    try:
        bp_val, brp_val, bbp_sig = _bull_bear_power(high, low, close)
        sub_signals["bull_bear_power"] = bbp_sig
        indicators["bull_power"] = round(bp_val, 4) if not np.isnan(bp_val) else float("nan")
        indicators["bear_power"] = round(brp_val, 4) if not np.isnan(brp_val) else float("nan")
    except Exception:
        sub_signals["bull_bear_power"] = "HOLD"
        indicators["bull_power"] = float("nan")
        indicators["bear_power"] = float("nan")

    # -- Majority vote -----------------------------------------------------
    votes = list(sub_signals.values())
    buy_count = votes.count("BUY")
    sell_count = votes.count("SELL")
    hold_count = votes.count("HOLD")
    total = len(votes)  # always 8

    if buy_count > sell_count and buy_count > hold_count:
        action = "BUY"
        confidence = buy_count / total
    elif sell_count > buy_count and sell_count > hold_count:
        action = "SELL"
        confidence = sell_count / total
    else:
        action = "HOLD"
        confidence = hold_count / total

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
