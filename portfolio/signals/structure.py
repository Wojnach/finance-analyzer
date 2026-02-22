"""Composite price-structure / breakout signal.

Combines four sub-indicators into a majority-vote composite:
  1. Period High/Low Breakout  (52-week or available range)
  2. Donchian Channel(55) Breakout
  3. RSI(14) Centerline Cross
  4. MACD(12,26,9) Zero-Line Cross

Each sub-indicator votes BUY / SELL / HOLD.  The composite action is the
majority vote; confidence is the fraction of non-HOLD votes that agree with
the majority direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, rsi

# ---------------------------------------------------------------------------
# Minimum data lengths for each sub-indicator
# ---------------------------------------------------------------------------
_MIN_BARS_RSI = 15       # RSI(14) needs at least 15 bars
_MIN_BARS_MACD = 35      # MACD(12,26,9): 26 + 9 warm-up
_MIN_BARS_DONCHIAN = 56  # Donchian(55) needs 56 to have two values
_MIN_BARS_HIGHLOW = 20   # Bare minimum for period high/low to be meaningful


def _macd_histogram(close: pd.Series,
                    fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram (MACD line minus signal line)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line


# ---------------------------------------------------------------------------
# Sub-indicator votes
# ---------------------------------------------------------------------------

def _highlow_breakout(df: pd.DataFrame) -> tuple[str, dict]:
    """Period high/low proximity check.

    Uses all available data (ideally 252 daily bars for a 52-week window).
    BUY if close is within 2% of the period high; SELL if within 2% of the
    period low; HOLD otherwise.
    """
    indicators: dict = {"period_high": np.nan, "period_low": np.nan}

    if len(df) < _MIN_BARS_HIGHLOW:
        return "HOLD", indicators

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    period_high = high.max()
    period_low = low.min()
    current_close = close.iloc[-1]

    indicators["period_high"] = float(period_high)
    indicators["period_low"] = float(period_low)

    if period_high == 0:
        return "HOLD", indicators

    pct_from_high = (period_high - current_close) / period_high
    pct_from_low = (current_close - period_low) / period_low if period_low != 0 else np.inf

    if pct_from_high <= 0.02:
        return "BUY", indicators
    if pct_from_low <= 0.02:
        return "SELL", indicators
    return "HOLD", indicators


def _donchian_breakout(df: pd.DataFrame, period: int = 55) -> tuple[str, dict]:
    """Donchian Channel(55) breakout.

    BUY when close breaks above the upper channel (highest high of prior
    *period* bars).  SELL when close breaks below the lower channel (lowest
    low of prior *period* bars).  HOLD when inside the channel.
    """
    indicators: dict = {"donchian_upper": np.nan, "donchian_lower": np.nan}

    if len(df) < period + 1:
        return "HOLD", indicators

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    # Channel based on the *previous* period bars (exclude the current bar)
    upper = high.iloc[-(period + 1):-1].max()
    lower = low.iloc[-(period + 1):-1].min()
    current_close = close.iloc[-1]

    indicators["donchian_upper"] = float(upper)
    indicators["donchian_lower"] = float(lower)

    if current_close > upper:
        return "BUY", indicators
    if current_close < lower:
        return "SELL", indicators
    return "HOLD", indicators


def _rsi_centerline(df: pd.DataFrame) -> tuple[str, dict]:
    """RSI(14) centerline cross.

    BUY when RSI > 60, SELL when RSI < 40 (wide deadband to filter noise).
    """
    indicators: dict = {"rsi": np.nan}

    close = df["close"].astype(float)
    if len(df) < _MIN_BARS_RSI:
        return "HOLD", indicators

    rsi_series = rsi(close)
    rsi_val = rsi_series.iloc[-1]
    indicators["rsi"] = float(rsi_val)

    if np.isnan(rsi_val):
        return "HOLD", indicators

    if rsi_val > 60.0:
        return "BUY", indicators
    if rsi_val < 40.0:
        return "SELL", indicators
    return "HOLD", indicators


def _macd_zeroline(df: pd.DataFrame) -> tuple[str, dict]:
    """MACD(12,26,9) histogram zero-line cross.

    BUY when histogram crosses from negative to positive (current > 0 and
    previous <= 0).  SELL when it crosses from positive to negative.
    If no cross occurred on the latest bar, HOLD.
    """
    indicators: dict = {"macd_hist": np.nan}

    close = df["close"].astype(float)
    if len(df) < _MIN_BARS_MACD:
        return "HOLD", indicators

    hist = _macd_histogram(close)

    current = hist.iloc[-1]
    previous = hist.iloc[-2]
    indicators["macd_hist"] = float(current)

    if np.isnan(current) or np.isnan(previous):
        return "HOLD", indicators

    # Bullish cross: histogram flips from non-positive to positive
    if current > 0 and previous <= 0:
        return "BUY", indicators
    # Bearish cross: histogram flips from non-negative to negative
    if current < 0 and previous >= 0:
        return "SELL", indicators
    return "HOLD", indicators


# ---------------------------------------------------------------------------
# Composite signal
# ---------------------------------------------------------------------------

def compute_structure_signal(df: pd.DataFrame) -> dict:
    """Compute the composite price-structure signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV candle data with columns ``open``, ``high``, ``low``,
        ``close``, ``volume``.  At least 20 rows recommended; ideally 100+.

    Returns
    -------
    dict
        ``action`` (BUY / SELL / HOLD), ``confidence`` (0.0-1.0),
        ``sub_signals`` dict, and ``indicators`` dict.
    """
    result: dict = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "high_low_breakout": "HOLD",
            "donchian_55": "HOLD",
            "rsi_centerline": "HOLD",
            "macd_zeroline": "HOLD",
        },
        "indicators": {
            "period_high": np.nan,
            "period_low": np.nan,
            "donchian_upper": np.nan,
            "donchian_lower": np.nan,
            "rsi": np.nan,
            "macd_hist": np.nan,
        },
    }

    # ---- Validate input ----
    if df is None or not isinstance(df, pd.DataFrame):
        return result

    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(set(df.columns)):
        return result

    if len(df) < 2:
        return result

    # ---- Compute each sub-indicator ----
    try:
        hl_action, hl_ind = _highlow_breakout(df)
    except Exception:
        hl_action, hl_ind = "HOLD", {"period_high": np.nan, "period_low": np.nan}

    try:
        dc_action, dc_ind = _donchian_breakout(df, period=55)
    except Exception:
        dc_action, dc_ind = "HOLD", {"donchian_upper": np.nan, "donchian_lower": np.nan}

    try:
        rsi_action, rsi_ind = _rsi_centerline(df)
    except Exception:
        rsi_action, rsi_ind = "HOLD", {"rsi": np.nan}

    try:
        macd_action, macd_ind = _macd_zeroline(df)
    except Exception:
        macd_action, macd_ind = "HOLD", {"macd_hist": np.nan}

    # ---- Populate sub-signals and indicators ----
    result["sub_signals"]["high_low_breakout"] = hl_action
    result["sub_signals"]["donchian_55"] = dc_action
    result["sub_signals"]["rsi_centerline"] = rsi_action
    result["sub_signals"]["macd_zeroline"] = macd_action

    result["indicators"].update(hl_ind)
    result["indicators"].update(dc_ind)
    result["indicators"].update(rsi_ind)
    result["indicators"].update(macd_ind)

    # ---- Majority vote ----
    votes = [hl_action, dc_action, rsi_action, macd_action]
    result["action"], result["confidence"] = majority_vote(votes)

    return result
