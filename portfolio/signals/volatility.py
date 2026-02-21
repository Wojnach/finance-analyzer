"""Composite volatility signal — 6 sub-indicators with majority voting.

Sub-indicators:
    1. BB Squeeze: detects low-volatility compression and breakout release
    2. BB Breakout: price closing outside Bollinger Bands
    3. ATR Expansion: volatility expansion combined with price direction
    4. Keltner Channel(20, 1.5): trend breakout via EMA + ATR envelope
    5. Historical Volatility: 20-day realized vol trend vs price direction
    6. Donchian Channel(20): high/low breakout over rolling window
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Minimum rows required.  BB squeeze lookback (120) is the binding constraint,
# plus a small buffer for warm-up of the underlying 20-period indicators.
# ---------------------------------------------------------------------------
MIN_ROWS = 50


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range (Wilder)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Sub-indicator implementations
# ---------------------------------------------------------------------------

def _bb_squeeze(close: pd.Series, bb_upper: pd.Series, bb_lower: pd.Series,
                bb_middle: pd.Series) -> tuple[str, dict]:
    """BB Squeeze: width < 0.5 * 120-period avg width => squeeze ON.

    On release (width expands back) + price above upper => BUY,
    price below lower => SELL.  During squeeze => HOLD.
    """
    bb_width = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
    current_width = bb_width.iloc[-1]

    # Use up to 120 periods for the average; fall back to whatever is available
    lookback = min(120, len(bb_width))
    avg_width = bb_width.iloc[-lookback:].mean()

    squeeze_on = current_width < 0.5 * avg_width if avg_width > 0 else False

    # Check the previous bar to detect squeeze *release*
    if len(bb_width) >= 2:
        prev_width = bb_width.iloc[-2]
        prev_squeeze = prev_width < 0.5 * avg_width if avg_width > 0 else False
    else:
        prev_squeeze = False

    price = close.iloc[-1]
    upper = bb_upper.iloc[-1]
    lower = bb_lower.iloc[-1]

    action = "HOLD"
    if squeeze_on:
        # Still compressed -- wait
        action = "HOLD"
    elif prev_squeeze and not squeeze_on:
        # Squeeze just released
        if price > upper:
            action = "BUY"
        elif price < lower:
            action = "SELL"
    # If no squeeze context at all, remain HOLD

    indicators = {"bb_width": float(current_width), "bb_squeeze_on": bool(squeeze_on)}
    return action, indicators


def _bb_breakout(close: pd.Series, bb_upper: pd.Series, bb_lower: pd.Series) -> str:
    """Price close above upper BB => BUY, below lower => SELL, else HOLD."""
    price = close.iloc[-1]
    if price > bb_upper.iloc[-1]:
        return "BUY"
    elif price < bb_lower.iloc[-1]:
        return "SELL"
    return "HOLD"


def _atr_expansion(close: pd.Series, high: pd.Series, low: pd.Series) -> tuple[str, dict]:
    """ATR(14) > 1.5x its 20-period SMA => expansion.

    Expansion + price up => BUY, expansion + price down => SELL.
    """
    atr_series = _atr(high, low, close, 14)
    atr_avg = _sma(atr_series, 20)

    current_atr = atr_series.iloc[-1]
    current_avg = atr_avg.iloc[-1]

    if np.isnan(current_atr) or np.isnan(current_avg) or current_avg == 0:
        return "HOLD", {"atr": float(current_atr) if not np.isnan(current_atr) else 0.0,
                        "atr_avg": float(current_avg) if not np.isnan(current_avg) else 0.0}

    expansion = current_atr > 1.5 * current_avg

    action = "HOLD"
    if expansion:
        # Determine price direction from recent closes
        price_change = close.iloc[-1] - close.iloc[-2] if len(close) >= 2 else 0.0
        if price_change > 0:
            action = "BUY"
        elif price_change < 0:
            action = "SELL"

    return action, {"atr": float(current_atr), "atr_avg": float(current_avg)}


def _keltner_channel(close: pd.Series, high: pd.Series,
                     low: pd.Series) -> tuple[str, dict]:
    """Keltner Channel(20, 1.5): EMA(20) +/- 1.5 * ATR(10).

    Price above upper => BUY, below lower => SELL.
    """
    middle = _ema(close, 20)
    atr_10 = _atr(high, low, close, 10)
    upper = middle + 1.5 * atr_10
    lower = middle - 1.5 * atr_10

    price = close.iloc[-1]
    kc_upper = upper.iloc[-1]
    kc_lower = lower.iloc[-1]

    if np.isnan(kc_upper) or np.isnan(kc_lower):
        return "HOLD", {"keltner_upper": 0.0, "keltner_lower": 0.0}

    action = "HOLD"
    if price > kc_upper:
        action = "BUY"
    elif price < kc_lower:
        action = "SELL"

    return action, {"keltner_upper": float(kc_upper), "keltner_lower": float(kc_lower)}


def _historical_volatility(close: pd.Series) -> tuple[str, dict]:
    """20-day realized vol (annualized std of log returns).

    HV increasing + price rising => BUY (healthy trend expansion).
    HV increasing + price falling => SELL.
    HV decreasing => HOLD.
    """
    log_returns = np.log(close / close.shift(1))
    hv = log_returns.rolling(window=20, min_periods=20).std() * np.sqrt(365)

    current_hv = hv.iloc[-1]

    if np.isnan(current_hv):
        return "HOLD", {"hist_vol": 0.0}

    # Check if HV is increasing (compare current vs 5 periods ago)
    lookback = min(5, len(hv) - 1)
    if lookback > 0 and not np.isnan(hv.iloc[-(lookback + 1)]):
        prev_hv = hv.iloc[-(lookback + 1)]
        hv_increasing = current_hv > prev_hv
    else:
        hv_increasing = False

    action = "HOLD"
    if hv_increasing:
        # Determine price direction over the same lookback
        price_change = close.iloc[-1] - close.iloc[-(lookback + 1)]
        if price_change > 0:
            action = "BUY"
        elif price_change < 0:
            action = "SELL"

    return action, {"hist_vol": float(current_hv)}


def _donchian_channel(high: pd.Series, low: pd.Series,
                      close: pd.Series) -> tuple[str, dict]:
    """Donchian Channel(20): highest high / lowest low over 20 periods.

    Price breaks above upper => BUY, below lower => SELL.
    """
    dc_upper = high.rolling(window=20, min_periods=20).max()
    dc_lower = low.rolling(window=20, min_periods=20).min()

    current_upper = dc_upper.iloc[-1]
    current_lower = dc_lower.iloc[-1]

    if np.isnan(current_upper) or np.isnan(current_lower):
        return "HOLD", {"donchian_upper": 0.0, "donchian_lower": 0.0}

    price = close.iloc[-1]
    current_high = high.iloc[-1]
    current_low = low.iloc[-1]

    action = "HOLD"
    # Breakout: current bar makes a new 20-period high and close confirms.
    # Compare against previous bar's channel to detect the *new* breakout.
    has_prev = len(dc_upper) >= 2 and not np.isnan(dc_upper.iloc[-2])

    if current_high >= current_upper:
        # New 20-period high -- confirm with close above prior upper
        if has_prev:
            if price > dc_upper.iloc[-2]:
                action = "BUY"
        elif price >= current_upper:
            action = "BUY"
    elif current_low <= current_lower:
        # New 20-period low -- confirm with close below prior lower
        if has_prev:
            if price < dc_lower.iloc[-2]:
                action = "SELL"
        elif price <= current_lower:
            action = "SELL"

    return action, {"donchian_upper": float(current_upper), "donchian_lower": float(current_lower)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_volatility_signal(df: pd.DataFrame) -> dict[str, Any]:
    """Compute composite volatility signal from OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume.
        Needs at least ``MIN_ROWS`` (50) rows.

    Returns
    -------
    dict with keys:
        action : str        -- 'BUY', 'SELL', or 'HOLD'
        confidence : float  -- 0.0-1.0 agreement ratio among sub-signals
        sub_signals : dict  -- per-indicator votes
        indicators : dict   -- numeric indicator values
    """
    # -- Validate input -------------------------------------------------------
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(c.lower() for c in df.columns)
    if missing:
        logger.warning("volatility signal: missing columns %s", missing)
        return _empty_result(f"missing columns: {missing}")

    # Normalise column names to lowercase
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    if len(df) < MIN_ROWS:
        logger.warning("volatility signal: only %d rows (need %d)", len(df), MIN_ROWS)
        return _empty_result(f"insufficient data ({len(df)} rows, need {MIN_ROWS})")

    # Drop rows where close is NaN or zero (bad data)
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    if len(df) < MIN_ROWS:
        return _empty_result("insufficient valid data after cleaning")

    close = df["close"]
    high = df["high"]
    low = df["low"]

    # -- Bollinger Bands (shared by squeeze + breakout) -----------------------
    bb_period = 20
    bb_std = 2.0
    bb_middle = _sma(close, bb_period)
    rolling_std = close.rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = bb_middle + bb_std * rolling_std
    bb_lower = bb_middle - bb_std * rolling_std

    if np.isnan(bb_upper.iloc[-1]) or np.isnan(bb_lower.iloc[-1]):
        return _empty_result("BB calculation produced NaN (insufficient data)")

    # -- Compute all 6 sub-signals -------------------------------------------
    sub_signals: dict[str, str] = {}
    indicators: dict[str, Any] = {}

    try:
        action, ind = _bb_squeeze(close, bb_upper, bb_lower, bb_middle)
        sub_signals["bb_squeeze"] = action
        indicators.update(ind)
    except Exception as exc:
        logger.error("bb_squeeze failed: %s", exc)
        sub_signals["bb_squeeze"] = "HOLD"
        indicators.update({"bb_width": 0.0, "bb_squeeze_on": False})

    try:
        sub_signals["bb_breakout"] = _bb_breakout(close, bb_upper, bb_lower)
    except Exception as exc:
        logger.error("bb_breakout failed: %s", exc)
        sub_signals["bb_breakout"] = "HOLD"

    try:
        action, ind = _atr_expansion(close, high, low)
        sub_signals["atr_expansion"] = action
        indicators.update(ind)
    except Exception as exc:
        logger.error("atr_expansion failed: %s", exc)
        sub_signals["atr_expansion"] = "HOLD"
        indicators.update({"atr": 0.0, "atr_avg": 0.0})

    try:
        action, ind = _keltner_channel(close, high, low)
        sub_signals["keltner"] = action
        indicators.update(ind)
    except Exception as exc:
        logger.error("keltner failed: %s", exc)
        sub_signals["keltner"] = "HOLD"
        indicators.update({"keltner_upper": 0.0, "keltner_lower": 0.0})

    try:
        action, ind = _historical_volatility(close)
        sub_signals["historical_vol"] = action
        indicators.update(ind)
    except Exception as exc:
        logger.error("historical_vol failed: %s", exc)
        sub_signals["historical_vol"] = "HOLD"
        indicators.update({"hist_vol": 0.0})

    try:
        action, ind = _donchian_channel(high, low, close)
        sub_signals["donchian"] = action
        indicators.update(ind)
    except Exception as exc:
        logger.error("donchian failed: %s", exc)
        sub_signals["donchian"] = "HOLD"
        indicators.update({"donchian_upper": 0.0, "donchian_lower": 0.0})

    # -- Majority vote --------------------------------------------------------
    votes = list(sub_signals.values())
    buy_count = votes.count("BUY")
    sell_count = votes.count("SELL")
    hold_count = votes.count("HOLD")
    total = len(votes)

    if buy_count > sell_count and buy_count > hold_count:
        composite_action = "BUY"
        confidence = buy_count / total
    elif sell_count > buy_count and sell_count > hold_count:
        composite_action = "SELL"
        confidence = sell_count / total
    else:
        composite_action = "HOLD"
        confidence = hold_count / total

    return {
        "action": composite_action,
        "confidence": round(confidence, 4),
        "sub_signals": sub_signals,
        "indicators": indicators,
    }


def _empty_result(reason: str = "") -> dict[str, Any]:
    """Return a neutral HOLD result when computation cannot proceed."""
    logger.debug("volatility signal returning empty: %s", reason)
    return {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "bb_squeeze": "HOLD",
            "bb_breakout": "HOLD",
            "atr_expansion": "HOLD",
            "keltner": "HOLD",
            "historical_vol": "HOLD",
            "donchian": "HOLD",
        },
        "indicators": {
            "bb_width": 0.0,
            "bb_squeeze_on": False,
            "atr": 0.0,
            "atr_avg": 0.0,
            "keltner_upper": 0.0,
            "keltner_lower": 0.0,
            "hist_vol": 0.0,
            "donchian_upper": 0.0,
            "donchian_lower": 0.0,
        },
    }
