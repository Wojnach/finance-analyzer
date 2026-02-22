"""Composite mean-reversion signal module.

Computes 7 mean-reversion sub-indicators and returns a majority-vote composite
BUY/SELL/HOLD signal with confidence score.

Sub-indicators:
    1. RSI(2) Mean Reversion       (extreme short-term oversold/overbought snap-back)
    2. RSI(3) Mean Reversion       (slightly less extreme short-term mean reversion)
    3. Internal Bar Strength (IBS)  (close position within bar range)
    4. Consecutive Down Days Entry  (3+ consecutive directional closes)
    5. Gap Fade / Gap Fill          (gap-fill after significant open gap)
    6. Bollinger Band %B MR         (price outside or near Bollinger Bands)
    7. IBS + RSI(2) Combined        (combined extreme reading confirmation)

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 20 rows of data (for Bollinger Band calculation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, rsi, safe_float, sma

# ---------------------------------------------------------------------------
# Minimum rows required for reliable computation.  The longest lookback is
# the Bollinger Band (20-period SMA + 2-std), so we require 20 bars.
# RSI(2) needs only 3 bars, but we enforce 20 for BB.
# ---------------------------------------------------------------------------
MIN_ROWS_BB = 20
MIN_ROWS_RSI2 = 3


# ---- sub-indicator 1: RSI(2) Mean Reversion --------------------------------

def _rsi2_mean_reversion(close: pd.Series) -> tuple[float, str]:
    """RSI with period=2.  Ultra-short-term mean reversion.

    Below 10 = BUY (extreme oversold snap-back expected).
    Above 90 = SELL (extreme overbought reversal expected).

    Returns (rsi2_value, signal).
    """
    if len(close) < MIN_ROWS_RSI2:
        return float("nan"), "HOLD"

    rsi2 = rsi(close, period=2)
    val = rsi2.iloc[-1]

    if np.isnan(val):
        return float("nan"), "HOLD"

    val = float(val)
    if val < 10:
        return val, "BUY"
    if val > 90:
        return val, "SELL"
    return val, "HOLD"


# ---- sub-indicator 2: RSI(3) Mean Reversion --------------------------------

def _rsi3_mean_reversion(close: pd.Series) -> tuple[float, str]:
    """RSI with period=3.  Short-term mean reversion.

    Below 15 = BUY (oversold snap-back expected).
    Above 85 = SELL (overbought reversal expected).

    Returns (rsi3_value, signal).
    """
    if len(close) < 4:
        return float("nan"), "HOLD"

    rsi3 = rsi(close, period=3)
    val = rsi3.iloc[-1]

    if np.isnan(val):
        return float("nan"), "HOLD"

    val = float(val)
    if val < 15:
        return val, "BUY"
    if val > 85:
        return val, "SELL"
    return val, "HOLD"


# ---- sub-indicator 3: Internal Bar Strength (IBS) ---------------------------

def _internal_bar_strength(high: pd.Series, low: pd.Series,
                           close: pd.Series) -> tuple[float, str]:
    """Internal Bar Strength: IBS = (close - low) / (high - low).

    Uses the last bar only.
    Below 0.2 = BUY (closed near low, likely bounce).
    Above 0.8 = SELL (closed near high, likely reversal).

    Returns (ibs_value, signal).
    """
    if len(close) < 1:
        return float("nan"), "HOLD"

    h = float(high.iloc[-1])
    l = float(low.iloc[-1])
    c = float(close.iloc[-1])

    bar_range = h - l
    if bar_range == 0 or np.isnan(bar_range):
        return float("nan"), "HOLD"

    ibs = (c - l) / bar_range

    if ibs < 0.2:
        return ibs, "BUY"
    if ibs > 0.8:
        return ibs, "SELL"
    return ibs, "HOLD"


# ---- sub-indicator 4: Consecutive Down Days Entry ---------------------------

def _consecutive_days(close: pd.Series) -> tuple[int, str]:
    """Count consecutive days where close < previous close (down) or > (up).

    3+ consecutive down days = BUY (mean reversion expected).
    3+ consecutive up days = SELL.

    Returns (consecutive_count, signal).
    A positive count means consecutive up days, negative means consecutive down.
    """
    if len(close) < 2:
        return 0, "HOLD"

    # Count consecutive direction from the end
    count = 0
    for i in range(len(close) - 1, 0, -1):
        curr = float(close.iloc[i])
        prev = float(close.iloc[i - 1])
        if np.isnan(curr) or np.isnan(prev):
            break
        if curr < prev:
            if count > 0:
                # Direction changed from up to down
                break
            count -= 1
        elif curr > prev:
            if count < 0:
                # Direction changed from down to up
                break
            count += 1
        else:
            # Flat day — break the streak
            break

    # 3+ consecutive down days (count <= -3) = BUY
    if count <= -3:
        return count, "BUY"
    # 3+ consecutive up days (count >= 3) = SELL
    if count >= 3:
        return count, "SELL"
    return count, "HOLD"


# ---- sub-indicator 5: Gap Fade / Gap Fill -----------------------------------

def _gap_fill(open_prices: pd.Series, close: pd.Series,
              high: pd.Series, low: pd.Series,
              gap_threshold: float = 0.005) -> tuple[float, float, str]:
    """Detect gap fill after a significant open gap.

    A gap occurs when today's open differs from yesterday's close by more
    than the threshold.  If price has filled >30% of the gap distance,
    signal in the gap-fill direction.

    Gap up + filling = SELL.  Gap down + filling = BUY.

    Parameters
    ----------
    gap_threshold : float
        Minimum gap size as a fraction (0.005 = 0.5% for stocks, caller can
        pass 0.01 for crypto).

    Returns (gap_pct, fill_pct, signal).
    """
    if len(close) < 2:
        return 0.0, 0.0, "HOLD"

    prev_close = float(close.iloc[-2])
    today_open = float(open_prices.iloc[-1])

    if np.isnan(prev_close) or np.isnan(today_open) or prev_close == 0:
        return 0.0, 0.0, "HOLD"

    gap_pct = (today_open - prev_close) / prev_close
    gap_distance = today_open - prev_close

    if abs(gap_pct) < gap_threshold:
        return safe_float(gap_pct), 0.0, "HOLD"

    # How much of the gap has been filled?
    # For a gap up, price needs to move down toward prev_close.
    # For a gap down, price needs to move up toward prev_close.
    today_close = float(close.iloc[-1])
    if np.isnan(today_close):
        return safe_float(gap_pct), 0.0, "HOLD"

    if gap_distance == 0:
        return safe_float(gap_pct), 0.0, "HOLD"

    # Fill amount: how much of the gap has been retraced
    fill_amount = today_open - today_close  # positive if price moved down from open
    fill_pct = fill_amount / gap_distance   # positive means filling the gap

    if fill_pct < 0.3:
        # Gap not filling yet
        return safe_float(gap_pct), safe_float(fill_pct), "HOLD"

    # Gap up + filling (price falling back) = SELL
    if gap_pct > 0:
        return safe_float(gap_pct), safe_float(fill_pct), "SELL"

    # Gap down + filling (price rising back) = BUY
    if gap_pct < 0:
        return safe_float(gap_pct), safe_float(fill_pct), "BUY"

    return safe_float(gap_pct), safe_float(fill_pct), "HOLD"


# ---- sub-indicator 6: Bollinger Band %B Mean Reversion ----------------------

def _bb_pct_b(close: pd.Series, period: int = 20,
              num_std: float = 2.0) -> tuple[float, str]:
    """Bollinger Band %B = (close - bb_lower) / (bb_upper - bb_lower).

    %B < 0.0 = BUY (below lower band — strong mean reversion).
    %B 0.0-0.2 = weak BUY (near lower band).
    %B 0.8-1.0 = weak SELL (near upper band).
    %B > 1.0 = SELL (above upper band — strong mean reversion).

    Returns (pct_b_value, signal).
    """
    if len(close) < period:
        return float("nan"), "HOLD"

    sma_val = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std()

    bb_upper = sma_val + num_std * std
    bb_lower = sma_val - num_std * std

    upper = bb_upper.iloc[-1]
    lower = bb_lower.iloc[-1]
    c = close.iloc[-1]

    if np.isnan(upper) or np.isnan(lower) or np.isnan(c):
        return float("nan"), "HOLD"

    band_width = float(upper) - float(lower)
    if band_width == 0:
        return float("nan"), "HOLD"

    pct_b = (float(c) - float(lower)) / band_width

    if pct_b < 0.0:
        return pct_b, "BUY"
    if pct_b > 1.0:
        return pct_b, "SELL"
    if pct_b < 0.2:
        return pct_b, "BUY"
    if pct_b > 0.8:
        return pct_b, "SELL"
    return pct_b, "HOLD"


# ---- sub-indicator 7: IBS + RSI(2) Combined --------------------------------

def _ibs_rsi2_combined(high: pd.Series, low: pd.Series,
                       close: pd.Series) -> tuple[float, float, str]:
    """Combined IBS + RSI(2) signal.  Only votes when both agree.

    BOTH IBS < 0.2 AND RSI(2) < 10 = strong BUY.
    BOTH IBS > 0.8 AND RSI(2) > 90 = strong SELL.
    Otherwise HOLD (does not vote).

    Returns (ibs_value, rsi2_value, signal).
    """
    if len(close) < MIN_ROWS_RSI2:
        return float("nan"), float("nan"), "HOLD"

    # IBS
    h = float(high.iloc[-1])
    l = float(low.iloc[-1])
    c = float(close.iloc[-1])

    bar_range = h - l
    if bar_range == 0 or np.isnan(bar_range):
        return float("nan"), float("nan"), "HOLD"

    ibs = (c - l) / bar_range

    # RSI(2)
    rsi2 = rsi(close, period=2)
    rsi2_val = rsi2.iloc[-1]

    if np.isnan(rsi2_val):
        return safe_float(ibs), float("nan"), "HOLD"

    rsi2_val = float(rsi2_val)

    if ibs < 0.2 and rsi2_val < 10:
        return safe_float(ibs), rsi2_val, "BUY"
    if ibs > 0.8 and rsi2_val > 90:
        return safe_float(ibs), rsi2_val, "SELL"

    return safe_float(ibs), rsi2_val, "HOLD"


# ---- public API ------------------------------------------------------------

def compute_mean_reversion_signal(df: pd.DataFrame) -> dict:
    """Compute composite mean-reversion signal from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``open``, ``high``, ``low``, ``close``, ``volume``
        with at least 20 rows of numeric data for full computation (3 rows
        minimum for RSI-only sub-signals).

    Returns
    -------
    dict
        ``action``       : ``'BUY'`` | ``'SELL'`` | ``'HOLD'``
        ``confidence``   : float 0.0-1.0 (proportion of voting sub-signals
                           agreeing with the majority action)
        ``sub_signals``  : per-indicator votes
        ``indicators``   : raw indicator values for downstream use
    """
    # -- Default / fallback result -----------------------------------------
    hold_result: dict = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "rsi2_mr": "HOLD",
            "rsi3_mr": "HOLD",
            "ibs": "HOLD",
            "consecutive_days": "HOLD",
            "gap_fill": "HOLD",
            "bb_pct_b": "HOLD",
            "ibs_rsi2_combined": "HOLD",
        },
        "indicators": {
            "rsi2": float("nan"),
            "rsi3": float("nan"),
            "ibs": float("nan"),
            "consecutive_days": 0,
            "gap_pct": 0.0,
            "gap_fill_pct": 0.0,
            "bb_pct_b": float("nan"),
            "combined_ibs": float("nan"),
            "combined_rsi2": float("nan"),
        },
    }

    # -- Input validation --------------------------------------------------
    if df is None or not isinstance(df, pd.DataFrame):
        return hold_result

    required_cols = {"open", "high", "low", "close", "volume"}
    col_map = {c.lower(): c for c in df.columns}
    missing = required_cols - set(col_map.keys())
    if missing:
        return hold_result

    if len(df) < MIN_ROWS_RSI2:
        return hold_result

    # Work on a clean copy to avoid mutating the caller's data
    df = df.copy()
    close = df[col_map["close"]].astype(float)
    high = df[col_map["high"]].astype(float)
    low = df[col_map["low"]].astype(float)
    open_prices = df[col_map["open"]].astype(float)

    # -- Compute each sub-indicator ----------------------------------------
    sub_signals: dict[str, str] = {}
    indicators: dict[str, object] = {}

    # 1. RSI(2) Mean Reversion
    try:
        rsi2_val, rsi2_sig = _rsi2_mean_reversion(close)
        sub_signals["rsi2_mr"] = rsi2_sig
        indicators["rsi2"] = safe_float(rsi2_val)
    except Exception:
        sub_signals["rsi2_mr"] = "HOLD"
        indicators["rsi2"] = float("nan")

    # 2. RSI(3) Mean Reversion
    try:
        rsi3_val, rsi3_sig = _rsi3_mean_reversion(close)
        sub_signals["rsi3_mr"] = rsi3_sig
        indicators["rsi3"] = safe_float(rsi3_val)
    except Exception:
        sub_signals["rsi3_mr"] = "HOLD"
        indicators["rsi3"] = float("nan")

    # 3. Internal Bar Strength
    try:
        ibs_val, ibs_sig = _internal_bar_strength(high, low, close)
        sub_signals["ibs"] = ibs_sig
        indicators["ibs"] = safe_float(ibs_val)
    except Exception:
        sub_signals["ibs"] = "HOLD"
        indicators["ibs"] = float("nan")

    # 4. Consecutive Down Days
    try:
        cons_count, cons_sig = _consecutive_days(close)
        sub_signals["consecutive_days"] = cons_sig
        indicators["consecutive_days"] = cons_count
    except Exception:
        sub_signals["consecutive_days"] = "HOLD"
        indicators["consecutive_days"] = 0

    # 5. Gap Fade / Gap Fill
    try:
        gap_pct_val, fill_pct_val, gap_sig = _gap_fill(
            open_prices, close, high, low
        )
        sub_signals["gap_fill"] = gap_sig
        indicators["gap_pct"] = safe_float(gap_pct_val)
        indicators["gap_fill_pct"] = safe_float(fill_pct_val)
    except Exception:
        sub_signals["gap_fill"] = "HOLD"
        indicators["gap_pct"] = 0.0
        indicators["gap_fill_pct"] = 0.0

    # 6. Bollinger Band %B Mean Reversion
    try:
        if len(df) >= MIN_ROWS_BB:
            pct_b_val, pct_b_sig = _bb_pct_b(close)
            sub_signals["bb_pct_b"] = pct_b_sig
            indicators["bb_pct_b"] = safe_float(pct_b_val)
        else:
            sub_signals["bb_pct_b"] = "HOLD"
            indicators["bb_pct_b"] = float("nan")
    except Exception:
        sub_signals["bb_pct_b"] = "HOLD"
        indicators["bb_pct_b"] = float("nan")

    # 7. IBS + RSI(2) Combined
    try:
        comb_ibs, comb_rsi2, comb_sig = _ibs_rsi2_combined(high, low, close)
        sub_signals["ibs_rsi2_combined"] = comb_sig
        indicators["combined_ibs"] = safe_float(comb_ibs)
        indicators["combined_rsi2"] = safe_float(comb_rsi2)
    except Exception:
        sub_signals["ibs_rsi2_combined"] = "HOLD"
        indicators["combined_ibs"] = float("nan")
        indicators["combined_rsi2"] = float("nan")

    # -- Majority vote -----------------------------------------------------
    votes = list(sub_signals.values())
    action, confidence = majority_vote(votes)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
