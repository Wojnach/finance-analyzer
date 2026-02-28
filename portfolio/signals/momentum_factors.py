"""Momentum-factor-based trading signal module.

Computes 7 momentum-factor sub-indicators and returns a majority-vote composite
BUY/SELL/HOLD signal with confidence score.

Sub-indicators:
    1. Time-Series Momentum (12-1)   -- classic trend-following lookback
    2. Rate of Change Momentum (ROC-20) -- 20-period rate of change
    3. 52-Week High Proximity        -- distance from period high
    4. 52-Week Low Reversal          -- reversal detection near period low
    5. Consecutive Bars Count        -- green/red bar streaks
    6. Price Acceleration            -- recent vs older ROC comparison
    7. Volume-Weighted Momentum      -- price change with volume confirmation

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 50 rows of data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote

# ---------------------------------------------------------------------------
# Minimum rows required for reliable computation.
# ---------------------------------------------------------------------------
MIN_ROWS = 50


# ---- sub-indicator implementations ----------------------------------------

def _time_series_momentum(close: pd.Series) -> tuple[float, str]:
    """Time-Series Momentum (12-1).

    Classic trend-following factor: compare current price to price 12 months
    ago, skipping the most recent month.  For hourly data (~750 bars = 12
    months, ~63 bars = 1 month), or daily data (~252 bars / ~22 bars).

    Uses whatever lookback is available, scaling proportionally.  Minimum 50
    bars.

    Returns (ts_momentum_pct, signal).
    Positive momentum = BUY, negative = SELL.
    """
    n = len(close)
    if n < MIN_ROWS:
        return float("nan"), "HOLD"

    # Determine skip and lookback proportionally to available data.
    # Ideal: lookback = 252 bars, skip = 22 bars (daily convention).
    # If fewer bars available, scale down but keep the 12:1 ratio.
    ideal_lookback = 252
    ideal_skip = 22

    if n >= ideal_lookback:
        lookback = ideal_lookback
        skip = ideal_skip
    else:
        # Scale proportionally -- keep ~8.7% skip ratio
        lookback = n - 1
        skip = max(1, int(lookback * (ideal_skip / ideal_lookback)))

    # Compare close at (-skip) to close at (-(lookback))
    # i.e. "recent" = close skipping last month, "old" = close 12 months ago
    recent_idx = -(skip + 1)  # skip the most recent `skip` bars
    old_idx = -(lookback + 1) if (lookback + 1) <= n else 0

    recent_price = close.iloc[recent_idx]
    old_price = close.iloc[old_idx]

    if old_price == 0 or np.isnan(old_price) or np.isnan(recent_price):
        return float("nan"), "HOLD"

    ts_mom = (recent_price - old_price) / old_price * 100.0

    if ts_mom > 0:
        return round(ts_mom, 4), "BUY"
    elif ts_mom < 0:
        return round(ts_mom, 4), "SELL"
    return 0.0, "HOLD"


def _roc_20(close: pd.Series) -> tuple[float, str]:
    """Rate of Change Momentum (ROC-20).

    ROC = (close[-1] - close[-20]) / close[-20] * 100.
    ROC > 5% = BUY (strong upward momentum).
    ROC < -5% = SELL.
    Between = HOLD.

    Returns (roc_value, signal).
    """
    if len(close) < 21:
        return float("nan"), "HOLD"

    current = close.iloc[-1]
    past = close.iloc[-21]  # 20 periods ago (index -21 since -1 is current)

    if past == 0 or np.isnan(past) or np.isnan(current):
        return float("nan"), "HOLD"

    roc = (current - past) / past * 100.0

    if roc > 5.0:
        return round(roc, 4), "BUY"
    elif roc < -5.0:
        return round(roc, 4), "SELL"
    return round(roc, 4), "HOLD"


def _high_proximity(close: pd.Series) -> tuple[float, str]:
    """52-Week High Proximity.

    Ratio = close / max(close over available window, up to 252 bars).
    Above 0.95 (within 5% of high) = BUY (momentum, new highs).
    Below 0.80 (20%+ from high) = SELL (broken momentum).
    Between = HOLD.

    Returns (proximity_ratio, signal).
    """
    if len(close) < MIN_ROWS:
        return float("nan"), "HOLD"

    lookback = min(len(close), 252)
    window = close.iloc[-lookback:]
    period_high = window.max()
    current = close.iloc[-1]

    if period_high == 0 or np.isnan(period_high) or np.isnan(current):
        return float("nan"), "HOLD"

    ratio = current / period_high

    if ratio >= 0.95:
        return round(ratio, 4), "BUY"
    elif ratio <= 0.80:
        return round(ratio, 4), "SELL"
    return round(ratio, 4), "HOLD"


def _low_reversal(df: pd.DataFrame) -> tuple[float, str]:
    """52-Week Low Reversal.

    Ratio = close / min(close over available window, up to 252 bars).
    If price is within 5% of 52-week low AND last 3 bars are green
    (close > open) = BUY (reversal from low).
    If making new 52-week lows (within 1%) = SELL (breakdown).
    Otherwise HOLD.

    Returns (low_ratio, signal).
    """
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)

    if len(close) < 4:
        return float("nan"), "HOLD"

    lookback = min(len(close), 252)
    window = close.iloc[-lookback:]
    period_low = window.min()
    current = close.iloc[-1]

    if period_low == 0 or np.isnan(period_low) or np.isnan(current):
        return float("nan"), "HOLD"

    ratio = current / period_low

    # Check if last 3 bars are green (close > open)
    last_3_green = all(
        close.iloc[-i] > open_.iloc[-i] for i in range(1, 4)
    )

    # Within 5% of low AND last 3 bars are green -> reversal BUY
    if ratio <= 1.05 and last_3_green:
        return round(ratio, 4), "BUY"

    # Making new lows (within 1% of period low) -> breakdown SELL
    if ratio <= 1.01:
        return round(ratio, 4), "SELL"

    return round(ratio, 4), "HOLD"


def _consecutive_bars(df: pd.DataFrame) -> tuple[int, str]:
    """Consecutive Bars Count.

    Count consecutive green bars (close > open) and red bars (close < open)
    at the end of the series.
    4+ consecutive green = BUY momentum.
    4+ consecutive red = SELL momentum.
    3 or fewer = HOLD.

    Returns (consecutive_count, signal).
    Positive count = green streak, negative = red streak.
    """
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)

    if len(close) < 2:
        return 0, "HOLD"

    # Determine if each bar is green (1) or red (-1) or doji (0)
    colors = np.sign(close.values - open_.values)

    # Count consecutive bars from the end
    last_color = colors[-1]
    if last_color == 0:
        return 0, "HOLD"

    count = 0
    for i in range(len(colors) - 1, -1, -1):
        if colors[i] == last_color:
            count += 1
        else:
            break

    signed_count = int(count * last_color)

    if count >= 4 and last_color > 0:
        return signed_count, "BUY"
    elif count >= 4 and last_color < 0:
        return signed_count, "SELL"
    return signed_count, "HOLD"


def _price_acceleration(close: pd.Series) -> tuple[float, float, str]:
    """Price Acceleration.

    Compare recent ROC (last 10 bars) to older ROC (10 bars before that).
    If ROC_recent > ROC_older AND both positive = BUY (accelerating up).
    If ROC_recent < ROC_older AND both negative = SELL (accelerating down).
    Otherwise HOLD.

    Returns (roc_recent, roc_older, signal).
    """
    if len(close) < 21:
        return float("nan"), float("nan"), "HOLD"

    # Recent ROC: change over last 10 bars
    recent_end = close.iloc[-1]
    recent_start = close.iloc[-11]
    if recent_start == 0 or np.isnan(recent_start) or np.isnan(recent_end):
        return float("nan"), float("nan"), "HOLD"
    roc_recent = (recent_end - recent_start) / recent_start * 100.0

    # Older ROC: change over 10 bars before the recent window
    older_end = close.iloc[-11]
    older_start = close.iloc[-21]
    if older_start == 0 or np.isnan(older_start) or np.isnan(older_end):
        return float("nan"), float("nan"), "HOLD"
    roc_older = (older_end - older_start) / older_start * 100.0

    if roc_recent > roc_older and roc_recent > 0 and roc_older > 0:
        return round(roc_recent, 4), round(roc_older, 4), "BUY"
    elif roc_recent < roc_older and roc_recent < 0 and roc_older < 0:
        return round(roc_recent, 4), round(roc_older, 4), "SELL"

    return round(roc_recent, 4), round(roc_older, 4), "HOLD"


def _volume_weighted_momentum(df: pd.DataFrame) -> tuple[float, float, str]:
    """Volume-Weighted Momentum.

    Combine 10-period price change with volume confirmation.
    If price change positive AND avg volume of last 10 bars > avg volume
    of prior 20 bars (expanding volume), BUY.
    If price change negative AND volume expanding, SELL.
    If volume contracting or price flat, HOLD.

    Returns (price_change_pct, vol_ratio, signal).
    """
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    if len(close) < 31:
        return float("nan"), float("nan"), "HOLD"

    # Price change over last 10 bars
    price_now = close.iloc[-1]
    price_10_ago = close.iloc[-11]

    if price_10_ago == 0 or np.isnan(price_10_ago) or np.isnan(price_now):
        return float("nan"), float("nan"), "HOLD"

    price_change = (price_now - price_10_ago) / price_10_ago * 100.0

    # Volume comparison
    vol_recent = volume.iloc[-10:].mean()
    vol_prior = volume.iloc[-30:-10].mean()

    if vol_prior == 0 or np.isnan(vol_prior) or np.isnan(vol_recent):
        return round(price_change, 4), float("nan"), "HOLD"

    vol_ratio = vol_recent / vol_prior

    # Volume expanding = ratio > 1.0
    if vol_ratio > 1.0:
        if price_change > 0:
            return round(price_change, 4), round(vol_ratio, 4), "BUY"
        elif price_change < 0:
            return round(price_change, 4), round(vol_ratio, 4), "SELL"

    return round(price_change, 4), round(vol_ratio, 4), "HOLD"


# ---- public API ------------------------------------------------------------

def compute_momentum_factors_signal(df: pd.DataFrame) -> dict:
    """Compute composite momentum-factor signal from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``open``, ``high``, ``low``, ``close``, ``volume``
        with at least 50 rows of numeric data.

    Returns
    -------
    dict
        ``action``       : ``'BUY'`` | ``'SELL'`` | ``'HOLD'``
        ``confidence``   : float 0.0-1.0 (proportion of active voters agreeing
                           with the majority direction)
        ``sub_signals``  : per-indicator votes
        ``indicators``   : raw indicator values for downstream use
    """
    # -- Default / fallback result -----------------------------------------
    hold_result: dict = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "ts_momentum_12_1": "HOLD",
            "roc_20": "HOLD",
            "high_proximity": "HOLD",
            "low_reversal": "HOLD",
            "consecutive_bars": "HOLD",
            "price_acceleration": "HOLD",
            "volume_weighted_momentum": "HOLD",
        },
        "indicators": {
            "ts_momentum_pct": float("nan"),
            "roc_20": float("nan"),
            "high_proximity": float("nan"),
            "low_proximity": float("nan"),
            "consecutive_bars": 0,
            "acceleration_recent": float("nan"),
            "acceleration_older": float("nan"),
            "vol_momentum_price_chg": float("nan"),
            "vol_momentum_ratio": float("nan"),
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

    # -- Compute each sub-indicator ----------------------------------------
    sub_signals: dict[str, str] = {}
    indicators: dict[str, object] = {}

    # 1. Time-Series Momentum (12-1)
    try:
        ts_val, ts_sig = _time_series_momentum(close)
        sub_signals["ts_momentum_12_1"] = ts_sig
        indicators["ts_momentum_pct"] = ts_val
    except Exception:
        sub_signals["ts_momentum_12_1"] = "HOLD"
        indicators["ts_momentum_pct"] = float("nan")

    # 2. Rate of Change Momentum (ROC-20)
    try:
        roc_val, roc_sig = _roc_20(close)
        sub_signals["roc_20"] = roc_sig
        indicators["roc_20"] = roc_val
    except Exception:
        sub_signals["roc_20"] = "HOLD"
        indicators["roc_20"] = float("nan")

    # 3. 52-Week High Proximity
    try:
        hp_val, hp_sig = _high_proximity(close)
        sub_signals["high_proximity"] = hp_sig
        indicators["high_proximity"] = hp_val
    except Exception:
        sub_signals["high_proximity"] = "HOLD"
        indicators["high_proximity"] = float("nan")

    # 4. 52-Week Low Reversal
    try:
        lr_val, lr_sig = _low_reversal(df)
        sub_signals["low_reversal"] = lr_sig
        indicators["low_proximity"] = lr_val
    except Exception:
        sub_signals["low_reversal"] = "HOLD"
        indicators["low_proximity"] = float("nan")

    # 5. Consecutive Bars Count
    try:
        cb_val, cb_sig = _consecutive_bars(df)
        sub_signals["consecutive_bars"] = cb_sig
        indicators["consecutive_bars"] = cb_val
    except Exception:
        sub_signals["consecutive_bars"] = "HOLD"
        indicators["consecutive_bars"] = 0

    # 6. Price Acceleration
    try:
        acc_recent, acc_older, acc_sig = _price_acceleration(close)
        sub_signals["price_acceleration"] = acc_sig
        indicators["acceleration_recent"] = acc_recent
        indicators["acceleration_older"] = acc_older
    except Exception:
        sub_signals["price_acceleration"] = "HOLD"
        indicators["acceleration_recent"] = float("nan")
        indicators["acceleration_older"] = float("nan")

    # 7. Volume-Weighted Momentum
    try:
        vwm_price, vwm_ratio, vwm_sig = _volume_weighted_momentum(df)
        sub_signals["volume_weighted_momentum"] = vwm_sig
        indicators["vol_momentum_price_chg"] = vwm_price
        indicators["vol_momentum_ratio"] = vwm_ratio
    except Exception:
        sub_signals["volume_weighted_momentum"] = "HOLD"
        indicators["vol_momentum_price_chg"] = float("nan")
        indicators["vol_momentum_ratio"] = float("nan")

    # -- Majority vote (among non-HOLD voters) -----------------------------
    votes = list(sub_signals.values())
    action, confidence = majority_vote(votes)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
