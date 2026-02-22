"""Composite Heikin-Ashi and advanced trend signal module.

Computes 7 sub-indicators using Heikin-Ashi candles, Hull Moving Average,
Williams Alligator, Elder Impulse System, and TTM Squeeze, then returns a
composite BUY/SELL/HOLD vote via majority voting.

Sub-indicators:
    1. Heikin-Ashi Trend       — 3 consecutive strong HA candles (no wick)
    2. Heikin-Ashi Doji        — HA doji reversal after streak
    3. Heikin-Ashi Color Change — HA candle color transition
    4. Hull Moving Average Cross(9, 21)
    5. Alligator Indicator     — SMMA(13/8/5) with forward shifts
    6. Elder Impulse System    — EMA(13) + MACD histogram direction
    7. TTM Squeeze             — Bollinger inside Keltner + momentum

Requires a DataFrame with columns: open, high, low, close, volume.
At least 50 rows recommended; returns HOLD with 0.0 confidence when
insufficient data is available.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_ROWS = 50          # minimum rows for meaningful output
_MIN_ROWS_BASIC = 10    # absolute minimum to attempt any calculation
_NUM_SUB_SIGNALS = 7
_DOJI_BODY_PCT = 0.10   # body < 10% of range = doji
_HA_STREAK_LEN = 3      # consecutive candles for strong trend


# ---------------------------------------------------------------------------
# Moving average helpers
# ---------------------------------------------------------------------------

from portfolio.signal_utils import ema, rma, safe_float, sma, true_range, wma


def _hma(series: pd.Series, period: int) -> pd.Series:
    """Hull Moving Average.

    HMA(n) = WMA( 2*WMA(n/2) - WMA(n), sqrt(n) )
    """
    half_period = max(int(round(period / 2)), 1)
    sqrt_period = max(int(round(math.sqrt(period))), 1)

    wma_half = wma(series, half_period)
    wma_full = wma(series, period)

    raw = 2.0 * wma_half - wma_full
    return wma(raw, sqrt_period)


# ---------------------------------------------------------------------------
# Heikin-Ashi candle computation
# ---------------------------------------------------------------------------

def _compute_ha_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heikin-Ashi OHLC from regular OHLC data.

    HA_Close = (O + H + L + C) / 4
    HA_Open[0] = (O[0] + C[0]) / 2
    HA_Open[i] = (HA_Open[i-1] + HA_Close[i-1]) / 2
    HA_High = max(H, HA_Open, HA_Close)
    HA_Low  = min(L, HA_Open, HA_Close)
    """
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    n = len(o)

    ha_close = (o + h + l + c) / 4.0

    ha_open = np.empty(n, dtype=float)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(l, np.minimum(ha_open, ha_close))

    return pd.DataFrame(
        {
            "ha_open": ha_open,
            "ha_high": ha_high,
            "ha_low": ha_low,
            "ha_close": ha_close,
        },
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Sub-signal 1: Heikin-Ashi Trend (strong candles with no opposing wicks)
# ---------------------------------------------------------------------------

def _ha_trend_signal(ha: pd.DataFrame) -> tuple[str, str, int]:
    """Check last N HA candles for strong trend (no opposing wicks).

    Strong BUY:  last 3 candles all green AND no lower wicks
                 (ha_low == ha_open for each candle).
    Strong SELL: last 3 candles all red AND no upper wicks
                 (ha_high == ha_open for each candle).

    Returns (signal, ha_color, ha_streak).
    """
    n = len(ha)
    if n < _HA_STREAK_LEN:
        last = ha.iloc[-1]
        color = "green" if last["ha_close"] > last["ha_open"] else "red"
        return "HOLD", color, 0

    # Determine color and streak of the last candle
    last = ha.iloc[-1]
    is_green_last = last["ha_close"] > last["ha_open"]
    color = "green" if is_green_last else "red"

    # Count consecutive same-color candles from the end
    streak = 0
    for i in range(n - 1, -1, -1):
        row = ha.iloc[i]
        row_green = row["ha_close"] > row["ha_open"]
        if row_green == is_green_last:
            streak += 1
        else:
            break

    # Check the last _HA_STREAK_LEN candles for the strong pattern
    tail = ha.iloc[-_HA_STREAK_LEN:]

    # Use a small tolerance for wick comparison (floating-point precision)
    tol = 1e-10

    all_green_no_lower_wick = True
    all_red_no_upper_wick = True

    for i in range(len(tail)):
        row = tail.iloc[i]
        is_green = row["ha_close"] > row["ha_open"]
        is_red = row["ha_close"] < row["ha_open"]
        ha_range = row["ha_high"] - row["ha_low"]

        if ha_range < tol:
            # Negligible range candle -- not a strong trend candle
            all_green_no_lower_wick = False
            all_red_no_upper_wick = False
            break

        # Green candle: no lower wick means ha_low == ha_open
        if not is_green or abs(row["ha_low"] - row["ha_open"]) > tol * ha_range + tol:
            all_green_no_lower_wick = False

        # Red candle: no upper wick means ha_high == ha_open
        if not is_red or abs(row["ha_high"] - row["ha_open"]) > tol * ha_range + tol:
            all_red_no_upper_wick = False

    if all_green_no_lower_wick:
        return "BUY", color, streak
    if all_red_no_upper_wick:
        return "SELL", color, streak
    return "HOLD", color, streak


# ---------------------------------------------------------------------------
# Sub-signal 2: Heikin-Ashi Doji (reversal after streak)
# ---------------------------------------------------------------------------

def _ha_doji_signal(ha: pd.DataFrame) -> str:
    """Detect HA doji candle and infer reversal direction from prior streak.

    Doji: body < 10% of range on the last candle.
    After green streak -> doji = potential SELL reversal.
    After red streak   -> doji = potential BUY reversal.
    """
    n = len(ha)
    if n < 2:
        return "HOLD"

    last = ha.iloc[-1]
    ha_range = last["ha_high"] - last["ha_low"]
    if ha_range <= 0:
        return "HOLD"

    body = abs(last["ha_close"] - last["ha_open"])
    body_pct = body / ha_range

    if body_pct >= _DOJI_BODY_PCT:
        return "HOLD"  # Not a doji

    # Doji detected -- determine prior streak direction
    # Look at the candle before the doji to determine the streak
    prev_streak_color = None
    streak_count = 0
    for i in range(n - 2, -1, -1):
        row = ha.iloc[i]
        is_green = row["ha_close"] > row["ha_open"]
        if prev_streak_color is None:
            prev_streak_color = "green" if is_green else "red"
            streak_count = 1
        elif (is_green and prev_streak_color == "green") or \
             (not is_green and prev_streak_color == "red"):
            streak_count += 1
        else:
            break

    # Need at least 2 consecutive candles in one direction before the doji
    if streak_count < 2:
        return "HOLD"

    if prev_streak_color == "green":
        return "SELL"  # Doji after green streak = potential top reversal
    elif prev_streak_color == "red":
        return "BUY"   # Doji after red streak = potential bottom reversal

    return "HOLD"


# ---------------------------------------------------------------------------
# Sub-signal 3: Heikin-Ashi Color Change
# ---------------------------------------------------------------------------

def _ha_color_change_signal(ha: pd.DataFrame) -> str:
    """Detect HA color transition on the most recent bar.

    Red -> Green = BUY.
    Green -> Red = SELL.
    Same color   = HOLD.
    """
    n = len(ha)
    if n < 2:
        return "HOLD"

    prev = ha.iloc[-2]
    curr = ha.iloc[-1]

    prev_green = prev["ha_close"] > prev["ha_open"]
    curr_green = curr["ha_close"] > curr["ha_open"]

    # Handle flat candles (close == open) as continuation
    if abs(prev["ha_close"] - prev["ha_open"]) < 1e-12:
        return "HOLD"
    if abs(curr["ha_close"] - curr["ha_open"]) < 1e-12:
        return "HOLD"

    if not prev_green and curr_green:
        return "BUY"   # Red to green
    if prev_green and not curr_green:
        return "SELL"  # Green to red

    return "HOLD"


# ---------------------------------------------------------------------------
# Sub-signal 4: Hull Moving Average Cross (9, 21)
# ---------------------------------------------------------------------------

def _hull_ma_signal(close: pd.Series) -> tuple[str, float, float]:
    """HMA(9) crossing HMA(21).

    HMA(9) > HMA(21) = BUY (fast above slow).
    HMA(9) < HMA(21) = SELL (fast below slow).

    Returns (signal, hull_fast, hull_slow).
    """
    hma_fast = _hma(close, 9)
    hma_slow = _hma(close, 21)

    fast_val = hma_fast.iloc[-1]
    slow_val = hma_slow.iloc[-1]

    if pd.isna(fast_val) or pd.isna(slow_val):
        return "HOLD", float("nan"), float("nan")

    fast_f = float(fast_val)
    slow_f = float(slow_val)

    if fast_f > slow_f:
        return "BUY", fast_f, slow_f
    elif fast_f < slow_f:
        return "SELL", fast_f, slow_f

    return "HOLD", fast_f, slow_f


# ---------------------------------------------------------------------------
# Sub-signal 5: Alligator Indicator
# ---------------------------------------------------------------------------

def _alligator_signal(close: pd.Series) -> tuple[str, float, float, float]:
    """Williams Alligator.

    Jaw   = SMMA(13) shifted forward 8 bars.
    Teeth = SMMA(8) shifted forward 5 bars.
    Lips  = SMMA(5) shifted forward 3 bars.

    Lips > Teeth > Jaw = BUY  (awakening upward).
    Lips < Teeth < Jaw = SELL (awakening downward).
    Otherwise          = HOLD (sleeping / intertwined).

    Returns (signal, lips, teeth, jaw).
    """
    jaw_raw = rma(close, 13)
    teeth_raw = rma(close, 8)
    lips_raw = rma(close, 5)

    # Shift forward (the Alligator projects lines into the future)
    jaw = jaw_raw.shift(8)
    teeth = teeth_raw.shift(5)
    lips = lips_raw.shift(3)

    jaw_val = jaw.iloc[-1]
    teeth_val = teeth.iloc[-1]
    lips_val = lips.iloc[-1]

    if pd.isna(jaw_val) or pd.isna(teeth_val) or pd.isna(lips_val):
        return "HOLD", float("nan"), float("nan"), float("nan")

    jaw_f = float(jaw_val)
    teeth_f = float(teeth_val)
    lips_f = float(lips_val)

    if lips_f > teeth_f > jaw_f:
        return "BUY", lips_f, teeth_f, jaw_f
    elif lips_f < teeth_f < jaw_f:
        return "SELL", lips_f, teeth_f, jaw_f

    return "HOLD", lips_f, teeth_f, jaw_f


# ---------------------------------------------------------------------------
# Sub-signal 6: Elder Impulse System
# ---------------------------------------------------------------------------

def _elder_impulse_signal(close: pd.Series) -> tuple[str, str]:
    """Elder Impulse System.

    Combines EMA(13) direction with MACD histogram direction:
      - EMA rising  + MACD-hist rising  = green (BUY)
      - EMA falling + MACD-hist falling = red   (SELL)
      - Mixed                           = blue  (HOLD)

    MACD uses standard (12, 26, 9) parameters.

    Returns (signal, elder_color).
    """
    ema13 = ema(close, 13)

    # MACD: fast EMA(12) - slow EMA(26)
    macd_line = ema(close, 12) - ema(close, 26)
    macd_signal = ema(macd_line, 9)
    macd_hist = macd_line - macd_signal

    if len(ema13.dropna()) < 2 or len(macd_hist.dropna()) < 2:
        return "HOLD", "blue"

    ema_curr = ema13.iloc[-1]
    ema_prev = ema13.iloc[-2]
    hist_curr = macd_hist.iloc[-1]
    hist_prev = macd_hist.iloc[-2]

    if pd.isna(ema_curr) or pd.isna(ema_prev) or \
       pd.isna(hist_curr) or pd.isna(hist_prev):
        return "HOLD", "blue"

    ema_rising = float(ema_curr) > float(ema_prev)
    ema_falling = float(ema_curr) < float(ema_prev)
    hist_rising = float(hist_curr) > float(hist_prev)
    hist_falling = float(hist_curr) < float(hist_prev)

    if ema_rising and hist_rising:
        return "BUY", "green"
    if ema_falling and hist_falling:
        return "SELL", "red"

    return "HOLD", "blue"


# ---------------------------------------------------------------------------
# Sub-signal 7: TTM Squeeze
# ---------------------------------------------------------------------------

def _ttm_squeeze_signal(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    bb_period: int = 20,
    bb_mult: float = 2.0,
    kc_period: int = 20,
    kc_mult: float = 1.5,
) -> tuple[str, bool, float]:
    """TTM Squeeze.

    Squeeze ON  = Bollinger Bands fully inside Keltner Channels.
    Squeeze OFF = Bollinger Bands expand outside Keltner Channels.

    When squeeze releases:
      - Momentum (close - midline of Donchian/linear regression) positive = BUY
      - Momentum negative = SELL
    While squeeze is ON = HOLD.

    Momentum is approximated as the difference between the close and the
    midpoint of the highest-high and lowest-low over the same period (a
    simple Donchian midline), which closely matches TTM Squeeze momentum.

    Returns (signal, squeeze_on, momentum).
    """
    # Bollinger Bands
    bb_mid = sma(close, bb_period)
    bb_std = close.rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = bb_mid + bb_mult * bb_std
    bb_lower = bb_mid - bb_mult * bb_std

    # Keltner Channels (using ATR)
    tr = true_range(high, low, close)
    atr = ema(tr, kc_period)
    kc_mid = ema(close, kc_period)
    kc_upper = kc_mid + kc_mult * atr
    kc_lower = kc_mid - kc_mult * atr

    # Check last bar values
    vals = [bb_upper.iloc[-1], bb_lower.iloc[-1],
            kc_upper.iloc[-1], kc_lower.iloc[-1]]
    if any(pd.isna(v) for v in vals):
        return "HOLD", False, 0.0

    # Squeeze detection: BB inside KC
    squeeze_on = (float(bb_upper.iloc[-1]) < float(kc_upper.iloc[-1]) and
                  float(bb_lower.iloc[-1]) > float(kc_lower.iloc[-1]))

    # Momentum: close vs Donchian midline
    donchian_high = high.rolling(window=bb_period, min_periods=bb_period).max()
    donchian_low = low.rolling(window=bb_period, min_periods=bb_period).min()
    donchian_mid = (donchian_high + donchian_low) / 2.0
    sma_mid = sma(close, bb_period)
    # TTM-style momentum: close - average(Donchian_mid, SMA)
    momentum = close - (donchian_mid + sma_mid) / 2.0

    mom_val = momentum.iloc[-1]
    if pd.isna(mom_val):
        return "HOLD", squeeze_on, 0.0

    mom_f = float(mom_val)

    if squeeze_on:
        # Squeeze is still on -- no trade signal yet
        return "HOLD", True, mom_f

    # Squeeze released -- direction based on momentum
    if mom_f > 0:
        return "BUY", False, mom_f
    elif mom_f < 0:
        return "SELL", False, mom_f

    return "HOLD", False, mom_f



# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------

def _majority_vote(signals: list[str]) -> tuple[str, float]:
    """Majority voting across sub-signals.

    Returns (action, confidence) where confidence is the proportion of
    sub-signals agreeing with the winning direction.
    """
    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    hold_count = signals.count("HOLD")
    total = len(signals)

    if total == 0:
        return "HOLD", 0.0

    if buy_count > sell_count and buy_count > hold_count:
        return "BUY", round(buy_count / total, 4)
    if sell_count > buy_count and sell_count > hold_count:
        return "SELL", round(sell_count / total, 4)

    # Ties: BUY == SELL with both > HOLD => HOLD (conflicting)
    if buy_count == sell_count and buy_count > hold_count:
        return "HOLD", 0.0

    # HOLD wins or is tied with a directional signal
    return "HOLD", round(max(buy_count, sell_count, hold_count) / total, 4)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_heikin_ashi_signal(df: pd.DataFrame) -> dict:
    """Compute a composite Heikin-Ashi and advanced trend signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV candle data with columns: open, high, low, close, volume.
        At least 50 rows recommended for full indicator coverage.
        Minimum 10 rows required for any useful output.

    Returns
    -------
    dict
        {
            'action': 'BUY' | 'SELL' | 'HOLD',
            'confidence': 0.0 - 1.0,
            'sub_signals': {
                'ha_trend': str,
                'ha_doji': str,
                'ha_color_change': str,
                'hull_ma': str,
                'alligator': str,
                'elder_impulse': str,
                'ttm_squeeze': str,
            },
            'indicators': {
                'ha_color': str,           # 'green' | 'red'
                'ha_streak': int,          # consecutive same-color candles
                'hull_fast': float,        # HMA(9)
                'hull_slow': float,        # HMA(21)
                'alligator_lips': float,   # SMMA(5) shifted 3
                'alligator_teeth': float,  # SMMA(8) shifted 5
                'alligator_jaw': float,    # SMMA(13) shifted 8
                'elder_color': str,        # 'green' | 'red' | 'blue'
                'ttm_squeeze_on': bool,
                'ttm_momentum': float,
            },
        }

    Notes
    -----
    - Returns HOLD with 0.0 confidence when data is insufficient.
    - Sub-indicators that lack enough data individually return HOLD.
    - Uses numpy/pandas only (no TA-Lib dependency).
    """
    # Default result for early returns
    default_result = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "ha_trend": "HOLD",
            "ha_doji": "HOLD",
            "ha_color_change": "HOLD",
            "hull_ma": "HOLD",
            "alligator": "HOLD",
            "elder_impulse": "HOLD",
            "ttm_squeeze": "HOLD",
        },
        "indicators": {
            "ha_color": "green",
            "ha_streak": 0,
            "hull_fast": float("nan"),
            "hull_slow": float("nan"),
            "alligator_lips": float("nan"),
            "alligator_teeth": float("nan"),
            "alligator_jaw": float("nan"),
            "elder_color": "blue",
            "ttm_squeeze_on": False,
            "ttm_momentum": 0.0,
        },
    }

    # --- Validate input ------------------------------------------------
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return default_result

    required_cols = {"open", "high", "low", "close", "volume"}
    col_map = {c.lower(): c for c in df.columns}
    missing = required_cols - set(col_map.keys())
    if missing:
        return default_result

    if len(df) < _MIN_ROWS_BASIC:
        return default_result

    # --- Normalise columns to float (work on a copy) ------------------
    try:
        work = pd.DataFrame(
            {
                "open": df[col_map["open"]].astype(float),
                "high": df[col_map["high"]].astype(float),
                "low": df[col_map["low"]].astype(float),
                "close": df[col_map["close"]].astype(float),
                "volume": df[col_map["volume"]].astype(float),
            },
            index=df.index,
        )
    except (ValueError, TypeError):
        return default_result

    # Drop rows with NaN in OHLC
    work = work.dropna(subset=["open", "high", "low", "close"])
    if len(work) < _MIN_ROWS_BASIC:
        return default_result

    close = work["close"]
    high = work["high"]
    low = work["low"]

    # --- Compute Heikin-Ashi candles -----------------------------------
    ha = _compute_ha_candles(work)

    # --- Sub-signal 1: HA Trend ----------------------------------------
    ha_trend_sig, ha_color, ha_streak = _ha_trend_signal(ha)

    # --- Sub-signal 2: HA Doji -----------------------------------------
    ha_doji_sig = _ha_doji_signal(ha)

    # --- Sub-signal 3: HA Color Change ---------------------------------
    ha_color_change_sig = _ha_color_change_signal(ha)

    # --- Sub-signal 4: Hull MA Cross -----------------------------------
    hull_sig, hull_fast, hull_slow = _hull_ma_signal(close)

    # --- Sub-signal 5: Alligator Indicator -----------------------------
    alligator_sig, lips_val, teeth_val, jaw_val = _alligator_signal(close)

    # --- Sub-signal 6: Elder Impulse System ----------------------------
    elder_sig, elder_color = _elder_impulse_signal(close)

    # --- Sub-signal 7: TTM Squeeze -------------------------------------
    ttm_sig, squeeze_on, ttm_momentum = _ttm_squeeze_signal(high, low, close)

    # --- Assemble sub-signals and indicators ---------------------------
    sub_signals = {
        "ha_trend": ha_trend_sig,
        "ha_doji": ha_doji_sig,
        "ha_color_change": ha_color_change_sig,
        "hull_ma": hull_sig,
        "alligator": alligator_sig,
        "elder_impulse": elder_sig,
        "ttm_squeeze": ttm_sig,
    }

    indicators = {
        "ha_color": ha_color,
        "ha_streak": ha_streak,
        "hull_fast": safe_float(hull_fast),
        "hull_slow": safe_float(hull_slow),
        "alligator_lips": safe_float(lips_val),
        "alligator_teeth": safe_float(teeth_val),
        "alligator_jaw": safe_float(jaw_val),
        "elder_color": elder_color,
        "ttm_squeeze_on": squeeze_on,
        "ttm_momentum": safe_float(ttm_momentum),
    }

    # --- Majority vote -------------------------------------------------
    sub_list = list(sub_signals.values())
    action, confidence = _majority_vote(sub_list)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
