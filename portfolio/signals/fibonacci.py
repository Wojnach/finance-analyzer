"""Composite Fibonacci and price-level signal module.

Computes 5 sub-indicators based on Fibonacci retracement/extension levels
and pivot points, returning a majority-vote composite BUY/SELL/HOLD signal
with confidence score.

Sub-indicators:
    1. Fibonacci Retracement (bounce off 50%/61.8% levels)
    2. Golden Pocket (61.8%-65% reversal zone)
    3. Fibonacci Extension (127.2%/161.8% exhaustion)
    4. Pivot Points — Standard (PP, S1/R1, S2/R2)
    5. Pivot Points — Camarilla (S3/R3 breakout levels)

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 50 rows of data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import sma

# ---------------------------------------------------------------------------
# Minimum rows required.  Fibonacci retracement uses up to 100 bars for
# swing detection; 50 is the bare minimum to get meaningful swings.
# ---------------------------------------------------------------------------
MIN_ROWS = 50

# Tolerance for "price is near a Fibonacci level" (1% by default)
_FIB_TOLERANCE = 0.01

# Lookback for swing high/low detection
_SWING_LOOKBACK = 100

# Window size for local peak/trough detection (bars on each side)
_PEAK_WINDOW = 5


def _detect_trend(close: pd.Series, period: int = 20) -> str:
    """Determine trend direction using slope of SMA(20) over last 20 bars.

    Returns 'up', 'down', or 'flat'.
    """
    if len(close) < period + 1:
        return "flat"

    sma_val = sma(close, period)
    recent_sma = sma_val.iloc[-period:].dropna()

    if len(recent_sma) < 2:
        return "flat"

    # Linear slope over the SMA values in the window
    if recent_sma.iloc[0] == 0:
        return "flat"
    slope = (recent_sma.iloc[-1] - recent_sma.iloc[0]) / recent_sma.iloc[0]

    if slope > 0.005:   # >0.5% rise over window
        return "up"
    elif slope < -0.005:
        return "down"
    return "flat"


def _find_swing_high_low(df: pd.DataFrame, lookback: int = _SWING_LOOKBACK
                         ) -> tuple[float, float, int, int]:
    """Find the most recent significant swing high and swing low.

    Uses a simple peak/trough algorithm: a swing high is a bar whose high
    is greater than the highs of the surrounding ``_PEAK_WINDOW`` bars on
    each side.  Similarly for swing low.

    Falls back to the absolute highest high / lowest low in the lookback
    window if no structural swing points are found.

    Returns (swing_high, swing_low, high_idx, low_idx) where idx values
    are positional indices into ``df``.
    """
    n = len(df)
    start = max(0, n - lookback)
    window = _PEAK_WINDOW

    high = df["high"].astype(float).values
    low = df["low"].astype(float).values

    # Find swing highs (local maxima)
    swing_highs: list[tuple[int, float]] = []
    for i in range(start + window, n - window):
        is_peak = True
        for j in range(1, window + 1):
            if high[i] <= high[i - j] or high[i] <= high[i + j]:
                is_peak = False
                break
        if is_peak:
            swing_highs.append((i, high[i]))

    # Find swing lows (local minima)
    swing_lows: list[tuple[int, float]] = []
    for i in range(start + window, n - window):
        is_trough = True
        for j in range(1, window + 1):
            if low[i] >= low[i - j] or low[i] >= low[i + j]:
                is_trough = False
                break
        if is_trough:
            swing_lows.append((i, low[i]))

    # Pick the highest swing high and lowest swing low in the window
    if swing_highs:
        best_high = max(swing_highs, key=lambda x: x[1])
        sh_val, sh_idx = best_high[1], best_high[0]
    else:
        # Fallback: absolute high in the lookback range
        segment = high[start:]
        rel_idx = int(np.argmax(segment))
        sh_idx = start + rel_idx
        sh_val = high[sh_idx]

    if swing_lows:
        best_low = min(swing_lows, key=lambda x: x[1])
        sl_val, sl_idx = best_low[1], best_low[0]
    else:
        segment = low[start:]
        rel_idx = int(np.argmin(segment))
        sl_idx = start + rel_idx
        sl_val = low[sl_idx]

    return float(sh_val), float(sl_val), sh_idx, sl_idx


def _compute_fib_levels(swing_high: float, swing_low: float) -> dict[float, float]:
    """Compute Fibonacci retracement levels between swing high and swing low.

    Levels are expressed as price values.  The ratio keys are the standard
    Fibonacci ratios (0.236, 0.382, 0.500, 0.618, 0.786).

    In an uptrend the swing_low < swing_high and the retracement levels
    measure pullback depth from the high:
        level = swing_high - ratio * (swing_high - swing_low)

    This convention is trend-agnostic — the caller determines direction.
    """
    diff = swing_high - swing_low
    return {
        0.236: swing_high - 0.236 * diff,
        0.382: swing_high - 0.382 * diff,
        0.500: swing_high - 0.500 * diff,
        0.618: swing_high - 0.618 * diff,
        0.786: swing_high - 0.786 * diff,
    }


def _compute_fib_extensions(swing_high: float, swing_low: float
                            ) -> dict[str, float]:
    """Compute Fibonacci extension levels beyond the swing range.

    Returns both upside and downside extensions:
        - ext_up_127 / ext_up_161: above swing_high
        - ext_down_127 / ext_down_161: below swing_low
    """
    diff = swing_high - swing_low
    return {
        "ext_up_127": swing_high + 0.272 * diff,    # 127.2% extension
        "ext_up_161": swing_high + 0.618 * diff,    # 161.8% extension
        "ext_down_127": swing_low - 0.272 * diff,   # 127.2% extension down
        "ext_down_161": swing_low - 0.618 * diff,   # 161.8% extension down
    }


def _near_level(price: float, level: float, tolerance: float = _FIB_TOLERANCE
                ) -> bool:
    """Check if price is within *tolerance* (fractional) of a level."""
    if level == 0:
        return False
    return abs(price - level) / abs(level) <= tolerance


# ---------------------------------------------------------------------------
# Sub-indicator votes
# ---------------------------------------------------------------------------

def _fib_retracement_signal(close: float, swing_high: float, swing_low: float,
                            fib_levels: dict[float, float], trend: str
                            ) -> tuple[str, dict]:
    """Fibonacci Retracement sub-signal.

    BUY if price bounces off 50% or 61.8% level in an uptrend pullback.
    SELL if price bounces off 50% or 61.8% level in a downtrend rally.
    HOLD otherwise.
    """
    info: dict = {
        "near_50": False,
        "near_618": False,
    }

    near_50 = _near_level(close, fib_levels[0.500])
    near_618 = _near_level(close, fib_levels[0.618])
    info["near_50"] = near_50
    info["near_618"] = near_618

    if near_50 or near_618:
        if trend == "up":
            # Uptrend pullback bouncing off support = BUY
            return "BUY", info
        elif trend == "down":
            # Downtrend rally hitting resistance = SELL
            return "SELL", info

    return "HOLD", info


def _golden_pocket_signal(close: float, swing_high: float, swing_low: float,
                          fib_levels: dict[float, float], trend: str
                          ) -> tuple[str, dict]:
    """Golden Pocket (61.8%-65% zone) sub-signal.

    The golden pocket is the zone between the 61.8% and 65% retracement
    levels — a high-probability reversal zone.

    BUY if price is in the golden pocket during an uptrend pullback.
    SELL if price is in the golden pocket during a downtrend rally.
    """
    diff = swing_high - swing_low
    gp_upper = swing_high - 0.618 * diff  # 61.8% retracement
    gp_lower = swing_high - 0.650 * diff  # 65.0% retracement

    # Ensure gp_upper > gp_lower (when swing_high > swing_low this holds)
    if gp_upper < gp_lower:
        gp_upper, gp_lower = gp_lower, gp_upper

    in_pocket = gp_lower <= close <= gp_upper

    info: dict = {
        "gp_upper": float(gp_upper),
        "gp_lower": float(gp_lower),
        "in_pocket": in_pocket,
    }

    if in_pocket:
        if trend == "up":
            return "BUY", info
        elif trend == "down":
            return "SELL", info

    return "HOLD", info


def _fib_extension_signal(close: float, swing_high: float, swing_low: float,
                          extensions: dict[str, float], trend: str
                          ) -> tuple[str, dict]:
    """Fibonacci Extension sub-signal.

    Checks whether price has pushed beyond the swing range into extension
    territory, which often signals potential exhaustion.

    Above swing high:
        - Approaching 161.8% upside extension = potential exhaustion (SELL)
    Below swing low:
        - Approaching 161.8% downside extension = potential exhaustion (BUY)

    Uses a 1.5% tolerance for "approaching".
    """
    ext_tolerance = 0.015  # slightly wider tolerance for extensions

    info: dict = {
        "above_swing_high": close > swing_high,
        "below_swing_low": close < swing_low,
        "near_ext_up_161": False,
        "near_ext_down_161": False,
    }

    # Price above the swing high — check upside extensions
    if close > swing_high:
        near_161_up = _near_level(close, extensions["ext_up_161"], ext_tolerance)
        past_161_up = close >= extensions["ext_up_161"]
        info["near_ext_up_161"] = near_161_up or past_161_up

        if near_161_up or past_161_up:
            # Exhaustion at 161.8% upside — bearish
            return "SELL", info

    # Price below the swing low — check downside extensions
    if close < swing_low:
        near_161_down = _near_level(close, extensions["ext_down_161"], ext_tolerance)
        past_161_down = close <= extensions["ext_down_161"]
        info["near_ext_down_161"] = near_161_down or past_161_down

        if near_161_down or past_161_down:
            # Exhaustion at 161.8% downside — bullish
            return "BUY", info

    return "HOLD", info


def _pivot_standard_signal(high_prev: float, low_prev: float,
                           close_prev: float, close_now: float
                           ) -> tuple[str, dict]:
    """Standard Pivot Points sub-signal.

    PP  = (H + L + C) / 3
    S1  = 2*PP - H
    R1  = 2*PP - L
    S2  = PP - (H - L)
    R2  = PP + (H - L)

    Price above R1 = BUY.  Below S1 = SELL.  Between = HOLD.
    """
    pp = (high_prev + low_prev + close_prev) / 3.0
    r1 = 2.0 * pp - low_prev
    s1 = 2.0 * pp - high_prev
    r2 = pp + (high_prev - low_prev)
    s2 = pp - (high_prev - low_prev)

    info: dict = {
        "pivot": float(pp),
        "r1": float(r1),
        "r2": float(r2),
        "s1": float(s1),
        "s2": float(s2),
    }

    if close_now > r1:
        return "BUY", info
    if close_now < s1:
        return "SELL", info
    return "HOLD", info


def _pivot_camarilla_signal(high_prev: float, low_prev: float,
                            close_prev: float, close_now: float
                            ) -> tuple[str, dict]:
    """Camarilla Pivot Points sub-signal.

    Uses Camarilla multipliers:
        R3 = C + (H - L) * 1.1 / 4
        S3 = C - (H - L) * 1.1 / 4
        R4 = C + (H - L) * 1.1 / 2   (not used for signal, for info)
        S4 = C - (H - L) * 1.1 / 2   (not used for signal, for info)

    R3/S3 are the breakout levels.
    Price above R3 = BUY (breakout).  Below S3 = SELL (breakdown).
    """
    hl_range = high_prev - low_prev
    r3 = close_prev + hl_range * 1.1 / 4.0
    s3 = close_prev - hl_range * 1.1 / 4.0
    r4 = close_prev + hl_range * 1.1 / 2.0
    s4 = close_prev - hl_range * 1.1 / 2.0

    info: dict = {
        "cam_r3": float(r3),
        "cam_s3": float(s3),
        "cam_r4": float(r4),
        "cam_s4": float(s4),
    }

    if close_now > r3:
        return "BUY", info
    if close_now < s3:
        return "SELL", info
    return "HOLD", info


# ---------------------------------------------------------------------------
# Composite signal — public API
# ---------------------------------------------------------------------------

def compute_fibonacci_signal(df: pd.DataFrame) -> dict:
    """Compute the composite Fibonacci and price-level signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV candle data with columns ``open``, ``high``, ``low``,
        ``close``, ``volume``.  At least 50 rows required; 100+ recommended
        for reliable swing detection.

    Returns
    -------
    dict
        ``action``       : ``'BUY'`` | ``'SELL'`` | ``'HOLD'``
        ``confidence``   : float 0.0-1.0 (fraction of active voters that
                           agree with the majority direction)
        ``sub_signals``  : per-indicator votes
        ``indicators``   : raw indicator values for downstream use
    """
    # -- Default / fallback result -----------------------------------------
    hold_result: dict = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "fib_retracement": "HOLD",
            "golden_pocket": "HOLD",
            "fib_extension": "HOLD",
            "pivot_standard": "HOLD",
            "pivot_camarilla": "HOLD",
        },
        "indicators": {
            "swing_high": float("nan"),
            "swing_low": float("nan"),
            "fib_levels": {},
            "pivot": float("nan"),
            "r1": float("nan"),
            "s1": float("nan"),
            "cam_r3": float("nan"),
            "cam_s3": float("nan"),
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

    current_close = float(close.iloc[-1])

    # -- Detect trend direction --------------------------------------------
    try:
        trend = _detect_trend(close, period=20)
    except Exception:
        trend = "flat"

    # -- Find swing high / low ---------------------------------------------
    try:
        swing_high, swing_low, sh_idx, sl_idx = _find_swing_high_low(df)
    except Exception:
        return hold_result

    # Sanity check: swing range must be non-zero
    if swing_high <= swing_low or swing_high == 0:
        return hold_result

    # -- Compute Fibonacci levels ------------------------------------------
    fib_levels = _compute_fib_levels(swing_high, swing_low)
    extensions = _compute_fib_extensions(swing_high, swing_low)

    # -- Previous bar for pivot calculations --------------------------------
    high_prev = float(high.iloc[-2])
    low_prev = float(low.iloc[-2])
    close_prev = float(close.iloc[-2])

    # -- Compute each sub-indicator ----------------------------------------
    sub_signals: dict[str, str] = {}
    indicators: dict = {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "trend": trend,
        "fib_levels": {str(k): round(v, 6) for k, v in fib_levels.items()},
    }

    # 1. Fibonacci Retracement
    try:
        fib_ret_action, fib_ret_info = _fib_retracement_signal(
            current_close, swing_high, swing_low, fib_levels, trend
        )
        sub_signals["fib_retracement"] = fib_ret_action
        indicators.update(fib_ret_info)
    except Exception:
        sub_signals["fib_retracement"] = "HOLD"

    # 2. Golden Pocket
    try:
        gp_action, gp_info = _golden_pocket_signal(
            current_close, swing_high, swing_low, fib_levels, trend
        )
        sub_signals["golden_pocket"] = gp_action
        indicators.update(gp_info)
    except Exception:
        sub_signals["golden_pocket"] = "HOLD"

    # 3. Fibonacci Extension
    try:
        ext_action, ext_info = _fib_extension_signal(
            current_close, swing_high, swing_low, extensions, trend
        )
        sub_signals["fib_extension"] = ext_action
        indicators.update(ext_info)
    except Exception:
        sub_signals["fib_extension"] = "HOLD"

    # 4. Standard Pivot Points
    try:
        piv_action, piv_info = _pivot_standard_signal(
            high_prev, low_prev, close_prev, current_close
        )
        sub_signals["pivot_standard"] = piv_action
        indicators.update(piv_info)
    except Exception:
        sub_signals["pivot_standard"] = "HOLD"
        indicators["pivot"] = float("nan")
        indicators["r1"] = float("nan")
        indicators["s1"] = float("nan")

    # 5. Camarilla Pivot Points
    try:
        cam_action, cam_info = _pivot_camarilla_signal(
            high_prev, low_prev, close_prev, current_close
        )
        sub_signals["pivot_camarilla"] = cam_action
        indicators.update(cam_info)
    except Exception:
        sub_signals["pivot_camarilla"] = "HOLD"
        indicators["cam_r3"] = float("nan")
        indicators["cam_s3"] = float("nan")

    # -- Majority vote -----------------------------------------------------
    votes = list(sub_signals.values())
    buy_count = votes.count("BUY")
    sell_count = votes.count("SELL")
    active_votes = buy_count + sell_count  # non-HOLD votes

    if active_votes == 0:
        action = "HOLD"
        confidence = 0.0
    elif buy_count > sell_count:
        action = "BUY"
        confidence = buy_count / active_votes
    elif sell_count > buy_count:
        action = "SELL"
        confidence = sell_count / active_votes
    else:
        # Tied between BUY and SELL — no clear direction
        action = "HOLD"
        confidence = 0.0

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
