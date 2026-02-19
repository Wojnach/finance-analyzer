"""Composite candlestick pattern signal.

Detects classic candlestick patterns on the last 3 bars of OHLCV data and
returns a composite BUY/SELL/HOLD vote based on the balance of bullish vs
bearish patterns found.

Patterns detected:
  - Hammer / Inverted Hammer (bullish reversal)
  - Shooting Star / Hanging Man (bearish reversal)
  - Bullish / Bearish Engulfing
  - Doji (context-dependent)
  - Morning Star / Evening Star (3-candle reversal)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Minimum rows required for full analysis (10-bar trend context + 3 pattern
# bars).  We degrade gracefully with fewer rows but need at least 3.
# ---------------------------------------------------------------------------
MIN_ROWS_FULL = 20
MIN_ROWS_BASIC = 3

# Thresholds -- intentionally conservative to avoid over-signalling.
_BODY_SHADOW_RATIO = 2.0      # shadow must be >= 2x body for hammer-family
_DOJI_BODY_PCT = 0.10         # body < 10% of range = doji
_LARGE_BODY_PCT = 0.60        # body > 60% of range = "large" candle
_SMALL_BODY_PCT = 0.30        # body < 30% of range = "small" candle
_TREND_SLOPE_THRESHOLD = 0.0  # any nonzero slope counts; real filter is sign


def compute_candlestick_signal(df: pd.DataFrame) -> dict:
    """Analyse the last 3 bars of *df* for candlestick patterns.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns ``open, high, low, close, volume``.
        Must contain at least 3 rows; 20+ rows recommended for trend context.

    Returns
    -------
    dict
        ``action``            – ``'BUY'``, ``'SELL'``, or ``'HOLD'``
        ``confidence``        – 0.0 – 1.0
        ``sub_signals``       – per-family sub-votes
        ``patterns_detected`` – list of pattern name strings found
    """
    default = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "hammer": "HOLD",
            "engulfing": "HOLD",
            "doji": "HOLD",
            "star": "HOLD",
        },
        "patterns_detected": [],
    }

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if df is None or not isinstance(df, pd.DataFrame):
        return default

    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return default

    if len(df) < MIN_ROWS_BASIC:
        return default

    # Work on a clean copy of the tail -- avoids mutating caller's data.
    try:
        tail = df[["open", "high", "low", "close"]].copy()
        tail = tail.astype(float)
    except (ValueError, TypeError):
        return default

    # Drop rows with NaN in OHLC (keeps things robust).
    tail = tail.dropna(subset=["open", "high", "low", "close"])
    if len(tail) < MIN_ROWS_BASIC:
        return default

    # ------------------------------------------------------------------
    # Helper: trend context via linear-regression slope on close prices
    # ------------------------------------------------------------------
    trend = _detect_trend(tail)

    # ------------------------------------------------------------------
    # Pattern detection on last 3 bars
    # ------------------------------------------------------------------
    last3 = tail.iloc[-3:]
    c0, c1, c2 = (
        last3.iloc[0],  # 3rd-to-last bar
        last3.iloc[1],  # 2nd-to-last bar
        last3.iloc[2],  # most recent bar
    )

    patterns: list[str] = []
    bullish = 0
    bearish = 0

    # --- Hammer family (check last bar = c2) ---
    hammer_result = _check_hammer_family(c2, trend)
    if hammer_result == "BUY":
        bullish += 1
    elif hammer_result == "SELL":
        bearish += 1

    # --- Engulfing (check c1 -> c2) ---
    engulfing_result = _check_engulfing(c1, c2)
    if engulfing_result == "BUY":
        bullish += 1
    elif engulfing_result == "SELL":
        bearish += 1

    # --- Doji (check last bar = c2) ---
    doji_result = _check_doji(c2, trend)
    if doji_result == "BUY":
        bullish += 1
    elif doji_result == "SELL":
        bearish += 1

    # --- Morning / Evening star (check c0, c1, c2) ---
    star_result = _check_star(c0, c1, c2)
    if star_result == "BUY":
        bullish += 1
    elif star_result == "SELL":
        bearish += 1

    # Collect pattern names and map sub-signals
    _collect_pattern_names(hammer_result, engulfing_result, doji_result,
                           star_result, trend, c1, c2, c0, patterns)

    sub_signals = {
        "hammer": hammer_result,
        "engulfing": engulfing_result,
        "doji": doji_result,
        "star": star_result,
    }

    # ------------------------------------------------------------------
    # Composite vote
    # ------------------------------------------------------------------
    if bullish > bearish and bullish > 0:
        action = "BUY"
    elif bearish > bullish and bearish > 0:
        action = "SELL"
    else:
        action = "HOLD"

    total_signals = bullish + bearish
    if total_signals == 0:
        confidence = 0.0
    else:
        # Confidence scales with: (a) how many patterns agree, and
        # (b) strength of the majority.  Max realistic confidence ~ 0.85
        # (4 patterns unanimously agreeing).
        majority = max(bullish, bearish)
        confidence = round(min(majority * 0.25, 1.0), 2)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "patterns_detected": patterns,
    }


# ======================================================================
# Internal helpers
# ======================================================================

def _detect_trend(df: pd.DataFrame) -> str:
    """Return ``'up'``, ``'down'``, or ``'flat'`` based on the slope of the
    close prices over the last 10 bars (or fewer if not available)."""
    lookback = min(len(df), 10)
    closes = df["close"].iloc[-lookback:].values
    if len(closes) < 2:
        return "flat"

    x = np.arange(len(closes), dtype=float)
    # Simple linear regression slope via numpy.
    try:
        slope = np.polyfit(x, closes, 1)[0]
    except (np.linalg.LinAlgError, ValueError):
        return "flat"

    # Normalise slope relative to average price to get a dimensionless
    # measure.  A slope of > 0.1% of price per bar is "trending".
    avg_price = np.mean(closes)
    if avg_price == 0:
        return "flat"
    normalised = slope / avg_price
    if normalised > 0.001:
        return "up"
    elif normalised < -0.001:
        return "down"
    return "flat"


def _body(bar) -> float:
    """Absolute body size."""
    return abs(bar["close"] - bar["open"])


def _range(bar) -> float:
    """Total high-low range (returns tiny epsilon if zero to avoid divzero)."""
    r = bar["high"] - bar["low"]
    return r if r > 0 else 1e-12


def _upper_shadow(bar) -> float:
    return bar["high"] - max(bar["open"], bar["close"])


def _lower_shadow(bar) -> float:
    return min(bar["open"], bar["close"]) - bar["low"]


def _is_green(bar) -> bool:
    return bar["close"] >= bar["open"]


def _is_red(bar) -> bool:
    return bar["close"] < bar["open"]


def _body_pct(bar) -> float:
    """Body as a fraction of total range."""
    return _body(bar) / _range(bar)


# ------------------------------------------------------------------
# Pattern detectors -- each returns 'BUY', 'SELL', or 'HOLD'
# ------------------------------------------------------------------

def _check_hammer_family(bar, trend: str) -> str:
    """Detect hammer, inverted hammer, shooting star, hanging man on a
    single bar and return a directional vote considering trend context.
    """
    body = _body(bar)
    rng = _range(bar)
    if rng == 0 or body == 0:
        return "HOLD"

    lower = _lower_shadow(bar)
    upper = _upper_shadow(bar)
    body_ratio = body / rng

    # Filter: body must be "small" relative to range.
    if body_ratio > _SMALL_BODY_PCT:
        return "HOLD"

    # Hammer: small body near top, long lower shadow.
    if lower >= _BODY_SHADOW_RATIO * body and upper < body:
        if trend == "down":
            return "BUY"   # classic hammer after downtrend
        elif trend == "up":
            return "SELL"  # hanging man after uptrend
        return "HOLD"

    # Inverted hammer / shooting star: small body near bottom, long upper shadow.
    if upper >= _BODY_SHADOW_RATIO * body and lower < body:
        if trend == "down":
            return "BUY"   # inverted hammer after downtrend
        elif trend == "up":
            return "SELL"  # shooting star after uptrend
        return "HOLD"

    return "HOLD"


def _check_engulfing(prev, curr) -> str:
    """Detect bullish or bearish engulfing between two consecutive bars."""
    prev_body = _body(prev)
    curr_body = _body(curr)

    # Need a meaningful body on both bars.
    if prev_body == 0 or curr_body == 0:
        return "HOLD"

    prev_open = prev["open"]
    prev_close = prev["close"]
    curr_open = curr["open"]
    curr_close = curr["close"]

    # Bullish engulfing: previous red, current green, current body engulfs
    # previous body.
    if (_is_red(prev) and _is_green(curr)
            and curr_open <= prev_close and curr_close >= prev_open):
        return "BUY"

    # Bearish engulfing: previous green, current red, current body engulfs.
    if (_is_green(prev) and _is_red(curr)
            and curr_open >= prev_close and curr_close <= prev_open):
        return "SELL"

    return "HOLD"


def _check_doji(bar, trend: str) -> str:
    """Detect doji and vote based on trend context."""
    bp = _body_pct(bar)
    if bp >= _DOJI_BODY_PCT:
        return "HOLD"  # not a doji

    # Doji detected -- signal depends on trend.
    if trend == "up":
        return "SELL"  # potential reversal after uptrend
    elif trend == "down":
        return "BUY"   # potential reversal after downtrend
    return "HOLD"      # no trend = indecision, abstain


def _check_star(c0, c1, c2) -> str:
    """Detect morning star (BUY) or evening star (SELL) across 3 bars."""
    body0 = _body(c0)
    body1 = _body(c1)
    body2 = _body(c2)
    rng0 = _range(c0)
    rng2 = _range(c2)

    # First and third candles must have large bodies.
    if rng0 == 0 or rng2 == 0:
        return "HOLD"
    if body0 / rng0 < _LARGE_BODY_PCT or body2 / rng2 < _LARGE_BODY_PCT:
        return "HOLD"

    # Middle candle must have a small body.
    rng1 = _range(c1)
    if rng1 == 0:
        return "HOLD"
    if body1 / rng1 > _SMALL_BODY_PCT:
        return "HOLD"

    # Morning star: red -> small (gap down) -> green closing above midpoint
    # of first candle.
    midpoint_c0 = (c0["open"] + c0["close"]) / 2
    if (_is_red(c0) and _is_green(c2)):
        # Gap down: middle candle's body (max of open/close) below c0's close.
        mid_top = max(c1["open"], c1["close"])
        if mid_top <= c0["close"] and c2["close"] >= midpoint_c0:
            return "BUY"

    # Evening star: green -> small (gap up) -> red closing below midpoint
    # of first candle.
    if (_is_green(c0) and _is_red(c2)):
        mid_bottom = min(c1["open"], c1["close"])
        if mid_bottom >= c0["close"] and c2["close"] <= midpoint_c0:
            return "SELL"

    return "HOLD"


# ------------------------------------------------------------------
# Pattern-name collector
# ------------------------------------------------------------------

def _collect_pattern_names(hammer_result, engulfing_result, doji_result,
                           star_result, trend, c1, c2, c0, patterns):
    """Populate *patterns* list with human-readable names of detected
    patterns.  Only adds names for non-HOLD results."""

    # Hammer family names
    if hammer_result != "HOLD":
        body = _body(c2)
        lower = _lower_shadow(c2)
        upper = _upper_shadow(c2)
        if lower >= _BODY_SHADOW_RATIO * body and upper < body:
            if trend == "down":
                patterns.append("hammer")
            elif trend == "up":
                patterns.append("hanging_man")
        elif upper >= _BODY_SHADOW_RATIO * body and lower < body:
            if trend == "down":
                patterns.append("inverted_hammer")
            elif trend == "up":
                patterns.append("shooting_star")

    # Engulfing names
    if engulfing_result == "BUY":
        patterns.append("bullish_engulfing")
    elif engulfing_result == "SELL":
        patterns.append("bearish_engulfing")

    # Doji
    if doji_result != "HOLD":
        if trend == "up":
            patterns.append("bearish_doji")
        elif trend == "down":
            patterns.append("bullish_doji")

    # Star names
    if star_result == "BUY":
        patterns.append("morning_star")
    elif star_result == "SELL":
        patterns.append("evening_star")
