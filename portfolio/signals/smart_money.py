"""Composite Smart Money Concepts (SMC) and market structure signal.

Combines five sub-indicators into a single BUY/SELL/HOLD vote via majority
voting:
    1. Break of Structure (BOS) - swing high/low breakouts
    2. Change of Character (CHoCH) - trend reversal detection
    3. Fair Value Gap (FVG) - unfilled 3-candle gaps being revisited
    4. Liquidity Sweep / Stop Hunt - wick-based fake-out reversals
    5. Supply and Demand Zones - institutional order flow zones

Requires a pandas DataFrame with columns: open, high, low, close, volume.
At least 50 rows recommended; returns HOLD on insufficient data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote

logger = logging.getLogger(__name__)

MIN_ROWS = 50
_SWING_LOOKBACK = 3          # bars on each side for swing detection
_FVG_SCAN_BARS = 20          # how far back to scan for unfilled FVGs
_LIQUIDITY_SWEEP_PCT = 0.005 # wick must exceed extreme by >0.5%
_STRONG_BODY_MULT = 2.0      # body > 2x avg body = strong candle
_SUPPLY_DEMAND_LOOKBACK = 30 # bars to scan for S/D zones
_ZONE_PROXIMITY_PCT = 0.005  # within 0.5% of zone boundary counts as "in zone"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _body(open_s: pd.Series, close_s: pd.Series) -> pd.Series:
    """Absolute candle body size."""
    return (close_s - open_s).abs()


def _find_swing_highs(highs: np.ndarray, lookback: int = _SWING_LOOKBACK) -> List[Tuple[int, float]]:
    """Find swing highs: bar whose high > high of `lookback` bars on each side.

    Returns list of (index, high_value) tuples sorted by index ascending.
    """
    swings: List[Tuple[int, float]] = []
    n = len(highs)
    for i in range(lookback, n - lookback):
        is_swing = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_swing = False
                break
        if is_swing:
            swings.append((i, float(highs[i])))
    return swings


def _find_swing_lows(lows: np.ndarray, lookback: int = _SWING_LOOKBACK) -> List[Tuple[int, float]]:
    """Find swing lows: bar whose low < low of `lookback` bars on each side.

    Returns list of (index, low_value) tuples sorted by index ascending.
    """
    swings: List[Tuple[int, float]] = []
    n = len(lows)
    for i in range(lookback, n - lookback):
        is_swing = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_swing = False
                break
        if is_swing:
            swings.append((i, float(lows[i])))
    return swings


# ---------------------------------------------------------------------------
# Sub-indicator 1: Break of Structure (BOS)
# ---------------------------------------------------------------------------

def _detect_bos(
    highs: np.ndarray,
    lows: np.ndarray,
    close: np.ndarray,
    swing_highs: List[Tuple[int, float]],
    swing_lows: List[Tuple[int, float]],
) -> Tuple[str, dict]:
    """Detect Break of Structure on the most recent bar.

    Bullish BOS: current close breaks above the most recent swing high.
    Bearish BOS: current close breaks below the most recent swing low.

    Returns (vote, indicators_dict).
    """
    indicators: dict = {"last_swing_high": np.nan, "last_swing_low": np.nan}

    if len(swing_highs) < 1 or len(swing_lows) < 1:
        return "HOLD", indicators

    last_sh_idx, last_sh_val = swing_highs[-1]
    last_sl_idx, last_sl_val = swing_lows[-1]
    indicators["last_swing_high"] = last_sh_val
    indicators["last_swing_low"] = last_sl_val

    current_close = float(close[-1])

    # Only count as BOS if the swing was detected *before* the current bar
    # (swing detection requires lookback bars on the right, so it is always
    # at least _SWING_LOOKBACK bars before the end).
    bullish_bos = current_close > last_sh_val
    bearish_bos = current_close < last_sl_val

    if bullish_bos and not bearish_bos:
        return "BUY", indicators
    if bearish_bos and not bullish_bos:
        return "SELL", indicators
    # Both or neither
    return "HOLD", indicators


# ---------------------------------------------------------------------------
# Sub-indicator 2: Change of Character (CHoCH)
# ---------------------------------------------------------------------------

def _detect_choch(
    swing_highs: List[Tuple[int, float]],
    swing_lows: List[Tuple[int, float]],
) -> Tuple[str, str]:
    """Detect Change of Character from swing sequence.

    Bullish CHoCH: bearish structure (lower highs + lower lows) makes a
        higher low followed by a higher high.
    Bearish CHoCH: bullish structure (higher highs + higher lows) makes a
        lower high followed by a lower low.

    Returns (vote, structure_label).
    """
    # Need at least 3 swing highs and 3 swing lows to assess structure change
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return "HOLD", "neutral"

    sh_vals = [v for _, v in swing_highs]
    sl_vals = [v for _, v in swing_lows]

    # Assess the last 3 swings for pattern
    # Previous structure (using swings [-3] and [-2])
    prev_hh = sh_vals[-2] > sh_vals[-3]  # was making higher highs
    prev_hl = sl_vals[-2] > sl_vals[-3]  # was making higher lows
    prev_lh = sh_vals[-2] < sh_vals[-3]  # was making lower highs
    prev_ll = sl_vals[-2] < sl_vals[-3]  # was making lower lows

    # Current structure change (using swings [-2] and [-1])
    curr_hh = sh_vals[-1] > sh_vals[-2]
    curr_hl = sl_vals[-1] > sl_vals[-2]
    curr_lh = sh_vals[-1] < sh_vals[-2]
    curr_ll = sl_vals[-1] < sl_vals[-2]

    # Bearish-to-bullish CHoCH: was making lower highs/lows, now higher
    # low AND higher high
    if (prev_lh or prev_ll) and curr_hl and curr_hh:
        return "BUY", "bullish"

    # Bullish-to-bearish CHoCH: was making higher highs/lows, now lower
    # high AND lower low
    if (prev_hh or prev_hl) and curr_lh and curr_ll:
        return "SELL", "bearish"

    # Determine current structure label without a change
    if curr_hh and curr_hl:
        return "HOLD", "bullish"
    if curr_lh and curr_ll:
        return "HOLD", "bearish"

    return "HOLD", "neutral"


# ---------------------------------------------------------------------------
# Sub-indicator 3: Fair Value Gap (FVG)
# ---------------------------------------------------------------------------

def _detect_fvg(
    highs: np.ndarray,
    lows: np.ndarray,
    close: np.ndarray,
    scan_bars: int = _FVG_SCAN_BARS,
) -> Tuple[str, int]:
    """Detect Fair Value Gaps and check if current price is filling one.

    Bullish FVG (gap up): candle[i+2].low > candle[i].high
        -> price dropping back into this gap = BUY (filling from above)
    Bearish FVG (gap down): candle[i+2].high < candle[i].low
        -> price rising back into this gap = SELL (filling from below)

    Returns (vote, unfilled_fvg_count).
    """
    n = len(highs)
    current_close = float(close[-1])
    unfilled_bullish: List[Tuple[float, float]] = []  # (gap_low, gap_high)
    unfilled_bearish: List[Tuple[float, float]] = []

    start = max(0, n - scan_bars - 2)

    for i in range(start, n - 2):
        candle1_high = float(highs[i])
        candle1_low = float(lows[i])
        candle3_low = float(lows[i + 2])
        candle3_high = float(highs[i + 2])

        # Bullish FVG: gap up — candle 3 low is above candle 1 high
        if candle3_low > candle1_high:
            gap_low = candle1_high
            gap_high = candle3_low
            # Check if gap has been filled by any subsequent bar
            filled = False
            for j in range(i + 3, n):
                if float(lows[j]) <= gap_low:
                    filled = True
                    break
            if not filled:
                unfilled_bullish.append((gap_low, gap_high))

        # Bearish FVG: gap down — candle 3 high is below candle 1 low
        if candle3_high < candle1_low:
            gap_high = candle1_low
            gap_low = candle3_high
            filled = False
            for j in range(i + 3, n):
                if float(highs[j]) >= gap_high:
                    filled = True
                    break
            if not filled:
                unfilled_bearish.append((gap_low, gap_high))

    total_unfilled = len(unfilled_bullish) + len(unfilled_bearish)

    # Check if current price is filling any unfilled FVG
    filling_bullish = any(
        gap_low <= current_close <= gap_high
        for gap_low, gap_high in unfilled_bullish
    )
    filling_bearish = any(
        gap_low <= current_close <= gap_high
        for gap_low, gap_high in unfilled_bearish
    )

    if filling_bullish and not filling_bearish:
        return "BUY", total_unfilled
    if filling_bearish and not filling_bullish:
        return "SELL", total_unfilled
    return "HOLD", total_unfilled


# ---------------------------------------------------------------------------
# Sub-indicator 4: Liquidity Sweep / Stop Hunt
# ---------------------------------------------------------------------------

def _detect_liquidity_sweep(
    highs: np.ndarray,
    lows: np.ndarray,
    opens: np.ndarray,
    close: np.ndarray,
    swing_highs: List[Tuple[int, float]],
    swing_lows: List[Tuple[int, float]],
    threshold_pct: float = _LIQUIDITY_SWEEP_PCT,
) -> str:
    """Detect liquidity sweeps on the most recent bar.

    Buy-side sweep (bearish): wick spikes above a recent swing high by >0.5%
        but close is back below = SELL (smart money grabbed liquidity above).
    Sell-side sweep (bullish): wick spikes below a recent swing low by >0.5%
        but close is back above = BUY (smart money grabbed liquidity below).

    Returns vote string.
    """
    if len(highs) < 2:
        return "HOLD"

    current_high = float(highs[-1])
    current_low = float(lows[-1])
    current_close = float(close[-1])
    current_open = float(opens[-1])

    # Check sell-side sweep (bullish signal): wick below recent swing low
    # then close back above it
    for _, sl_val in reversed(swing_lows[-5:]):  # check last 5 swing lows
        if sl_val <= 0:
            continue
        penetration = (sl_val - current_low) / sl_val
        if penetration > threshold_pct and current_close > sl_val:
            # Wick went below the swing low by >0.5% but closed above
            return "BUY"

    # Check buy-side sweep (bearish signal): wick above recent swing high
    # then close back below it
    for _, sh_val in reversed(swing_highs[-5:]):  # check last 5 swing highs
        if sh_val <= 0:
            continue
        penetration = (current_high - sh_val) / sh_val
        if penetration > threshold_pct and current_close < sh_val:
            # Wick went above the swing high by >0.5% but closed below
            return "SELL"

    return "HOLD"


# ---------------------------------------------------------------------------
# Sub-indicator 5: Supply and Demand Zones
# ---------------------------------------------------------------------------

def _detect_supply_demand(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    close: np.ndarray,
    lookback: int = _SUPPLY_DEMAND_LOOKBACK,
    strong_mult: float = _STRONG_BODY_MULT,
    proximity_pct: float = _ZONE_PROXIMITY_PCT,
) -> Tuple[str, bool, bool]:
    """Identify supply/demand zones and check if price is in one.

    Demand zone: base of a strong bullish candle (body > 2x avg).
        Zone spans from the candle's low to the candle's open (or close,
        whichever is lower — the base of the body).
    Supply zone: top of a strong bearish candle. Zone spans from the
        candle's close (or open, whichever is higher — the top of body)
        to the candle's high.

    Returns (vote, in_demand_zone, in_supply_zone).
    """
    n = len(close)
    if n < lookback:
        return "HOLD", False, False

    # Compute average body size over the lookback
    bodies = np.abs(close - opens)
    avg_body = np.mean(bodies[max(0, n - lookback):n])
    if avg_body <= 0:
        return "HOLD", False, False

    current_close = float(close[-1])
    demand_zones: List[Tuple[float, float]] = []  # (zone_low, zone_high)
    supply_zones: List[Tuple[float, float]] = []

    start = max(0, n - lookback)
    for i in range(start, n - 1):  # exclude current bar
        body_i = float(bodies[i])
        if body_i <= avg_body * strong_mult:
            continue  # not a strong candle

        o = float(opens[i])
        c = float(close[i])
        h = float(highs[i])
        low_i = float(lows[i])

        if c > o:
            # Bullish strong candle -> demand zone at base
            zone_low = low_i
            zone_high = o  # open is the base of a bullish body
            if zone_high > zone_low:
                demand_zones.append((zone_low, zone_high))
        elif o > c:
            # Bearish strong candle -> supply zone at top
            zone_low = o  # open is the top of a bearish body
            zone_high = h
            if zone_high > zone_low:
                supply_zones.append((zone_low, zone_high))

    # Expand zones slightly by proximity_pct to catch near-touches
    in_demand = False
    for z_low, z_high in demand_zones:
        margin = (z_high - z_low) * proximity_pct / 0.005 if z_high > z_low else 0
        # Use a small expansion: max of proximity_pct * price or a fraction
        # of zone width
        expand = max(current_close * proximity_pct, margin * 0.1)
        if (z_low - expand) <= current_close <= (z_high + expand):
            in_demand = True
            break

    in_supply = False
    for z_low, z_high in supply_zones:
        margin = (z_high - z_low) * proximity_pct / 0.005 if z_high > z_low else 0
        expand = max(current_close * proximity_pct, margin * 0.1)
        if (z_low - expand) <= current_close <= (z_high + expand):
            in_supply = True
            break

    if in_demand and not in_supply:
        return "BUY", True, False
    if in_supply and not in_demand:
        return "SELL", False, True
    return "HOLD", in_demand, in_supply


# ---------------------------------------------------------------------------
# Composite signal
# ---------------------------------------------------------------------------

def compute_smart_money_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute the composite Smart Money Concepts signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV candle data with columns ``open``, ``high``, ``low``,
        ``close``, ``volume``.  At least 50 rows recommended.

    Returns
    -------
    dict
        ``action`` (BUY / SELL / HOLD), ``confidence`` (0.0-1.0),
        ``sub_signals`` dict with individual votes, and ``indicators``
        dict with raw values.
    """
    default_result: Dict[str, Any] = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "bos": "HOLD",
            "choch": "HOLD",
            "fvg": "HOLD",
            "liquidity_sweep": "HOLD",
            "supply_demand": "HOLD",
        },
        "indicators": {
            "last_swing_high": np.nan,
            "last_swing_low": np.nan,
            "structure": "neutral",
            "unfilled_fvgs": 0,
            "in_demand_zone": False,
            "in_supply_zone": False,
        },
    }

    # --- Validate input ---------------------------------------------------
    if df is None or not isinstance(df, pd.DataFrame):
        logger.warning("smart_money: input is not a DataFrame")
        return default_result

    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("smart_money: missing columns %s", missing)
        return default_result

    if len(df) < MIN_ROWS:
        logger.info(
            "smart_money: insufficient data (%d rows, need %d)",
            len(df), MIN_ROWS,
        )
        return default_result

    # Cast to float
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    df = df.dropna(subset=list(required_cols), how="all")
    if len(df) < MIN_ROWS:
        logger.info("smart_money: too many NaN rows, only %d remain", len(df))
        return default_result

    try:
        # Extract numpy arrays for performance
        opens = df["open"].values
        highs = df["high"].values
        lows = df["low"].values
        close = df["close"].values

        # --- Swing detection (shared by BOS, CHoCH, Liquidity) ------------
        swing_highs = _find_swing_highs(highs, lookback=_SWING_LOOKBACK)
        swing_lows = _find_swing_lows(lows, lookback=_SWING_LOOKBACK)

        # --- Sub-indicator 1: Break of Structure --------------------------
        try:
            bos_vote, bos_ind = _detect_bos(
                highs, lows, close, swing_highs, swing_lows,
            )
        except Exception:
            logger.debug("smart_money: BOS detection failed", exc_info=True)
            bos_vote = "HOLD"
            bos_ind = {"last_swing_high": np.nan, "last_swing_low": np.nan}

        # --- Sub-indicator 2: Change of Character -------------------------
        try:
            choch_vote, structure_label = _detect_choch(swing_highs, swing_lows)
        except Exception:
            logger.debug("smart_money: CHoCH detection failed", exc_info=True)
            choch_vote = "HOLD"
            structure_label = "neutral"

        # --- Sub-indicator 3: Fair Value Gap ------------------------------
        try:
            fvg_vote, unfilled_count = _detect_fvg(
                highs, lows, close, scan_bars=_FVG_SCAN_BARS,
            )
        except Exception:
            logger.debug("smart_money: FVG detection failed", exc_info=True)
            fvg_vote = "HOLD"
            unfilled_count = 0

        # --- Sub-indicator 4: Liquidity Sweep -----------------------------
        try:
            sweep_vote = _detect_liquidity_sweep(
                highs, lows, opens, close,
                swing_highs, swing_lows,
                threshold_pct=_LIQUIDITY_SWEEP_PCT,
            )
        except Exception:
            logger.debug("smart_money: liquidity sweep detection failed", exc_info=True)
            sweep_vote = "HOLD"

        # --- Sub-indicator 5: Supply and Demand Zones ---------------------
        try:
            sd_vote, in_demand, in_supply = _detect_supply_demand(
                opens, highs, lows, close,
                lookback=_SUPPLY_DEMAND_LOOKBACK,
                strong_mult=_STRONG_BODY_MULT,
                proximity_pct=_ZONE_PROXIMITY_PCT,
            )
        except Exception:
            logger.debug("smart_money: S/D zone detection failed", exc_info=True)
            sd_vote = "HOLD"
            in_demand = False
            in_supply = False

        # --- Populate result ----------------------------------------------
        sub_signals = {
            "bos": bos_vote,
            "choch": choch_vote,
            "fvg": fvg_vote,
            "liquidity_sweep": sweep_vote,
            "supply_demand": sd_vote,
        }

        indicators = {
            "last_swing_high": bos_ind.get("last_swing_high", np.nan),
            "last_swing_low": bos_ind.get("last_swing_low", np.nan),
            "structure": structure_label,
            "unfilled_fvgs": unfilled_count,
            "in_demand_zone": in_demand,
            "in_supply_zone": in_supply,
        }

        # --- Majority vote ------------------------------------------------
        votes = list(sub_signals.values())
        action, confidence = majority_vote(votes)

        return {
            "action": action,
            "confidence": confidence,
            "sub_signals": sub_signals,
            "indicators": indicators,
        }

    except Exception:
        logger.exception("smart_money: unexpected error computing signal")
        return default_result
