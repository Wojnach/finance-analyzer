"""Composite volume flow signal.

Combines six volume-based sub-indicators into a single BUY/SELL/HOLD vote
via majority voting:
    1. On-Balance Volume (OBV) vs its 20-period SMA
    2. VWAP Cross (price vs session VWAP)
    3. Accumulation/Distribution Line vs its 20-period SMA
    4. Chaikin Money Flow (20-period)
    5. Money Flow Index (14-period)
    6. Volume RSI (14-period)

Requires a pandas DataFrame with columns: open, high, low, close, volume.
At least 50 rows recommended; returns HOLD on insufficient data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_ROWS = 50


def _safe_series(series: pd.Series) -> pd.Series:
    """Replace inf/-inf with NaN and forward-fill."""
    return series.replace([np.inf, -np.inf], np.nan).ffill()


# ---------------------------------------------------------------------------
# Sub-indicator calculations
# ---------------------------------------------------------------------------

def _compute_obv(close: pd.Series, volume: pd.Series) -> tuple[pd.Series, pd.Series]:
    """On-Balance Volume and its 20-period SMA."""
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()
    obv_sma = obv.rolling(window=20, min_periods=20).mean()
    return obv, obv_sma


def _compute_vwap(high: pd.Series, low: pd.Series, close: pd.Series,
                  volume: pd.Series) -> pd.Series:
    """Session VWAP: cumulative(volume * typical_price) / cumulative(volume)."""
    typical_price = (high + low + close) / 3.0
    cum_vol = volume.cumsum()
    cum_vp = (volume * typical_price).cumsum()
    # Avoid division by zero
    vwap = cum_vp / cum_vol.replace(0, np.nan)
    return _safe_series(vwap)


def _compute_ad_line(high: pd.Series, low: pd.Series, close: pd.Series,
                     volume: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Accumulation/Distribution Line and its 20-period SMA."""
    hl_range = high - low
    # Avoid division by zero when high == low (doji candles)
    clv = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
    clv = clv.fillna(0.0)
    ad = (clv * volume).cumsum()
    ad_sma = ad.rolling(window=20, min_periods=20).mean()
    return ad, ad_sma


def _compute_cmf(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow over *period* bars."""
    hl_range = high - low
    clv = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
    clv = clv.fillna(0.0)
    mf_volume = clv * volume
    cmf = mf_volume.rolling(window=period, min_periods=period).sum() / \
          volume.rolling(window=period, min_periods=period).sum().replace(0, np.nan)
    return _safe_series(cmf)


def _compute_mfi(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index (volume-weighted RSI)."""
    typical_price = (high + low + close) / 3.0
    raw_mf = typical_price * volume
    tp_diff = typical_price.diff()

    pos_mf = pd.Series(np.where(tp_diff > 0, raw_mf, 0.0), index=raw_mf.index)
    neg_mf = pd.Series(np.where(tp_diff < 0, raw_mf, 0.0), index=raw_mf.index)

    pos_sum = pos_mf.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(window=period, min_periods=period).sum()

    mf_ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + mf_ratio))
    return _safe_series(mfi)


def _compute_volume_rsi(volume: pd.Series, period: int = 14) -> pd.Series:
    """RSI applied to volume rather than price."""
    delta = volume.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    vrsi = 100.0 - (100.0 / (1.0 + rs))
    return _safe_series(vrsi)


# ---------------------------------------------------------------------------
# Sub-signal voting
# ---------------------------------------------------------------------------

def _vote_obv(obv_val: float, obv_sma_val: float) -> str:
    if np.isnan(obv_val) or np.isnan(obv_sma_val):
        return "HOLD"
    if obv_val > obv_sma_val:
        return "BUY"
    if obv_val < obv_sma_val:
        return "SELL"
    return "HOLD"


def _vote_vwap(close_val: float, vwap_val: float) -> str:
    if np.isnan(close_val) or np.isnan(vwap_val):
        return "HOLD"
    if close_val > vwap_val:
        return "BUY"
    if close_val < vwap_val:
        return "SELL"
    return "HOLD"


def _vote_ad(ad_val: float, ad_sma_val: float) -> str:
    if np.isnan(ad_val) or np.isnan(ad_sma_val):
        return "HOLD"
    if ad_val > ad_sma_val:
        return "BUY"
    if ad_val < ad_sma_val:
        return "SELL"
    return "HOLD"


def _vote_cmf(cmf_val: float) -> str:
    if np.isnan(cmf_val):
        return "HOLD"
    if cmf_val > 0.05:
        return "BUY"
    if cmf_val < -0.05:
        return "SELL"
    return "HOLD"


def _vote_mfi(mfi_val: float) -> str:
    if np.isnan(mfi_val):
        return "HOLD"
    if mfi_val < 20.0:
        return "BUY"
    if mfi_val > 80.0:
        return "SELL"
    return "HOLD"


def _vote_volume_rsi(vrsi_val: float, price_up: bool) -> str:
    """Volume RSI > 70 with price up = BUY (strong buying pressure).
    Volume RSI > 70 with price down = SELL (strong selling pressure).
    Otherwise HOLD."""
    if np.isnan(vrsi_val):
        return "HOLD"
    if vrsi_val > 70.0:
        return "BUY" if price_up else "SELL"
    return "HOLD"


# ---------------------------------------------------------------------------
# Composite signal
# ---------------------------------------------------------------------------

def compute_volume_flow_signal(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute composite volume flow signal from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: open, high, low, close, volume.
        At least 50 rows recommended.

    Returns
    -------
    dict with keys:
        action : str          'BUY', 'SELL', or 'HOLD'
        confidence : float    0.0 - 1.0 (proportion agreeing with majority)
        sub_signals : dict    Per-sub-indicator vote
        indicators : dict     Raw indicator values (latest bar)
    """
    default_result: Dict[str, Any] = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "obv": "HOLD",
            "vwap": "HOLD",
            "ad_line": "HOLD",
            "cmf": "HOLD",
            "mfi": "HOLD",
            "volume_rsi": "HOLD",
        },
        "indicators": {
            "obv": np.nan,
            "obv_sma": np.nan,
            "vwap": np.nan,
            "ad_line": np.nan,
            "ad_sma": np.nan,
            "cmf": np.nan,
            "mfi": np.nan,
            "volume_rsi": np.nan,
        },
    }

    # --- Validate input ------------------------------------------------
    required_cols = {"open", "high", "low", "close", "volume"}
    if df is None or not isinstance(df, pd.DataFrame):
        logger.warning("volume_flow: input is not a DataFrame")
        return default_result

    # Normalise column names to lowercase for robustness
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("volume_flow: missing columns %s", missing)
        return default_result

    if len(df) < MIN_ROWS:
        logger.info("volume_flow: insufficient data (%d rows, need %d)", len(df), MIN_ROWS)
        return default_result

    # Cast to float to avoid integer overflow / dtype surprises
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # Drop rows where all OHLCV are NaN
    df = df.dropna(subset=list(required_cols), how="all")
    if len(df) < MIN_ROWS:
        logger.info("volume_flow: too many NaN rows, only %d remain", len(df))
        return default_result

    try:
        # --- Compute indicators ----------------------------------------
        obv, obv_sma = _compute_obv(df["close"], df["volume"])
        vwap = _compute_vwap(df["high"], df["low"], df["close"], df["volume"])
        ad, ad_sma = _compute_ad_line(df["high"], df["low"], df["close"], df["volume"])
        cmf = _compute_cmf(df["high"], df["low"], df["close"], df["volume"], period=20)
        mfi = _compute_mfi(df["high"], df["low"], df["close"], df["volume"], period=14)
        vrsi = _compute_volume_rsi(df["volume"], period=14)

        # Latest values
        obv_val = float(obv.iloc[-1])
        obv_sma_val = float(obv_sma.iloc[-1]) if not np.isnan(obv_sma.iloc[-1]) else np.nan
        vwap_val = float(vwap.iloc[-1])
        ad_val = float(ad.iloc[-1])
        ad_sma_val = float(ad_sma.iloc[-1]) if not np.isnan(ad_sma.iloc[-1]) else np.nan
        cmf_val = float(cmf.iloc[-1])
        mfi_val = float(mfi.iloc[-1])
        vrsi_val = float(vrsi.iloc[-1])
        close_val = float(df["close"].iloc[-1])

        # Determine price direction for volume RSI vote
        # Use 1-bar change; fall back to neutral if insufficient data
        price_change = df["close"].diff().iloc[-1]
        price_up = price_change > 0 if not np.isnan(price_change) else True  # default neutral bias

        # --- Sub-signal votes ------------------------------------------
        sub_signals = {
            "obv": _vote_obv(obv_val, obv_sma_val),
            "vwap": _vote_vwap(close_val, vwap_val),
            "ad_line": _vote_ad(ad_val, ad_sma_val),
            "cmf": _vote_cmf(cmf_val),
            "mfi": _vote_mfi(mfi_val),
            "volume_rsi": _vote_volume_rsi(vrsi_val, price_up),
        }

        indicators = {
            "obv": obv_val,
            "obv_sma": obv_sma_val,
            "vwap": vwap_val,
            "ad_line": ad_val,
            "ad_sma": ad_sma_val,
            "cmf": cmf_val,
            "mfi": mfi_val,
            "volume_rsi": vrsi_val,
        }

        # --- Majority vote ---------------------------------------------
        votes = list(sub_signals.values())
        buy_count = votes.count("BUY")
        sell_count = votes.count("SELL")
        hold_count = votes.count("HOLD")
        total = len(votes)  # always 6

        if buy_count > sell_count and buy_count > hold_count:
            action = "BUY"
            confidence = buy_count / total
        elif sell_count > buy_count and sell_count > hold_count:
            action = "SELL"
            confidence = sell_count / total
        else:
            # Tie or HOLD majority -> HOLD
            action = "HOLD"
            confidence = hold_count / total

        return {
            "action": action,
            "confidence": round(confidence, 4),
            "sub_signals": sub_signals,
            "indicators": indicators,
        }

    except Exception:
        logger.exception("volume_flow: unexpected error computing signal")
        return default_result
