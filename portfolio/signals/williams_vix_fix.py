"""Williams VIX Fix signal module.

Synthetic volatility indicator for bottom/capitulation detection.
Calculates how far the current low has dropped from the highest close
in the lookback window, expressing it as a percentage.  Spikes signal
extreme fear/capitulation → BUY opportunity.

Sub-indicators:
    1. WVF BB Spike       — WVF exceeds upper Bollinger Band (extreme reading)
    2. WVF Percentile     — WVF in top 3% of 50-bar range (rare extreme)
    3. WVF+RSI Confirm    — WVF spike while RSI is oversold but recovering
    4. WVF Complacency    — WVF near zero for extended period → SELL (top risk)

Research basis:
    Larry Williams (2007).  QuantifiedStrategies backtest:
    profit factor 2.0, 322 trades 1993–2024, avg 0.5%/trade, max DD 23%.
    Works on any instrument — pure OHLCV, no options data needed.

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 25 rows of data (22-bar WVF lookback + 3 for shift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, rsi, safe_float, sma

MIN_ROWS = 25  # 22-bar WVF lookback + 3 for BB/RSI warmup


# ---- core WVF computation ---------------------------------------------------

def _compute_wvf(close: pd.Series, low: pd.Series,
                 lookback: int = 22) -> pd.Series:
    """Compute Williams VIX Fix.

    WVF = (highest_close_N - low) / highest_close_N * 100
    """
    highest_close = close.rolling(window=lookback, min_periods=lookback).max()
    wvf = (highest_close - low) / highest_close.replace(0, np.nan) * 100
    return wvf


# ---- sub-indicator 1: WVF Bollinger Band Spike ------------------------------

def _wvf_bb_spike(wvf: pd.Series, bb_length: int = 20,
                  bb_mult: float = 2.0) -> tuple[float, str]:
    """WVF spike above upper Bollinger Band = extreme fear = BUY.

    Returns (wvf_value, signal).
    """
    if len(wvf.dropna()) < bb_length:
        return float("nan"), "HOLD"

    wvf_sma = sma(wvf, bb_length)
    wvf_std = wvf.rolling(window=bb_length, min_periods=bb_length).std()
    upper_band = wvf_sma + bb_mult * wvf_std

    current_wvf = safe_float(wvf.iloc[-1])
    current_upper = safe_float(upper_band.iloc[-1])

    if np.isnan(current_wvf) or np.isnan(current_upper):
        return float("nan"), "HOLD"

    if current_wvf > current_upper:
        return current_wvf, "BUY"
    return current_wvf, "HOLD"


# ---- sub-indicator 2: WVF Percentile Rank -----------------------------------

def _wvf_percentile(wvf: pd.Series, pct_lookback: int = 50,
                    pct_threshold: float = 0.97) -> tuple[float, str]:
    """WVF in top 3% of recent range = rare extreme = BUY.

    Returns (percentile_value, signal).
    """
    valid = wvf.dropna()
    if len(valid) < pct_lookback:
        return float("nan"), "HOLD"

    window = valid.iloc[-pct_lookback:]
    current = safe_float(valid.iloc[-1])
    if np.isnan(current):
        return float("nan"), "HOLD"

    pct_rank = (window < current).sum() / len(window)
    pct_val = safe_float(pct_rank)

    if pct_val >= pct_threshold:
        return pct_val, "BUY"
    return pct_val, "HOLD"


# ---- sub-indicator 3: WVF + RSI Confirmation --------------------------------

def _wvf_rsi_confirm(wvf: pd.Series, close: pd.Series,
                     bb_length: int = 20, bb_mult: float = 2.0,
                     rsi_period: int = 14) -> tuple[float, str]:
    """WVF spike combined with RSI confirmation.

    BUY when WVF is elevated AND RSI is oversold (25–45 range).
    The sweet spot: fear is high (WVF spike) but RSI shows the sell-off
    is losing momentum (not deeply collapsed below 20).

    Returns (rsi_value, signal).
    """
    if len(close) < max(bb_length, rsi_period) + 5:
        return float("nan"), "HOLD"

    # Check if WVF is elevated (above 75th percentile of recent range)
    valid_wvf = wvf.dropna()
    if len(valid_wvf) < 20:
        return float("nan"), "HOLD"

    current_wvf = safe_float(valid_wvf.iloc[-1])
    wvf_median = safe_float(valid_wvf.iloc[-30:].median()) if len(valid_wvf) >= 30 else safe_float(valid_wvf.median())
    if np.isnan(current_wvf) or np.isnan(wvf_median):
        return float("nan"), "HOLD"

    wvf_elevated = current_wvf > wvf_median * 1.5

    # Check RSI
    rsi_series = rsi(close, period=rsi_period)
    rsi_val = safe_float(rsi_series.iloc[-1])
    if np.isnan(rsi_val):
        return float("nan"), "HOLD"

    # BUY: WVF elevated + RSI in oversold-but-recovering zone (25-45)
    if wvf_elevated and 25 <= rsi_val <= 45:
        return rsi_val, "BUY"
    return rsi_val, "HOLD"


# ---- sub-indicator 4: WVF Complacency (SELL) --------------------------------

def _wvf_complacency(wvf: pd.Series, close: pd.Series,
                     complacency_bars: int = 10,
                     low_wvf_threshold: float = 0.5,
                     rsi_period: int = 14) -> tuple[float, str]:
    """WVF near zero for extended period = complacency = SELL risk.

    When WVF stays very low for 10+ bars AND RSI > 70, the market
    is complacent — a correction becomes more likely.

    Returns (complacency_count, signal).
    """
    valid_wvf = wvf.dropna()
    if len(valid_wvf) < complacency_bars:
        return float("nan"), "HOLD"

    # Count consecutive bars where WVF < threshold
    recent = valid_wvf.iloc[-complacency_bars:]
    low_count = (recent < low_wvf_threshold).sum()
    low_count_val = safe_float(low_count)

    # Check RSI for overbought confirmation
    rsi_series = rsi(close, period=rsi_period)
    rsi_val = safe_float(rsi_series.iloc[-1])
    if np.isnan(rsi_val):
        return low_count_val, "HOLD"

    # SELL: extended complacency + overbought
    if low_count >= complacency_bars * 0.8 and rsi_val > 70:
        return low_count_val, "SELL"
    return low_count_val, "HOLD"


# ---- composite ---------------------------------------------------------------

def compute_williams_vix_fix_signal(
    df: pd.DataFrame, context: dict = None,
) -> dict:
    """Compute Williams VIX Fix signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        context: Optional dict with keys: ticker, config, asset_class, regime

    Returns:
        dict with keys:
            action: "BUY" | "SELL" | "HOLD"
            confidence: float 0.0–1.0
            sub_signals: dict of sub-indicator votes
            indicators: dict of raw indicator values
    """
    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    close = df["close"].astype(float)
    low = df["low"].astype(float)

    # Compute core WVF
    wvf = _compute_wvf(close, low, lookback=22)

    # Compute sub-signals
    bb_val, bb_vote = _wvf_bb_spike(wvf)
    pct_val, pct_vote = _wvf_percentile(wvf)
    rsi_val, rsi_vote = _wvf_rsi_confirm(wvf, close)
    comp_val, comp_vote = _wvf_complacency(wvf, close)

    votes = [bb_vote, pct_vote, rsi_vote, comp_vote]
    action, confidence = majority_vote(votes, count_hold=False)

    # Cap confidence at 0.7 (convention for new signals)
    confidence = min(confidence, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "wvf_bb_spike": bb_vote,
            "wvf_percentile": pct_vote,
            "wvf_rsi_confirm": rsi_vote,
            "wvf_complacency": comp_vote,
        },
        "indicators": {
            "wvf": safe_float(wvf.iloc[-1]) if len(wvf.dropna()) > 0 else float("nan"),
            "wvf_bb_val": safe_float(bb_val),
            "wvf_pct_rank": safe_float(pct_val),
            "rsi_val": safe_float(rsi_val),
            "complacency_count": safe_float(comp_val),
        },
    }
