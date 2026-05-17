"""TSI + Choppiness Index regime-adaptive mean reversion signal.

Combines the True Strength Index (double-smoothed momentum oscillator) with
the Choppiness Index (ATR/range ratio) as a regime gate. In ranging markets,
TSI extremes generate mean-reversion signals.  In trending markets, the signal
is suppressed to HOLD to avoid fighting strong directional moves.

Sub-indicators:
    1. TSI Extreme MR      (TSI < -25 BUY, > 25 SELL — ranging only)
    2. TSI Signal Cross     (TSI crossing its 7-EMA signal line)
    3. TSI Divergence       (price vs TSI divergence)
    4. CHOP Regime Gate     (< 38.2 trending → HOLD, > 55 ranging → full confidence)

Academic basis:
    - Blau (1991), True Strength Index — double-smoothed momentum.
    - Dreiss (1993), Choppiness Index — ATR/range regime classifier.
    - Requejo (2024), SSRN 4708400 — TSI MR on SPY/QQQ, 1996-2023.
    - FMZQuant (2024), dual-regime MR with ADX gate — 71% WR ranging mode.

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 40 rows of data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import ema, majority_vote, safe_float, true_range

MIN_ROWS = 40


def _compute_tsi(close: pd.Series, long_period: int = 25,
                 short_period: int = 13) -> pd.Series:
    """True Strength Index: 100 * EMA(EMA(delta, long), short) / EMA(EMA(|delta|, long), short)."""
    delta = close.diff()
    smooth1 = ema(delta, long_period)
    double_smooth = ema(smooth1, short_period)
    abs_smooth1 = ema(delta.abs(), long_period)
    abs_double_smooth = ema(abs_smooth1, short_period)
    denom = abs_double_smooth.replace(0, np.nan)
    return 100.0 * double_smooth / denom


def _compute_chop(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
    """Choppiness Index: 100 * log10(sum(TR, n) / (HH - LL)) / log10(n)."""
    tr = true_range(high, low, close)
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    hh = high.rolling(window=period, min_periods=period).max()
    ll = low.rolling(window=period, min_periods=period).min()
    price_range = (hh - ll).replace(0, np.nan)
    return 100.0 * np.log10(tr_sum / price_range) / np.log10(period)


def _tsi_extreme(tsi_val: float, buy_threshold: float = -25.0,
                 sell_threshold: float = 25.0) -> str:
    """MR signal from TSI extremes."""
    if np.isnan(tsi_val):
        return "HOLD"
    if tsi_val < buy_threshold:
        return "BUY"
    if tsi_val > sell_threshold:
        return "SELL"
    return "HOLD"


def _tsi_signal_cross(tsi_series: pd.Series, signal_period: int = 7) -> str:
    """TSI crossing its signal line (EMA of TSI)."""
    if len(tsi_series.dropna()) < signal_period + 2:
        return "HOLD"
    signal_line = ema(tsi_series, signal_period)
    curr_tsi = tsi_series.iloc[-1]
    prev_tsi = tsi_series.iloc[-2]
    curr_sig = signal_line.iloc[-1]
    prev_sig = signal_line.iloc[-2]
    if np.isnan(curr_tsi) or np.isnan(curr_sig) or np.isnan(prev_tsi) or np.isnan(prev_sig):
        return "HOLD"
    if prev_tsi <= prev_sig and curr_tsi > curr_sig:
        return "BUY"
    if prev_tsi >= prev_sig and curr_tsi < curr_sig:
        return "SELL"
    return "HOLD"


def _tsi_divergence(close: pd.Series, tsi_series: pd.Series,
                    lookback: int = 14) -> str:
    """Detect bullish/bearish divergence between price and TSI."""
    if len(close) < lookback + 1 or len(tsi_series.dropna()) < lookback + 1:
        return "HOLD"
    recent_close = close.iloc[-lookback:]
    recent_tsi = tsi_series.iloc[-lookback:]
    if recent_tsi.isna().any():
        return "HOLD"
    close_min_idx = recent_close.idxmin()
    close_max_idx = recent_close.idxmax()
    price_at_end = close.iloc[-1]
    tsi_at_end = tsi_series.iloc[-1]
    price_at_min = close.loc[close_min_idx]
    tsi_at_min = tsi_series.loc[close_min_idx]
    price_at_max = close.loc[close_max_idx]
    tsi_at_max = tsi_series.loc[close_max_idx]
    if price_at_end <= price_at_min * 1.005 and tsi_at_end > tsi_at_min + 3:
        return "BUY"
    if price_at_end >= price_at_max * 0.995 and tsi_at_end < tsi_at_max - 3:
        return "SELL"
    return "HOLD"


def compute_tsi_chop_mr_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute TSI + Choppiness regime-adaptive mean reversion signal."""
    if df is None or len(df) < MIN_ROWS:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {},
                "indicators": {}}

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    tsi_series = _compute_tsi(close)
    chop_series = _compute_chop(high, low, close)

    tsi_val = safe_float(tsi_series.iloc[-1])
    chop_val = safe_float(chop_series.iloc[-1])

    if np.isnan(tsi_val) or np.isnan(chop_val):
        return {"action": "HOLD", "confidence": 0.0,
                "sub_signals": {}, "indicators": {"tsi": tsi_val, "chop": chop_val}}

    tsi_signal_line = ema(tsi_series, 7)
    tsi_sig_val = safe_float(tsi_signal_line.iloc[-1])

    is_trending = chop_val < 38.2
    is_ranging = chop_val >= 55.0
    is_transition = not is_trending and not is_ranging

    if is_trending:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {
                "tsi_extreme": "HOLD (trending)",
                "tsi_signal_cross": "HOLD (trending)",
                "tsi_divergence": "HOLD (trending)",
                "chop_regime": "trending",
            },
            "indicators": {
                "tsi": tsi_val,
                "tsi_signal": tsi_sig_val,
                "chop": chop_val,
                "regime": "trending",
            },
        }

    extreme_vote = _tsi_extreme(tsi_val)
    cross_vote = _tsi_signal_cross(tsi_series)
    div_vote = _tsi_divergence(close, tsi_series)

    votes = [extreme_vote, cross_vote, div_vote]
    action, raw_confidence = majority_vote(votes, count_hold=False)

    chop_multiplier = 1.0 if is_ranging else 0.65
    confidence = min(round(raw_confidence * chop_multiplier, 4), 0.7)

    regime_label = "ranging" if is_ranging else "transition"

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "tsi_extreme": extreme_vote,
            "tsi_signal_cross": cross_vote,
            "tsi_divergence": div_vote,
            "chop_regime": regime_label,
        },
        "indicators": {
            "tsi": tsi_val,
            "tsi_signal": tsi_sig_val,
            "chop": chop_val,
            "regime": regime_label,
        },
    }
