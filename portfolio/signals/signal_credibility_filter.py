"""Signal credibility filter — meta-signal distinguishing genuine vs noise moves.

Three sub-signals assess whether recent price action is credible:
  1. Persistence ratio   — fraction of initial move sustained after lag bars
  2. Volume distribution — HHI of per-bar volume shares (low = genuine, high = manipulative)
  3. Follow-through      — whether the move continued beyond the initial impulse

When credibility is high, confirms the move direction.
When credibility is low, fades the move (contrarian).

Source: Nechepurenko 2026, arxiv:2604.27041 (Signal Credibility Index).

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 25 rows of data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote

MIN_ROWS = 25
_MAX_CONFIDENCE = 0.7

_PERSISTENCE_LAG = 5
_HHI_WINDOW = 10
_LOOKBACK = 20
_CREDIBILITY_HIGH = 0.55
_CREDIBILITY_LOW = 0.25
_MIN_MOVE_ATR = 0.3


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def _persistence_vote(close: pd.Series) -> tuple[str, dict]:
    if len(close) < _LOOKBACK + _PERSISTENCE_LAG:
        return "HOLD", {}

    initial_move = float(close.iloc[-_PERSISTENCE_LAG] - close.iloc[-_LOOKBACK])
    current_move = float(close.iloc[-1] - close.iloc[-_LOOKBACK])

    if abs(initial_move) < 1e-10:
        return "HOLD", {"persistence_ratio": 0.0}

    ratio = current_move / initial_move
    ratio = max(min(ratio, 3.0), -1.0)

    indicators = {"persistence_ratio": round(ratio, 4)}

    if ratio > 0.7:
        direction = "BUY" if current_move > 0 else "SELL"
        return direction, indicators
    elif ratio < 0.2:
        direction = "SELL" if initial_move > 0 else "BUY"
        return direction, indicators

    return "HOLD", indicators


def _volume_hhi_vote(volume: pd.Series, close: pd.Series) -> tuple[str, dict]:
    if len(volume) < _HHI_WINDOW + 1:
        return "HOLD", {}

    window_vol = volume.iloc[-_HHI_WINDOW:]
    total = window_vol.sum()
    if total <= 0:
        return "HOLD", {"volume_hhi": 0.0}

    shares = window_vol / total
    hhi = float((shares ** 2).sum())
    hhi_uniform = 1.0 / _HHI_WINDOW
    hhi_normalized = (hhi - hhi_uniform) / (1.0 - hhi_uniform) if (1.0 - hhi_uniform) > 1e-10 else 0.0
    hhi_normalized = max(0.0, min(1.0, hhi_normalized))

    recent_direction = float(close.iloc[-1] - close.iloc[-_HHI_WINDOW])

    indicators = {"volume_hhi": round(hhi, 4), "hhi_normalized": round(hhi_normalized, 4)}

    if hhi_normalized > 0.5:
        direction = "SELL" if recent_direction > 0 else "BUY"
        return direction, indicators
    elif hhi_normalized < 0.15:
        direction = "BUY" if recent_direction > 0 else "SELL"
        return direction, indicators

    return "HOLD", indicators


def _follow_through_vote(close: pd.Series) -> tuple[str, dict]:
    if len(close) < _PERSISTENCE_LAG * 2 + 1:
        return "HOLD", {}

    move_1 = float(close.iloc[-_PERSISTENCE_LAG] - close.iloc[-_PERSISTENCE_LAG * 2])
    move_2 = float(close.iloc[-1] - close.iloc[-_PERSISTENCE_LAG])

    if abs(move_1) < 1e-10:
        return "HOLD", {"follow_through": 0.0}

    ft_ratio = move_2 / abs(move_1)
    ft_ratio = max(min(ft_ratio, 3.0), -3.0)

    indicators = {"follow_through": round(ft_ratio, 4)}

    if ft_ratio > 0.3:
        direction = "BUY" if move_2 > 0 else "SELL"
        return direction, indicators
    elif ft_ratio < -0.3:
        direction = "SELL" if move_1 > 0 else "BUY"
        return direction, indicators

    return "HOLD", indicators


def compute_signal_credibility_filter_signal(
    df: pd.DataFrame, context: dict = None
) -> dict:
    empty = {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    if df is None or len(df) < MIN_ROWS:
        return empty

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    if close.isna().sum() > len(close) * 0.2:
        return empty

    close = close.ffill()
    volume = volume.fillna(0)

    atr_series = _atr(high, low, close, period=14)
    current_atr = atr_series.iloc[-1] if not np.isnan(atr_series.iloc[-1]) else 0
    recent_move = abs(float(close.iloc[-1] - close.iloc[-_LOOKBACK])) if len(close) >= _LOOKBACK else 0

    if current_atr <= 0 or recent_move / current_atr < _MIN_MOVE_ATR:
        return empty

    p_vote, p_ind = _persistence_vote(close)
    h_vote, h_ind = _volume_hhi_vote(volume, close)
    f_vote, f_ind = _follow_through_vote(close)

    votes = [p_vote, h_vote, f_vote]
    action, confidence = majority_vote(votes, count_hold=False)
    confidence = min(confidence, _MAX_CONFIDENCE)

    sub_signals = {
        "persistence": p_vote,
        "volume_distribution": h_vote,
        "follow_through": f_vote,
    }
    indicators = {}
    indicators.update(p_ind)
    indicators.update(h_ind)
    indicators.update(f_ind)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
