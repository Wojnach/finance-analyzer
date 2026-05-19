"""ConnorsRSI(2) — ultra-short-term mean reversion signal for crypto.

RSI with period=2 is extremely sensitive to 1-2 bar pullbacks. At sub-3
bar horizons, BTC and ETH mean-revert reliably after extreme moves.

Sub-indicators:
    1. RSI(2) Level   — raw extreme detection (<10 BUY, >90 SELL)
    2. RSI(2) Streak  — consecutive up/down closes amplify signal
    3. Price vs SMA(5) — short-term mean anchor for confirmation

Source: Connors & Alvarez (2008) "Short Term Trading Strategies That Work".
       Backtested: RSI(2)<5 on SPY yielded 83.4% win rate (1993-2008).

Applicable: BTC-USD, ETH-USD only (crypto assets with 24/7 liquidity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, rsi, safe_float, sma

MIN_ROWS = 10

_APPLICABLE_TICKERS = frozenset({"BTC-USD", "BTCUSDT", "ETH-USD", "ETHUSDT"})

RSI2_BUY = 10.0
RSI2_SELL = 90.0
RSI2_STRONG_BUY = 5.0
RSI2_STRONG_SELL = 95.0
STREAK_THRESHOLD = 3
SMA_PERIOD = 5
MAX_CONFIDENCE = 0.75


def _rsi2_level(close: pd.Series) -> tuple[float, str]:
    """RSI(2) extreme detection."""
    if len(close) < MIN_ROWS:
        return float("nan"), "HOLD"

    r = rsi(close, period=2)
    val = safe_float(r.iloc[-1])
    if np.isnan(val):
        return float("nan"), "HOLD"

    if val <= RSI2_STRONG_BUY:
        return val, "BUY"
    if val <= RSI2_BUY:
        return val, "BUY"
    if val >= RSI2_STRONG_SELL:
        return val, "SELL"
    if val >= RSI2_SELL:
        return val, "SELL"
    return val, "HOLD"


def _close_streak(close: pd.Series) -> tuple[float, str]:
    """Count consecutive up/down closes. 3+ down = BUY, 3+ up = SELL."""
    if len(close) < 4:
        return float("nan"), "HOLD"

    diffs = close.diff().dropna()
    last_sign = np.sign(diffs.iloc[-1])
    if last_sign == 0:
        return 0.0, "HOLD"

    streak = 0
    for val in reversed(diffs.values):
        if np.sign(val) == last_sign:
            streak += 1
        else:
            break

    streak_signed = streak * last_sign
    if streak >= STREAK_THRESHOLD and last_sign < 0:
        return safe_float(streak_signed), "BUY"
    if streak >= STREAK_THRESHOLD and last_sign > 0:
        return safe_float(streak_signed), "SELL"
    return safe_float(streak_signed), "HOLD"


def _price_vs_sma5(close: pd.Series) -> tuple[float, str]:
    """Price distance from SMA(5) as confirmation."""
    if len(close) < SMA_PERIOD + 2:
        return float("nan"), "HOLD"

    ma = sma(close, SMA_PERIOD)
    ma_val = safe_float(ma.iloc[-1])
    price = safe_float(close.iloc[-1])
    if np.isnan(ma_val) or np.isnan(price) or ma_val == 0:
        return float("nan"), "HOLD"

    pct_diff = (price - ma_val) / ma_val * 100
    pct_val = safe_float(pct_diff)

    if pct_val < -2.0:
        return pct_val, "BUY"
    if pct_val > 2.0:
        return pct_val, "SELL"
    return pct_val, "HOLD"


def compute_connors_rsi2_signal(df: pd.DataFrame, ticker: str = "",
                                **kwargs) -> dict:
    """Compute ConnorsRSI(2) signal for crypto assets.

    Returns standard signal dict with action, confidence, sub_signals.
    Non-crypto tickers get HOLD with feature_unavailable=True.
    """
    if ticker and not any(ticker.startswith(t.split("-")[0])
                          for t in _APPLICABLE_TICKERS):
        if ticker not in _APPLICABLE_TICKERS:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "feature_unavailable": True,
                "reason": "connors_rsi2 only applies to BTC/ETH",
            }

    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "reason": f"insufficient data ({len(df) if df is not None else 0} rows, need {MIN_ROWS})",
        }

    close = df["close"]

    rsi2_val, rsi2_sig = _rsi2_level(close)
    streak_val, streak_sig = _close_streak(close)
    sma5_val, sma5_sig = _price_vs_sma5(close)

    votes = [rsi2_sig, streak_sig, sma5_sig]
    action, conf = majority_vote(votes)

    if rsi2_val <= RSI2_STRONG_BUY or rsi2_val >= RSI2_STRONG_SELL:
        conf = min(conf * 1.2, 1.0)

    conf = min(conf, MAX_CONFIDENCE)

    return {
        "action": action,
        "confidence": round(conf, 4),
        "sub_signals": {
            "rsi2_level": {"value": round(rsi2_val, 2) if not np.isnan(rsi2_val) else None,
                           "signal": rsi2_sig},
            "close_streak": {"value": round(streak_val, 1) if not np.isnan(streak_val) else None,
                             "signal": streak_sig},
            "price_vs_sma5": {"value": round(sma5_val, 2) if not np.isnan(sma5_val) else None,
                              "signal": sma5_sig},
        },
    }
