"""Cross-asset equity trend spillover signal.

SPY/QQQ technical indicators predict commodity, crypto, and equity returns.
Academic basis: Fieberg et al. (2025), "Cross-Asset Trend Spillover" —
robust across 1.3M research designs with 20 countries of data.

Mechanism: equity market trend state (via RSI, MACD, price vs EMA) spills
over into non-equity assets.  For risk-on assets (BTC, ETH, MSTR), bullish
equities => BUY.  For safe-haven assets (XAU, XAG), bullish equities =>
SELL (risk-on reduces safe-haven demand).

4 sub-indicators via majority vote:
    1. SPY RSI(14)      — overbought/oversold
    2. SPY MACD hist    — momentum direction
    3. QQQ RSI(14)      — tech sector confirmation
    4. SPY trend        — price vs EMA(50)

Data: yfinance for SPY and QQQ (free, no API key).  Cached 1 hour via
shared_state._cached() to avoid rate limits.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.xtrend_equity_spillover")

MIN_ROWS = 20
_CACHE_TTL = 3600  # 1 hour

_RSI_PERIOD = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_EMA_TREND = 50
_RSI_OVERBOUGHT = 65  # Slightly looser than classic 70 — captures trend continuation
_RSI_OVERSOLD = 35

# Safe-haven assets get INVERTED signal: bullish equities = risk-on = SELL safe havens
_SAFE_HAVEN_TICKERS = {"XAU-USD", "XAG-USD"}


def _fetch_equity_data() -> dict | None:
    """Fetch SPY and QQQ daily OHLCV via yfinance (90 days).

    Returns dict with RSI, MACD hist, trend for each, or None on failure.
    Cached for 1 hour.
    """
    def _do_fetch():
        try:
            import yfinance as yf

            data = yf.download(
                ["SPY", "QQQ"], period="6mo", progress=False, threads=True,
            )
            if data is None or data.empty:
                return None

            result = {}
            for ticker in ("SPY", "QQQ"):
                try:
                    if hasattr(data.columns, "levels") and len(data.columns.levels) > 1:
                        close = data["Close"][ticker].dropna()
                    else:
                        close = data["Close"].dropna()
                except (KeyError, TypeError):
                    continue

                if len(close) < _EMA_TREND + 10:
                    continue

                close_series = close.astype(float)

                # RSI(14)
                rsi_val = _compute_rsi(close_series, _RSI_PERIOD)

                # MACD histogram
                macd_hist = _compute_macd_hist(close_series)

                # Trend: price vs EMA(50)
                ema_50 = close_series.ewm(span=_EMA_TREND, adjust=False).mean()
                above_ema = float(close_series.iloc[-1]) > float(ema_50.iloc[-1])

                # 5d momentum
                if len(close_series) >= 6:
                    mom_5d = float(close_series.iloc[-1]) / float(close_series.iloc[-6]) - 1
                else:
                    mom_5d = 0.0

                result[ticker] = {
                    "rsi": rsi_val,
                    "macd_hist": macd_hist,
                    "above_ema50": above_ema,
                    "mom_5d": mom_5d,
                    "close": float(close_series.iloc[-1]),
                }

            return result if result else None
        except Exception as e:
            logger.warning("xtrend_equity_spillover: yfinance fetch failed: %s", e)
            return None

    return _cached("xtrend_equity_spillover_yf", _CACHE_TTL, _do_fetch)


def _compute_rsi(close: pd.Series, period: int = 14) -> float:
    """Compute last RSI value from a close series."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - 100 / (1 + rs)
    val = rsi_series.iloc[-1]
    return float(val) if np.isfinite(val) else 50.0


def _compute_macd_hist(close: pd.Series) -> float:
    """Compute last MACD histogram value."""
    ema_fast = close.ewm(span=_MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=_MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=_MACD_SIGNAL, adjust=False).mean()
    hist = macd_line - signal_line
    val = hist.iloc[-1]
    return float(val) if np.isfinite(val) else 0.0


def _sub_spy_rsi(equity_data: dict) -> str:
    """Sub-indicator 1: SPY RSI(14) direction."""
    spy = equity_data.get("SPY")
    if not spy:
        return "HOLD"
    rsi_val = spy["rsi"]
    if rsi_val < _RSI_OVERSOLD:
        return "BUY"
    if rsi_val > _RSI_OVERBOUGHT:
        return "SELL"
    return "HOLD"


def _sub_spy_macd(equity_data: dict) -> str:
    """Sub-indicator 2: SPY MACD histogram direction."""
    spy = equity_data.get("SPY")
    if not spy:
        return "HOLD"
    hist = spy["macd_hist"]
    if hist > 0.5:
        return "BUY"
    if hist < -0.5:
        return "SELL"
    return "HOLD"


def _sub_qqq_rsi(equity_data: dict) -> str:
    """Sub-indicator 3: QQQ RSI(14) for tech sector confirmation."""
    qqq = equity_data.get("QQQ")
    if not qqq:
        return "HOLD"
    rsi_val = qqq["rsi"]
    if rsi_val < _RSI_OVERSOLD:
        return "BUY"
    if rsi_val > _RSI_OVERBOUGHT:
        return "SELL"
    return "HOLD"


def _sub_spy_trend(equity_data: dict) -> str:
    """Sub-indicator 4: SPY price vs EMA(50) trend."""
    spy = equity_data.get("SPY")
    if not spy:
        return "HOLD"
    if spy["above_ema50"] and spy["mom_5d"] > 0.005:
        return "BUY"
    if not spy["above_ema50"] and spy["mom_5d"] < -0.005:
        return "SELL"
    return "HOLD"


def compute_xtrend_equity_spillover_signal(
    df: pd.DataFrame, context: dict | None = None,
) -> dict[str, Any]:
    """Compute cross-asset equity trend spillover signal.

    Args:
        df: OHLCV DataFrame for target asset (used for MIN_ROWS check only).
        context: dict with ``ticker``, ``asset_class`` keys.

    Returns:
        dict with action, confidence, sub_signals, indicators.
    """
    empty = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {},
        "indicators": {},
    }

    if df is None or len(df) < MIN_ROWS:
        return empty

    context = context or {}
    ticker = context.get("ticker", "")

    # Fetch SPY/QQQ data (cached 1 hour)
    equity_data = _fetch_equity_data()
    if not equity_data:
        return empty

    # Compute 4 sub-indicators (equity perspective)
    spy_rsi_vote = _sub_spy_rsi(equity_data)
    spy_macd_vote = _sub_spy_macd(equity_data)
    qqq_rsi_vote = _sub_qqq_rsi(equity_data)
    spy_trend_vote = _sub_spy_trend(equity_data)

    votes = [spy_rsi_vote, spy_macd_vote, qqq_rsi_vote, spy_trend_vote]
    action, confidence = majority_vote(votes, count_hold=False)

    # For safe-haven assets: INVERT the signal
    # Bullish equities = risk-on = bearish for gold/silver
    if ticker in _SAFE_HAVEN_TICKERS and action != "HOLD":
        action = "SELL" if action == "BUY" else "BUY"

    # Build indicators dict
    spy = equity_data.get("SPY", {})
    qqq = equity_data.get("QQQ", {})
    indicators = {
        "spy_rsi": safe_float(spy.get("rsi", 0)),
        "spy_macd_hist": safe_float(spy.get("macd_hist", 0)),
        "spy_above_ema50": 1.0 if spy.get("above_ema50") else 0.0,
        "spy_mom_5d": safe_float(spy.get("mom_5d", 0)),
        "qqq_rsi": safe_float(qqq.get("rsi", 0)),
        "inverted": 1.0 if ticker in _SAFE_HAVEN_TICKERS else 0.0,
    }

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "spy_rsi": spy_rsi_vote,
            "spy_macd": spy_macd_vote,
            "qqq_rsi": qqq_rsi_vote,
            "spy_trend": spy_trend_vote,
        },
        "indicators": indicators,
    }
