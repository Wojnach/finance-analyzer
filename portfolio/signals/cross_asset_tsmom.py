"""Cross-asset time-series momentum signal.

Uses cross-asset return momentum to predict target asset direction.
Academic basis: Pitkajarvi, Suominen, Vaittinen (2020), JFE — cross-asset
filtered TSMOM yields 45% higher Sharpe than standard TSMOM.

4 sub-indicators via majority vote:
    1. Own TSMOM (252d)       — target asset's 12-month momentum
    2. Cross-pair momentum    — paired asset's 3-month momentum
    3. Bond momentum (TLT)   — 3-month treasury bond returns
    4. Equity momentum (SPY) — 3-month equity market returns

Cross-asset pairs:
    XAU-USD  -> TLT   (gold follows bonds in risk-off)
    XAG-USD  -> GC=F  (silver follows gold futures, 24h coverage)
    BTC-USD  -> SPY   (crypto correlates with risk-on)
    ETH-USD  -> BTC   (ETH follows BTC)
    MSTR     -> BTC   (MSTR is leveraged BTC)

Data: yfinance for TLT/SPY/GC=F/BTC-USD (free, no API key). Cached 1 hour.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.cross_asset_tsmom")

MIN_ROWS = 60
_CACHE_TTL = 3600

_TSMOM_LOOKBACK = 252
_CROSS_PAIR_LOOKBACK = 63
_BOND_LOOKBACK = 63
_EQUITY_LOOKBACK = 63

_CROSS_PAIRS = {
    "XAU-USD": "TLT",
    "XAG-USD": "GC=F",
    "BTC-USD": "SPY",
    "ETH-USD": "BTC-USD",
    "MSTR": "BTC-USD",
}

_YF_TICKERS = ["TLT", "SPY", "GC=F", "BTC-USD"]


def _fetch_yf_returns() -> dict[str, dict] | None:
    """Fetch 12-month daily closes for TLT, SPY, GLD, BTC-USD via yfinance.

    Returns dict mapping ticker -> {"ret_63d": float, "ret_252d": float}
    or None on failure.  Cached for 1 hour.
    """
    def _do_fetch():
        try:
            import yfinance as yf

            tickers = list(_YF_TICKERS)
            data = yf.download(tickers, period="13mo", progress=False, threads=True)
            if data is None or data.empty:
                return None

            close_col = "Close"
            if hasattr(data.columns, "levels") and len(data.columns.levels) > 1:
                close = data[close_col]
            else:
                close = data[[close_col]]
                close.columns = tickers[:1]

            result = {}
            for t in tickers:
                col = t
                if col not in close.columns:
                    continue
                series = close[col].dropna()
                if len(series) < _CROSS_PAIR_LOOKBACK + 1:
                    continue

                cur = float(series.iloc[-1])
                idx_63 = max(0, len(series) - _CROSS_PAIR_LOOKBACK - 1)
                idx_252 = max(0, len(series) - _TSMOM_LOOKBACK - 1)
                prev_63 = float(series.iloc[idx_63])
                prev_252 = float(series.iloc[idx_252])

                ret_63d = (cur / prev_63 - 1) if prev_63 > 0 else 0.0
                ret_252d = (cur / prev_252 - 1) if prev_252 > 0 else 0.0

                result[t] = {"ret_63d": ret_63d, "ret_252d": ret_252d}

            missing = set(_YF_TICKERS) - set(result)
            if missing:
                logger.warning("cross_asset_tsmom: missing tickers %s", missing)
            return result if result else None
        except Exception as e:
            logger.warning("cross_asset_tsmom yfinance fetch failed: %s", e)
            return None

    return _cached("cross_asset_tsmom_yf", _CACHE_TTL, _do_fetch)


def _compute_own_tsmom(close: pd.Series) -> str:
    """Sub-indicator 1: target asset's own 252d momentum."""
    n = len(close)
    lookback = min(_TSMOM_LOOKBACK, n - 1)
    if lookback < 20:
        return "HOLD"

    try:
        cur = float(close.iloc[-1])
        prev = float(close.iloc[-lookback - 1])
    except (TypeError, ValueError):
        return "HOLD"
    if prev <= 0 or not np.isfinite(cur) or not np.isfinite(prev):
        return "HOLD"

    ret = cur / prev - 1
    if ret > 0.005:
        return "BUY"
    if ret < -0.005:
        return "SELL"
    return "HOLD"


def _compute_cross_pair(ticker: str, yf_data: dict) -> str:
    """Sub-indicator 2: cross-asset pair's 63d momentum."""
    pair = _CROSS_PAIRS.get(ticker)
    if not pair or not yf_data:
        return "HOLD"

    pair_data = yf_data.get(pair)
    if not pair_data:
        return "HOLD"

    ret = pair_data["ret_63d"]
    if ret > 0.005:
        return "BUY"
    if ret < -0.005:
        return "SELL"
    return "HOLD"


def _compute_bond_momentum(yf_data: dict) -> str:
    """Sub-indicator 3: TLT 63d return as macro risk appetite gauge."""
    if not yf_data or "TLT" not in yf_data:
        return "HOLD"

    ret = yf_data["TLT"]["ret_63d"]
    if ret > 0.005:
        return "BUY"
    if ret < -0.005:
        return "SELL"
    return "HOLD"


def _compute_equity_momentum(yf_data: dict) -> str:
    """Sub-indicator 4: SPY 63d return as risk-on/risk-off gauge."""
    if not yf_data or "SPY" not in yf_data:
        return "HOLD"

    ret = yf_data["SPY"]["ret_63d"]
    if ret > 0.005:
        return "BUY"
    if ret < -0.005:
        return "SELL"
    return "HOLD"


def compute_cross_asset_tsmom_signal(
    df: pd.DataFrame, context: dict | None = None
) -> dict[str, Any]:
    """Compute cross-asset time-series momentum signal."""
    if df is None or len(df) < MIN_ROWS:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {},
            "indicators": {},
        }

    close = df["close"]
    ticker = (context or {}).get("ticker", "")

    yf_data = _fetch_yf_returns()

    own_tsmom = _compute_own_tsmom(close)
    cross_pair = _compute_cross_pair(ticker, yf_data)
    bond_mom = _compute_bond_momentum(yf_data)
    equity_mom = _compute_equity_momentum(yf_data)

    votes = [own_tsmom, cross_pair, bond_mom, equity_mom]
    action, confidence = majority_vote(votes, count_hold=False)

    n = len(close)
    cur = float(close.iloc[-1])
    idx_252 = max(0, n - _TSMOM_LOOKBACK - 1)
    idx_63 = max(0, n - _CROSS_PAIR_LOOKBACK - 1)
    base_252 = float(close.iloc[idx_252])
    base_63 = float(close.iloc[idx_63])
    own_ret_252 = (cur / base_252 - 1) if base_252 > 0 else 0.0
    own_ret_63 = (cur / base_63 - 1) if base_63 > 0 else 0.0

    def _yf_ret(t):
        return yf_data[t]["ret_63d"] if yf_data and t in yf_data else None

    pair_ticker = _CROSS_PAIRS.get(ticker, "none")
    pair_ret = _yf_ret(pair_ticker) if pair_ticker != "none" else None

    return {
        "action": action,
        "confidence": min(confidence, 0.7),
        "sub_signals": {
            "own_tsmom_252d": own_tsmom,
            "cross_pair_63d": cross_pair,
            "bond_momentum": bond_mom,
            "equity_momentum": equity_mom,
        },
        "indicators": {
            "own_ret_252d": safe_float(own_ret_252),
            "own_ret_63d": safe_float(own_ret_63),
            "tlt_ret_63d": safe_float(_yf_ret("TLT")),
            "spy_ret_63d": safe_float(_yf_ret("SPY")),
            "gld_ret_63d": safe_float(_yf_ret("GLD")),
            "btc_ret_63d": safe_float(_yf_ret("BTC-USD")),
            "cross_pair_ticker": pair_ticker,
            "cross_pair_ret_63d": safe_float(pair_ret),
        },
    }
