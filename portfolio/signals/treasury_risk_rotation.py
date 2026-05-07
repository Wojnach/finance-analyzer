"""Treasury yield curve risk rotation signal.

Uses relative performance of IEF (7-10Y Treasury) vs TLT (20Y+ Treasury)
as a cross-asset risk regime detector.  Steepening curve (TLT outperforms)
signals risk-on; flattening/inverting (IEF outperforms) signals risk-off.

4 sub-indicators via majority vote:
    1. Curve Slope Direction   — 65d relative return sign
    2. Slope Momentum          — 21d rate of change of the spread
    3. Slope Z-Score           — 252d z-score of current spread
    4. Regime Persistence      — days since last regime flip

Direction depends on asset class:
- Risk-on assets (BTC, ETH, MSTR): steepening = BUY, flattening = SELL
- Safe havens (XAU, XAG):          steepening = SELL, flattening = BUY

Data: yfinance IEF + TLT (free, no API key). Cached 1 hour.
Source: Gayed (2014). An Intermarket Approach to Tactical Risk Rotation.
        SSRN 2431022.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.treasury_risk_rotation")

MIN_ROWS = 30
_CACHE_TTL = 3600

_SLOPE_LOOKBACK = 65
_MOM_LOOKBACK = 21
_ZSCORE_LOOKBACK = 252
_SPREAD_THRESHOLD = 0.01
_ZSCORE_THRESHOLD = 1.0
_MOM_THRESHOLD = 0.005
_PERSISTENCE_DAYS = 5

_SAFE_HAVENS = frozenset({"XAU-USD", "XAG-USD"})


def _fetch_treasury_data() -> dict | None:
    """Fetch IEF and TLT daily closes via yfinance.  Cached 1 hour."""

    def _do_fetch():
        try:
            import yfinance as yf

            data = yf.download(
                ["IEF", "TLT"], period="14mo", progress=False, threads=True,
            )
            if data is None or data.empty:
                return None

            close = data["Close"]
            if "IEF" not in close.columns or "TLT" not in close.columns:
                logger.warning("treasury_risk_rotation: missing IEF or TLT column")
                return None

            ief = close["IEF"].dropna()
            tlt = close["TLT"].dropna()

            if len(ief) < _SLOPE_LOOKBACK + 1 or len(tlt) < _SLOPE_LOOKBACK + 1:
                logger.warning("treasury_risk_rotation: insufficient data rows")
                return None

            return {"ief": ief, "tlt": tlt}
        except Exception as e:
            logger.warning("treasury_risk_rotation yfinance fetch failed: %s", e)
            return None

    return _cached("treasury_risk_rotation_yf", _CACHE_TTL, _do_fetch)


def _compute_spread_series(ief: pd.Series, tlt: pd.Series) -> pd.Series:
    """Compute rolling spread = TLT_return - IEF_return for each lookback window."""
    ief_ret = ief.pct_change(_SLOPE_LOOKBACK)
    tlt_ret = tlt.pct_change(_SLOPE_LOOKBACK)
    return tlt_ret - ief_ret


def _sub_slope_direction(spread_current: float) -> str:
    if spread_current > _SPREAD_THRESHOLD:
        return "BUY"
    if spread_current < -_SPREAD_THRESHOLD:
        return "SELL"
    return "HOLD"


def _sub_slope_momentum(spread_series: pd.Series) -> str:
    if len(spread_series) < _MOM_LOOKBACK + 1:
        return "HOLD"
    mom = float(spread_series.iloc[-1]) - float(spread_series.iloc[-_MOM_LOOKBACK - 1])
    if mom > _MOM_THRESHOLD:
        return "BUY"
    if mom < -_MOM_THRESHOLD:
        return "SELL"
    return "HOLD"


def _sub_slope_zscore(spread_series: pd.Series) -> tuple[float, str]:
    n = min(_ZSCORE_LOOKBACK, len(spread_series))
    if n < 30:
        return 0.0, "HOLD"
    window = spread_series.iloc[-n:]
    mean = float(window.mean())
    std = float(window.std())
    if std < 1e-9:
        return 0.0, "HOLD"
    z = (float(spread_series.iloc[-1]) - mean) / std
    if z > _ZSCORE_THRESHOLD:
        return z, "BUY"
    if z < -_ZSCORE_THRESHOLD:
        return z, "SELL"
    return z, "HOLD"


def _sub_regime_persistence(spread_series: pd.Series) -> str:
    """Count consecutive days the spread has stayed on the same side."""
    if len(spread_series) < 2:
        return "HOLD"
    last = float(spread_series.iloc[-1])
    if last == 0.0:
        return "HOLD"
    current_sign = 1 if last > 0 else -1
    days = 0
    for i in range(len(spread_series) - 1, -1, -1):
        val = float(spread_series.iloc[i])
        if val == 0.0:
            break
        s = 1 if val > 0 else -1
        if s != current_sign:
            break
        days += 1
    if days >= _PERSISTENCE_DAYS:
        return "BUY" if current_sign > 0 else "SELL"
    return "HOLD"


def _invert(action: str) -> str:
    if action == "BUY":
        return "SELL"
    if action == "SELL":
        return "BUY"
    return "HOLD"


def compute_treasury_risk_rotation_signal(
    df: pd.DataFrame, context: dict | None = None,
) -> dict:
    """Compute treasury risk rotation signal."""
    if df is None or len(df) < MIN_ROWS:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    treasury = _fetch_treasury_data()
    if treasury is None:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    ief = treasury["ief"]
    tlt = treasury["tlt"]
    spread_series = _compute_spread_series(ief, tlt)
    spread_series = spread_series.replace([np.inf, -np.inf], np.nan).dropna()

    if len(spread_series) < 30:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    spread_current = float(spread_series.iloc[-1])

    v_direction = _sub_slope_direction(spread_current)
    v_momentum = _sub_slope_momentum(spread_series)
    zscore_val, v_zscore = _sub_slope_zscore(spread_series)
    v_persistence = _sub_regime_persistence(spread_series)

    votes = [v_direction, v_momentum, v_zscore, v_persistence]
    action, confidence = majority_vote(votes, count_hold=False)

    ticker = (context or {}).get("ticker", "")
    is_safe_haven = ticker in _SAFE_HAVENS
    if is_safe_haven:
        action = _invert(action)

    ief_ret_65 = float(ief.iloc[-1] / ief.iloc[-min(_SLOPE_LOOKBACK, len(ief) - 1) - 1] - 1)
    tlt_ret_65 = float(tlt.iloc[-1] / tlt.iloc[-min(_SLOPE_LOOKBACK, len(tlt) - 1) - 1] - 1)

    return {
        "action": action,
        "confidence": min(confidence, 0.7),
        "sub_signals": {
            "slope_direction": v_direction,
            "slope_momentum": v_momentum,
            "slope_zscore": v_zscore,
            "regime_persistence": v_persistence,
        },
        "indicators": {
            "spread_65d": safe_float(spread_current),
            "zscore": safe_float(zscore_val),
            "ief_ret_65d": safe_float(ief_ret_65),
            "tlt_ret_65d": safe_float(tlt_ret_65),
            "is_safe_haven": is_safe_haven,
        },
    }
