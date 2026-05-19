"""Cross-asset Absorption Ratio regime signal.

Academic basis: Kritzman, Li, Page, Rigobon (2011), "Principal Components
as a Measure of Systemic Risk" (SSRN 1633027, 500+ citations). Also
validated in Hammond et al. (2026, arXiv:2605.17117) as 2nd-best regime
detector (Cohen's d = 0.80) across 17 historical crises.

The Absorption Ratio measures how much of total return variance is
concentrated in the top eigenvectors of the correlation matrix. High AR
means assets are tightly coupled (fragile/correlated regime), low AR means
diversified returns (stable regime).

Backtest evidence: Sharpe 0.54 -> 0.85, max DD -55% -> -15% on SPY
(2000-2022) using AR-percentile allocation.

3 sub-indicators via majority vote:
    1. AR z-score     -- deviation from expanding historical mean
    2. AR delta       -- 5-day change in AR (rising = deteriorating)
    3. AR percentile  -- current AR rank in expanding history

Asset-class aware:
    - Safe havens (XAU/XAG): high AR (fragile) = BUY (flight to quality)
    - Risk assets (BTC/ETH/MSTR): high AR (fragile) = SELL (risk-off)

Data: 5-asset daily closes via yfinance (free, no API key).
Cached 1 hour since cross-asset correlation structure is stable intraday.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.absorption_ratio_regime")

MIN_ROWS = 65
_CACHE_TTL = 3600

_COV_WINDOW = 252
_Z_WINDOW = 60
_DELTA_WINDOW = 5

_Z_THRESHOLD = 1.5
_PCTILE_HIGH = 85
_PCTILE_LOW = 15

_SAFE_HAVEN_TICKERS = {"XAU-USD", "XAG-USD"}

_YF_TICKERS = ["BTC-USD", "ETH-USD", "GC=F", "SI=F", "SPY"]


def _fetch_multi_asset_closes() -> pd.DataFrame | None:
    """Fetch ~14 months of daily closes for the 5-asset universe."""
    def _do_fetch():
        try:
            import yfinance as yf
            data = yf.download(
                _YF_TICKERS, period="14mo", progress=False, threads=True
            )
            if data is None or data.empty:
                return None
            if hasattr(data.columns, "levels") and len(data.columns.levels) > 1:
                close = data["Close"]
            else:
                close = data[["Close"]]
                close.columns = _YF_TICKERS[:1]
            close = close.dropna(how="all")
            if len(close) < MIN_ROWS:
                return None
            return close
        except Exception as e:
            logger.warning("absorption_ratio_regime: fetch failed: %s", e)
            return None

    return _cached("absorption_ratio_closes", _CACHE_TTL, _do_fetch)


def _compute_absorption_ratio_series(closes: pd.DataFrame) -> pd.Series | None:
    """Compute rolling Absorption Ratio from multi-asset closes.

    AR = sum(top ceil(N/5) eigenvalues) / sum(all eigenvalues)
    computed on the correlation matrix of rolling daily log returns.
    """
    if closes is None or len(closes) < MIN_ROWS:
        return None

    returns = np.log(closes / closes.shift(1)).dropna()
    if len(returns) < 60 + 5:
        return None

    effective_window = min(_COV_WINDOW, len(returns) - 5)
    if effective_window < 60:
        return None

    ar_vals = []
    indices = []

    for i in range(effective_window, len(returns)):
        window = returns.iloc[i - effective_window:i]

        valid_cols = window.dropna(axis=1, how="all")
        if valid_cols.shape[1] < 3:
            continue

        valid_cols = valid_cols.ffill().bfill().dropna(axis=1, how="any")
        if valid_cols.shape[1] < 3:
            continue

        n = valid_cols.shape[1]
        n_eigenvectors = max(1, int(np.ceil(n / 5)))

        try:
            corr = valid_cols.corr().values
            if np.any(np.isnan(corr)):
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 1.0)
            eigenvalues = np.linalg.eigvalsh(corr)
            eigenvalues = np.sort(eigenvalues)[::-1]
            total_var = np.sum(np.maximum(eigenvalues, 0))
            top_var = np.sum(np.maximum(eigenvalues[:n_eigenvectors], 0))
            ar = top_var / total_var if total_var > 0 else 0.0
        except np.linalg.LinAlgError:
            continue

        ar_vals.append(ar)
        indices.append(returns.index[i])

    if len(ar_vals) < 10:
        return None

    return pd.Series(ar_vals, index=indices, name="absorption_ratio")


def _ar_zscore_vote(ar_z: float, is_safe_haven: bool) -> str:
    if ar_z >= _Z_THRESHOLD:
        return "BUY" if is_safe_haven else "SELL"
    if ar_z <= -_Z_THRESHOLD:
        return "SELL" if is_safe_haven else "BUY"
    return "HOLD"


def _ar_delta_vote(ar_delta: float, is_safe_haven: bool) -> str:
    if ar_delta > 0.02:
        return "BUY" if is_safe_haven else "SELL"
    if ar_delta < -0.02:
        return "SELL" if is_safe_haven else "BUY"
    return "HOLD"


def _ar_percentile_vote(ar_pctile: float, is_safe_haven: bool) -> str:
    if ar_pctile >= _PCTILE_HIGH:
        return "BUY" if is_safe_haven else "SELL"
    if ar_pctile <= _PCTILE_LOW:
        return "SELL" if is_safe_haven else "BUY"
    return "HOLD"


def compute_absorption_ratio_regime_signal(
    df: pd.DataFrame, context: dict = None
) -> dict:
    """Compute cross-asset absorption ratio regime signal."""
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }
    if df is None or len(df) < MIN_ROWS:
        return empty

    ticker = (context or {}).get("ticker", "")
    is_safe_haven = ticker in _SAFE_HAVEN_TICKERS

    closes = _fetch_multi_asset_closes()
    ar_series = _compute_absorption_ratio_series(closes)
    if ar_series is None or len(ar_series) < _Z_WINDOW:
        return empty

    current_ar = float(ar_series.iloc[-1])

    expanding_mean = float(ar_series.iloc[:-1].mean())
    expanding_std = float(ar_series.iloc[:-1].std())
    if expanding_std < 1e-10:
        return empty

    ar_z = (current_ar - expanding_mean) / expanding_std

    ar_delta = 0.0
    if len(ar_series) > _DELTA_WINDOW:
        ar_delta = float(ar_series.iloc[-1] - ar_series.iloc[-_DELTA_WINDOW - 1])

    ar_pctile = float(
        (ar_series.iloc[:-1] < current_ar).sum() / max(1, len(ar_series) - 1) * 100
    )

    vote_z = _ar_zscore_vote(ar_z, is_safe_haven)
    vote_delta = _ar_delta_vote(ar_delta, is_safe_haven)
    vote_pctile = _ar_percentile_vote(ar_pctile, is_safe_haven)

    votes = [vote_z, vote_delta, vote_pctile]
    action, confidence = majority_vote(votes, count_hold=False)

    confidence = min(confidence, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "ar_zscore": vote_z,
            "ar_delta": vote_delta,
            "ar_percentile": vote_pctile,
        },
        "indicators": {
            "absorption_ratio": safe_float(current_ar),
            "ar_z_score": safe_float(ar_z),
            "ar_delta_5d": safe_float(ar_delta),
            "ar_percentile": safe_float(ar_pctile),
            "ar_expanding_mean": safe_float(expanding_mean),
            "n_history_points": len(ar_series),
        },
    }
