"""Complexity gap regime signal — Random Matrix Theory market structure detector.

Academic basis: Mukhia, Ansari, Nurujjaman (2026), "Structural Dynamics of
G5 Stock Markets During Exogenous Shocks: A Random Matrix Theory-Based
Complexity Gap Approach", arXiv:2604.19107.

Complexity gap = (max eigenvalue / N) - mean pairwise correlation of a
multi-asset return correlation matrix.  When markets synchronize during
stress, correlations spike and the largest eigenvalue dominates → gap
collapses.  Three-phase pattern: collapse → false recovery → stabilization.

3 sub-indicators via majority vote:
    1. Gap z-score    — distance of current gap from 60d rolling mean
    2. Gap slope      — 5-day slope (widening vs narrowing trend)
    3. Corr regime    — average pairwise correlation level (absolute)

Asset-class aware:
    - Safe havens (XAU/XAG): gap collapse = BUY (flight to quality)
    - Risk assets (BTC/ETH/MSTR): gap collapse = SELL (risk-off)

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 65 rows of data.

Data: 5-asset daily closes via yfinance (free, no API key).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.complexity_gap_regime")

MIN_ROWS = 65
_CACHE_TTL = 3600  # 1 hour — cross-asset data doesn't change fast

# Rolling window for correlation matrix
_CORR_WINDOW = 60

# Z-score thresholds (symmetric)
_Z_THRESHOLD = 1.5  # Gap collapse (< -1.5) or widening (> +1.5)
_Z_STRONG = 2.0     # Strong conviction threshold

# Correlation regime thresholds
_CORR_HIGH = 0.45   # Average pairwise corr above this = synchronized
_CORR_LOW = 0.15    # Below this = diversified

# Safe-haven tickers (invert signal direction)
_SAFE_HAVEN_TICKERS = {"XAU-USD", "XAG-USD"}

# Peer tickers for yfinance download
_YF_TICKERS = ["BTC-USD", "ETH-USD", "GC=F", "SI=F", "SPY"]


def _fetch_multi_asset_closes() -> pd.DataFrame | None:
    """Fetch ~4 months of daily closes for the 5-asset universe.

    Returns DataFrame with columns = ticker symbols, index = dates.
    Cached for 1 hour via shared_state.
    """
    def _do_fetch():
        try:
            import yfinance as yf

            data = yf.download(
                _YF_TICKERS, period="4mo", progress=False, threads=True
            )
            if data is None or data.empty:
                return None

            # Extract Close prices — handle MultiIndex (yfinance multi-ticker)
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
            logger.warning("complexity_gap: multi-asset fetch failed: %s", e)
            return None

    return _cached("complexity_gap_closes", _do_fetch, ttl=_CACHE_TTL)


def _compute_complexity_gap_series(closes: pd.DataFrame) -> pd.DataFrame | None:
    """Compute rolling complexity gap from multi-asset close prices.

    Returns DataFrame with columns: gap, avg_corr, max_eig_norm.
    """
    if closes is None or len(closes) < MIN_ROWS:
        return None

    # Daily log returns
    returns = np.log(closes / closes.shift(1)).dropna()
    if len(returns) < _CORR_WINDOW + 5:
        return None

    n_assets = returns.shape[1]
    gaps = []
    avg_corrs = []
    max_eig_norms = []
    indices = []

    for i in range(_CORR_WINDOW, len(returns)):
        window = returns.iloc[i - _CORR_WINDOW:i]

        # Skip if too many NaNs
        valid_cols = window.dropna(axis=1, how="all")
        if valid_cols.shape[1] < 3:
            continue

        # Compute correlation matrix
        corr = valid_cols.corr().values
        n = corr.shape[0]

        # Handle NaN in correlation matrix
        if np.any(np.isnan(corr)):
            corr = np.nan_to_num(corr, nan=0.0)
            np.fill_diagonal(corr, 1.0)

        # Eigenvalues
        try:
            eigenvalues = np.linalg.eigvalsh(corr)
        except np.linalg.LinAlgError:
            continue

        max_eig = eigenvalues[-1]  # Largest eigenvalue
        norm_max_eig = max_eig / n

        # Average off-diagonal correlation
        mask = ~np.eye(n, dtype=bool)
        avg_corr = np.mean(np.abs(corr[mask]))

        # Complexity gap
        gap = norm_max_eig - avg_corr

        gaps.append(gap)
        avg_corrs.append(avg_corr)
        max_eig_norms.append(norm_max_eig)
        indices.append(returns.index[i])

    if len(gaps) < 10:
        return None

    return pd.DataFrame({
        "gap": gaps,
        "avg_corr": avg_corrs,
        "max_eig_norm": max_eig_norms,
    }, index=indices)


def _gap_zscore_vote(gap_z: float, is_safe_haven: bool) -> str:
    """Vote based on gap z-score."""
    if gap_z <= -_Z_THRESHOLD:
        # Gap collapse → synchronization/crisis
        return "BUY" if is_safe_haven else "SELL"
    if gap_z >= _Z_THRESHOLD:
        # Gap widening → diversification/healthy
        return "SELL" if is_safe_haven else "BUY"
    return "HOLD"


def _gap_slope_vote(gap_series: pd.Series, is_safe_haven: bool) -> str:
    """Vote based on 5-day slope of gap."""
    if len(gap_series) < 6:
        return "HOLD"

    recent = gap_series.iloc[-5:]
    slope = (recent.iloc[-1] - recent.iloc[0]) / 5

    if slope < -0.02:
        # Gap narrowing → increasing synchronization
        return "BUY" if is_safe_haven else "SELL"
    if slope > 0.02:
        # Gap widening → de-synchronization
        return "SELL" if is_safe_haven else "BUY"
    return "HOLD"


def _corr_regime_vote(avg_corr: float, is_safe_haven: bool) -> str:
    """Vote based on average pairwise correlation level."""
    if avg_corr > _CORR_HIGH:
        # High correlation → synchronized market
        return "BUY" if is_safe_haven else "SELL"
    if avg_corr < _CORR_LOW:
        # Low correlation → diversified market
        return "SELL" if is_safe_haven else "BUY"
    return "HOLD"


def compute_complexity_gap_regime_signal(
    df: pd.DataFrame, context: dict | None = None,
) -> dict[str, Any]:
    """Compute complexity gap regime signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        context: Optional dict with keys: ticker, config, asset_class, regime

    Returns:
        dict with keys: action, confidence, sub_signals, indicators
    """
    hold = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {},
        "indicators": {},
    }

    if df is None or len(df) < MIN_ROWS:
        return hold

    # Determine if this ticker is a safe haven
    ticker = (context or {}).get("ticker", "")
    is_safe_haven = ticker in _SAFE_HAVEN_TICKERS

    # Fetch multi-asset closes
    closes = _fetch_multi_asset_closes()
    if closes is None:
        return hold

    # Compute complexity gap series
    gap_df = _compute_complexity_gap_series(closes)
    if gap_df is None or len(gap_df) < 20:
        return hold

    # Current values
    current_gap = gap_df["gap"].iloc[-1]
    current_avg_corr = gap_df["avg_corr"].iloc[-1]
    current_max_eig = gap_df["max_eig_norm"].iloc[-1]

    # Z-score of gap
    gap_mean = gap_df["gap"].rolling(60, min_periods=20).mean().iloc[-1]
    gap_std = gap_df["gap"].rolling(60, min_periods=20).std().iloc[-1]
    if np.isnan(gap_std) or gap_std < 1e-8:
        return hold
    gap_z = (current_gap - gap_mean) / gap_std

    # Sub-signal votes
    zscore_vote = _gap_zscore_vote(gap_z, is_safe_haven)
    slope_vote = _gap_slope_vote(gap_df["gap"], is_safe_haven)
    corr_vote = _corr_regime_vote(current_avg_corr, is_safe_haven)

    votes = [zscore_vote, slope_vote, corr_vote]
    action, confidence = majority_vote(votes, count_hold=False)

    # Boost confidence for strong z-scores
    if abs(gap_z) > _Z_STRONG and action != "HOLD":
        confidence = min(confidence * 1.2, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "gap_zscore": zscore_vote,
            "gap_slope": slope_vote,
            "corr_regime": corr_vote,
        },
        "indicators": {
            "complexity_gap": safe_float(current_gap),
            "gap_z": safe_float(gap_z),
            "avg_corr": safe_float(current_avg_corr),
            "max_eig_norm": safe_float(current_max_eig),
            "gap_mean_60d": safe_float(gap_mean),
            "gap_std_60d": safe_float(gap_std),
            "is_safe_haven": is_safe_haven,
        },
    }
