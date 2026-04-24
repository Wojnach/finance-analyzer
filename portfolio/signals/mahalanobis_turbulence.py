"""Mahalanobis turbulence index — cross-asset regime detection signal.

Academic basis: Kritzman & Li (2010), "Skulls, Financial Turbulence, and
Risk Management". Also: Kritzman, Li, Page, Rigobon (2010), "Principal
Components as a Measure of Systemic Risk".

The turbulence index measures the statistical unusualness of multi-asset
returns by computing the Mahalanobis distance of today's return vector
from the historical distribution. Unlike correlation-based regime signals,
this captures both magnitude anomalies AND correlation breakdowns in a
single metric.

Backtest evidence: turbulence-adjusted portfolio achieves Sharpe 2.20 vs
1.0 buy-and-hold, max drawdown 6% vs 32% (SPY/SHY, 2000-2022).

3 sub-indicators via majority vote:
    1. Turbulence z-score — distance from 60d rolling mean/std
    2. Turbulence trend   — 5-day slope (rising = deteriorating)
    3. Absorption ratio   — PCA complement (fraction of variance in top eigenvector)

Asset-class aware:
    - Safe havens (XAU/XAG): high turbulence = BUY (flight to quality)
    - Risk assets (BTC/ETH/MSTR): high turbulence = SELL (risk-off)

Requires a pandas DataFrame with at least 65 rows (for rolling covariance).

Data: 5-asset daily closes via yfinance (free, no API key).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.mahalanobis_turbulence")

MIN_ROWS = 65
_CACHE_TTL = 3600  # 1 hour — cross-asset data stable intraday

# Rolling window for covariance estimation
_COV_WINDOW = 252

# Z-score computation window
_Z_WINDOW = 60

# Thresholds
_Z_HIGH = 2.0       # Extreme turbulence
_Z_ELEVATED = 1.5   # Elevated turbulence
_Z_CALM = -1.5      # Abnormally calm (complacency)

# Absorption ratio threshold (percentile)
_AR_HIGH_PCTILE = 90
_AR_LOW_PCTILE = 20

# Safe-haven tickers (invert signal direction)
_SAFE_HAVEN_TICKERS = {"XAU-USD", "XAG-USD"}

# Peer tickers for yfinance download (proxy for our 5 assets)
_YF_TICKERS = ["BTC-USD", "ETH-USD", "GC=F", "SI=F", "SPY"]


def _fetch_multi_asset_closes() -> pd.DataFrame | None:
    """Fetch ~14 months of daily closes for the 5-asset universe.

    Returns DataFrame with columns = ticker symbols, index = dates.
    Cached for 1 hour via shared_state.
    """
    def _do_fetch():
        try:
            import yfinance as yf

            data = yf.download(
                _YF_TICKERS, period="14mo", progress=False, threads=True
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
            logger.warning("mahalanobis_turbulence: multi-asset fetch failed: %s", e)
            return None

    return _cached("mahalanobis_turb_closes", _do_fetch, ttl=_CACHE_TTL)


def _compute_turbulence_series(closes: pd.DataFrame) -> pd.DataFrame | None:
    """Compute rolling Mahalanobis turbulence index from multi-asset closes.

    d_t = (1/n) * (r_t - mu)^T * Sigma^-1 * (r_t - mu)

    Also computes absorption ratio as complementary sub-signal.

    Returns DataFrame with columns: turbulence, absorption_ratio.
    """
    if closes is None or len(closes) < MIN_ROWS:
        return None

    # Daily log returns
    returns = np.log(closes / closes.shift(1)).dropna()
    if len(returns) < _COV_WINDOW + 5:
        # Not enough data for full rolling window; use what we have
        min_window = max(60, len(returns) - 5)
        if len(returns) < min_window + 5:
            return None
        effective_window = min_window
    else:
        effective_window = _COV_WINDOW

    n_assets = returns.shape[1]
    turbulence_vals = []
    ar_vals = []
    indices = []

    for i in range(effective_window, len(returns)):
        window = returns.iloc[i - effective_window:i]

        # Drop columns with all NaN in window
        valid_cols = window.dropna(axis=1, how="all")
        if valid_cols.shape[1] < 3:
            continue

        # Forward-fill within window to handle sparse NaNs
        valid_cols = valid_cols.ffill().bfill().dropna(axis=1, how="any")
        if valid_cols.shape[1] < 3:
            continue

        n = valid_cols.shape[1]

        # Rolling mean and covariance
        mu = valid_cols.mean().values
        cov = valid_cols.cov().values

        # Regularize covariance matrix for numerical stability
        cov += np.eye(n) * 1e-8

        # Today's return vector (must match valid_cols columns)
        today_ret = returns.iloc[i][valid_cols.columns].values
        if np.any(np.isnan(today_ret)):
            continue

        # Mahalanobis distance: d = (1/n) * (r - mu)^T * Sigma^-1 * (r - mu)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Singular matrix — use pseudo-inverse
            cov_inv = np.linalg.pinv(cov)

        diff = today_ret - mu
        d_t = float(diff @ cov_inv @ diff) / n

        # Absorption ratio: fraction of variance explained by top eigenvector
        try:
            corr = valid_cols.corr().values
            if np.any(np.isnan(corr)):
                corr = np.nan_to_num(corr, nan=0.0)
                np.fill_diagonal(corr, 1.0)
            eigenvalues = np.linalg.eigvalsh(corr)
            total_var = np.sum(eigenvalues)
            ar = eigenvalues[-1] / total_var if total_var > 0 else 0.0
        except np.linalg.LinAlgError:
            ar = 0.0

        turbulence_vals.append(d_t)
        ar_vals.append(ar)
        indices.append(returns.index[i])

    if len(turbulence_vals) < 10:
        return None

    return pd.DataFrame({
        "turbulence": turbulence_vals,
        "absorption_ratio": ar_vals,
    }, index=indices)


# --------------- Sub-signal voters ---------------

def _turbulence_z_vote(turb_z: float, is_safe_haven: bool) -> str:
    """Vote based on turbulence z-score."""
    if turb_z >= _Z_HIGH:
        # Extreme turbulence
        return "BUY" if is_safe_haven else "SELL"
    if turb_z >= _Z_ELEVATED:
        # Elevated turbulence
        return "BUY" if is_safe_haven else "SELL"
    if turb_z <= _Z_CALM:
        # Abnormally calm — complacency / risk-on
        return "SELL" if is_safe_haven else "BUY"
    return "HOLD"


def _turbulence_trend_vote(turb_series: pd.Series, is_safe_haven: bool) -> str:
    """Vote based on 5-day slope of turbulence."""
    if len(turb_series) < 6:
        return "HOLD"

    recent = turb_series.iloc[-5:]
    # Use relative change to handle scale differences
    if recent.iloc[0] == 0:
        return "HOLD"

    slope = (recent.iloc[-1] - recent.iloc[0]) / max(abs(recent.iloc[0]), 1e-8)

    if slope > 0.5:
        # Rapidly rising turbulence
        return "BUY" if is_safe_haven else "SELL"
    if slope < -0.3:
        # Rapidly falling turbulence (normalizing)
        return "SELL" if is_safe_haven else "BUY"
    return "HOLD"


def _absorption_ratio_vote(ar_series: pd.Series, is_safe_haven: bool) -> str:
    """Vote based on absorption ratio percentile."""
    if len(ar_series) < _Z_WINDOW:
        return "HOLD"

    current_ar = ar_series.iloc[-1]
    pctile = (ar_series < current_ar).sum() / len(ar_series) * 100

    if pctile > _AR_HIGH_PCTILE:
        # Markets tightly coupled — systemic risk
        return "BUY" if is_safe_haven else "SELL"
    if pctile < _AR_LOW_PCTILE:
        # Markets diversified — healthy
        return "SELL" if is_safe_haven else "BUY"
    return "HOLD"


def compute_mahalanobis_turbulence_signal(
    df: pd.DataFrame, context: dict | None = None,
) -> dict[str, Any]:
    """Compute Mahalanobis turbulence regime signal.

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

    # Compute turbulence series
    turb_df = _compute_turbulence_series(closes)
    if turb_df is None or len(turb_df) < _Z_WINDOW:
        return hold

    # Current values
    turb_series = turb_df["turbulence"]
    ar_series = turb_df["absorption_ratio"]

    current_turb = turb_series.iloc[-1]
    current_ar = ar_series.iloc[-1]

    # Z-score of turbulence against recent window
    recent_turb = turb_series.iloc[-_Z_WINDOW:]
    turb_mean = recent_turb.mean()
    turb_std = recent_turb.std()

    if turb_std == 0 or np.isnan(turb_std):
        return hold

    turb_z = (current_turb - turb_mean) / turb_std

    # Sub-signal votes
    v1 = _turbulence_z_vote(turb_z, is_safe_haven)
    v2 = _turbulence_trend_vote(turb_series, is_safe_haven)
    v3 = _absorption_ratio_vote(ar_series, is_safe_haven)

    votes = [v1, v2, v3]
    action, confidence = majority_vote(votes, count_hold=False)

    # Cap confidence at 0.7 (external/cross-asset data signal)
    confidence = min(confidence, 0.7)

    # Boost confidence for extreme turbulence
    if abs(turb_z) > _Z_HIGH:
        confidence = min(confidence * 1.15, 0.7)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "turbulence_z": v1,
            "turbulence_trend": v2,
            "absorption_ratio": v3,
        },
        "indicators": {
            "turbulence_raw": safe_float(current_turb),
            "turbulence_z": safe_float(turb_z),
            "turbulence_mean_60d": safe_float(turb_mean),
            "absorption_ratio": safe_float(current_ar),
            "ar_percentile": safe_float(
                (ar_series < current_ar).sum() / len(ar_series) * 100
                if len(ar_series) > 0 else 0.0
            ),
            "is_safe_haven": is_safe_haven,
        },
    }
