"""Residual pair reversion signal — cointegration-based pairs trading.

Regresses a target asset's returns on its natural "driver" pair using
rolling OLS, then z-scores the residual to detect mean-reversion
opportunities.  This is regime-neutral: it works in both trending and
ranging markets because the signal is relative, not directional.

Pairs:
    ETH-USD  ↔  BTC-USD   (correlation ~0.85)
    XAG-USD  ↔  XAU-USD   (correlation ~0.90)
    XAU-USD  ↔  XAG-USD   (inverse: gold relative to silver)
    MSTR     ↔  BTC-USD   (correlation ~0.97)
    BTC-USD  ↔  ETH-USD   (inverse: BTC relative to ETH)

Sub-indicators:
    1. Residual Z-Score   — z-scored OLS residual (primary)
    2. Beta Stability     — rolling beta stddev (high = relationship breaking)
    3. Half-Life          — Ornstein-Uhlenbeck half-life (MR speed quality)

Academic basis:
    - Leung & Nguyen (2018), "Constructing Cointegrated Cryptocurrency
      Portfolios for Statistical Arbitrage", SSRN 3235890.
    - Amberdata crypto pairs: Sharpe 0.93, 16% annual, 15.7% max DD.
    - BTC-neutral ETH residual reversion: Sharpe 2.3 post-2021 (6hr).

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 200 rows of data (180-bar OLS window + 20 for z-scoring).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from portfolio.shared_state import _cached
from portfolio.signal_utils import majority_vote, safe_float

logger = logging.getLogger("portfolio.signals.residual_pair_reversion")

MIN_ROWS = 200  # 180-bar OLS window + 20 for z-scoring

_CACHE_TTL = 3600  # 1 hour

# OLS rolling regression window
_OLS_WINDOW = 180

# Z-score normalisation window
_Z_LOOKBACK = 60

# Z-score thresholds for entry
_Z_BUY = -2.0    # residual is below mean → target underpriced vs driver
_Z_SELL = 2.0     # residual is above mean → target overpriced vs driver

# Beta stability threshold (stddev of rolling beta)
_BETA_UNSTABLE = 0.15  # if beta stddev > this, relationship is breaking

# Half-life thresholds (in bars)
_HL_FAST = 5       # MR is fast enough to trade
_HL_SLOW = 60      # MR too slow to be actionable

# Pair mapping: target ticker → yfinance ticker for the driver asset
_PAIR_MAP = {
    "ETH-USD": "BTC-USD",
    "XAG-USD": "GC=F",      # gold futures (yfinance symbol)
    "XAU-USD": "SI=F",      # silver futures (inverse pair)
    "MSTR": "BTC-USD",
    "BTC-USD": "ETH-USD",   # BTC relative to ETH
}


def _fetch_driver_closes(yf_ticker: str) -> pd.Series | None:
    """Fetch ~1 year of daily closes for the driver asset via yfinance.

    Cached for 1 hour to avoid redundant API calls.
    """
    def _do_fetch():
        try:
            import yfinance as yf
            data = yf.download(
                yf_ticker, period="1y", interval="1d",
                progress=False, auto_adjust=True,
            )
            if data is None or data.empty:
                return None
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return close.dropna()
        except Exception as exc:
            logger.warning("Failed to fetch driver %s: %s", yf_ticker, exc)
            return None

    return _cached(f"pair_reversion_driver_{yf_ticker}", _CACHE_TTL, _do_fetch)


def _rolling_ols_beta(target_ret: pd.Series, driver_ret: pd.Series,
                      window: int) -> tuple[pd.Series, pd.Series]:
    """Compute rolling OLS beta and residual.

    Uses vectorised rolling covariance/variance for efficiency.

    Returns:
        (rolling_beta, rolling_residual) as pd.Series
    """
    cov = target_ret.rolling(window=window, min_periods=window).cov(driver_ret)
    var = driver_ret.rolling(window=window, min_periods=window).var()

    # Avoid division by zero
    var_safe = var.replace(0, np.nan)
    beta = cov / var_safe

    residual = target_ret - beta * driver_ret
    return beta, residual


def _compute_half_life(residual: pd.Series) -> float:
    """Estimate Ornstein-Uhlenbeck half-life from residual series.

    Half-life = -ln(2) / ln(1 + theta) where theta is the AR(1) coefficient
    of the residual series (lag regression).

    Returns half-life in bars, or NaN if estimation fails.
    """
    resid = residual.dropna()
    if len(resid) < 30:
        return float("nan")

    y = resid.values[1:]
    x = resid.values[:-1]

    # Simple AR(1): y_t = alpha + theta * y_{t-1} + epsilon
    # theta = cov(y, x) / var(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov_xy = np.mean((x - x_mean) * (y - y_mean))
    var_x = np.var(x)

    if var_x == 0:
        return float("nan")

    theta = cov_xy / var_x

    # theta should be < 1 for mean-reverting (and > -1 for stability)
    if theta >= 1.0 or theta <= -1.0:
        return float("nan")

    if theta <= 0:
        # Negative theta means very fast MR — use absolute value
        theta = abs(theta)
        if theta >= 1.0:
            return float("nan")

    half_life = -np.log(2) / np.log(abs(theta)) if abs(theta) > 0 else float("nan")

    return safe_float(half_life)


# ---- sub-indicator 1: Residual Z-Score ----------------------------------------

def _residual_z_signal(residual: pd.Series, z_lookback: int) -> tuple[float, str]:
    """Z-score of the OLS residual.

    When z < -2.0: target is underpriced vs driver → BUY.
    When z > 2.0: target is overpriced vs driver → SELL.
    """
    if len(residual.dropna()) < z_lookback:
        return float("nan"), "HOLD"

    mean = residual.rolling(window=z_lookback, min_periods=z_lookback).mean()
    std = residual.rolling(window=z_lookback, min_periods=z_lookback).std()

    std_safe = std.replace(0, np.nan)
    z = (residual - mean) / std_safe

    z_val = safe_float(z.iloc[-1])
    if np.isnan(z_val):
        return float("nan"), "HOLD"

    if z_val < _Z_BUY:
        return z_val, "BUY"
    if z_val > _Z_SELL:
        return z_val, "SELL"
    return z_val, "HOLD"


# ---- sub-indicator 2: Beta Stability ------------------------------------------

def _beta_stability_signal(beta: pd.Series,
                           stability_window: int = 30) -> tuple[float, str]:
    """Rolling stddev of beta as relationship stability measure.

    High beta instability means the cointegration is breaking down →
    suppress signal (HOLD).  Stable beta confirms the pair relationship.
    """
    if len(beta.dropna()) < stability_window:
        return float("nan"), "HOLD"

    beta_std = beta.rolling(window=stability_window,
                            min_periods=stability_window).std()
    std_val = safe_float(beta_std.iloc[-1])

    if np.isnan(std_val):
        return float("nan"), "HOLD"

    if std_val > _BETA_UNSTABLE:
        # Relationship breaking down — do not trade
        return std_val, "HOLD"

    # Stable relationship — confirm directional signal
    # We use the latest residual direction to confirm
    return std_val, "HOLD"  # Stability alone doesn't generate direction


# ---- sub-indicator 3: Half-Life Quality ----------------------------------------

def _half_life_signal(half_life: float) -> tuple[float, str]:
    """Half-life as MR speed quality filter.

    Fast half-life (< 5 bars): MR is fast → confirms trade.
    Moderate (5-60 bars): acceptable.
    Slow (> 60 bars): MR too slow → suppress.
    """
    hl_val = safe_float(half_life)
    if np.isnan(hl_val) or hl_val <= 0:
        return float("nan"), "HOLD"

    if hl_val > _HL_SLOW:
        return hl_val, "HOLD"  # Too slow for actionable MR

    # Half-life is acceptable — doesn't generate direction on its own
    return hl_val, "HOLD"


# ---- composite ---------------------------------------------------------------

def compute_residual_pair_reversion_signal(
    df: pd.DataFrame, context: dict = None,
) -> dict:
    """Compute residual pair reversion signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
        context: Optional dict with keys: ticker, config, asset_class, regime

    Returns:
        dict with keys:
            action: "BUY" | "SELL" | "HOLD"
            confidence: float 0.0-1.0
            sub_signals: dict of sub-indicator votes
            indicators: dict of raw indicator values
    """
    empty = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {},
        "indicators": {},
    }

    if df is None or len(df) < MIN_ROWS:
        return empty

    # Determine ticker from context
    ticker = (context or {}).get("ticker", "")
    if not ticker or ticker not in _PAIR_MAP:
        return empty

    driver_yf = _PAIR_MAP[ticker]

    # Fetch driver asset data
    driver_close = _fetch_driver_closes(driver_yf)
    if driver_close is None or len(driver_close) < MIN_ROWS:
        return empty

    # Align target and driver by date
    target_close = df["close"].copy()
    target_close.index = pd.to_datetime(df.index) if not isinstance(
        df.index, pd.DatetimeIndex) else df.index

    # Create aligned DataFrame
    aligned = pd.DataFrame({
        "target": target_close,
        "driver": driver_close,
    }).dropna()

    if len(aligned) < MIN_ROWS:
        return empty

    # Compute log returns
    target_ret = np.log(aligned["target"] / aligned["target"].shift(1)).dropna()
    driver_ret = np.log(aligned["driver"] / aligned["driver"].shift(1)).dropna()

    # Align returns
    common_idx = target_ret.index.intersection(driver_ret.index)
    if len(common_idx) < MIN_ROWS - 1:
        return empty

    target_ret = target_ret.loc[common_idx]
    driver_ret = driver_ret.loc[common_idx]

    # Rolling OLS regression
    beta, residual = _rolling_ols_beta(target_ret, driver_ret, _OLS_WINDOW)

    # Sub-indicator 1: Residual z-score
    z_val, z_vote = _residual_z_signal(residual, _Z_LOOKBACK)

    # Sub-indicator 2: Beta stability
    beta_std_val, beta_vote = _beta_stability_signal(beta)

    # Sub-indicator 3: Half-life
    recent_residual = residual.dropna().iloc[-_Z_LOOKBACK:]
    half_life = _compute_half_life(recent_residual)
    hl_val, hl_vote = _half_life_signal(half_life)

    # Combine: z-score drives direction; beta stability and half-life are
    # quality filters that suppress (HOLD) if conditions are poor.
    # If beta is unstable OR half-life is too slow, override to HOLD.
    if beta_std_val is not None and not np.isnan(beta_std_val):
        if beta_std_val > _BETA_UNSTABLE:
            return {
                "action": "HOLD",
                "confidence": 0.0,
                "sub_signals": {
                    "residual_z": z_vote,
                    "beta_stability": "HOLD (unstable)",
                    "half_life": hl_vote,
                },
                "indicators": {
                    "residual_z": safe_float(z_val),
                    "beta_std": safe_float(beta_std_val),
                    "half_life": safe_float(hl_val),
                    "latest_beta": safe_float(beta.dropna().iloc[-1]
                                              if len(beta.dropna()) > 0
                                              else float("nan")),
                    "driver": driver_yf,
                },
            }

    if not np.isnan(safe_float(hl_val)) and safe_float(hl_val) > _HL_SLOW:
        return {
            "action": "HOLD",
            "confidence": 0.0,
            "sub_signals": {
                "residual_z": z_vote,
                "beta_stability": beta_vote,
                "half_life": "HOLD (too slow)",
            },
            "indicators": {
                "residual_z": safe_float(z_val),
                "beta_std": safe_float(beta_std_val),
                "half_life": safe_float(hl_val),
                "latest_beta": safe_float(beta.dropna().iloc[-1]
                                          if len(beta.dropna()) > 0
                                          else float("nan")),
                "driver": driver_yf,
            },
        }

    # z-score is the primary directional signal
    action = z_vote
    if action == "HOLD":
        confidence = 0.0
    else:
        # Confidence based on z-score magnitude (clipped to 0.7 max)
        z_abs = abs(safe_float(z_val)) if not np.isnan(safe_float(z_val)) else 0.0
        # Scale: z=2.0 → 0.5, z=3.0 → 0.65, z=4.0+ → 0.7
        confidence = min(0.7, 0.35 + 0.15 * (z_abs - 2.0))
        confidence = max(0.0, confidence)

        # Boost if half-life is fast (confirms strong MR)
        if not np.isnan(safe_float(hl_val)) and safe_float(hl_val) < _HL_FAST:
            confidence = min(0.7, confidence * 1.15)

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "residual_z": z_vote,
            "beta_stability": beta_vote,
            "half_life": hl_vote,
        },
        "indicators": {
            "residual_z": safe_float(z_val),
            "beta_std": safe_float(beta_std_val),
            "half_life": safe_float(hl_val),
            "latest_beta": safe_float(beta.dropna().iloc[-1]
                                      if len(beta.dropna()) > 0
                                      else float("nan")),
            "driver": driver_yf,
        },
    }
