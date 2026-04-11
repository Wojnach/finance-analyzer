"""Hurst exponent regime detector signal module.

Computes the rolling Hurst exponent via Rescaled Range (R/S) analysis to
classify the current market regime as trending, mean-reverting, or random walk.
Produces directional votes based on the detected regime:

    - Trending (H > 0.55): vote in trend direction (EMA slope)
    - Mean-reverting (H < 0.45): vote contrarian at RSI extremes
    - Random walk (0.45 <= H <= 0.55): HOLD (no edge)

Sub-signals:
    1. Hurst Regime     — regime classification from R/S analysis
    2. Trend Direction   — EMA(9)/EMA(21) slope when trending
    3. MR Extreme        — RSI(14) contrarian when mean-reverting
    4. Hurst Momentum    — rate-of-change of H (regime shift detection)

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 160 rows of data (for 150-bar R/S window + buffer).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import ema, majority_vote, rsi, safe_float

# Minimum rows for the R/S analysis window plus lookback buffer.
# The largest sub-window is 128 bars, and we need at least 2 chunks,
# so 256 minimum for R/S.  We use 160 rows as the practical minimum
# (smaller sub-windows still produce valid estimates).
MIN_ROWS = 160

# R/S sub-window sizes (powers of 2 for clean chunking)
SUB_WINDOWS = (8, 16, 32, 64, 128)

# Regime thresholds (from FractalCycles guide + literature consensus)
TRENDING_THRESHOLD = 0.55
MR_THRESHOLD = 0.45

# Hurst momentum lookback for rate-of-change
HURST_ROC_LOOKBACK = 10


# ── R/S Analysis ─────────────────────────────────────────────────────────

def _rescaled_range(returns: np.ndarray, n: int) -> float:
    """Compute mean R/S statistic for sub-window size *n*.

    Splits *returns* into non-overlapping chunks of size *n*, computes the
    rescaled range for each chunk, and returns the mean.
    """
    n_chunks = len(returns) // n
    if n_chunks < 1:
        return np.nan

    rs_values = []
    for i in range(n_chunks):
        chunk = returns[i * n : (i + 1) * n]
        mean_c = chunk.mean()
        cumdev = np.cumsum(chunk - mean_c)
        r = cumdev.max() - cumdev.min()
        s = chunk.std(ddof=1)
        if s > 0 and np.isfinite(r):
            rs_values.append(r / s)

    return float(np.mean(rs_values)) if rs_values else np.nan


def _compute_hurst(returns: np.ndarray,
                   sub_windows: tuple[int, ...] = SUB_WINDOWS) -> float:
    """Compute Hurst exponent via R/S analysis.

    For each sub-window size, compute the mean R/S.  Then fit
    log(R/S) = H * log(n) + c  via OLS to get the Hurst exponent H.

    Returns NaN if fewer than 3 valid sub-window sizes.
    """
    log_rs = []
    log_n = []

    for n in sub_windows:
        if len(returns) < n * 2:
            continue
        rs = _rescaled_range(returns, n)
        if np.isfinite(rs) and rs > 0:
            log_rs.append(np.log(rs))
            log_n.append(np.log(n))

    if len(log_n) < 3:
        return np.nan

    # OLS: H = slope of log(R/S) vs log(n)
    x = np.array(log_n)
    y = np.array(log_rs)
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx == 0:
        return np.nan

    h = float(np.sum((x - x_mean) * (y - y_mean)) / ss_xx)
    return np.clip(h, 0.0, 1.0)


# ── Sub-signal 1: Hurst Regime Classification ────────────────────────────

def _hurst_regime(h: float) -> str:
    """Classify regime from Hurst exponent value."""
    if np.isnan(h):
        return "unknown"
    if h > TRENDING_THRESHOLD:
        return "trending"
    if h < MR_THRESHOLD:
        return "mean_reverting"
    return "random_walk"


# ── Sub-signal 2: Trend Direction (when trending) ────────────────────────

def _trend_direction(close: pd.Series) -> tuple[float, str]:
    """Determine trend direction via EMA(9) vs EMA(21) spread.

    Returns (spread_pct, vote).
    """
    if len(close) < 21:
        return 0.0, "HOLD"

    ema9 = ema(close, span=9)
    ema21 = ema(close, span=21)
    e9 = float(ema9.iloc[-1])
    e21 = float(ema21.iloc[-1])

    if np.isnan(e9) or np.isnan(e21) or e21 == 0:
        return 0.0, "HOLD"

    spread = (e9 - e21) / e21 * 100  # percent spread

    if spread > 0.3:
        return safe_float(spread), "BUY"
    if spread < -0.3:
        return safe_float(spread), "SELL"
    return safe_float(spread), "HOLD"


# ── Sub-signal 3: Mean-Reversion Extreme (when MR) ──────────────────────

def _mr_extreme(close: pd.Series) -> tuple[float, str]:
    """Contrarian signal at RSI extremes (for mean-reverting regime).

    Returns (rsi_value, vote).
    """
    if len(close) < 15:
        return float("nan"), "HOLD"

    rsi_vals = rsi(close, period=14)
    val = rsi_vals.iloc[-1]

    if np.isnan(val):
        return float("nan"), "HOLD"

    val = float(val)
    if val < 30:
        return val, "BUY"
    if val > 70:
        return val, "SELL"
    return val, "HOLD"


# ── Sub-signal 4: Hurst Momentum (regime shift detection) ───────────────

def _hurst_momentum(close: pd.Series, window: int = MIN_ROWS,
                    roc_lookback: int = HURST_ROC_LOOKBACK) -> tuple[float, float, str]:
    """Compute rate-of-change of Hurst exponent.

    Rising H (positive ROC) = trend strengthening -> favor trend direction.
    Falling H (negative ROC) = trend weakening -> caution / favor MR.

    Returns (hurst_roc, current_h, vote).
    """
    needed = window + roc_lookback
    if len(close) < needed:
        return float("nan"), float("nan"), "HOLD"

    returns = close.pct_change(fill_method=None).dropna().values

    # Current Hurst
    h_now = _compute_hurst(returns[-window:])
    # Hurst roc_lookback bars ago
    h_prev = _compute_hurst(returns[-(window + roc_lookback):-roc_lookback])

    if np.isnan(h_now) or np.isnan(h_prev):
        return float("nan"), safe_float(h_now), "HOLD"

    roc = h_now - h_prev

    # Strong rising Hurst = trend strengthening -> vote with trend
    # Strong falling Hurst = trend weakening -> favor caution
    if roc > 0.05:
        return safe_float(roc), safe_float(h_now), "BUY"   # trend strengthening
    if roc < -0.05:
        return safe_float(roc), safe_float(h_now), "SELL"   # trend weakening
    return safe_float(roc), safe_float(h_now), "HOLD"


# ── Public API ───────────────────────────────────────────────────────────

def compute_hurst_regime_signal(df: pd.DataFrame,
                                context: dict | None = None) -> dict:
    """Compute Hurst regime detector signal from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``open``, ``high``, ``low``, ``close``, ``volume``
        with at least 160 rows.
    context : dict, optional
        Signal context (ticker, regime, config).  Not used directly but
        accepted for interface compatibility.

    Returns
    -------
    dict with keys: action, confidence, sub_signals, indicators
    """
    hold_result: dict = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "hurst_regime": "HOLD",
            "trend_direction": "HOLD",
            "mr_extreme": "HOLD",
            "hurst_momentum": "HOLD",
        },
        "indicators": {
            "hurst_exponent": float("nan"),
            "regime": "unknown",
            "ema_spread_pct": float("nan"),
            "rsi14": float("nan"),
            "hurst_roc": float("nan"),
        },
    }

    if df is None or not isinstance(df, pd.DataFrame):
        return hold_result

    required = {"open", "high", "low", "close", "volume"}
    col_map = {c.lower(): c for c in df.columns}
    if required - set(col_map.keys()):
        return hold_result

    if len(df) < MIN_ROWS:
        return hold_result

    close = df[col_map["close"]].astype(float)

    # ── Compute Hurst exponent ────────────────────────────────────────────
    returns = close.pct_change(fill_method=None).dropna().values
    if len(returns) < MIN_ROWS - 1:
        return hold_result

    h = _compute_hurst(returns[-MIN_ROWS:])

    if np.isnan(h):
        return hold_result

    regime = _hurst_regime(h)

    # ── Sub-signals ───────────────────────────────────────────────────────
    sub_signals: dict[str, str] = {}
    indicators: dict[str, object] = {}

    indicators["hurst_exponent"] = safe_float(h)
    indicators["regime"] = regime

    # 1. Regime vote: trending or MR environment
    if regime == "trending":
        # In trending regime, vote with trend direction
        try:
            spread, trend_vote = _trend_direction(close)
            sub_signals["hurst_regime"] = trend_vote
            sub_signals["trend_direction"] = trend_vote
            indicators["ema_spread_pct"] = spread
        except Exception:
            sub_signals["hurst_regime"] = "HOLD"
            sub_signals["trend_direction"] = "HOLD"
            indicators["ema_spread_pct"] = float("nan")

        # MR sub-signal is inactive in trending regime
        sub_signals["mr_extreme"] = "HOLD"
        indicators["rsi14"] = float("nan")

    elif regime == "mean_reverting":
        # In MR regime, vote contrarian
        try:
            rsi_val, mr_vote = _mr_extreme(close)
            sub_signals["hurst_regime"] = mr_vote
            sub_signals["mr_extreme"] = mr_vote
            indicators["rsi14"] = safe_float(rsi_val)
        except Exception:
            sub_signals["hurst_regime"] = "HOLD"
            sub_signals["mr_extreme"] = "HOLD"
            indicators["rsi14"] = float("nan")

        # Trend sub-signal is inactive in MR regime
        sub_signals["trend_direction"] = "HOLD"
        indicators["ema_spread_pct"] = float("nan")

    else:
        # Random walk: no edge, force HOLD on all sub-signals
        sub_signals["hurst_regime"] = "HOLD"
        sub_signals["trend_direction"] = "HOLD"
        sub_signals["mr_extreme"] = "HOLD"
        indicators["ema_spread_pct"] = float("nan")
        indicators["rsi14"] = float("nan")

    # 4. Hurst momentum (regime shift detection)
    try:
        h_roc, _, h_mom_vote = _hurst_momentum(close)
        sub_signals["hurst_momentum"] = h_mom_vote
        indicators["hurst_roc"] = safe_float(h_roc)
    except Exception:
        sub_signals["hurst_momentum"] = "HOLD"
        indicators["hurst_roc"] = float("nan")

    # ── Majority vote ─────────────────────────────────────────────────────
    votes = list(sub_signals.values())
    action, confidence = majority_vote(votes)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": indicators,
    }
