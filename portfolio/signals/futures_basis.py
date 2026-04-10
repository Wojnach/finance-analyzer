"""Futures basis regime signal — mark-index spread analysis.

Computes the perpetual futures basis (mark price vs index price spread) from
Binance FAPI premiumIndexKlines, detects contango/backwardation regimes, and
generates contrarian signals from statistical extremes.

Sub-indicators:
    1. basis_z_extreme   — Z-score of current basis vs 7-day rolling distribution
    2. basis_velocity    — Rate of basis change over 24h (accelerating = confirms)
    3. sustained_regime  — Backwardation/contango sustained for 6+ of last 8 periods
    4. basis_acceleration— Second derivative of basis (regime shift detection)

Applicable to all Binance FAPI tickers (BTC-USD, ETH-USD, XAU-USD, XAG-USD).

Research basis: CF Benchmarks 2025 (Sharpe 1.52 sentiment-gated basis strategy);
CoinDesk backwardation-bottom analysis (Nov 2022, Mar 2023, Aug 2023 bottoms).

Requires context dict with 'ticker' key for Binance symbol lookup.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from portfolio.api_utils import BINANCE_FAPI_BASE
from portfolio.http_retry import fetch_json
from portfolio.signal_utils import majority_vote, safe_float
from portfolio.tickers import BINANCE_FAPI_MAP, BINANCE_SPOT_MAP

logger = logging.getLogger("portfolio.signals.futures_basis")

# ── Configuration ────────────────────────────────────────────────────────
_MAX_CONFIDENCE = 0.7
_LOOKBACK_HOURS = 168        # 7 days of hourly candles
_Z_THRESHOLD_BUY = -1.5      # Extreme backwardation → contrarian BUY
_Z_THRESHOLD_SELL = 1.5      # Extreme contango → overheated SELL
_VELOCITY_WINDOW = 24        # 24h window for velocity computation
_SUSTAINED_THRESHOLD = 7     # 7 of last 8 periods in same regime (stricter)
_SUSTAINED_WINDOW = 8        # Window for sustained regime check
_SUSTAINED_MIN_ABS = 0.0002  # Minimum absolute basis for sustained regime to count
_MIN_KLINES = 48             # Minimum 2 days of data for reliable z-score

# Map tickers to their FAPI symbols
_SYMBOL_MAP = {**BINANCE_FAPI_MAP, **BINANCE_SPOT_MAP}


def _fetch_premium_klines(symbol: str, interval: str = "1h",
                          limit: int = _LOOKBACK_HOURS) -> list | None:
    """Fetch premium index klines from Binance FAPI.

    Returns list of [open_time, open, high, low, close, ...] candles
    where values represent the premium index (basis fraction).
    """
    data = fetch_json(
        f"{BINANCE_FAPI_BASE}/premiumIndexKlines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=15,
        label="premium_klines",
    )
    if not data or not isinstance(data, list):
        return None
    return data


def _basis_z_extreme(basis_values: np.ndarray) -> tuple[str, float]:
    """Sub-1: Z-score of current basis vs rolling distribution.

    Extreme backwardation (z < -1.8) → BUY (market bottoming).
    Extreme contango (z > 1.8) → SELL (overheated/overleveraged).
    """
    if len(basis_values) < _MIN_KLINES:
        return "HOLD", 0.0

    mean = np.nanmean(basis_values)
    std = np.nanstd(basis_values)
    if std < 1e-10:
        return "HOLD", 0.0

    current = basis_values[-1]
    z = (current - mean) / std

    if z < _Z_THRESHOLD_BUY:
        return "BUY", z
    elif z > _Z_THRESHOLD_SELL:
        return "SELL", z
    return "HOLD", z


def _basis_velocity(basis_values: np.ndarray) -> tuple[str, float]:
    """Sub-2: Rate of basis change over 24h.

    Rapidly decreasing basis (moving toward backwardation) → BUY confirmation.
    Rapidly increasing basis (moving toward contango) → SELL confirmation.
    """
    if len(basis_values) < _VELOCITY_WINDOW + 1:
        return "HOLD", 0.0

    current = basis_values[-1]
    past = basis_values[-_VELOCITY_WINDOW - 1]

    if math.isnan(current) or math.isnan(past):
        return "HOLD", 0.0

    velocity = current - past  # positive = moving toward contango

    # Normalize velocity by std of basis for comparability across assets
    std = np.nanstd(basis_values)
    if std < 1e-10:
        return "HOLD", 0.0

    v_normalized = velocity / std

    if v_normalized < -2.0:  # Rapid move toward backwardation
        return "BUY", v_normalized
    elif v_normalized > 2.0:  # Rapid move toward contango
        return "SELL", v_normalized
    return "HOLD", v_normalized


def _sustained_regime(basis_values: np.ndarray) -> tuple[str, float]:
    """Sub-3: Sustained backwardation or contango regime.

    If basis is meaningfully negative (< -0.02%) for 7+ of last 8 periods → BUY.
    If basis is meaningfully positive (> +0.02%) for 7+ of last 8 periods → SELL.

    Sustained backwardation historically aligns with market bottoms.
    Requires minimum absolute basis to filter out noise near zero.
    """
    if len(basis_values) < _SUSTAINED_WINDOW:
        return "HOLD", 0.0

    window = basis_values[-_SUSTAINED_WINDOW:]
    valid = window[~np.isnan(window)]
    if len(valid) < _SUSTAINED_WINDOW:
        return "HOLD", 0.0

    n_backwardation = np.sum(valid < -_SUSTAINED_MIN_ABS)
    n_contango = np.sum(valid > _SUSTAINED_MIN_ABS)

    if n_backwardation >= _SUSTAINED_THRESHOLD:
        return "BUY", float(n_backwardation / len(valid))
    elif n_contango >= _SUSTAINED_THRESHOLD:
        return "SELL", float(n_contango / len(valid))
    return "HOLD", 0.0


def _basis_acceleration(basis_values: np.ndarray) -> tuple[str, float]:
    """Sub-4: Second derivative of basis — regime shift detection.

    Accelerating move into backwardation (negative acceleration when already
    negative basis) → BUY confirmation. Vice versa for contango.
    """
    if len(basis_values) < _VELOCITY_WINDOW * 2 + 1:
        return "HOLD", 0.0

    # First derivative (velocity) at two points
    v_current = basis_values[-1] - basis_values[-_VELOCITY_WINDOW - 1]
    v_past = (
        basis_values[-_VELOCITY_WINDOW - 1]
        - basis_values[-2 * _VELOCITY_WINDOW - 1]
    )

    if math.isnan(v_current) or math.isnan(v_past):
        return "HOLD", 0.0

    acceleration = v_current - v_past  # positive = accelerating toward contango

    std = np.nanstd(basis_values)
    if std < 1e-10:
        return "HOLD", 0.0

    a_normalized = acceleration / std

    if a_normalized < -1.0:
        return "BUY", a_normalized
    elif a_normalized > 1.0:
        return "SELL", a_normalized
    return "HOLD", a_normalized


def compute_futures_basis_signal(df=None, context: dict = None) -> dict:
    """Compute futures basis regime signal.

    This signal fetches its own data from Binance premiumIndexKlines rather
    than using the OHLCV DataFrame (df is ignored but accepted for interface
    compatibility).

    Args:
        df: Ignored (interface compatibility). May be None.
        context: Dict with 'ticker' key for Binance symbol lookup.

    Returns:
        dict with action, confidence, sub_signals, indicators.
    """
    hold = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {},
        "indicators": {},
    }

    # Need ticker from context
    if not context or "ticker" not in context:
        return hold

    ticker = context["ticker"]
    symbol = _SYMBOL_MAP.get(ticker)
    if not symbol:
        return hold

    # Fetch premium index klines
    klines = _fetch_premium_klines(symbol, "1h", _LOOKBACK_HOURS)
    if not klines or len(klines) < _MIN_KLINES:
        return hold

    # Extract close values (index 4) — these are the basis fractions
    basis_values = np.array(
        [safe_float(k[4]) for k in klines], dtype=float
    )

    # Filter out NaN
    if np.isnan(basis_values).all():
        return hold

    # ── Compute sub-indicators ───────────────────────────────────────────
    z_action, z_val = _basis_z_extreme(basis_values)
    vel_action, vel_val = _basis_velocity(basis_values)
    regime_action, regime_val = _sustained_regime(basis_values)
    accel_action, accel_val = _basis_acceleration(basis_values)

    # ── Majority vote ────────────────────────────────────────────────────
    votes = [z_action, vel_action, regime_action, accel_action]
    action, confidence = majority_vote(votes, count_hold=False)

    # Cap confidence
    confidence = min(confidence, _MAX_CONFIDENCE)

    # Current basis stats
    current_basis = safe_float(basis_values[-1])
    mean_basis = safe_float(np.nanmean(basis_values))
    std_basis = safe_float(np.nanstd(basis_values))
    basis_pct = safe_float(current_basis * 100)  # Convert to percentage

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": {
            "basis_z_extreme": z_action,
            "basis_velocity": vel_action,
            "sustained_regime": regime_action,
            "basis_acceleration": accel_action,
        },
        "indicators": {
            "basis_current": current_basis,
            "basis_pct": basis_pct,
            "basis_mean_7d": mean_basis,
            "basis_std_7d": std_basis,
            "basis_z_score": safe_float(z_val),
            "basis_velocity_norm": safe_float(vel_val),
            "regime_strength": safe_float(regime_val),
            "acceleration_norm": safe_float(accel_val),
            "n_klines": len(klines),
        },
    }
