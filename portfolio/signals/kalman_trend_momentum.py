"""Kalman Trend Momentum signal module.

State-space model with [price, velocity] state vector. Kalman filter
extracts smooth trend + velocity from noisy price data. 3-regime
classification (TREND_UP / TREND_DOWN / RANGE) based on velocity z-score.

Sub-indicators:
    1. Velocity Z-Score  — z-scored Kalman velocity vs rolling baseline
    2. Trend Regime      — UP/DOWN/RANGE from velocity sign + magnitude
    3. Price vs Filtered — divergence between raw and filtered price
    4. Velocity Momentum — acceleration (change in velocity)

Academic source: Singha et al (2025), arxiv:2511.08571.
Gold futures Sharpe 2.88, 43% CAGR, 0.52% max DD (2015-2025 OOS).

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 60 rows of data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote

MIN_ROWS = 60
VELOCITY_Z_BUY = 1.0
VELOCITY_Z_SELL = -1.0
NORM_WINDOW = 60
ACCEL_LOOKBACK = 5
DIVERGENCE_THRESHOLD = 0.015

PROCESS_NOISE_PRICE = 0.01
PROCESS_NOISE_VELOCITY = 0.001
OBSERVATION_NOISE = 1.0


def _run_kalman(close: np.ndarray, q_price: float, q_vel: float,
                r_obs: float) -> tuple[np.ndarray, np.ndarray]:
    """Run Kalman filter with constant-velocity model.

    State: [price, velocity]
    Transition: price_t = price_{t-1} + velocity_{t-1}
                velocity_t = velocity_{t-1}
    Observation: z_t = price_t + noise

    Returns filtered price and velocity arrays.
    """
    n = len(close)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[q_price, 0.0], [0.0, q_vel]])
    R = np.array([[r_obs]])

    x = np.array([close[0], 0.0])
    P = np.eye(2) * 100.0

    filtered_price = np.empty(n)
    filtered_velocity = np.empty(n)

    for t in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        z = close[t]
        y = z - (H @ x_pred)[0]
        S = (H @ P_pred @ H.T + R)[0, 0]

        if abs(S) < 1e-12:
            S = 1e-12

        K = (P_pred @ H.T) / S
        x = x_pred + K.flatten() * y
        P = (np.eye(2) - K @ H) @ P_pred

        filtered_price[t] = x[0]
        filtered_velocity[t] = x[1]

    return filtered_price, filtered_velocity


def _sub_velocity_direction(velocity: np.ndarray, close: np.ndarray) -> tuple[str, float, dict]:
    """Sub-1: Velocity direction and magnitude relative to price scale."""
    if len(velocity) < 20:
        return "HOLD", 0.0, {"velocity_ratio": np.nan}

    current_vel = velocity[-1]
    price_scale = np.nanmean(close[-20:])
    if price_scale < 1e-10:
        return "HOLD", 0.0, {"velocity_ratio": 0.0}

    vel_ratio = current_vel / price_scale
    recent_vel_ratios = velocity[-20:] / price_scale
    consistency = np.mean(recent_vel_ratios > 0) if vel_ratio > 0 else np.mean(recent_vel_ratios < 0)

    indicators = {
        "velocity_ratio": round(float(vel_ratio), 6),
        "direction_consistency": round(float(consistency), 3),
    }

    if vel_ratio > 0.002 and consistency > 0.65:
        conf = min(consistency, 0.85)
        return "BUY", conf, indicators
    elif vel_ratio < -0.002 and consistency > 0.65:
        conf = min(consistency, 0.85)
        return "SELL", conf, indicators
    return "HOLD", 0.0, indicators


def _sub_trend_regime(velocity: np.ndarray) -> tuple[str, float, dict]:
    """Sub-2: Trend regime from velocity sign + magnitude."""
    if len(velocity) < 20:
        return "HOLD", 0.0, {"regime": "unknown"}

    recent_vel = velocity[-20:]
    pos_frac = np.mean(recent_vel > 0)
    mean_vel = np.nanmean(recent_vel)

    indicators = {
        "pos_fraction": round(float(pos_frac), 3),
        "mean_velocity": round(float(mean_vel), 6),
    }

    if pos_frac > 0.7 and mean_vel > 0:
        indicators["regime"] = "trend_up"
        return "BUY", min(pos_frac, 0.85), indicators
    elif pos_frac < 0.3 and mean_vel < 0:
        indicators["regime"] = "trend_down"
        return "SELL", min(1 - pos_frac, 0.85), indicators
    else:
        indicators["regime"] = "range"
        return "HOLD", 0.0, indicators


def _sub_price_divergence(close: np.ndarray, filtered: np.ndarray,
                          velocity: np.ndarray) -> tuple[str, float, dict]:
    """Sub-3: Trend-aware divergence (dip/rip detector within trend)."""
    if len(close) < 2 or abs(close[-1]) < 1e-10:
        return "HOLD", 0.0, {"divergence_pct": np.nan}

    div_pct = (close[-1] - filtered[-1]) / close[-1]
    vel_sign = 1 if velocity[-1] > 0 else (-1 if velocity[-1] < 0 else 0)

    indicators = {"divergence_pct": round(float(div_pct), 5), "vel_sign": vel_sign}

    if vel_sign > 0 and div_pct < -DIVERGENCE_THRESHOLD:
        return "BUY", min(abs(div_pct) / 0.05, 0.7), indicators
    elif vel_sign < 0 and div_pct > DIVERGENCE_THRESHOLD:
        return "SELL", min(abs(div_pct) / 0.05, 0.7), indicators
    return "HOLD", 0.0, indicators


def _sub_velocity_momentum(velocity: np.ndarray) -> tuple[str, float, dict]:
    """Sub-4: Velocity acceleration (change in velocity)."""
    if len(velocity) < ACCEL_LOOKBACK + 1:
        return "HOLD", 0.0, {"acceleration": np.nan}

    accel = velocity[-1] - velocity[-ACCEL_LOOKBACK - 1]
    accel_std = np.nanstd(np.diff(velocity[-NORM_WINDOW:])) if len(velocity) >= NORM_WINDOW else 1.0
    if accel_std < 1e-12:
        accel_std = 1e-12

    accel_z = accel / accel_std
    indicators = {
        "acceleration": round(float(accel), 6),
        "acceleration_z": round(float(accel_z), 4),
    }

    if accel_z > 2.0:
        return "BUY", min(abs(accel_z) / 5.0, 0.7), indicators
    elif accel_z < -2.0:
        return "SELL", min(abs(accel_z) / 5.0, 0.7), indicators
    return "HOLD", 0.0, indicators


def compute_kalman_trend_momentum_signal(
    df: pd.DataFrame, context: dict = None
) -> dict:
    if df is None or len(df) < MIN_ROWS:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    close = df["close"].dropna().values.astype(float)
    if len(close) < MIN_ROWS:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    try:
        filtered_price, velocity = _run_kalman(
            close, PROCESS_NOISE_PRICE, PROCESS_NOISE_VELOCITY, OBSERVATION_NOISE
        )
    except Exception:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    sub_signals = {}
    all_indicators = {}
    votes = []

    action1, conf1, ind1 = _sub_velocity_direction(velocity, close)
    votes.append(action1)
    sub_signals["velocity_direction"] = action1
    all_indicators.update(ind1)

    action2, conf2, ind2 = _sub_trend_regime(velocity)
    votes.append(action2)
    sub_signals["trend_regime"] = action2
    all_indicators.update(ind2)

    action3, conf3, ind3 = _sub_price_divergence(close, filtered_price, velocity)
    votes.append(action3)
    sub_signals["price_divergence"] = action3
    all_indicators.update(ind3)

    action4, conf4, ind4 = _sub_velocity_momentum(velocity)
    votes.append(action4)
    sub_signals["velocity_momentum"] = action4
    all_indicators.update(ind4)

    action, confidence = majority_vote(votes, count_hold=False)

    all_indicators["kalman_filtered_price"] = round(float(filtered_price[-1]), 4)
    all_indicators["kalman_velocity"] = round(float(velocity[-1]), 6)

    return {
        "action": action,
        "confidence": confidence,
        "sub_signals": sub_signals,
        "indicators": all_indicators,
    }
