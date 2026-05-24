"""BOCPD Regime Switch signal module.

Bayesian Online Changepoint Detection (Adams & MacKay, 2007) applied to
return series. Detects regime breaks in real-time via posterior run-length.
On changepoint: switch from trend-following to mean-reversion for N bars.

Sub-signals:
    1. Changepoint detector — BOCPD posterior run-length
    2. Trend follower — momentum direction when no changepoint
    3. Mean reverter — z-score reversion when changepoint active
    4. Regime classifier — combines CPD + momentum/MR into action

Source: Wood, Roberts, Zohren (2021), "Online CPD for Momentum Trading",
Journal of Financial Data Science. Sharpe improvement ~33%.

Requires: open, high, low, close, volume — at least 50 rows.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float, sma, rsi

logger = logging.getLogger(__name__)

MIN_ROWS = 50
HAZARD_LAMBDA = 100
MR_WINDOW = 20
MR_ZSCORE_ENTRY = 2.0
MOM_LOOKBACK = 20
CHANGEPOINT_THRESHOLD = 0.3


def _bocpd_run_lengths(returns: np.ndarray, hazard_lambda: float = HAZARD_LAMBDA) -> np.ndarray:
    """Compute BOCPD posterior max run-length for each observation.

    Uses a Gaussian predictive model with online sufficient statistics
    and constant hazard rate h = 1/hazard_lambda.

    Returns array of max-posterior run-lengths (higher = more stable regime).
    """
    n = len(returns)
    if n == 0:
        return np.array([])

    hazard = 1.0 / hazard_lambda

    mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, 1.0

    max_rl = np.zeros(n)

    rl_probs = np.array([1.0])

    muT = np.array([mu0])
    kappaT = np.array([kappa0])
    alphaT = np.array([alpha0])
    betaT = np.array([beta0])

    for t in range(n):
        x = returns[t]
        if np.isnan(x):
            max_rl[t] = max_rl[t - 1] if t > 0 else 0
            continue

        cur_len = len(rl_probs)

        pred_var = betaT * (kappaT + 1) / (alphaT * kappaT)
        pred_var = np.maximum(pred_var, 1e-10)
        pred_mean = muT

        z = (x - pred_mean) / np.sqrt(pred_var)
        nu = 2 * alphaT
        log_pred = -0.5 * np.log(pred_var) - (nu + 1) / 2 * np.log(1 + z ** 2 / nu)

        log_pred = np.clip(log_pred, -500, 0)
        pred_probs = np.exp(log_pred)
        pred_probs = np.maximum(pred_probs, 1e-300)

        growth = rl_probs * pred_probs * (1 - hazard)
        changepoint = np.sum(rl_probs * pred_probs * hazard)

        new_rl_probs = np.empty(cur_len + 1)
        new_rl_probs[0] = changepoint
        new_rl_probs[1:] = growth

        total = np.sum(new_rl_probs)
        if total > 0:
            new_rl_probs /= total

        new_mu = np.empty(cur_len + 1)
        new_kappa = np.empty(cur_len + 1)
        new_alpha = np.empty(cur_len + 1)
        new_beta = np.empty(cur_len + 1)

        new_mu[0] = mu0
        new_kappa[0] = kappa0
        new_alpha[0] = alpha0
        new_beta[0] = beta0

        new_kappa[1:] = kappaT + 1
        new_mu[1:] = (kappaT * muT + x) / new_kappa[1:]
        new_alpha[1:] = alphaT + 0.5
        new_beta[1:] = betaT + kappaT * (x - muT) ** 2 / (2 * new_kappa[1:])

        max_idx = np.argmax(new_rl_probs)
        max_rl[t] = max_idx

        max_keep = min(cur_len + 1, 300)
        rl_probs = new_rl_probs[:max_keep]
        muT = new_mu[:max_keep]
        kappaT = new_kappa[:max_keep]
        alphaT = new_alpha[:max_keep]
        betaT = new_beta[:max_keep]

        total = np.sum(rl_probs)
        if total > 0:
            rl_probs /= total

    return max_rl


def _detect_changepoint(max_run_lengths: np.ndarray, threshold: float = CHANGEPOINT_THRESHOLD) -> tuple[bool, float]:
    """Detect if a changepoint occurred recently.

    A changepoint is detected when the max run-length drops sharply
    relative to its recent history.
    """
    if len(max_run_lengths) < 5:
        return False, 0.0

    current_rl = max_run_lengths[-1]
    recent_rl = np.mean(max_run_lengths[-10:])
    if recent_rl < 1:
        return False, 0.0

    drop_ratio = current_rl / max(recent_rl, 1.0)

    is_changepoint = drop_ratio < threshold
    severity = max(0.0, 1.0 - drop_ratio)

    return is_changepoint, severity


def _trend_signal(close: pd.Series, lookback: int = MOM_LOOKBACK) -> tuple[str, float]:
    """Simple momentum: direction of lookback-period return."""
    if len(close) < lookback + 1:
        return "HOLD", 0.0

    ret = (close.iloc[-1] / close.iloc[-lookback] - 1)
    if abs(ret) < 0.001:
        return "HOLD", 0.0

    rsi_val = rsi(close, period=14).iloc[-1]
    if np.isnan(rsi_val):
        rsi_conf = 0.0
    else:
        rsi_conf = abs(rsi_val - 50) / 50

    conf = min(abs(ret) * 10, 0.7)
    conf = (conf + rsi_conf) / 2

    if ret > 0:
        return "BUY", min(conf, 0.7)
    else:
        return "SELL", min(conf, 0.7)


def _mr_signal(close: pd.Series, window: int = MR_WINDOW, zscore_threshold: float = MR_ZSCORE_ENTRY) -> tuple[str, float]:
    """Mean-reversion signal: z-score from rolling mean."""
    if len(close) < window:
        return "HOLD", 0.0

    mean = close.iloc[-window:].mean()
    std = close.iloc[-window:].std()
    if std < 1e-10:
        return "HOLD", 0.0

    z = (close.iloc[-1] - mean) / std

    if z < -zscore_threshold:
        conf = min(abs(z) / (zscore_threshold * 2), 0.7)
        return "BUY", conf
    elif z > zscore_threshold:
        conf = min(abs(z) / (zscore_threshold * 2), 0.7)
        return "SELL", conf
    return "HOLD", 0.0


def compute_bocpd_regime_switch_signal(df: pd.DataFrame, context: dict = None) -> dict:
    """Compute BOCPD regime switch signal.

    Uses Bayesian Online Changepoint Detection to switch between
    trend-following and mean-reversion modes.
    """
    if df is None or len(df) < MIN_ROWS:
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

    try:
        close = df["close"].astype(float)
        returns = close.pct_change(fill_method=None).dropna().values

        if len(returns) < MIN_ROWS - 1:
            return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}

        max_rl = _bocpd_run_lengths(returns, HAZARD_LAMBDA)
        is_changepoint, severity = _detect_changepoint(max_rl)

        trend_action, trend_conf = _trend_signal(close)
        mr_action, mr_conf = _mr_signal(close)

        if is_changepoint and severity > 0.3:
            action = mr_action
            confidence = mr_conf * min(severity + 0.5, 1.0)
            regime = "changepoint_mr"
        else:
            action = trend_action
            confidence = trend_conf
            regime = "trend_following"

        recent_rl = float(max_rl[-1]) if len(max_rl) > 0 else 0.0
        avg_rl = float(np.mean(max_rl[-20:])) if len(max_rl) >= 20 else recent_rl

        confidence = min(confidence, 0.7)

        sub_signals = {
            "changepoint_detector": "BREAK" if is_changepoint else "STABLE",
            "trend_follower": trend_action,
            "mean_reverter": mr_action,
            "regime_classifier": regime,
        }

        indicators = {
            "max_run_length": safe_float(recent_rl),
            "avg_run_length_20": safe_float(avg_rl),
            "changepoint_severity": safe_float(severity),
            "is_changepoint": is_changepoint,
            "trend_conf": safe_float(trend_conf),
            "mr_conf": safe_float(mr_conf),
            "regime": regime,
        }

        return {
            "action": action,
            "confidence": confidence,
            "sub_signals": sub_signals,
            "indicators": indicators,
        }

    except Exception as e:
        logger.warning("bocpd_regime_switch error: %s", e)
        return {"action": "HOLD", "confidence": 0.0, "sub_signals": {}, "indicators": {}}
