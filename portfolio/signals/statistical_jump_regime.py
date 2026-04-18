"""Statistical Jump Model regime signal module.

Detects market regime (bull/bear/neutral) using statistical jump detection
with a persistence penalty to reduce whiplash from frequent regime flips.

Based on: Shu, Yu, Mulvey 2024 — "Statistical Jump Model for regime detection"
(12 citations). Simplified implementation using threshold-based jump detection
instead of full EM algorithm. Core insight preserved: persistence penalty
penalizes frequent regime switches.

Sub-indicators:
    1. Jump Detection     — returns exceeding k*vol threshold
    2. Regime Persistence — consecutive bars in current regime (min N to act)
    3. Volatility Regime  — low/normal/high vol classification
    4. Trend Confirmation — SMA slope direction alignment

Requires a pandas DataFrame with columns: open, high, low, close, volume
and at least 50 rows of data (for rolling volatility + SMA baseline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote, safe_float, sma

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MIN_ROWS = 50          # Need enough history for rolling stats
RETURN_WINDOW = 20     # Window for computing rolling returns
VOL_WINDOW = 20        # Window for rolling volatility
JUMP_THRESHOLD = 2.0   # Returns > k * vol = jump event
PERSISTENCE_MIN = 3    # Minimum consecutive bars in regime to act
SMA_PERIOD = 20        # Trend confirmation SMA period
VOL_LOW_PCTILE = 25    # Below this percentile = low-vol regime
VOL_HIGH_PCTILE = 75   # Above this percentile = high-vol regime
REGIME_DECAY = 10      # Bars before regime confidence starts decaying


def _detect_jumps(returns: pd.Series, vol: pd.Series,
                  threshold: float = JUMP_THRESHOLD) -> pd.Series:
    """Detect jump events where abs(return) > threshold * rolling vol.

    Returns a Series of +1 (positive jump), -1 (negative jump), 0 (no jump).
    """
    jump_up = (returns > threshold * vol).astype(int)
    jump_down = (returns < -threshold * vol).astype(int)
    return jump_up - jump_down


def _classify_vol_regime(vol: pd.Series, window: int = 252) -> pd.Series:
    """Classify volatility regime using rolling percentile rank.

    Returns: 'low_vol', 'normal', or 'high_vol' for each bar.
    """
    # Use available history up to `window` bars for percentile rank
    effective_window = min(window, len(vol))
    if effective_window < 20:
        return pd.Series("normal", index=vol.index)

    rank = vol.rolling(window=effective_window, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    result = pd.Series("normal", index=vol.index)
    result[rank < VOL_LOW_PCTILE / 100] = "low_vol"
    result[rank > VOL_HIGH_PCTILE / 100] = "high_vol"
    return result


def _compute_regime_with_persistence(jumps: pd.Series,
                                     persistence_min: int = PERSISTENCE_MIN
                                     ) -> tuple[list[str], list[int]]:
    """Apply persistence penalty: track regime and count consecutive bars.

    Regime only changes after persistence_min consecutive opposing signals.

    Returns:
        regimes: list of 'bull', 'bear', or 'neutral'
        persistence: list of consecutive bars in current regime
    """
    n = len(jumps)
    regimes = ["neutral"] * n
    persistence = [0] * n
    current_regime = "neutral"
    current_count = 0
    opposing_count = 0

    for i in range(n):
        j = jumps.iloc[i]

        if j == 0:
            # No jump — maintain current regime, reset opposing count slowly
            opposing_count = max(0, opposing_count - 1)
        elif current_regime == "neutral":
            # In neutral, any jump starts a directional regime attempt
            if j > 0:
                opposing_count = 0
                current_count += 1
                if current_count >= persistence_min:
                    current_regime = "bull"
                    current_count = persistence_min
            elif j < 0:
                opposing_count = 0
                current_count += 1
                if current_count >= persistence_min:
                    current_regime = "bear"
                    current_count = persistence_min
        elif current_regime == "bull":
            if j > 0:
                current_count += 1
                opposing_count = 0
            elif j < 0:
                opposing_count += 1
                if opposing_count >= persistence_min:
                    current_regime = "bear"
                    current_count = opposing_count
                    opposing_count = 0
        elif current_regime == "bear":
            if j < 0:
                current_count += 1
                opposing_count = 0
            elif j > 0:
                opposing_count += 1
                if opposing_count >= persistence_min:
                    current_regime = "bull"
                    current_count = opposing_count
                    opposing_count = 0

        regimes[i] = current_regime
        persistence[i] = current_count

    return regimes, persistence


def compute_statistical_jump_regime_signal(
    df: pd.DataFrame, context: dict = None
) -> dict:
    """Compute Statistical Jump Model regime signal.

    Args:
        df: DataFrame with columns: open, high, low, close, volume
            (minimum 50 rows)
        context: Optional dict with keys: ticker, config, asset_class, regime

    Returns:
        dict with keys:
            action: "BUY" | "SELL" | "HOLD"
            confidence: float 0.0-1.0
            sub_signals: dict of sub-indicator votes
            indicators: dict of raw indicator values
    """
    empty = {
        "action": "HOLD", "confidence": 0.0,
        "sub_signals": {}, "indicators": {},
    }

    if df is None or len(df) < MIN_ROWS:
        return empty

    close = df["close"].astype(float)
    if close.isna().sum() > len(close) * 0.3:
        return empty

    # Forward-fill small NaN gaps
    close = close.ffill()

    # --- Sub-indicator 1: Jump Detection ---
    log_returns = np.log(close / close.shift(1)).dropna()
    if len(log_returns) < RETURN_WINDOW:
        return empty

    rolling_vol = log_returns.rolling(window=VOL_WINDOW, min_periods=10).std()
    jumps = _detect_jumps(log_returns, rolling_vol)

    # --- Sub-indicator 2: Regime with Persistence ---
    regimes, persistence = _compute_regime_with_persistence(jumps)
    current_regime = regimes[-1]
    current_persistence = persistence[-1]

    # Jump detection vote
    if current_regime == "bull" and current_persistence >= PERSISTENCE_MIN:
        jump_vote = "BUY"
    elif current_regime == "bear" and current_persistence >= PERSISTENCE_MIN:
        jump_vote = "SELL"
    else:
        jump_vote = "HOLD"

    # --- Sub-indicator 3: Volatility Regime ---
    vol_regime = _classify_vol_regime(rolling_vol)
    current_vol_regime = vol_regime.iloc[-1] if len(vol_regime) > 0 else "normal"

    # In high-vol regimes, trend signals are less reliable → bias toward HOLD
    # In low-vol regimes, trend signals are more reliable → amplify
    if current_vol_regime == "high_vol":
        vol_vote = "HOLD"  # High vol = uncertain, don't add directional signal
    elif current_vol_regime == "low_vol":
        # Low vol tends to precede breakouts — use trend direction
        sma_val = sma(close, SMA_PERIOD)
        if len(sma_val) > 0 and not np.isnan(sma_val.iloc[-1]):
            slope = (sma_val.iloc[-1] - sma_val.iloc[-5]) if len(sma_val) >= 5 else 0
            vol_vote = "BUY" if slope > 0 else "SELL" if slope < 0 else "HOLD"
        else:
            vol_vote = "HOLD"
    else:
        vol_vote = "HOLD"

    # --- Sub-indicator 4: Trend Confirmation ---
    sma_series = sma(close, SMA_PERIOD)
    if len(sma_series) >= 5:
        sma_current = sma_series.iloc[-1]
        sma_prev = sma_series.iloc[-5]
        if not (np.isnan(sma_current) or np.isnan(sma_prev)):
            sma_slope = (sma_current - sma_prev) / sma_prev if sma_prev != 0 else 0
            if sma_slope > 0.005:
                trend_vote = "BUY"
            elif sma_slope < -0.005:
                trend_vote = "SELL"
            else:
                trend_vote = "HOLD"
        else:
            trend_vote = "HOLD"
    else:
        trend_vote = "HOLD"

    # --- Majority vote ---
    votes = [jump_vote, vol_vote, trend_vote]
    action, raw_confidence = majority_vote(votes, count_hold=False)

    # --- Confidence adjustment ---
    # Persistence-based confidence: longer regime = higher confidence
    persistence_factor = min(current_persistence / (PERSISTENCE_MIN * 3), 1.0)

    # Regime decay: after REGIME_DECAY bars without a new jump, confidence decays
    recent_jumps = jumps.iloc[-REGIME_DECAY:] if len(jumps) >= REGIME_DECAY else jumps
    jump_recency = (recent_jumps != 0).sum() / len(recent_jumps)

    confidence = raw_confidence * (0.5 + 0.3 * persistence_factor + 0.2 * jump_recency)
    confidence = min(max(confidence, 0.0), 1.0)

    # In high-vol regime, cap confidence
    if current_vol_regime == "high_vol":
        confidence = min(confidence, 0.5)

    # --- Indicators ---
    last_vol = safe_float(rolling_vol.iloc[-1]) if len(rolling_vol) > 0 else 0.0
    total_jumps = int((jumps != 0).sum())
    recent_jump_count = int((recent_jumps != 0).sum())

    return {
        "action": action,
        "confidence": round(confidence, 4),
        "sub_signals": {
            "jump_regime": jump_vote,
            "vol_regime": vol_vote,
            "trend_confirm": trend_vote,
        },
        "indicators": {
            "regime": current_regime,
            "persistence": current_persistence,
            "vol_regime": current_vol_regime,
            "rolling_vol": round(last_vol, 6),
            "total_jumps": total_jumps,
            "recent_jumps": recent_jump_count,
            "jump_recency": round(jump_recency, 4),
        },
    }
