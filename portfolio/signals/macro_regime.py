"""Macro-regime-based trading signal.

Combines six sub-indicators into a majority-vote composite:
  1. MA Regime Filter (adaptive)     (price vs longest available SMA, 50-200)
  2. DXY vs Risk Assets              (Dollar Index 5d change, ±0.3% threshold)
  3. Yield Curve Inversion Signal    (2s10s spread)
  4. 10Y Yield Momentum              (5d yield change direction, ±1.5%)
  5. FOMC Proximity                  (≤2 days = risk-off SELL)
  6. Golden/Death Cross (adaptive)   (fast/slow SMA cross, 20/50 or 50/200)

Each sub-indicator votes BUY / SELL / HOLD.  The composite action is the
majority vote; confidence is the fraction of non-HOLD votes that agree with
the majority direction.

The ``macro`` parameter is an optional dict containing DXY, treasury, and
Fed calendar data.  When absent or incomplete, affected sub-signals simply
vote HOLD — the signal degrades gracefully rather than failing.

SMA-based sub-signals adapt their periods to available data length, so
they work on short timeframes (100x 15m candles) as well as long ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio.signal_utils import majority_vote

# ---------------------------------------------------------------------------
# Minimum data lengths
# ---------------------------------------------------------------------------
_MIN_BARS_SMA50 = 50
_MIN_BARS_SMA200 = 200
_SMA200_TRANSITION_PCT = 0.01  # 1% band around 200-SMA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average with NaN for insufficient data."""
    return series.rolling(window=period, min_periods=period).mean()


def _safe_get(d: dict | None, *keys, default=None):
    """Safely traverse nested dict keys, returning *default* on any miss."""
    current = d
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
        if current is default:
            return default
    return current


# ---------------------------------------------------------------------------
# Sub-indicator votes
# ---------------------------------------------------------------------------

def _sma_regime(df: pd.DataFrame) -> tuple[str, dict]:
    """Moving-Average Regime Filter (adaptive period).

    Uses the longest available SMA up to 200 bars.  Falls back to shorter
    periods (min 50) when the dataframe has fewer than 200 rows — this is
    the common case on the Now timeframe (100x 15m candles).

    Price above SMA = bullish regime (BUY).
    Price below SMA = bearish regime (SELL).
    Price within 1% of SMA = HOLD (transition zone).
    """
    indicators: dict = {"sma_period": np.nan, "sma_value": np.nan,
                        "price_vs_sma_pct": np.nan}

    close = df["close"].astype(float)
    n = len(close)

    # Pick the longest SMA we can compute (min 50 bars)
    if n >= _MIN_BARS_SMA200:
        period = _MIN_BARS_SMA200
    elif n >= _MIN_BARS_SMA50:
        period = n  # use all available bars
    else:
        return "HOLD", indicators

    sma = _safe_sma(close, period)
    sma_val = sma.iloc[-1]

    if np.isnan(sma_val) or sma_val == 0:
        return "HOLD", indicators

    current_close = close.iloc[-1]
    pct_diff = (current_close - sma_val) / sma_val

    indicators["sma_period"] = period
    indicators["sma_value"] = float(sma_val)
    indicators["price_vs_sma_pct"] = float(pct_diff)

    if pct_diff > _SMA200_TRANSITION_PCT:
        return "BUY", indicators
    if pct_diff < -_SMA200_TRANSITION_PCT:
        return "SELL", indicators
    return "HOLD", indicators


def _dxy_risk(macro: dict | None) -> tuple[str, dict]:
    """DXY vs Risk Assets.

    Strong dollar (DXY rising, change_5d_pct > 0.3%) = SELL for risk assets.
    Weak dollar (change_5d_pct < -0.3%) = BUY for risk assets.
    Otherwise HOLD.
    """
    indicators: dict = {"dxy_value": np.nan, "dxy_change_5d_pct": np.nan}

    dxy_change = _safe_get(macro, "dxy", "change_5d_pct")
    dxy_value = _safe_get(macro, "dxy", "value")

    if dxy_change is None:
        return "HOLD", indicators

    try:
        dxy_change = float(dxy_change)
    except (TypeError, ValueError):
        return "HOLD", indicators

    if dxy_value is not None:
        try:
            indicators["dxy_value"] = float(dxy_value)
        except (TypeError, ValueError):
            pass
    indicators["dxy_change_5d_pct"] = dxy_change

    if dxy_change > 0.3:
        return "SELL", indicators
    if dxy_change < -0.3:
        return "BUY", indicators
    return "HOLD", indicators


def _yield_curve(macro: dict | None) -> tuple[str, dict]:
    """Yield Curve Inversion Signal.

    2s10s spread < 0 (inverted) = SELL (recession risk).
    2s10s spread > 0.5 = BUY (normal, healthy).
    Between 0 and 0.5 = HOLD (watch zone).
    """
    indicators: dict = {"yield_curve_2s10s": np.nan}

    spread = _safe_get(macro, "treasury", "spread_2s10s")
    if spread is None:
        return "HOLD", indicators

    try:
        spread = float(spread)
    except (TypeError, ValueError):
        return "HOLD", indicators

    indicators["yield_curve_2s10s"] = spread

    if spread < 0:
        return "SELL", indicators
    if spread > 0.5:
        return "BUY", indicators
    return "HOLD", indicators


def _yield_10y_momentum(macro: dict | None) -> tuple[str, dict]:
    """10Y Yield Momentum (direction-based).

    Uses 5-day yield change instead of absolute level, because 10Y
    spends most of its time between 3.5-5.0% where absolute thresholds
    produce permanent HOLD.

    Yields rising sharply (change_5d > +1.5%) = SELL (tightening).
    Yields falling sharply (change_5d < -1.5%) = BUY (easing).
    Otherwise HOLD.
    """
    indicators: dict = {"treasury_10y": np.nan, "treasury_10y_change_5d": np.nan}

    yield_pct = _safe_get(macro, "treasury", "10y", "yield_pct")
    change_5d = _safe_get(macro, "treasury", "10y", "change_5d")

    if yield_pct is not None:
        try:
            indicators["treasury_10y"] = float(yield_pct)
        except (TypeError, ValueError):
            pass

    if change_5d is None:
        return "HOLD", indicators

    try:
        change_5d = float(change_5d)
    except (TypeError, ValueError):
        return "HOLD", indicators

    indicators["treasury_10y_change_5d"] = change_5d

    if change_5d > 1.5:
        return "SELL", indicators
    if change_5d < -1.5:
        return "BUY", indicators
    return "HOLD", indicators


def _fomc_proximity(macro: dict | None) -> tuple[str, dict]:
    """FOMC Proximity.

    Within 2 days of FOMC = SELL (risk-off caution before high-vol event).
    3-7 days away = HOLD (too early to position).
    More than 7 days away = HOLD (no informational value).

    This is a short-lived risk-off signal that only activates ~4 days per
    FOMC cycle (8 cycles/year), preventing the permanent BUY bias that
    the old >14-days-away logic created.
    """
    indicators: dict = {"fomc_days_until": np.nan}

    days_until = _safe_get(macro, "fed", "days_until")
    if days_until is None:
        return "HOLD", indicators

    try:
        days_until = float(days_until)
    except (TypeError, ValueError):
        return "HOLD", indicators

    indicators["fomc_days_until"] = days_until

    if days_until <= 2:
        return "SELL", indicators
    return "HOLD", indicators


def _golden_death_cross(df: pd.DataFrame) -> tuple[str, dict]:
    """Golden/Death Cross Regime (adaptive periods).

    Uses SMA(fast) vs SMA(slow) cross with price confirmation.
    Adapts periods to available data: 50/200 when possible, falls
    back to 20/50 when fewer than 200 bars are available (common on
    the Now timeframe with 100x 15m candles).

    Fast > slow AND price > fast = golden cross (BUY).
    Fast < slow AND price < fast = death cross (SELL).
    Otherwise HOLD (transitioning).
    """
    indicators: dict = {"sma_fast": np.nan, "sma_slow": np.nan,
                        "sma_fast_period": np.nan, "sma_slow_period": np.nan,
                        "golden_cross": False, "death_cross": False}

    close = df["close"].astype(float)
    n = len(close)

    # Pick fast/slow periods based on available data
    if n >= _MIN_BARS_SMA200:
        fast_period, slow_period = 50, 200
    elif n >= _MIN_BARS_SMA50:
        fast_period, slow_period = 20, 50
    else:
        return "HOLD", indicators

    sma_fast = _safe_sma(close, fast_period)
    sma_slow = _safe_sma(close, slow_period)

    fast_val = sma_fast.iloc[-1]
    slow_val = sma_slow.iloc[-1]

    if np.isnan(fast_val) or np.isnan(slow_val):
        return "HOLD", indicators

    current_close = close.iloc[-1]
    indicators["sma_fast"] = float(fast_val)
    indicators["sma_slow"] = float(slow_val)
    indicators["sma_fast_period"] = fast_period
    indicators["sma_slow_period"] = slow_period

    if fast_val > slow_val and current_close > fast_val:
        indicators["golden_cross"] = True
        return "BUY", indicators
    if fast_val < slow_val and current_close < fast_val:
        indicators["death_cross"] = True
        return "SELL", indicators
    return "HOLD", indicators


# ---------------------------------------------------------------------------
# Composite signal
# ---------------------------------------------------------------------------

def compute_macro_regime_signal(df: pd.DataFrame, macro: dict = None) -> dict:
    """Compute the composite macro-regime signal.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV candle data with columns ``open``, ``high``, ``low``,
        ``close``, ``volume``, and optionally ``time``.  At least 200 rows
        recommended for full SMA coverage; degrades gracefully with less.
    macro : dict, optional
        Macro context dict that may contain ``dxy``, ``treasury``, and
        ``fed`` sub-dicts.  When None or missing keys, affected sub-signals
        vote HOLD.

    Returns
    -------
    dict
        ``action`` (BUY / SELL / HOLD), ``confidence`` (0.0-1.0),
        ``sub_signals`` dict mapping each sub-signal name to its vote,
        and ``indicators`` dict with computed values.
    """
    result: dict = {
        "action": "HOLD",
        "confidence": 0.0,
        "sub_signals": {
            "sma_regime": "HOLD",
            "dxy_risk": "HOLD",
            "yield_curve": "HOLD",
            "yield_10y_momentum": "HOLD",
            "fomc_proximity": "HOLD",
            "golden_death_cross": "HOLD",
        },
        "indicators": {
            "sma_period": np.nan,
            "sma_value": np.nan,
            "price_vs_sma_pct": np.nan,
            "sma_fast": np.nan,
            "sma_slow": np.nan,
            "sma_fast_period": np.nan,
            "sma_slow_period": np.nan,
            "golden_cross": False,
            "death_cross": False,
            "dxy_value": np.nan,
            "dxy_change_5d_pct": np.nan,
            "yield_curve_2s10s": np.nan,
            "treasury_10y": np.nan,
            "treasury_10y_change_5d": np.nan,
            "fomc_days_until": np.nan,
        },
    }

    # ---- Validate input ----
    if df is None or not isinstance(df, pd.DataFrame):
        return result

    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(set(df.columns)):
        return result

    if len(df) < 2:
        return result

    # ---- Compute each sub-indicator ----
    try:
        sma_action, sma_ind = _sma_regime(df)
    except Exception:
        sma_action, sma_ind = "HOLD", {}

    try:
        dxy_action, dxy_ind = _dxy_risk(macro)
    except Exception:
        dxy_action, dxy_ind = "HOLD", {}

    try:
        yc_action, yc_ind = _yield_curve(macro)
    except Exception:
        yc_action, yc_ind = "HOLD", {}

    try:
        y10_action, y10_ind = _yield_10y_momentum(macro)
    except Exception:
        y10_action, y10_ind = "HOLD", {}

    try:
        fomc_action, fomc_ind = _fomc_proximity(macro)
    except Exception:
        fomc_action, fomc_ind = "HOLD", {}

    try:
        gdc_action, gdc_ind = _golden_death_cross(df)
    except Exception:
        gdc_action, gdc_ind = "HOLD", {}

    # ---- Populate sub-signals and indicators ----
    result["sub_signals"]["sma_regime"] = sma_action
    result["sub_signals"]["dxy_risk"] = dxy_action
    result["sub_signals"]["yield_curve"] = yc_action
    result["sub_signals"]["yield_10y_momentum"] = y10_action
    result["sub_signals"]["fomc_proximity"] = fomc_action
    result["sub_signals"]["golden_death_cross"] = gdc_action

    result["indicators"].update(sma_ind)
    result["indicators"].update(dxy_ind)
    result["indicators"].update(yc_ind)
    result["indicators"].update(y10_ind)
    result["indicators"].update(fomc_ind)
    result["indicators"].update(gdc_ind)

    # ---- Majority vote ----
    votes = [sma_action, dxy_action, yc_action, y10_action, fomc_action, gdc_action]
    result["action"], result["confidence"] = majority_vote(votes)

    return result
