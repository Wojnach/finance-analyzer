"""Macro-regime-based trading signal.

Combines six sub-indicators into a majority-vote composite:
  1. 200-Day MA Regime Filter       (price vs 200-SMA)
  2. DXY vs Risk Assets             (Dollar Index momentum)
  3. Yield Curve Inversion Signal    (2s10s spread)
  4. 10Y Yield Momentum             (absolute yield level)
  5. FOMC Proximity                  (days until next meeting)
  6. Golden/Death Cross Regime       (50-SMA vs 200-SMA + price)

Each sub-indicator votes BUY / SELL / HOLD.  The composite action is the
majority vote; confidence is the fraction of non-HOLD votes that agree with
the majority direction.

The ``macro`` parameter is an optional dict containing DXY, treasury, and
Fed calendar data.  When absent or incomplete, affected sub-signals simply
vote HOLD — the signal degrades gracefully rather than failing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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

def _sma200_regime(df: pd.DataFrame) -> tuple[str, dict]:
    """200-Day MA Regime Filter.

    Price above 200-SMA = bullish regime (BUY).
    Price below 200-SMA = bearish regime (SELL).
    Price within 1% of 200-SMA = HOLD (transition zone).
    """
    indicators: dict = {"sma200": np.nan, "price_vs_sma200_pct": np.nan}

    close = df["close"].astype(float)
    if len(close) < _MIN_BARS_SMA200:
        return "HOLD", indicators

    sma200 = _safe_sma(close, _MIN_BARS_SMA200)
    sma200_val = sma200.iloc[-1]

    if np.isnan(sma200_val) or sma200_val == 0:
        return "HOLD", indicators

    current_close = close.iloc[-1]
    pct_diff = (current_close - sma200_val) / sma200_val

    indicators["sma200"] = float(sma200_val)
    indicators["price_vs_sma200_pct"] = float(pct_diff)

    if pct_diff > _SMA200_TRANSITION_PCT:
        return "BUY", indicators
    if pct_diff < -_SMA200_TRANSITION_PCT:
        return "SELL", indicators
    return "HOLD", indicators


def _dxy_risk(macro: dict | None) -> tuple[str, dict]:
    """DXY vs Risk Assets.

    Strong dollar (DXY rising, change_5d_pct > 0.5%) = SELL for risk assets.
    Weak dollar (change_5d_pct < -0.5%) = BUY for risk assets.
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

    if dxy_change > 0.5:
        return "SELL", indicators
    if dxy_change < -0.5:
        return "BUY", indicators
    return "HOLD", indicators


def _yield_curve(macro: dict | None) -> tuple[str, dict]:
    """Yield Curve Inversion Signal.

    2s10s spread < 0 (inverted) = SELL (recession risk).
    2s10s spread > 0.5 = BUY (normal, healthy).
    Between 0 and 0.5 = HOLD (watch zone).
    """
    indicators: dict = {"yield_curve_2s10s": np.nan}

    spread = _safe_get(macro, "treasury", "2s10s")
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
    """10Y Yield Momentum.

    10Y > 5.0% = SELL (tight financial conditions).
    10Y < 3.5% = BUY (easy money).
    Otherwise HOLD.
    """
    indicators: dict = {"treasury_10y": np.nan}

    yield_10y = _safe_get(macro, "treasury", "10y")
    if yield_10y is None:
        return "HOLD", indicators

    try:
        yield_10y = float(yield_10y)
    except (TypeError, ValueError):
        return "HOLD", indicators

    indicators["treasury_10y"] = yield_10y

    if yield_10y > 5.0:
        return "SELL", indicators
    if yield_10y < 3.5:
        return "BUY", indicators
    return "HOLD", indicators


def _fomc_proximity(macro: dict | None) -> tuple[str, dict]:
    """FOMC Proximity.

    Within 3 days of FOMC = HOLD (uncertainty).
    More than 14 days until FOMC = weak BUY (favorable for risk).
    3-14 days = HOLD.
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

    if days_until <= 3:
        return "HOLD", indicators
    if days_until > 14:
        return "BUY", indicators
    return "HOLD", indicators


def _golden_death_cross(df: pd.DataFrame) -> tuple[str, dict]:
    """Golden/Death Cross Regime.

    50-SMA > 200-SMA AND price > 50-SMA = strong BUY (golden cross regime).
    50-SMA < 200-SMA AND price < 50-SMA = strong SELL (death cross regime).
    Otherwise HOLD (transitioning).
    """
    indicators: dict = {"sma50": np.nan, "sma200_cross": np.nan,
                        "golden_cross": False, "death_cross": False}

    close = df["close"].astype(float)
    if len(close) < _MIN_BARS_SMA200:
        return "HOLD", indicators

    sma50 = _safe_sma(close, _MIN_BARS_SMA50)
    sma200 = _safe_sma(close, _MIN_BARS_SMA200)

    sma50_val = sma50.iloc[-1]
    sma200_val = sma200.iloc[-1]

    if np.isnan(sma50_val) or np.isnan(sma200_val):
        return "HOLD", indicators

    current_close = close.iloc[-1]
    indicators["sma50"] = float(sma50_val)
    indicators["sma200_cross"] = float(sma200_val)

    if sma50_val > sma200_val and current_close > sma50_val:
        indicators["golden_cross"] = True
        return "BUY", indicators
    if sma50_val < sma200_val and current_close < sma50_val:
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
            "sma200_regime": "HOLD",
            "dxy_risk": "HOLD",
            "yield_curve": "HOLD",
            "yield_10y_momentum": "HOLD",
            "fomc_proximity": "HOLD",
            "golden_death_cross": "HOLD",
        },
        "indicators": {
            "sma200": np.nan,
            "price_vs_sma200_pct": np.nan,
            "sma50": np.nan,
            "sma200_cross": np.nan,
            "golden_cross": False,
            "death_cross": False,
            "dxy_value": np.nan,
            "dxy_change_5d_pct": np.nan,
            "yield_curve_2s10s": np.nan,
            "treasury_10y": np.nan,
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
        sma200_action, sma200_ind = _sma200_regime(df)
    except Exception:
        sma200_action, sma200_ind = "HOLD", {"sma200": np.nan, "price_vs_sma200_pct": np.nan}

    try:
        dxy_action, dxy_ind = _dxy_risk(macro)
    except Exception:
        dxy_action, dxy_ind = "HOLD", {"dxy_value": np.nan, "dxy_change_5d_pct": np.nan}

    try:
        yc_action, yc_ind = _yield_curve(macro)
    except Exception:
        yc_action, yc_ind = "HOLD", {"yield_curve_2s10s": np.nan}

    try:
        y10_action, y10_ind = _yield_10y_momentum(macro)
    except Exception:
        y10_action, y10_ind = "HOLD", {"treasury_10y": np.nan}

    try:
        fomc_action, fomc_ind = _fomc_proximity(macro)
    except Exception:
        fomc_action, fomc_ind = "HOLD", {"fomc_days_until": np.nan}

    try:
        gdc_action, gdc_ind = _golden_death_cross(df)
    except Exception:
        gdc_action, gdc_ind = "HOLD", {"sma50": np.nan, "sma200_cross": np.nan,
                                        "golden_cross": False, "death_cross": False}

    # ---- Populate sub-signals and indicators ----
    result["sub_signals"]["sma200_regime"] = sma200_action
    result["sub_signals"]["dxy_risk"] = dxy_action
    result["sub_signals"]["yield_curve"] = yc_action
    result["sub_signals"]["yield_10y_momentum"] = y10_action
    result["sub_signals"]["fomc_proximity"] = fomc_action
    result["sub_signals"]["golden_death_cross"] = gdc_action

    result["indicators"].update(sma200_ind)
    result["indicators"].update(dxy_ind)
    result["indicators"].update(yc_ind)
    result["indicators"].update(y10_ind)
    result["indicators"].update(fomc_ind)
    result["indicators"].update(gdc_ind)

    # ---- Majority vote ----
    votes = [sma200_action, dxy_action, yc_action, y10_action, fomc_action, gdc_action]
    buy_count = votes.count("BUY")
    sell_count = votes.count("SELL")
    active_votes = buy_count + sell_count  # non-HOLD votes

    if active_votes == 0:
        # All sub-indicators abstain
        result["action"] = "HOLD"
        result["confidence"] = 0.0
    elif buy_count > sell_count:
        result["action"] = "BUY"
        result["confidence"] = round(buy_count / active_votes, 2)
    elif sell_count > buy_count:
        result["action"] = "SELL"
        result["confidence"] = round(sell_count / active_votes, 2)
    else:
        # Tied between BUY and SELL -- no clear direction
        result["action"] = "HOLD"
        result["confidence"] = 0.0

    return result
