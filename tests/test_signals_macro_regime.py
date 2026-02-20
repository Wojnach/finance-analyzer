"""Tests for the macro-regime signal module.

Covers:
- Basic computation with sufficient data and no macro context
- Insufficient data returns HOLD
- Price above 200-SMA triggers BUY
- Price below 200-SMA triggers SELL
- DXY rising triggers SELL with macro data
- Yield curve inverted triggers SELL
- FOMC within 3 days triggers HOLD
- Golden cross regime detection
- Death cross regime detection
- No macro data gracefully handled
- Tie between BUY and SELL returns HOLD
- Confidence calculation correctness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.macro_regime import (
    compute_macro_regime_signal,
    _sma200_regime,
    _dxy_risk,
    _yield_curve,
    _yield_10y_momentum,
    _fomc_proximity,
    _golden_death_cross,
)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 250, close_base: float = 100.0,
             trend: float = 0.0, volatility: float = 1.0) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame.

    Parameters
    ----------
    n : int
        Number of rows.
    close_base : float
        Starting price level.
    trend : float
        Per-bar drift added to close (positive = uptrend).
    volatility : float
        Noise scale.
    """
    np.random.seed(42)
    noise = np.random.randn(n) * volatility
    close = close_base + np.cumsum(noise) + np.arange(n) * trend
    close = np.maximum(close, 1.0)  # keep prices positive
    high = close + np.abs(np.random.randn(n) * volatility)
    low = close - np.abs(np.random.randn(n) * volatility)
    low = np.maximum(low, 0.5)
    opn = close + np.random.randn(n) * 0.3
    volume = np.random.randint(100, 10000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "time": dates,
    })


def _make_df_above_sma200(n: int = 250) -> pd.DataFrame:
    """Create a DataFrame where the latest close is well above the 200-SMA.

    Uses a strong uptrend so price ends far above the 200-period average.
    """
    np.random.seed(42)
    # Linear uptrend with small noise: starts at 100, ends ~100+0.5*250=225
    close = 100.0 + np.arange(n) * 0.5 + np.random.randn(n) * 0.5
    close = np.maximum(close, 1.0)
    high = close + 1.0
    low = close - 1.0
    low = np.maximum(low, 0.5)
    opn = close + np.random.randn(n) * 0.2
    volume = np.random.randint(100, 10000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": opn, "high": high, "low": low,
        "close": close, "volume": volume, "time": dates,
    })


def _make_df_below_sma200(n: int = 250) -> pd.DataFrame:
    """Create a DataFrame where the latest close is well below the 200-SMA.

    Uses a strong downtrend so price ends far below the 200-period average.
    """
    np.random.seed(42)
    # Linear downtrend: starts at 200, ends ~200-0.5*250=75
    close = 200.0 - np.arange(n) * 0.5 + np.random.randn(n) * 0.5
    close = np.maximum(close, 1.0)
    high = close + 1.0
    low = close - 1.0
    low = np.maximum(low, 0.5)
    opn = close + np.random.randn(n) * 0.2
    volume = np.random.randint(100, 10000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": opn, "high": high, "low": low,
        "close": close, "volume": volume, "time": dates,
    })


def _make_golden_cross_df(n: int = 250) -> pd.DataFrame:
    """Create a DataFrame exhibiting golden cross: 50-SMA > 200-SMA, price > 50-SMA.

    The price starts low, then ramps up steeply in the second half so that:
    - The 50-SMA (recent) is well above the 200-SMA (all-time)
    - The current price is above both
    """
    np.random.seed(42)
    # Flat then ramp: first 150 bars at 100, then steep ramp to ~250
    close = np.concatenate([
        np.full(150, 100.0) + np.random.randn(150) * 0.3,
        100.0 + np.arange(100) * 1.5 + np.random.randn(100) * 0.3,
    ])
    close = np.maximum(close, 1.0)
    high = close + 1.0
    low = close - 1.0
    low = np.maximum(low, 0.5)
    opn = close + np.random.randn(n) * 0.2
    volume = np.random.randint(100, 10000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": opn, "high": high, "low": low,
        "close": close, "volume": volume, "time": dates,
    })


def _make_death_cross_df(n: int = 250) -> pd.DataFrame:
    """Create a DataFrame exhibiting death cross: 50-SMA < 200-SMA, price < 50-SMA.

    The price starts high, then drops steeply in the second half.
    """
    np.random.seed(42)
    # Flat then drop: first 150 bars at 200, then steep decline
    close = np.concatenate([
        np.full(150, 200.0) + np.random.randn(150) * 0.3,
        200.0 - np.arange(100) * 1.5 + np.random.randn(100) * 0.3,
    ])
    close = np.maximum(close, 1.0)
    high = close + 1.0
    low = close - 1.0
    low = np.maximum(low, 0.5)
    opn = close + np.random.randn(n) * 0.2
    volume = np.random.randint(100, 10000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": opn, "high": high, "low": low,
        "close": close, "volume": volume, "time": dates,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicComputation:
    """Basic computation with sufficient data and no macro context."""

    def test_returns_required_keys(self):
        df = _make_df(n=250)
        result = compute_macro_regime_signal(df)
        assert "action" in result
        assert "confidence" in result
        assert "sub_signals" in result
        assert "indicators" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_sub_signals_has_all_six(self):
        df = _make_df(n=250)
        result = compute_macro_regime_signal(df)
        expected = {
            "sma200_regime", "dxy_risk", "yield_curve",
            "yield_10y_momentum", "fomc_proximity", "golden_death_cross",
        }
        assert set(result["sub_signals"].keys()) == expected

    def test_all_sub_signals_are_valid_votes(self):
        df = _make_df(n=250)
        result = compute_macro_regime_signal(df)
        for name, vote in result["sub_signals"].items():
            assert vote in ("BUY", "SELL", "HOLD"), f"{name} has invalid vote: {vote}"

    def test_indicators_populated_with_data(self):
        df = _make_df(n=250)
        result = compute_macro_regime_signal(df)
        # SMA indicators should be computed when we have 250 bars
        assert not np.isnan(result["indicators"]["sma200"])
        assert not np.isnan(result["indicators"]["sma50"])

    def test_no_macro_means_macro_sub_signals_hold(self):
        """Without macro dict, DXY/yield/FOMC sub-signals should all be HOLD."""
        df = _make_df(n=250)
        result = compute_macro_regime_signal(df, macro=None)
        assert result["sub_signals"]["dxy_risk"] == "HOLD"
        assert result["sub_signals"]["yield_curve"] == "HOLD"
        assert result["sub_signals"]["yield_10y_momentum"] == "HOLD"
        assert result["sub_signals"]["fomc_proximity"] == "HOLD"


class TestInsufficientData:
    """Insufficient data returns HOLD with confidence 0.0."""

    def test_very_short_df(self):
        df = _make_df(n=5)
        result = compute_macro_regime_signal(df)
        # SMA-based sub-signals should be HOLD with insufficient data
        assert result["sub_signals"]["sma200_regime"] == "HOLD"
        assert result["sub_signals"]["golden_death_cross"] == "HOLD"

    def test_none_df_returns_hold(self):
        result = compute_macro_regime_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_empty_df_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_macro_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_missing_columns_returns_hold(self):
        df = pd.DataFrame({"close": [100.0], "volume": [1000.0]})
        result = compute_macro_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_one_row_returns_hold(self):
        df = _make_df(n=1)
        result = compute_macro_regime_signal(df)
        assert result["action"] == "HOLD"


class TestSMA200Regime:
    """Price above/below 200-SMA triggers BUY/SELL."""

    def test_price_above_sma200_triggers_buy(self):
        df = _make_df_above_sma200()
        action, indicators = _sma200_regime(df)
        assert action == "BUY"
        assert indicators["price_vs_sma200_pct"] > 0.01

    def test_price_below_sma200_triggers_sell(self):
        df = _make_df_below_sma200()
        action, indicators = _sma200_regime(df)
        assert action == "SELL"
        assert indicators["price_vs_sma200_pct"] < -0.01

    def test_insufficient_data_returns_hold(self):
        df = _make_df(n=50)
        action, indicators = _sma200_regime(df)
        assert action == "HOLD"
        assert np.isnan(indicators["sma200"])

    def test_composite_buy_with_above_sma200(self):
        """Full composite with uptrend data (no macro) should lean BUY."""
        df = _make_df_above_sma200()
        result = compute_macro_regime_signal(df)
        # Both sma200_regime and golden_death_cross should be BUY
        assert result["sub_signals"]["sma200_regime"] == "BUY"


class TestDXYRisk:
    """DXY rising triggers SELL with macro data."""

    def test_strong_dollar_triggers_sell(self):
        macro = {"dxy": {"value": 105.0, "change_5d_pct": 1.2}}
        action, indicators = _dxy_risk(macro)
        assert action == "SELL"
        assert indicators["dxy_change_5d_pct"] == 1.2

    def test_weak_dollar_triggers_buy(self):
        macro = {"dxy": {"value": 99.0, "change_5d_pct": -0.8}}
        action, indicators = _dxy_risk(macro)
        assert action == "BUY"
        assert indicators["dxy_change_5d_pct"] == -0.8

    def test_neutral_dollar_triggers_hold(self):
        macro = {"dxy": {"value": 102.0, "change_5d_pct": 0.2}}
        action, indicators = _dxy_risk(macro)
        assert action == "HOLD"

    def test_no_macro_returns_hold(self):
        action, indicators = _dxy_risk(None)
        assert action == "HOLD"
        assert np.isnan(indicators["dxy_value"])

    def test_partial_macro_missing_dxy_returns_hold(self):
        macro = {"treasury": {"10y": 4.5}}
        action, indicators = _dxy_risk(macro)
        assert action == "HOLD"


class TestYieldCurve:
    """Yield curve inverted triggers SELL."""

    def test_inverted_curve_triggers_sell(self):
        macro = {"treasury": {"spread_2s10s": -0.5}}
        action, indicators = _yield_curve(macro)
        assert action == "SELL"
        assert indicators["yield_curve_2s10s"] == -0.5

    def test_normal_curve_triggers_buy(self):
        macro = {"treasury": {"spread_2s10s": 1.0}}
        action, indicators = _yield_curve(macro)
        assert action == "BUY"

    def test_watch_zone_triggers_hold(self):
        macro = {"treasury": {"spread_2s10s": 0.3}}
        action, indicators = _yield_curve(macro)
        assert action == "HOLD"

    def test_no_macro_returns_hold(self):
        action, indicators = _yield_curve(None)
        assert action == "HOLD"

    def test_exactly_zero_triggers_hold(self):
        """2s10s at exactly 0 is between 0 and 0.5, so HOLD."""
        macro = {"treasury": {"spread_2s10s": 0.0}}
        action, indicators = _yield_curve(macro)
        assert action == "HOLD"


class TestYield10YMomentum:
    """10Y yield levels trigger BUY or SELL."""

    def test_high_yield_triggers_sell(self):
        macro = {"treasury": {"10y": 5.5}}
        action, indicators = _yield_10y_momentum(macro)
        assert action == "SELL"

    def test_low_yield_triggers_buy(self):
        macro = {"treasury": {"10y": 2.8}}
        action, indicators = _yield_10y_momentum(macro)
        assert action == "BUY"

    def test_middle_yield_triggers_hold(self):
        macro = {"treasury": {"10y": 4.2}}
        action, indicators = _yield_10y_momentum(macro)
        assert action == "HOLD"

    def test_no_macro_returns_hold(self):
        action, indicators = _yield_10y_momentum(None)
        assert action == "HOLD"


class TestFOMCProximity:
    """FOMC within 3 days triggers HOLD."""

    def test_within_3_days_triggers_hold(self):
        macro = {"fed": {"days_until": 2}}
        action, indicators = _fomc_proximity(macro)
        assert action == "HOLD"
        assert indicators["fomc_days_until"] == 2.0

    def test_exactly_3_days_triggers_hold(self):
        macro = {"fed": {"days_until": 3}}
        action, indicators = _fomc_proximity(macro)
        assert action == "HOLD"

    def test_far_from_fomc_triggers_hold(self):
        macro = {"fed": {"days_until": 30}}
        action, indicators = _fomc_proximity(macro)
        assert action == "HOLD"

    def test_medium_distance_triggers_hold(self):
        macro = {"fed": {"days_until": 10}}
        action, indicators = _fomc_proximity(macro)
        assert action == "HOLD"

    def test_no_macro_returns_hold(self):
        action, indicators = _fomc_proximity(None)
        assert action == "HOLD"


class TestGoldenDeathCross:
    """Golden cross and death cross regime detection."""

    def test_golden_cross_detected(self):
        df = _make_golden_cross_df()
        action, indicators = _golden_death_cross(df)
        assert action == "BUY"
        assert indicators["golden_cross"] is True
        assert indicators["death_cross"] is False
        # 50-SMA should be above 200-SMA
        assert indicators["sma50"] > indicators["sma200_cross"]

    def test_death_cross_detected(self):
        df = _make_death_cross_df()
        action, indicators = _golden_death_cross(df)
        assert action == "SELL"
        assert indicators["death_cross"] is True
        assert indicators["golden_cross"] is False
        # 50-SMA should be below 200-SMA
        assert indicators["sma50"] < indicators["sma200_cross"]

    def test_insufficient_data_returns_hold(self):
        df = _make_df(n=100)
        action, indicators = _golden_death_cross(df)
        assert action == "HOLD"


class TestNoMacroGraceful:
    """No macro data is handled gracefully across the full composite."""

    def test_none_macro(self):
        df = _make_df(n=250)
        result = compute_macro_regime_signal(df, macro=None)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        # All macro-dependent sub-signals should be HOLD
        assert result["sub_signals"]["dxy_risk"] == "HOLD"
        assert result["sub_signals"]["yield_curve"] == "HOLD"
        assert result["sub_signals"]["yield_10y_momentum"] == "HOLD"
        assert result["sub_signals"]["fomc_proximity"] == "HOLD"

    def test_empty_macro(self):
        df = _make_df(n=250)
        result = compute_macro_regime_signal(df, macro={})
        assert result["sub_signals"]["dxy_risk"] == "HOLD"
        assert result["sub_signals"]["yield_curve"] == "HOLD"
        assert result["sub_signals"]["yield_10y_momentum"] == "HOLD"
        assert result["sub_signals"]["fomc_proximity"] == "HOLD"

    def test_partial_macro_only_dxy(self):
        df = _make_df(n=250)
        macro = {"dxy": {"value": 105.0, "change_5d_pct": 1.5}}
        result = compute_macro_regime_signal(df, macro=macro)
        assert result["sub_signals"]["dxy_risk"] == "SELL"
        assert result["sub_signals"]["yield_curve"] == "HOLD"
        assert result["sub_signals"]["yield_10y_momentum"] == "HOLD"
        assert result["sub_signals"]["fomc_proximity"] == "HOLD"


class TestVotingLogic:
    """Verify majority voting and confidence calculations."""

    def test_all_hold_gives_zero_confidence(self):
        df = _make_df(n=5)  # insufficient for SMA signals
        result = compute_macro_regime_signal(df, macro=None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_confidence_is_majority_fraction(self):
        """With uptrend data + favorable macro, expect BUY majority."""
        df = _make_golden_cross_df()
        macro = {
            "dxy": {"value": 99.0, "change_5d_pct": -1.0},   # BUY
            "treasury": {"spread_2s10s": 1.0, "10y": 3.0},           # BUY, BUY
            "fed": {"days_until": 30},                          # BUY
        }
        result = compute_macro_regime_signal(df, macro=macro)
        # sma200_regime=BUY, dxy=BUY, yield_curve=BUY, yield_10y=BUY,
        # fomc=BUY, golden_death_cross=BUY => 6 BUY, 0 SELL
        assert result["action"] == "BUY"
        assert result["confidence"] == 1.0

    def test_tie_returns_hold(self):
        """When BUY and SELL are tied, action should be HOLD."""
        # Craft macro so we get exactly 1 BUY and 1 SELL from macro sub-signals
        df = _make_df(n=5)  # Not enough for SMA signals -> both SMA subs = HOLD
        macro = {
            "dxy": {"value": 105.0, "change_5d_pct": 1.5},   # SELL
            "treasury": {"spread_2s10s": 1.0, "10y": 4.2},           # yield_curve=BUY, 10y=HOLD
            "fed": {"days_until": 10},                          # HOLD
        }
        result = compute_macro_regime_signal(df, macro=macro)
        # sma200=HOLD, dxy=SELL, yield_curve=BUY, 10y=HOLD, fomc=HOLD, gdc=HOLD
        # 1 BUY, 1 SELL -> tie -> HOLD
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_sell_majority(self):
        """Strong bearish macro should produce SELL majority."""
        df = _make_death_cross_df()  # sma200=SELL, gdc=SELL
        macro = {
            "dxy": {"value": 106.0, "change_5d_pct": 2.0},   # SELL
            "treasury": {"spread_2s10s": -0.5, "10y": 5.5},          # SELL, SELL
            "fed": {"days_until": 2},                           # HOLD
        }
        result = compute_macro_regime_signal(df, macro=macro)
        # sma200=SELL, dxy=SELL, yield_curve=SELL, 10y=SELL, fomc=HOLD, gdc=SELL
        # 5 SELL, 0 BUY
        assert result["action"] == "SELL"
        assert result["confidence"] == 1.0
