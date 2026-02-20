"""Tests for the macro-regime signal module.

Covers:
- Basic computation with sufficient data and no macro context
- Insufficient data returns HOLD
- Adaptive SMA regime (works with 50-200+ bars)
- DXY rising triggers SELL (±0.3% threshold)
- Yield curve inverted triggers SELL
- 10Y yield momentum (direction-based, ±1.5% change_5d)
- FOMC within 2 days triggers SELL (risk-off)
- Adaptive golden/death cross (20/50 or 50/200 periods)
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
    _sma_regime,
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
    """Create a synthetic OHLCV DataFrame."""
    np.random.seed(42)
    noise = np.random.randn(n) * volatility
    close = close_base + np.cumsum(noise) + np.arange(n) * trend
    close = np.maximum(close, 1.0)
    high = close + np.abs(np.random.randn(n) * volatility)
    low = close - np.abs(np.random.randn(n) * volatility)
    low = np.maximum(low, 0.5)
    opn = close + np.random.randn(n) * 0.3
    volume = np.random.randint(100, 10000, n).astype(float)
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame({
        "open": opn, "high": high, "low": low,
        "close": close, "volume": volume, "time": dates,
    })


def _make_df_above_sma(n: int = 250) -> pd.DataFrame:
    """Create a DataFrame where the latest close is well above the SMA."""
    np.random.seed(42)
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


def _make_df_below_sma(n: int = 250) -> pd.DataFrame:
    """Create a DataFrame where the latest close is well below the SMA."""
    np.random.seed(42)
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
    """Create a DataFrame exhibiting golden cross: fast SMA > slow SMA, price > fast."""
    np.random.seed(42)
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
    """Create a DataFrame exhibiting death cross: fast SMA < slow SMA, price < fast."""
    np.random.seed(42)
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
            "sma_regime", "dxy_risk", "yield_curve",
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
        assert not np.isnan(result["indicators"]["sma_value"])
        assert result["indicators"]["sma_period"] == 200

    def test_no_macro_means_macro_sub_signals_hold(self):
        df = _make_df(n=250)
        result = compute_macro_regime_signal(df, macro=None)
        assert result["sub_signals"]["dxy_risk"] == "HOLD"
        assert result["sub_signals"]["yield_curve"] == "HOLD"
        assert result["sub_signals"]["yield_10y_momentum"] == "HOLD"
        assert result["sub_signals"]["fomc_proximity"] == "HOLD"


class TestInsufficientData:
    def test_very_short_df(self):
        df = _make_df(n=5)
        result = compute_macro_regime_signal(df)
        assert result["sub_signals"]["sma_regime"] == "HOLD"
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


class TestSMARegime:
    """Adaptive SMA regime: works with 50-200+ bars."""

    def test_price_above_sma_triggers_buy(self):
        df = _make_df_above_sma()
        action, indicators = _sma_regime(df)
        assert action == "BUY"
        assert indicators["price_vs_sma_pct"] > 0.01

    def test_price_below_sma_triggers_sell(self):
        df = _make_df_below_sma()
        action, indicators = _sma_regime(df)
        assert action == "SELL"
        assert indicators["price_vs_sma_pct"] < -0.01

    def test_200_bars_uses_sma200(self):
        df = _make_df(n=250)
        action, indicators = _sma_regime(df)
        assert indicators["sma_period"] == 200

    def test_100_bars_uses_adaptive_sma(self):
        """100 bars (Now TF typical) should use SMA(100), not HOLD."""
        df = _make_df_above_sma(n=100)
        action, indicators = _sma_regime(df)
        assert indicators["sma_period"] == 100
        assert action in ("BUY", "SELL", "HOLD")

    def test_50_bars_uses_sma50(self):
        df = _make_df_above_sma(n=50)
        action, indicators = _sma_regime(df)
        assert indicators["sma_period"] == 50

    def test_30_bars_returns_hold(self):
        """Below 50 bars minimum, should return HOLD."""
        df = _make_df(n=30)
        action, indicators = _sma_regime(df)
        assert action == "HOLD"

    def test_composite_buy_with_uptrend(self):
        df = _make_df_above_sma()
        result = compute_macro_regime_signal(df)
        assert result["sub_signals"]["sma_regime"] == "BUY"


class TestDXYRisk:
    """DXY thresholds at ±0.3%."""

    def test_strong_dollar_triggers_sell(self):
        macro = {"dxy": {"value": 105.0, "change_5d_pct": 1.2}}
        action, indicators = _dxy_risk(macro)
        assert action == "SELL"

    def test_weak_dollar_triggers_buy(self):
        macro = {"dxy": {"value": 99.0, "change_5d_pct": -0.8}}
        action, indicators = _dxy_risk(macro)
        assert action == "BUY"

    def test_borderline_above_triggers_sell(self):
        """0.32% > 0.3 threshold → SELL (was HOLD at old 0.5 threshold)."""
        macro = {"dxy": {"value": 97.0, "change_5d_pct": 0.32}}
        action, indicators = _dxy_risk(macro)
        assert action == "SELL"

    def test_neutral_dollar_triggers_hold(self):
        macro = {"dxy": {"value": 102.0, "change_5d_pct": 0.1}}
        action, indicators = _dxy_risk(macro)
        assert action == "HOLD"

    def test_no_macro_returns_hold(self):
        action, indicators = _dxy_risk(None)
        assert action == "HOLD"


class TestYieldCurve:
    def test_inverted_curve_triggers_sell(self):
        macro = {"treasury": {"spread_2s10s": -0.5}}
        action, indicators = _yield_curve(macro)
        assert action == "SELL"

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


class TestYield10YMomentum:
    """10Y yield direction-based (change_5d ±1.5%)."""

    def test_rising_yields_triggers_sell(self):
        macro = {"treasury": {"10y": {"yield_pct": 4.5, "change_5d": 2.0}}}
        action, indicators = _yield_10y_momentum(macro)
        assert action == "SELL"
        assert indicators["treasury_10y_change_5d"] == 2.0

    def test_falling_yields_triggers_buy(self):
        macro = {"treasury": {"10y": {"yield_pct": 4.0, "change_5d": -2.29}}}
        action, indicators = _yield_10y_momentum(macro)
        assert action == "BUY"
        assert indicators["treasury_10y"] == 4.0

    def test_flat_yields_triggers_hold(self):
        macro = {"treasury": {"10y": {"yield_pct": 4.2, "change_5d": 0.5}}}
        action, indicators = _yield_10y_momentum(macro)
        assert action == "HOLD"

    def test_no_macro_returns_hold(self):
        action, indicators = _yield_10y_momentum(None)
        assert action == "HOLD"

    def test_missing_change_5d_returns_hold(self):
        macro = {"treasury": {"10y": {"yield_pct": 4.0}}}
        action, indicators = _yield_10y_momentum(macro)
        assert action == "HOLD"


class TestFOMCProximity:
    """FOMC within 2 days → SELL (risk-off), >2 days → HOLD."""

    def test_within_2_days_triggers_sell(self):
        macro = {"fed": {"days_until": 1}}
        action, indicators = _fomc_proximity(macro)
        assert action == "SELL"

    def test_exactly_2_days_triggers_sell(self):
        macro = {"fed": {"days_until": 2}}
        action, indicators = _fomc_proximity(macro)
        assert action == "SELL"
        assert indicators["fomc_days_until"] == 2.0

    def test_3_days_triggers_hold(self):
        macro = {"fed": {"days_until": 3}}
        action, indicators = _fomc_proximity(macro)
        assert action == "HOLD"

    def test_far_from_fomc_triggers_hold(self):
        macro = {"fed": {"days_until": 30}}
        action, indicators = _fomc_proximity(macro)
        assert action == "HOLD"

    def test_no_macro_returns_hold(self):
        action, indicators = _fomc_proximity(None)
        assert action == "HOLD"


class TestGoldenDeathCross:
    """Adaptive golden/death cross: 50/200 with 200+ bars, 20/50 with 50-199."""

    def test_golden_cross_detected_250_bars(self):
        df = _make_golden_cross_df()
        action, indicators = _golden_death_cross(df)
        assert action == "BUY"
        assert indicators["golden_cross"] is True
        assert indicators["sma_fast_period"] == 50
        assert indicators["sma_slow_period"] == 200
        assert indicators["sma_fast"] > indicators["sma_slow"]

    def test_death_cross_detected_250_bars(self):
        df = _make_death_cross_df()
        action, indicators = _golden_death_cross(df)
        assert action == "SELL"
        assert indicators["death_cross"] is True
        assert indicators["sma_fast"] < indicators["sma_slow"]

    def test_100_bars_uses_20_50_periods(self):
        """100 bars (Now TF) should use SMA(20)/SMA(50) instead of HOLD."""
        df = _make_df_above_sma(n=100)
        action, indicators = _golden_death_cross(df)
        assert indicators["sma_fast_period"] == 20
        assert indicators["sma_slow_period"] == 50
        assert action in ("BUY", "SELL", "HOLD")

    def test_30_bars_returns_hold(self):
        df = _make_df(n=30)
        action, indicators = _golden_death_cross(df)
        assert action == "HOLD"


class TestNoMacroGraceful:
    def test_none_macro(self):
        df = _make_df(n=250)
        result = compute_macro_regime_signal(df, macro=None)
        assert result["action"] in ("BUY", "SELL", "HOLD")
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


class TestVotingLogic:
    def test_all_hold_gives_zero_confidence(self):
        df = _make_df(n=5)
        result = compute_macro_regime_signal(df, macro=None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_confidence_is_majority_fraction(self):
        df = _make_golden_cross_df()
        macro = {
            "dxy": {"value": 99.0, "change_5d_pct": -1.0},            # BUY
            "treasury": {
                "spread_2s10s": 1.0,                                    # BUY
                "10y": {"yield_pct": 4.0, "change_5d": -2.0},         # BUY
            },
            "fed": {"days_until": 30},                                  # HOLD
        }
        result = compute_macro_regime_signal(df, macro=macro)
        # sma_regime=BUY, dxy=BUY, yield_curve=BUY, yield_10y=BUY,
        # fomc=HOLD, golden_death_cross=BUY => 5 BUY, 0 SELL
        assert result["action"] == "BUY"
        assert result["confidence"] == 1.0

    def test_tie_returns_hold(self):
        df = _make_df(n=5)
        macro = {
            "dxy": {"value": 105.0, "change_5d_pct": 1.5},            # SELL
            "treasury": {
                "spread_2s10s": 1.0,                                    # BUY
                "10y": {"yield_pct": 4.2, "change_5d": 0.5},          # HOLD
            },
            "fed": {"days_until": 10},                                  # HOLD
        }
        result = compute_macro_regime_signal(df, macro=macro)
        # sma=HOLD, dxy=SELL, yield_curve=BUY, 10y=HOLD, fomc=HOLD, gdc=HOLD
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_sell_majority(self):
        df = _make_death_cross_df()
        macro = {
            "dxy": {"value": 106.0, "change_5d_pct": 2.0},            # SELL
            "treasury": {
                "spread_2s10s": -0.5,                                   # SELL
                "10y": {"yield_pct": 5.5, "change_5d": 3.0},          # SELL
            },
            "fed": {"days_until": 1},                                   # SELL
        }
        result = compute_macro_regime_signal(df, macro=macro)
        # sma=SELL, dxy=SELL, yield_curve=SELL, 10y=SELL, fomc=SELL, gdc=SELL
        assert result["action"] == "SELL"
        assert result["confidence"] == 1.0
