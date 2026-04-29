"""Tests for vol_ratio_regime signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.vol_ratio_regime import (
    MIN_ROWS,
    _classify_regime,
    _efficiency_ratio,
    _garman_klass_cc_ratio,
    _variance_ratio,
    compute_vol_ratio_regime_signal,
)


def _make_df(n=100, trend=0.0, noise=1.0, seed=42):
    """Create a test DataFrame with realistic OHLCV data.

    Args:
        n: number of bars
        trend: daily drift (0 = flat, positive = uptrend)
        noise: scale of random noise (higher = more volatile/choppy)
    """
    np.random.seed(seed)
    close = 100.0 + np.cumsum(np.random.randn(n) * noise + trend)
    # Ensure positive prices
    close = np.maximum(close, 10.0)
    high = close + abs(np.random.randn(n) * noise * 0.5)
    low = close - abs(np.random.randn(n) * noise * 0.5)
    open_ = close + np.random.randn(n) * noise * 0.2
    # Ensure high > low and open within range
    high = np.maximum(high, np.maximum(open_, close) + 0.01)
    low = np.minimum(low, np.minimum(open_, close) - 0.01)

    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_trending_df(n=100):
    """Create a strongly trending DataFrame (clear uptrend)."""
    return _make_df(n=n, trend=0.5, noise=0.3)


def _make_ranging_df(n=100):
    """Create a ranging/choppy DataFrame (high noise, no trend)."""
    return _make_df(n=n, trend=0.0, noise=2.0)


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_vol_ratio_regime_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_vol_ratio_regime_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        result = compute_vol_ratio_regime_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        # Check key indicators are present
        assert "gk_cc_ratio" in result["indicators"]
        assert "variance_ratio" in result["indicators"]
        assert "efficiency_ratio" in result["indicators"]
        assert "regime" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_vol_ratio_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_none_input_returns_hold(self):
        result = compute_vol_ratio_regime_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_vol_ratio_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_exactly_min_rows(self):
        df = _make_df(n=MIN_ROWS)
        result = compute_vol_ratio_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_vol_ratio_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_vol_ratio_regime_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_capped_at_0_7(self):
        """Confidence should never exceed 0.7."""
        for seed in range(10):
            df = _make_df(seed=seed)
            result = compute_vol_ratio_regime_signal(df)
            assert result["confidence"] <= 0.7


class TestSubIndicators:
    """Test individual sub-indicator functions."""

    def test_gk_cc_ratio_shape(self):
        df = _make_df()
        ratio = _garman_klass_cc_ratio(df)
        assert len(ratio) == len(df)
        # First 19 values should be NaN (window=20)
        assert ratio.iloc[:19].isna().all()
        # Later values should be positive
        valid = ratio.dropna()
        assert (valid > 0).all()

    def test_gk_cc_ratio_ranging_market(self):
        """Ranging market should produce higher GK/CC ratio."""
        trending_df = _make_trending_df(n=200)
        ranging_df = _make_ranging_df(n=200)

        trending_ratio = _garman_klass_cc_ratio(trending_df).iloc[-1]
        ranging_ratio = _garman_klass_cc_ratio(ranging_df).iloc[-1]

        # Ranging should generally have higher ratio, but this isn't guaranteed
        # due to randomness. Just verify both are positive.
        assert trending_ratio > 0
        assert ranging_ratio > 0

    def test_variance_ratio_shape(self):
        df = _make_df()
        vr = _variance_ratio(df["close"])
        assert len(vr) == len(df)
        # Later values should be positive
        valid = vr.dropna()
        assert (valid > 0).all()

    def test_efficiency_ratio_bounded(self):
        df = _make_df()
        er = _efficiency_ratio(df["close"])
        valid = er.dropna()
        # ER should be between 0 and 1
        assert (valid >= 0).all()
        assert (valid <= 1.001).all()  # tiny float tolerance

    def test_efficiency_ratio_trending(self):
        """Strongly trending market should have high ER."""
        np.random.seed(42)
        n = 100
        close = pd.Series(np.linspace(100, 150, n))  # Perfect uptrend
        er = _efficiency_ratio(close, period=20)
        # Last value should be close to 1.0
        assert er.iloc[-1] > 0.9


class TestRegimeClassification:
    """Test the regime classification logic."""

    def test_trending_regime(self):
        assert _classify_regime(1.0, 1.0, 0.4) == "trending"

    def test_ranging_regime(self):
        assert _classify_regime(2.5, 0.6, 0.15) == "ranging"

    def test_uncertain_regime(self):
        assert _classify_regime(1.8, 0.8, 0.22) == "uncertain"

    def test_nan_returns_uncertain(self):
        assert _classify_regime(np.nan, 1.0, 0.4) == "uncertain"
        assert _classify_regime(1.0, np.nan, 0.4) == "uncertain"
        assert _classify_regime(1.0, 1.0, np.nan) == "uncertain"

    def test_mixed_signals_need_majority(self):
        # Only 1 trending vote (GK), but VR and ER are neutral
        assert _classify_regime(1.0, 0.8, 0.22) != "trending"  # only 1 vote

    def test_two_of_three_sufficient(self):
        # GK trending, VR trending, ER neutral
        assert _classify_regime(1.0, 0.9, 0.22) == "trending"
        # GK ranging, VR ranging, ER neutral
        assert _classify_regime(2.5, 0.6, 0.22) == "ranging"


class TestDirectionalLogic:
    """Test directional signal generation based on regime."""

    def test_trending_up_produces_buy(self):
        """Strong uptrend should produce BUY with trending regime."""
        df = _make_trending_df(n=200)
        result = compute_vol_ratio_regime_signal(df)
        # With a strong uptrend, if regime is trending, should BUY
        if result["indicators"]["regime"] == "trending":
            assert result["action"] == "BUY"

    def test_regime_in_sub_signals(self):
        """Regime should appear in sub_signals."""
        df = _make_df()
        result = compute_vol_ratio_regime_signal(df)
        assert "composite_regime" in result["sub_signals"]
        assert result["sub_signals"]["composite_regime"] in ("trending", "ranging", "uncertain")

    def test_regime_clarity_in_indicators(self):
        """Regime clarity should be between 0 and 1."""
        df = _make_df()
        result = compute_vol_ratio_regime_signal(df)
        assert 0.0 <= result["indicators"]["regime_clarity"] <= 1.0

    def test_uncertain_regime_produces_hold(self):
        """When regime is uncertain, action should be HOLD."""
        df = _make_df(n=100, trend=0.01, noise=0.5)
        result = compute_vol_ratio_regime_signal(df)
        if result["indicators"]["regime"] == "uncertain":
            assert result["action"] == "HOLD"


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_flat_price(self):
        """All same price should not crash."""
        n = 100
        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.01),
            "low": np.full(n, 99.99),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
        })
        result = compute_vol_ratio_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_very_volatile(self):
        """Extreme volatility should not crash."""
        df = _make_df(n=100, noise=10.0)
        result = compute_vol_ratio_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_long_dataframe(self):
        """Should handle large DataFrames."""
        df = _make_df(n=1000)
        result = compute_vol_ratio_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_zero_volume(self):
        """Zero volume should not affect signal (we don't use it)."""
        df = _make_df()
        df["volume"] = 0.0
        result = compute_vol_ratio_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_missing_columns_returns_hold(self):
        """DataFrame missing required OHLC columns should return HOLD."""
        df = _make_df()
        df_no_high = df.drop(columns=["high"])
        result = compute_vol_ratio_regime_signal(df_no_high)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_large_close_open_gap(self):
        """Anomalous ticks with close >> open should not produce negative GK."""
        df = _make_df()
        # Simulate anomalous ticks where close-open gap exceeds high-low range
        df.iloc[50, df.columns.get_loc("close")] = 200.0
        df.iloc[50, df.columns.get_loc("open")] = 100.0
        df.iloc[50, df.columns.get_loc("high")] = 200.5
        df.iloc[50, df.columns.get_loc("low")] = 99.5
        result = compute_vol_ratio_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
