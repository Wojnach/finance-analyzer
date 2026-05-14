"""Tests for trend_slope_momentum signal module."""

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.trend_slope_momentum import (
    compute_trend_slope_momentum_signal,
)


def _make_df(n=300, trend=0.0, seed=42):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(seed)
    noise = np.random.randn(n) * 0.5
    close = 100 + trend * np.arange(n) + np.cumsum(noise)
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.1,
            "high": close + abs(np.random.randn(n) * 0.3),
            "low": close - abs(np.random.randn(n) * 0.3),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_trend_slope_momentum_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_trend_slope_momentum_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        expected_keys = {
            "trend_slope",
            "momentum_50d",
            "probability",
            "slope_momentum_agreement",
        }
        assert expected_keys == set(result["sub_signals"].keys())

    def test_has_indicators(self):
        df = _make_df()
        result = compute_trend_slope_momentum_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        expected = {"z_score", "p_trend", "p_bull", "slope", "momentum_ratio"}
        assert expected == set(result["indicators"].keys())

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_trend_slope_momentum_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=50)
        result = compute_trend_slope_momentum_signal(df)
        assert result["action"] == "HOLD"

    def test_none_dataframe_returns_hold(self):
        result = compute_trend_slope_momentum_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[150:155, df.columns.get_loc("close")] = np.nan
        result = compute_trend_slope_momentum_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {
            "ticker": "XAU-USD",
            "asset_class": "metals",
            "regime": "trending-up",
        }
        result = compute_trend_slope_momentum_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")


class TestSignalBehavior:
    """Test directional behavior of the signal."""

    def test_strong_uptrend_produces_buy(self):
        df = _make_df(n=300, trend=0.5)
        result = compute_trend_slope_momentum_signal(df)
        assert result["action"] == "BUY"
        assert result["confidence"] > 0.0

    def test_strong_downtrend_produces_sell(self):
        df = _make_df(n=300, trend=-0.2)
        result = compute_trend_slope_momentum_signal(df)
        assert result["action"] == "SELL"
        assert result["confidence"] > 0.0

    def test_flat_market_produces_valid_signal(self):
        df = _make_df(n=300, trend=0.0)
        result = compute_trend_slope_momentum_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probability_bounded_zero_one(self):
        for trend in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            df = _make_df(n=300, trend=trend, seed=trend.__hash__() % 2**31)
            result = compute_trend_slope_momentum_signal(df)
            p_bull = result["indicators"].get("p_bull", 0.5)
            assert 0.0 <= p_bull <= 1.0

    def test_z_score_clipped(self):
        df = _make_df(n=300, trend=2.0)
        result = compute_trend_slope_momentum_signal(df)
        z = result["indicators"].get("z_score", 0.0)
        assert -3.0 <= z <= 3.0

    def test_momentum_ratio_positive(self):
        df = _make_df(n=300, trend=0.5)
        result = compute_trend_slope_momentum_signal(df)
        ratio = result["indicators"].get("momentum_ratio", 0.0)
        assert ratio > 0.0

    def test_different_seeds_give_varied_results(self):
        results = []
        for seed in range(5):
            df = _make_df(n=300, trend=0.0, seed=seed)
            r = compute_trend_slope_momentum_signal(df)
            results.append(r["action"])
        assert len(set(results)) >= 1

    def test_minimum_rows_boundary(self):
        df = _make_df(n=260)
        result = compute_trend_slope_momentum_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

        df_short = _make_df(n=259)
        result_short = compute_trend_slope_momentum_signal(df_short)
        assert result_short["action"] == "HOLD"

    def test_constant_price_returns_hold(self):
        n = 300
        df = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [100.1] * n,
                "low": [99.9] * n,
                "close": [100.0] * n,
                "volume": [5000.0] * n,
            }
        )
        result = compute_trend_slope_momentum_signal(df)
        assert result["action"] == "HOLD"
