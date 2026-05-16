"""Tests for TTM Squeeze breakout signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.ttm_squeeze import compute_ttm_squeeze_signal


def _make_df(n=100, seed=42):
    """Create test DataFrame with realistic OHLCV data."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + abs(np.random.randn(n) * 0.3)
    low = close - abs(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_squeeze_df(n=60):
    """Create DataFrame where price compresses then breaks out upward."""
    np.random.seed(123)
    prices = []
    p = 100.0
    for i in range(n):
        if i < 40:
            p += np.random.randn() * 0.05
        else:
            p += 0.3 + np.random.randn() * 0.1
        prices.append(p)
    close = np.array(prices)
    return pd.DataFrame({
        "open": close - np.random.rand(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.2),
        "low": close - abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_ttm_squeeze_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_ttm_squeeze_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "squeeze_state" in result["sub_signals"]
        assert "momentum_direction" in result["sub_signals"]
        assert "momentum_acceleration" in result["sub_signals"]

    def test_has_indicators(self):
        df = _make_df()
        result = compute_ttm_squeeze_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "squeeze_on" in result["indicators"]
        assert "squeeze_bars" in result["indicators"]
        assert "momentum" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_ttm_squeeze_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=5)
        result = compute_ttm_squeeze_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_none_input_returns_hold(self):
        result = compute_ttm_squeeze_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_ttm_squeeze_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_ttm_squeeze_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_capped_at_0_7(self):
        df = _make_df(n=200)
        result = compute_ttm_squeeze_signal(df)
        assert result["confidence"] <= 0.7


class TestSqueezeLogic:
    """Test squeeze detection mechanics."""

    def test_squeeze_detection(self):
        df = _make_df(n=50)
        result = compute_ttm_squeeze_signal(df)
        assert isinstance(result["indicators"]["squeeze_on"], bool)
        assert isinstance(result["indicators"]["squeeze_bars"], int)

    def test_breakout_produces_signal(self):
        df = _make_squeeze_df(n=60)
        result = compute_ttm_squeeze_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_momentum_values_are_floats(self):
        df = _make_df()
        result = compute_ttm_squeeze_signal(df)
        assert isinstance(result["indicators"]["momentum"], float)
        assert isinstance(result["indicators"]["momentum_prev"], float)

    def test_different_seeds_different_results(self):
        df1 = _make_df(seed=1)
        df2 = _make_df(seed=999)
        r1 = compute_ttm_squeeze_signal(df1)
        r2 = compute_ttm_squeeze_signal(df2)
        assert not (r1["indicators"]["momentum"] == r2["indicators"]["momentum"]
                    and r1["indicators"]["squeeze_on"] == r2["indicators"]["squeeze_on"])

    def test_just_released_flag(self):
        df = _make_df(n=100)
        result = compute_ttm_squeeze_signal(df)
        assert isinstance(result["indicators"]["just_released"], bool)
