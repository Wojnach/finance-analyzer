"""Tests for autotune_adaptive_cycle signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.autotune_adaptive_cycle import (
    compute_autotune_adaptive_cycle_signal,
    _highpass_filter,
    _autocorrelation_periodogram,
    _adaptive_bandpass,
)


def _make_df(n=200):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_cyclic_df(n=200, period=20):
    """Create DataFrame with embedded sine cycle for testing cycle detection."""
    np.random.seed(123)
    t = np.arange(n)
    close = 100 + 5 * np.sin(2 * np.pi * t / period) + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.2),
        "low": close - abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_autotune_adaptive_cycle_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_autotune_adaptive_cycle_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        result = compute_autotune_adaptive_cycle_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_autotune_adaptive_cycle_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_autotune_adaptive_cycle_signal(df)
        assert result["action"] == "HOLD"

    def test_none_dataframe_returns_hold(self):
        result = compute_autotune_adaptive_cycle_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_autotune_adaptive_cycle_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_autotune_adaptive_cycle_signal(df, context=ctx)
        assert isinstance(result, dict)


class TestIndicators:
    """Test that indicators are computed correctly."""

    def test_dominant_cycle_in_range(self):
        df = _make_df(n=200)
        result = compute_autotune_adaptive_cycle_signal(df)
        if result["indicators"]:
            dc = result["indicators"].get("dominant_cycle", 0)
            assert 8 <= dc <= 48

    def test_min_correlation_range(self):
        df = _make_df(n=200)
        result = compute_autotune_adaptive_cycle_signal(df)
        if result["indicators"]:
            mc = result["indicators"].get("min_correlation", 0)
            assert -1.0 <= mc <= 1.0

    def test_confidence_capped_at_07(self):
        df = _make_df(n=200)
        result = compute_autotune_adaptive_cycle_signal(df)
        assert result["confidence"] <= 0.7

    def test_bandpass_roc_zscore_nonneg(self):
        df = _make_df(n=200)
        result = compute_autotune_adaptive_cycle_signal(df)
        if result["indicators"]:
            zscore = result["indicators"].get("bandpass_roc_zscore", 0)
            assert zscore >= 0


class TestCycleDetection:
    """Test that the cycle detection works on synthetic data."""

    def test_detects_embedded_cycle(self):
        df = _make_cyclic_df(n=300, period=20)
        result = compute_autotune_adaptive_cycle_signal(df)
        dc = result["indicators"].get("dominant_cycle", 0)
        assert 14 <= dc <= 28, f"Expected ~20, got {dc}"

    def test_strong_cycle_has_negative_correlation(self):
        df = _make_cyclic_df(n=300, period=20)
        result = compute_autotune_adaptive_cycle_signal(df)
        mc = result["indicators"].get("min_correlation", 0)
        assert mc < -0.3, f"Expected strong negative correlation, got {mc}"


class TestInternalFunctions:
    """Test internal helper functions."""

    def test_highpass_filter_removes_trend(self):
        n = 200
        trend = np.linspace(100, 200, n)
        hp = _highpass_filter(trend, 48.0)
        assert abs(hp[-1]) < 10.0

    def test_adaptive_bandpass_output_shape(self):
        data = np.random.randn(100)
        bp = _adaptive_bandpass(data, 20.0)
        assert len(bp) == 100

    def test_autocorrelation_returns_valid_period(self):
        np.random.seed(42)
        n = 200
        t = np.arange(n)
        signal = np.sin(2 * np.pi * t / 16) + np.random.randn(n) * 0.2
        hp = _highpass_filter(signal, 48.0)
        dc, mc = _autocorrelation_periodogram(hp, 50, 8, 48)
        assert 8 <= dc <= 48


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_constant_price_returns_hold(self):
        n = 100
        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.0),
            "low": np.full(n, 100.0),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 1000.0),
        })
        result = compute_autotune_adaptive_cycle_signal(df)
        assert result["action"] == "HOLD"

    def test_large_dataframe(self):
        df = _make_df(n=1000)
        result = compute_autotune_adaptive_cycle_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_minimum_rows(self):
        df = _make_df(n=60)
        result = compute_autotune_adaptive_cycle_signal(df)
        assert isinstance(result, dict)
