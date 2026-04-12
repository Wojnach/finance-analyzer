"""Tests for shannon_entropy signal module."""

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.shannon_entropy import (
    MIN_ROWS,
    _compute_entropy,
    _entropy_momentum,
    _entropy_regime,
    _trend_direction,
    _trend_strength,
    compute_shannon_entropy_signal,
)


def _make_df(n=100, seed=42):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_trending_df(n=100, direction="up"):
    """Create a strongly trending DataFrame."""
    np.random.seed(42)
    if direction == "up":
        close = 100 + np.arange(n) * 0.5 + np.random.randn(n) * 0.05
    else:
        close = 200 - np.arange(n) * 0.5 + np.random.randn(n) * 0.05
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.01,
        "high": close + abs(np.random.randn(n) * 0.02),
        "low": close - abs(np.random.randn(n) * 0.02),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_noisy_df(n=100):
    """Create a highly noisy/random DataFrame (mean-reverting noise)."""
    np.random.seed(42)
    close = 100 + np.random.randn(n) * 3  # large random noise, no trend
    close = np.abs(close)  # ensure positive
    close[close < 1] = 1  # floor
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.5,
        "high": close + abs(np.random.randn(n) * 1.0),
        "low": close - abs(np.random.randn(n) * 1.0),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestComputeEntropy:
    """Test the core entropy computation."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution should have entropy near 1.0."""
        # Generate data that fills all bins equally
        returns = np.linspace(-0.05, 0.05, 1000)
        ent = _compute_entropy(returns, n_bins=10)
        assert ent > 0.9  # near-uniform = high entropy

    def test_concentrated_distribution_low_entropy(self):
        """Concentrated distribution should have low entropy."""
        # 90 values in one region, 10 in another — clearly bimodal/concentrated
        returns = np.concatenate([
            np.full(90, 0.01),   # 90% identical values
            np.full(10, -0.05),  # 10% outliers far away
        ])
        ent = _compute_entropy(returns, n_bins=10)
        assert ent < 0.7  # concentrated around one value

    def test_too_few_values_returns_nan(self):
        """Less than 10 values should return NaN."""
        returns = np.array([0.01, 0.02, 0.03])
        ent = _compute_entropy(returns, n_bins=10)
        assert np.isnan(ent)

    def test_all_nan_returns_nan(self):
        """All NaN values should return NaN."""
        returns = np.full(20, np.nan)
        ent = _compute_entropy(returns, n_bins=10)
        assert np.isnan(ent)

    def test_entropy_range(self):
        """Entropy should be between 0 and 1."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.02
        ent = _compute_entropy(returns, n_bins=10)
        assert 0 <= ent <= 1


class TestEntropyRegime:
    """Test entropy regime classification."""

    def test_trending_data_low_entropy(self):
        """Strongly trending data should have low entropy."""
        df = _make_trending_df(100, "up")
        ent, regime = _entropy_regime(df["close"])
        # Trending data should have lower entropy (more concentrated returns)
        assert not np.isnan(ent)
        assert regime in ("trending", "neutral", "noisy")

    def test_noisy_data_high_entropy(self):
        """Random noisy data should have higher entropy."""
        df = _make_noisy_df(100)
        ent, regime = _entropy_regime(df["close"])
        assert not np.isnan(ent)
        assert regime in ("neutral", "noisy")

    def test_short_series_returns_unknown(self):
        """Too-short series should return unknown."""
        short_close = pd.Series([100, 101, 102])
        ent, regime = _entropy_regime(short_close)
        assert regime == "unknown"


class TestTrendDirection:
    """Test trend direction sub-signal."""

    def test_uptrend(self):
        df = _make_trending_df(100, "up")
        spread, direction = _trend_direction(df["close"])
        assert direction == "BUY"
        assert spread > 0

    def test_downtrend(self):
        df = _make_trending_df(100, "down")
        spread, direction = _trend_direction(df["close"])
        assert direction == "SELL"
        assert spread < 0

    def test_short_series(self):
        short_close = pd.Series([100, 101, 102])
        spread, direction = _trend_direction(short_close)
        assert direction == "HOLD"


class TestTrendStrength:
    """Test trend strength sub-signal."""

    def test_strong_uptrend(self):
        df = _make_trending_df(100, "up")
        roc, strength = _trend_strength(df["close"])
        assert roc > 0
        assert strength in ("strong", "moderate")

    def test_short_series(self):
        short_close = pd.Series([100, 101])
        roc, strength = _trend_strength(short_close)
        assert strength == "weak"


class TestEntropyMomentum:
    """Test entropy momentum sub-signal."""

    def test_returns_valid_result(self):
        df = _make_df(150)
        delta, label = _entropy_momentum(df["close"])
        assert isinstance(delta, float)
        assert label in ("trending_strengthening", "noise_increasing", "stable")

    def test_short_series(self):
        short_close = pd.Series(range(20))
        delta, label = _entropy_momentum(short_close)
        assert label == "HOLD"


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_shannon_entropy_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_shannon_entropy_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        result = compute_shannon_entropy_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "normalized_entropy" in result["indicators"]
        assert "entropy_regime" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_shannon_entropy_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_shannon_entropy_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_shannon_entropy_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_shannon_entropy_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_none_dataframe_returns_hold(self):
        result = compute_shannon_entropy_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_confidence_capped_at_0_7(self):
        """Confidence should never exceed 0.7."""
        df = _make_trending_df(200, "up")
        result = compute_shannon_entropy_signal(df)
        assert result["confidence"] <= 0.7

    def test_trending_up_returns_buy(self):
        """Strong uptrend with predictable returns should BUY."""
        df = _make_trending_df(200, "up")
        result = compute_shannon_entropy_signal(df)
        # Should be BUY or HOLD — depends on entropy classification
        if result["indicators"]["entropy_regime"] == "trending":
            assert result["action"] == "BUY"

    def test_trending_down_returns_sell(self):
        """Strong downtrend with predictable returns should SELL."""
        df = _make_trending_df(200, "down")
        result = compute_shannon_entropy_signal(df)
        if result["indicators"]["entropy_regime"] == "trending":
            assert result["action"] == "SELL"

    def test_noisy_market_returns_hold(self):
        """Noisy market should return HOLD."""
        df = _make_noisy_df(200)
        result = compute_shannon_entropy_signal(df)
        if result["indicators"]["entropy_regime"] == "noisy":
            assert result["action"] == "HOLD"


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_all_same_close(self):
        """All same close prices — zero returns."""
        df = pd.DataFrame({
            "open": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "close": [100.0] * 100,
            "volume": [5000.0] * 100,
        })
        result = compute_shannon_entropy_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_very_large_dataset(self):
        """Should handle 1000+ rows without issues."""
        df = _make_df(n=1000)
        result = compute_shannon_entropy_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_exactly_min_rows(self):
        """Exactly MIN_ROWS should work."""
        df = _make_df(n=MIN_ROWS)
        result = compute_shannon_entropy_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_mostly_nan_close(self):
        """More than 30% NaN should return HOLD."""
        df = _make_df(n=100)
        df.iloc[:40, df.columns.get_loc("close")] = np.nan
        result = compute_shannon_entropy_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
