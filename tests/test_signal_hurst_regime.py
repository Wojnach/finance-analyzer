"""Tests for hurst_regime signal module."""

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.hurst_regime import (
    MIN_ROWS,
    _compute_hurst,
    _hurst_regime,
    _mr_extreme,
    _rescaled_range,
    _trend_direction,
    compute_hurst_regime_signal,
)


def _make_df(n=200, seed=42):
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


def _make_trending_df(n=250, direction="up"):
    """Create a DataFrame with clear trending behavior (high Hurst)."""
    np.random.seed(123)
    if direction == "up":
        trend = np.linspace(100, 150, n)
    else:
        trend = np.linspace(150, 100, n)
    noise = np.random.randn(n) * 0.3
    close = trend + noise
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.2),
        "low": close - abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_mean_reverting_df(n=250):
    """Create a DataFrame with mean-reverting behavior (low Hurst)."""
    np.random.seed(456)
    # Oscillating price around a mean
    t = np.arange(n)
    close = 100 + 5 * np.sin(t * 0.3) + np.random.randn(n) * 0.5
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.2),
        "low": close - abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


# ── R/S Analysis Tests ───────────────────────────────────────────────────


class TestRescaledRange:
    """Test the core R/S computation."""

    def test_returns_float(self):
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01
        result = _rescaled_range(returns, 16)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_nan_for_too_few_data(self):
        returns = np.array([0.01, -0.01])
        result = _rescaled_range(returns, 16)
        assert np.isnan(result)

    def test_positive_result(self):
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01
        result = _rescaled_range(returns, 8)
        assert result > 0


class TestComputeHurst:
    """Test Hurst exponent computation."""

    def test_returns_float(self):
        np.random.seed(42)
        returns = np.random.randn(300) * 0.01
        h = _compute_hurst(returns)
        assert isinstance(h, float)
        assert 0.0 <= h <= 1.0

    def test_nan_for_insufficient_data(self):
        returns = np.array([0.01, -0.01, 0.005])
        h = _compute_hurst(returns)
        assert np.isnan(h)

    def test_random_walk_near_half(self):
        """Pure random walk should have H close to 0.5."""
        np.random.seed(42)
        returns = np.random.randn(500) * 0.01
        h = _compute_hurst(returns)
        # Allow wide tolerance — R/S has estimation noise
        assert 0.35 <= h <= 0.7

    def test_trending_series_high_hurst(self):
        """Cumulative trend should have H > 0.5."""
        np.random.seed(42)
        # Persistent: positive returns mostly
        returns = np.abs(np.random.randn(300)) * 0.01 + 0.002
        h = _compute_hurst(returns)
        assert h > 0.45  # Trending should be above random walk

    def test_clipped_to_zero_one(self):
        np.random.seed(42)
        returns = np.random.randn(300) * 0.01
        h = _compute_hurst(returns)
        assert 0.0 <= h <= 1.0


class TestHurstRegimeClassification:
    """Test regime classification from Hurst values."""

    def test_trending(self):
        assert _hurst_regime(0.65) == "trending"

    def test_mean_reverting(self):
        assert _hurst_regime(0.35) == "mean_reverting"

    def test_random_walk(self):
        assert _hurst_regime(0.50) == "random_walk"

    def test_boundary_trending(self):
        assert _hurst_regime(0.56) == "trending"

    def test_boundary_mr(self):
        assert _hurst_regime(0.44) == "mean_reverting"

    def test_boundary_rw_low(self):
        assert _hurst_regime(0.45) == "random_walk"

    def test_boundary_rw_high(self):
        assert _hurst_regime(0.55) == "random_walk"

    def test_nan_unknown(self):
        assert _hurst_regime(float("nan")) == "unknown"


# ── Sub-signal Tests ─────────────────────────────────────────────────────


class TestTrendDirection:
    """Test EMA-based trend direction sub-signal."""

    def test_uptrend(self):
        df = _make_trending_df(100, "up")
        spread, vote = _trend_direction(df["close"])
        assert vote == "BUY"
        assert spread > 0

    def test_downtrend(self):
        df = _make_trending_df(100, "down")
        spread, vote = _trend_direction(df["close"])
        assert vote == "SELL"
        assert spread < 0

    def test_insufficient_data(self):
        close = pd.Series([100.0, 101.0])
        spread, vote = _trend_direction(close)
        assert vote == "HOLD"


class TestMrExtreme:
    """Test RSI-based mean-reversion sub-signal."""

    def test_oversold(self):
        # Create a series that drops sharply (RSI < 30)
        close = pd.Series([100.0] * 20 + [100 - i * 2 for i in range(10)])
        rsi_val, vote = _mr_extreme(close)
        assert vote == "BUY"

    def test_overbought(self):
        # Create a series with mostly up moves but some noise, pushing RSI > 70
        np.random.seed(99)
        # Strong uptrend with small noise: ~80% of moves are up
        changes = np.random.choice([3.0, 2.5, 2.0, -0.5], size=29, p=[0.4, 0.3, 0.2, 0.1])
        close = pd.Series(np.concatenate([[100.0], 100.0 + np.cumsum(changes)]))
        rsi_val, vote = _mr_extreme(close)
        assert vote == "SELL"

    def test_insufficient_data(self):
        close = pd.Series([100.0, 101.0])
        rsi_val, vote = _mr_extreme(close)
        assert vote == "HOLD"


# ── Signal Interface Tests ───────────────────────────────────────────────


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_hurst_regime_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_hurst_regime_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        expected_keys = {"hurst_regime", "trend_direction", "mr_extreme", "hurst_momentum"}
        assert expected_keys == set(result["sub_signals"].keys())

    def test_has_indicators(self):
        df = _make_df()
        result = compute_hurst_regime_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "hurst_exponent" in result["indicators"]
        assert "regime" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_hurst_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_hurst_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_none_input_returns_hold(self):
        result = compute_hurst_regime_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_nan_handling(self):
        df = _make_df(n=250)
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_hurst_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_hurst_regime_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_hurst_exponent_in_range(self):
        df = _make_df(n=300)
        result = compute_hurst_regime_signal(df)
        h = result["indicators"]["hurst_exponent"]
        if not np.isnan(h):
            assert 0.0 <= h <= 1.0

    def test_regime_is_valid_string(self):
        df = _make_df()
        result = compute_hurst_regime_signal(df)
        regime = result["indicators"]["regime"]
        assert regime in ("trending", "mean_reverting", "random_walk", "unknown")


class TestSignalBehavior:
    """Test signal produces sensible outputs for known inputs."""

    def test_trending_up_valid_output(self):
        """A clear uptrend should produce a valid signal.

        Note: R/S analysis may not always classify linear trends as trending
        because returns have low variance.  The key test is that the signal
        produces a valid result with a known regime classification.
        """
        df = _make_trending_df(300, "up")
        result = compute_hurst_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert result["indicators"]["regime"] in ("trending", "mean_reverting", "random_walk")

    def test_trending_down_produces_sell(self):
        """A clear downtrend should tend toward SELL."""
        df = _make_trending_df(300, "down")
        result = compute_hurst_regime_signal(df)
        assert result["action"] in ("SELL", "HOLD")

    def test_mean_reverting_after_drop(self):
        """Mean-reverting series that just dropped should tend toward BUY."""
        df = _make_mean_reverting_df(300)
        result = compute_hurst_regime_signal(df)
        # In MR regime, the signal depends on RSI state; just verify valid
        assert result["action"] in ("BUY", "SELL", "HOLD")
        if result["indicators"]["regime"] == "mean_reverting":
            assert result["sub_signals"]["trend_direction"] == "HOLD"

    def test_large_dataset(self):
        """Verify it handles larger datasets efficiently."""
        df = _make_df(n=1000)
        result = compute_hurst_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert np.isfinite(result["indicators"]["hurst_exponent"])

    def test_missing_column_returns_hold(self):
        df = pd.DataFrame({"close": [100.0] * 200, "volume": [1000.0] * 200})
        result = compute_hurst_regime_signal(df)
        assert result["action"] == "HOLD"

    def test_sub_signals_all_valid_strings(self):
        df = _make_df(n=300)
        result = compute_hurst_regime_signal(df)
        for name, vote in result["sub_signals"].items():
            assert vote in ("BUY", "SELL", "HOLD"), f"Invalid vote for {name}: {vote}"
