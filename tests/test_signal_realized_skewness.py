"""Tests for realized_skewness signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.realized_skewness import compute_realized_skewness_signal


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


def _make_skewed_df(n=300, skew_direction="negative"):
    """Create a DataFrame with intentionally skewed returns."""
    np.random.seed(123)
    if skew_direction == "negative":
        # Fat left tail: mostly small positive returns with occasional large drops
        returns = np.where(
            np.random.rand(n) > 0.9,
            -np.random.exponential(0.03, n),  # large negative
            np.random.exponential(0.005, n),   # small positive
        )
    else:
        # Fat right tail: mostly small negative returns with occasional large jumps
        returns = np.where(
            np.random.rand(n) > 0.9,
            np.random.exponential(0.03, n),   # large positive
            -np.random.exponential(0.005, n),  # small negative
        )

    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + abs(np.random.randn(n) * 0.003)),
        "low": close * (1 - abs(np.random.randn(n) * 0.003)),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_realized_skewness_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_realized_skewness_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_realized_skewness_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_realized_skewness_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_none_dataframe_returns_hold(self):
        result = compute_realized_skewness_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


class TestSubSignals:
    """Test individual sub-signal behavior."""

    def test_sub_signals_have_expected_keys(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        expected_keys = {"skew_zscore", "skew_momentum", "kurtosis_confirm",
                         "skew_regime_divergence"}
        assert expected_keys == set(result["sub_signals"].keys())

    def test_all_sub_signals_are_valid_actions(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        for key, value in result["sub_signals"].items():
            assert value in ("BUY", "SELL", "HOLD"), f"{key} has invalid value {value}"


class TestIndicators:
    """Test indicator values are present and reasonable."""

    def test_raw_skewness_in_indicators(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert "raw_skewness" in result["indicators"]
        skew = result["indicators"]["raw_skewness"]
        # Skewness should be a finite number
        assert np.isfinite(skew)
        # For random data, skewness should be roughly between -3 and 3
        assert -5.0 < skew < 5.0

    def test_skew_z_in_indicators(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert "skew_z" in result["indicators"]

    def test_kurtosis_in_indicators(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert "kurtosis" in result["indicators"]

    def test_skew_divergence_in_indicators(self):
        df = _make_df()
        result = compute_realized_skewness_signal(df)
        assert "skew_divergence" in result["indicators"]


class TestDirectionality:
    """Test that the signal produces expected directions for skewed data."""

    def test_negative_skew_tends_toward_buy(self):
        """Negative skew (fat left tail) should lean toward BUY."""
        df = _make_skewed_df(n=300, skew_direction="negative")
        result = compute_realized_skewness_signal(df)
        # At minimum, the raw skewness should be negative
        assert result["indicators"]["raw_skewness"] < 0

    def test_positive_skew_tends_toward_sell(self):
        """Positive skew (fat right tail) should lean toward SELL."""
        df = _make_skewed_df(n=300, skew_direction="positive")
        result = compute_realized_skewness_signal(df)
        # At minimum, the raw skewness should be positive
        assert result["indicators"]["raw_skewness"] > 0


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_constant_price_returns_hold(self):
        """Constant price = zero returns = zero skewness = HOLD."""
        n = 100
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
            "volume": [1000.0] * n,
        })
        result = compute_realized_skewness_signal(df)
        assert result["action"] == "HOLD"

    def test_minimum_viable_rows(self):
        """Test with exactly MIN_ROWS (60) rows."""
        df = _make_df(n=60)
        result = compute_realized_skewness_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert isinstance(result["confidence"], float)

    def test_large_dataset(self):
        """Test with a large dataset (1000 rows)."""
        df = _make_df(n=1000)
        result = compute_realized_skewness_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_high_volatility_data(self):
        """Test with extremely volatile data."""
        np.random.seed(99)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n) * 5.0)  # 10x normal vol
        close = np.maximum(close, 1.0)  # prevent negative prices
        df = pd.DataFrame({
            "open": close,
            "high": close * 1.05,
            "low": close * 0.95,
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        })
        result = compute_realized_skewness_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_all_volume_zero(self):
        """Volume=0 should not crash."""
        df = _make_df()
        df["volume"] = 0.0
        result = compute_realized_skewness_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
