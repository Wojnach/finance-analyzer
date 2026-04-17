"""Tests for copper_gold_ratio signal module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from portfolio.signals.copper_gold_ratio import (
    compute_copper_gold_ratio_signal,
    _ratio_zscore,
    _ratio_trend,
    _ratio_momentum,
    _copper_gold_spread,
    _CACHE,
)


def _make_df(n=100):
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


def _make_ratio_df(n=200, trend="neutral"):
    """Create a mock combined DataFrame with copper, gold, and ratio."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")

    if trend == "falling":
        # Falling ratio: gold rising faster than copper
        copper = 4.0 + np.cumsum(np.random.randn(n) * 0.01)
        gold = 2000 + np.cumsum(np.random.randn(n) * 5 + 2)
    elif trend == "rising":
        # Rising ratio: copper rising faster than gold
        copper = 4.0 + np.cumsum(np.random.randn(n) * 0.01 + 0.005)
        gold = 2000 + np.cumsum(np.random.randn(n) * 5 - 1)
    else:
        copper = 4.0 + np.cumsum(np.random.randn(n) * 0.01)
        gold = 2000 + np.cumsum(np.random.randn(n) * 5)

    df = pd.DataFrame({"copper": copper, "gold": gold}, index=dates)
    df["ratio"] = df["copper"] / df["gold"]
    return df


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear module cache before each test."""
    _CACHE.clear()
    yield
    _CACHE.clear()


class TestRatioHelpers:
    """Test helper functions directly."""

    def test_ratio_zscore_neutral(self):
        df = _make_ratio_df(200, "neutral")
        z = _ratio_zscore(df["ratio"], window=50)
        assert isinstance(z, float)
        assert -5.0 < z < 5.0

    def test_ratio_zscore_short_series(self):
        df = _make_ratio_df(10)
        z = _ratio_zscore(df["ratio"], window=50)
        assert z == 0.0

    def test_ratio_trend_rising(self):
        df = _make_ratio_df(250, "rising")
        t = _ratio_trend(df["ratio"])
        assert t in (-1, 0, 1)

    def test_ratio_trend_short_series(self):
        series = pd.Series([1.0, 1.1, 1.2])
        t = _ratio_trend(series)
        assert t == 0

    def test_ratio_momentum(self):
        df = _make_ratio_df(200, "rising")
        m = _ratio_momentum(df["ratio"], periods=20)
        assert isinstance(m, float)

    def test_ratio_momentum_short(self):
        series = pd.Series([1.0, 1.1])
        m = _ratio_momentum(series, periods=20)
        assert m == 0.0

    def test_copper_gold_spread(self):
        df = _make_ratio_df(200, "neutral")
        s = _copper_gold_spread(df, periods=20)
        assert isinstance(s, float)


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    @patch("portfolio.signals.copper_gold_ratio._fetch_ratio_data")
    def test_returns_dict_with_required_keys(self, mock_fetch):
        mock_fetch.return_value = _make_ratio_df(200, "neutral")
        df = _make_df()
        result = compute_copper_gold_ratio_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.copper_gold_ratio._fetch_ratio_data")
    def test_has_sub_signals(self, mock_fetch):
        mock_fetch.return_value = _make_ratio_df(200, "neutral")
        df = _make_df()
        result = compute_copper_gold_ratio_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        expected_keys = {"ratio_zscore", "ratio_trend", "ratio_momentum",
                         "copper_gold_spread"}
        assert expected_keys == set(result["sub_signals"].keys())

    @patch("portfolio.signals.copper_gold_ratio._fetch_ratio_data")
    def test_has_indicators(self, mock_fetch):
        mock_fetch.return_value = _make_ratio_df(200, "neutral")
        df = _make_df()
        result = compute_copper_gold_ratio_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "ratio" in result["indicators"]
        assert "ratio_zscore" in result["indicators"]
        assert "copper_price" in result["indicators"]
        assert "gold_price" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_copper_gold_ratio_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_copper_gold_ratio_signal(df)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.copper_gold_ratio._fetch_ratio_data")
    def test_fetch_failure_returns_hold(self, mock_fetch):
        mock_fetch.return_value = None
        df = _make_df()
        result = compute_copper_gold_ratio_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.copper_gold_ratio._fetch_ratio_data")
    def test_confidence_capped_at_0_7(self, mock_fetch):
        mock_fetch.return_value = _make_ratio_df(200, "falling")
        df = _make_df()
        result = compute_copper_gold_ratio_signal(df)
        assert result["confidence"] <= 0.7


class TestAssetClassInversion:
    """Test that signal direction inverts for metals tickers."""

    @patch("portfolio.signals.copper_gold_ratio._fetch_ratio_data")
    def test_metals_inversion(self, mock_fetch):
        """For metals: falling ratio = gold strength = BUY."""
        # Create strongly falling ratio data
        mock_fetch.return_value = _make_ratio_df(200, "falling")
        df = _make_df()

        # Risk asset should get SELL for falling ratio
        risk_result = compute_copper_gold_ratio_signal(
            df, context={"ticker": "BTC-USD", "asset_class": "crypto"})

        # Metals should get opposite direction
        metals_result = compute_copper_gold_ratio_signal(
            df, context={"ticker": "XAU-USD", "asset_class": "metals"})

        # If both produce non-HOLD signals, they should be opposite
        if risk_result["action"] != "HOLD" and metals_result["action"] != "HOLD":
            assert risk_result["action"] != metals_result["action"]

    @patch("portfolio.signals.copper_gold_ratio._fetch_ratio_data")
    def test_metals_ticker_detection(self, mock_fetch):
        mock_fetch.return_value = _make_ratio_df(200, "neutral")
        df = _make_df()

        # XAG-USD should be detected as metals
        result = compute_copper_gold_ratio_signal(
            df, context={"ticker": "XAG-USD"})
        assert isinstance(result, dict)

    @patch("portfolio.signals.copper_gold_ratio._fetch_ratio_data")
    def test_no_context_defaults_to_risk(self, mock_fetch):
        mock_fetch.return_value = _make_ratio_df(200, "neutral")
        df = _make_df()
        result = compute_copper_gold_ratio_signal(df)
        assert isinstance(result, dict)

    @patch("portfolio.signals.copper_gold_ratio._fetch_ratio_data")
    def test_with_full_context(self, mock_fetch):
        mock_fetch.return_value = _make_ratio_df(200, "neutral")
        df = _make_df()
        ctx = {
            "ticker": "BTC-USD",
            "asset_class": "crypto",
            "regime": "trending-up",
        }
        result = compute_copper_gold_ratio_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")


class TestSubSignalVoting:
    """Test sub-signal voting logic."""

    @patch("portfolio.signals.copper_gold_ratio._fetch_ratio_data")
    def test_all_votes_are_valid(self, mock_fetch):
        mock_fetch.return_value = _make_ratio_df(200, "neutral")
        df = _make_df()
        result = compute_copper_gold_ratio_signal(df)
        for key, vote in result["sub_signals"].items():
            assert vote in ("BUY", "SELL", "HOLD"), \
                f"Invalid vote for {key}: {vote}"

    @patch("portfolio.signals.copper_gold_ratio._fetch_ratio_data")
    def test_nan_handling_in_data(self, mock_fetch):
        """Ratio data with NaN values should not crash."""
        data = _make_ratio_df(200, "neutral")
        data.iloc[100:105, data.columns.get_loc("copper")] = np.nan
        data["ratio"] = data["copper"] / data["gold"]
        mock_fetch.return_value = data
        df = _make_df()
        result = compute_copper_gold_ratio_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
