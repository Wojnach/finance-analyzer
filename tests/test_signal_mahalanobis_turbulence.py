"""Tests for mahalanobis_turbulence signal module."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.mahalanobis_turbulence import (
    _absorption_ratio_vote,
    _compute_turbulence_series,
    _turbulence_trend_vote,
    _turbulence_z_vote,
    compute_mahalanobis_turbulence_signal,
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


def _make_multi_asset_closes(n=400):
    """Create synthetic multi-asset close prices for 5 assets.

    n=400 ensures enough data for 252-day rolling covariance + 60-day z-score.
    """
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    # Correlated random walks
    base = np.cumsum(np.random.randn(n) * 0.01)
    closes = pd.DataFrame({
        "BTC-USD": 50000 * np.exp(base + np.cumsum(np.random.randn(n) * 0.02)),
        "ETH-USD": 3000 * np.exp(base * 0.8 + np.cumsum(np.random.randn(n) * 0.025)),
        "GC=F": 2000 * np.exp(-base * 0.3 + np.cumsum(np.random.randn(n) * 0.005)),
        "SI=F": 25 * np.exp(-base * 0.2 + np.cumsum(np.random.randn(n) * 0.01)),
        "SPY": 500 * np.exp(base * 0.5 + np.cumsum(np.random.randn(n) * 0.008)),
    }, index=dates)
    return closes


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    @patch("portfolio.signals.mahalanobis_turbulence._fetch_multi_asset_closes")
    def test_returns_dict_with_required_keys(self, mock_fetch):
        mock_fetch.return_value = _make_multi_asset_closes()
        df = _make_df()
        result = compute_mahalanobis_turbulence_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.mahalanobis_turbulence._fetch_multi_asset_closes")
    def test_has_sub_signals(self, mock_fetch):
        mock_fetch.return_value = _make_multi_asset_closes()
        df = _make_df()
        result = compute_mahalanobis_turbulence_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    @patch("portfolio.signals.mahalanobis_turbulence._fetch_multi_asset_closes")
    def test_has_indicators(self, mock_fetch):
        mock_fetch.return_value = _make_multi_asset_closes()
        df = _make_df()
        result = compute_mahalanobis_turbulence_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_mahalanobis_turbulence_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_mahalanobis_turbulence_signal(df)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.mahalanobis_turbulence._fetch_multi_asset_closes")
    def test_fetch_failure_returns_hold(self, mock_fetch):
        mock_fetch.return_value = None
        df = _make_df()
        result = compute_mahalanobis_turbulence_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.mahalanobis_turbulence._fetch_multi_asset_closes")
    def test_with_context_risk_asset(self, mock_fetch):
        mock_fetch.return_value = _make_multi_asset_closes(400)
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_mahalanobis_turbulence_signal(df, context=ctx)
        assert isinstance(result, dict)
        # Should have indicators if turbulence computed successfully
        if result["indicators"]:
            assert result["indicators"]["is_safe_haven"] is False

    @patch("portfolio.signals.mahalanobis_turbulence._fetch_multi_asset_closes")
    def test_with_context_safe_haven(self, mock_fetch):
        mock_fetch.return_value = _make_multi_asset_closes(400)
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_mahalanobis_turbulence_signal(df, context=ctx)
        assert isinstance(result, dict)
        if result["indicators"]:
            assert result["indicators"]["is_safe_haven"] is True

    @patch("portfolio.signals.mahalanobis_turbulence._fetch_multi_asset_closes")
    def test_confidence_capped_at_07(self, mock_fetch):
        mock_fetch.return_value = _make_multi_asset_closes()
        df = _make_df()
        result = compute_mahalanobis_turbulence_signal(df)
        assert result["confidence"] <= 0.7


class TestTurbulenceComputation:
    """Test the core turbulence computation."""

    def test_compute_turbulence_series_normal(self):
        closes = _make_multi_asset_closes(300)
        result = _compute_turbulence_series(closes)
        assert result is not None
        assert "turbulence" in result.columns
        assert "absorption_ratio" in result.columns
        assert len(result) > 0
        # Turbulence should be non-negative (squared Mahalanobis distance)
        assert (result["turbulence"] >= 0).all()
        # Absorption ratio between 0 and 1
        assert (result["absorption_ratio"] >= 0).all()
        assert (result["absorption_ratio"] <= 1.0).all()

    def test_compute_turbulence_series_too_short(self):
        closes = _make_multi_asset_closes(30)
        result = _compute_turbulence_series(closes)
        assert result is None

    def test_compute_turbulence_none_input(self):
        result = _compute_turbulence_series(None)
        assert result is None

    def test_compute_turbulence_with_nans(self):
        closes = _make_multi_asset_closes(300)
        # Introduce some NaN values
        closes.iloc[50:55, 0] = np.nan
        closes.iloc[100:102, 2] = np.nan
        result = _compute_turbulence_series(closes)
        # Should still work with partial NaN
        assert result is not None
        assert len(result) > 0

    def test_turbulence_spike_on_shock(self):
        """Verify turbulence spikes when we inject an extreme return."""
        closes = _make_multi_asset_closes(300)
        # Inject a massive shock on the last day
        for col in closes.columns:
            closes.iloc[-1, closes.columns.get_loc(col)] *= 1.10  # 10% spike

        result = _compute_turbulence_series(closes)
        assert result is not None
        # Last turbulence value should be elevated vs mean
        turb = result["turbulence"]
        assert turb.iloc[-1] > turb.mean()


class TestSubSignalVotes:
    """Test individual sub-signal voting functions."""

    def test_z_vote_high_turbulence_risk_asset(self):
        assert _turbulence_z_vote(2.5, is_safe_haven=False) == "SELL"

    def test_z_vote_high_turbulence_safe_haven(self):
        assert _turbulence_z_vote(2.5, is_safe_haven=True) == "BUY"

    def test_z_vote_calm_risk_asset(self):
        assert _turbulence_z_vote(-2.0, is_safe_haven=False) == "BUY"

    def test_z_vote_calm_safe_haven(self):
        assert _turbulence_z_vote(-2.0, is_safe_haven=True) == "SELL"

    def test_z_vote_normal(self):
        assert _turbulence_z_vote(0.5, is_safe_haven=False) == "HOLD"
        assert _turbulence_z_vote(0.5, is_safe_haven=True) == "HOLD"

    def test_trend_vote_rising(self):
        series = pd.Series([1.0, 1.5, 2.0, 3.0, 5.0, 8.0])
        assert _turbulence_trend_vote(series, is_safe_haven=False) == "SELL"
        assert _turbulence_trend_vote(series, is_safe_haven=True) == "BUY"

    def test_trend_vote_falling(self):
        series = pd.Series([8.0, 6.0, 4.0, 3.0, 2.0, 1.0])
        assert _turbulence_trend_vote(series, is_safe_haven=False) == "BUY"
        assert _turbulence_trend_vote(series, is_safe_haven=True) == "SELL"

    def test_trend_vote_flat(self):
        series = pd.Series([5.0, 5.1, 5.0, 5.1, 5.0, 5.1])
        assert _turbulence_trend_vote(series, is_safe_haven=False) == "HOLD"

    def test_trend_vote_too_short(self):
        series = pd.Series([1.0, 2.0])
        assert _turbulence_trend_vote(series, is_safe_haven=False) == "HOLD"

    def test_ar_vote_high_percentile(self):
        # Create series where current value is in top 90th percentile
        series = pd.Series(np.linspace(0.3, 0.5, 100))
        series.iloc[-1] = 0.55  # Above all others
        assert _absorption_ratio_vote(series, is_safe_haven=False) == "SELL"
        assert _absorption_ratio_vote(series, is_safe_haven=True) == "BUY"

    def test_ar_vote_low_percentile(self):
        series = pd.Series(np.linspace(0.3, 0.5, 100))
        series.iloc[-1] = 0.25  # Below all others
        assert _absorption_ratio_vote(series, is_safe_haven=False) == "BUY"
        assert _absorption_ratio_vote(series, is_safe_haven=True) == "SELL"

    def test_ar_vote_normal(self):
        # Last value in middle of distribution — should be HOLD
        series = pd.Series(np.linspace(0.3, 0.5, 100))
        series.iloc[-1] = 0.4  # Median of [0.3, 0.5] range — ~50th percentile
        assert _absorption_ratio_vote(series, is_safe_haven=False) == "HOLD"

    def test_ar_vote_too_short(self):
        series = pd.Series([0.4, 0.5])
        assert _absorption_ratio_vote(series, is_safe_haven=False) == "HOLD"


class TestEdgeCases:
    """Test edge cases and robustness."""

    @patch("portfolio.signals.mahalanobis_turbulence._fetch_multi_asset_closes")
    def test_none_df(self, mock_fetch):
        result = compute_mahalanobis_turbulence_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.mahalanobis_turbulence._fetch_multi_asset_closes")
    def test_all_nan_closes(self, mock_fetch):
        closes = pd.DataFrame(
            np.nan,
            index=pd.date_range("2025-01-01", periods=300, freq="B"),
            columns=["A", "B", "C", "D", "E"],
        )
        mock_fetch.return_value = closes
        df = _make_df()
        result = compute_mahalanobis_turbulence_signal(df)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.mahalanobis_turbulence._fetch_multi_asset_closes")
    def test_constant_prices(self, mock_fetch):
        """Constant prices = zero returns = degenerate covariance."""
        closes = pd.DataFrame(
            100.0,
            index=pd.date_range("2025-01-01", periods=300, freq="B"),
            columns=["A", "B", "C", "D", "E"],
        )
        mock_fetch.return_value = closes
        df = _make_df()
        result = compute_mahalanobis_turbulence_signal(df)
        assert result["action"] == "HOLD"
