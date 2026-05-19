"""Tests for absorption_ratio_regime signal module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.absorption_ratio_regime import (
    _ar_delta_vote,
    _ar_percentile_vote,
    _ar_zscore_vote,
    _compute_absorption_ratio_series,
    compute_absorption_ratio_regime_signal,
)


def _make_df(n=100, seed=42):
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
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


def _make_multi_asset_closes(n=300, seed=42):
    """Synthetic 5-asset daily close prices for AR computation."""
    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    base = 100 + np.cumsum(np.random.randn(n, 5) * 0.5, axis=0)
    base = np.maximum(base, 1.0)
    return pd.DataFrame(
        base,
        index=dates,
        columns=["BTC-USD", "ETH-USD", "GC=F", "SI=F", "SPY"],
    )


def _make_correlated_closes(n=300, seed=42, correlation=0.9):
    """Highly correlated multi-asset closes (high AR expected)."""
    np.random.seed(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    common = np.cumsum(np.random.randn(n) * 0.5)
    cols = []
    for _ in range(5):
        noise = np.random.randn(n) * 0.1
        cols.append(100 + correlation * common + (1 - correlation) * noise)
    data = np.column_stack(cols)
    data = np.maximum(data, 1.0)
    return pd.DataFrame(
        data,
        index=dates,
        columns=["BTC-USD", "ETH-USD", "GC=F", "SI=F", "SPY"],
    )


class TestSignalInterface:
    @patch("portfolio.signals.absorption_ratio_regime._fetch_multi_asset_closes")
    def test_returns_dict_with_required_keys(self, mock_fetch):
        mock_fetch.return_value = _make_multi_asset_closes()
        df = _make_df()
        result = compute_absorption_ratio_regime_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.absorption_ratio_regime._fetch_multi_asset_closes")
    def test_has_sub_signals(self, mock_fetch):
        mock_fetch.return_value = _make_multi_asset_closes()
        df = _make_df()
        result = compute_absorption_ratio_regime_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    @patch("portfolio.signals.absorption_ratio_regime._fetch_multi_asset_closes")
    def test_has_indicators(self, mock_fetch):
        mock_fetch.return_value = _make_multi_asset_closes(n=400)
        df = _make_df()
        result = compute_absorption_ratio_regime_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "absorption_ratio" in result["indicators"]
        assert "ar_z_score" in result["indicators"]
        assert "ar_delta_5d" in result["indicators"]
        assert "ar_percentile" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_absorption_ratio_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_absorption_ratio_regime_signal(df)
        assert result["action"] == "HOLD"

    def test_none_input_returns_hold(self):
        result = compute_absorption_ratio_regime_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.absorption_ratio_regime._fetch_multi_asset_closes")
    def test_confidence_capped_at_07(self, mock_fetch):
        mock_fetch.return_value = _make_multi_asset_closes()
        df = _make_df(n=200)
        result = compute_absorption_ratio_regime_signal(df)
        assert result["confidence"] <= 0.7

    @patch("portfolio.signals.absorption_ratio_regime._fetch_multi_asset_closes")
    def test_safe_haven_context(self, mock_fetch):
        mock_fetch.return_value = _make_multi_asset_closes()
        df = _make_df()
        ctx = {"ticker": "XAU-USD"}
        result = compute_absorption_ratio_regime_signal(df, context=ctx)
        assert isinstance(result, dict)

    @patch("portfolio.signals.absorption_ratio_regime._fetch_multi_asset_closes")
    def test_risk_asset_context(self, mock_fetch):
        mock_fetch.return_value = _make_multi_asset_closes()
        df = _make_df()
        ctx = {"ticker": "BTC-USD"}
        result = compute_absorption_ratio_regime_signal(df, context=ctx)
        assert isinstance(result, dict)

    @patch("portfolio.signals.absorption_ratio_regime._fetch_multi_asset_closes")
    def test_fetch_returns_none_graceful(self, mock_fetch):
        mock_fetch.return_value = None
        df = _make_df()
        result = compute_absorption_ratio_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


class TestARComputation:
    def test_ar_series_from_synthetic(self):
        closes = _make_multi_asset_closes(n=300)
        ar = _compute_absorption_ratio_series(closes)
        assert ar is not None
        assert len(ar) > 10
        assert all(0.0 <= v <= 1.0 for v in ar)

    def test_ar_series_none_on_insufficient_data(self):
        closes = _make_multi_asset_closes(n=30)
        ar = _compute_absorption_ratio_series(closes)
        assert ar is None

    def test_ar_series_none_on_none_input(self):
        ar = _compute_absorption_ratio_series(None)
        assert ar is None

    def test_high_correlation_raises_ar(self):
        uncorr = _make_multi_asset_closes(n=300, seed=42)
        corr = _make_correlated_closes(n=300, seed=42, correlation=0.95)
        ar_uncorr = _compute_absorption_ratio_series(uncorr)
        ar_corr = _compute_absorption_ratio_series(corr)
        assert ar_uncorr is not None and ar_corr is not None
        assert ar_corr.iloc[-1] > ar_uncorr.iloc[-1]


class TestSubSignalVotes:
    def test_zscore_risk_asset(self):
        assert _ar_zscore_vote(2.0, False) == "SELL"
        assert _ar_zscore_vote(-2.0, False) == "BUY"
        assert _ar_zscore_vote(0.5, False) == "HOLD"

    def test_zscore_safe_haven(self):
        assert _ar_zscore_vote(2.0, True) == "BUY"
        assert _ar_zscore_vote(-2.0, True) == "SELL"
        assert _ar_zscore_vote(0.5, True) == "HOLD"

    def test_delta_risk_asset(self):
        assert _ar_delta_vote(0.05, False) == "SELL"
        assert _ar_delta_vote(-0.05, False) == "BUY"
        assert _ar_delta_vote(0.01, False) == "HOLD"

    def test_delta_safe_haven(self):
        assert _ar_delta_vote(0.05, True) == "BUY"
        assert _ar_delta_vote(-0.05, True) == "SELL"

    def test_percentile_risk_asset(self):
        assert _ar_percentile_vote(90.0, False) == "SELL"
        assert _ar_percentile_vote(10.0, False) == "BUY"
        assert _ar_percentile_vote(50.0, False) == "HOLD"

    def test_percentile_safe_haven(self):
        assert _ar_percentile_vote(90.0, True) == "BUY"
        assert _ar_percentile_vote(10.0, True) == "SELL"

    def test_boundary_values(self):
        assert _ar_zscore_vote(1.5, False) == "SELL"
        assert _ar_zscore_vote(-1.5, False) == "BUY"
        assert _ar_percentile_vote(85.0, False) == "SELL"
        assert _ar_percentile_vote(15.0, False) == "BUY"


class TestDeterminism:
    @patch("portfolio.signals.absorption_ratio_regime._fetch_multi_asset_closes")
    def test_same_input_same_output(self, mock_fetch):
        closes = _make_multi_asset_closes(n=300, seed=99)
        mock_fetch.return_value = closes
        df = _make_df(n=100, seed=99)
        r1 = compute_absorption_ratio_regime_signal(df.copy(), {"ticker": "BTC-USD"})
        mock_fetch.return_value = closes.copy()
        r2 = compute_absorption_ratio_regime_signal(df.copy(), {"ticker": "BTC-USD"})
        assert r1["action"] == r2["action"]
        assert r1["confidence"] == r2["confidence"]
