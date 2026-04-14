"""Tests for gold_real_yield_paradox signal module."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.gold_real_yield_paradox import (
    _compute_baseline_correlation,
    _correlation_break,
    _momentum_split,
    _paradox_spread,
    compute_gold_real_yield_paradox_signal,
)


def _make_df(n=200, trend=0.001):
    """Create a test DataFrame with realistic gold OHLCV data."""
    np.random.seed(42)
    close = 2000.0 + np.cumsum(np.random.randn(n) * 5 + trend * 100)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 2,
        "high": close + abs(np.random.randn(n) * 5),
        "low": close - abs(np.random.randn(n) * 5),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_yield_values(n=200, base=2.0, trend=0.001):
    """Fake FRED real yield data (newest first)."""
    np.random.seed(123)
    vals = base + np.cumsum(np.random.randn(n) * 0.02 + trend)
    return list(reversed(vals.tolist()))


class TestSignalInterface:
    """Test standard signal interface compliance."""

    @patch("portfolio.signals.gold_real_yield_paradox._fetch_real_yield")
    def test_returns_dict_with_required_keys(self, mock_fetch):
        mock_fetch.return_value = _make_yield_values()
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_gold_real_yield_paradox_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.gold_real_yield_paradox._fetch_real_yield")
    def test_has_sub_signals(self, mock_fetch):
        mock_fetch.return_value = _make_yield_values()
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_gold_real_yield_paradox_signal(df, context=ctx)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        if result["action"] != "HOLD" or result["confidence"] > 0:
            assert "paradox_spread" in result["sub_signals"]
            assert "correlation_break" in result["sub_signals"]
            assert "momentum_split" in result["sub_signals"]

    @patch("portfolio.signals.gold_real_yield_paradox._fetch_real_yield")
    def test_has_indicators(self, mock_fetch):
        mock_fetch.return_value = _make_yield_values()
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_gold_real_yield_paradox_signal(df, context=ctx)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        if result["action"] != "HOLD" or result["confidence"] > 0:
            assert "gydi" in result["indicators"]
            assert "gydi_regime" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_gold_real_yield_paradox_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_gold_real_yield_paradox_signal(df)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.gold_real_yield_paradox._fetch_real_yield")
    def test_nan_handling(self, mock_fetch):
        mock_fetch.return_value = _make_yield_values()
        df = _make_df()
        df.iloc[100:105, df.columns.get_loc("close")] = np.nan
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_gold_real_yield_paradox_signal(df, context=ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    @patch("portfolio.signals.gold_real_yield_paradox._fetch_real_yield")
    def test_confidence_capped_at_0_7(self, mock_fetch):
        mock_fetch.return_value = _make_yield_values()
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_gold_real_yield_paradox_signal(df, context=ctx)
        assert result["confidence"] <= 0.7

    def test_non_applicable_ticker_returns_hold(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_gold_real_yield_paradox_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.gold_real_yield_paradox._fetch_real_yield")
    def test_xag_ticker_accepted(self, mock_fetch):
        mock_fetch.return_value = _make_yield_values()
        df = _make_df()
        ctx = {"ticker": "XAG-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_gold_real_yield_paradox_signal(df, context=ctx)
        assert isinstance(result, dict)

    @patch("portfolio.signals.gold_real_yield_paradox._fetch_real_yield", return_value=None)
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_no_fred_key_returns_hold(self, _mock_open, _mock_fetch):
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {}}
        result = compute_gold_real_yield_paradox_signal(df, context=ctx)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.gold_real_yield_paradox._fetch_real_yield")
    def test_no_context_returns_valid(self, mock_fetch):
        mock_fetch.return_value = _make_yield_values()
        df = _make_df()
        result = compute_gold_real_yield_paradox_signal(df)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")


class TestSubIndicators:
    """Test individual sub-indicators."""

    def test_paradox_spread_both_positive(self):
        action, ind = _paradox_spread(0.05, 0.3)
        assert action == "BUY"
        assert ind["paradox_score"] > 0

    def test_paradox_spread_both_negative(self):
        action, ind = _paradox_spread(-0.05, -0.3)
        assert action == "SELL"
        assert ind["paradox_score"] > 0

    def test_paradox_spread_diverging(self):
        action, ind = _paradox_spread(0.05, -0.3)
        assert action == "HOLD"
        assert ind["paradox_score"] == 0.0

    def test_paradox_spread_magnitude_caps(self):
        action, ind = _paradox_spread(0.20, 1.5)
        assert ind["paradox_score"] <= 100.0

    def test_correlation_break_strong_deviation(self):
        np.random.seed(42)
        gold_ret = np.random.randn(60) * 0.01
        yield_chg = gold_ret * 0.5 + np.random.randn(60) * 0.002
        action, ind = _correlation_break(gold_ret, yield_chg, baseline_corr=-0.45)
        assert ind["corr_break_score"] > 0

    def test_correlation_break_insufficient_data(self):
        action, ind = _correlation_break(np.array([1.0] * 10), np.array([1.0] * 10), -0.45)
        assert action == "HOLD"

    def test_momentum_split_both_up(self):
        df = _make_df(n=250, trend=0.01)
        yields = _make_yield_values(n=100, trend=0.01)
        action, ind = _momentum_split(df["close"], yields)
        assert action in ("BUY", "SELL", "HOLD")
        assert "momentum_split_score" in ind

    def test_momentum_split_insufficient_data(self):
        df = _make_df(n=50)
        yields = _make_yield_values(n=20)
        action, ind = _momentum_split(df["close"], yields)
        assert action == "HOLD"


class TestBaselineCorrelation:
    """Test baseline correlation computation."""

    def test_returns_float(self):
        np.random.seed(42)
        gold = np.random.randn(200)
        yields = np.random.randn(200)
        result = _compute_baseline_correlation(gold, yields)
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_insufficient_data_returns_default(self):
        result = _compute_baseline_correlation(np.array([1.0] * 50), np.array([1.0] * 50))
        assert result == -0.45

    def test_correlated_data(self):
        np.random.seed(42)
        gold = np.random.randn(300)
        yields = gold * 0.8 + np.random.randn(300) * 0.2
        result = _compute_baseline_correlation(gold, yields)
        assert result > 0.3


class TestGYDIRegime:
    """Test GYDI score and regime classification."""

    @patch("portfolio.signals.gold_real_yield_paradox._fetch_real_yield")
    def test_gydi_regime_in_indicators(self, mock_fetch):
        mock_fetch.return_value = _make_yield_values()
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_gold_real_yield_paradox_signal(df, context=ctx)
        if "gydi_regime" in result["indicators"]:
            assert result["indicators"]["gydi_regime"] in ("LOW", "ELEVATED", "HIGH", "CRITICAL")

    @patch("portfolio.signals.gold_real_yield_paradox._fetch_real_yield")
    def test_gydi_score_bounded(self, mock_fetch):
        mock_fetch.return_value = _make_yield_values()
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_gold_real_yield_paradox_signal(df, context=ctx)
        if "gydi" in result["indicators"]:
            assert 0.0 <= result["indicators"]["gydi"] <= 100.0
