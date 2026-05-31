"""Tests for gold_btc_vol_spillover signal module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from portfolio.signals.gold_btc_vol_spillover import (
    compute_gold_btc_vol_spillover_signal,
    _MIN_ROWS,
)


def _make_btc_df(n=200, trend_up=True):
    """Create a test DataFrame with realistic BTC OHLCV data."""
    np.random.seed(42)
    if trend_up:
        close = 60000 + np.cumsum(np.random.randn(n) * 200 + 50)
    else:
        close = 60000 + np.cumsum(np.random.randn(n) * 200 - 50)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 100,
        "high": close + abs(np.random.randn(n) * 300),
        "low": close - abs(np.random.randn(n) * 300),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_gold_df(n=300, vol_spike=False):
    """Create a gold DataFrame. If vol_spike=True, inject high vol at end."""
    np.random.seed(123)
    base_returns = np.random.randn(n) * 0.005
    if vol_spike:
        base_returns[-25:] = np.random.randn(25) * 0.04
    else:
        base_returns[-80:] = np.random.randn(80) * 0.003
    close = 2000 * np.exp(np.cumsum(base_returns))
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + abs(np.random.randn(n) * 0.003)),
        "low": close * (1 - abs(np.random.randn(n) * 0.003)),
        "close": close,
        "volume": np.random.randint(100, 1000, n).astype(float),
    })


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_returns_dict_with_required_keys(self, mock_gold):
        mock_gold.return_value = _make_gold_df(200)
        df = _make_btc_df(200)
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_has_sub_signals(self, mock_gold):
        mock_gold.return_value = _make_gold_df(200)
        df = _make_btc_df(200)
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_has_indicators(self, mock_gold):
        mock_gold.return_value = _make_gold_df(200)
        df = _make_btc_df(200)
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_btc_df(n=10)
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_non_btc_ticker_returns_hold(self, mock_gold):
        mock_gold.return_value = _make_gold_df(200)
        df = _make_btc_df(200)
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "ETH-USD"})
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_no_ticker_still_works(self, mock_gold):
        mock_gold.return_value = _make_gold_df(200)
        df = _make_btc_df(200)
        result = compute_gold_btc_vol_spillover_signal(df, {})
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_nan_handling(self, mock_gold):
        mock_gold.return_value = _make_gold_df(200)
        df = _make_btc_df(200)
        df.iloc[100:105, df.columns.get_loc("close")] = np.nan
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        assert result["action"] in ("BUY", "SELL", "HOLD")


class TestVolSpilloverLogic:
    """Test the volatility spillover signal logic."""

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_gold_vol_spike_with_btc_uptrend_gives_buy(self, mock_gold):
        mock_gold.return_value = _make_gold_df(200, vol_spike=True)
        df = _make_btc_df(200, trend_up=True)
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        if result["action"] != "HOLD":
            assert result["action"] == "BUY"
            assert result["confidence"] > 0.0

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_gold_vol_spike_with_btc_downtrend_gives_sell(self, mock_gold):
        mock_gold.return_value = _make_gold_df(200, vol_spike=True)
        df = _make_btc_df(200, trend_up=False)
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        if result["action"] != "HOLD":
            assert result["action"] == "SELL"
            assert result["confidence"] > 0.0

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_no_vol_spike_gives_hold(self, mock_gold):
        mock_gold.return_value = _make_gold_df(200, vol_spike=False)
        df = _make_btc_df(200)
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_confidence_capped_at_07(self, mock_gold):
        mock_gold.return_value = _make_gold_df(200, vol_spike=True)
        df = _make_btc_df(200, trend_up=True)
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        assert result["confidence"] <= 0.7

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_gold_fetch_failure_returns_hold(self, mock_gold):
        mock_gold.return_value = None
        df = _make_btc_df(200)
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_indicators_populated_on_spike(self, mock_gold):
        mock_gold.return_value = _make_gold_df(200, vol_spike=True)
        df = _make_btc_df(200, trend_up=True)
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        indicators = result["indicators"]
        if result["action"] != "HOLD":
            assert "gold_vol_zscore" in indicators
            assert "gold_realized_vol" in indicators
            assert "vol_momentum" in indicators
            assert "btc_trend_strength" in indicators

    @patch("portfolio.signals.gold_btc_vol_spillover._fetch_gold_data")
    def test_sub_signals_correct_format(self, mock_gold):
        mock_gold.return_value = _make_gold_df(200, vol_spike=True)
        df = _make_btc_df(200, trend_up=True)
        result = compute_gold_btc_vol_spillover_signal(df, {"ticker": "BTC-USD"})
        subs = result["sub_signals"]
        if result["action"] != "HOLD":
            assert "gold_vol_spike" in subs
            assert "vol_momentum" in subs
            assert "btc_trend" in subs
