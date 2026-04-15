"""Tests for cross_asset_tsmom signal module."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.cross_asset_tsmom import (
    _compute_bond_momentum,
    _compute_cross_pair,
    _compute_equity_momentum,
    _compute_own_tsmom,
    compute_cross_asset_tsmom_signal,
)


def _make_df(n=300, trend=0.0):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5 + trend)
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_yf_data(tlt_ret=0.02, spy_ret=0.03, gcf_ret=0.01, btc_ret=0.05):
    """Create mock yfinance return data."""
    return {
        "TLT": {"ret_63d": tlt_ret, "ret_252d": tlt_ret * 3},
        "SPY": {"ret_63d": spy_ret, "ret_252d": spy_ret * 3},
        "GC=F": {"ret_63d": gcf_ret, "ret_252d": gcf_ret * 3},
        "BTC-USD": {"ret_63d": btc_ret, "ret_252d": btc_ret * 3},
    }


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_returns_dict_with_required_keys(self, mock_yf):
        mock_yf.return_value = _make_yf_data()
        df = _make_df()
        result = compute_cross_asset_tsmom_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_has_sub_signals(self, mock_yf):
        mock_yf.return_value = _make_yf_data()
        df = _make_df()
        result = compute_cross_asset_tsmom_signal(df)
        assert "sub_signals" in result
        subs = result["sub_signals"]
        assert isinstance(subs, dict)
        assert "own_tsmom_252d" in subs
        assert "cross_pair_63d" in subs
        assert "bond_momentum" in subs
        assert "equity_momentum" in subs

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_has_indicators(self, mock_yf):
        mock_yf.return_value = _make_yf_data()
        df = _make_df()
        result = compute_cross_asset_tsmom_signal(df)
        assert "indicators" in result
        inds = result["indicators"]
        assert isinstance(inds, dict)
        assert "own_ret_252d" in inds
        assert "tlt_ret_63d" in inds
        assert "spy_ret_63d" in inds

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_cross_asset_tsmom_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_none_dataframe_returns_hold(self):
        result = compute_cross_asset_tsmom_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_cross_asset_tsmom_signal(df)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_nan_handling(self, mock_yf):
        mock_yf.return_value = _make_yf_data()
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_cross_asset_tsmom_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_confidence_capped_at_0_7(self, mock_yf):
        mock_yf.return_value = _make_yf_data()
        df = _make_df(n=300, trend=0.1)
        result = compute_cross_asset_tsmom_signal(df)
        assert result["confidence"] <= 0.7


class TestWithContext:
    """Test context-dependent behavior for different tickers."""

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_xau_uses_tlt_cross_pair(self, mock_yf):
        mock_yf.return_value = _make_yf_data()
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_cross_asset_tsmom_signal(df, context=ctx)
        assert result["indicators"]["cross_pair_ticker"] == "TLT"

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_xag_uses_gcf_cross_pair(self, mock_yf):
        mock_yf.return_value = _make_yf_data()
        df = _make_df()
        ctx = {"ticker": "XAG-USD", "asset_class": "metals"}
        result = compute_cross_asset_tsmom_signal(df, context=ctx)
        assert result["indicators"]["cross_pair_ticker"] == "GC=F"

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_btc_uses_spy_cross_pair(self, mock_yf):
        mock_yf.return_value = _make_yf_data()
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_cross_asset_tsmom_signal(df, context=ctx)
        assert result["indicators"]["cross_pair_ticker"] == "SPY"

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_eth_uses_btc_cross_pair(self, mock_yf):
        mock_yf.return_value = _make_yf_data()
        df = _make_df()
        ctx = {"ticker": "ETH-USD", "asset_class": "crypto"}
        result = compute_cross_asset_tsmom_signal(df, context=ctx)
        assert result["indicators"]["cross_pair_ticker"] == "BTC-USD"

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_mstr_uses_btc_cross_pair(self, mock_yf):
        mock_yf.return_value = _make_yf_data()
        df = _make_df()
        ctx = {"ticker": "MSTR", "asset_class": "stocks"}
        result = compute_cross_asset_tsmom_signal(df, context=ctx)
        assert result["indicators"]["cross_pair_ticker"] == "BTC-USD"

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_unknown_ticker_gets_none_pair(self, mock_yf):
        mock_yf.return_value = _make_yf_data()
        df = _make_df()
        ctx = {"ticker": "AAPL", "asset_class": "stocks"}
        result = compute_cross_asset_tsmom_signal(df, context=ctx)
        assert result["indicators"]["cross_pair_ticker"] == "none"

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_no_context_still_works(self, mock_yf):
        mock_yf.return_value = _make_yf_data()
        df = _make_df()
        result = compute_cross_asset_tsmom_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")


class TestSubIndicators:
    """Test individual sub-indicator computation."""

    def test_own_tsmom_uptrend_buy(self):
        np.random.seed(42)
        close = pd.Series(np.linspace(80, 120, 300))
        assert _compute_own_tsmom(close) == "BUY"

    def test_own_tsmom_downtrend_sell(self):
        close = pd.Series(np.linspace(120, 80, 300))
        assert _compute_own_tsmom(close) == "SELL"

    def test_own_tsmom_flat_hold(self):
        close = pd.Series([100.0] * 300)
        assert _compute_own_tsmom(close) == "HOLD"

    def test_own_tsmom_insufficient_data(self):
        close = pd.Series([100.0] * 15)
        assert _compute_own_tsmom(close) == "HOLD"

    def test_cross_pair_positive(self):
        yf_data = _make_yf_data(gcf_ret=0.05)
        assert _compute_cross_pair("XAG-USD", yf_data) == "BUY"

    def test_cross_pair_negative(self):
        yf_data = _make_yf_data(spy_ret=-0.05)
        assert _compute_cross_pair("BTC-USD", yf_data) == "SELL"

    def test_cross_pair_unknown_ticker(self):
        yf_data = _make_yf_data()
        assert _compute_cross_pair("UNKNOWN", yf_data) == "HOLD"

    def test_cross_pair_no_data(self):
        assert _compute_cross_pair("XAU-USD", None) == "HOLD"

    def test_bond_momentum_positive(self):
        yf_data = _make_yf_data(tlt_ret=0.03)
        assert _compute_bond_momentum(yf_data) == "BUY"

    def test_bond_momentum_negative(self):
        yf_data = _make_yf_data(tlt_ret=-0.03)
        assert _compute_bond_momentum(yf_data) == "SELL"

    def test_bond_momentum_no_data(self):
        assert _compute_bond_momentum(None) == "HOLD"
        assert _compute_bond_momentum({}) == "HOLD"

    def test_equity_momentum_positive(self):
        yf_data = _make_yf_data(spy_ret=0.05)
        assert _compute_equity_momentum(yf_data) == "BUY"

    def test_equity_momentum_negative(self):
        yf_data = _make_yf_data(spy_ret=-0.05)
        assert _compute_equity_momentum(yf_data) == "SELL"

    def test_equity_momentum_no_data(self):
        assert _compute_equity_momentum(None) == "HOLD"


class TestYFFailure:
    """Test behavior when yfinance data is unavailable."""

    @patch("portfolio.signals.cross_asset_tsmom._fetch_yf_returns")
    def test_yf_failure_still_returns_valid(self, mock_yf):
        mock_yf.return_value = None
        df = _make_df(n=300, trend=0.1)
        result = compute_cross_asset_tsmom_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert result["sub_signals"]["bond_momentum"] == "HOLD"
        assert result["sub_signals"]["equity_momentum"] == "HOLD"
        assert result["sub_signals"]["cross_pair_63d"] == "HOLD"
