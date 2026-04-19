"""Tests for network_momentum signal module."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.network_momentum import (
    _compute_correlation_regime,
    _compute_network_divergence,
    _compute_own_tsmom,
    _vol_scaled_return,
    compute_network_momentum_signal,
)


def _make_df(n=100, trend=0.0, seed=42):
    """Create a test DataFrame with realistic OHLCV data."""
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n) * 0.5 + trend)
    close = np.maximum(close, 1.0)  # prevent negative prices
    return pd.DataFrame({
        "open": close + rng.randn(n) * 0.1,
        "high": close + abs(rng.randn(n) * 0.3),
        "low": close - abs(rng.randn(n) * 0.3),
        "close": close,
        "volume": rng.randint(1000, 10000, n).astype(float),
    })


def _make_peer_closes(n=100, seed=42):
    """Create a mock peer closes DataFrame."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    data = {}
    for ticker in ["BTC-USD", "ETH-USD", "GC=F", "SI=F", "SPY"]:
        base = 100 + rng.randn() * 20
        prices = base + np.cumsum(rng.randn(n) * 0.5)
        data[ticker] = np.maximum(prices, 1.0)
    return pd.DataFrame(data, index=dates)


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    @patch("portfolio.signals.network_momentum._fetch_peer_closes", return_value=None)
    def test_returns_dict_with_required_keys(self, _mock):
        df = _make_df()
        result = compute_network_momentum_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.network_momentum._fetch_peer_closes", return_value=None)
    def test_has_sub_signals(self, _mock):
        df = _make_df()
        result = compute_network_momentum_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    @patch("portfolio.signals.network_momentum._fetch_peer_closes", return_value=None)
    def test_has_indicators(self, _mock):
        df = _make_df()
        result = compute_network_momentum_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    @patch("portfolio.signals.network_momentum._fetch_peer_closes", return_value=None)
    def test_empty_dataframe_returns_hold(self, _mock):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_network_momentum_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.network_momentum._fetch_peer_closes", return_value=None)
    def test_insufficient_rows_returns_hold(self, _mock):
        df = _make_df(n=3)
        result = compute_network_momentum_signal(df)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.network_momentum._fetch_peer_closes", return_value=None)
    def test_nan_handling(self, _mock):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_network_momentum_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    @patch("portfolio.signals.network_momentum._fetch_peer_closes", return_value=None)
    def test_with_context(self, _mock):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_network_momentum_signal(df, context=ctx)
        assert isinstance(result, dict)

    @patch("portfolio.signals.network_momentum._fetch_peer_closes", return_value=None)
    def test_confidence_capped_at_0_7(self, _mock):
        df = _make_df(n=200, trend=0.1)
        result = compute_network_momentum_signal(df)
        assert result["confidence"] <= 0.7

    @patch("portfolio.signals.network_momentum._fetch_peer_closes", return_value=None)
    def test_none_dataframe_returns_hold(self, _mock):
        result = compute_network_momentum_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


class TestVolScaledReturn:
    """Test volatility-scaled return computation."""

    def test_basic_computation(self):
        close = pd.Series(range(1, 102), dtype=float)
        result = _vol_scaled_return(close, 20)
        assert np.isfinite(result)
        assert result > 0  # uptrend should give positive

    def test_insufficient_data(self):
        close = pd.Series([100.0, 101.0])
        result = _vol_scaled_return(close, 20)
        assert result == 0.0

    def test_flat_series(self):
        close = pd.Series([100.0] * 50)
        result = _vol_scaled_return(close, 20)
        assert result == 0.0  # zero vol -> zero return


class TestOwnTsmom:
    """Test own multi-scale momentum sub-indicator."""

    def test_uptrend_returns_buy(self):
        close = pd.Series(range(1, 202), dtype=float)
        action, conf = _compute_own_tsmom(close)
        assert action == "BUY"
        assert conf > 0.0

    def test_downtrend_returns_sell(self):
        close = pd.Series(range(200, 0, -1), dtype=float)
        action, conf = _compute_own_tsmom(close)
        assert action == "SELL"
        assert conf > 0.0

    def test_short_series_returns_hold(self):
        close = pd.Series([100.0, 101.0, 102.0])
        action, conf = _compute_own_tsmom(close)
        assert action == "HOLD"
        assert conf == 0.0


class TestNetworkDivergence:
    """Test network divergence computation."""

    def test_with_peer_data(self):
        own_close = pd.Series(range(1, 102), dtype=float)
        peer_closes = _make_peer_closes(n=100)
        action, conf, indicators = _compute_network_divergence(
            own_close, peer_closes, "BTC-USD"
        )
        assert action in ("BUY", "SELL", "HOLD")
        assert isinstance(indicators, dict)

    def test_unknown_ticker(self):
        own_close = pd.Series(range(1, 102), dtype=float)
        peer_closes = _make_peer_closes(n=100)
        action, conf, indicators = _compute_network_divergence(
            own_close, peer_closes, "UNKNOWN"
        )
        assert action in ("BUY", "SELL", "HOLD")

    def test_empty_peer_closes(self):
        own_close = pd.Series(range(1, 102), dtype=float)
        peer_closes = pd.DataFrame()
        action, conf, indicators = _compute_network_divergence(
            own_close, peer_closes, "BTC-USD"
        )
        assert action == "HOLD"
        assert conf == 0.0


class TestCorrelationRegime:
    """Test correlation regime sub-indicator."""

    def test_with_correlated_peers(self):
        peer_closes = _make_peer_closes(n=100, seed=42)
        action, conf = _compute_correlation_regime(peer_closes, "BTC-USD")
        assert action in ("BUY", "SELL", "HOLD")

    def test_unknown_ticker(self):
        peer_closes = _make_peer_closes(n=100)
        action, conf = _compute_correlation_regime(peer_closes, "UNKNOWN")
        assert action == "HOLD"
        assert conf == 0.0

    def test_short_data(self):
        peer_closes = _make_peer_closes(n=10)
        action, conf = _compute_correlation_regime(peer_closes, "BTC-USD")
        assert action == "HOLD"


class TestWithMockPeers:
    """Test full signal with mocked peer data."""

    def test_with_peer_data_btc(self):
        df = _make_df(n=100)
        peer_closes = _make_peer_closes(n=100)
        with patch(
            "portfolio.signals.network_momentum._fetch_peer_closes",
            return_value=peer_closes,
        ):
            result = compute_network_momentum_signal(
                df, context={"ticker": "BTC-USD"}
            )
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 0.7
        assert "own_tsmom" in result["sub_signals"]
        assert "network_divergence" in result["sub_signals"]
        assert "correlation_regime" in result["sub_signals"]

    def test_with_peer_data_eth(self):
        df = _make_df(n=100, seed=99)
        peer_closes = _make_peer_closes(n=100, seed=99)
        with patch(
            "portfolio.signals.network_momentum._fetch_peer_closes",
            return_value=peer_closes,
        ):
            result = compute_network_momentum_signal(
                df, context={"ticker": "ETH-USD"}
            )
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_peer_data_xau(self):
        df = _make_df(n=100, seed=123)
        peer_closes = _make_peer_closes(n=100, seed=123)
        with patch(
            "portfolio.signals.network_momentum._fetch_peer_closes",
            return_value=peer_closes,
        ):
            result = compute_network_momentum_signal(
                df, context={"ticker": "XAU-USD"}
            )
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_all_tickers(self):
        """Test all 5 tickers produce valid output."""
        peer_closes = _make_peer_closes(n=100)
        for ticker in ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"]:
            df = _make_df(n=100)
            with patch(
                "portfolio.signals.network_momentum._fetch_peer_closes",
                return_value=peer_closes,
            ):
                result = compute_network_momentum_signal(
                    df, context={"ticker": ticker}
                )
            assert result["action"] in ("BUY", "SELL", "HOLD"), (
                f"Failed for {ticker}"
            )
            assert 0.0 <= result["confidence"] <= 0.7, (
                f"Confidence out of range for {ticker}"
            )

    def test_indicators_present(self):
        df = _make_df(n=100)
        peer_closes = _make_peer_closes(n=100)
        with patch(
            "portfolio.signals.network_momentum._fetch_peer_closes",
            return_value=peer_closes,
        ):
            result = compute_network_momentum_signal(
                df, context={"ticker": "BTC-USD"}
            )
        ind = result["indicators"]
        assert "own_mom_5d" in ind
        assert "own_mom_20d" in ind
        assert "own_mom_60d" in ind
