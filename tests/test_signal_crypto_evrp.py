"""Tests for crypto_evrp signal module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from portfolio.signals.crypto_evrp import (
    compute_crypto_evrp_signal,
    _compute_realized_vol,
    _evrp_level_signal,
    _evrp_momentum_signal,
)


def _make_df(n=100, base_price=50000.0, vol=0.02):
    """Create a test DataFrame with realistic BTC-like OHLCV data."""
    np.random.seed(42)
    returns = np.random.randn(n) * vol
    close = base_price * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + abs(np.random.randn(n) * 0.005)),
        "low": close * (1 - abs(np.random.randn(n) * 0.005)),
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })


class TestRealizedVol:
    """Test the realized vol computation."""

    def test_basic_rv(self):
        df = _make_df(50)
        rv = _compute_realized_vol(df["close"], window=10)
        assert not np.isnan(rv)
        assert rv > 0
        # For 2% daily vol, annualized should be roughly 2% * sqrt(365) * 100
        assert 10 < rv < 200  # Reasonable range for crypto

    def test_insufficient_data(self):
        df = _make_df(5)
        rv = _compute_realized_vol(df["close"], window=10)
        assert np.isnan(rv)

    def test_constant_price(self):
        close = pd.Series([100.0] * 20)
        rv = _compute_realized_vol(close, window=10)
        # Constant price = 0 vol (log returns are 0, std is 0)
        assert rv == 0.0 or np.isnan(rv)


class TestEvrpLevelSignal:
    """Test the eVRP level sub-signal."""

    def test_high_evrp_sells(self):
        assert _evrp_level_signal(15.0) == "SELL"

    def test_low_evrp_buys(self):
        assert _evrp_level_signal(-15.0) == "BUY"

    def test_neutral_holds(self):
        assert _evrp_level_signal(0.0) == "HOLD"
        assert _evrp_level_signal(5.0) == "HOLD"
        assert _evrp_level_signal(-5.0) == "HOLD"

    def test_boundary_holds(self):
        assert _evrp_level_signal(10.0) == "HOLD"
        assert _evrp_level_signal(-10.0) == "HOLD"


class TestEvrpMomentum:
    """Test the eVRP momentum sub-signal."""

    def test_rising_dvol_sells(self):
        history = pd.Series([50.0, 52.0, 54.0, 56.0, 58.0, 60.0])
        vote, change = _evrp_momentum_signal(history)
        assert vote == "SELL"
        assert change > 5.0

    def test_falling_dvol_buys(self):
        history = pd.Series([60.0, 58.0, 56.0, 54.0, 52.0, 50.0])
        vote, change = _evrp_momentum_signal(history)
        assert vote == "BUY"
        assert change < -5.0

    def test_stable_dvol_holds(self):
        history = pd.Series([50.0, 50.5, 49.5, 50.2, 49.8, 50.1])
        vote, change = _evrp_momentum_signal(history)
        assert vote == "HOLD"

    def test_insufficient_history(self):
        history = pd.Series([50.0, 51.0])
        vote, change = _evrp_momentum_signal(history)
        assert vote == "HOLD"
        assert change == 0.0

    def test_none_history(self):
        vote, change = _evrp_momentum_signal(None)
        assert vote == "HOLD"


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    @patch("portfolio.signals.crypto_evrp._fetch_dvol_latest", return_value=55.0)
    @patch("portfolio.signals.crypto_evrp._fetch_dvol_history", return_value=None)
    def test_returns_dict_with_required_keys(self, mock_hist, mock_dvol):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.crypto_evrp._fetch_dvol_latest", return_value=55.0)
    @patch("portfolio.signals.crypto_evrp._fetch_dvol_history", return_value=None)
    def test_has_sub_signals(self, mock_hist, mock_dvol):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    @patch("portfolio.signals.crypto_evrp._fetch_dvol_latest", return_value=55.0)
    @patch("portfolio.signals.crypto_evrp._fetch_dvol_history", return_value=None)
    def test_has_indicators(self, mock_hist, mock_dvol):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "dvol_30d" in result["indicators"]
        assert "rv_10d" in result["indicators"]
        assert "evrp" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_crypto_evrp_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_crypto_evrp_signal(df)
        assert result["action"] == "HOLD"

    def test_none_dataframe_returns_hold(self):
        result = compute_crypto_evrp_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.crypto_evrp._fetch_dvol_latest", return_value=55.0)
    @patch("portfolio.signals.crypto_evrp._fetch_dvol_history", return_value=None)
    def test_nan_handling(self, mock_hist, mock_dvol):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    @patch("portfolio.signals.crypto_evrp._fetch_dvol_latest", return_value=55.0)
    @patch("portfolio.signals.crypto_evrp._fetch_dvol_history", return_value=None)
    def test_confidence_capped_at_0_7(self, mock_hist, mock_dvol):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        assert result["confidence"] <= 0.7


class TestAssetFiltering:
    """Test that signal only applies to crypto assets."""

    def test_metals_returns_hold(self):
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "asset_class": "metals"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_stocks_returns_hold(self):
        df = _make_df()
        ctx = {"ticker": "MSTR", "asset_class": "stocks"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @patch("portfolio.signals.crypto_evrp._fetch_dvol_latest", return_value=55.0)
    @patch("portfolio.signals.crypto_evrp._fetch_dvol_history", return_value=None)
    def test_btc_processes(self, mock_hist, mock_dvol):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        assert result["indicators"].get("currency") == "BTC"

    @patch("portfolio.signals.crypto_evrp._fetch_dvol_latest", return_value=45.0)
    @patch("portfolio.signals.crypto_evrp._fetch_dvol_history", return_value=None)
    def test_eth_processes(self, mock_hist, mock_dvol):
        df = _make_df(base_price=3000.0)
        ctx = {"ticker": "ETH-USD", "asset_class": "crypto"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        assert result["indicators"].get("currency") == "ETH"


class TestDvolFailure:
    """Test graceful handling when Deribit is unavailable."""

    @patch("portfolio.signals.crypto_evrp._fetch_dvol_latest", return_value=None)
    def test_dvol_unavailable_returns_hold(self, mock_dvol):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0
        assert result["indicators"]["dvol"] is None

    @patch("portfolio.signals.crypto_evrp._fetch_dvol_latest", return_value=0.0)
    def test_dvol_zero_returns_hold(self, mock_dvol):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        assert result["action"] == "HOLD"


class TestEvrpSignalLogic:
    """Test the actual signal generation logic with known values."""

    @patch("portfolio.signals.crypto_evrp._fetch_dvol_history", return_value=None)
    @patch("portfolio.signals.crypto_evrp._fetch_dvol_latest")
    def test_high_evrp_tends_sell(self, mock_dvol, mock_hist):
        # Very high DVOL (80) vs low realized vol should produce SELL
        mock_dvol.return_value = 80.0
        df = _make_df(n=50, vol=0.005)  # Low vol data
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        # With DVOL=80 and low RV, eVRP should be very positive
        evrp = result["indicators"].get("evrp")
        if evrp is not None:
            assert evrp > 10  # High eVRP

    @patch("portfolio.signals.crypto_evrp._fetch_dvol_history", return_value=None)
    @patch("portfolio.signals.crypto_evrp._fetch_dvol_latest")
    def test_low_evrp_tends_buy(self, mock_dvol, mock_hist):
        # Low DVOL (20) vs high realized vol should produce BUY
        mock_dvol.return_value = 20.0
        df = _make_df(n=50, vol=0.05)  # High vol data
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto"}
        result = compute_crypto_evrp_signal(df, context=ctx)
        evrp = result["indicators"].get("evrp")
        if evrp is not None:
            assert evrp < 0  # Negative eVRP
