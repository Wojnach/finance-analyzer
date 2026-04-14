"""Tests for portfolio.price_source — the yfinance-router."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from portfolio.price_source import (
    SourceUnavailableError,
    fetch_klines,
    is_yfinance_allowed,
    resolve_source,
)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame({
        "open": [1.0], "high": [1.0], "low": [1.0],
        "close": [1.0], "volume": [1.0],
    })


class TestResolveSource:
    @pytest.mark.parametrize("ticker,expected", [
        ("XAG-USD", "binance_fapi"),
        ("SI=F", "binance_fapi"),
        ("XAGUSDT", "binance_fapi"),
        ("XAU-USD", "binance_fapi"),
        ("GC=F", "binance_fapi"),
        ("BTC-USD", "binance_spot"),
        ("ETH-USD", "binance_spot"),
        ("BTCUSDT", "binance_spot"),
    ])
    def test_binance_routing(self, ticker, expected):
        assert resolve_source(ticker) == expected

    @pytest.mark.parametrize("ticker", ["MSTR", "SPY", "QQQ", "USO", "TLT", "AAPL"])
    def test_stocks_route_to_alpaca(self, ticker):
        assert resolve_source(ticker) == "alpaca"

    @pytest.mark.parametrize("ticker", ["^VIX", "^VIX3M", "^OVX", "^GVZ", "^RVX"])
    def test_cboe_indices_route_to_yfinance(self, ticker):
        assert resolve_source(ticker) == "yfinance"

    @pytest.mark.parametrize("ticker", ["HG=F", "DX-Y.NYB", "EURUSD=X"])
    def test_last_resort_tickers_route_to_yfinance(self, ticker):
        assert resolve_source(ticker) == "yfinance"

    def test_unknown_caret_prefix_routes_to_yfinance(self):
        """^SOMETHINGWEIRD defaults to yfinance — it's almost certainly an index."""
        assert resolve_source("^DJI") == "yfinance"


class TestIsYfinanceAllowed:
    @pytest.mark.parametrize("ticker,allowed", [
        ("^VIX", True),
        ("^VIX3M", True),
        ("^OVX", True),
        ("HG=F", True),
        ("DX-Y.NYB", True),
        ("XAG-USD", False),
        ("BTC-USD", False),
        ("MSTR", False),
    ])
    def test_matrix(self, ticker, allowed):
        assert is_yfinance_allowed(ticker) is allowed


class TestFetchKlines:
    """Verify dispatcher calls the correct backend fetcher for each class."""

    @patch("portfolio.price_source._fetch_binance_fapi")
    def test_xag_goes_to_binance_fapi(self, mock_fapi):
        mock_fapi.return_value = _empty_df()
        fetch_klines("XAG-USD", interval="1m", limit=10)
        mock_fapi.assert_called_once_with("XAGUSDT", "1m", 10)

    @patch("portfolio.price_source._fetch_binance_fapi")
    def test_si_future_alias_goes_to_binance_fapi(self, mock_fapi):
        mock_fapi.return_value = _empty_df()
        fetch_klines("SI=F", interval="1h", limit=5)
        mock_fapi.assert_called_once_with("XAGUSDT", "1h", 5)

    @patch("portfolio.price_source._fetch_binance_spot")
    def test_btc_goes_to_binance_spot(self, mock_spot):
        mock_spot.return_value = _empty_df()
        fetch_klines("BTC-USD", interval="5m", limit=20)
        mock_spot.assert_called_once_with("BTCUSDT", "5m", 20)

    @patch("portfolio.price_source._fetch_alpaca")
    def test_stock_goes_to_alpaca(self, mock_alpaca):
        mock_alpaca.return_value = _empty_df()
        fetch_klines("MSTR", interval="1h", limit=30)
        mock_alpaca.assert_called_once_with("MSTR", "1h", 30)

    @patch("portfolio.price_source._fetch_yfinance")
    def test_cboe_vix_goes_to_yfinance(self, mock_yf):
        mock_yf.return_value = _empty_df()
        fetch_klines("^VIX", interval="1d", limit=60, period="2mo")
        mock_yf.assert_called_once_with("^VIX", "1d", period="2mo", limit=60)


class TestFailoverBehavior:
    """When a primary source fails, fall back to yfinance as last resort."""

    @patch("portfolio.price_source._fetch_yfinance")
    @patch("portfolio.price_source._fetch_binance_fapi")
    def test_binance_failure_falls_back_to_yfinance(self, mock_fapi, mock_yf):
        mock_fapi.side_effect = ConnectionError("Binance down")
        mock_yf.return_value = _empty_df()
        result = fetch_klines("XAG-USD", interval="1h", limit=10)
        assert not result.empty
        mock_yf.assert_called_once()

    @patch("portfolio.price_source._fetch_yfinance")
    @patch("portfolio.price_source._fetch_alpaca")
    def test_alpaca_failure_falls_back_to_yfinance(self, mock_alpaca, mock_yf):
        mock_alpaca.side_effect = ConnectionError("Alpaca down")
        mock_yf.return_value = _empty_df()
        result = fetch_klines("MSTR", interval="1d", limit=30)
        assert not result.empty
        mock_yf.assert_called_once()

    @patch("portfolio.price_source._fetch_yfinance")
    @patch("portfolio.price_source._fetch_binance_fapi")
    def test_both_sources_fail_raises_source_unavailable(self, mock_fapi, mock_yf):
        mock_fapi.side_effect = ConnectionError("Binance down")
        mock_yf.side_effect = ConnectionError("yfinance down")
        with pytest.raises(SourceUnavailableError):
            fetch_klines("XAG-USD", interval="1h", limit=10)

    @patch("portfolio.price_source._fetch_yfinance")
    def test_yfinance_only_ticker_failure_raises_source_unavailable(self, mock_yf):
        """For ^VIX etc, a yfinance failure has no fallback — raises directly."""
        mock_yf.side_effect = ConnectionError("yfinance down")
        with pytest.raises(SourceUnavailableError):
            fetch_klines("^VIX", interval="1d", limit=60)


class TestReturnContract:
    """The dispatcher must return the DataFrame as-is from the backend;
    lowercase OHLCV columns are the contract."""

    @patch("portfolio.price_source._fetch_binance_fapi")
    def test_passthrough_dataframe_from_binance(self, mock_fapi):
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]})
        mock_fapi.return_value = df
        result = fetch_klines("XAG-USD", "1m", 1)
        pd.testing.assert_frame_equal(result, df)
