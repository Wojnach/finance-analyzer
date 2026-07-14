"""Fail-closed behavior for traded-instrument price fetches (2026-07-14)."""

from unittest.mock import patch

import pytest

from portfolio.price_source import SourceUnavailableError, fetch_klines


class TestFailClosed:
    @patch("portfolio.price_source._price_fail_closed", return_value=True)
    @patch(
        "portfolio.price_source._fetch_binance_spot",
        side_effect=RuntimeError("binance down"),
    )
    @patch("portfolio.price_source._fetch_yfinance")
    def test_traded_instrument_fails_closed_no_yfinance(self, mock_yf, _b, _fc):
        # BTC-USD (binance_spot) primary fails -> must RAISE, never call yfinance
        with pytest.raises(SourceUnavailableError, match="fail-closed"):
            fetch_klines("BTC-USD", interval="15m")
        mock_yf.assert_not_called()

    @patch("portfolio.price_source._price_fail_closed", return_value=True)
    @patch(
        "portfolio.price_source._fetch_alpaca", side_effect=RuntimeError("alpaca down")
    )
    @patch("portfolio.price_source._fetch_yfinance")
    def test_stock_fails_closed(self, mock_yf, _a, _fc):
        with pytest.raises(SourceUnavailableError, match="fail-closed"):
            fetch_klines("MSTR", interval="15m")
        mock_yf.assert_not_called()

    @patch("portfolio.price_source._price_fail_closed", return_value=False)
    @patch(
        "portfolio.price_source._fetch_binance_spot",
        side_effect=RuntimeError("binance down"),
    )
    @patch("portfolio.price_source._fetch_yfinance")
    def test_opt_out_restores_fallback(self, mock_yf, _b, _fc):
        # config price_source.fail_closed=false -> old behavior, yfinance used
        import pandas as pd

        mock_yf.return_value = pd.DataFrame({"close": [1.0]})
        df = fetch_klines("BTC-USD", interval="15m")
        mock_yf.assert_called_once()
        assert df.attrs["_source"] == "yfinance_fallback"

    @patch("portfolio.price_source._fetch_yfinance")
    def test_yfinance_native_ticker_unaffected(self, mock_yf):
        # A yfinance-only ticker (VIX) must still use yfinance directly —
        # fail-closed only guards binance/alpaca sources.
        import pandas as pd

        mock_yf.return_value = pd.DataFrame({"close": [20.0]})
        df = fetch_klines("^VIX", interval="1d")
        mock_yf.assert_called_once()
        assert "close" in df.columns
