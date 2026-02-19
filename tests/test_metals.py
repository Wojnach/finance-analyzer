"""Tests for Gold/Silver (XAU/XAG) tracking via Binance Futures API."""

import json
from unittest.mock import patch, MagicMock
import pytest
import pandas as pd

from portfolio.main import (
    binance_fapi_klines,
    _fetch_klines,
    METALS_SYMBOLS,
    BINANCE_FAPI_BASE,
)


def _make_mock_kline_row():
    """Return a single mock kline row as Binance API would return."""
    return [
        1706000000000, "2650.5", "2655.0", "2648.0", "2652.3", "100.5",
        1706003599999, "266500", 500, "60.2", "159600", "0",
    ]


class TestBinanceFapiKlines:
    """Tests for binance_fapi_klines function."""

    def test_fapi_klines_calls_correct_url(self):
        """binance_fapi_klines hits fapi.binance.com."""
        mock_response = MagicMock()
        mock_response.json.return_value = [_make_mock_kline_row() for _ in range(30)]
        mock_response.raise_for_status = MagicMock()

        with patch("portfolio.main.requests.get", return_value=mock_response) as mock_get:
            df = binance_fapi_klines("XAUUSDT", interval="1h", limit=30)
            mock_get.assert_called_once()
            call_url = mock_get.call_args[0][0]
            assert "fapi.binance.com" in call_url
            assert len(df) == 30
            assert "close" in df.columns

    def test_fapi_klines_returns_float_columns(self):
        """Returned DataFrame has float columns for OHLCV."""
        mock_response = MagicMock()
        mock_response.json.return_value = [_make_mock_kline_row() for _ in range(30)]
        mock_response.raise_for_status = MagicMock()

        with patch("portfolio.main.requests.get", return_value=mock_response):
            df = binance_fapi_klines("XAUUSDT")
            assert df["close"].dtype == float
            assert df["volume"].dtype == float

    def test_fapi_klines_has_time_column(self):
        """Returned DataFrame includes a parsed 'time' column."""
        mock_response = MagicMock()
        mock_response.json.return_value = [_make_mock_kline_row() for _ in range(5)]
        mock_response.raise_for_status = MagicMock()

        with patch("portfolio.main.requests.get", return_value=mock_response):
            df = binance_fapi_klines("XAGUSDT", interval="5m", limit=5)
            assert "time" in df.columns
            assert len(df) == 5

    def test_fapi_klines_passes_params(self):
        """binance_fapi_klines passes symbol, interval, limit as query params."""
        mock_response = MagicMock()
        mock_response.json.return_value = [_make_mock_kline_row() for _ in range(10)]
        mock_response.raise_for_status = MagicMock()

        with patch("portfolio.main.requests.get", return_value=mock_response) as mock_get:
            binance_fapi_klines("XAUUSDT", interval="4h", limit=50)
            call_kwargs = mock_get.call_args
            params = call_kwargs[1]["params"] if "params" in call_kwargs[1] else call_kwargs.kwargs["params"]
            assert params["symbol"] == "XAUUSDT"
            assert params["interval"] == "4h"
            assert params["limit"] == 50


class TestFetchKlinesMetals:
    """Tests for _fetch_klines with metals sources."""

    def test_fetch_klines_routes_binance_fapi(self):
        """_fetch_klines routes binance_fapi source correctly."""
        with patch("portfolio.main.binance_fapi_klines") as mock_fapi:
            mock_fapi.return_value = pd.DataFrame({"close": [100.0]})
            result = _fetch_klines({"binance_fapi": "XAUUSDT"}, "1h", 100)
            mock_fapi.assert_called_once_with("XAUUSDT", interval="1h", limit=100)

    def test_fetch_klines_binance_still_works(self):
        """Regular binance source still works after adding binance_fapi."""
        with patch("portfolio.main.binance_klines") as mock_binance:
            mock_binance.return_value = pd.DataFrame({"close": [100.0]})
            result = _fetch_klines({"binance": "BTCUSDT"}, "1h", 100)
            mock_binance.assert_called_once_with("BTCUSDT", interval="1h", limit=100)

    def test_fetch_klines_alpaca_still_works(self):
        """Alpaca source still works after adding binance_fapi."""
        with patch("portfolio.main.alpaca_klines") as mock_alpaca:
            mock_alpaca.return_value = pd.DataFrame({"close": [100.0]})
            result = _fetch_klines({"alpaca": "MSTR"}, "1d", 100)
            mock_alpaca.assert_called_once_with("MSTR", interval="1d", limit=100)

    def test_fetch_klines_unknown_source_raises(self):
        """_fetch_klines raises ValueError for unknown source types."""
        with pytest.raises(ValueError, match="Unknown source"):
            _fetch_klines({"unknown": "FOO"}, "1h", 100)


class TestMetalsConfig:
    """Tests for metals configuration."""

    def test_metals_symbols_defined(self):
        assert "XAU-USD" in METALS_SYMBOLS
        assert "XAG-USD" in METALS_SYMBOLS

    def test_metals_not_in_crypto(self):
        from portfolio.main import CRYPTO_SYMBOLS
        assert "XAU-USD" not in CRYPTO_SYMBOLS
        assert "XAG-USD" not in CRYPTO_SYMBOLS

    def test_metals_not_in_stocks(self):
        from portfolio.main import STOCK_SYMBOLS
        assert "XAU-USD" not in STOCK_SYMBOLS
        assert "XAG-USD" not in STOCK_SYMBOLS

    def test_metals_in_symbols_dict(self):
        from portfolio.main import SYMBOLS
        assert "XAU-USD" in SYMBOLS
        assert "XAG-USD" in SYMBOLS
        assert "binance_fapi" in SYMBOLS["XAU-USD"]
        assert "binance_fapi" in SYMBOLS["XAG-USD"]
        assert SYMBOLS["XAU-USD"]["binance_fapi"] == "XAUUSDT"
        assert SYMBOLS["XAG-USD"]["binance_fapi"] == "XAGUSDT"


class TestMetalsMarketState:
    """Tests for metals inclusion in market state."""

    def test_metals_active_on_weekends(self):
        """Metals should be active on weekends like crypto."""
        from portfolio.main import get_market_state
        from datetime import datetime, timezone
        # Saturday 12:00 UTC
        with patch("portfolio.main.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 21, 12, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            state, symbols, interval = get_market_state()
            assert "XAU-USD" in symbols
            assert "XAG-USD" in symbols
            assert state == "weekend"

    def test_metals_active_when_market_closed(self):
        """Metals should be active during off-hours like crypto."""
        from portfolio.main import get_market_state
        from datetime import datetime, timezone
        # Wednesday 3:00 UTC (market closed)
        with patch("portfolio.main.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 18, 3, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            state, symbols, interval = get_market_state()
            assert "XAU-USD" in symbols
            assert "XAG-USD" in symbols
            assert state == "closed"

    def test_metals_active_during_market_hours(self):
        """Metals should be active during market hours (all symbols are)."""
        from portfolio.main import get_market_state
        from datetime import datetime, timezone
        # Wednesday 14:00 UTC (market open)
        with patch("portfolio.main.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 18, 14, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            state, symbols, interval = get_market_state()
            assert "XAU-USD" in symbols
            assert "XAG-USD" in symbols
            assert state == "open"


class TestMetalsSignalConfig:
    """Tests for metals-specific signal configuration."""

    def test_metals_total_applicable_is_5(self):
        """Metals should have 5 applicable signals (RSI, MACD, EMA, BB, Volume)."""
        from portfolio.main import generate_signal, METALS_SYMBOLS
        # Create minimal indicators
        ind = {
            "close": 2650.0,
            "rsi": 50.0,
            "rsi_p20": 30.0,
            "rsi_p80": 70.0,
            "macd_hist": 0.5,
            "macd_hist_prev": 0.3,
            "ema9": 2645.0,
            "ema21": 2640.0,
            "bb_upper": 2700.0,
            "bb_lower": 2600.0,
            "bb_mid": 2650.0,
            "price_vs_bb": "inside",
            "atr": 15.0,
            "atr_pct": 0.57,
        }
        action, conf, extra = generate_signal(ind, ticker="XAU-USD")
        assert extra["_total_applicable"] == 5

    def test_stocks_total_applicable_unchanged(self):
        """Stocks should still have 7 applicable signals."""
        from portfolio.main import generate_signal
        ind = {
            "close": 130.0,
            "rsi": 50.0,
            "rsi_p20": 30.0,
            "rsi_p80": 70.0,
            "macd_hist": 0.5,
            "macd_hist_prev": 0.3,
            "ema9": 129.0,
            "ema21": 128.0,
            "bb_upper": 135.0,
            "bb_lower": 125.0,
            "bb_mid": 130.0,
            "price_vs_bb": "inside",
            "atr": 3.0,
            "atr_pct": 2.3,
        }
        action, conf, extra = generate_signal(ind, ticker="MSTR")
        assert extra["_total_applicable"] == 7

    def test_crypto_total_applicable_unchanged(self):
        """Crypto should still have 11 applicable signals."""
        from portfolio.main import generate_signal
        ind = {
            "close": 67000.0,
            "rsi": 50.0,
            "rsi_p20": 30.0,
            "rsi_p80": 70.0,
            "macd_hist": 50.0,
            "macd_hist_prev": 30.0,
            "ema9": 66500.0,
            "ema21": 66000.0,
            "bb_upper": 70000.0,
            "bb_lower": 63000.0,
            "bb_mid": 66500.0,
            "price_vs_bb": "inside",
            "atr": 1500.0,
            "atr_pct": 2.24,
        }
        action, conf, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra["_total_applicable"] == 11
