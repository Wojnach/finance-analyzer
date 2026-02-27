"""Comprehensive tests for portfolio/data_collector.py.

Covers:
  - _binance_fetch: spot/fapi kline parsing, circuit breaker open, ConnectionError on None,
    raise_for_status, failure recording
  - binance_klines / binance_fapi_klines: wrapper delegation
  - alpaca_klines: bar parsing, circuit breaker, empty bars, unsupported interval
  - yfinance_klines: download mocking, empty result, MultiIndex flattening, unsupported interval
  - _fetch_klines: dispatcher routing (binance/fapi/alpaca/yfinance), market-closed yfinance
    fallback, unknown source error
  - collect_timeframes: cache hit, cache miss, compute_indicators returning None (insufficient
    data logging), error handling, stock vs crypto timeframe selection
"""

import time
from unittest import mock
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

import portfolio.shared_state as _ss
from portfolio.data_collector import (
    _binance_fetch,
    binance_klines,
    binance_fapi_klines,
    alpaca_klines,
    yfinance_klines,
    _fetch_klines,
    collect_timeframes,
    _BINANCE_KLINE_COLS,
    ALPACA_INTERVAL_MAP,
    _YF_INTERVAL_MAP,
    TIMEFRAMES,
    STOCK_TIMEFRAMES,
    binance_spot_cb,
    binance_fapi_cb,
    alpaca_cb,
)
from portfolio.circuit_breaker import CircuitBreaker


# ---------------------------------------------------------------------------
# Helpers — realistic API response builders
# ---------------------------------------------------------------------------

def _make_binance_kline_row(open_time=1700000000000, o=68000.0, h=68500.0,
                             l=67500.0, c=68200.0, v=123.45):
    """Build one row of Binance kline data (12-element list)."""
    return [
        open_time, str(o), str(h), str(l), str(c), str(v),
        open_time + 300000,   # close_time
        str(v * c),           # quote_vol
        100,                  # trades
        str(v * 0.6),         # taker_buy_vol
        str(v * 0.6 * c),    # taker_buy_quote_vol
        "0",                  # ignore
    ]


def _make_binance_response(n=5, base_open=68000.0):
    """Build a list of n Binance kline rows with increasing prices."""
    rows = []
    for i in range(n):
        rows.append(_make_binance_kline_row(
            open_time=1700000000000 + i * 300000,
            o=base_open + i * 10,
            h=base_open + i * 10 + 50,
            l=base_open + i * 10 - 50,
            c=base_open + i * 10 + 5,
            v=100.0 + i,
        ))
    return rows


def _make_alpaca_bars(n=5, base_close=185.0):
    """Build a list of Alpaca bar dicts."""
    bars = []
    for i in range(n):
        bars.append({
            "o": base_close + i * 0.5,
            "h": base_close + i * 0.5 + 1.0,
            "l": base_close + i * 0.5 - 1.0,
            "c": base_close + i * 0.5 + 0.2,
            "v": 50000 + i * 100,
            "t": f"2026-02-27T14:{i:02d}:00Z",
        })
    return bars


def _make_mock_response(json_data, status_code=200):
    """Create a mock HTTP response object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_circuit_breakers():
    """Reset all circuit breakers to CLOSED before each test."""
    for cb in (binance_spot_cb, binance_fapi_cb, alpaca_cb):
        cb._state = cb._state.__class__("CLOSED")
        cb._failure_count = 0
        cb._last_failure_time = None
    yield


@pytest.fixture(autouse=True)
def _clear_tool_cache():
    """Clear shared state tool cache before each test."""
    _ss._tool_cache.clear()
    yield
    _ss._tool_cache.clear()


@pytest.fixture
def fresh_cb():
    """Provide a fresh CircuitBreaker for isolated tests."""
    return CircuitBreaker("test_cb", failure_threshold=3, recovery_timeout=10)


# ===========================================================================
# _binance_fetch
# ===========================================================================

class TestBinanceFetch:

    @patch("portfolio.data_collector.fetch_with_retry")
    def test_successful_fetch_returns_dataframe(self, mock_fetch):
        """Successful Binance kline fetch returns DataFrame with correct columns."""
        data = _make_binance_response(10)
        mock_fetch.return_value = _make_mock_response(data)
        cb = CircuitBreaker("test", failure_threshold=5, recovery_timeout=60)

        df = _binance_fetch("https://api.binance.com/api/v3", cb, "spot",
                            "BTCUSDT", interval="15m", limit=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns
            assert df[col].dtype == float
        assert "time" in df.columns

    @patch("portfolio.data_collector.fetch_with_retry")
    def test_fetch_records_success_on_cb(self, mock_fetch):
        """Circuit breaker records success after a good fetch."""
        data = _make_binance_response(3)
        mock_fetch.return_value = _make_mock_response(data)
        cb = CircuitBreaker("test", failure_threshold=5, recovery_timeout=60)
        cb._failure_count = 2  # simulate prior failures

        _binance_fetch("https://api.binance.com/api/v3", cb, "spot", "BTCUSDT")

        assert cb._failure_count == 0

    @patch("portfolio.data_collector.fetch_with_retry")
    def test_circuit_breaker_open_raises_connection_error(self, mock_fetch):
        """When circuit breaker is open, raises ConnectionError without calling API."""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=600)
        # Trip the CB
        cb.record_failure()
        cb.record_failure()

        with pytest.raises(ConnectionError, match="circuit open"):
            _binance_fetch("https://api.binance.com/api/v3", cb, "spot", "BTCUSDT")

        mock_fetch.assert_not_called()

    @patch("portfolio.data_collector.fetch_with_retry")
    def test_fetch_returns_none_raises_connection_error(self, mock_fetch):
        """When fetch_with_retry returns None, raises ConnectionError."""
        mock_fetch.return_value = None
        cb = CircuitBreaker("test", failure_threshold=5, recovery_timeout=60)

        with pytest.raises(ConnectionError, match="request failed"):
            _binance_fetch("https://api.binance.com/api/v3", cb, "spot", "BTCUSDT")

    @patch("portfolio.data_collector.fetch_with_retry")
    def test_fetch_records_failure_on_exception(self, mock_fetch):
        """Circuit breaker records failure when fetch raises."""
        mock_fetch.return_value = None
        cb = CircuitBreaker("test", failure_threshold=5, recovery_timeout=60)

        with pytest.raises(ConnectionError):
            _binance_fetch("https://api.binance.com/api/v3", cb, "spot", "ETHUSDT")

        assert cb._failure_count == 1

    @patch("portfolio.data_collector.fetch_with_retry")
    def test_raise_for_status_propagates(self, mock_fetch):
        """HTTP error from raise_for_status propagates and records failure."""
        resp = _make_mock_response([], status_code=500)
        mock_fetch.return_value = resp
        cb = CircuitBreaker("test", failure_threshold=5, recovery_timeout=60)

        with pytest.raises(Exception, match="HTTP 500"):
            _binance_fetch("https://api.binance.com/api/v3", cb, "spot", "BTCUSDT")

        assert cb._failure_count == 1

    @patch("portfolio.data_collector.fetch_with_retry")
    def test_ohlcv_values_are_correct(self, mock_fetch):
        """Verify that OHLCV values are correctly parsed from the raw data."""
        row = _make_binance_kline_row(o=68000.0, h=68500.0, l=67500.0, c=68200.0, v=123.45)
        mock_fetch.return_value = _make_mock_response([row])
        cb = CircuitBreaker("test", failure_threshold=5, recovery_timeout=60)

        df = _binance_fetch("https://api.binance.com/api/v3", cb, "spot", "BTCUSDT")

        assert df.iloc[0]["open"] == pytest.approx(68000.0)
        assert df.iloc[0]["high"] == pytest.approx(68500.0)
        assert df.iloc[0]["low"] == pytest.approx(67500.0)
        assert df.iloc[0]["close"] == pytest.approx(68200.0)
        assert df.iloc[0]["volume"] == pytest.approx(123.45)


# ===========================================================================
# binance_klines / binance_fapi_klines wrappers
# ===========================================================================

class TestBinanceWrappers:

    @patch("portfolio.data_collector._binance_fetch")
    def test_binance_klines_delegates_to_spot(self, mock_bf):
        """binance_klines calls _binance_fetch with spot URL and CB."""
        mock_bf.return_value = pd.DataFrame()
        binance_klines("BTCUSDT", interval="1h", limit=50)

        mock_bf.assert_called_once()
        args = mock_bf.call_args
        assert "api.binance.com" in args[0][0]
        assert args[0][2] == "spot"
        assert args[0][3] == "BTCUSDT"

    @patch("portfolio.data_collector._binance_fetch")
    def test_binance_fapi_klines_delegates_to_fapi(self, mock_bf):
        """binance_fapi_klines calls _binance_fetch with FAPI URL and CB."""
        mock_bf.return_value = pd.DataFrame()
        binance_fapi_klines("XAUUSDT", interval="4h", limit=100)

        mock_bf.assert_called_once()
        args = mock_bf.call_args
        assert "fapi.binance.com" in args[0][0]
        assert args[0][2] == "FAPI"
        assert args[0][3] == "XAUUSDT"


# ===========================================================================
# alpaca_klines
# ===========================================================================

class TestAlpacaKlines:

    @patch("portfolio.data_collector.get_alpaca_headers", return_value={"APCA-API-KEY-ID": "k"})
    @patch("portfolio.data_collector.fetch_with_retry")
    def test_successful_alpaca_fetch(self, mock_fetch, mock_headers):
        """Successful Alpaca fetch returns DataFrame with renamed columns."""
        bars = _make_alpaca_bars(10)
        mock_fetch.return_value = _make_mock_response({"bars": bars})

        df = alpaca_klines("NVDA", interval="1d", limit=100)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        for col in ["open", "high", "low", "close", "volume", "time"]:
            assert col in df.columns
        assert df["close"].dtype == float

    @patch("portfolio.data_collector.get_alpaca_headers", return_value={})
    @patch("portfolio.data_collector.fetch_with_retry")
    def test_alpaca_empty_bars_raises_value_error(self, mock_fetch, mock_headers):
        """Empty bars list raises ValueError."""
        mock_fetch.return_value = _make_mock_response({"bars": []})

        with pytest.raises(ValueError, match="No Alpaca data"):
            alpaca_klines("NVDA", interval="1d", limit=100)

    @patch("portfolio.data_collector.get_alpaca_headers", return_value={})
    @patch("portfolio.data_collector.fetch_with_retry")
    def test_alpaca_none_bars_raises_value_error(self, mock_fetch, mock_headers):
        """Null bars in response raises ValueError."""
        mock_fetch.return_value = _make_mock_response({"bars": None})

        with pytest.raises(ValueError, match="No Alpaca data"):
            alpaca_klines("NVDA", interval="1d", limit=100)

    def test_alpaca_unsupported_interval_raises(self):
        """Unsupported interval raises ValueError before any network call."""
        with pytest.raises(ValueError, match="Unsupported Alpaca interval"):
            alpaca_klines("NVDA", interval="3d", limit=100)

    @patch("portfolio.data_collector.get_alpaca_headers", return_value={})
    @patch("portfolio.data_collector.fetch_with_retry")
    def test_alpaca_circuit_breaker_open(self, mock_fetch, mock_headers):
        """When Alpaca CB is open, raises ConnectionError without calling API."""
        # Trip the CB by recording 5 failures
        for _ in range(5):
            alpaca_cb.record_failure()

        with pytest.raises(ConnectionError, match="circuit open"):
            alpaca_klines("NVDA", interval="1d", limit=100)

        mock_fetch.assert_not_called()

    @patch("portfolio.data_collector.get_alpaca_headers", return_value={})
    @patch("portfolio.data_collector.fetch_with_retry")
    def test_alpaca_returns_none_raises_connection_error(self, mock_fetch, mock_headers):
        """fetch_with_retry returning None raises ConnectionError."""
        mock_fetch.return_value = None

        with pytest.raises(ConnectionError, match="Alpaca request failed"):
            alpaca_klines("NVDA", interval="1d", limit=100)

    @patch("portfolio.data_collector.get_alpaca_headers", return_value={})
    @patch("portfolio.data_collector.fetch_with_retry")
    def test_alpaca_records_success_on_cb(self, mock_fetch, mock_headers):
        """Circuit breaker records success after a good Alpaca fetch."""
        bars = _make_alpaca_bars(5)
        mock_fetch.return_value = _make_mock_response({"bars": bars})
        alpaca_cb._failure_count = 2

        alpaca_klines("NVDA", interval="1d", limit=100)

        assert alpaca_cb._failure_count == 0

    @patch("portfolio.data_collector.get_alpaca_headers", return_value={})
    @patch("portfolio.data_collector.fetch_with_retry")
    def test_alpaca_tail_limits_result(self, mock_fetch, mock_headers):
        """Result is tail-limited to requested limit."""
        bars = _make_alpaca_bars(20)
        mock_fetch.return_value = _make_mock_response({"bars": bars})

        df = alpaca_klines("NVDA", interval="1d", limit=5)

        assert len(df) == 5


# ===========================================================================
# yfinance_klines
# ===========================================================================

class TestYfinanceKlines:

    def _make_yf_mock(self):
        """Create a mock yfinance module."""
        mock_yf = MagicMock()
        return mock_yf

    def _make_yf_df(self, n=10, multi_index=False):
        """Build a DataFrame resembling yfinance output."""
        idx = pd.date_range("2026-02-20", periods=n, freq="1D")
        if multi_index:
            arrays = [
                ["Open", "High", "Low", "Close", "Volume"],
                ["NVDA", "NVDA", "NVDA", "NVDA", "NVDA"],
            ]
            tuples = list(zip(*arrays))
            cols = pd.MultiIndex.from_tuples(tuples)
            data = [[100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 50000] for i in range(n)]
            return pd.DataFrame(data, columns=cols, index=idx)
        return pd.DataFrame({
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [50000] * n,
        }, index=idx)

    def test_successful_yfinance_fetch(self):
        """Successful yfinance download returns DataFrame with correct columns."""
        mock_yf = self._make_yf_mock()
        mock_yf.download.return_value = self._make_yf_df(10)

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("portfolio.tickers.YF_MAP", {"NVDA": "NVDA"}):
                result = yfinance_klines("NVDA", interval="1d", limit=100)

        assert isinstance(result, pd.DataFrame)
        assert "close" in result.columns
        assert "time" in result.columns
        assert len(result) == 10

    def test_yfinance_empty_raises_value_error(self):
        """Empty yfinance result raises ValueError."""
        mock_yf = self._make_yf_mock()
        mock_yf.download.return_value = pd.DataFrame()

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("portfolio.tickers.YF_MAP", {"NVDA": "NVDA"}):
                with pytest.raises(ValueError, match="No yfinance data"):
                    yfinance_klines("NVDA", interval="1d", limit=100)

    def test_yfinance_none_raises_value_error(self):
        """None yfinance result raises ValueError."""
        mock_yf = self._make_yf_mock()
        mock_yf.download.return_value = None

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("portfolio.tickers.YF_MAP", {"NVDA": "NVDA"}):
                with pytest.raises(ValueError, match="No yfinance data"):
                    yfinance_klines("NVDA", interval="1d", limit=100)

    def test_yfinance_multiindex_flattened(self):
        """MultiIndex columns (common in yfinance) are properly flattened."""
        mock_yf = self._make_yf_mock()
        mock_yf.download.return_value = self._make_yf_df(5, multi_index=True)

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("portfolio.tickers.YF_MAP", {"NVDA": "NVDA"}):
                result = yfinance_klines("NVDA", interval="1d", limit=100)

        assert "close" in result.columns
        assert len(result) == 5

    def test_yfinance_unsupported_interval_raises(self):
        """Unsupported interval raises ValueError."""
        mock_yf = self._make_yf_mock()

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("portfolio.tickers.YF_MAP", {"NVDA": "NVDA"}):
                with pytest.raises(ValueError, match="Unsupported yfinance interval"):
                    yfinance_klines("NVDA", interval="3d", limit=100)

    def test_yfinance_tail_limits_result(self):
        """Result is tail-limited to requested limit."""
        mock_yf = self._make_yf_mock()
        mock_yf.download.return_value = self._make_yf_df(20)

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("portfolio.tickers.YF_MAP", {"NVDA": "NVDA"}):
                result = yfinance_klines("NVDA", interval="1d", limit=5)

        assert len(result) == 5

    def test_yfinance_uses_yf_map(self):
        """The YF_MAP is consulted for ticker translation."""
        mock_yf = self._make_yf_mock()
        mock_yf.download.return_value = pd.DataFrame()

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            with patch("portfolio.tickers.YF_MAP", {"NVDA": "NVDA-CUSTOM"}):
                with pytest.raises(ValueError):
                    yfinance_klines("NVDA", interval="1d", limit=100)

        # Verify download was called with the mapped ticker
        mock_yf.download.assert_called_once()
        call_args = mock_yf.download.call_args
        assert call_args[0][0] == "NVDA-CUSTOM"


# ===========================================================================
# _fetch_klines dispatcher
# ===========================================================================

class TestFetchKlinesDispatcher:

    @patch("portfolio.data_collector.binance_klines")
    def test_dispatches_to_binance_spot(self, mock_bk):
        """Source with 'binance' key dispatches to binance_klines."""
        mock_bk.return_value = pd.DataFrame()
        _ss._binance_limiter.last_call = 0.0  # avoid rate limit wait

        _fetch_klines({"binance": "BTCUSDT"}, "15m", 100)

        mock_bk.assert_called_once_with("BTCUSDT", interval="15m", limit=100)

    @patch("portfolio.data_collector.binance_fapi_klines")
    def test_dispatches_to_binance_fapi(self, mock_fapi):
        """Source with 'binance_fapi' key dispatches to binance_fapi_klines."""
        mock_fapi.return_value = pd.DataFrame()
        _ss._binance_limiter.last_call = 0.0

        _fetch_klines({"binance_fapi": "XAUUSDT"}, "1h", 100)

        mock_fapi.assert_called_once_with("XAUUSDT", interval="1h", limit=100)

    @patch("portfolio.data_collector.alpaca_klines")
    def test_dispatches_to_alpaca_when_market_open(self, mock_ak):
        """Source with 'alpaca' key dispatches to alpaca_klines when market is open."""
        mock_ak.return_value = pd.DataFrame()
        original_state = _ss._current_market_state
        _ss._current_market_state = "open"
        _ss._alpaca_limiter.last_call = 0.0

        try:
            _fetch_klines({"alpaca": "NVDA"}, "1d", 100)
        finally:
            _ss._current_market_state = original_state

        mock_ak.assert_called_once_with("NVDA", interval="1d", limit=100)

    @patch("portfolio.data_collector.yfinance_klines")
    def test_dispatches_to_yfinance_when_market_closed(self, mock_yfk):
        """Source with 'alpaca' key falls back to yfinance when market is closed."""
        mock_yfk.return_value = pd.DataFrame()
        original_state = _ss._current_market_state
        _ss._current_market_state = "closed"
        _ss._yfinance_limiter.last_call = 0.0

        try:
            _fetch_klines({"alpaca": "NVDA"}, "1d", 100)
        finally:
            _ss._current_market_state = original_state

        mock_yfk.assert_called_once_with("NVDA", interval="1d", limit=100)

    @patch("portfolio.data_collector.yfinance_klines")
    def test_dispatches_to_yfinance_when_weekend(self, mock_yfk):
        """Source with 'alpaca' key falls back to yfinance on weekends."""
        mock_yfk.return_value = pd.DataFrame()
        original_state = _ss._current_market_state
        _ss._current_market_state = "weekend"
        _ss._yfinance_limiter.last_call = 0.0

        try:
            _fetch_klines({"alpaca": "NVDA"}, "1d", 100)
        finally:
            _ss._current_market_state = original_state

        mock_yfk.assert_called_once_with("NVDA", interval="1d", limit=100)

    def test_unknown_source_raises_value_error(self):
        """Unknown source dict raises ValueError."""
        with pytest.raises(ValueError, match="Unknown source"):
            _fetch_klines({"unknown": "TICKER"}, "1d", 100)

    @patch("portfolio.data_collector.binance_fapi_klines")
    @patch("portfolio.data_collector.binance_klines")
    def test_fapi_takes_precedence_over_spot(self, mock_bk, mock_fapi):
        """When source has both binance_fapi and binance, fapi is preferred."""
        mock_fapi.return_value = pd.DataFrame()
        _ss._binance_limiter.last_call = 0.0

        _fetch_klines({"binance_fapi": "XAUUSDT", "binance": "BTCUSDT"}, "1h", 100)

        mock_fapi.assert_called_once()
        mock_bk.assert_not_called()


# ===========================================================================
# collect_timeframes
# ===========================================================================

class TestCollectTimeframes:

    @patch("portfolio.data_collector._fetch_klines")
    @patch("portfolio.data_collector.compute_indicators")
    @patch("portfolio.data_collector.technical_signal")
    def test_crypto_uses_crypto_timeframes(self, mock_ts, mock_ci, mock_fk):
        """Crypto source uses TIMEFRAMES (not STOCK_TIMEFRAMES)."""
        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
        mock_ts.return_value = ("HOLD", 0.5)

        results = collect_timeframes({"binance": "BTCUSDT"})

        # Should have entries for all crypto timeframes
        labels = [label for label, _ in results]
        expected_labels = [label for label, _, _, _ in TIMEFRAMES]
        assert labels == expected_labels

    @patch("portfolio.data_collector._fetch_klines")
    @patch("portfolio.data_collector.compute_indicators")
    @patch("portfolio.data_collector.technical_signal")
    def test_stock_uses_stock_timeframes(self, mock_ts, mock_ci, mock_fk):
        """Alpaca source uses STOCK_TIMEFRAMES."""
        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
        mock_ts.return_value = ("HOLD", 0.5)

        results = collect_timeframes({"alpaca": "NVDA"})

        labels = [label for label, _ in results]
        expected_labels = [label for label, _, _, _ in STOCK_TIMEFRAMES]
        assert labels == expected_labels

    @patch("portfolio.data_collector._fetch_klines")
    @patch("portfolio.data_collector.compute_indicators")
    @patch("portfolio.data_collector.technical_signal")
    def test_now_timeframe_skips_technical_signal(self, mock_ts, mock_ci, mock_fk):
        """The 'Now' timeframe does not call technical_signal (action/conf are None)."""
        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
        mock_ts.return_value = ("BUY", 0.8)

        results = collect_timeframes({"binance": "BTCUSDT"})

        now_entry = None
        for label, data in results:
            if label == "Now":
                now_entry = data
                break

        assert now_entry is not None
        assert now_entry["action"] is None
        assert now_entry["confidence"] is None
        assert "_df" in now_entry  # raw DataFrame preserved for enhanced signals

    @patch("portfolio.data_collector._fetch_klines")
    @patch("portfolio.data_collector.compute_indicators")
    @patch("portfolio.data_collector.technical_signal")
    def test_non_now_timeframes_call_technical_signal(self, mock_ts, mock_ci, mock_fk):
        """Non-Now timeframes call technical_signal and include action/confidence."""
        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
        mock_ts.return_value = ("BUY", 0.75)

        results = collect_timeframes({"binance": "BTCUSDT"})

        non_now_entries = [(label, data) for label, data in results if label != "Now"]
        assert len(non_now_entries) > 0
        for label, data in non_now_entries:
            if "error" not in data:
                assert data["action"] == "BUY"
                assert data["confidence"] == 0.75

    @patch("portfolio.data_collector._fetch_klines")
    @patch("portfolio.data_collector.compute_indicators")
    def test_insufficient_data_skips_timeframe(self, mock_ci, mock_fk):
        """When compute_indicators returns None (insufficient data), the timeframe is skipped."""
        mock_fk.return_value = pd.DataFrame({"close": [1.0, 2.0]})  # too few rows
        mock_ci.return_value = None  # insufficient data

        results = collect_timeframes({"binance": "BTCUSDT"})

        # All timeframes should be skipped (none in results)
        assert len(results) == 0

    @patch("portfolio.data_collector._fetch_klines")
    @patch("portfolio.data_collector.compute_indicators")
    @patch("portfolio.data_collector.technical_signal")
    def test_cache_hit_returns_cached_data(self, mock_ts, mock_ci, mock_fk):
        """Cached timeframes are returned without calling _fetch_klines again."""
        # Pre-populate cache for "12h" timeframe (TTL=300s)
        cache_key = "tf_BTCUSDT_12h"
        cached_entry = {"indicators": {"rsi": 42.0}, "action": "SELL", "confidence": 0.6}
        _ss._tool_cache[cache_key] = {"data": cached_entry, "time": time.time()}

        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
        mock_ts.return_value = ("HOLD", 0.5)

        results = collect_timeframes({"binance": "BTCUSDT"})

        # Find the 12h entry — it should be the cached version
        for label, data in results:
            if label == "12h":
                assert data["indicators"]["rsi"] == 42.0
                assert data["action"] == "SELL"
                break

    @patch("portfolio.data_collector._fetch_klines")
    @patch("portfolio.data_collector.compute_indicators")
    @patch("portfolio.data_collector.technical_signal")
    def test_expired_cache_refetches(self, mock_ts, mock_ci, mock_fk):
        """Expired cache entries trigger a fresh fetch."""
        cache_key = "tf_BTCUSDT_12h"
        cached_entry = {"indicators": {"rsi": 42.0}, "action": "SELL", "confidence": 0.6}
        # Expired cache (time well in the past)
        _ss._tool_cache[cache_key] = {"data": cached_entry, "time": time.time() - 9999}

        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
        mock_ci.return_value = {"rsi": 55.0, "close": 100.0}
        mock_ts.return_value = ("BUY", 0.8)

        results = collect_timeframes({"binance": "BTCUSDT"})

        # 12h entry should have the new data, not the stale cached data
        for label, data in results:
            if label == "12h":
                assert data["indicators"]["rsi"] == 55.0
                assert data["action"] == "BUY"
                break

    @patch("portfolio.data_collector._fetch_klines")
    def test_fetch_error_appends_error_entry(self, mock_fk):
        """When _fetch_klines raises, the timeframe gets an error entry."""
        mock_fk.side_effect = ConnectionError("API down")

        results = collect_timeframes({"binance": "BTCUSDT"})

        # All entries should have error
        assert len(results) == len(TIMEFRAMES)
        for label, data in results:
            assert "error" in data
            assert "API down" in data["error"]

    @patch("portfolio.data_collector._fetch_klines")
    @patch("portfolio.data_collector.compute_indicators")
    @patch("portfolio.data_collector.technical_signal")
    def test_now_timeframe_never_cached(self, mock_ts, mock_ci, mock_fk):
        """The 'Now' timeframe (TTL=0) is never cached — always fetches fresh."""
        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
        mock_ts.return_value = ("HOLD", 0.5)

        # Call twice
        collect_timeframes({"binance": "BTCUSDT"})
        collect_timeframes({"binance": "BTCUSDT"})

        # _fetch_klines should be called for "Now" both times.
        # Non-Now timeframes may be cached on second call.
        # Count: first call = 7 TFs, second call = at least 1 (Now) + maybe cached others
        assert mock_fk.call_count >= len(TIMEFRAMES) + 1

    @patch("portfolio.data_collector._fetch_klines")
    @patch("portfolio.data_collector.compute_indicators")
    @patch("portfolio.data_collector.technical_signal")
    def test_cache_written_for_non_now_timeframes(self, mock_ts, mock_ci, mock_fk):
        """Non-Now timeframes with TTL > 0 are written to the cache."""
        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
        mock_ts.return_value = ("HOLD", 0.5)

        collect_timeframes({"binance": "BTCUSDT"})

        # Check that 12h was cached
        assert "tf_BTCUSDT_12h" in _ss._tool_cache
        assert "tf_BTCUSDT_2d" in _ss._tool_cache

    @patch("portfolio.data_collector._fetch_klines")
    @patch("portfolio.data_collector.compute_indicators")
    @patch("portfolio.data_collector.technical_signal")
    def test_fapi_source_key_in_cache(self, mock_ts, mock_ci, mock_fk):
        """FAPI sources use the correct source_key in cache keys."""
        mock_fk.return_value = pd.DataFrame({"close": [100.0]})
        mock_ci.return_value = {"rsi": 50.0, "close": 100.0}
        mock_ts.return_value = ("HOLD", 0.5)

        collect_timeframes({"binance_fapi": "XAUUSDT"})

        assert "tf_XAUUSDT_12h" in _ss._tool_cache


# ===========================================================================
# Interval maps — static sanity checks
# ===========================================================================

class TestIntervalMaps:

    def test_alpaca_interval_map_has_required_intervals(self):
        """ALPACA_INTERVAL_MAP has all expected intervals."""
        for interval in ("15m", "1h", "1d", "1w", "1M"):
            assert interval in ALPACA_INTERVAL_MAP
            tf_name, lookback = ALPACA_INTERVAL_MAP[interval]
            assert isinstance(tf_name, str)
            assert lookback > 0

    def test_yf_interval_map_has_required_intervals(self):
        """_YF_INTERVAL_MAP has all expected intervals."""
        for interval in ("15m", "1h", "1d", "1w", "1M"):
            assert interval in _YF_INTERVAL_MAP
            yf_interval, yf_period = _YF_INTERVAL_MAP[interval]
            assert isinstance(yf_interval, str)
            assert isinstance(yf_period, str)

    def test_crypto_timeframes_have_seven_entries(self):
        """TIMEFRAMES has 7 entries (Now through 6mo)."""
        assert len(TIMEFRAMES) == 7
        labels = [t[0] for t in TIMEFRAMES]
        assert labels == ["Now", "12h", "2d", "7d", "1mo", "3mo", "6mo"]

    def test_stock_timeframes_have_seven_entries(self):
        """STOCK_TIMEFRAMES has 7 entries."""
        assert len(STOCK_TIMEFRAMES) == 7
        labels = [t[0] for t in STOCK_TIMEFRAMES]
        assert labels == ["Now", "12h", "2d", "7d", "1mo", "3mo", "6mo"]

    def test_now_timeframe_has_zero_ttl(self):
        """The 'Now' timeframe has TTL=0 (never cached)."""
        for tfs in (TIMEFRAMES, STOCK_TIMEFRAMES):
            now_tf = [t for t in tfs if t[0] == "Now"][0]
            assert now_tf[3] == 0  # TTL is the 4th element

    def test_binance_kline_cols_has_twelve_elements(self):
        """_BINANCE_KLINE_COLS has exactly 12 elements matching Binance API."""
        assert len(_BINANCE_KLINE_COLS) == 12
        assert "open" in _BINANCE_KLINE_COLS
        assert "close" in _BINANCE_KLINE_COLS
        assert "volume" in _BINANCE_KLINE_COLS
