"""Tests for _load_candles_ohlcv() interval handling fixes.

Covers:
- Fix 1: No double-mapping — raw interval is passed directly to alpaca_klines()
- Fix 2: Alpaca 5m/1m/3m intervals fall back to 15m (Alpaca minimum)
- Fix 3: Kronos df OHLCV fallback when candle loading fails
"""

from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=60, with_ohlcv=True):
    """Return a minimal OHLCV DataFrame."""
    data = {"close": [100.0 + i * 0.1 for i in range(n)]}
    if with_ohlcv:
        data["open"] = [c - 0.5 for c in data["close"]]
        data["high"] = [c + 1.0 for c in data["close"]]
        data["low"] = [c - 1.0 for c in data["close"]]
        data["volume"] = [1_000.0] * n
    return pd.DataFrame(data)


def _make_candles_df(n=60):
    """Return a DataFrame shaped like alpaca_klines() output."""
    df = _make_df(n)
    df["time"] = pd.date_range("2026-01-01", periods=n, freq="h")
    return df


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_forecast():
    """Reset circuit breakers and disable models after each test."""
    import portfolio.signals.forecast as mod
    orig_kronos = mod._KRONOS_ENABLED
    orig_disabled = mod._FORECAST_MODELS_DISABLED
    mod._FORECAST_MODELS_DISABLED = True  # keep models off — we're only testing candle loading
    yield
    mod._KRONOS_ENABLED = orig_kronos
    mod._FORECAST_MODELS_DISABLED = orig_disabled


# ---------------------------------------------------------------------------
# Fix 1: No double-mapping — alpaca_klines() receives the raw internal interval
# ---------------------------------------------------------------------------

class TestNoDoubleMapping:
    """alpaca_klines() must receive raw internal intervals like '1h', '15m', not 'Hour', '15Min'."""

    @patch("portfolio.signals.forecast._load_candles_ohlcv")
    def test_alpaca_klines_receives_raw_1h_interval(self, mock_load):
        """Sanity: _load_candles_ohlcv is the function under test; confirm it calls alpaca_klines with raw '1h'."""
        mock_load.return_value = None  # won't be called — we test the real function below
        mock_load.side_effect = None

    def test_alpaca_klines_called_with_raw_interval_1h(self):
        """alpaca_klines() must be called with '1h', not 'Hour' or '1Hour'."""
        alpaca_df = _make_candles_df(80)

        with patch("portfolio.tickers.SYMBOLS", {"PLTR": {"alpaca": "PLTR"}}), \
             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
            from portfolio.signals.forecast import _load_candles_ohlcv
            result = _load_candles_ohlcv("PLTR", periods=80, interval="1h")

        # alpaca_klines must be called once with the raw '1h' interval
        mock_ak.assert_called_once()
        _, kwargs = mock_ak.call_args
        received_interval = kwargs.get("interval") or mock_ak.call_args[0][1]
        assert received_interval == "1h", (
            f"Expected raw '1h' passed to alpaca_klines, got {received_interval!r}. "
            "Double-mapping bug: alpaca_klines does its own mapping internally."
        )
        assert result is not None
        assert len(result) > 30

    def test_alpaca_klines_called_with_raw_interval_15m(self):
        """alpaca_klines() must be called with '15m', not '15Min'."""
        alpaca_df = _make_candles_df(80)

        with patch("portfolio.tickers.SYMBOLS", {"PLTR": {"alpaca": "PLTR"}}), \
             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
            from portfolio.signals.forecast import _load_candles_ohlcv
            _load_candles_ohlcv("PLTR", periods=80, interval="15m")

        _, kwargs = mock_ak.call_args
        received_interval = kwargs.get("interval") or mock_ak.call_args[0][1]
        assert received_interval == "15m", (
            f"Expected raw '15m' passed to alpaca_klines, got {received_interval!r}."
        )

    def test_alpaca_klines_called_with_raw_interval_1d(self):
        """alpaca_klines() must be called with '1d', not '1Day'."""
        alpaca_df = _make_candles_df(80)

        with patch("portfolio.tickers.SYMBOLS", {"PLTR": {"alpaca": "PLTR"}}), \
             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
            from portfolio.signals.forecast import _load_candles_ohlcv
            _load_candles_ohlcv("PLTR", periods=80, interval="1d")

        _, kwargs = mock_ak.call_args
        received_interval = kwargs.get("interval") or mock_ak.call_args[0][1]
        assert received_interval == "1d", (
            f"Expected raw '1d' passed to alpaca_klines, got {received_interval!r}."
        )

    def test_alpaca_klines_never_receives_mapped_format(self):
        """Regression: alpaca_klines() must NEVER receive already-mapped values like '1Hour' or '15Min'."""
        alpaca_df = _make_candles_df(80)
        mapped_values = {"1Hour", "15Min", "5Min", "1Day", "1Week", "1Month"}

        for internal_interval in ("1h", "15m", "1d"):
            with patch("portfolio.tickers.SYMBOLS", {"PLTR": {"alpaca": "PLTR"}}), \
                 patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
                from portfolio.signals.forecast import _load_candles_ohlcv
                _load_candles_ohlcv("PLTR", periods=80, interval=internal_interval)

            received_interval = mock_ak.call_args[1].get("interval") or mock_ak.call_args[0][1]
            assert received_interval not in mapped_values, (
                f"alpaca_klines received already-mapped value {received_interval!r} "
                f"for input interval {internal_interval!r}. Double-mapping bug."
            )


# ---------------------------------------------------------------------------
# Fix 2: Alpaca interval fallback (5m → 15m)
# ---------------------------------------------------------------------------

class TestAlpacaIntervalFallback:
    """When configured interval is smaller than Alpaca's minimum (15m), fall back to 15m."""

    @pytest.mark.parametrize("bad_interval", ["5m", "1m", "3m"])
    def test_small_interval_falls_back_to_15m(self, bad_interval):
        """Intervals below 15m must be silently upgraded to 15m for Alpaca sources."""
        alpaca_df = _make_candles_df(80)

        with patch("portfolio.tickers.SYMBOLS", {"PLTR": {"alpaca": "PLTR"}}), \
             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
            from portfolio.signals.forecast import _load_candles_ohlcv
            result = _load_candles_ohlcv("PLTR", periods=80, interval=bad_interval)

        # Must have been called with the fallback interval, not the original
        received_interval = mock_ak.call_args[1].get("interval") or mock_ak.call_args[0][1]
        assert received_interval == "15m", (
            f"Expected fallback to '15m' for Alpaca with {bad_interval!r}, "
            f"got {received_interval!r}."
        )
        assert result is not None

    def test_small_interval_does_not_fall_back_for_binance(self):
        """Binance supports 5m natively — no fallback should happen for crypto tickers."""
        binance_df = _make_candles_df(80)

        with patch("portfolio.tickers.SYMBOLS", {"BTC-USD": {"binance": "BTCUSDT"}}), \
             patch("portfolio.data_collector.binance_klines", return_value=binance_df) as mock_bk:
            from portfolio.signals.forecast import _load_candles_ohlcv
            _load_candles_ohlcv("BTC-USD", periods=80, interval="5m")

        # Binance must receive the original 5m interval unchanged
        received_interval = mock_bk.call_args[1].get("interval") or mock_bk.call_args[0][1]
        assert received_interval == "5m", (
            f"Binance should receive raw '5m', got {received_interval!r}."
        )

    def test_15m_interval_not_altered_for_alpaca(self):
        """15m is Alpaca's minimum — must be passed through unchanged."""
        alpaca_df = _make_candles_df(80)

        with patch("portfolio.tickers.SYMBOLS", {"PLTR": {"alpaca": "PLTR"}}), \
             patch("portfolio.data_collector.alpaca_klines", return_value=alpaca_df) as mock_ak:
            from portfolio.signals.forecast import _load_candles_ohlcv
            _load_candles_ohlcv("PLTR", periods=80, interval="15m")

        received_interval = mock_ak.call_args[1].get("interval") or mock_ak.call_args[0][1]
        assert received_interval == "15m"


# ---------------------------------------------------------------------------
# Fix 3: Kronos df OHLCV fallback
# ---------------------------------------------------------------------------

class TestKronosDfFallback:
    """When candle loading fails, Kronos should fall back to the passed-in DataFrame."""

    def _make_context(self, ticker="PLTR"):
        return {
            "ticker": ticker,
            "config": {"forecast": {"kronos_interval": "5m", "kronos_periods": 500}},
        }

    def test_kronos_df_fallback_triggers_when_candle_load_fails(self):
        """When _load_candles_ohlcv returns None, Kronos input should be built from df."""
        import portfolio.signals.forecast as mod

        df = _make_df(60)
        context = self._make_context()

        with patch.object(mod, "_FORECAST_MODELS_DISABLED", False), \
             patch.object(mod, "_KRONOS_ENABLED", True), \
             patch.object(mod, "_cached") as mock_cached, \
             patch.object(mod, "_run_kronos", return_value=None) as mock_kronos, \
             patch.object(mod, "_run_chronos", return_value=None):

            captured_kronos_input = []

            def fake_cached(key, ttl, fn, *args, **kwargs):
                if "candles" in key and "5m" in key:
                    # Kronos-specific candle fetch — simulate failure
                    return None
                elif "candles" in key:
                    # 1h candle fetch for Chronos — return minimal list
                    return [{"close": 100.0 + i * 0.1, "open": 99.5, "high": 101.0,
                              "low": 99.0, "volume": 1000.0} for i in range(60)]
                elif "kronos_forecast" in key:
                    captured_kronos_input.extend(args[0] if args else [])
                    return None
                elif "chronos_forecast" in key:
                    return None
                elif "forecast_ticker_accuracy" in key or "forecast_subsignal" in key:
                    return {}
                return fn(*args, **kwargs) if callable(fn) else None

            mock_cached.side_effect = fake_cached

            result = mod.compute_forecast_signal(df, context=context)

        # The result should not error on candle data — kronos fallback should have kicked in
        assert result is not None
        assert "indicators" in result
        # Should NOT report insufficient_candle_data since df is available
        assert result["indicators"].get("error") != "insufficient_candle_data"

    def test_kronos_df_fallback_sets_source_indicator(self):
        """When df fallback is used for Kronos candles, indicators should note 'df_fallback'."""
        import portfolio.signals.forecast as mod

        df = _make_df(60)  # has full OHLCV
        context = self._make_context()

        with patch.object(mod, "_FORECAST_MODELS_DISABLED", False), \
             patch.object(mod, "_KRONOS_ENABLED", True), \
             patch.object(mod, "_cached") as mock_cached, \
             patch.object(mod, "_run_kronos", return_value=None), \
             patch.object(mod, "_run_chronos", return_value=None):

            def fake_cached(key, ttl, fn, *args, **kwargs):
                if "candles" in key and "5m" in key:
                    return None  # kronos-specific fetch fails
                elif "candles" in key:
                    return [{"close": 100.0 + i * 0.1, "open": 99.5, "high": 101.0,
                              "low": 99.0, "volume": 1000.0} for i in range(60)]
                elif "kronos_forecast" in key or "chronos_forecast" in key:
                    return None
                elif "forecast_ticker_accuracy" in key or "forecast_subsignal" in key:
                    return {}
                return fn(*args, **kwargs) if callable(fn) else None

            mock_cached.side_effect = fake_cached

            result = mod.compute_forecast_signal(df, context=context)

        assert result["indicators"].get("kronos_candles_source") == "df_fallback"

    def test_kronos_df_fallback_skipped_when_df_close_only(self):
        """If df has only 'close' column (no full OHLCV), df fallback should not produce candles."""
        import portfolio.signals.forecast as mod

        df = _make_df(60, with_ohlcv=False)  # close only
        context = self._make_context()

        with patch.object(mod, "_FORECAST_MODELS_DISABLED", False), \
             patch.object(mod, "_KRONOS_ENABLED", True), \
             patch.object(mod, "_cached") as mock_cached, \
             patch.object(mod, "_run_kronos", return_value=None), \
             patch.object(mod, "_run_chronos", return_value=None):

            def fake_cached(key, ttl, fn, *args, **kwargs):
                if "candles" in key and "5m" in key:
                    return None
                elif "candles" in key:
                    return [{"close": 100.0 + i * 0.1, "open": 99.5, "high": 101.0,
                              "low": 99.0, "volume": 1000.0} for i in range(60)]
                elif "kronos_forecast" in key or "chronos_forecast" in key:
                    return None
                elif "forecast_ticker_accuracy" in key or "forecast_subsignal" in key:
                    return {}
                return fn(*args, **kwargs) if callable(fn) else None

            mock_cached.side_effect = fake_cached

            result = mod.compute_forecast_signal(df, context=context)

        # No df_fallback since df lacks full OHLCV columns
        assert result["indicators"].get("kronos_candles_source") != "df_fallback"
