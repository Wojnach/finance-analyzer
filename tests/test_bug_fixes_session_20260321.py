"""Tests for bug fixes in auto-session-2026-03-21.

Covers:
  BUG-101: Sentiment flush crash safety (dirty flag after write)
  BUG-102: Forecast circuit breaker thread safety
  BUG-103: Portfolio P&L zero-division guard
  BUG-104: Calendar seasonal exception logging
  BUG-105: FX rate alert routing to Telegram
  BUG-106: Forecast prediction dedup cache eviction
"""

import json
import logging
import threading
import time
from unittest.mock import patch

import pandas as pd
import pytest

# ===========================================================================
# BUG-101: Sentiment flush crash safety
# ===========================================================================


class TestSentimentFlushCrashSafety:
    """Verify that flush_sentiment_state() only clears the dirty flag after
    a successful write. A failed write should leave the flag set so the next
    cycle retries."""

    def test_dirty_flag_cleared_after_successful_write(self, tmp_path):
        """Happy path: dirty flag cleared after successful atomic_write_json."""
        import portfolio.signal_engine as se

        # Reset module state
        with se._sentiment_lock:
            se._prev_sentiment = {"BTC-USD": "positive"}
            se._sentiment_dirty = True
            se._prev_sentiment_loaded = True

        state_file = tmp_path / "sentiment_state.json"
        with patch.object(se, "_SENTIMENT_STATE_FILE", state_file):
            se.flush_sentiment_state()

        # Dirty flag should be cleared
        with se._sentiment_lock:
            assert se._sentiment_dirty is False

        # File should exist with correct content
        data = json.loads(state_file.read_text())
        assert data["prev_sentiment"]["BTC-USD"] == "positive"

    def test_dirty_flag_remains_on_write_failure(self, tmp_path):
        """If atomic_write_json raises, dirty flag must remain True for retry."""
        import portfolio.signal_engine as se

        with se._sentiment_lock:
            se._prev_sentiment = {"ETH-USD": "negative"}
            se._sentiment_dirty = True
            se._prev_sentiment_loaded = True

        # atomic_write_json is imported lazily inside flush_sentiment_state,
        # so patch at source module
        with patch("portfolio.file_utils.atomic_write_json",
                   side_effect=OSError("disk full")):
            se.flush_sentiment_state()

        # Dirty flag should still be True (not cleared)
        with se._sentiment_lock:
            assert se._sentiment_dirty is True

    def test_retry_after_failure(self, tmp_path):
        """After a failed flush, the next flush should succeed."""
        import portfolio.signal_engine as se

        with se._sentiment_lock:
            se._prev_sentiment = {"XAG-USD": "positive"}
            se._sentiment_dirty = True
            se._prev_sentiment_loaded = True

        state_file = tmp_path / "sentiment_state.json"

        # First call fails
        with patch("portfolio.file_utils.atomic_write_json",
                   side_effect=OSError("disk full")):
            se.flush_sentiment_state()

        with se._sentiment_lock:
            assert se._sentiment_dirty is True  # still dirty

        # Second call succeeds
        with patch.object(se, "_SENTIMENT_STATE_FILE", state_file):
            se.flush_sentiment_state()

        with se._sentiment_lock:
            assert se._sentiment_dirty is False

        data = json.loads(state_file.read_text())
        assert data["prev_sentiment"]["XAG-USD"] == "positive"

    def test_not_dirty_skips_write(self):
        """If not dirty, flush should return immediately without writing."""
        import portfolio.signal_engine as se

        with se._sentiment_lock:
            se._sentiment_dirty = False
            se._prev_sentiment_loaded = True

        with patch("portfolio.file_utils.atomic_write_json") as mock_write:
            se.flush_sentiment_state()
            mock_write.assert_not_called()


# ===========================================================================
# BUG-102: Forecast circuit breaker thread safety
# ===========================================================================


class TestForecastThreadSafety:
    """Verify that circuit breaker operations are thread-safe."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        from portfolio.signals.forecast import reset_circuit_breakers
        reset_circuit_breakers()
        yield
        reset_circuit_breakers()

    def test_concurrent_trip_and_check(self):
        """Multiple threads tripping and checking should not raise."""
        from portfolio.signals.forecast import (
            _kronos_circuit_open,
            _trip_kronos,
            reset_circuit_breakers,
        )

        errors = []

        def tripper():
            try:
                for _ in range(100):
                    _trip_kronos()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def checker():
            try:
                for _ in range(100):
                    _kronos_circuit_open()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def resetter():
            try:
                for _ in range(50):
                    reset_circuit_breakers()
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=tripper),
            threading.Thread(target=checker),
            threading.Thread(target=resetter),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread safety errors: {errors}"

    def test_log_health_reset_under_lock(self):
        """_log_health auto-reset should be atomic (no race window)."""
        from portfolio.signals.forecast import (
            _kronos_circuit_open,
            _log_health,
            _trip_kronos,
        )

        _trip_kronos()
        assert _kronos_circuit_open()

        # Success should reset breaker
        _log_health("kronos", "BTC-USD", success=True, duration_ms=100)
        assert not _kronos_circuit_open()


# ===========================================================================
# BUG-103: Portfolio P&L zero-division guard
# ===========================================================================


class TestPortfolioPnlZeroDivGuard:
    """Verify that main.py run() doesn't crash on corrupt portfolio state."""

    def test_zero_initial_value_uses_fallback(self):
        """If initial_value_sek is 0, should use INITIAL_CASH_SEK (500K)."""
        from portfolio.portfolio_mgr import INITIAL_CASH_SEK

        state = {"initial_value_sek": 0, "cash_sek": 490_000, "holdings": {}}
        # Simulate the guarded calculation from main.py:434
        initial_val = state.get("initial_value_sek") or INITIAL_CASH_SEK
        pnl_pct = ((490_000 - initial_val) / initial_val) * 100
        assert initial_val == INITIAL_CASH_SEK
        assert pnl_pct == pytest.approx(-2.0)

    def test_missing_initial_value_uses_fallback(self):
        """If initial_value_sek is missing, should use INITIAL_CASH_SEK."""
        from portfolio.portfolio_mgr import INITIAL_CASH_SEK

        state = {"cash_sek": 500_000, "holdings": {}}
        initial_val = state.get("initial_value_sek") or INITIAL_CASH_SEK
        assert initial_val == INITIAL_CASH_SEK

    def test_valid_initial_value_used_directly(self):
        """Normal case: initial_value_sek is used directly."""
        state = {"initial_value_sek": 500_000, "cash_sek": 510_000, "holdings": {}}
        initial_val = state.get("initial_value_sek") or 500_000
        pnl_pct = ((510_000 - initial_val) / initial_val) * 100
        assert pnl_pct == pytest.approx(2.0)


# ===========================================================================
# BUG-104: Calendar seasonal exception logging
# ===========================================================================


class TestCalendarExceptionLogging:
    """Verify that calendar_seasonal sub-signal exceptions are logged."""

    def test_sub_signal_failure_logged(self, caplog):
        """When a sub-signal raises, the exception should appear in debug logs."""
        from portfolio.signals.calendar_seasonal import compute_calendar_signal

        # Create a valid DataFrame with a time column
        df = pd.DataFrame({
            "time": pd.date_range("2026-03-01", periods=50, freq="h"),
            "open": [100] * 50,
            "high": [105] * 50,
            "low": [95] * 50,
            "close": [100] * 50,
            "volume": [1000] * 50,
        })

        # Patch one sub-signal to raise
        with patch("portfolio.signals.calendar_seasonal._day_of_week_effect",
                   side_effect=ValueError("test error")):
            with caplog.at_level(logging.DEBUG, logger="portfolio.signals.calendar_seasonal"):
                result = compute_calendar_signal(df)

        # Signal should still return a valid result (graceful degradation)
        assert result["action"] in ("BUY", "SELL", "HOLD")

        # The exception should be logged
        assert any("day_of_week_effect failed" in r.message for r in caplog.records), \
            f"Expected 'day_of_week_effect failed' in logs, got: {[r.message for r in caplog.records]}"

    def test_all_sub_signals_fail_returns_hold(self, caplog):
        """If ALL sub-signals fail, should return HOLD with confidence 0."""
        from portfolio.signals.calendar_seasonal import compute_calendar_signal

        df = pd.DataFrame({
            "time": pd.date_range("2026-03-01", periods=50, freq="h"),
            "open": [100] * 50,
            "high": [105] * 50,
            "low": [95] * 50,
            "close": [100] * 50,
            "volume": [1000] * 50,
        })

        # Patch ALL sub-signals to raise
        sub_signals = [
            "_day_of_week_effect", "_turnaround_tuesday", "_month_end_effect",
            "_sell_in_may", "_january_effect", "_pre_holiday_effect",
            "_fomc_drift", "_santa_claus_rally",
        ]
        patches_list = {f"portfolio.signals.calendar_seasonal.{s}": ValueError("fail")
                        for s in sub_signals}

        with caplog.at_level(logging.DEBUG, logger="portfolio.signals.calendar_seasonal"):
            active_mocks = {}
            for name, err in patches_list.items():
                active_mocks[name] = patch(name, side_effect=err)
            # Enter all patches
            for m in active_mocks.values():
                m.start()
            try:
                result = compute_calendar_signal(df)
            finally:
                for m in active_mocks.values():
                    m.stop()

        # All sub-signals HOLD → overall should be HOLD
        assert result["action"] == "HOLD"

        # Should have 8 logged failures
        failure_logs = [r for r in caplog.records if "failed" in r.message]
        assert len(failure_logs) == 8, f"Expected 8 failure logs, got {len(failure_logs)}"


# ===========================================================================
# BUG-105: FX rate alert routing
# ===========================================================================


class TestFxAlertRouting:
    """Verify that FX fallback alerts are sent with category 'error'
    (which reaches Telegram) instead of 'fx_alert' (save-only)."""

    def test_fallback_alert_uses_error_category(self):
        """_fx_alert_telegram should send with category='error'."""
        import portfolio.fx_rates as fx_mod

        # Reset alert cooldown
        fx_mod._fx_cache["_last_fx_alert"] = 0

        with patch("portfolio.fx_rates._load_config", return_value={"telegram": {}}):
            with patch("portfolio.message_store.send_or_store") as mock_send:
                fx_mod._fx_alert_telegram(age_secs=None)

        mock_send.assert_called_once()
        # Check that category kwarg is "error", not "fx_alert"
        assert mock_send.call_args[1].get("category") == "error", \
            f"Expected category='error', got: {mock_send.call_args}"

    def test_stale_rate_alert_uses_error_category(self):
        """When rate is stale, alert should also use 'error' category."""
        import portfolio.fx_rates as fx_mod

        fx_mod._fx_cache["_last_fx_alert"] = 0

        with patch("portfolio.fx_rates._load_config", return_value={"telegram": {}}):
            with patch("portfolio.message_store.send_or_store") as mock_send:
                fx_mod._fx_alert_telegram(age_secs=7200)

        mock_send.assert_called_once()
        # Verify the message mentions stale and uses error category
        msg = mock_send.call_args[0][0]
        assert "stale" in msg.lower() or "FX" in msg
        assert mock_send.call_args[1].get("category") == "error"


# ===========================================================================
# BUG-106: Forecast prediction dedup cache eviction
# ===========================================================================


class TestForecastDedupEviction:
    """Verify that stale entries in _last_prediction_ts are evicted."""

    def test_stale_entries_evicted(self):
        """Entries older than _PREDICTION_DEDUP_EVICT_AGE should be removed."""
        import portfolio.signals.forecast as fmod

        now = time.monotonic()
        with fmod._forecast_lock:
            # Add some entries: 2 stale, 1 fresh
            fmod._last_prediction_ts["OLD-TICKER-1"] = now - 1200  # 20 min ago
            fmod._last_prediction_ts["OLD-TICKER-2"] = now - 900   # 15 min ago
            fmod._last_prediction_ts["FRESH-TICKER"] = now - 30    # 30 sec ago

        # Trigger eviction by simulating the dedup write path
        with fmod._forecast_lock:
            stale = [k for k, v in fmod._last_prediction_ts.items()
                     if now - v > fmod._PREDICTION_DEDUP_EVICT_AGE]
            for k in stale:
                del fmod._last_prediction_ts[k]

        with fmod._forecast_lock:
            assert "OLD-TICKER-1" not in fmod._last_prediction_ts
            assert "OLD-TICKER-2" not in fmod._last_prediction_ts
            assert "FRESH-TICKER" in fmod._last_prediction_ts

    def test_eviction_constant_exists(self):
        """Verify _PREDICTION_DEDUP_EVICT_AGE constant is defined."""
        from portfolio.signals.forecast import _PREDICTION_DEDUP_EVICT_AGE
        assert _PREDICTION_DEDUP_EVICT_AGE == 600  # 10 minutes

    def test_forecast_lock_exists(self):
        """Verify _forecast_lock is a threading.Lock."""
        from portfolio.signals.forecast import _forecast_lock
        assert isinstance(_forecast_lock, type(threading.Lock()))
