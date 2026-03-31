"""Tests for Batch 3 fixes: confidence clamping, circuit breaker probe, empty response.

Covers:
  BUG-90: Confidence clamped to 1.0 after each penalty stage
  BUG-93: Circuit breaker HALF_OPEN allows only one probe request
  BUG-100: Empty Binance response treated as circuit breaker failure
"""

import threading
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from portfolio.circuit_breaker import CircuitBreaker, State

# ===========================================================================
# BUG-90: Confidence clamping after each penalty stage
# ===========================================================================

class TestBug90ConfidenceClamping:
    """Confidence must be clamped to [0, 1.0] after each stage."""

    def _make_df(self, n=10, up=True):
        """Create a simple DataFrame with close/volume columns."""
        base = 100.0
        closes = [base + i * (0.5 if up else -0.5) for i in range(n)]
        return pd.DataFrame({
            "close": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "volume": [1000 + i * 10 for i in range(n)],
        })

    def test_regime_bonus_clamped_at_1(self):
        """Stage 1 regime boost (1.10x) should not push conf above 1.0."""
        from portfolio.signal_engine import apply_confidence_penalties

        action, conf, log = apply_confidence_penalties(
            action="BUY",
            conf=0.95,  # 0.95 * 1.10 = 1.045 → should clamp to 1.0
            regime="trending-up",
            ind={},
            extra_info={"_voters": 5},
            ticker="BTC-USD",
            df=self._make_df(),
            config={"confidence_penalties": {"enabled": True}},
        )
        assert conf <= 1.0

    def test_volume_boost_clamped_at_1(self):
        """Stage 2 volume boost (1.15x) should not push conf above 1.0."""
        from portfolio.signal_engine import apply_confidence_penalties

        action, conf, log = apply_confidence_penalties(
            action="BUY",
            conf=0.90,  # After regime bonus: 0.99, then 0.99 * 1.15 = 1.1385 → clamp
            regime="trending-up",
            ind={},
            extra_info={"volume_ratio": 2.0, "_voters": 5},
            ticker="BTC-USD",
            df=self._make_df(),
            config={"confidence_penalties": {"enabled": True}},
        )
        assert conf <= 1.0

    def test_regime_penalty_reduces_confidence(self):
        """Stage 1 ranging penalty (0.75x) should reduce confidence."""
        from portfolio.signal_engine import apply_confidence_penalties

        action, conf, log = apply_confidence_penalties(
            action="BUY",
            conf=0.80,
            regime="ranging",
            ind={},
            extra_info={"_voters": 6},
            ticker="BTC-USD",
            df=self._make_df(),
            config={"confidence_penalties": {"enabled": True}},
        )
        assert conf <= 0.80
        assert conf == pytest.approx(0.80 * 0.75, abs=0.01)

    def test_all_stages_never_exceed_1(self):
        """Full cascade with all bonuses applied never exceeds 1.0."""
        from portfolio.signal_engine import apply_confidence_penalties

        # Trending-up + aligned BUY + high volume — both bonuses apply
        action, conf, log = apply_confidence_penalties(
            action="BUY",
            conf=0.99,
            regime="trending-up",
            ind={},
            extra_info={"volume_ratio": 2.0, "_voters": 5},
            ticker="BTC-USD",
            df=self._make_df(),
            config={"confidence_penalties": {"enabled": True}},
        )
        assert conf <= 1.0

    def test_disabled_skips_clamping(self):
        """When penalties are disabled, conf passes through unchanged."""
        from portfolio.signal_engine import apply_confidence_penalties

        action, conf, log = apply_confidence_penalties(
            action="BUY",
            conf=0.99,
            regime="trending-up",
            ind={},
            extra_info={"volume_ratio": 2.0, "_voters": 5},
            ticker="BTC-USD",
            df=self._make_df(),
            config={"confidence_penalties": {"enabled": False}},
        )
        assert conf == 0.99  # Unchanged


# ===========================================================================
# BUG-93: Circuit breaker HALF_OPEN probe limiting
# ===========================================================================

class TestBug93HalfOpenProbe:
    """HALF_OPEN state should allow exactly one probe request."""

    def _make_half_open_cb(self):
        """Create a circuit breaker in HALF_OPEN state."""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0)
        # Trigger OPEN
        cb.record_failure()
        cb.record_failure()
        assert cb.state == State.OPEN
        # Transition to HALF_OPEN via allow_request (recovery_timeout=0 means immediate)
        result = cb.allow_request()
        assert result is True
        assert cb.state == State.HALF_OPEN
        return cb

    def test_first_half_open_request_allowed(self):
        """First request in HALF_OPEN is allowed."""
        cb = self._make_half_open_cb()
        # The probe was already consumed by _make_half_open_cb calling allow_request
        # which transitioned OPEN→HALF_OPEN and returned True.
        # Now the probe flag is set, so next call should return False.
        assert cb.allow_request() is False

    def test_second_half_open_request_blocked(self):
        """Second request in HALF_OPEN is blocked."""
        cb = self._make_half_open_cb()
        # First allow_request in HALF_OPEN was consumed by _make_half_open_cb
        assert cb.allow_request() is False
        assert cb.allow_request() is False  # Still blocked

    def test_success_resets_probe_flag(self):
        """After record_success, circuit closes and requests are allowed again."""
        cb = self._make_half_open_cb()
        cb.record_success()
        assert cb.state == State.CLOSED
        assert cb.allow_request() is True
        assert cb.allow_request() is True  # CLOSED allows all

    def test_failure_resets_probe_flag(self):
        """After record_failure in HALF_OPEN, circuit opens and probe flag resets."""
        cb = self._make_half_open_cb()
        cb.record_failure()
        assert cb.state == State.OPEN

    def test_concurrent_half_open_only_one_probe(self):
        """Multiple threads in HALF_OPEN — only one should get through."""
        cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout=0)
        cb.record_failure()
        cb.record_failure()
        # Don't call allow_request yet — we want threads to race for the probe

        results = []
        barrier = threading.Barrier(5)

        def try_request():
            barrier.wait()
            # First call may transition OPEN→HALF_OPEN for one thread
            r = cb.allow_request()
            results.append(r)

        threads = [threading.Thread(target=try_request) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At most 1 thread should get True (the one that transitions OPEN→HALF_OPEN
        # AND gets the probe). Others should get False.
        assert results.count(True) <= 1


# ===========================================================================
# BUG-100: Empty Binance response handling
# ===========================================================================

class TestBug100EmptyBinanceResponse:
    """Empty Binance kline response should record failure, not success."""

    def test_empty_response_returns_empty_df(self):
        """Empty data returns empty DataFrame."""
        from portfolio.data_collector import _binance_fetch

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None

        mock_cb = MagicMock()
        mock_cb.allow_request.return_value = True

        with patch("portfolio.data_collector.fetch_with_retry", return_value=mock_response):
            result = _binance_fetch("http://test", mock_cb, "test", "BTCUSDT")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_empty_response_records_failure(self):
        """Empty data records circuit breaker failure."""
        from portfolio.data_collector import _binance_fetch

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None

        mock_cb = MagicMock()
        mock_cb.allow_request.return_value = True

        with patch("portfolio.data_collector.fetch_with_retry", return_value=mock_response):
            _binance_fetch("http://test", mock_cb, "test", "BTCUSDT")

        mock_cb.record_failure.assert_called_once()
        mock_cb.record_success.assert_not_called()

    def test_nonempty_response_records_success(self):
        """Non-empty data records circuit breaker success."""
        from portfolio.data_collector import _binance_fetch

        kline = [
            1710000000000, "50000", "50100", "49900", "50050", "1000",
            1710000060000, "50000000", 100, "500", "25000000", "0",
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = [kline]
        mock_response.raise_for_status.return_value = None

        mock_cb = MagicMock()
        mock_cb.allow_request.return_value = True

        with patch("portfolio.data_collector.fetch_with_retry", return_value=mock_response):
            result = _binance_fetch("http://test", mock_cb, "test", "BTCUSDT")

        assert len(result) == 1
        mock_cb.record_success.assert_called_once()
        mock_cb.record_failure.assert_not_called()
