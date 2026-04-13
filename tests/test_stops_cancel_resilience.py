"""Tests for the 2026-04-13 Bug 2a stop-snapshot retry.

`_capture_stop_snapshot` now retries with `close_playwright()` between
attempts when the first call raises (typically Playwright "sync API
inside asyncio loop" or TargetClosedError). Today's BULL_SILVER_X5
position got stuck because the first attempt raised and the legacy
single-attempt path failed closed.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock


class TestCaptureStopSnapshotRetry:

    def test_first_call_succeeds(self):
        """No retry needed when the first call returns cleanly."""
        from data import metals_loop as ml

        sl_data = [
            {"orderbook": {"id": "1650161"}, "stop_id": "abc"},
            {"orderbook": {"id": "9999"}, "stop_id": "other"},
        ]
        with patch("portfolio.avanza_session.get_stop_losses_strict", return_value=sl_data) as mock_strict:
            ok, snap = ml._capture_stop_snapshot("1650161")
            assert ok is True
            assert len(snap) == 1
            assert snap[0]["stop_id"] == "abc"
            mock_strict.assert_called_once()

    def test_retry_recovers_from_first_exception(self):
        """First call raises → close_playwright + retry succeeds → returns ok=True."""
        from data import metals_loop as ml

        sl_data = [{"orderbook": {"id": "1650161"}, "stop_id": "abc"}]
        # Simulate raise-then-success
        call_count = {"n": 0}

        def get_strict_side_effect():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("Playwright sync API inside asyncio loop")
            return sl_data

        with patch("portfolio.avanza_session.get_stop_losses_strict", side_effect=get_strict_side_effect), \
             patch("portfolio.avanza_session.close_playwright") as mock_close, \
             patch("time.sleep"):  # speed up test
            ok, snap = ml._capture_stop_snapshot("1650161")
            assert ok is True
            assert len(snap) == 1
            mock_close.assert_called_once()
            assert call_count["n"] == 2

    def test_both_calls_fail_returns_false(self):
        """If retry also fails, return (False, []) — preserves existing fail-closed behavior."""
        from data import metals_loop as ml

        with patch("portfolio.avanza_session.get_stop_losses_strict",
                   side_effect=RuntimeError("persistent failure")), \
             patch("portfolio.avanza_session.close_playwright"), \
             patch("time.sleep"):
            ok, snap = ml._capture_stop_snapshot("1650161")
            assert ok is False
            assert snap == []

    def test_close_playwright_failure_does_not_block_retry(self):
        """If close_playwright itself raises, the retry still attempts."""
        from data import metals_loop as ml

        sl_data = [{"orderbook": {"id": "1650161"}, "stop_id": "abc"}]
        call_count = {"n": 0}

        def get_strict_side_effect():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first attempt died")
            return sl_data

        with patch("portfolio.avanza_session.get_stop_losses_strict", side_effect=get_strict_side_effect), \
             patch("portfolio.avanza_session.close_playwright",
                   side_effect=RuntimeError("close also failed")), \
             patch("time.sleep"):
            ok, snap = ml._capture_stop_snapshot("1650161")
            # Retry still happens despite close_playwright failing
            assert ok is True
            assert len(snap) == 1
            assert call_count["n"] == 2

    def test_empty_ob_id_short_circuits(self):
        """Empty ob_id returns (True, []) without any API call."""
        from data import metals_loop as ml
        with patch("portfolio.avanza_session.get_stop_losses_strict") as mock_strict:
            ok, snap = ml._capture_stop_snapshot("")
            assert ok is True
            assert snap == []
            mock_strict.assert_not_called()

    def test_filters_to_target_orderbook(self):
        """Only stops matching the requested ob_id are returned."""
        from data import metals_loop as ml
        sl_data = [
            {"orderbook": {"id": "1650161"}, "stop_id": "wanted"},
            {"orderbook": {"id": "9999"}, "stop_id": "ignore_me"},
            {"orderbook": {"id": "1650161"}, "stop_id": "wanted2"},
        ]
        with patch("portfolio.avanza_session.get_stop_losses_strict", return_value=sl_data):
            ok, snap = ml._capture_stop_snapshot("1650161")
            assert ok is True
            assert len(snap) == 2
            ids = {s["stop_id"] for s in snap}
            assert ids == {"wanted", "wanted2"}
