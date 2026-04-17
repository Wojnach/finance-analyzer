"""Tests for monotonic clock usage in trigger._update_sustained()."""

import time
from unittest.mock import patch

from portfolio.trigger import SUSTAINED_CHECKS, SUSTAINED_DURATION_S, _update_sustained


class TestUpdateSustainedMonotonic:
    """Verify _update_sustained uses time.monotonic() for duration gating."""

    def test_duration_gate_fires_on_monotonic_exceeding_threshold(self):
        """Duration gate triggers when monotonic elapsed >= SUSTAINED_DURATION_S."""
        state = {}
        mono_start = 1000.0
        # First call — sets baseline
        with patch("portfolio.trigger.time.monotonic", return_value=mono_start):
            _update_sustained(state, "sig", "BUY", now_ts=100.0)

        # Second call — monotonic exceeds threshold
        with patch(
            "portfolio.trigger.time.monotonic",
            return_value=mono_start + SUSTAINED_DURATION_S,
        ):
            count_ok, duration_ok = _update_sustained(state, "sig", "BUY", now_ts=200.0)

        assert duration_ok is True

    def test_duration_gate_ignores_wall_clock(self):
        """Wall-clock (now_ts) exceeding threshold does NOT fire duration gate
        if monotonic time hasn't elapsed enough."""
        state = {}
        mono_start = 5000.0
        # First call — sets baseline
        with patch("portfolio.trigger.time.monotonic", return_value=mono_start):
            _update_sustained(state, "sig", "SELL", now_ts=100.0)

        # Wall-clock jumps way past threshold, but monotonic barely moves
        with patch(
            "portfolio.trigger.time.monotonic",
            return_value=mono_start + 1.0,
        ):
            count_ok, duration_ok = _update_sustained(
                state, "sig", "SELL", now_ts=100.0 + SUSTAINED_DURATION_S + 500
            )

        assert duration_ok is False

    def test_mono_start_resets_on_value_change(self):
        """_mono_start resets when the sustained value changes."""
        state = {}
        mono_start = 2000.0
        # First call with BUY
        with patch("portfolio.trigger.time.monotonic", return_value=mono_start):
            _update_sustained(state, "sig", "BUY", now_ts=0)

        # Advance monotonic past threshold
        mono_after = mono_start + SUSTAINED_DURATION_S + 10
        # Change value to SELL — resets _mono_start
        with patch("portfolio.trigger.time.monotonic", return_value=mono_after):
            _update_sustained(state, "sig", "SELL", now_ts=0)

        assert state["sig"]["_mono_start"] == mono_after
        assert state["sig"]["count"] == 1

        # Next call with same SELL, tiny monotonic advance — should NOT fire
        with patch(
            "portfolio.trigger.time.monotonic", return_value=mono_after + 1.0
        ):
            _, duration_ok = _update_sustained(state, "sig", "SELL", now_ts=0)

        assert duration_ok is False

    def test_count_gate_works_independently(self):
        """Count gate fires after SUSTAINED_CHECKS regardless of duration."""
        state = {}
        mono = 9000.0
        for i in range(SUSTAINED_CHECKS):
            with patch(
                "portfolio.trigger.time.monotonic", return_value=mono + i * 0.01
            ):
                count_ok, duration_ok = _update_sustained(
                    state, "sig", "BUY", now_ts=0
                )

        # Count should have reached threshold
        assert count_ok is True
        # Duration should NOT have reached threshold (only fractions of a second)
        assert duration_ok is False
