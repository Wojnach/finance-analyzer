"""Tests for the sustained-debounce gate in portfolio.trigger."""

from unittest.mock import patch

from portfolio.trigger import (
    RANGING_CONSENSUS_MIN_CONFIDENCE,
    SUSTAINED_CHECKS,
    SUSTAINED_DURATION_S,
    _update_sustained,
)


class TestSustainedIncrementsOnSameValue:
    """Calling _update_sustained with the same value SUSTAINED_CHECKS times
    should set count_ok=True on the third call."""

    def test_count_ok_on_third_call(self):
        state = {}
        now = 1000.0
        for _ in range(SUSTAINED_CHECKS - 1):
            count_ok, duration_ok = _update_sustained(state, "k", "BUY", now)
        # After 2 calls count is 2 — not yet enough
        assert count_ok is False
        # Third call crosses the threshold
        count_ok, duration_ok = _update_sustained(state, "k", "BUY", now)
        assert count_ok is True


class TestSustainedResetsOnValueChange:
    """Changing the value must reset the counter to 1."""

    def test_reset_after_change(self):
        state = {}
        now = 1000.0
        _update_sustained(state, "k", "A", now)
        _update_sustained(state, "k", "A", now)
        # Count is 2 after two identical calls
        assert state["k"]["count"] == 2

        # Value changes — count resets to 1
        count_ok, _ = _update_sustained(state, "k", "B", now)
        assert state["k"]["count"] == 1
        assert count_ok is False


class TestSustainedDurationGate:
    """Duration gate fires when wall-clock elapsed >= SUSTAINED_DURATION_S.

    2026-06-11 (audit B5): _update_sustained now anchors on the caller's
    wall-clock now_ts (epoch seconds) instead of time.monotonic(), so the
    persisted start survives loop restarts.
    """

    def test_duration_ok_after_120s(self):
        state = {}
        wall_start = 1.7e9

        # First call — anchors _wall_start
        _update_sustained(state, "k", "X", wall_start)

        # Second call — still within window
        _, duration_ok = _update_sustained(
            state, "k", "X", wall_start + SUSTAINED_DURATION_S - 1
        )
        assert duration_ok is False

        # Third call — crosses the duration boundary
        count_ok, duration_ok = _update_sustained(
            state, "k", "X", wall_start + SUSTAINED_DURATION_S
        )
        assert duration_ok is True
        # Three calls total with same value — count gate also passes
        assert count_ok is True


class TestSustainedFirstCallNotOk:
    """The very first call always has count=1 and zero elapsed time —
    both gates must be False."""

    def test_first_call_both_false(self):
        state = {}
        count_ok, duration_ok = _update_sustained(state, "k", "V", 0.0)
        assert count_ok is False
        assert duration_ok is False
        assert state["k"]["count"] == 1


class TestRangingDampeningConstant:
    """RANGING_CONSENSUS_MIN_CONFIDENCE must be 0.40."""

    def test_value(self):
        assert RANGING_CONSENSUS_MIN_CONFIDENCE == 0.40
