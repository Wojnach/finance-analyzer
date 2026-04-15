"""Tests for BUG-178 phase log helpers in signal_engine.

The phase log is a per-ticker list of (phase_name, duration_seconds) tuples
recorded during the post-dispatch code in generate_signal(). It exists so
the BUG-178 slow-cycle diagnostic in main.py can tell us WHICH phase burned
the time instead of lumping everything under __post_dispatch__.

These tests exercise the helpers directly (no generate_signal invocation)
to avoid pulling in the full signal engine dependency chain.
"""
from __future__ import annotations

import time

import pytest

from portfolio.signal_engine import (
    _PHASE_WARN_THRESHOLD_S,
    _record_phase,
    _reset_phase_log,
    get_phase_log,
)


class TestResetPhaseLog:
    def test_reset_creates_empty_list(self):
        _reset_phase_log("TEST1-USD")
        assert get_phase_log("TEST1-USD") == []

    def test_reset_clears_prior_entries(self):
        _reset_phase_log("TEST2-USD")
        _record_phase("TEST2-USD", "acc_load", time.monotonic())
        assert len(get_phase_log("TEST2-USD")) == 1
        _reset_phase_log("TEST2-USD")
        assert get_phase_log("TEST2-USD") == []

    def test_reset_ignores_empty_ticker(self):
        _reset_phase_log("")
        _reset_phase_log(None)  # type: ignore[arg-type]
        # Must not raise; no state written


class TestRecordPhase:
    def test_record_appends_entry(self):
        _reset_phase_log("TEST3-USD")
        # Fake a known elapsed duration by back-dating the start — sleep()-based
        # timing is unreliable on Windows (monotonic() has ~15.6 ms granularity).
        start = time.monotonic() - 0.05
        dur = _record_phase("TEST3-USD", "weighted", start)
        log = get_phase_log("TEST3-USD")
        assert len(log) == 1
        assert log[0][0] == "weighted"
        assert log[0][1] == dur
        assert dur >= 0.04  # leave a little slack against clock jitter

    def test_record_preserves_insertion_order(self):
        _reset_phase_log("TEST4-USD")
        for phase in ("acc_load", "utility_overlay", "weighted", "penalties"):
            start = time.monotonic()
            _record_phase("TEST4-USD", phase, start)
        log = get_phase_log("TEST4-USD")
        assert [p for p, _ in log] == ["acc_load", "utility_overlay", "weighted", "penalties"]

    def test_record_returns_duration(self):
        _reset_phase_log("TEST5-USD")
        start = time.monotonic()
        dur = _record_phase("TEST5-USD", "x", start)
        # Duration is non-negative and close to actual elapsed
        assert dur >= 0.0
        assert dur < 1.0  # way less than a second for a no-op

    def test_record_empty_ticker_is_noop(self):
        # Empty-ticker fast path must not raise and must not create a "" entry.
        before = get_phase_log("")
        assert _record_phase("", "x", time.monotonic()) == 0.0
        assert _record_phase(None, "x", time.monotonic()) == 0.0  # type: ignore[arg-type]
        after = get_phase_log("")
        assert before == after

    def test_slow_phase_emits_warning(self, caplog):
        """Phases exceeding _PHASE_WARN_THRESHOLD_S should log a WARNING.

        This is what surfaces slow individual phases (cold accuracy load, lock
        contention) in portfolio.log without needing a BUG-178 pool timeout.
        """
        _reset_phase_log("TEST6-USD")
        import logging as _logging
        # Synthesize a slow start by subtracting threshold + 0.5s from monotonic.
        fake_start = time.monotonic() - (_PHASE_WARN_THRESHOLD_S + 0.5)
        with caplog.at_level(_logging.WARNING, logger="portfolio.signal_engine"):
            dur = _record_phase("TEST6-USD", "slow_phase", fake_start)
        assert dur >= _PHASE_WARN_THRESHOLD_S
        # Expect a SLOW-PHASE warning.
        assert any("[SLOW-PHASE] TEST6-USD/slow_phase" in r.message for r in caplog.records)


class TestGetPhaseLog:
    def test_returns_empty_for_unknown_ticker(self):
        assert get_phase_log("NEVER-SEEN-TICKER-XYZ") == []

    def test_returns_copy_not_reference(self):
        """Callers must not be able to mutate internal state.

        Returning the live list would let the caller corrupt the record for
        the next invocation.
        """
        _reset_phase_log("TEST7-USD")
        _record_phase("TEST7-USD", "a", time.monotonic())
        snap = get_phase_log("TEST7-USD")
        snap.append(("tampered", 99.0))  # should not affect internal state
        fresh = get_phase_log("TEST7-USD")
        assert fresh == [(snap[0][0], snap[0][1])]
        assert ("tampered", 99.0) not in fresh


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
