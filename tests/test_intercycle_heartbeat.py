"""Tests for the inter-cycle heartbeat fix (2026-05-29).

Covers portfolio.main._sleep_for_next_cycle's chunked beat behavior and the
module-level _heartbeat_beat helper. The premortem narratives this pins:
  #4 — monotonic deadline-anchoring survives an OS-suspend clock jump.
  #5 — KeyboardInterrupt from a beat propagates (not swallowed).
  #3 — persistent beat failure escalates to critical_errors.jsonl after 3 strikes.
  #2 — slow beat is surfaced (best-effort; no assertion on the log).
"""

from __future__ import annotations

import pytest

from portfolio import main


class FakeClock:
    """Deterministic monotonic + sleep. `jumps[i]` adds extra seconds AFTER the
    i-th sleep call to simulate an OS suspend / NTP step mid-sleep."""

    def __init__(self, start: float = 1000.0, jumps: dict[int, float] | None = None):
        self.t = start
        self.sleeps: list[float] = []
        self.jumps = jumps or {}

    def monotonic(self) -> float:
        return self.t

    def sleep(self, secs: float) -> None:
        self.sleeps.append(secs)
        self.t += secs
        idx = len(self.sleeps) - 1
        if idx in self.jumps:
            self.t += self.jumps[idx]


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(main, "time", c)  # main.py calls time.monotonic()/time.sleep()
    return c


# ---------------------------------------------------------------------------
# _sleep_for_next_cycle
# ---------------------------------------------------------------------------

class TestSleepForNextCycle:
    def test_no_beat_single_sleep_unchanged(self, clock):
        # beat=None -> behaviour unchanged: one sleep of the full remaining.
        main._sleep_for_next_cycle(clock.monotonic(), interval_s=600)
        assert clock.sleeps == [600]

    def test_overrun_does_not_sleep(self, clock):
        # Work already exceeded the interval -> no sleep, just a warning.
        clock.t = 1700  # elapsed 700 > interval 600
        main._sleep_for_next_cycle(1000.0, interval_s=600, beat=lambda: None)
        assert clock.sleeps == []

    def test_chunked_beats_each_slice_and_respects_deadline(self, clock):
        beats = []
        main._sleep_for_next_cycle(clock.monotonic(), interval_s=600,
                                   beat=lambda: beats.append(1), beat_interval_s=60)
        # 600s / 60s = 10 slices -> 10 sleeps of 60 -> 10 beats. Total slept 600.
        assert clock.sleeps == [60] * 10
        assert len(beats) == 10
        assert clock.t == 1600  # deadline exactly reached

    def test_partial_final_slice(self, clock):
        beats = []
        main._sleep_for_next_cycle(clock.monotonic(), interval_s=130,
                                   beat=lambda: beats.append(1), beat_interval_s=60)
        assert clock.sleeps == [60, 60, 10]
        assert len(beats) == 3
        assert clock.t == 1130

    def test_premortem4_clock_jump_returns_promptly(self, monkeypatch):
        # OS suspend / NTP step: monotonic jumps +100000s after the 2nd sleep.
        c = FakeClock(jumps={1: 100_000})
        monkeypatch.setattr(main, "time", c)
        beats = []
        main._sleep_for_next_cycle(c.monotonic(), interval_s=600,
                                   beat=lambda: beats.append(1), beat_interval_s=60)
        # After the jump the deadline is far in the past -> loop breaks; no extra
        # sleeps queued and crucially NO negative / huge sleep was ever issued.
        assert clock_all_sleeps_bounded(c.sleeps, lo=0.0, hi=60.0)
        assert len(c.sleeps) == 2  # only the two pre-jump slices ran

    def test_premortem5_keyboardinterrupt_propagates(self, clock):
        def ki_beat():
            raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            main._sleep_for_next_cycle(clock.monotonic(), interval_s=600,
                                       beat=ki_beat, beat_interval_s=60)

    def test_ordinary_beat_exception_is_swallowed(self, clock):
        calls = []

        def boom():
            calls.append(1)
            raise RuntimeError("disk full")
        # Must NOT raise — a beat failure can't abort the loop.
        main._sleep_for_next_cycle(clock.monotonic(), interval_s=600,
                                   beat=boom, beat_interval_s=60)
        assert clock.sleeps == [60] * 10
        assert len(calls) == 10  # kept beating despite every beat raising


def clock_all_sleeps_bounded(sleeps, lo, hi):
    return all(lo <= s <= hi for s in sleeps)


# ---------------------------------------------------------------------------
# _heartbeat_beat (module-level, testable)
# ---------------------------------------------------------------------------

class TestHeartbeatBeat:
    def test_success_resets_streak(self, monkeypatch):
        import portfolio.health as health
        monkeypatch.setattr(health, "heartbeat", lambda: None)
        assert main._heartbeat_beat(2) == 0  # prior failures cleared on success

    def test_failure_increments_streak(self, monkeypatch):
        import portfolio.health as health

        def boom():
            raise OSError("disk full")
        monkeypatch.setattr(health, "heartbeat", boom)
        # Suppress the escalation path's writer for the non-threshold call.
        import portfolio.claude_gate as cg
        monkeypatch.setattr(cg, "record_critical_error", lambda **k: True)
        assert main._heartbeat_beat(0) == 1
        assert main._heartbeat_beat(1) == 2

    def test_premortem3_escalates_at_threshold(self, monkeypatch):
        import portfolio.health as health
        import portfolio.claude_gate as cg

        monkeypatch.setattr(health, "heartbeat", lambda: (_ for _ in ()).throw(OSError("x")))
        recorded = []
        monkeypatch.setattr(cg, "record_critical_error",
                            lambda **k: recorded.append(k) or True)
        # 3rd consecutive failure -> exactly one escalation.
        assert main._heartbeat_beat(2) == 3
        assert len(recorded) == 1
        assert recorded[0]["category"] == "heartbeat_write_failing"
        # 4th failure does NOT double-escalate (only fires at == threshold).
        assert main._heartbeat_beat(3) == 4
        assert len(recorded) == 1

    def test_keyboardinterrupt_propagates(self, monkeypatch):
        import portfolio.health as health

        def ki():
            raise KeyboardInterrupt
        monkeypatch.setattr(health, "heartbeat", ki)
        with pytest.raises(KeyboardInterrupt):
            main._heartbeat_beat(0)
