"""Tests for wall-clock duration gating in trigger._update_sustained().

History: this file originally pinned the time.monotonic() implementation.
2026-06-11 (audit B5): _update_sustained switched to wall-clock epoch
seconds (the caller's now_ts) because the monotonic origin was persisted
into trigger_state.json where it is meaningless across restarts/reboots
(QPC-since-boot on Windows). These tests pin the new contract:

  * duration gate driven by now_ts deltas;
  * persisted start survives a simulated restart (state round-trip);
  * legacy persisted monotonic values (< 1e9) are detected and re-seeded
    instead of firing the gate spuriously;
  * value changes reset the start; count gate independent of duration.
"""

from portfolio.trigger import SUSTAINED_CHECKS, SUSTAINED_DURATION_S, _update_sustained

_EPOCH = 1.7e9  # plausible wall-clock anchor (2023-11-ish)


class TestUpdateSustainedWallClock:

    def test_duration_gate_fires_on_wall_clock_exceeding_threshold(self):
        state = {}
        _update_sustained(state, "sig", "BUY", _EPOCH)
        count_ok, duration_ok = _update_sustained(
            state, "sig", "BUY", _EPOCH + SUSTAINED_DURATION_S
        )
        assert duration_ok is True

    def test_duration_gate_not_fired_before_threshold(self):
        state = {}
        _update_sustained(state, "sig", "SELL", _EPOCH)
        count_ok, duration_ok = _update_sustained(
            state, "sig", "SELL", _EPOCH + SUSTAINED_DURATION_S - 1
        )
        assert duration_ok is False

    def test_wall_start_resets_on_value_change(self):
        state = {}
        _update_sustained(state, "sig", "BUY", _EPOCH)
        later = _EPOCH + SUSTAINED_DURATION_S + 10
        # Change value to SELL — resets _wall_start and count
        _update_sustained(state, "sig", "SELL", later)
        assert state["sig"]["_wall_start"] == later
        assert state["sig"]["count"] == 1
        # Tiny advance with same SELL — should NOT fire
        _, duration_ok = _update_sustained(state, "sig", "SELL", later + 1.0)
        assert duration_ok is False

    def test_count_gate_works_independently(self):
        state = {}
        count_ok = duration_ok = False
        for i in range(SUSTAINED_CHECKS):
            count_ok, duration_ok = _update_sustained(
                state, "sig", "BUY", _EPOCH + i * 0.01
            )
        assert count_ok is True
        assert duration_ok is False

    def test_persisted_start_survives_restart(self):
        """Wall-clock start written by a previous process remains valid:
        a restart no longer resets the duration debounce (the old monotonic
        origin made this impossible)."""
        # State as persisted by a previous loop process 900s ago.
        state = {
            "sig": {
                "value": "BUY",
                "count": 2,
                "_wall_start": _EPOCH - SUSTAINED_DURATION_S,
            }
        }
        _, duration_ok = _update_sustained(state, "sig", "BUY", _EPOCH)
        assert duration_ok is True
        assert state["sig"]["count"] == 3

    def test_legacy_monotonic_value_is_reseeded_not_fired(self):
        """A legacy _mono_start (monotonic seconds-since-boot, far below 1e9)
        in persisted state must NOT make the duration gate fire — it is
        re-seeded to now_ts."""
        state = {
            "sig": {
                "value": "SELL",
                "count": 5,
                "_mono_start": 84641.2,  # typical monotonic magnitude
            }
        }
        count_ok, duration_ok = _update_sustained(state, "sig", "SELL", _EPOCH)
        assert duration_ok is False  # re-seeded, zero elapsed
        assert state["sig"]["_wall_start"] == _EPOCH
        # Count path unaffected by the re-seed
        assert state["sig"]["count"] == 6
        assert count_ok is True

    def test_implausible_future_start_is_reseeded(self):
        """A persisted start absurdly far in the future (clock damage) is
        re-seeded rather than producing a permanently dead gate."""
        state = {
            "sig": {
                "value": "BUY",
                "count": 1,
                "_wall_start": _EPOCH + 10 * 86400,
            }
        }
        _, duration_ok = _update_sustained(state, "sig", "BUY", _EPOCH)
        assert duration_ok is False
        assert state["sig"]["_wall_start"] == _EPOCH
