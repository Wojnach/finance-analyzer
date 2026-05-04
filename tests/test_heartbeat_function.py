"""Tests for portfolio.health.heartbeat() — pre-Layer-2 watchdog touch.

Added 2026-05-04 alongside the call site in main.py:828. Layer 2 T2/T3
invocations can block 600-900s while update_health() (the normal heartbeat
write) only runs at end-of-cycle. heartbeat() advances last_heartbeat
without churning any other field so the dashboard stale flag stops
flapping during triggering cycles.

xdist safety: every test patches HEALTH_FILE to tmp_path. health.py is
otherwise stateless apart from the module-level lock (which is fine to
share across tests).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime

import portfolio.health as health_mod
from portfolio.file_utils import atomic_write_json


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(health_mod, "HEALTH_FILE", tmp_path / "health.json")


class TestHeartbeatTouchesOnlyLastHeartbeat:
    def test_advances_last_heartbeat_on_pre_populated_state(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)

        # Pre-populate state with a STALE heartbeat + non-trivial counters.
        stale_iso = "2026-01-01T00:00:00+00:00"
        atomic_write_json(health_mod.HEALTH_FILE, {
            "last_heartbeat": stale_iso,
            "cycle_count": 42,
            "signals_ok": 5,
            "signals_failed": 0,
            "uptime_seconds": 3600.0,
            "errors": [{"ts": "2026-01-01T00:00:00+00:00", "error": "old"}],
            "error_count": 1,
            "start_time": time.time() - 3600,
            "last_invocation_ts": stale_iso,
            "last_trigger_reason": "old_trigger",
        })

        before = datetime.now(UTC)
        health_mod.heartbeat()
        after = datetime.now(UTC)

        with open(health_mod.HEALTH_FILE) as f:
            state = json.load(f)

        # last_heartbeat advanced.
        new_hb = datetime.fromisoformat(state["last_heartbeat"])
        assert before <= new_hb <= after, (
            f"heartbeat ts {new_hb} not in window [{before}, {after}]"
        )

        # All other fields preserved verbatim.
        assert state["cycle_count"] == 42
        assert state["signals_ok"] == 5
        assert state["signals_failed"] == 0
        assert state["uptime_seconds"] == 3600.0
        assert state["errors"] == [{"ts": "2026-01-01T00:00:00+00:00", "error": "old"}]
        assert state["error_count"] == 1
        assert state["last_invocation_ts"] == stale_iso
        assert state["last_trigger_reason"] == "old_trigger"

    def test_does_not_call_update_health(self, monkeypatch, tmp_path):
        """heartbeat() must be cheap — no cycle-counter machinery, no
        last_invocation_ts mutation. If a refactor accidentally routes
        heartbeat() through update_health() we lose the cheap-and-cheerful
        guarantee and this test fires."""
        _isolate(monkeypatch, tmp_path)
        atomic_write_json(health_mod.HEALTH_FILE, {"cycle_count": 0})

        update_calls: list = []

        def boom(*args, **kwargs):
            update_calls.append((args, kwargs))
            raise AssertionError("heartbeat must not call update_health")

        monkeypatch.setattr(health_mod, "update_health", boom)

        health_mod.heartbeat()

        assert update_calls == []


class TestHeartbeatCreatesStateWhenMissing:
    def test_creates_file_on_first_call(self, monkeypatch, tmp_path):
        _isolate(monkeypatch, tmp_path)
        assert not health_mod.HEALTH_FILE.exists()

        health_mod.heartbeat()

        assert health_mod.HEALTH_FILE.exists()
        with open(health_mod.HEALTH_FILE) as f:
            state = json.load(f)
        assert "last_heartbeat" in state
        # load_health's default fields propagate.
        assert "start_time" in state
        assert state["cycle_count"] == 0
        assert state["error_count"] == 0


class TestHeartbeatAtomicity:
    def test_uses_atomic_write_json(self, monkeypatch, tmp_path):
        """If a future refactor replaces atomic_write_json with raw open(),
        a crash mid-write could leave a torn file that crashes the dashboard.
        Test catches that regression."""
        _isolate(monkeypatch, tmp_path)
        # Pre-populate so load_health doesn't need to fall back to defaults.
        atomic_write_json(health_mod.HEALTH_FILE, {"last_heartbeat": "2026-01-01T00:00:00+00:00"})

        atomic_calls: list = []
        original = health_mod.atomic_write_json

        def tracker(path, data):
            atomic_calls.append(path)
            return original(path, data)

        monkeypatch.setattr(health_mod, "atomic_write_json", tracker)

        health_mod.heartbeat()

        assert health_mod.HEALTH_FILE in atomic_calls


class TestHeartbeatThreadSafety:
    def test_concurrent_calls_produce_valid_state(self, monkeypatch, tmp_path):
        """N threads racing on heartbeat() must not produce torn JSON.
        atomic_write_json + the module-level _health_lock should serialize
        cleanly. If a regression dropped the lock, we'd see KeyError or
        corrupted JSON."""
        _isolate(monkeypatch, tmp_path)
        atomic_write_json(health_mod.HEALTH_FILE, {
            "last_heartbeat": "2026-01-01T00:00:00+00:00",
            "cycle_count": 0,
            "signals_ok": 0,
            "signals_failed": 0,
        })

        exceptions: list = []

        def worker():
            try:
                for _ in range(20):
                    health_mod.heartbeat()
            except Exception as e:
                exceptions.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not exceptions, f"thread exceptions: {exceptions}"

        # Final state file is valid JSON with a parseable timestamp.
        with open(health_mod.HEALTH_FILE) as f:
            state = json.load(f)
        # Just confirms the timestamp parses; no exception is the success.
        datetime.fromisoformat(state["last_heartbeat"])


class TestStalenessGateInteraction:
    def test_check_staleness_returns_fresh_after_heartbeat(self, monkeypatch, tmp_path):
        """End-to-end: a stale state file flips fresh after a single heartbeat()
        call. Mirrors the production flow (loop calls heartbeat before L2 →
        dashboard /api/health reads check_staleness → returns 'healthy')."""
        _isolate(monkeypatch, tmp_path)
        atomic_write_json(health_mod.HEALTH_FILE, {
            "last_heartbeat": "2026-01-01T00:00:00+00:00",
        })

        # Pre-condition: stale.
        is_stale, _, _ = health_mod.check_staleness(max_age_seconds=300)
        assert is_stale is True

        health_mod.heartbeat()

        # Post-condition: fresh.
        is_stale, age, _ = health_mod.check_staleness(max_age_seconds=300)
        assert is_stale is False
        assert age < 5.0


class TestHeartbeatKeepalive:
    """Codex P1: a single pre-Layer-2 heartbeat only delays staleness by 5 min.
    Layer 2 T2 has a 600s timeout, T3 has 900s. A keepalive context manager
    that ticks every interval seconds is the only correct shape — tests below
    verify that contract, especially that the heartbeat advances while the
    wrapped block is in flight."""

    def test_keepalive_ticks_during_long_running_block(self, monkeypatch, tmp_path):
        """During a 0.3s wrapped sleep, with a 0.05s interval, expect the
        heartbeat timestamp to advance multiple times. This is the core
        invariant: it's not "one beat then quiet for 9 min." A regression
        that drops the daemon thread or sets the interval too high will fail
        this assertion.
        """
        _isolate(monkeypatch, tmp_path)
        atomic_write_json(health_mod.HEALTH_FILE, {
            "last_heartbeat": "2026-01-01T00:00:00+00:00",
        })

        timestamps: list[str] = []

        def record_then_sleep():
            # Wrapped work: simulate a multi-second blocking subprocess.
            # During this block the keepalive thread should beat ~5-6 times.
            for _ in range(6):
                with open(health_mod.HEALTH_FILE) as f:
                    timestamps.append(json.load(f)["last_heartbeat"])
                time.sleep(0.05)

        with health_mod.heartbeat_keepalive(interval=0.02):
            record_then_sleep()

        # Distinct timestamps prove the daemon thread is actually beating.
        # We allow slight duplication (sampling between beats) but require
        # MORE than one unique timestamp.
        unique = set(timestamps)
        assert len(unique) >= 3, (
            f"keepalive didn't tick during wrapped block — got {len(unique)} "
            f"distinct timestamps over 0.3s with 0.02s interval. timestamps={timestamps}"
        )

    def test_keepalive_initial_beat_is_synchronous(self, monkeypatch, tmp_path):
        """Even if the wrapped block returns BEFORE the first interval tick,
        we still want one heartbeat. Otherwise a fast-returning Layer 2 (T1
        succeeds in 5s with interval=60s) would not bump the heartbeat at
        all under this design."""
        _isolate(monkeypatch, tmp_path)
        atomic_write_json(health_mod.HEALTH_FILE, {
            "last_heartbeat": "2026-01-01T00:00:00+00:00",
        })

        # Long interval — the only beat we'll see is the synchronous one
        # in __enter__.
        with health_mod.heartbeat_keepalive(interval=999.0):
            pass  # block returns instantly

        with open(health_mod.HEALTH_FILE) as f:
            state = json.load(f)
        # Heartbeat advanced from 2026-01-01.
        assert state["last_heartbeat"].startswith("20"), state["last_heartbeat"]
        assert state["last_heartbeat"] != "2026-01-01T00:00:00+00:00"

    def test_keepalive_thread_stops_on_exit(self, monkeypatch, tmp_path):
        """The daemon thread MUST stop within the join timeout. A leak would
        accumulate across cycles and (over a long-running loop) exhaust
        process handles."""
        _isolate(monkeypatch, tmp_path)
        atomic_write_json(health_mod.HEALTH_FILE, {
            "last_heartbeat": "2026-01-01T00:00:00+00:00",
        })

        ka = health_mod.heartbeat_keepalive(interval=0.05)
        with ka:
            pass

        # __exit__ joined with timeout=2.0; thread should be done.
        assert ka._thread is not None
        assert not ka._thread.is_alive(), "keepalive thread did not stop on exit"

    def test_keepalive_propagates_wrapped_exceptions(self, monkeypatch, tmp_path):
        """The context manager must NOT swallow exceptions from the wrapped
        block — Layer 2 errors need to bubble up to main.py's existing
        handlers. __exit__ still runs (cleanup happens), exception
        re-raises."""
        _isolate(monkeypatch, tmp_path)
        atomic_write_json(health_mod.HEALTH_FILE, {
            "last_heartbeat": "2026-01-01T00:00:00+00:00",
        })

        class BoomFromInside(RuntimeError):
            pass

        ka = health_mod.heartbeat_keepalive(interval=0.05)
        try:
            with ka:
                raise BoomFromInside("simulated L2 failure")
        except BoomFromInside:
            pass
        else:
            raise AssertionError("exception was swallowed")

        # Cleanup still happened.
        assert ka._thread is not None
        assert not ka._thread.is_alive()

    def test_keepalive_tick_failure_does_not_kill_thread(self, monkeypatch, tmp_path):
        """A single tick failure (e.g. transient disk full) must not stop
        keepalive — the loop has to keep ticking once disk recovers, OR at
        least keep trying so we don't silently lose the watchdog mid-T3."""
        _isolate(monkeypatch, tmp_path)
        atomic_write_json(health_mod.HEALTH_FILE, {
            "last_heartbeat": "2026-01-01T00:00:00+00:00",
        })

        call_count = [0]
        original = health_mod.heartbeat

        def flaky_heartbeat():
            call_count[0] += 1
            if call_count[0] == 2:
                raise OSError("simulated disk full")
            return original()

        monkeypatch.setattr(health_mod, "heartbeat", flaky_heartbeat)

        with health_mod.heartbeat_keepalive(interval=0.02):
            time.sleep(0.15)  # ~7 ticks expected; tick #2 fails

        # call_count > 2 proves the loop kept going past the failed tick.
        assert call_count[0] >= 3, (
            f"keepalive aborted after first tick failure (call_count={call_count[0]})"
        )

    def test_keepalive_keeps_health_fresh_through_simulated_l2_window(self, monkeypatch, tmp_path):
        """End-to-end regression check for codex P1: with the 300s stale
        threshold, a 0.4s simulated L2 block + 0.05s interval must keep
        check_staleness returning False the whole time.

        Scaled-down version of the production scenario: production has
        interval=60s, threshold=300s, T2_max=600s. This test compresses
        to interval=0.05s, threshold=0.2s, "T2"=0.4s — same ratios.
        """
        _isolate(monkeypatch, tmp_path)
        atomic_write_json(health_mod.HEALTH_FILE, {
            "last_heartbeat": "2026-01-01T00:00:00+00:00",
        })

        readings: list[bool] = []

        def simulated_l2_with_health_polling():
            for _ in range(8):
                # Dashboard would do this every poll cycle.
                is_stale, _, _ = health_mod.check_staleness(max_age_seconds=0.2)
                readings.append(is_stale)
                time.sleep(0.05)

        with health_mod.heartbeat_keepalive(interval=0.05):
            simulated_l2_with_health_polling()

        # Every reading must be fresh. Even one stale reading reproduces
        # the bug codex caught — that single-shot heartbeat lets the gate
        # trip mid-T2.
        assert not any(readings), (
            f"health went stale during keepalive — readings={readings}"
        )
