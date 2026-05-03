"""Tests for loop_health.write_heartbeat() — the shared helper that
metals_loop and golddigger (and any future loop) call to emit the
watchdog-compatible heartbeat file.

Coverage:
- Schema written matches what read_loop_status / read_loop_health expect.
- n_positions / cycle / ok / extra fields round-trip correctly.
- Failure path swallows exceptions and returns False.
- Override timestamp produces deterministic output.
"""
from __future__ import annotations

import datetime
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio import loop_health


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_writes_minimal_payload(tmp_path):
    hb = tmp_path / "x_loop.heartbeat"
    ok = loop_health.write_heartbeat(hb, cycle=1)
    assert ok is True
    assert hb.exists()

    payload = _read(hb)
    # Required by read_loop_status:
    assert isinstance(payload["ts"], str)
    datetime.datetime.fromisoformat(payload["ts"].replace("Z", "+00:00"))
    # Default operator fields:
    assert payload["status"] == "ok"
    assert payload["cycle"] == 1
    assert payload["ok"] is True
    assert payload["n_positions"] == 0


def test_writes_extra_fields(tmp_path):
    hb = tmp_path / "x_loop.heartbeat"
    loop_health.write_heartbeat(
        hb, cycle=42, n_positions=3, extra={"phase": "shadow", "regime": "trending_up"}
    )
    payload = _read(hb)
    assert payload["cycle"] == 42
    assert payload["n_positions"] == 3
    assert payload["phase"] == "shadow"
    assert payload["regime"] == "trending_up"


def test_extra_cannot_override_required_fields(tmp_path):
    """`extra` is a merge — caller can override defaults like ok if they
    really want to. This is intentional to keep the helper flexible, but
    we document the behavior here so a future refactor doesn't quietly
    break it."""
    hb = tmp_path / "x_loop.heartbeat"
    loop_health.write_heartbeat(
        hb, cycle=1, ok=True, extra={"ok": False, "custom_status": "degraded"}
    )
    payload = _read(hb)
    # extra wins — caller signaled they wanted to override.
    assert payload["ok"] is False
    assert payload["custom_status"] == "degraded"


def test_uses_provided_timestamp(tmp_path):
    """Caller can pin `now` for reproducible tests."""
    hb = tmp_path / "x_loop.heartbeat"
    pinned = datetime.datetime(2026, 5, 3, 12, 0, 0, tzinfo=datetime.UTC)
    loop_health.write_heartbeat(hb, cycle=1, now=pinned)
    payload = _read(hb)
    assert payload["ts"] == "2026-05-03T12:00:00+00:00"


def test_failure_returns_false_and_does_not_raise(tmp_path, monkeypatch):
    """If atomic_write_json fails, the helper must NOT propagate — live
    trading loops cannot crash on telemetry failure."""
    hb = tmp_path / "x_loop.heartbeat"

    def _boom(*_a, **_kw):
        raise OSError("disk full (simulated)")

    from portfolio import file_utils
    monkeypatch.setattr(file_utils, "atomic_write_json", _boom)

    result = loop_health.write_heartbeat(hb, cycle=1)
    assert result is False
    assert not hb.exists()


def test_round_trips_through_read_loop_status(tmp_path):
    """End-to-end: write_heartbeat output is read as 'fresh' by the
    consumer side — guarantees the schema match."""
    hb = tmp_path / "metals_loop.heartbeat"
    pinned = datetime.datetime(2026, 5, 3, 12, 0, 0, tzinfo=datetime.UTC)
    loop_health.write_heartbeat(hb, cycle=99, n_positions=2, now=pinned)

    # Read 60s later — should be 'fresh'.
    later = pinned + datetime.timedelta(seconds=60)
    status = loop_health.read_loop_status("metals", hb, now=later)
    assert status["state"] == "fresh"
    assert status["age_seconds"] == 60.0
    assert status["payload"]["cycle"] == 99
    assert status["payload"]["n_positions"] == 2


def test_round_trips_through_read_loop_health(tmp_path):
    """All five default loops can be assembled into a healthy rollup."""
    pinned = datetime.datetime(2026, 5, 3, 12, 0, 0, tzinfo=datetime.UTC)

    # Materialize one heartbeat per default loop in a fake repo root.
    (tmp_path / "data").mkdir(exist_ok=True)
    for name, rel in loop_health.DEFAULT_HEARTBEAT_FILES.items():
        full = tmp_path / rel
        loop_health.write_heartbeat(full, cycle=1, now=pinned)

    later = pinned + datetime.timedelta(seconds=30)
    rollup = loop_health.read_loop_health(repo_root=tmp_path, now=later)

    assert rollup["any_unhealthy"] is False, rollup["unhealthy"]
    for name in loop_health.DEFAULT_HEARTBEAT_FILES:
        assert rollup["loops"][name]["state"] == "fresh", name
