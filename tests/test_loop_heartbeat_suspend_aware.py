"""The per-loop heartbeats need the same awake-time anchor heartbeat.txt got.

read_loop_health() feeds scripts/loop_health_watchdog.py, which ALERTS. On a
host that suspends for hours (this Deck spends ~half its uptime suspended) a
wall-clock age makes every loop look "stale" the moment it resumes, so the
watchdog pages about loops that are running fine.

Exposure is narrower than for heartbeat.txt — these loops cycle every 60s and
self-heal on the next write — but the watchdog can fire inside that window,
and a false page is exactly what erodes trust in a real one.

Same design as bf768f97: an exact per-write CLOCK_MONOTONIC anchor scoped by
boot_id. Never a blanket subtraction of suspend-since-boot, which would mask a
genuine stall that began after a long suspend.
"""

import json

import pytest

from portfolio import loop_health


def test_write_heartbeat_includes_awake_anchor(tmp_path):
    p = tmp_path / "crypto_loop.heartbeat"
    loop_health.write_heartbeat(p, cycle=7)

    payload = json.loads(p.read_text())
    assert payload["cycle"] == 7
    assert isinstance(payload["awake_s"], float)
    assert payload["boot_id"]


def test_write_heartbeat_keeps_its_existing_schema(tmp_path):
    """The anchor is additive — watchdog/dashboard readers must not break."""
    p = tmp_path / "oil_loop.heartbeat"
    loop_health.write_heartbeat(
        p, cycle=3, ok=False, n_positions=2, extra={"phase": "x"}
    )

    payload = json.loads(p.read_text())
    for k in ("ts", "cycle", "ok", "n_positions", "phase"):
        assert k in payload, f"{k} missing — existing readers depend on it"
    assert payload["ok"] is False
    assert payload["n_positions"] == 2


def test_status_is_fresh_when_only_suspend_elapsed(tmp_path, monkeypatch):
    p = tmp_path / "metals_loop.heartbeat"
    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 900.0)
    loop_health.write_heartbeat(p, cycle=1)

    # Two hours of wall clock passed, but only 30 awake seconds.
    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 930.0)
    import datetime

    later = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=2)
    st = loop_health.read_loop_status("metals", p, now=later)
    assert st["state"] == "fresh", st
    assert st["age_seconds"] == pytest.approx(30.0)


def test_a_genuinely_stalled_loop_is_still_stale(tmp_path, monkeypatch):
    p = tmp_path / "metals_loop.heartbeat"
    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 900.0)
    loop_health.write_heartbeat(p, cycle=1)

    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 900.0 + 1800)
    st = loop_health.read_loop_status("metals", p)
    assert st["state"] == "stale", st
    assert st["age_seconds"] == pytest.approx(1800.0)


def test_heartbeat_from_a_previous_boot_falls_back_to_wall_clock(tmp_path, monkeypatch):
    """No comparable anchor across a reboot — must not invent an awake age."""
    import datetime

    p = tmp_path / "crypto_loop.heartbeat"
    monkeypatch.setattr(loop_health, "_boot_id", lambda: "boot-A")
    old = datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=4000)
    loop_health.write_heartbeat(p, cycle=1, now=old)

    monkeypatch.setattr(loop_health, "_boot_id", lambda: "boot-B")
    st = loop_health.read_loop_status("crypto", p)
    assert st["state"] == "stale"
    assert st["age_seconds"] == pytest.approx(4000, abs=30)


def test_legacy_heartbeat_without_anchor_still_reads(tmp_path):
    """Files written before this change must keep working on wall clock."""
    import datetime

    p = tmp_path / "mstr_loop.heartbeat"
    ts = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(seconds=120)
    ).isoformat()
    p.write_text(json.dumps({"ts": ts, "cycle": 5, "ok": True}))

    st = loop_health.read_loop_status("mstr", p)
    assert st["state"] == "fresh"
    assert st["age_seconds"] == pytest.approx(120, abs=20)
