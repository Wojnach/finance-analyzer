"""Suspend-aware staleness for data/heartbeat.txt.

This Steam Deck deep-suspends constantly — measured 2026-08-17, it had spent
48 of its last 96 uptime hours suspended. Every wall-clock staleness check
therefore mis-reports a perfectly healthy loop as dead after each resume:

  * portfolio/main.py sent "_LOOP RESTARTED_ ... Possible crash" to Telegram
  * dashboard/system_status.py marked heartbeat.txt "frozen"

The loop itself was always correct — _sleep_for_next_cycle() is anchored to
time.monotonic(), which excludes suspend, so cadence never actually drifted.
Only the readers were wrong.

The fix records CLOCK_MONOTONIC and the host boot_id alongside the wall
timestamp, so age can be measured in AWAKE seconds. Subtracting total
suspend-since-boot would not do: after an 8h overnight suspend it would
mask a genuine 30-minute stall the next morning.
"""

import json
import time

import pytest

from portfolio import loop_health


def test_write_heartbeat_records_wall_monotonic_and_boot_id(tmp_path):
    p = tmp_path / "heartbeat.txt"
    loop_health.write_wall_heartbeat(p)

    payload = json.loads(p.read_text())
    assert payload["ts"].endswith("+00:00")
    assert isinstance(payload["awake_s"], float)
    assert payload["boot_id"]


def test_awake_age_is_measured_on_the_monotonic_clock(tmp_path, monkeypatch):
    """Wall clock may jump; awake age must not."""
    p = tmp_path / "heartbeat.txt"
    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 1000.0)
    loop_health.write_wall_heartbeat(p)

    # 90 awake seconds later
    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 1090.0)
    info = loop_health.read_heartbeat_age(p)
    assert info["awake_age_s"] == pytest.approx(90.0)
    assert info["same_boot"] is True


def test_suspend_does_not_inflate_awake_age(tmp_path, monkeypatch):
    """The whole point: 2h of suspend must not read as 2h of staleness."""
    p = tmp_path / "heartbeat.txt"
    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 500.0)
    loop_health.write_wall_heartbeat(p)

    # Host suspended 2h then resumed: wall clock advanced, monotonic barely.
    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 560.0)
    info = loop_health.read_heartbeat_age(p)
    assert info["awake_age_s"] == pytest.approx(60.0)


def test_real_stall_after_a_suspend_is_still_detected(tmp_path, monkeypatch):
    """Guard against over-suppression — a genuine stall must survive the fix."""
    p = tmp_path / "heartbeat.txt"
    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 100.0)
    loop_health.write_wall_heartbeat(p)

    # Loop wedged for 1800 awake seconds, regardless of any suspend before it.
    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 1900.0)
    info = loop_health.read_heartbeat_age(p)
    assert info["awake_age_s"] == pytest.approx(1800.0)
    assert info["awake_age_s"] > 1200


def test_reboot_is_reported_as_different_boot_not_a_stall(tmp_path, monkeypatch):
    p = tmp_path / "heartbeat.txt"
    monkeypatch.setattr(loop_health, "_boot_id", lambda: "boot-A")
    loop_health.write_wall_heartbeat(p)

    monkeypatch.setattr(loop_health, "_boot_id", lambda: "boot-B")
    info = loop_health.read_heartbeat_age(p)
    assert info["same_boot"] is False
    # Awake age across a reboot is unknowable — must not be invented.
    assert info["awake_age_s"] is None
    assert info["wall_age_s"] is not None


def test_legacy_bare_iso_file_still_parses(tmp_path):
    """Old heartbeats are plain ISO text; readers must not crash on them."""
    from datetime import UTC, datetime, timedelta

    p = tmp_path / "heartbeat.txt"
    p.write_text((datetime.now(UTC) - timedelta(seconds=300)).isoformat())

    info = loop_health.read_heartbeat_age(p)
    assert info["awake_age_s"] is None, "cannot know awake age without an anchor"
    assert info["wall_age_s"] == pytest.approx(300, abs=20)
    assert info["same_boot"] is False


def test_missing_file_reports_nothing_rather_than_zero(tmp_path):
    info = loop_health.read_heartbeat_age(tmp_path / "nope.txt")
    assert info["wall_age_s"] is None
    assert info["awake_age_s"] is None


def test_garbage_file_does_not_raise(tmp_path):
    p = tmp_path / "heartbeat.txt"
    p.write_text("{not json at all")
    info = loop_health.read_heartbeat_age(p)
    assert info["wall_age_s"] is None
    assert info["awake_age_s"] is None


def test_awake_seconds_excludes_suspend_on_this_host():
    """CLOCK_MONOTONIC must be the suspend-excluding clock we think it is."""
    boottime = time.clock_gettime(time.CLOCK_BOOTTIME)
    awake = loop_health._awake_seconds()
    assert awake <= boottime + 1.0


# --- the two readers that were lying ---------------------------------------


def test_main_startup_check_ignores_suspend(monkeypatch, tmp_path):
    """The Telegram "Possible crash" alarm must not fire on a resume."""
    import portfolio.main as main_mod

    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 1000.0)
    loop_health.write_wall_heartbeat(tmp_path / "heartbeat.txt")

    sent = []
    monkeypatch.setattr(main_mod, "_load_config", lambda *a, **k: {}, raising=False)
    import portfolio.message_store as ms

    monkeypatch.setattr(ms, "send_or_store", lambda msg, cfg, **kw: sent.append(msg))

    # Only 120 awake seconds passed, but hours of wall clock did.
    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 1120.0)
    main_mod._warn_if_heartbeat_stale()
    assert sent == [], f"false crash alarm after suspend: {sent}"


def test_main_startup_check_still_reports_a_real_stall(monkeypatch, tmp_path):
    import portfolio.main as main_mod

    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 1000.0)
    loop_health.write_wall_heartbeat(tmp_path / "heartbeat.txt")

    sent = []
    monkeypatch.setattr(main_mod, "_load_config", lambda *a, **k: {}, raising=False)
    import portfolio.message_store as ms

    monkeypatch.setattr(ms, "send_or_store", lambda msg, cfg, **kw: sent.append(msg))

    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 1000.0 + 3600)
    main_mod._warn_if_heartbeat_stale()
    assert len(sent) == 1, "a genuine hour-long stall must still alert"
    assert "LOOP RESTARTED" in sent[0]


def test_dashboard_heartbeat_freshness_is_suspend_aware(tmp_path, monkeypatch):
    """system_status must not paint a healthy loop 'frozen' after a resume."""
    from dashboard import system_status

    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 5000.0)
    loop_health.write_wall_heartbeat(tmp_path / "heartbeat.txt")

    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 5060.0)
    out = system_status._source_freshness(tmp_path, "heartbeat.txt")
    assert out["frozen"] is False, out


def test_dashboard_still_freezes_on_a_real_stall(tmp_path, monkeypatch):
    from dashboard import system_status

    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 5000.0)
    loop_health.write_wall_heartbeat(tmp_path / "heartbeat.txt")

    monkeypatch.setattr(loop_health, "_awake_seconds", lambda: 5000.0 + 3600)
    out = system_status._source_freshness(tmp_path, "heartbeat.txt")
    assert out["frozen"] is True, out


def test_dashboard_layer1_reports_parsed_timestamp_not_raw_json(tmp_path):
    """last_cycle_ts is rendered as a timestamp — it must not be a JSON blob."""
    from dashboard import system_status

    loop_health.write_wall_heartbeat(tmp_path / "heartbeat.txt")
    out = system_status._layer1(tmp_path)

    ts = out["last_cycle_ts"]
    assert ts and not ts.startswith("{"), ts
    from datetime import datetime

    datetime.fromisoformat(ts)  # raises if this is not a real timestamp


def test_legacy_file_is_labelled_legacy_not_a_reboot(tmp_path, caplog):
    """A bare-ISO heartbeat has no anchor — that is not evidence of a reboot.

    Shipped wrong once: the first deploy logged "predates this boot" for a
    heartbeat written 0 minutes earlier in the same boot, purely because the
    legacy format carries no boot_id.
    """
    import logging

    import portfolio.main as main_mod

    (tmp_path / "heartbeat.txt").write_text(
        __import__("datetime").datetime.now(
            __import__("datetime").UTC
        ).isoformat()
    )

    info = loop_health.read_heartbeat_age(tmp_path / "heartbeat.txt")
    assert info["has_anchor"] is False

    caplog.set_level(logging.INFO)
    import unittest.mock as mock

    with mock.patch.object(main_mod, "DATA_DIR", tmp_path):
        main_mod._warn_if_heartbeat_stale()
    text = caplog.text.lower()
    assert "reboot" not in text, f"legacy file mislabelled as a reboot: {caplog.text}"


def test_genuine_reboot_still_says_reboot(tmp_path, caplog, monkeypatch):
    import logging

    import portfolio.main as main_mod

    monkeypatch.setattr(loop_health, "_boot_id", lambda: "boot-A")
    loop_health.write_wall_heartbeat(tmp_path / "heartbeat.txt")
    monkeypatch.setattr(loop_health, "_boot_id", lambda: "boot-B")

    info = loop_health.read_heartbeat_age(tmp_path / "heartbeat.txt")
    assert info["has_anchor"] is True
    assert info["same_boot"] is False

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(main_mod, "DATA_DIR", tmp_path)
    main_mod._warn_if_heartbeat_stale()
    assert "reboot" in caplog.text.lower(), caplog.text
