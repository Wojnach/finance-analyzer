"""Tests for the MSTR loop heartbeat — consumed by loop_health watchdog."""
from __future__ import annotations

import datetime
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio import loop_health
from portfolio.mstr_loop import config, loop, state


def _bot_state_with(positions: dict | None = None) -> state.BotState:
    s = state.default_state()
    if positions is not None:
        s.positions = positions
    return s


def test_heartbeat_written_to_configured_path(tmp_path, monkeypatch):
    """_write_heartbeat() writes JSON with the contract loop_health expects."""
    hb_path = tmp_path / "mstr_loop.heartbeat"
    monkeypatch.setattr(config, "HEARTBEAT_FILE", str(hb_path))

    loop._write_heartbeat(_bot_state_with(), cycle_count=42)

    assert hb_path.exists()
    payload = json.loads(hb_path.read_text())
    # Required by loop_health.read_loop_status:
    assert isinstance(payload["ts"], str)
    datetime.datetime.fromisoformat(payload["ts"].replace("Z", "+00:00"))
    # Operator-facing fields:
    assert payload["status"] == "ok"
    assert payload["cycle"] == 42
    assert payload["ok"] is True
    assert payload["phase"] == config.PHASE
    assert payload["n_positions"] == 0


def test_heartbeat_reports_position_count(tmp_path, monkeypatch):
    hb_path = tmp_path / "mstr_loop.heartbeat"
    monkeypatch.setattr(config, "HEARTBEAT_FILE", str(hb_path))

    bs = _bot_state_with(positions={"momentum_rider": object(), "mean_reversion": object()})
    loop._write_heartbeat(bs, cycle_count=1)

    payload = json.loads(hb_path.read_text())
    assert payload["n_positions"] == 2


def test_heartbeat_write_failure_does_not_raise(tmp_path, monkeypatch):
    """Heartbeat is best-effort telemetry — must never propagate exceptions."""
    monkeypatch.setattr(config, "HEARTBEAT_FILE",
                         str(tmp_path / "mstr_loop.heartbeat"))

    def _boom(*_a, **_kw):
        raise OSError("disk full (simulated)")

    # The helper imports atomic_write_json lazily inside the function body,
    # so patch it on the source module (portfolio.file_utils).
    from portfolio import file_utils
    monkeypatch.setattr(file_utils, "atomic_write_json", _boom)

    # Must NOT raise — failure path is logged at debug level and swallowed.
    loop._write_heartbeat(_bot_state_with(), cycle_count=99)


def test_heartbeat_consumed_by_loop_health(tmp_path, monkeypatch):
    """End-to-end: heartbeat written by mstr_loop is read as 'fresh' by loop_health."""
    hb_path = tmp_path / "data" / "mstr_loop.heartbeat"
    hb_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "HEARTBEAT_FILE", str(hb_path))

    loop._write_heartbeat(_bot_state_with(), cycle_count=7)

    rollup = loop_health.read_loop_health(
        repo_root=tmp_path,
        files={"mstr": "data/mstr_loop.heartbeat"},
    )
    assert rollup["any_unhealthy"] is False
    assert rollup["loops"]["mstr"]["state"] == "fresh"
    assert rollup["loops"]["mstr"]["payload"]["cycle"] == 7


def test_default_loop_health_includes_mstr():
    """Sanity: mstr is wired into the default rollup so the watchdog sees it."""
    assert "mstr" in loop_health.DEFAULT_HEARTBEAT_FILES
    assert loop_health.DEFAULT_HEARTBEAT_FILES["mstr"] == "data/mstr_loop.heartbeat"
