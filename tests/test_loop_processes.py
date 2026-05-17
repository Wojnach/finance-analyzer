"""Tests for portfolio.loop_processes scan logic.

The function is psutil-shaped but the tests monkey-patch
_iter_processes so they run identically on Linux CI and Windows prod.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from portfolio import loop_processes


def _fake_proc(pid: int, cmdline: list[str], create_time: float = 1_700_000_000.0):
    """Shape returned by loop_processes._iter_processes()."""
    return {"pid": pid, "name": "python.exe", "cmdline": cmdline, "create_time": create_time}


def _patch_processes(monkeypatch: pytest.MonkeyPatch, procs):
    monkeypatch.setattr(loop_processes, "_iter_processes", lambda: procs)


def test_scan_empty_returns_all_loops_with_count_zero(monkeypatch):
    _patch_processes(monkeypatch, [])
    payload = loop_processes.scan(now_utc=datetime(2026, 5, 18, 12, 0, tzinfo=UTC))
    assert payload["any_duplicate"] is False
    assert payload["checked_at"] == "2026-05-18T12:00:00+00:00"
    names = [L["name"] for L in payload["loops"]]
    # Every KNOWN_LOOPS key must appear, even with count=0 — that's how
    # the tile renders missing loops as red ("expected, not running").
    for n in loop_processes.KNOWN_LOOPS:
        assert n in names
    for L in payload["loops"]:
        assert L["count"] == 0
        assert L["duplicate"] is False
        assert L["pids"] == []


def test_scan_finds_single_match(monkeypatch):
    procs = [
        _fake_proc(123, ["C:\\py.exe", "-u", "Q:\\finance-analyzer\\portfolio\\main.py", "--loop"]),
    ]
    _patch_processes(monkeypatch, procs)
    payload = loop_processes.scan(now_utc=datetime(2026, 5, 18, 12, 0, tzinfo=UTC))
    main = next(L for L in payload["loops"] if L["name"] == "main")
    assert main["count"] == 1
    assert main["pids"] == [123]
    assert main["duplicate"] is False
    assert payload["any_duplicate"] is False


def test_scan_detects_duplicate(monkeypatch):
    procs = [
        _fake_proc(123, ["py", "-u", "data/metals_loop.py"]),
        _fake_proc(124, ["py", "-u", "data\\metals_loop.py"]),  # backslash form
    ]
    _patch_processes(monkeypatch, procs)
    payload = loop_processes.scan()
    metals = next(L for L in payload["loops"] if L["name"] == "metals")
    assert metals["count"] == 2
    assert set(metals["pids"]) == {123, 124}
    assert metals["duplicate"] is True
    assert payload["any_duplicate"] is True


def test_scan_path_separator_normalisation(monkeypatch):
    """KNOWN_LOOPS uses forward slashes; cmdline may contain either."""
    procs = [
        _fake_proc(11, ["python", "data\\crypto_loop.py", "--loop"]),
    ]
    _patch_processes(monkeypatch, procs)
    payload = loop_processes.scan()
    crypto = next(L for L in payload["loops"] if L["name"] == "crypto")
    assert crypto["count"] == 1


def test_scan_omits_own_pid(monkeypatch):
    """Dashboard process must not self-match `dashboard/app.py`."""
    import os
    own = os.getpid()
    procs = [
        _fake_proc(own, ["py", "dashboard/app.py"]),  # this is us
        _fake_proc(999, ["py", "dashboard/app.py"]),  # the real dashboard
    ]
    _patch_processes(monkeypatch, procs)
    payload = loop_processes.scan()
    dash = next(L for L in payload["loops"] if L["name"] == "dashboard")
    # Own pid excluded → count is 1, not 2 → not a false-positive
    # duplicate.
    assert own not in dash["pids"]
    assert dash["count"] == 1
    assert dash["duplicate"] is False


def test_scan_uptime_uses_oldest_create_time(monkeypatch):
    """When multiple PIDs match, oldest_uptime_seconds reflects the
    oldest process so the user sees "this loop has been running 2h"
    even if a duplicate fork is only 30s old."""
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    procs = [
        # 2h old
        _fake_proc(1, ["py", "data/oil_loop.py"], create_time=now.timestamp() - 7200),
        # 30s old (the orphan from premortem N5)
        _fake_proc(2, ["py", "data/oil_loop.py"], create_time=now.timestamp() - 30),
    ]
    _patch_processes(monkeypatch, procs)
    payload = loop_processes.scan(now_utc=now)
    oil = next(L for L in payload["loops"] if L["name"] == "oil")
    assert oil["oldest_uptime_seconds"] == 7200
    assert oil["duplicate"] is True


def test_scan_ignores_empty_cmdline(monkeypatch):
    """psutil sometimes returns [] for a process we can't read; that
    must not crash and must not match any pattern."""
    _patch_processes(monkeypatch, [_fake_proc(42, [])])
    payload = loop_processes.scan()
    assert payload["any_duplicate"] is False
    for L in payload["loops"]:
        assert L["count"] == 0
