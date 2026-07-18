"""Tests for portfolio.loop_processes scan logic.

The function is psutil-shaped but the tests monkey-patch
_iter_processes so they run identically on Linux CI and Windows prod.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from portfolio import loop_processes


def _fake_proc(pid: int, cmdline: list[str], create_time: float = 1_700_000_000.0, ppid: int = 0):
    """Shape returned by loop_processes._iter_processes()."""
    return {"pid": pid, "name": "python.exe", "cmdline": cmdline, "create_time": create_time, "ppid": ppid}


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


def test_scan_collapses_venv_shim_child_pair(monkeypatch):
    """Windows venv python.exe re-execs the base interpreter, so the
    same loop script appears in TWO processes with identical argv tails:
    the .venv shim (parent) and the Python3xx child. They are ONE logical
    loop, not a duplicate. The child's ppid == the shim's pid; the
    scanner must drop the shim and report count=1.

    Regression: every loop on the live box was flagged duplicate
    2026-06-05 because both halves of the pair matched the substring.
    """
    procs = [
        # .venv shim (parent) — spawned the real interpreter
        _fake_proc(16708, [".venv\\Scripts\\python.exe", "-u", "portfolio\\main.py", "--loop"]),
        # base-interpreter child — its parent IS the shim above
        _fake_proc(16724, ["C:\\Python312\\python.exe", "-u", "portfolio\\main.py", "--loop"], ppid=16708),
    ]
    _patch_processes(monkeypatch, procs)
    payload = loop_processes.scan()
    main = next(L for L in payload["loops"] if L["name"] == "main")
    assert main["count"] == 1
    assert main["pids"] == [16724]  # leaf interpreter, not the shim
    assert main["duplicate"] is False
    assert payload["any_duplicate"] is False


def test_scan_true_duplicate_of_shim_child_pairs(monkeypatch):
    """Two independent shim+child pairs = a REAL duplicate loop. After
    collapsing each pair to its child, two leaf interpreters remain →
    duplicate must still fire."""
    procs = [
        _fake_proc(100, [".venv\\Scripts\\python.exe", "-u", "data\\metals_loop.py"]),
        _fake_proc(101, ["C:\\Python312\\python.exe", "-u", "data\\metals_loop.py"], ppid=100),
        _fake_proc(200, [".venv\\Scripts\\python.exe", "-u", "data\\metals_loop.py"]),
        _fake_proc(201, ["C:\\Python312\\python.exe", "-u", "data\\metals_loop.py"], ppid=200),
    ]
    _patch_processes(monkeypatch, procs)
    payload = loop_processes.scan()
    metals = next(L for L in payload["loops"] if L["name"] == "metals")
    assert metals["count"] == 2
    assert set(metals["pids"]) == {101, 201}
    assert metals["duplicate"] is True


def test_scan_matches_powershell_hw_monitor(monkeypatch):
    """hw monitoring runs as a PowerShell script (read_temps.ps1) under
    wscript/cmd, not a python module. The pattern must match the .ps1
    path so the tile shows it green when PF-HWMonitor is live."""
    procs = [
        _fake_proc(
            777,
            ["powershell.exe", "-File", "Q:\\finance-analyzer\\data\\read_temps.ps1"],
        ),
    ]
    _patch_processes(monkeypatch, procs)
    payload = loop_processes.scan()
    hw = next(L for L in payload["loops"] if L["name"] == "hw_monitor")
    assert hw["count"] == 1
    assert hw["duplicate"] is False


def test_scan_matches_module_launched_dashboard(monkeypatch):
    """Dashboard launches via `-m dashboard.app`; the pattern must match
    the module token, not the legacy file path."""
    procs = [
        _fake_proc(888, ["C:\\py.exe", "-m", "dashboard.app"]),
    ]
    _patch_processes(monkeypatch, procs)
    payload = loop_processes.scan()
    dash = next(L for L in payload["loops"] if L["name"] == "dashboard")
    assert dash["count"] == 1


def test_scan_matches_deck_module_launched_main(monkeypatch):
    """On the Deck, pf-dataloop.service launches main via
    `-m portfolio.main --loop` (module form), not the Windows
    `portfolio/main.py --loop` file-path form. Both must match
    (2026-07-18: the module form never matched, so the live Deck loop
    showed count=0 while pf-dataloop was running)."""
    procs = [
        _fake_proc(
            555,
            [
                "/home/deck/projects/finance-analyzer/.venv/bin/python",
                "-u",
                "-m",
                "portfolio.main",
                "--loop",
            ],
        ),
    ]
    _patch_processes(monkeypatch, procs)
    payload = loop_processes.scan()
    main = next(L for L in payload["loops"] if L["name"] == "main")
    assert main["count"] == 1
    assert main["pids"] == [555]


def test_scan_matches_deck_file_launched_dashboard(monkeypatch):
    """On the Deck, pf-dashboard.service launches the dashboard as a
    bare script (`dashboard/app.py`), not the Windows `-m dashboard.app`
    module form. Both must match (2026-07-18 regression: the
    module-only pattern never matched the live Deck process)."""
    procs = [
        _fake_proc(
            556,
            [
                "/home/deck/projects/finance-analyzer/.venv/bin/python",
                "dashboard/app.py",
            ],
        ),
    ]
    _patch_processes(monkeypatch, procs)
    payload = loop_processes.scan()
    dash = next(L for L in payload["loops"] if L["name"] == "dashboard")
    assert dash["count"] == 1
    assert dash["pids"] == [556]


def test_log_rotation_not_process_checked(monkeypatch):
    """log_rotation is NOT process-checked (removed 2026-07-18) — it
    runs inline inside the main loop's cycle (portfolio/main.py ->
    rotate_all()), never as a standalone process on either OS. Same
    rationale as telegram_poller above; a permanent entry here could
    only ever show a false grey row."""
    _patch_processes(monkeypatch, [])
    payload = loop_processes.scan()
    assert "log_rotation" not in [L["name"] for L in payload["loops"]]
    assert "log_rotation" not in loop_processes.KNOWN_LOOPS


def test_telegram_poller_not_process_checked(monkeypatch):
    """telegram_poller is a daemon thread inside main.py, never a
    standalone process — it must not appear as a tile row."""
    _patch_processes(monkeypatch, [])
    payload = loop_processes.scan()
    assert "telegram_poller" not in [L["name"] for L in payload["loops"]]


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
        _fake_proc(own, ["py", "-m", "dashboard.app"]),  # this is us
        _fake_proc(999, ["py", "-m", "dashboard.app"]),  # the real dashboard
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


def test_scan_handles_zero_create_time_without_crashing(monkeypatch):
    """If psutil returns create_time=0 for all matched processes
    (AccessDenied fallback), the scanner must NOT crash on min() of
    an empty generator. Regression test for cavecrew-reviewer P1."""
    procs = [
        _fake_proc(123, ["py", "data/metals_loop.py"], create_time=0.0),
        _fake_proc(124, ["py", "data/metals_loop.py"], create_time=0.0),
    ]
    _patch_processes(monkeypatch, procs)
    payload = loop_processes.scan()  # must not raise
    metals = next(L for L in payload["loops"] if L["name"] == "metals")
    # Duplicate detection still works without create_time
    assert metals["count"] == 2
    assert metals["duplicate"] is True
    # But uptime is unknown (None) instead of crashing
    assert metals["oldest_uptime_seconds"] is None
