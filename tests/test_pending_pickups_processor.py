"""Tests for `scripts/process_pending_pickups.py`.

Covers:
* due-filtering (pickup with future due_ts is skipped)
* status="completed" gating (completed pickups do not re-run)
* `--force ID` bypasses both due and status
* handler whitelist (an unknown handler returns verdict=error without
  importing arbitrary code -- regression for the CWE-706 fix)
* atomic write to `data/pending_pickups.json` after a run
* exit code 1 when a handler returns verdict=error
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path


def _import_processor():
    import scripts.process_pending_pickups as proc
    return importlib.reload(proc)


def _write_pickups(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pickups": rows}), encoding="utf-8")


def _stub_handler(returns):
    """Build an in-memory handler module that returns a fixed dict."""
    mod = types.ModuleType("scripts.pickups._stub_for_test")
    mod.run = lambda pickup, repo_root: returns  # type: ignore[attr-defined]
    return mod


def test_future_due_not_processed(monkeypatch, tmp_path):
    proc = _import_processor()
    monkeypatch.setattr(proc, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(proc, "_PICKUPS_PATH", tmp_path / "data" / "pending_pickups.json")
    monkeypatch.setattr(proc, "_SESSION_PROGRESS", tmp_path / "docs" / "SESSION_PROGRESS.md")
    monkeypatch.setattr(proc, "_send_telegram", lambda lines: None)
    _write_pickups(
        proc._PICKUPS_PATH,
        [{
            "id": "TEST-FUTURE",
            "title": "due later",
            "due_ts": "2099-01-01T00:00:00+00:00",
            "handler": "llm_cryptotrader_72h",
            "status": "pending",
        }],
    )
    rc = proc.main([])
    assert rc == 0
    data = json.loads(proc._PICKUPS_PATH.read_text(encoding="utf-8"))
    assert data["pickups"][0]["status"] == "pending"
    assert "history" not in data["pickups"][0] or data["pickups"][0]["history"] == []


def test_completed_status_not_rerun(monkeypatch, tmp_path):
    proc = _import_processor()
    monkeypatch.setattr(proc, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(proc, "_PICKUPS_PATH", tmp_path / "data" / "pending_pickups.json")
    monkeypatch.setattr(proc, "_SESSION_PROGRESS", tmp_path / "docs" / "SESSION_PROGRESS.md")
    monkeypatch.setattr(proc, "_send_telegram", lambda lines: None)
    monkeypatch.setattr(proc, "_HANDLERS", {"llm_cryptotrader_72h": _stub_handler({
        "verdict": "promote", "summary": "stub", "details": {}, "telegram_lines": [],
    })})
    _write_pickups(
        proc._PICKUPS_PATH,
        [{
            "id": "TEST-DONE",
            "due_ts": "1999-01-01T00:00:00+00:00",  # past
            "handler": "llm_cryptotrader_72h",
            "status": "completed",
            "history": [{"verdict": "promote"}],
        }],
    )
    rc = proc.main([])
    assert rc == 0
    data = json.loads(proc._PICKUPS_PATH.read_text(encoding="utf-8"))
    # history length unchanged
    assert len(data["pickups"][0]["history"]) == 1


def test_due_pickup_dispatched(monkeypatch, tmp_path):
    proc = _import_processor()
    monkeypatch.setattr(proc, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(proc, "_PICKUPS_PATH", tmp_path / "data" / "pending_pickups.json")
    monkeypatch.setattr(proc, "_SESSION_PROGRESS", tmp_path / "docs" / "SESSION_PROGRESS.md")
    monkeypatch.setattr(proc, "_send_telegram", lambda lines: None)
    monkeypatch.setattr(proc, "_HANDLERS", {"llm_cryptotrader_72h": _stub_handler({
        "verdict": "defer",
        "summary": "not enough data",
        "details": {"n": 0},
        "telegram_lines": ["test line"],
    })})
    _write_pickups(
        proc._PICKUPS_PATH,
        [{
            "id": "TEST-DUE",
            "title": "due now",
            "due_ts": "1999-01-01T00:00:00+00:00",
            "handler": "llm_cryptotrader_72h",
            "status": "pending",
        }],
    )
    rc = proc.main([])
    assert rc == 0
    data = json.loads(proc._PICKUPS_PATH.read_text(encoding="utf-8"))
    p = data["pickups"][0]
    assert p["status"] == "completed"
    assert len(p["history"]) == 1
    assert p["history"][0]["verdict"] == "defer"
    assert p["last_run_ts"]


def test_unknown_handler_returns_error_not_import(monkeypatch, tmp_path):
    """Regression for the CWE-706 semgrep finding: arbitrary handler names
    in the JSON file must NOT trigger an importlib import. They must be
    rejected via the _HANDLERS whitelist."""
    proc = _import_processor()
    monkeypatch.setattr(proc, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(proc, "_PICKUPS_PATH", tmp_path / "data" / "pending_pickups.json")
    monkeypatch.setattr(proc, "_SESSION_PROGRESS", tmp_path / "docs" / "SESSION_PROGRESS.md")
    monkeypatch.setattr(proc, "_send_telegram", lambda lines: None)
    # Even if a malicious actor names a real module:
    _write_pickups(
        proc._PICKUPS_PATH,
        [{
            "id": "TEST-EVIL",
            "title": "would import os",
            "due_ts": "1999-01-01T00:00:00+00:00",
            "handler": "os",
            "status": "pending",
        }],
    )
    rc = proc.main([])
    # error verdict -> exit 1
    assert rc == 1
    data = json.loads(proc._PICKUPS_PATH.read_text(encoding="utf-8"))
    p = data["pickups"][0]
    # 2026-06-10 bounded retry: a single failure no longer parks the pickup
    # at status=error — it stays pending with an attempts counter.
    assert p["status"] == "pending"
    assert p["attempts"] == 1
    assert p["history"][0]["verdict"] == "error"
    assert "whitelist" in p["history"][0]["summary"].lower()


def test_force_id_bypasses_due_and_status(monkeypatch, tmp_path):
    proc = _import_processor()
    monkeypatch.setattr(proc, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(proc, "_PICKUPS_PATH", tmp_path / "data" / "pending_pickups.json")
    monkeypatch.setattr(proc, "_SESSION_PROGRESS", tmp_path / "docs" / "SESSION_PROGRESS.md")
    monkeypatch.setattr(proc, "_send_telegram", lambda lines: None)
    monkeypatch.setattr(proc, "_HANDLERS", {"llm_cryptotrader_72h": _stub_handler({
        "verdict": "promote", "summary": "forced", "details": {}, "telegram_lines": [],
    })})
    _write_pickups(
        proc._PICKUPS_PATH,
        [{
            "id": "TEST-FORCED",
            "due_ts": "2099-01-01T00:00:00+00:00",  # future
            "handler": "llm_cryptotrader_72h",
            "status": "completed",  # already done
            "history": [],
        }],
    )
    rc = proc.main(["--force", "TEST-FORCED"])
    assert rc == 0
    data = json.loads(proc._PICKUPS_PATH.read_text(encoding="utf-8"))
    p = data["pickups"][0]
    assert p["history"][-1]["verdict"] == "promote"


def test_dry_run_does_not_mutate(monkeypatch, tmp_path):
    proc = _import_processor()
    monkeypatch.setattr(proc, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(proc, "_PICKUPS_PATH", tmp_path / "data" / "pending_pickups.json")
    monkeypatch.setattr(proc, "_SESSION_PROGRESS", tmp_path / "docs" / "SESSION_PROGRESS.md")
    monkeypatch.setattr(proc, "_send_telegram", lambda lines: None)
    monkeypatch.setattr(proc, "_HANDLERS", {"llm_cryptotrader_72h": _stub_handler({
        "verdict": "promote", "summary": "x", "details": {}, "telegram_lines": [],
    })})
    _write_pickups(
        proc._PICKUPS_PATH,
        [{
            "id": "TEST-DRY",
            "due_ts": "1999-01-01T00:00:00+00:00",
            "handler": "llm_cryptotrader_72h",
            "status": "pending",
        }],
    )
    original = proc._PICKUPS_PATH.read_text(encoding="utf-8")
    rc = proc.main(["--dry-run"])
    assert rc == 0
    assert proc._PICKUPS_PATH.read_text(encoding="utf-8") == original


def test_force_unknown_id_exits_2(monkeypatch, tmp_path, capsys):
    """2026-06-10: --force with a typo'd id used to print the routine
    'no due pickups' and exit 0 — a success-looking no-op on the documented
    manual recovery path. Must now print an error listing valid ids and
    exit 2."""
    proc = _import_processor()
    monkeypatch.setattr(proc, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(proc, "_PICKUPS_PATH", tmp_path / "data" / "pending_pickups.json")
    monkeypatch.setattr(proc, "_SESSION_PROGRESS", tmp_path / "docs" / "SESSION_PROGRESS.md")
    monkeypatch.setattr(proc, "_send_telegram", lambda lines: None)
    _write_pickups(
        proc._PICKUPS_PATH,
        [{
            "id": "REAL-ID",
            "due_ts": "2099-01-01T00:00:00+00:00",
            "handler": "llm_cryptotrader_72h",
            "status": "pending",
        }],
    )
    rc = proc.main(["--force", "TYPO-ID"])
    assert rc == 2
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "TYPO-ID" in out
    assert "REAL-ID" in out


def test_error_retries_then_parks_with_critical_entry(monkeypatch, tmp_path):
    """2026-06-10 bounded retry: a failing handler keeps the pickup pending
    (with an attempts counter) for the first _MAX_ATTEMPTS-1 failures, then
    parks it at status=error and appends a pickup_failed critical entry."""
    proc = _import_processor()
    monkeypatch.setattr(proc, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(proc, "_PICKUPS_PATH", tmp_path / "data" / "pending_pickups.json")
    monkeypatch.setattr(proc, "_SESSION_PROGRESS", tmp_path / "docs" / "SESSION_PROGRESS.md")
    monkeypatch.setattr(proc, "_send_telegram", lambda lines: None)
    monkeypatch.setattr(proc, "_HANDLERS", {"llm_cryptotrader_72h": _stub_handler({
        "verdict": "error", "summary": "transient failure",
        "details": {}, "telegram_lines": [],
    })})
    # Redirect the critical-errors journal that record_critical_error writes.
    crit_file = tmp_path / "critical_errors.jsonl"
    import portfolio.claude_gate as claude_gate
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", crit_file)
    _write_pickups(
        proc._PICKUPS_PATH,
        [{
            "id": "TEST-RETRY",
            "due_ts": "1999-01-01T00:00:00+00:00",
            "handler": "llm_cryptotrader_72h",
            "status": "pending",
        }],
    )

    # Attempts 1 and 2: stays pending, no critical entry.
    for expected_attempts in (1, 2):
        rc = proc.main([])
        assert rc == 1
        data = json.loads(proc._PICKUPS_PATH.read_text(encoding="utf-8"))
        p = data["pickups"][0]
        assert p["status"] == "pending"
        assert p["attempts"] == expected_attempts
        assert not crit_file.exists()

    # Attempt 3: parked at error + pickup_failed critical entry.
    rc = proc.main([])
    assert rc == 1
    data = json.loads(proc._PICKUPS_PATH.read_text(encoding="utf-8"))
    p = data["pickups"][0]
    assert p["status"] == "error"
    assert p["attempts"] == 3
    assert len(p["history"]) == 3
    rows = [json.loads(line) for line in
            crit_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["category"] == "pickup_failed"
    assert rows[0]["level"] == "critical"
    assert rows[0]["context"]["pickup_id"] == "TEST-RETRY"

    # And the parked pickup is no longer retried.
    rc = proc.main([])
    assert rc == 0
    data = json.loads(proc._PICKUPS_PATH.read_text(encoding="utf-8"))
    assert len(data["pickups"][0]["history"]) == 3


def test_raising_handler_contained_as_error(monkeypatch, tmp_path):
    """2026-06-10: a handler exception used to propagate out of main() —
    the script crashed and _save_pickups never ran. Must now record an
    error verdict and apply the bounded-retry path."""
    proc = _import_processor()
    monkeypatch.setattr(proc, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(proc, "_PICKUPS_PATH", tmp_path / "data" / "pending_pickups.json")
    monkeypatch.setattr(proc, "_SESSION_PROGRESS", tmp_path / "docs" / "SESSION_PROGRESS.md")
    monkeypatch.setattr(proc, "_send_telegram", lambda lines: None)
    boom = types.ModuleType("scripts.pickups._raises_for_test")
    def _raise(pickup, repo_root):
        raise RuntimeError("data file briefly missing")
    boom.run = _raise  # type: ignore[attr-defined]
    monkeypatch.setattr(proc, "_HANDLERS", {"llm_cryptotrader_72h": boom})
    _write_pickups(
        proc._PICKUPS_PATH,
        [{
            "id": "TEST-RAISES",
            "due_ts": "1999-01-01T00:00:00+00:00",
            "handler": "llm_cryptotrader_72h",
            "status": "pending",
        }],
    )
    rc = proc.main([])
    assert rc == 1
    data = json.loads(proc._PICKUPS_PATH.read_text(encoding="utf-8"))
    p = data["pickups"][0]
    assert p["status"] == "pending"
    assert p["attempts"] == 1
    assert "RuntimeError" in p["history"][0]["summary"]
