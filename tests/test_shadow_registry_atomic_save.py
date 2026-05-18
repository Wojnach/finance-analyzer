"""Pin `portfolio.shadow_registry.save_registry` to atomic_write_json.

Premortem N2 mitigation from the 2026-05-18 plan
(`.worktrees/shadow-gate-lora-20260518/docs/PLAN.md`): the registry
mutation path must go through `file_utils.atomic_write_json` so that
a half-written file cannot be loaded by the 03:30 promotion cron
mid-write and silently drop voters.

This test guards the contract: if a future refactor swaps to a plain
`json.dump`/`open(...).write()` the test fails immediately.
"""
from __future__ import annotations

import json
from pathlib import Path


def test_save_registry_routes_through_atomic_write_json(monkeypatch, tmp_path):
    """Patch `atomic_write_json` and confirm save_registry invokes it."""
    from portfolio import file_utils
    from portfolio import shadow_registry as sr

    calls = []
    real_atomic = file_utils.atomic_write_json

    def spy_atomic(path, data):
        calls.append((str(path), data))
        return real_atomic(path, data)

    monkeypatch.setattr(sr, "atomic_write_json", spy_atomic)

    target = tmp_path / "shadow_registry.json"
    payload = {"shadows": {"probe_signal": {"status": "shadow"}}}
    sr.save_registry(payload, path=target)

    assert calls, "save_registry must call atomic_write_json"
    assert calls[0][0] == str(target)
    assert calls[0][1] == payload
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_save_registry_uses_tempfile_rename(tmp_path):
    """End-to-end: save twice and confirm the file is replaced atomically.

    We can't directly inspect the rename system call from Python, so instead
    verify the empirical guarantee: between writes there is never a state where
    the file exists but is unreadable as JSON. We test by writing under
    contention is overkill; the file-utils unit tests already cover that. Here
    we just confirm the writer succeeds on a path that already exists.
    """
    from portfolio import shadow_registry as sr

    target = tmp_path / "shadow_registry.json"
    sr.save_registry({"shadows": {"a": {"status": "shadow"}}}, path=target)
    assert target.exists()
    sr.save_registry({"shadows": {"b": {"status": "promoted"}}}, path=target)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["shadows"] == {"b": {"status": "promoted"}}


def test_atomic_write_json_import_present_in_shadow_registry():
    """The atomic_write_json import in shadow_registry.py is critical for
    safety. If a refactor removes it (and silently switches to plain json),
    this test catches it before merge."""
    from portfolio import shadow_registry as sr
    # The function must be bound at module level for save_registry to use it.
    assert getattr(sr, "atomic_write_json", None) is not None
    # And callable (not just a string reference).
    assert callable(sr.atomic_write_json)
