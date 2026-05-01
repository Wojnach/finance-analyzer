"""Tests for the loop-hardening additions to crypto_loop and oil_loop.

Verifies:
  - run_loop returns EXIT_LOCK_CONFLICT (11) when another instance holds
    the singleton lock.
  - main() propagates that exit code so the .bat wrapper can stop
    fork-bombing.
  - write_heartbeat persists JSON without raising.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pytest

# data/ is script-style.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import crypto_loop
import oil_loop


# ---------------------------------------------------------------------------
# Exit code 11 — both loops
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("loop_module", [crypto_loop, oil_loop])
def test_run_loop_returns_exit_lock_conflict_when_lock_held(loop_module, monkeypatch):
    """run_loop must return EXIT_LOCK_CONFLICT (11) instead of None/0 when
    acquire_singleton_lock returns None — otherwise the .bat wrapper
    cannot detect the duplicate-instance case and will respawn forever."""
    monkeypatch.setattr(loop_module, "acquire_singleton_lock", lambda *_a, **_k: None)
    rc = loop_module.run_loop()
    assert rc == 11
    assert rc == loop_module.EXIT_LOCK_CONFLICT


@pytest.mark.parametrize("loop_module", [crypto_loop, oil_loop])
def test_main_propagates_exit_lock_conflict(loop_module, monkeypatch):
    """When --loop hits the lock-conflict path, main() must return 11."""
    monkeypatch.setattr(sys, "argv", [loop_module.__name__, "--loop"])
    monkeypatch.setattr(loop_module, "acquire_singleton_lock", lambda *_a, **_k: None)
    rc = loop_module.main()
    assert rc == 11


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("loop_module", [crypto_loop, oil_loop])
def test_write_heartbeat_creates_file_with_timestamp(tmp_path, monkeypatch,
                                                      loop_module):
    hb_file = tmp_path / "loop.heartbeat"
    monkeypatch.setattr(loop_module, "HEARTBEAT_FILE", str(hb_file))
    loop_module.write_heartbeat({"cycle": 5, "ok": True})
    assert hb_file.exists()
    payload = json.loads(hb_file.read_text())
    assert "ts" in payload
    assert payload["status"] == "ok"
    assert payload["cycle"] == 5


@pytest.mark.parametrize("loop_module", [crypto_loop, oil_loop])
def test_write_heartbeat_swallows_io_error(tmp_path, monkeypatch, loop_module):
    """Heartbeat write failure must NEVER crash the loop."""
    monkeypatch.setattr(loop_module, "HEARTBEAT_FILE", "/no/such/dir/hb.json")
    # Should not raise even when atomic_write fails internally.
    loop_module.write_heartbeat({"cycle": 1})


# ---------------------------------------------------------------------------
# Telegram notify wiring — no exception when config absent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("loop_module", [crypto_loop, oil_loop])
def test_main_loop_does_not_crash_when_config_missing(loop_module, monkeypatch):
    """If config.json doesn't exist or has no telegram token, main() must
    still proceed (notify falls back to None)."""
    monkeypatch.setattr(sys, "argv", [loop_module.__name__, "--loop"])
    # Force run_loop to return immediately with the lock-conflict path so
    # we don't actually enter the cycle.
    monkeypatch.setattr(loop_module, "acquire_singleton_lock", lambda *_a, **_k: None)
    # Even if loading the canonical config blows up, main must not raise.
    with patch("portfolio.file_utils.load_json", side_effect=RuntimeError("no config")):
        rc = loop_module.main()
    assert rc == 11
