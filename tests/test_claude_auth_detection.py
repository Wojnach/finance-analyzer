"""Tests for the 2026-04-13 claude auth-failure detection.

Covers ``detect_auth_failure``, ``record_critical_error``, and the wiring
into ``invoke_claude`` / ``invoke_claude_text``. Root cause context is in
``docs/plans/2026-04-13-claude-auth-detection.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from portfolio import claude_gate
from portfolio.claude_gate import (
    _AUTH_ERROR_MARKERS,
    detect_auth_failure,
    record_critical_error,
)


# ---------------------------------------------------------------------------
# detect_auth_failure — pure function
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("marker", _AUTH_ERROR_MARKERS)
def test_detect_auth_failure_matches_each_marker(marker, monkeypatch, tmp_path):
    """Every marker in _AUTH_ERROR_MARKERS must trigger detection."""
    # Redirect the critical-errors journal so we don't pollute data/.
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", tmp_path / "crit.jsonl")

    output = f"some preamble\n{marker}\ntrailing noise\n"
    assert detect_auth_failure(output, caller="test") is True


def test_detect_auth_failure_benign_output(monkeypatch, tmp_path):
    """Normal claude output without the markers must NOT trigger detection."""
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", tmp_path / "crit.jsonl")

    output = "Here is my analysis of BTC-USD:\n- RSI 52\n- MACD +0.3\nRecommendation: HOLD"
    assert detect_auth_failure(output, caller="test") is False


def test_detect_auth_failure_empty_output(monkeypatch, tmp_path):
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", tmp_path / "crit.jsonl")
    assert detect_auth_failure("", caller="test") is False
    assert detect_auth_failure(None, caller="test") is False  # type: ignore[arg-type]


def test_detect_auth_failure_writes_to_critical_errors_journal(monkeypatch, tmp_path):
    """On detection, an entry must be appended to critical_errors.jsonl."""
    journal = tmp_path / "crit.jsonl"
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", journal)

    detect_auth_failure("Not logged in — Please run /login", caller="layer2_t3",
                        context={"tier": 3, "exit_code": 0})

    assert journal.exists()
    lines = journal.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["level"] == "critical"
    assert entry["category"] == "auth_failure"
    assert entry["caller"] == "layer2_t3"
    assert entry["resolution"] is None
    assert "Not logged in" in entry["context"]["marker"]
    assert entry["context"]["tier"] == 3


# ---------------------------------------------------------------------------
# record_critical_error — pure function
# ---------------------------------------------------------------------------

def test_record_critical_error_appends(monkeypatch, tmp_path):
    """Two calls should append, not overwrite."""
    journal = tmp_path / "crit.jsonl"
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", journal)

    record_critical_error("auth_failure", "caller-a", "first", {"k": 1})
    record_critical_error("contract_violation", "caller-b", "second", {"k": 2})

    lines = journal.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["category"] == "auth_failure"
    assert second["category"] == "contract_violation"
    assert first["context"]["k"] == 1
    assert second["context"]["k"] == 2


def test_record_critical_error_never_raises(monkeypatch):
    """File-system failures must NOT propagate — callers can't handle them."""
    class BoomPath:
        def __truediv__(self, other):
            return self

        def __fspath__(self):
            raise OSError("simulated")

    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", BoomPath())
    # Must not raise
    record_critical_error("auth_failure", "x", "y")


# ---------------------------------------------------------------------------
# invoke_claude wiring — auth-error overrides exit_code=0 success
# ---------------------------------------------------------------------------

def test_invoke_claude_overrides_exit0_on_auth_failure(monkeypatch, tmp_path):
    """The exact regression guard: exit_code=0 + 'Not logged in' in stdout
    must come back as (False, non-zero) with status='auth_error'."""
    journal = tmp_path / "crit.jsonl"
    inv_log = tmp_path / "inv.jsonl"
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", journal)
    monkeypatch.setattr(claude_gate, "INVOCATIONS_LOG", inv_log)
    monkeypatch.setattr(claude_gate, "_find_claude_cmd", lambda: "claude")
    monkeypatch.setattr(claude_gate, "_load_config_layer2_enabled", lambda: True)

    def fake_run(cmd, *, timeout, env, cwd, label):
        return 0, "Not logged in\nPlease run /login\n", "", False

    monkeypatch.setattr(claude_gate, "_run_with_tree_kill", fake_run)

    success, exit_code = claude_gate.invoke_claude(
        prompt="hi", caller="test", model="sonnet", max_turns=1, timeout=5,
    )

    assert success is False, "auth failure must not look successful"
    assert exit_code != 0, "must not return the claude-CLI's fake exit-0"

    # Invocation log should record status='auth_error'.
    inv_lines = inv_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(inv_lines) == 1
    assert json.loads(inv_lines[0])["status"] == "auth_error"

    # Critical errors journal should have an auth_failure entry.
    crit_lines = journal.read_text(encoding="utf-8").strip().splitlines()
    assert len(crit_lines) == 1
    assert json.loads(crit_lines[0])["category"] == "auth_failure"


def test_invoke_claude_success_still_works(monkeypatch, tmp_path):
    """Benign 0-exit must still return (True, 0) — no false positives."""
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", tmp_path / "crit.jsonl")
    monkeypatch.setattr(claude_gate, "INVOCATIONS_LOG", tmp_path / "inv.jsonl")
    monkeypatch.setattr(claude_gate, "_find_claude_cmd", lambda: "claude")
    monkeypatch.setattr(claude_gate, "_load_config_layer2_enabled", lambda: True)

    def fake_run(cmd, *, timeout, env, cwd, label):
        return 0, "analysis complete", "", False

    monkeypatch.setattr(claude_gate, "_run_with_tree_kill", fake_run)

    success, exit_code = claude_gate.invoke_claude(
        prompt="hi", caller="test", model="sonnet", max_turns=1, timeout=5,
    )
    assert success is True
    assert exit_code == 0


# ---------------------------------------------------------------------------
# invoke_claude_text wiring
# ---------------------------------------------------------------------------

def test_invoke_claude_text_overrides_exit0_on_auth_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", tmp_path / "crit.jsonl")
    monkeypatch.setattr(claude_gate, "INVOCATIONS_LOG", tmp_path / "inv.jsonl")
    monkeypatch.setattr(claude_gate, "_find_claude_cmd", lambda: "claude")
    monkeypatch.setattr(claude_gate, "_load_config_layer2_enabled", lambda: True)

    def fake_run(cmd, *, timeout, env, cwd, label):
        return 0, "Not logged in", "", False

    monkeypatch.setattr(claude_gate, "_run_with_tree_kill", fake_run)

    text, success, exit_code = claude_gate.invoke_claude_text(
        prompt="hi", caller="test", model="sonnet", timeout=5,
    )
    assert success is False
    assert exit_code != 0
    # Error text must NOT leak back as the "analysis result" string.
    assert text == ""
