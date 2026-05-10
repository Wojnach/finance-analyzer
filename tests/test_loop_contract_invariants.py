"""Tests for the trust-hardening invariants added 2026-05-10.

Each invariant has:
  - one positive case (no violations on healthy state)
  - one negative case per failure mode (violation fires with correct severity)

These tests are EXAMPLE-based on purpose — the invariants themselves are
the universal-truth gate; this file just confirms each gate routes right.
Property-based coverage of portfolio arithmetic lives in
``tests/test_property_invariants.py``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from portfolio import loop_contract


# ──────────────────────────────────────────────────────────────────────
# check_portfolio_arithmetic_safe
# ──────────────────────────────────────────────────────────────────────


def _write_state(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_portfolio_arithmetic_passes_on_healthy_state(tmp_path, monkeypatch):
    state = tmp_path / "portfolio_state.json"
    _write_state(state, {
        "cash_sek": 250_000.0,
        "holdings": {
            "BTC-USD": {"shares": 0.5, "avg_cost_usd": 60_000},
        },
    })
    monkeypatch.setattr(loop_contract, "_PORTFOLIO_STATE_FILES", (state,))
    assert loop_contract.check_portfolio_arithmetic_safe() == []


def test_portfolio_arithmetic_flags_negative_cash(tmp_path, monkeypatch):
    state = tmp_path / "portfolio_state.json"
    _write_state(state, {"cash_sek": -1000.0, "holdings": {}})
    monkeypatch.setattr(loop_contract, "_PORTFOLIO_STATE_FILES", (state,))
    violations = loop_contract.check_portfolio_arithmetic_safe()
    assert len(violations) == 1
    assert violations[0].severity == "CRITICAL"
    assert violations[0].invariant == "portfolio_arithmetic"
    assert "negative" in violations[0].message.lower()


def test_portfolio_arithmetic_flags_wrong_field_name(tmp_path, monkeypatch):
    """Regression: codex 2026-05-10 caught us using "cash" instead of "cash_sek".

    A state file with the OLD wrong field name must trigger a CRITICAL
    "missing or non-numeric" violation — proves we read the correct key.
    """
    state = tmp_path / "portfolio_state.json"
    _write_state(state, {"cash": 100.0, "holdings": {}})  # legacy-shaped
    monkeypatch.setattr(loop_contract, "_PORTFOLIO_STATE_FILES", (state,))
    violations = loop_contract.check_portfolio_arithmetic_safe()
    assert len(violations) == 1
    assert violations[0].severity == "CRITICAL"
    assert "cash_sek" in violations[0].message


def test_portfolio_arithmetic_flags_nan_shares(tmp_path, monkeypatch):
    state = tmp_path / "portfolio_state.json"
    # NaN can't survive json.dumps directly without allow_nan; emit raw.
    state.write_text(
        '{"cash_sek": 100.0, "holdings": {"BTC-USD": {"shares": NaN}}}',
        encoding="utf-8",
    )
    # Codex 2026-05-10: explicitly verify the field name; original test
    # used "cash" which masked the schema mismatch the production code had.
    # load_json wraps json.loads with default kwargs (allow_nan=True), so
    # NaN parses through. Patch path tuple and run.
    monkeypatch.setattr(loop_contract, "_PORTFOLIO_STATE_FILES", (state,))
    violations = loop_contract.check_portfolio_arithmetic_safe()
    assert any(
        v.invariant == "portfolio_arithmetic"
        and v.severity == "CRITICAL"
        and "NaN" in v.message
        for v in violations
    )


def test_portfolio_arithmetic_flags_negative_shares(tmp_path, monkeypatch):
    state = tmp_path / "portfolio_state.json"
    _write_state(state, {
        "cash_sek": 100.0,
        "holdings": {"ETH-USD": {"shares": -0.5}},
    })
    monkeypatch.setattr(loop_contract, "_PORTFOLIO_STATE_FILES", (state,))
    violations = loop_contract.check_portfolio_arithmetic_safe()
    assert any(
        v.severity == "CRITICAL" and "long-only" in v.message.lower()
        for v in violations
    )


def test_portfolio_arithmetic_skips_missing_files(tmp_path, monkeypatch):
    # Pure day-1 deployment with no state files yet should NOT emit any
    # violation — that would page the user every cycle on a fresh install.
    monkeypatch.setattr(
        loop_contract,
        "_PORTFOLIO_STATE_FILES",
        (tmp_path / "missing.json",),
    )
    assert loop_contract.check_portfolio_arithmetic_safe() == []


def test_portfolio_arithmetic_handles_malformed_state(tmp_path, monkeypatch):
    state = tmp_path / "portfolio_state.json"
    state.write_text("not even close to json", encoding="utf-8")
    monkeypatch.setattr(loop_contract, "_PORTFOLIO_STATE_FILES", (state,))
    # Should NOT raise. Either flags malformed OR returns empty (both are
    # acceptable — a try/except in the safe wrapper degrades to []).
    out = loop_contract.check_portfolio_arithmetic_safe()
    assert isinstance(out, list)


# ──────────────────────────────────────────────────────────────────────
# check_atomic_write_residue_safe
# ──────────────────────────────────────────────────────────────────────


def test_atomic_write_residue_passes_with_no_tmp_files(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_contract, "DATA_DIR", tmp_path)
    assert loop_contract.check_atomic_write_residue_safe() == []


def test_atomic_write_residue_skips_in_flight_writes(tmp_path, monkeypatch):
    # Fresh .tmp file (mtime = now) should NOT fire — atomic write may
    # be in flight legitimately.
    fresh = tmp_path / "portfolio_state.json.tmp"
    fresh.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(loop_contract, "DATA_DIR", tmp_path)
    assert loop_contract.check_atomic_write_residue_safe() == []


def test_atomic_write_residue_flags_orphaned_tmp(tmp_path, monkeypatch):
    stale = tmp_path / "portfolio_state.json.tmp"
    stale.write_text("{}", encoding="utf-8")
    # Backdate mtime by 10 min — past the 5 min cutoff.
    old = time.time() - 10 * 60
    os.utime(stale, (old, old))
    monkeypatch.setattr(loop_contract, "DATA_DIR", tmp_path)
    violations = loop_contract.check_atomic_write_residue_safe()
    assert len(violations) == 1
    v = violations[0]
    assert v.invariant == "atomic_write_residue"
    assert v.severity == "WARNING"
    assert v.details["count"] == 1


def test_atomic_write_residue_truncates_long_lists(tmp_path, monkeypatch):
    # 12 stale .tmp files. Message must truncate at _TMP_RESIDUE_MAX_REPORT
    # (5) with a "…" indicator; details lists at most that many.
    old = time.time() - 10 * 60
    for i in range(12):
        p = tmp_path / f"file_{i}.json.tmp"
        p.write_text("{}", encoding="utf-8")
        os.utime(p, (old, old))
    monkeypatch.setattr(loop_contract, "DATA_DIR", tmp_path)
    violations = loop_contract.check_atomic_write_residue_safe()
    assert len(violations) == 1
    v = violations[0]
    assert v.details["count"] == 12
    assert len(v.details["stale_files"]) == loop_contract._TMP_RESIDUE_MAX_REPORT


# ──────────────────────────────────────────────────────────────────────
# check_journal_uniqueness_safe
# ──────────────────────────────────────────────────────────────────────


def _write_journal(path: Path, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_journal_uniqueness_passes_on_unique_triggers(tmp_path, monkeypatch):
    journal = tmp_path / "layer2_journal.jsonl"
    _write_journal(journal, [
        {"trigger_id": "T1", "timestamp": "2026-05-10T10:00:00+00:00"},
        {"trigger_id": "T2", "timestamp": "2026-05-10T10:01:00+00:00"},
        {"trigger_id": "T3", "timestamp": "2026-05-10T10:02:00+00:00"},
    ])
    monkeypatch.setattr(loop_contract, "LAYER2_JOURNAL_FILE", journal)
    assert loop_contract.check_journal_uniqueness_safe() == []


def test_journal_uniqueness_flags_duplicate_inside_window(tmp_path, monkeypatch):
    journal = tmp_path / "layer2_journal.jsonl"
    _write_journal(journal, [
        {"trigger_id": "T1", "timestamp": "2026-05-10T10:00:00+00:00"},
        {"trigger_id": "T1", "timestamp": "2026-05-10T10:03:00+00:00"},  # dup
    ])
    monkeypatch.setattr(loop_contract, "LAYER2_JOURNAL_FILE", journal)
    violations = loop_contract.check_journal_uniqueness_safe()
    assert len(violations) == 1
    v = violations[0]
    assert v.invariant == "journal_uniqueness"
    assert v.severity == "WARNING"
    assert v.details["duplicates"][0]["trigger_id"] == "T1"


def test_journal_uniqueness_ignores_old_dup_outside_window(tmp_path, monkeypatch):
    # Same trigger_id, but timestamps are >10 min apart → not a retry
    # storm; legitimate re-emit on a manual restart.
    journal = tmp_path / "layer2_journal.jsonl"
    _write_journal(journal, [
        {"trigger_id": "T1", "timestamp": "2026-05-10T10:00:00+00:00"},
        {"trigger_id": "T1", "timestamp": "2026-05-10T11:00:00+00:00"},
    ])
    monkeypatch.setattr(loop_contract, "LAYER2_JOURNAL_FILE", journal)
    assert loop_contract.check_journal_uniqueness_safe() == []


def test_journal_uniqueness_handles_malformed_lines(tmp_path, monkeypatch):
    # Mix of valid + malformed lines must NOT raise; the malformed line
    # is skipped and the valid duplicate is still detected.
    journal = tmp_path / "layer2_journal.jsonl"
    journal.write_text(
        '{"trigger_id": "T1", "timestamp": "2026-05-10T10:00:00+00:00"}\n'
        'not-valid-json\n'
        '{"trigger_id": "T1", "timestamp": "2026-05-10T10:02:00+00:00"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(loop_contract, "LAYER2_JOURNAL_FILE", journal)
    violations = loop_contract.check_journal_uniqueness_safe()
    assert len(violations) == 1


def test_journal_uniqueness_skips_missing_journal(tmp_path, monkeypatch):
    # Day-1 deployment — journal doesn't exist. Must not violate.
    monkeypatch.setattr(
        loop_contract,
        "LAYER2_JOURNAL_FILE",
        tmp_path / "missing.jsonl",
    )
    assert loop_contract.check_journal_uniqueness_safe() == []
