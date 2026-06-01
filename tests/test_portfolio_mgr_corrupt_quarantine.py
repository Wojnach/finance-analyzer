"""Tests for the corrupt-state quarantine path in portfolio_mgr.

Added 2026-06-01. Background: a hand-edit left ``portfolio_state.json``
unparseable while NO ``.bak`` existed on disk. The old ``_load_state_from``
fall-through returned fresh defaults with only a ``logger.critical`` line, so
the next ``save_state`` would have silently wiped the entire portfolio. These
tests pin the new behaviour: when a corrupt file has no recoverable backup, the
corrupt bytes are quarantined (content-addressed, once) and a critical journal
entry is written BEFORE defaults are returned — and the read path never raises.

All file paths are tmp_path-scoped and ``CRITICAL_ERRORS_LOG`` is monkeypatched
so the suite is xdist-safe and never touches the live journal.
"""

import json

import pytest

import portfolio.portfolio_mgr as pm
from portfolio.portfolio_mgr import INITIAL_CASH_SEK, _load_state_from

CORRUPT = b'{"cash_sek": 467803.17,\n  ],\n  {"orphan": true}\n'  # the real-world shape


@pytest.fixture
def journal(tmp_path, monkeypatch):
    """Redirect the critical-errors journal into tmp_path."""
    jpath = tmp_path / "critical_errors.jsonl"
    monkeypatch.setattr(pm, "CRITICAL_ERRORS_LOG", jpath)
    return jpath


def _read_journal(jpath):
    if not jpath.exists():
        return []
    return [json.loads(ln) for ln in jpath.read_text(encoding="utf-8").splitlines() if ln.strip()]


class TestCorruptQuarantine:
    def test_no_backup_quarantines_and_journals_then_defaults(self, tmp_path, journal):
        state = tmp_path / "portfolio_state.json"
        state.write_bytes(CORRUPT)

        result = _load_state_from(state)

        # Loop keeps running on fresh defaults.
        assert result["cash_sek"] == INITIAL_CASH_SEK
        assert result["holdings"] == {}

        # Corrupt bytes preserved verbatim under a content-addressed name.
        quarantines = list(tmp_path.glob("portfolio_state.json.corrupt-*"))
        assert len(quarantines) == 1
        assert quarantines[0].read_bytes() == CORRUPT

        # A single critical journal entry surfaces it.
        entries = _read_journal(journal)
        assert len(entries) == 1
        e = entries[0]
        assert e["level"] == "critical"
        assert e["category"] == "portfolio_state_corrupt"
        assert e["context"]["path"] == str(state)
        assert e["context"]["quarantine"] == str(quarantines[0])

    def test_idempotent_across_repeated_cycles(self, tmp_path, journal):
        """The corrupt branch fires every 60s cycle — must act exactly once."""
        state = tmp_path / "portfolio_state.json"
        state.write_bytes(CORRUPT)

        for _ in range(5):
            _load_state_from(state)

        assert len(list(tmp_path.glob("portfolio_state.json.corrupt-*"))) == 1
        assert len(_read_journal(journal)) == 1

    def test_distinct_corruptions_quarantined_separately(self, tmp_path, journal):
        state = tmp_path / "portfolio_state.json"
        state.write_bytes(CORRUPT)
        _load_state_from(state)
        state.write_bytes(CORRUPT + b"different garbage")
        _load_state_from(state)

        assert len(list(tmp_path.glob("portfolio_state.json.corrupt-*"))) == 2
        assert len(_read_journal(journal)) == 2

    def test_valid_backup_recovers_without_quarantine(self, tmp_path, journal):
        """Existing C7 recovery must be untouched: a good .bak short-circuits."""
        state = tmp_path / "portfolio_state.json"
        state.write_bytes(CORRUPT)
        good = {"cash_sek": 123456.0, "holdings": {"XAG-USD": {"shares": 5.0}},
                "transactions": []}
        (tmp_path / "portfolio_state.json.bak").write_text(json.dumps(good), encoding="utf-8")

        result = _load_state_from(state)

        assert result["cash_sek"] == 123456.0
        assert result["holdings"]["XAG-USD"]["shares"] == 5.0
        assert list(tmp_path.glob("portfolio_state.json.corrupt-*")) == []
        assert _read_journal(journal) == []

    def test_missing_file_no_quarantine(self, tmp_path, journal):
        """A simply-absent file (day-1) is not corruption — no quarantine, no journal."""
        state = tmp_path / "portfolio_state.json"
        result = _load_state_from(state)
        assert result["cash_sek"] == INITIAL_CASH_SEK
        assert list(tmp_path.glob("portfolio_state.json.corrupt-*")) == []
        assert _read_journal(journal) == []

    def test_journal_failure_does_not_crash_read_path(self, tmp_path, monkeypatch):
        """Evidence preservation is best-effort: a journal write error must not
        propagate into the loop's read path."""
        state = tmp_path / "portfolio_state.json"
        state.write_bytes(CORRUPT)
        monkeypatch.setattr(pm, "CRITICAL_ERRORS_LOG", tmp_path / "critical_errors.jsonl")

        def _boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(pm, "atomic_append_jsonl", _boom)

        # Must return defaults, not raise.
        result = _load_state_from(state)
        assert result["cash_sek"] == INITIAL_CASH_SEK
