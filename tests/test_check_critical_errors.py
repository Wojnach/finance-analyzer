"""Tests for scripts/check_critical_errors.py."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_critical_errors as cce  # noqa: E402


def _write_entries(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def test_naive_timestamp_does_not_crash(tmp_path):
    """2026-05-28 fix #13: a resolution/error line whose ts omits the tz
    offset (naive) must not crash find_unresolved by comparing naive vs
    aware datetimes — it would hide ALL unresolved errors behind a traceback."""
    journal = tmp_path / "crit.jsonl"
    naive_now = datetime.now().replace(microsecond=0).isoformat()  # no tz offset
    _write_entries(journal, [
        {"ts": naive_now, "level": "critical", "category": "x",
         "caller": "c", "message": "m", "resolution": None, "context": {}},
    ])
    # Must not raise; the naive ts is coerced to UTC and surfaces as unresolved.
    unresolved = cce.find_unresolved(cce._load_entries(journal), days=7)
    assert len(unresolved) == 1


def test_no_journal_returns_zero(tmp_path):
    exit_code = cce.main(["--journal", str(tmp_path / "missing.jsonl")])
    assert exit_code == 0


def test_empty_journal_returns_zero(tmp_path):
    journal = tmp_path / "crit.jsonl"
    journal.write_text("", encoding="utf-8")
    assert cce.main(["--journal", str(journal)]) == 0


def test_unresolved_recent_returns_nonzero(tmp_path, capsys):
    journal = tmp_path / "crit.jsonl"
    _write_entries(journal, [
        {"ts": _iso(datetime.now(UTC) - timedelta(hours=2)),
         "level": "critical", "category": "auth_failure",
         "caller": "layer2_t3", "resolution": None,
         "message": "Not logged in", "context": {}},
    ])
    exit_code = cce.main(["--journal", str(journal)])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "1 unresolved" in out
    assert "auth_failure" in out


def test_old_entry_ignored(tmp_path):
    journal = tmp_path / "crit.jsonl"
    _write_entries(journal, [
        {"ts": _iso(datetime.now(UTC) - timedelta(days=14)),
         "level": "critical", "category": "auth_failure",
         "caller": "x", "resolution": None, "message": "old", "context": {}},
    ])
    assert cce.main(["--journal", str(journal), "--days", "7"]) == 0


def test_resolved_entry_ignored(tmp_path):
    """An entry with resolution != null is considered closed."""
    journal = tmp_path / "crit.jsonl"
    _write_entries(journal, [
        {"ts": _iso(datetime.now(UTC) - timedelta(hours=1)),
         "level": "critical", "category": "auth_failure",
         "caller": "x", "resolution": "fixed", "message": "y", "context": {}},
    ])
    assert cce.main(["--journal", str(journal)]) == 0


def test_resolves_ts_retroactive_resolution(tmp_path):
    """A later entry with resolves_ts pointing at an earlier ts closes it."""
    journal = tmp_path / "crit.jsonl"
    t1 = _iso(datetime.now(UTC) - timedelta(hours=3))
    t2 = _iso(datetime.now(UTC) - timedelta(hours=1))
    _write_entries(journal, [
        {"ts": t1, "level": "critical", "category": "auth_failure",
         "caller": "x", "resolution": None, "message": "was broken", "context": {}},
        {"ts": t2, "level": "info", "category": "resolution",
         "caller": "x", "resolution": "fixed via --bare removal",
         "resolves_ts": t1, "message": "fix", "context": {}},
    ])
    assert cce.main(["--journal", str(journal)]) == 0


def test_json_output_mode(tmp_path, capsys):
    journal = tmp_path / "crit.jsonl"
    _write_entries(journal, [
        {"ts": _iso(datetime.now(UTC) - timedelta(hours=1)),
         "level": "critical", "category": "auth_failure",
         "caller": "x", "resolution": None, "message": "y", "context": {"k": 1}},
    ])
    cce.main(["--journal", str(journal), "--json"])
    out = capsys.readouterr().out.strip()
    parsed = json.loads(out)  # must be valid JSON, one per line
    assert parsed["category"] == "auth_failure"
    assert parsed["context"]["k"] == 1


def test_malformed_line_is_skipped(tmp_path):
    journal = tmp_path / "crit.jsonl"
    journal.write_text("not json\n" + json.dumps({
        "ts": _iso(datetime.now(UTC) - timedelta(hours=1)),
        "level": "critical", "category": "auth_failure",
        "caller": "x", "resolution": None, "message": "y", "context": {},
    }) + "\n", encoding="utf-8")
    # Malformed line is silently skipped; real entry still surfaces.
    assert cce.main(["--journal", str(journal)]) == 1


class TestAutoResolveStaleCategories:
    """Auto-resolve categories that haven't fired in 3+ days with a post-fix."""

    def test_stale_category_auto_resolved(self):
        now = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        entries = [
            {"ts": _iso(now - timedelta(days=5)), "level": "critical",
             "category": "contract_violation", "resolution": None},
            {"ts": _iso(now - timedelta(days=4)), "level": "info",
             "category": "contract_violation", "resolution": "fixed"},
        ]
        unresolved = cce.find_unresolved(entries, days=7, now=now)
        assert len(unresolved) == 0

    def test_recent_category_not_auto_resolved(self):
        now = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        entries = [
            {"ts": _iso(now - timedelta(hours=12)), "level": "critical",
             "category": "contract_violation", "resolution": None},
            {"ts": _iso(now - timedelta(days=4)), "level": "info",
             "category": "contract_violation", "resolution": "fixed"},
        ]
        unresolved = cce.find_unresolved(entries, days=7, now=now)
        assert len(unresolved) == 1

    def test_stale_category_without_fix_not_auto_resolved(self):
        now = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        entries = [
            {"ts": _iso(now - timedelta(days=5)), "level": "critical",
             "category": "unknown_bug", "resolution": None},
        ]
        unresolved = cce.find_unresolved(entries, days=7, now=now)
        assert len(unresolved) == 1

    def test_fix_predating_critical_does_not_auto_resolve(self):
        now = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        entries = [
            {"ts": _iso(now - timedelta(days=6)), "level": "info",
             "category": "test_cat", "resolution": "fixed"},
            {"ts": _iso(now - timedelta(days=4)), "level": "critical",
             "category": "test_cat", "resolution": None},
        ]
        unresolved = cce.find_unresolved(entries, days=7, now=now)
        assert len(unresolved) == 1

    def test_mixed_categories(self):
        now = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
        entries = [
            {"ts": _iso(now - timedelta(days=5)), "level": "critical",
             "category": "contract_violation", "resolution": None},
            {"ts": _iso(now - timedelta(days=4)), "level": "info",
             "category": "contract_violation", "resolution": "fixed"},
            {"ts": _iso(now - timedelta(hours=6)), "level": "critical",
             "category": "accuracy_degradation", "resolution": None},
        ]
        unresolved = cce.find_unresolved(entries, days=7, now=now)
        assert len(unresolved) == 1
        assert unresolved[0]["category"] == "accuracy_degradation"
