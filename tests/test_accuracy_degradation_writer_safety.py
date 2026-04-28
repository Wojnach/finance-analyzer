"""Snapshot writer bulletproofing — defense against silent state-without-write desync.

Caught 2026-04-28 during the contract-alert root-cause investigation:
``data/accuracy_snapshot_state.json`` claimed today's snapshot was written
but ``data/accuracy_snapshots.jsonl`` had no entry for today (last entry
was Apr 21, 7 days stale). Once the state file says today is done,
``maybe_save_daily_snapshot`` returns False on every subsequent cycle —
so a single state-without-write desync silences the writer for the rest
of the day, and if it recurs, indefinitely.

The fix verifies the JSONL actually grew before persisting state. If the
writer "succeeds" but doesn't append a line (stub bypass, partial atomic
write, exception swallowed somewhere downstream), state is left untouched
and a row goes to ``critical_errors.jsonl`` so the dispatcher engages.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from portfolio import accuracy_degradation as deg


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect every state file the writer touches to ``tmp_path``."""
    snap_state = tmp_path / "accuracy_snapshot_state.json"
    alert_state = tmp_path / "degradation_alert_state.json"
    snapshots_jsonl = tmp_path / "accuracy_snapshots.jsonl"
    critical_errors = tmp_path / "critical_errors.jsonl"

    monkeypatch.setattr(deg, "SNAPSHOT_STATE_FILE", snap_state)
    monkeypatch.setattr(deg, "ALERT_STATE_FILE", alert_state)

    from portfolio import accuracy_stats
    monkeypatch.setattr(accuracy_stats, "ACCURACY_SNAPSHOTS_FILE", snapshots_jsonl)

    from portfolio import claude_gate
    monkeypatch.setattr(claude_gate, "CRITICAL_ERRORS_LOG", critical_errors)

    return {
        "snap_state": snap_state,
        "alert_state": alert_state,
        "snapshots_jsonl": snapshots_jsonl,
        "critical_errors": critical_errors,
    }


def _now_after_target_hour() -> datetime:
    """A fixed wall-clock past the default 06:00 UTC snapshot gate."""
    return datetime(2026, 4, 28, 10, 0, 0, tzinfo=UTC)


class TestWriterBulletproofing:

    def test_silent_writer_failure_blocks_state_update(
        self, isolated_state, monkeypatch
    ):
        """Stub ``save_full_accuracy_snapshot`` to a no-op that doesn't grow
        the JSONL. The state file must NOT be updated and a critical_errors
        row must appear so the dispatcher can engage on the silent failure.
        """
        # Pre-create the JSONL so the size-check has a defined "before"
        isolated_state["snapshots_jsonl"].write_text("", encoding="utf-8")

        monkeypatch.setattr(
            deg, "save_full_accuracy_snapshot",
            lambda **_: {"ts": datetime.now(UTC).isoformat()},
        )

        wrote = deg.maybe_save_daily_snapshot(config={}, now=_now_after_target_hour())

        assert wrote is False, "writer must report failure on silent no-op"
        # last_snapshot_date_utc must NOT be set to today (else writer
        # skips for the rest of the day). The state file MAY exist
        # with a last_silent_failure_ts field for journal-write rate
        # limiting, plus _load_snapshot_state's default empty
        # last_snapshot_date_utc — both are fine; what matters is we
        # still retry next cycle.
        if isolated_state["snap_state"].exists():
            saved = json.loads(
                isolated_state["snap_state"].read_text(encoding="utf-8"),
            )
            assert saved.get("last_snapshot_date_utc", "") != "2026-04-28", (
                "last_snapshot_date_utc must NOT be set to today on silent "
                "failure or the writer skips retries for the rest of the day"
            )

        # critical_errors row landed
        assert isolated_state["critical_errors"].exists(), \
            "silent writer failure must surface to critical_errors.jsonl"
        rows = [
            json.loads(line)
            for line in isolated_state["critical_errors"].read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert rows[0]["level"] == "critical"
        assert rows[0]["category"] == "snapshot_writer_silent_failure"
        assert rows[0]["caller"] == "maybe_save_daily_snapshot"
        assert "didn't grow" in rows[0]["message"].lower() or \
               "did not grow" in rows[0]["message"].lower()

    def test_real_append_updates_state(self, isolated_state, monkeypatch):
        """A real write that grows the file by ≥1 line MUST persist state."""
        from portfolio.file_utils import atomic_append_jsonl

        snapshots_jsonl = isolated_state["snapshots_jsonl"]

        def fake_save(**_):
            atomic_append_jsonl(snapshots_jsonl, {"ts": datetime.now(UTC).isoformat()})
            return {"ts": datetime.now(UTC).isoformat()}

        monkeypatch.setattr(deg, "save_full_accuracy_snapshot", fake_save)

        wrote = deg.maybe_save_daily_snapshot(config={}, now=_now_after_target_hour())

        assert wrote is True
        assert isolated_state["snap_state"].exists()
        state = json.loads(isolated_state["snap_state"].read_text(encoding="utf-8"))
        assert state["last_snapshot_date_utc"] == "2026-04-28"

        # No critical_errors row
        assert not isolated_state["critical_errors"].exists() or \
               isolated_state["critical_errors"].read_text(encoding="utf-8").strip() == ""

    def test_already_today_short_circuits_without_size_check(
        self, isolated_state, monkeypatch
    ):
        """If state says today is done, return False before the writer runs."""
        from portfolio.file_utils import atomic_write_json

        atomic_write_json(
            isolated_state["snap_state"], {"last_snapshot_date_utc": "2026-04-28"}
        )

        called = {"count": 0}

        def fake_save(**_):
            called["count"] += 1
            return {}

        monkeypatch.setattr(deg, "save_full_accuracy_snapshot", fake_save)

        wrote = deg.maybe_save_daily_snapshot(config={}, now=_now_after_target_hour())

        assert wrote is False
        assert called["count"] == 0, "save_full_accuracy_snapshot must not run"

    def test_pre_target_hour_short_circuits(self, isolated_state, monkeypatch):
        """Before 06:00 UTC the writer must not run and must not touch state."""
        called = {"count": 0}

        def fake_save(**_):
            called["count"] += 1
            return {}

        monkeypatch.setattr(deg, "save_full_accuracy_snapshot", fake_save)

        wrote = deg.maybe_save_daily_snapshot(
            config={}, now=datetime(2026, 4, 28, 4, 0, tzinfo=UTC),
        )

        assert wrote is False
        assert called["count"] == 0
        assert not isolated_state["snap_state"].exists()

    def test_writer_raises_returns_false_no_state_update(
        self, isolated_state, monkeypatch
    ):
        """If save_full_accuracy_snapshot raises, state stays untouched.

        We don't write a critical_errors row in this path because the
        existing logger.warning('Daily accuracy snapshot failed') already
        surfaces the exception — and crash recovery is handled by the
        next-cycle retry. We only journal the *silent* failure mode,
        because that's the one that hides for days.
        """
        def boom(**_):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(deg, "save_full_accuracy_snapshot", boom)

        wrote = deg.maybe_save_daily_snapshot(config={}, now=_now_after_target_hour())

        assert wrote is False
        assert not isolated_state["snap_state"].exists()
        # No critical_errors row for the raise path — the logger already
        # surfaces it.
        assert not isolated_state["critical_errors"].exists() or \
               isolated_state["critical_errors"].read_text(encoding="utf-8").strip() == ""

    def test_silent_failure_journal_rate_limit(self, isolated_state, monkeypatch):
        """Codex round 7 P3 (2026-04-28): on a recurring silent failure,
        the writer is called every cycle and would otherwise append a
        fresh critical_errors row every minute (~1440/day). We
        rate-limit journal writes to one per 30 minutes via a
        last_silent_failure_ts field in the snapshot state."""
        # Pre-create the JSONL so size check has a defined "before"
        isolated_state["snapshots_jsonl"].write_text("", encoding="utf-8")

        monkeypatch.setattr(
            deg, "save_full_accuracy_snapshot",
            lambda **_: {"ts": datetime.now(UTC).isoformat()},
        )

        first_call = datetime(2026, 4, 28, 10, 0, 0, tzinfo=UTC)
        cycle_5min = datetime(2026, 4, 28, 10, 5, 0, tzinfo=UTC)
        cycle_31min = datetime(2026, 4, 28, 10, 31, 0, tzinfo=UTC)

        deg.maybe_save_daily_snapshot(config={}, now=first_call)
        deg.maybe_save_daily_snapshot(config={}, now=cycle_5min)
        deg.maybe_save_daily_snapshot(config={}, now=cycle_31min)

        rows = [
            json.loads(line)
            for line in isolated_state["critical_errors"].read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        assert len(rows) == 2, (
            f"Expected 2 journal rows: 1 for first failure, 1 after the "
            f"30-min cooldown (the 5-min replay should be deduped). "
            f"Got {len(rows)}."
        )

    def test_jsonl_missing_treated_as_zero(self, isolated_state, monkeypatch):
        """If the JSONL file doesn't exist yet at the start, treat it as 0
        bytes — a real first append will grow it from nothing to a line.
        """
        from portfolio.file_utils import atomic_append_jsonl

        # Don't pre-create the file
        snapshots_jsonl = isolated_state["snapshots_jsonl"]
        assert not snapshots_jsonl.exists()

        def fake_save(**_):
            atomic_append_jsonl(snapshots_jsonl, {"ts": "first ever"})
            return {}

        monkeypatch.setattr(deg, "save_full_accuracy_snapshot", fake_save)

        wrote = deg.maybe_save_daily_snapshot(config={}, now=_now_after_target_hour())

        assert wrote is True
        assert snapshots_jsonl.exists()
