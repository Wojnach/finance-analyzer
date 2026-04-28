"""Tests for portfolio.llm_outcome_backfill."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from portfolio import llm_outcome_backfill as mod


def _write_prob_log(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def _write_snapshot(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def test_backfill_noop_when_log_missing(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    stats = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats["processed"] == 0
    assert stats["written"] == 0
    assert not out.exists()


def test_backfill_writes_outcome_when_horizon_elapsed(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"

    now = datetime.now(UTC)
    # Entry 2 days ago, horizon 1d (24h) — should be backfill-eligible.
    entry_time = (now - timedelta(days=2)).isoformat()
    target_time = (datetime.fromisoformat(entry_time) + timedelta(hours=24)).isoformat()

    _write_prob_log(log, [{
        "ts": entry_time,
        "signal": "ministral",
        "ticker": "BTC-USD",
        "horizon": "1d",
        "probs": {"BUY": 0.7, "HOLD": 0.2, "SELL": 0.1},
        "chosen": "BUY",
        "confidence": 0.7,
    }])
    _write_snapshot(snap, [
        {"ts": entry_time, "prices": {"BTC-USD": 100.0}},
        {"ts": target_time, "prices": {"BTC-USD": 102.0}},
    ])

    stats = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats["written"] == 1
    assert stats["skipped_too_recent"] == 0

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["outcome"] == "BUY"  # +2 % is above the 0.5 % buy threshold
    assert rows[0]["pct_change"] == pytest.approx(0.02)
    assert rows[0]["entry_price"] == 100.0
    assert rows[0]["target_price"] == 102.0


def test_backfill_skips_too_recent(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    _write_snapshot(snap, [])
    now = datetime.now(UTC)
    # Entry 1 hour ago, horizon 1d — horizon hasn't elapsed.
    _write_prob_log(log, [{
        "ts": (now - timedelta(hours=1)).isoformat(),
        "signal": "ministral", "ticker": "BTC-USD", "horizon": "1d",
        "probs": {"BUY": 0.5, "HOLD": 0.3, "SELL": 0.2},
        "chosen": "BUY", "confidence": 0.5,
    }])
    stats = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats["written"] == 0
    assert stats["skipped_too_recent"] == 1


def test_backfill_skips_missing_price(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    now = datetime.now(UTC)
    entry_time = (now - timedelta(days=2)).isoformat()
    _write_prob_log(log, [{
        "ts": entry_time,
        "signal": "ministral", "ticker": "BTC-USD", "horizon": "1d",
        "probs": {"BUY": 0.5, "HOLD": 0.3, "SELL": 0.2},
        "chosen": "BUY", "confidence": 0.5,
    }])
    _write_snapshot(snap, [])  # no snapshot data
    stats = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats["written"] == 0
    assert stats["skipped_missing_price"] == 1


def test_backfill_is_idempotent(tmp_path):
    """Rerunning the backfill must not duplicate rows."""
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    now = datetime.now(UTC)
    entry_time = (now - timedelta(days=2)).isoformat()
    target_time = (datetime.fromisoformat(entry_time) + timedelta(hours=24)).isoformat()
    _write_prob_log(log, [{
        "ts": entry_time,
        "signal": "ministral", "ticker": "BTC-USD", "horizon": "1d",
        "probs": {"BUY": 0.5, "HOLD": 0.3, "SELL": 0.2},
        "chosen": "BUY", "confidence": 0.5,
    }])
    _write_snapshot(snap, [
        {"ts": entry_time, "prices": {"BTC-USD": 100.0}},
        {"ts": target_time, "prices": {"BTC-USD": 103.0}},
    ])
    mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    stats2 = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats2["written"] == 0
    assert stats2["skipped_already_present"] == 1
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1


def test_backfill_skips_bad_horizon(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    _write_prob_log(log, [{
        "ts": "2026-01-01T00:00:00+00:00",
        "signal": "ministral", "ticker": "BTC-USD", "horizon": "99y",
        "probs": {"BUY": 0.5, "HOLD": 0.3, "SELL": 0.2},
        "chosen": "BUY", "confidence": 0.5,
    }])
    _write_snapshot(snap, [])
    stats = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats["skipped_bad_row"] >= 1
    assert stats["written"] == 0


def test_outcome_lookup_reads_outcomes_file(tmp_path):
    out = tmp_path / "out.jsonl"
    out.write_text(json.dumps({
        "ts": "2026-04-21T12:00:00+00:00",
        "signal": "ministral", "ticker": "BTC-USD", "horizon": "1d",
        "outcome": "BUY",
    }) + "\n")
    lookup = mod.outcome_lookup(outcomes_path=out)
    assert lookup("2026-04-21T12:00:00+00:00", "BTC-USD", "1d") == "BUY"
    assert lookup("2026-04-21T12:00:00+00:00", "BTC-USD", "3d") is None


def test_outcome_lookup_empty_when_file_missing(tmp_path):
    lookup = mod.outcome_lookup(outcomes_path=tmp_path / "nope.jsonl")
    assert lookup("whatever", "BTC-USD", "1d") is None


def test_backfill_sell_outcome(tmp_path):
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    now = datetime.now(UTC)
    entry_time = (now - timedelta(days=2)).isoformat()
    target_time = (datetime.fromisoformat(entry_time) + timedelta(hours=24)).isoformat()
    _write_prob_log(log, [{
        "ts": entry_time,
        "signal": "forecast", "ticker": "BTC-USD", "horizon": "1d",
        "probs": {"BUY": 0.2, "HOLD": 0.3, "SELL": 0.5},
        "chosen": "SELL", "confidence": 0.5,
    }])
    _write_snapshot(snap, [
        {"ts": entry_time, "prices": {"BTC-USD": 100.0}},
        {"ts": target_time, "prices": {"BTC-USD": 97.0}},  # -3 %
    ])
    mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["outcome"] == "SELL"
    assert rows[0]["pct_change"] == pytest.approx(-0.03)


def test_backfill_dedups_null_horizon_rows(tmp_path):
    """Regression for the null-horizon dedup bug found 2026-04-28.

    Early production rows had `horizon: null`. The write side normalizes
    null -> '1d', but the dedup-key reader used dict.get(default) which
    only honors the default when the key is ABSENT, not when its value
    is null. So the second backfill run computed key (..., None) and
    didn't match the existing outcome row's key (..., '1d') — re-writing
    the same outcome on every run. 30 keys x 91 hourly cycles = 2,700
    duplicate rows over 7 days, manufacturing a fake -25pp accuracy drop
    on claude_fundamental BTC-USD before this was found.
    """
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"

    now = datetime.now(UTC)
    entry_time = (now - timedelta(days=2)).isoformat()
    target_time = (datetime.fromisoformat(entry_time) + timedelta(hours=24)).isoformat()

    _write_prob_log(log, [{
        "ts": entry_time,
        "signal": "claude_fundamental",
        "ticker": "XAG-USD",
        "horizon": None,  # the bug case — null, not absent
        "probs": {"BUY": 0.333, "HOLD": 0.333, "SELL": 0.334},
        "chosen": "HOLD",
        "confidence": 0.0,
    }])
    _write_snapshot(snap, [
        {"ts": entry_time, "prices": {"XAG-USD": 77.11}},
        {"ts": target_time, "prices": {"XAG-USD": 78.21}},
    ])

    # First backfill: writes one outcome row.
    stats1 = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats1["written"] == 1
    rows_after_first = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows_after_first) == 1

    # Second backfill: must NOT re-write the same outcome.
    stats2 = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats2["written"] == 0, (
        "null-horizon row was re-backfilled — _row_key dedup is broken"
    )
    assert stats2["skipped_already_present"] == 1
    rows_after_second = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows_after_second) == 1

    # Five more runs to mimic an hourly cron over 5 hours; should still be one row.
    for _ in range(5):
        mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    rows_final = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows_final) == 1, (
        "null-horizon row accumulated duplicates across multiple backfill runs"
    )
