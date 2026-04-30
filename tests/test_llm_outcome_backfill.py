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


# ----------------------------------------------------------------------------
# Per-asset tolerance fix (2026-05-01) — recovery from snapshot gaps wider
# than the default 2h tolerance, without losing semantic correctness.
# ----------------------------------------------------------------------------


def test_backfill_recovers_btc_with_loop_downtime_gap(tmp_path):
    """A 5h gap in the snapshot stream (loop down) should not orphan
    backfill rows for crypto/metals tickers (24/7 markets).

    Real-world case: 2026-04-25 saw a 14.8h BTC snapshot gap. With 8h
    tolerance, we still find a price within window.
    """
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    now = datetime.now(UTC)
    # Entry 2 days ago, target 1d later, snapshots placed 5h before each.
    entry_time = now - timedelta(days=2)
    target_time = entry_time + timedelta(hours=24)
    _write_prob_log(log, [{
        "ts": entry_time.isoformat(),
        "signal": "ministral",
        "ticker": "BTC-USD",
        "horizon": "1d",
        "probs": {"BUY": 0.6, "HOLD": 0.3, "SELL": 0.1},
        "chosen": "BUY",
        "confidence": 0.6,
    }])
    _write_snapshot(snap, [
        {"ts": (entry_time - timedelta(hours=5)).isoformat(),
         "prices": {"BTC-USD": 100.0}},
        {"ts": (target_time - timedelta(hours=5)).isoformat(),
         "prices": {"BTC-USD": 103.0}},
    ])
    stats = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats["written"] == 1, (
        "5h snapshot gap on a crypto ticker should be tolerable; "
        "default 2h tolerance was too tight."
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["entry_price"] == 100.0
    assert rows[0]["target_price"] == 103.0


def test_backfill_recovers_mstr_overnight_gap(tmp_path):
    """A 16h gap is normal for MSTR (after-hours) and must not orphan
    1d-horizon outcome rows. The right comparison price for a 1d-ahead
    stock prediction is the next-trading-day close, which lands 16-22h
    after the entry timestamp when the entry was logged after-hours.
    """
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    now = datetime.now(UTC)
    entry_time = now - timedelta(days=2)
    target_time = entry_time + timedelta(hours=24)
    _write_prob_log(log, [{
        "ts": entry_time.isoformat(),
        "signal": "ministral",
        "ticker": "MSTR",
        "horizon": "1d",
        "probs": {"BUY": 0.4, "HOLD": 0.4, "SELL": 0.2},
        "chosen": "HOLD",
        "confidence": 0.4,
    }])
    # Snapshots placed 16h before each (mimics MSTR median offset).
    _write_snapshot(snap, [
        {"ts": (entry_time - timedelta(hours=16)).isoformat(),
         "prices": {"MSTR": 280.0}},
        {"ts": (target_time - timedelta(hours=16)).isoformat(),
         "prices": {"MSTR": 287.0}},
    ])
    stats = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats["written"] == 1, (
        "16h MSTR overnight gap should not orphan 1d outcome backfill"
    )


def test_backfill_does_not_match_decommissioned_ticker(tmp_path):
    """Tickers fully outside the snapshot window (e.g. NVDA after the
    Apr-09 decommission) should still be skipped — the wider tolerance
    must NOT bridge multi-week gaps and pretend stale prices are valid.
    """
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    now = datetime.now(UTC)
    entry_time = now - timedelta(days=2)
    _write_prob_log(log, [{
        "ts": entry_time.isoformat(),
        "signal": "ministral",
        "ticker": "NVDA",
        "horizon": "1d",
        "probs": {"BUY": 0.5, "HOLD": 0.3, "SELL": 0.2},
        "chosen": "BUY",
        "confidence": 0.5,
    }])
    # Last NVDA snapshot 15 days ago — way outside any reasonable tolerance.
    _write_snapshot(snap, [
        {"ts": (now - timedelta(days=15)).isoformat(),
         "prices": {"NVDA": 175.0}},
    ])
    stats = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats["written"] == 0
    assert stats["skipped_missing_price"] == 1, (
        "stale snapshots from before a ticker was decommissioned must not "
        "be matched as 'current' prices via wider tolerance"
    )


def test_backfill_uses_crypto_tolerance_for_eth(tmp_path):
    """Confirm ETH-USD (24/7 market) gets the same 8h tolerance as BTC-USD."""
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    now = datetime.now(UTC)
    entry_time = now - timedelta(days=2)
    target_time = entry_time + timedelta(hours=24)
    _write_prob_log(log, [{
        "ts": entry_time.isoformat(),
        "signal": "qwen3",
        "ticker": "ETH-USD",
        "horizon": "1d",
        "probs": {"BUY": 0.45, "HOLD": 0.4, "SELL": 0.15},
        "chosen": "BUY",
        "confidence": 0.45,
    }])
    _write_snapshot(snap, [
        {"ts": (entry_time - timedelta(hours=6)).isoformat(),
         "prices": {"ETH-USD": 2200.0}},
        {"ts": (target_time - timedelta(hours=7)).isoformat(),
         "prices": {"ETH-USD": 2266.0}},
    ])
    stats = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats["written"] == 1


def test_backfill_uses_metals_tolerance_for_gold(tmp_path):
    """Confirm XAU-USD gets the same 8h tolerance — metals are 24/7."""
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    now = datetime.now(UTC)
    entry_time = now - timedelta(days=2)
    target_time = entry_time + timedelta(hours=24)
    _write_prob_log(log, [{
        "ts": entry_time.isoformat(),
        "signal": "claude_fundamental",
        "ticker": "XAU-USD",
        "horizon": "1d",
        "probs": {"BUY": 0.35, "HOLD": 0.35, "SELL": 0.30},
        "chosen": "BUY",
        "confidence": 0.35,
    }])
    _write_snapshot(snap, [
        {"ts": (entry_time - timedelta(hours=4, minutes=30)).isoformat(),
         "prices": {"XAU-USD": 4500.0}},
        {"ts": (target_time - timedelta(hours=4, minutes=30)).isoformat(),
         "prices": {"XAU-USD": 4540.0}},
    ])
    stats = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats["written"] == 1


def test_backfill_does_not_widen_tolerance_for_crypto_beyond_8h(tmp_path):
    """Crypto/metals tolerance is 8h; a 9h gap should still be skipped
    — wider would risk matching stale prices from a multi-day outage.
    """
    log = tmp_path / "log.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    now = datetime.now(UTC)
    entry_time = now - timedelta(days=2)
    target_time = entry_time + timedelta(hours=24)
    _write_prob_log(log, [{
        "ts": entry_time.isoformat(),
        "signal": "ministral",
        "ticker": "BTC-USD",
        "horizon": "1d",
        "probs": {"BUY": 0.5, "HOLD": 0.3, "SELL": 0.2},
        "chosen": "BUY",
        "confidence": 0.5,
    }])
    # Target snapshot 9h before target_time — outside crypto's 8h tolerance.
    _write_snapshot(snap, [
        {"ts": entry_time.isoformat(),
         "prices": {"BTC-USD": 100.0}},
        {"ts": (target_time - timedelta(hours=9)).isoformat(),
         "prices": {"BTC-USD": 103.0}},
    ])
    stats = mod.backfill(log_path=log, outcomes_path=out, snapshot_path=snap)
    assert stats["written"] == 0
    assert stats["skipped_missing_price"] == 1
