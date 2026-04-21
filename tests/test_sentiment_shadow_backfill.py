"""Tests for portfolio.sentiment_shadow_backfill."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from portfolio import sentiment_shadow_backfill as mod


def _write_ab(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def _write_snap(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""))


def test_noop_when_ab_log_missing(tmp_path):
    out = tmp_path / "out.jsonl"
    stats = mod.backfill(
        ab_log_path=tmp_path / "nope.jsonl",
        outcomes_path=out,
        snapshot_path=tmp_path / "snap.jsonl",
    )
    assert stats["outcomes_written"] == 0
    assert not out.exists()


def test_backfill_writes_primary_plus_shadows(tmp_path):
    ab = tmp_path / "ab.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"

    now = datetime.now(UTC)
    entry_time = (now - timedelta(days=2)).isoformat()
    target_time = (datetime.fromisoformat(entry_time) + timedelta(hours=24)).isoformat()

    _write_ab(ab, [{
        "ts": entry_time,
        "ticker": "BTC",
        "primary": {"model": "CryptoBERT", "sentiment": "positive", "confidence": 0.7},
        "shadow": [
            {"model": "FinBERT", "sentiment": "neutral", "confidence": 0.5,
             "avg_scores": {"positive": 0.3, "negative": 0.2, "neutral": 0.5}},
            {"model": "fingpt:finance-llama-8b", "sentiment": "positive", "confidence": 0.6,
             "avg_scores": {"positive": 0.6, "negative": 0.1, "neutral": 0.3}},
        ],
    }])
    _write_snap(snap, [
        {"ts": entry_time, "prices": {"BTC-USD": 100.0}},
        {"ts": target_time, "prices": {"BTC-USD": 102.0}},
    ])

    stats = mod.backfill(ab_log_path=ab, outcomes_path=out, snapshot_path=snap)
    assert stats["outcomes_written"] == 3  # primary + 2 shadows
    assert stats["skipped_too_recent"] == 0

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 3

    by_model = {r["model"]: r for r in rows}
    # Realized move +2% → BUY
    assert by_model["CryptoBERT"]["outcome"] == "BUY"
    assert by_model["CryptoBERT"]["predicted_class"] == "BUY"
    assert by_model["CryptoBERT"]["correct"] is True
    assert by_model["CryptoBERT"]["kind"] == "primary"
    assert by_model["CryptoBERT"]["agreement_with_primary"] is None

    assert by_model["FinBERT"]["predicted_class"] == "HOLD"  # neutral → HOLD
    assert by_model["FinBERT"]["correct"] is False  # HOLD ≠ BUY
    assert by_model["FinBERT"]["kind"] == "shadow"
    assert by_model["FinBERT"]["agreement_with_primary"] is False  # HOLD ≠ BUY

    assert by_model["fingpt:finance-llama-8b"]["predicted_class"] == "BUY"
    assert by_model["fingpt:finance-llama-8b"]["correct"] is True
    assert by_model["fingpt:finance-llama-8b"]["agreement_with_primary"] is True


def test_ticker_short_form_expands(tmp_path):
    ab = tmp_path / "ab.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"

    now = datetime.now(UTC)
    entry_time = (now - timedelta(days=2)).isoformat()
    target_time = (datetime.fromisoformat(entry_time) + timedelta(hours=24)).isoformat()

    _write_ab(ab, [{
        "ts": entry_time,
        "ticker": "ETH",  # short form
        "primary": {"model": "CryptoBERT", "sentiment": "negative", "confidence": 0.6},
        "shadow": [],
    }])
    _write_snap(snap, [
        {"ts": entry_time, "prices": {"ETH-USD": 200.0}},
        {"ts": target_time, "prices": {"ETH-USD": 198.0}},
    ])

    stats = mod.backfill(ab_log_path=ab, outcomes_path=out, snapshot_path=snap)
    assert stats["outcomes_written"] == 1
    row = json.loads(out.read_text().splitlines()[0])
    assert row["ticker"] == "ETH-USD"  # expanded
    # -1% is below sell threshold → SELL
    assert row["outcome"] == "SELL"
    assert row["predicted_class"] == "SELL"
    assert row["correct"] is True


def test_too_recent_skipped_once_per_row(tmp_path):
    """Too-recent skip should count once per row, not per shadow."""
    ab = tmp_path / "ab.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    _write_snap(snap, [])
    now = datetime.now(UTC)
    _write_ab(ab, [{
        "ts": (now - timedelta(hours=1)).isoformat(),
        "ticker": "BTC",
        "primary": {"model": "CryptoBERT", "sentiment": "positive", "confidence": 0.7},
        "shadow": [
            {"model": "FinBERT", "sentiment": "neutral", "confidence": 0.5},
            {"model": "fingpt:x", "sentiment": "positive", "confidence": 0.6},
        ],
    }])
    stats = mod.backfill(ab_log_path=ab, outcomes_path=out, snapshot_path=snap)
    assert stats["outcomes_written"] == 0
    assert stats["skipped_too_recent"] == 1  # ← not 3


def test_backfill_is_idempotent(tmp_path):
    ab = tmp_path / "ab.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    now = datetime.now(UTC)
    entry_time = (now - timedelta(days=2)).isoformat()
    target_time = (datetime.fromisoformat(entry_time) + timedelta(hours=24)).isoformat()
    _write_ab(ab, [{
        "ts": entry_time, "ticker": "BTC",
        "primary": {"model": "CryptoBERT", "sentiment": "positive", "confidence": 0.7},
        "shadow": [{"model": "FinBERT", "sentiment": "neutral", "confidence": 0.5}],
    }])
    _write_snap(snap, [
        {"ts": entry_time, "prices": {"BTC-USD": 100.0}},
        {"ts": target_time, "prices": {"BTC-USD": 101.0}},
    ])
    mod.backfill(ab_log_path=ab, outcomes_path=out, snapshot_path=snap)
    stats = mod.backfill(ab_log_path=ab, outcomes_path=out, snapshot_path=snap)
    assert stats["outcomes_written"] == 0
    assert stats["skipped_already_present"] == 2


def test_compute_model_accuracy(tmp_path):
    out = tmp_path / "out.jsonl"
    now = datetime.now(UTC)
    # Three rows — FinBERT 2/3 correct, CryptoBERT 3/3 correct
    rows = [
        {"ts": now.isoformat(), "ticker": "BTC-USD", "model": "CryptoBERT",
         "kind": "primary", "correct": True, "agreement_with_primary": None,
         "horizon": "1d"},
        {"ts": now.isoformat(), "ticker": "BTC-USD", "model": "FinBERT",
         "kind": "shadow", "correct": True, "agreement_with_primary": True,
         "horizon": "1d"},
        {"ts": (now - timedelta(days=1)).isoformat(), "ticker": "BTC-USD",
         "model": "CryptoBERT", "kind": "primary", "correct": True,
         "agreement_with_primary": None, "horizon": "1d"},
        {"ts": (now - timedelta(days=1)).isoformat(), "ticker": "BTC-USD",
         "model": "FinBERT", "kind": "shadow", "correct": False,
         "agreement_with_primary": False, "horizon": "1d"},
        {"ts": (now - timedelta(days=2)).isoformat(), "ticker": "BTC-USD",
         "model": "CryptoBERT", "kind": "primary", "correct": True,
         "agreement_with_primary": None, "horizon": "1d"},
        {"ts": (now - timedelta(days=2)).isoformat(), "ticker": "BTC-USD",
         "model": "FinBERT", "kind": "shadow", "correct": True,
         "agreement_with_primary": True, "horizon": "1d"},
    ]
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    summary = mod.compute_model_accuracy(outcomes_path=out, days=30)
    assert summary["CryptoBERT"]["accuracy"] == pytest.approx(1.0)
    assert summary["CryptoBERT"]["samples"] == 3
    assert summary["CryptoBERT"]["agreement_with_primary"] is None
    assert summary["FinBERT"]["accuracy"] == pytest.approx(2 / 3)
    assert summary["FinBERT"]["agreement_with_primary"] == pytest.approx(2 / 3)


def test_compute_model_accuracy_empty_when_missing(tmp_path):
    assert mod.compute_model_accuracy(outcomes_path=tmp_path / "nope.jsonl") == {}


def test_backfill_respects_horizon_override(tmp_path):
    ab = tmp_path / "ab.jsonl"
    out = tmp_path / "out.jsonl"
    snap = tmp_path / "snap.jsonl"
    now = datetime.now(UTC)
    entry_time = (now - timedelta(days=4)).isoformat()
    target_3d = (datetime.fromisoformat(entry_time) + timedelta(hours=72)).isoformat()
    _write_ab(ab, [{
        "ts": entry_time, "ticker": "BTC",
        "primary": {"model": "CryptoBERT", "sentiment": "positive", "confidence": 0.7},
        "shadow": [],
    }])
    _write_snap(snap, [
        {"ts": entry_time, "prices": {"BTC-USD": 100.0}},
        {"ts": target_3d, "prices": {"BTC-USD": 105.0}},
    ])
    stats = mod.backfill(ab_log_path=ab, outcomes_path=out, snapshot_path=snap, horizon="3d")
    assert stats["outcomes_written"] == 1
    row = json.loads(out.read_text().splitlines()[0])
    assert row["horizon"] == "3d"
    assert row["outcome"] == "BUY"
