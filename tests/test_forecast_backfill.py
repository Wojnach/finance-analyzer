"""Tests for forecast outcome backfill logic.

Tests backfill_forecast_outcomes() which reads forecast_predictions.jsonl,
looks up actual prices at horizon times, and writes outcomes back.

Complements test_forecast_accuracy.py with additional edge-case coverage
for the backfill pipeline, idempotency, partial fills, and file I/O.
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from portfolio.forecast_accuracy import (
    backfill_forecast_outcomes,
    load_predictions,
    _lookup_price_at_time,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path, entries):
    """Write a list of dicts as JSONL to the given path."""
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    """Read a JSONL file and return list of dicts."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def _ago(hours=0, days=0):
    return (datetime.now(timezone.utc) - timedelta(hours=hours, days=days)).isoformat()


def _make_prediction(ticker="BTC-USD", hours_ago=26, price=67000.0,
                     sub_signals=None, outcome=None):
    """Create a prediction entry at the given time offset."""
    entry = {
        "ts": _ago(hours=hours_ago),
        "ticker": ticker,
        "current_price": price,
        "sub_signals": sub_signals or {"chronos_1h": "BUY", "chronos_24h": "BUY"},
        "action": "BUY",
        "confidence": 0.5,
    }
    if outcome is not None:
        entry["outcome"] = outcome
    return entry


def _make_snapshot(ticker, price, hours_ago=0):
    """Create a price snapshot entry."""
    return {
        "ts": _ago(hours=hours_ago),
        "prices": {ticker: price},
    }


@pytest.fixture
def setup_files(tmp_path):
    """Create temp prediction and snapshot file paths."""
    pred_file = tmp_path / "forecast_predictions.jsonl"
    snap_file = tmp_path / "price_snapshots_hourly.jsonl"
    return pred_file, snap_file, tmp_path


# ---------------------------------------------------------------------------
# TestBackfillForecasts
# ---------------------------------------------------------------------------

class TestBackfillForecasts:
    """Core tests for backfill_forecast_outcomes()."""

    def test_empty_file_returns_zero(self, setup_files):
        pred_file, snap_file, tmp_dir = setup_files
        pred_file.write_text("", encoding="utf-8")
        result = backfill_forecast_outcomes(predictions_file=pred_file)
        assert result == 0

    def test_missing_file_returns_zero(self, setup_files):
        pred_file, snap_file, tmp_dir = setup_files
        # pred_file does not exist
        result = backfill_forecast_outcomes(
            predictions_file=tmp_dir / "nonexistent.jsonl"
        )
        assert result == 0

    def test_all_already_backfilled_returns_zero(self, setup_files):
        pred_file, snap_file, tmp_dir = setup_files
        entries = [_make_prediction(outcome={
            "1h": {"actual_price": 67100.0, "change_pct": 0.149},
            "24h": {"actual_price": 67500.0, "change_pct": 0.746},
        })]
        _write_jsonl(pred_file, entries)
        result = backfill_forecast_outcomes(predictions_file=pred_file)
        assert result == 0

    def test_backfills_1h_outcome(self, setup_files):
        """Prediction from 2 hours ago should get 1h backfilled."""
        pred_file, snap_file, tmp_dir = setup_files
        entry_time = datetime.now(timezone.utc) - timedelta(hours=2)
        pred = {
            "ts": entry_time.isoformat(),
            "ticker": "XAG-USD",
            "current_price": 30.0,
            "sub_signals": {"chronos_1h": "BUY", "chronos_24h": "BUY"},
            "action": "BUY",
            "confidence": 0.5,
        }
        _write_jsonl(pred_file, [pred])

        # Snapshot at horizon time (1h after prediction)
        snap_time = entry_time + timedelta(hours=1)
        snap = {"ts": snap_time.isoformat(), "prices": {"XAG-USD": 30.5}}
        _write_jsonl(snap_file, [snap])

        result = backfill_forecast_outcomes(
            predictions_file=pred_file, snapshot_file=snap_file
        )
        assert result == 1

        updated = _read_jsonl(pred_file)
        assert len(updated) == 1
        assert "1h" in updated[0]["outcome"]
        assert updated[0]["outcome"]["1h"]["actual_price"] == 30.5
        assert updated[0]["outcome"]["1h"]["change_pct"] == pytest.approx(
            1.6667, rel=0.01
        )

    def test_too_recent_not_backfilled(self, setup_files):
        """Prediction from 30 minutes ago should not be backfilled for any horizon."""
        pred_file, snap_file, tmp_dir = setup_files
        entries = [_make_prediction(hours_ago=0.5)]
        _write_jsonl(pred_file, entries)
        result = backfill_forecast_outcomes(predictions_file=pred_file)
        assert result == 0

    def test_max_entries_limit_respected(self, setup_files):
        """max_entries stops processing after the entry where the count is reached.

        Each entry can fill up to 2 horizons (1h + 24h), so the update count
        increments per-horizon. The break happens after the entry loop, meaning
        the entry that crosses the threshold finishes both horizons first.
        With 10 entries * 2 horizons = 20 possible, max_entries=3 should stop
        after the 2nd entry (which reaches 4 >= 3), yielding 4.
        """
        pred_file, snap_file, tmp_dir = setup_files

        base_time = datetime.now(timezone.utc) - timedelta(hours=30)
        preds = []
        snaps = []
        for i in range(10):
            t = base_time - timedelta(hours=i * 30)
            preds.append({
                "ts": t.isoformat(),
                "ticker": "BTC-USD",
                "current_price": 67000.0,
            })
            snaps.append({
                "ts": (t + timedelta(hours=1)).isoformat(),
                "prices": {"BTC-USD": 67100.0},
            })
            snaps.append({
                "ts": (t + timedelta(hours=24)).isoformat(),
                "prices": {"BTC-USD": 67500.0},
            })
        _write_jsonl(pred_file, preds)
        _write_jsonl(snap_file, snaps)

        result = backfill_forecast_outcomes(
            max_entries=3, predictions_file=pred_file, snapshot_file=snap_file
        )
        # First entry: 1h + 24h = 2 updates. Second entry: 1h + 24h = 4 updates.
        # 4 >= 3, so break after 2nd entry. Result is 4, not 3.
        assert result == 4
        # Verify it did NOT process all 10 entries (20 updates)
        assert result < 20

    def test_missing_ticker_skips(self, setup_files):
        pred_file, snap_file, tmp_dir = setup_files
        entries = [{"ts": _ago(hours=26), "current_price": 67000.0}]
        _write_jsonl(pred_file, entries)
        result = backfill_forecast_outcomes(predictions_file=pred_file)
        assert result == 0

    def test_missing_current_price_skips(self, setup_files):
        pred_file, snap_file, tmp_dir = setup_files
        entries = [{"ts": _ago(hours=26), "ticker": "BTC-USD"}]
        _write_jsonl(pred_file, entries)
        result = backfill_forecast_outcomes(predictions_file=pred_file)
        assert result == 0

    def test_missing_timestamp_skips(self, setup_files):
        pred_file, snap_file, tmp_dir = setup_files
        entries = [{"ticker": "BTC-USD", "current_price": 67000.0}]
        _write_jsonl(pred_file, entries)
        result = backfill_forecast_outcomes(predictions_file=pred_file)
        assert result == 0

    def test_idempotent_no_double_backfill(self, setup_files):
        """Running backfill twice should not re-backfill already-filled entries."""
        pred_file, snap_file, tmp_dir = setup_files

        entry_time = datetime.now(timezone.utc) - timedelta(hours=26)
        preds = [{
            "ts": entry_time.isoformat(),
            "ticker": "BTC-USD",
            "current_price": 67000.0,
        }]
        _write_jsonl(pred_file, preds)

        snaps = [
            {"ts": (entry_time + timedelta(hours=1)).isoformat(),
             "prices": {"BTC-USD": 67100.0}},
            {"ts": (entry_time + timedelta(hours=24)).isoformat(),
             "prices": {"BTC-USD": 67500.0}},
        ]
        _write_jsonl(snap_file, snaps)

        # First run
        first_result = backfill_forecast_outcomes(
            predictions_file=pred_file, snapshot_file=snap_file
        )
        assert first_result == 2  # 1h + 24h

        # Second run -- nothing to do
        second_result = backfill_forecast_outcomes(
            predictions_file=pred_file, snapshot_file=snap_file
        )
        assert second_result == 0

    def test_partial_backfill_only_fills_missing(self, setup_files):
        """If 1h is already filled but 24h is not, only fill 24h."""
        pred_file, snap_file, tmp_dir = setup_files

        entry_time = datetime.now(timezone.utc) - timedelta(hours=26)
        preds = [{
            "ts": entry_time.isoformat(),
            "ticker": "BTC-USD",
            "current_price": 67000.0,
            "outcome": {
                "1h": {"actual_price": 67100.0, "change_pct": 0.149},
            },
        }]
        _write_jsonl(pred_file, preds)

        snaps = [{
            "ts": (entry_time + timedelta(hours=24)).isoformat(),
            "prices": {"BTC-USD": 68000.0},
        }]
        _write_jsonl(snap_file, snaps)

        result = backfill_forecast_outcomes(
            predictions_file=pred_file, snapshot_file=snap_file
        )
        assert result == 1  # only 24h filled

        updated = _read_jsonl(pred_file)
        assert updated[0]["outcome"]["1h"]["actual_price"] == 67100.0  # untouched
        assert updated[0]["outcome"]["24h"]["actual_price"] == 68000.0  # newly filled

    def test_backfill_writes_file_correctly(self, setup_files):
        """Verify the output file has correct JSONL format after backfill."""
        pred_file, snap_file, tmp_dir = setup_files

        entry_time = datetime.now(timezone.utc) - timedelta(hours=3)
        preds = [
            {
                "ts": entry_time.isoformat(),
                "ticker": "BTC-USD",
                "current_price": 67000.0,
            },
            {
                "ts": (entry_time - timedelta(hours=1)).isoformat(),
                "ticker": "ETH-USD",
                "current_price": 2000.0,
            },
        ]
        _write_jsonl(pred_file, preds)

        snaps = [
            {"ts": (entry_time + timedelta(hours=1)).isoformat(),
             "prices": {"BTC-USD": 67200.0, "ETH-USD": 2010.0}},
        ]
        _write_jsonl(snap_file, snaps)

        backfill_forecast_outcomes(
            predictions_file=pred_file, snapshot_file=snap_file
        )

        # Verify valid JSONL
        lines = pred_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            parsed = json.loads(line)
            assert "ticker" in parsed

    def test_invalid_timestamp_format_skips(self, setup_files):
        """Entry with non-ISO timestamp should be skipped without error."""
        pred_file, snap_file, tmp_dir = setup_files
        entries = [{
            "ts": "not-a-timestamp",
            "ticker": "BTC-USD",
            "current_price": 67000.0,
        }]
        _write_jsonl(pred_file, entries)
        result = backfill_forecast_outcomes(predictions_file=pred_file)
        assert result == 0

    def test_no_price_snapshots_available(self, setup_files):
        """When snapshot file does not exist, no backfill occurs."""
        pred_file, snap_file, tmp_dir = setup_files
        entries = [_make_prediction(hours_ago=26)]
        _write_jsonl(pred_file, entries)
        # snap_file not created -- _lookup_price_at_time will look in DATA_DIR
        # by default, but with snapshot_file param we can point to missing file
        missing_snap = tmp_dir / "nonexistent_snaps.jsonl"
        result = backfill_forecast_outcomes(
            predictions_file=pred_file, snapshot_file=missing_snap
        )
        assert result == 0

    def test_snapshot_outside_tolerance_not_used(self, setup_files):
        """Snapshot more than 2h from target time should not be used."""
        pred_file, snap_file, tmp_dir = setup_files

        entry_time = datetime.now(timezone.utc) - timedelta(hours=5)
        preds = [{
            "ts": entry_time.isoformat(),
            "ticker": "BTC-USD",
            "current_price": 67000.0,
        }]
        _write_jsonl(pred_file, preds)

        # Snapshot at entry_time + 4h (3h away from 1h horizon target)
        bad_snap = {
            "ts": (entry_time + timedelta(hours=4)).isoformat(),
            "prices": {"BTC-USD": 67500.0},
        }
        _write_jsonl(snap_file, [bad_snap])

        result = backfill_forecast_outcomes(
            predictions_file=pred_file, snapshot_file=snap_file
        )
        assert result == 0

    def test_correct_change_pct_calculation(self, setup_files):
        """Verify change_pct = (actual - current) / current * 100."""
        pred_file, snap_file, tmp_dir = setup_files

        entry_time = datetime.now(timezone.utc) - timedelta(hours=3)
        preds = [{
            "ts": entry_time.isoformat(),
            "ticker": "XAG-USD",
            "current_price": 30.0,
        }]
        _write_jsonl(pred_file, preds)

        snaps = [{
            "ts": (entry_time + timedelta(hours=1)).isoformat(),
            "prices": {"XAG-USD": 31.5},
        }]
        _write_jsonl(snap_file, snaps)

        backfill_forecast_outcomes(
            predictions_file=pred_file, snapshot_file=snap_file
        )

        updated = _read_jsonl(pred_file)
        # (31.5 - 30.0) / 30.0 * 100 = 5.0
        assert updated[0]["outcome"]["1h"]["change_pct"] == pytest.approx(5.0, rel=0.01)

    def test_negative_price_change_correctly_computed(self, setup_files):
        pred_file, snap_file, tmp_dir = setup_files

        entry_time = datetime.now(timezone.utc) - timedelta(hours=3)
        preds = [{
            "ts": entry_time.isoformat(),
            "ticker": "ETH-USD",
            "current_price": 2000.0,
        }]
        _write_jsonl(pred_file, preds)

        snaps = [{
            "ts": (entry_time + timedelta(hours=1)).isoformat(),
            "prices": {"ETH-USD": 1900.0},
        }]
        _write_jsonl(snap_file, snaps)

        backfill_forecast_outcomes(
            predictions_file=pred_file, snapshot_file=snap_file
        )

        updated = _read_jsonl(pred_file)
        # (1900 - 2000) / 2000 * 100 = -5.0
        assert updated[0]["outcome"]["1h"]["change_pct"] == pytest.approx(
            -5.0, rel=0.01
        )

    def test_both_1h_and_24h_backfilled_in_single_pass(self, setup_files):
        pred_file, snap_file, tmp_dir = setup_files

        entry_time = datetime.now(timezone.utc) - timedelta(hours=26)
        preds = [{
            "ts": entry_time.isoformat(),
            "ticker": "BTC-USD",
            "current_price": 67000.0,
        }]
        _write_jsonl(pred_file, preds)

        snaps = [
            {"ts": (entry_time + timedelta(hours=1)).isoformat(),
             "prices": {"BTC-USD": 67200.0}},
            {"ts": (entry_time + timedelta(hours=24)).isoformat(),
             "prices": {"BTC-USD": 68000.0}},
        ]
        _write_jsonl(snap_file, snaps)

        result = backfill_forecast_outcomes(
            predictions_file=pred_file, snapshot_file=snap_file
        )
        assert result == 2  # both horizons

        updated = _read_jsonl(pred_file)
        assert "1h" in updated[0]["outcome"]
        assert "24h" in updated[0]["outcome"]
        assert updated[0]["outcome"]["1h"]["actual_price"] == 67200.0
        assert updated[0]["outcome"]["24h"]["actual_price"] == 68000.0

    def test_multiple_tickers_in_same_file(self, setup_files):
        pred_file, snap_file, tmp_dir = setup_files

        entry_time = datetime.now(timezone.utc) - timedelta(hours=3)
        preds = [
            {
                "ts": entry_time.isoformat(),
                "ticker": "BTC-USD",
                "current_price": 67000.0,
            },
            {
                "ts": entry_time.isoformat(),
                "ticker": "XAG-USD",
                "current_price": 30.0,
            },
        ]
        _write_jsonl(pred_file, preds)

        snaps = [{
            "ts": (entry_time + timedelta(hours=1)).isoformat(),
            "prices": {"BTC-USD": 67500.0, "XAG-USD": 30.3},
        }]
        _write_jsonl(snap_file, snaps)

        result = backfill_forecast_outcomes(
            predictions_file=pred_file, snapshot_file=snap_file
        )
        assert result == 2  # 1h for BTC + 1h for XAG

        updated = _read_jsonl(pred_file)
        assert updated[0]["outcome"]["1h"]["actual_price"] == 67500.0
        assert updated[1]["outcome"]["1h"]["actual_price"] == 30.3

    def test_writes_backfilled_at_timestamp(self, setup_files):
        """Each backfilled outcome should contain a backfilled_at ISO timestamp."""
        pred_file, snap_file, tmp_dir = setup_files

        entry_time = datetime.now(timezone.utc) - timedelta(hours=3)
        preds = [{
            "ts": entry_time.isoformat(),
            "ticker": "BTC-USD",
            "current_price": 67000.0,
        }]
        _write_jsonl(pred_file, preds)

        snaps = [{
            "ts": (entry_time + timedelta(hours=1)).isoformat(),
            "prices": {"BTC-USD": 67100.0},
        }]
        _write_jsonl(snap_file, snaps)

        before = datetime.now(timezone.utc)
        backfill_forecast_outcomes(
            predictions_file=pred_file, snapshot_file=snap_file
        )
        after = datetime.now(timezone.utc)

        updated = _read_jsonl(pred_file)
        bf_at = updated[0]["outcome"]["1h"]["backfilled_at"]
        bf_time = datetime.fromisoformat(bf_at)
        # backfilled_at should be between before and after
        assert before <= bf_time <= after

    def test_zero_price_entry_skipped(self, setup_files):
        """Entry with current_price = 0 should be skipped (falsy)."""
        pred_file, snap_file, tmp_dir = setup_files
        entries = [{
            "ts": _ago(hours=26),
            "ticker": "BTC-USD",
            "current_price": 0,
        }]
        _write_jsonl(pred_file, entries)
        result = backfill_forecast_outcomes(predictions_file=pred_file)
        assert result == 0

    def test_empty_ticker_skipped(self, setup_files):
        """Entry with empty string ticker should be skipped."""
        pred_file, snap_file, tmp_dir = setup_files
        entries = [{
            "ts": _ago(hours=26),
            "ticker": "",
            "current_price": 67000.0,
        }]
        _write_jsonl(pred_file, entries)
        result = backfill_forecast_outcomes(predictions_file=pred_file)
        assert result == 0


# ---------------------------------------------------------------------------
# TestLookupPriceAtTimeEdgeCases
# ---------------------------------------------------------------------------

class TestLookupPriceAtTimeEdgeCases:
    """Additional edge cases for _lookup_price_at_time beyond test_forecast_accuracy.py."""

    def test_empty_snapshot_file(self, tmp_path):
        snap_file = tmp_path / "snaps.jsonl"
        snap_file.write_text("", encoding="utf-8")
        target = datetime.now(timezone.utc)
        result = _lookup_price_at_time("BTC-USD", target, snapshot_file=snap_file)
        assert result is None

    def test_picks_closest_from_multiple(self, tmp_path):
        """When multiple snapshots are within tolerance, pick the closest."""
        snap_file = tmp_path / "snaps.jsonl"
        target = datetime.now(timezone.utc)
        snaps = [
            {"ts": (target - timedelta(minutes=90)).isoformat(),
             "prices": {"BTC-USD": 66000.0}},
            {"ts": (target - timedelta(minutes=30)).isoformat(),
             "prices": {"BTC-USD": 67000.0}},
            {"ts": (target - timedelta(minutes=5)).isoformat(),
             "prices": {"BTC-USD": 67500.0}},
        ]
        _write_jsonl(snap_file, snaps)
        result = _lookup_price_at_time("BTC-USD", target, snapshot_file=snap_file)
        assert result == 67500.0  # 5 min away, closest

    def test_boundary_exactly_2h_tolerance(self, tmp_path):
        """Snapshot exactly at the 2h boundary is NOT included (strict <)."""
        snap_file = tmp_path / "snaps.jsonl"
        target = datetime.now(timezone.utc)
        snap = {
            "ts": (target - timedelta(hours=2)).isoformat(),
            "prices": {"BTC-USD": 67000.0},
        }
        _write_jsonl(snap_file, [snap])
        result = _lookup_price_at_time("BTC-USD", target, snapshot_file=snap_file)
        # timedelta comparison: exactly 2h == 2h, not strictly less, so None
        assert result is None

    def test_snapshot_just_under_2h_included(self, tmp_path):
        """Snapshot at 1h59m should be within tolerance."""
        snap_file = tmp_path / "snaps.jsonl"
        target = datetime.now(timezone.utc)
        snap = {
            "ts": (target - timedelta(hours=1, minutes=59)).isoformat(),
            "prices": {"BTC-USD": 67000.0},
        }
        _write_jsonl(snap_file, [snap])
        result = _lookup_price_at_time("BTC-USD", target, snapshot_file=snap_file)
        assert result == 67000.0
