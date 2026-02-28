"""Tests for forecast_accuracy module.

Covers: load_predictions, load_health_stats, compute_forecast_accuracy,
backfill_forecast_outcomes, get_forecast_accuracy_summary,
print_forecast_accuracy_report, _lookup_price_at_time, _write_predictions.
"""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from portfolio.forecast_accuracy import (
    load_predictions,
    load_health_stats,
    compute_forecast_accuracy,
    backfill_forecast_outcomes,
    get_forecast_accuracy_summary,
    print_forecast_accuracy_report,
    _lookup_price_at_time,
    _write_predictions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path, entries):
    """Write a list of dicts as JSONL to the given path."""
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _ago(hours=0, days=0):
    return (datetime.now(timezone.utc) - timedelta(hours=hours, days=days)).isoformat()


# ---------------------------------------------------------------------------
# load_predictions
# ---------------------------------------------------------------------------

class TestLoadPredictions:
    def test_missing_file(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        assert load_predictions(path) == []

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert load_predictions(path) == []

    def test_blank_lines(self, tmp_path):
        path = tmp_path / "blank.jsonl"
        path.write_text("\n\n\n", encoding="utf-8")
        assert load_predictions(path) == []

    def test_valid_entries(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = [{"ticker": "BTC-USD", "ts": _now_iso()},
                   {"ticker": "ETH-USD", "ts": _now_iso()}]
        _write_jsonl(path, entries)
        result = load_predictions(path)
        assert len(result) == 2
        assert result[0]["ticker"] == "BTC-USD"

    def test_invalid_json_skipped(self, tmp_path):
        path = tmp_path / "mixed.jsonl"
        path.write_text(
            '{"ticker": "BTC-USD"}\n'
            'not json\n'
            '{"ticker": "ETH-USD"}\n',
            encoding="utf-8",
        )
        result = load_predictions(path)
        assert len(result) == 2

    def test_mixed_with_empty_lines(self, tmp_path):
        path = tmp_path / "mixed2.jsonl"
        path.write_text(
            '\n{"ticker": "BTC-USD"}\n\n{"ticker": "ETH-USD"}\n\n',
            encoding="utf-8",
        )
        result = load_predictions(path)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# load_health_stats
# ---------------------------------------------------------------------------

class TestLoadHealthStats:
    def test_missing_file(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        assert load_health_stats(path) == {}

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert load_health_stats(path) == {}

    def test_single_model_all_success(self, tmp_path):
        path = tmp_path / "health.jsonl"
        entries = [
            {"model": "chronos", "ok": True, "ms": 100},
            {"model": "chronos", "ok": True, "ms": 150},
        ]
        _write_jsonl(path, entries)
        result = load_health_stats(path)
        assert result["chronos"]["ok"] == 2
        assert result["chronos"]["fail"] == 0
        assert result["chronos"]["total"] == 2
        assert result["chronos"]["success_rate"] == 1.0

    def test_single_model_mixed(self, tmp_path):
        path = tmp_path / "health.jsonl"
        entries = [
            {"model": "kronos", "ok": True, "ms": 100},
            {"model": "kronos", "ok": False, "ms": 6000, "error": "timeout"},
            {"model": "kronos", "ok": False, "ms": 5500, "error": "empty"},
        ]
        _write_jsonl(path, entries)
        result = load_health_stats(path)
        assert result["kronos"]["ok"] == 1
        assert result["kronos"]["fail"] == 2
        assert result["kronos"]["total"] == 3
        assert result["kronos"]["success_rate"] == 0.333

    def test_multiple_models(self, tmp_path):
        path = tmp_path / "health.jsonl"
        entries = [
            {"model": "kronos", "ok": True, "ms": 100},
            {"model": "chronos", "ok": True, "ms": 50},
            {"model": "kronos", "ok": False, "ms": 6000},
            {"model": "chronos", "ok": True, "ms": 60},
        ]
        _write_jsonl(path, entries)
        result = load_health_stats(path)
        assert len(result) == 2
        assert result["kronos"]["total"] == 2
        assert result["chronos"]["total"] == 2
        assert result["chronos"]["success_rate"] == 1.0

    def test_invalid_json_skipped(self, tmp_path):
        path = tmp_path / "health.jsonl"
        path.write_text(
            '{"model": "kronos", "ok": true, "ms": 100}\n'
            'not json\n',
            encoding="utf-8",
        )
        result = load_health_stats(path)
        assert result["kronos"]["total"] == 1

    def test_missing_model_field(self, tmp_path):
        path = tmp_path / "health.jsonl"
        entries = [{"ok": True, "ms": 100}]
        _write_jsonl(path, entries)
        result = load_health_stats(path)
        assert "unknown" in result


# ---------------------------------------------------------------------------
# compute_forecast_accuracy
# ---------------------------------------------------------------------------

class TestComputeForecastAccuracy:
    def test_empty_predictions(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        path.write_text("", encoding="utf-8")
        result = compute_forecast_accuracy(predictions_file=path)
        assert result == {}

    def test_no_outcomes(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = [{"ticker": "BTC-USD", "ts": _now_iso(),
                    "sub_signals": {"chronos_24h": "BUY"}}]
        _write_jsonl(path, entries)
        result = compute_forecast_accuracy(predictions_file=path)
        assert result == {}

    def test_correct_prediction(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = [{
            "ticker": "BTC-USD",
            "ts": _ago(hours=25),
            "sub_signals": {"chronos_24h": "BUY"},
            "outcome": {"24h": {"change_pct": 2.5}},
        }]
        _write_jsonl(path, entries)
        result = compute_forecast_accuracy(predictions_file=path, horizon="24h")
        assert "chronos_24h" in result
        assert result["chronos_24h"]["accuracy"] == 1.0
        assert result["chronos_24h"]["correct"] == 1
        assert result["chronos_24h"]["total"] == 1

    def test_incorrect_prediction(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = [{
            "ticker": "BTC-USD",
            "ts": _ago(hours=25),
            "sub_signals": {"chronos_24h": "BUY"},
            "outcome": {"24h": {"change_pct": -1.5}},
        }]
        _write_jsonl(path, entries)
        result = compute_forecast_accuracy(predictions_file=path, horizon="24h")
        assert result["chronos_24h"]["accuracy"] == 0.0

    def test_hold_votes_excluded(self, tmp_path):
        """HOLD votes should not be counted as predictions."""
        path = tmp_path / "pred.jsonl"
        entries = [{
            "ticker": "BTC-USD",
            "ts": _ago(hours=25),
            "sub_signals": {"chronos_24h": "HOLD", "kronos_24h": "BUY"},
            "outcome": {"24h": {"change_pct": 2.0}},
        }]
        _write_jsonl(path, entries)
        result = compute_forecast_accuracy(predictions_file=path, horizon="24h")
        assert "chronos_24h" not in result  # HOLD excluded
        assert "kronos_24h" in result
        assert result["kronos_24h"]["accuracy"] == 1.0

    def test_ticker_filter(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = [
            {"ticker": "BTC-USD", "ts": _ago(hours=25),
             "sub_signals": {"chronos_24h": "BUY"},
             "outcome": {"24h": {"change_pct": 2.0}}},
            {"ticker": "ETH-USD", "ts": _ago(hours=25),
             "sub_signals": {"chronos_24h": "SELL"},
             "outcome": {"24h": {"change_pct": -1.0}}},
        ]
        _write_jsonl(path, entries)
        result = compute_forecast_accuracy(
            ticker="BTC-USD", predictions_file=path, horizon="24h"
        )
        assert "chronos_24h" in result
        assert result["chronos_24h"]["total"] == 1
        assert result["chronos_24h"]["by_ticker"].get("ETH-USD") is None

    def test_days_filter(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = [
            {"ticker": "BTC-USD", "ts": _ago(days=10),
             "sub_signals": {"chronos_24h": "BUY"},
             "outcome": {"24h": {"change_pct": 2.0}}},
            {"ticker": "BTC-USD", "ts": _ago(days=2),
             "sub_signals": {"chronos_24h": "SELL"},
             "outcome": {"24h": {"change_pct": -1.0}}},
        ]
        _write_jsonl(path, entries)
        result = compute_forecast_accuracy(
            days=7, predictions_file=path, horizon="24h"
        )
        # Only the 2-day-old entry should be counted
        assert result["chronos_24h"]["total"] == 1

    def test_multiple_entries_accuracy(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = [
            {"ticker": "XAG-USD", "ts": _ago(hours=25),
             "sub_signals": {"chronos_24h": "BUY"},
             "outcome": {"24h": {"change_pct": 1.0}}},
            {"ticker": "XAG-USD", "ts": _ago(hours=50),
             "sub_signals": {"chronos_24h": "BUY"},
             "outcome": {"24h": {"change_pct": 0.5}}},
            {"ticker": "XAG-USD", "ts": _ago(hours=75),
             "sub_signals": {"chronos_24h": "BUY"},
             "outcome": {"24h": {"change_pct": -0.3}}},
        ]
        _write_jsonl(path, entries)
        result = compute_forecast_accuracy(predictions_file=path, horizon="24h")
        # 2 correct (positive change), 1 incorrect (negative change)
        assert result["chronos_24h"]["correct"] == 2
        assert result["chronos_24h"]["total"] == 3
        assert abs(result["chronos_24h"]["accuracy"] - 0.667) < 0.01

    def test_by_ticker_breakdown(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = [
            {"ticker": "BTC-USD", "ts": _ago(hours=25),
             "sub_signals": {"chronos_24h": "BUY"},
             "outcome": {"24h": {"change_pct": 1.0}}},
            {"ticker": "XAG-USD", "ts": _ago(hours=25),
             "sub_signals": {"chronos_24h": "BUY"},
             "outcome": {"24h": {"change_pct": -0.5}}},
        ]
        _write_jsonl(path, entries)
        result = compute_forecast_accuracy(predictions_file=path, horizon="24h")
        by_ticker = result["chronos_24h"]["by_ticker"]
        assert by_ticker["BTC-USD"]["accuracy"] == 1.0
        assert by_ticker["XAG-USD"]["accuracy"] == 0.0

    def test_horizon_1h(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = [{
            "ticker": "BTC-USD",
            "ts": _ago(hours=2),
            "sub_signals": {"chronos_1h": "SELL", "chronos_24h": "BUY"},
            "outcome": {"1h": {"change_pct": -0.5}},
        }]
        _write_jsonl(path, entries)
        result = compute_forecast_accuracy(
            predictions_file=path, horizon="1h"
        )
        assert "chronos_1h" in result
        assert result["chronos_1h"]["accuracy"] == 1.0
        # chronos_24h should NOT be counted for 1h horizon
        assert "chronos_24h" not in result

    def test_horizon_filtering_excludes_wrong_horizon(self, tmp_path):
        """Sub-signals for a different horizon should be excluded."""
        path = tmp_path / "pred.jsonl"
        entries = [{
            "ticker": "BTC-USD",
            "ts": _ago(hours=25),
            "sub_signals": {"kronos_1h": "BUY", "kronos_24h": "SELL"},
            "outcome": {"24h": {"change_pct": -1.0}},
        }]
        _write_jsonl(path, entries)
        result = compute_forecast_accuracy(
            predictions_file=path, horizon="24h"
        )
        # kronos_1h should NOT be included in 24h evaluation
        assert "kronos_1h" not in result
        assert "kronos_24h" in result
        assert result["kronos_24h"]["accuracy"] == 1.0  # SELL + negative = correct

    def test_sell_prediction_correct(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = [{
            "ticker": "BTC-USD",
            "ts": _ago(hours=25),
            "sub_signals": {"chronos_24h": "SELL"},
            "outcome": {"24h": {"change_pct": -2.0}},
        }]
        _write_jsonl(path, entries)
        result = compute_forecast_accuracy(predictions_file=path, horizon="24h")
        assert result["chronos_24h"]["accuracy"] == 1.0

    def test_sell_prediction_incorrect(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        entries = [{
            "ticker": "BTC-USD",
            "ts": _ago(hours=25),
            "sub_signals": {"chronos_24h": "SELL"},
            "outcome": {"24h": {"change_pct": 3.0}},
        }]
        _write_jsonl(path, entries)
        result = compute_forecast_accuracy(predictions_file=path, horizon="24h")
        assert result["chronos_24h"]["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# _lookup_price_at_time
# ---------------------------------------------------------------------------

class TestLookupPriceAtTime:
    def test_missing_file(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        target = datetime.now(timezone.utc)
        assert _lookup_price_at_time("BTC-USD", target, snapshot_file=path) is None

    def test_empty_file(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        path.write_text("", encoding="utf-8")
        target = datetime.now(timezone.utc)
        assert _lookup_price_at_time("BTC-USD", target, snapshot_file=path) is None

    def test_exact_match(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        now = datetime.now(timezone.utc)
        snap = {"ts": now.isoformat(), "prices": {"BTC-USD": 67500.0}}
        _write_jsonl(path, [snap])
        result = _lookup_price_at_time("BTC-USD", now, snapshot_file=path)
        assert result == 67500.0

    def test_closest_within_tolerance(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        now = datetime.now(timezone.utc)
        # Snapshot 30 minutes before target
        snap_time = now - timedelta(minutes=30)
        snap = {"ts": snap_time.isoformat(), "prices": {"BTC-USD": 68000.0}}
        _write_jsonl(path, [snap])
        result = _lookup_price_at_time("BTC-USD", now, snapshot_file=path)
        assert result == 68000.0

    def test_outside_tolerance(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        now = datetime.now(timezone.utc)
        # Snapshot 3 hours before target (exceeds 2h tolerance)
        snap_time = now - timedelta(hours=3)
        snap = {"ts": snap_time.isoformat(), "prices": {"BTC-USD": 68000.0}}
        _write_jsonl(path, [snap])
        result = _lookup_price_at_time("BTC-USD", now, snapshot_file=path)
        assert result is None

    def test_ticker_not_in_snapshot(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        now = datetime.now(timezone.utc)
        snap = {"ts": now.isoformat(), "prices": {"ETH-USD": 2000.0}}
        _write_jsonl(path, [snap])
        result = _lookup_price_at_time("BTC-USD", now, snapshot_file=path)
        assert result is None

    def test_picks_closest(self, tmp_path):
        """When multiple snapshots exist, pick the closest to target time."""
        path = tmp_path / "snaps.jsonl"
        now = datetime.now(timezone.utc)
        snaps = [
            {"ts": (now - timedelta(hours=1, minutes=30)).isoformat(),
             "prices": {"BTC-USD": 66000.0}},
            {"ts": (now - timedelta(minutes=10)).isoformat(),
             "prices": {"BTC-USD": 67000.0}},
            {"ts": (now + timedelta(minutes=30)).isoformat(),
             "prices": {"BTC-USD": 68000.0}},
        ]
        _write_jsonl(path, snaps)
        result = _lookup_price_at_time("BTC-USD", now, snapshot_file=path)
        assert result == 67000.0  # 10 min away is closest

    def test_invalid_json_skipped(self, tmp_path):
        path = tmp_path / "snaps.jsonl"
        now = datetime.now(timezone.utc)
        path.write_text(
            'not json\n'
            f'{{"ts": "{now.isoformat()}", "prices": {{"BTC-USD": 67000.0}}}}\n',
            encoding="utf-8",
        )
        result = _lookup_price_at_time("BTC-USD", now, snapshot_file=path)
        assert result == 67000.0


# ---------------------------------------------------------------------------
# _write_predictions
# ---------------------------------------------------------------------------

class TestWritePredictions:
    def test_writes_jsonl(self, tmp_path):
        path = tmp_path / "out.jsonl"
        entries = [{"ticker": "BTC-USD"}, {"ticker": "ETH-USD"}]
        _write_predictions(entries, path)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["ticker"] == "BTC-USD"

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "out.jsonl"
        _write_predictions([{"a": 1}], path)
        _write_predictions([{"b": 2}], path)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"b": 2}

    def test_empty_list(self, tmp_path):
        path = tmp_path / "out.jsonl"
        _write_predictions([], path)
        assert path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# backfill_forecast_outcomes
# ---------------------------------------------------------------------------

class TestBackfillForecastOutcomes:
    def test_empty_file(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        path.write_text("", encoding="utf-8")
        assert backfill_forecast_outcomes(predictions_file=path) == 0

    def test_missing_file(self, tmp_path):
        path = tmp_path / "missing.jsonl"
        assert backfill_forecast_outcomes(predictions_file=path) == 0

    def test_already_backfilled(self, tmp_path):
        pred_path = tmp_path / "pred.jsonl"
        entries = [{
            "ticker": "BTC-USD",
            "ts": _ago(hours=25),
            "current_price": 67000.0,
            "outcome": {"1h": {"change_pct": 0.5}, "24h": {"change_pct": 1.0}},
        }]
        _write_jsonl(pred_path, entries)
        updated = backfill_forecast_outcomes(predictions_file=pred_path)
        assert updated == 0

    def test_backfills_with_snapshot(self, tmp_path):
        pred_path = tmp_path / "pred.jsonl"
        snap_path = tmp_path / "price_snapshots_hourly.jsonl"

        entry_time = datetime.now(timezone.utc) - timedelta(hours=26)
        entries = [{
            "ticker": "BTC-USD",
            "ts": entry_time.isoformat(),
            "current_price": 67000.0,
        }]
        _write_jsonl(pred_path, entries)

        # Snapshot at entry_time + 1h
        snap_1h = {
            "ts": (entry_time + timedelta(hours=1)).isoformat(),
            "prices": {"BTC-USD": 67500.0},
        }
        # Snapshot at entry_time + 24h
        snap_24h = {
            "ts": (entry_time + timedelta(hours=24)).isoformat(),
            "prices": {"BTC-USD": 68000.0},
        }
        _write_jsonl(snap_path, [snap_1h, snap_24h])

        updated = backfill_forecast_outcomes(
            predictions_file=pred_path, snapshot_file=snap_path
        )
        assert updated == 2  # 1h and 24h

        # Verify the written data
        result = load_predictions(pred_path)
        assert len(result) == 1
        assert "1h" in result[0]["outcome"]
        assert "24h" in result[0]["outcome"]
        assert abs(result[0]["outcome"]["1h"]["change_pct"] - 0.7463) < 0.01
        assert abs(result[0]["outcome"]["24h"]["change_pct"] - 1.4925) < 0.01

    def test_no_snapshot_available(self, tmp_path):
        pred_path = tmp_path / "pred.jsonl"
        snap_path = tmp_path / "price_snapshots_hourly.jsonl"

        entry_time = datetime.now(timezone.utc) - timedelta(hours=26)
        entries = [{
            "ticker": "BTC-USD",
            "ts": entry_time.isoformat(),
            "current_price": 67000.0,
        }]
        _write_jsonl(pred_path, entries)
        # Empty snapshot file
        snap_path.write_text("", encoding="utf-8")

        updated = backfill_forecast_outcomes(
            predictions_file=pred_path, snapshot_file=snap_path
        )
        assert updated == 0

    def test_max_entries_limit(self, tmp_path):
        pred_path = tmp_path / "pred.jsonl"
        snap_path = tmp_path / "price_snapshots_hourly.jsonl"

        entry_time = datetime.now(timezone.utc) - timedelta(hours=26)
        entries = []
        for i in range(5):
            entries.append({
                "ticker": "BTC-USD",
                "ts": (entry_time - timedelta(hours=i * 25)).isoformat(),
                "current_price": 67000.0 + i * 100,
            })
        _write_jsonl(pred_path, entries)

        # Create snapshots for all entries
        snaps = []
        for i in range(5):
            t = entry_time - timedelta(hours=i * 25)
            snaps.append({
                "ts": (t + timedelta(hours=1)).isoformat(),
                "prices": {"BTC-USD": 67100.0 + i * 100},
            })
            snaps.append({
                "ts": (t + timedelta(hours=24)).isoformat(),
                "prices": {"BTC-USD": 67500.0 + i * 100},
            })
        _write_jsonl(snap_path, snaps)

        updated = backfill_forecast_outcomes(
            max_entries=2, predictions_file=pred_path, snapshot_file=snap_path
        )
        assert updated == 2

    def test_skips_entry_without_ts(self, tmp_path):
        pred_path = tmp_path / "pred.jsonl"
        entries = [{"ticker": "BTC-USD", "current_price": 67000.0}]
        _write_jsonl(pred_path, entries)
        updated = backfill_forecast_outcomes(predictions_file=pred_path)
        assert updated == 0

    def test_skips_entry_without_price(self, tmp_path):
        pred_path = tmp_path / "pred.jsonl"
        entries = [{"ticker": "BTC-USD", "ts": _ago(hours=25)}]
        _write_jsonl(pred_path, entries)
        updated = backfill_forecast_outcomes(predictions_file=pred_path)
        assert updated == 0

    def test_skips_entry_without_ticker(self, tmp_path):
        pred_path = tmp_path / "pred.jsonl"
        entries = [{"ts": _ago(hours=25), "current_price": 67000.0}]
        _write_jsonl(pred_path, entries)
        updated = backfill_forecast_outcomes(predictions_file=pred_path)
        assert updated == 0

    def test_future_entries_not_backfilled(self, tmp_path):
        pred_path = tmp_path / "pred.jsonl"
        # Entry from 10 minutes ago -- not enough time for 1h or 24h
        entries = [{
            "ticker": "BTC-USD",
            "ts": _ago(hours=0),
            "current_price": 67000.0,
        }]
        _write_jsonl(pred_path, entries)
        updated = backfill_forecast_outcomes(predictions_file=pred_path)
        assert updated == 0


# ---------------------------------------------------------------------------
# get_forecast_accuracy_summary
# ---------------------------------------------------------------------------

class TestGetForecastAccuracySummary:
    def test_empty_data(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "portfolio.forecast_accuracy.HEALTH_FILE", tmp_path / "health.jsonl"
        )
        monkeypatch.setattr(
            "portfolio.forecast_accuracy.PREDICTIONS_FILE", tmp_path / "pred.jsonl"
        )
        result = get_forecast_accuracy_summary()
        assert result["health"] == {}
        assert result["accuracy"] == {}

    def test_with_health_and_accuracy(self, tmp_path, monkeypatch):
        health_path = tmp_path / "health.jsonl"
        pred_path = tmp_path / "pred.jsonl"
        monkeypatch.setattr(
            "portfolio.forecast_accuracy.HEALTH_FILE", health_path
        )
        monkeypatch.setattr(
            "portfolio.forecast_accuracy.PREDICTIONS_FILE", pred_path
        )

        _write_jsonl(health_path, [
            {"model": "chronos", "ok": True, "ms": 100},
            {"model": "kronos", "ok": False, "ms": 6000},
        ])
        _write_jsonl(pred_path, [{
            "ticker": "XAG-USD",
            "ts": _ago(hours=25),
            "sub_signals": {"chronos_24h": "BUY"},
            "outcome": {"24h": {"change_pct": 2.0}},
        }])

        result = get_forecast_accuracy_summary(days=7)
        assert "chronos" in result["health"]
        assert "kronos" in result["health"]
        assert "chronos_24h" in result["accuracy"]
        assert result["accuracy"]["chronos_24h"]["accuracy"] == 1.0

    def test_focus_tickers_filter(self, tmp_path, monkeypatch):
        health_path = tmp_path / "health.jsonl"
        pred_path = tmp_path / "pred.jsonl"
        monkeypatch.setattr(
            "portfolio.forecast_accuracy.HEALTH_FILE", health_path
        )
        monkeypatch.setattr(
            "portfolio.forecast_accuracy.PREDICTIONS_FILE", pred_path
        )

        health_path.write_text("", encoding="utf-8")
        _write_jsonl(pred_path, [
            {"ticker": "XAG-USD", "ts": _ago(hours=25),
             "sub_signals": {"chronos_24h": "BUY"},
             "outcome": {"24h": {"change_pct": 2.0}}},
            {"ticker": "BTC-USD", "ts": _ago(hours=25),
             "sub_signals": {"chronos_24h": "SELL"},
             "outcome": {"24h": {"change_pct": -1.0}}},
        ])

        result = get_forecast_accuracy_summary(
            focus_tickers=["XAG-USD"], days=30
        )
        acc = result["accuracy"]["chronos_24h"]
        # Both entries counted in overall accuracy
        assert acc["samples"] == 2
        # But by_ticker only shows XAG-USD
        assert "XAG-USD" in acc["by_ticker"]
        assert "BTC-USD" not in acc["by_ticker"]


# ---------------------------------------------------------------------------
# print_forecast_accuracy_report
# ---------------------------------------------------------------------------

class TestPrintForecastAccuracyReport:
    def test_no_data(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            "portfolio.forecast_accuracy.HEALTH_FILE", tmp_path / "health.jsonl"
        )
        monkeypatch.setattr(
            "portfolio.forecast_accuracy.PREDICTIONS_FILE", tmp_path / "pred.jsonl"
        )
        print_forecast_accuracy_report()
        captured = capsys.readouterr()
        assert "No health data" in captured.out
        assert "No outcome data" in captured.out

    def test_with_data(self, tmp_path, monkeypatch, capsys):
        health_path = tmp_path / "health.jsonl"
        pred_path = tmp_path / "pred.jsonl"
        monkeypatch.setattr(
            "portfolio.forecast_accuracy.HEALTH_FILE", health_path
        )
        monkeypatch.setattr(
            "portfolio.forecast_accuracy.PREDICTIONS_FILE", pred_path
        )

        _write_jsonl(health_path, [
            {"model": "chronos", "ok": True, "ms": 100},
        ])
        _write_jsonl(pred_path, [{
            "ticker": "XAG-USD",
            "ts": _ago(hours=25),
            "sub_signals": {"chronos_24h": "BUY"},
            "outcome": {"24h": {"change_pct": 2.0}},
        }])

        print_forecast_accuracy_report()
        captured = capsys.readouterr()
        assert "chronos" in captured.out
        assert "100.0%" in captured.out
        assert "XAG-USD" in captured.out
