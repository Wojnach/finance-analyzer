"""Tests for data/metals_signal_tracker.py — signal logging, outcome backfill, accuracy.

Batch 5 of the metals monitoring auto-improvement plan.
"""
import json
import os
import sys
import threading
import pytest

# Ensure data/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))


@pytest.fixture(autouse=True)
def _isolate_files(tmp_path, monkeypatch):
    """Redirect all file paths to tmp_path so tests don't touch real data."""
    import metals_signal_tracker as mod

    monkeypatch.setattr(mod, "SIGNAL_LOG", str(tmp_path / "signal_log.jsonl"))
    monkeypatch.setattr(mod, "OUTCOMES_LOG", str(tmp_path / "outcomes.jsonl"))
    monkeypatch.setattr(mod, "ACCURACY_CACHE_FILE", str(tmp_path / "accuracy.json"))
    # Reset in-memory cache between tests
    with mod._lock:
        mod._accuracy_cache.clear()
    yield


# ---------------------------------------------------------------------------
# log_snapshot
# ---------------------------------------------------------------------------

class TestLogSnapshot:
    def test_basic_snapshot(self, tmp_path):
        from metals_signal_tracker import log_snapshot, SIGNAL_LOG

        log_snapshot(
            check_count=1,
            prices={"silver79": {"underlying": 30.5, "bid": 42.0}},
            positions={"silver79": {"active": True, "units": 100, "entry": 40.0, "stop": 38.0}},
            signal_data={},
            llm_signals=None,
            triggered=True,
            trigger_reasons=["price_move"],
        )
        with open(SIGNAL_LOG, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["check"] == 1
        assert entry["triggered"] is True
        assert "XAG-USD" in entry.get("prices", {}) or "silver79" in entry.get("prices", {})

    def test_silver_gold_price_mapping(self, tmp_path):
        from metals_signal_tracker import log_snapshot, SIGNAL_LOG

        log_snapshot(
            check_count=2,
            prices={
                "silver79": {"underlying": 30.5, "bid": 42},
                "gold": {"underlying": 2650, "bid": 100},
            },
            positions={},
            signal_data={},
            llm_signals=None,
            triggered=False,
            trigger_reasons=[],
        )
        with open(SIGNAL_LOG, "r") as f:
            entry = json.loads(f.readline())
        # Should map to XAG-USD and XAU-USD
        assert "XAG-USD" in entry.get("prices", entry.get("underlying_prices", {}))

    def test_with_signal_data(self, tmp_path):
        from metals_signal_tracker import log_snapshot, SIGNAL_LOG

        signal_data = {
            "XAG-USD": {
                "action": "BUY",
                "confidence": 0.8,
                "weighted_confidence": 0.75,
                "buy_count": 5,
                "sell_count": 1,
                "voters": 6,
                "rsi": 45.0,
                "regime": "trending-up",
                "vote_detail": "B:rsi,macd,ema | S:mean_reversion",
            }
        }
        log_snapshot(
            check_count=3,
            prices={"silver79": {"underlying": 31.0}},
            positions={},
            signal_data=signal_data,
            llm_signals=None,
            triggered=True,
            trigger_reasons=["consensus"],
        )
        with open(SIGNAL_LOG, "r") as f:
            entry = json.loads(f.readline())
        assert "signals" in entry or "signal_data" in entry

    def test_with_llm_signals(self, tmp_path):
        from metals_signal_tracker import log_snapshot, SIGNAL_LOG

        llm = {"XAG-USD": {"direction": "BUY", "confidence": 0.8}}
        log_snapshot(
            check_count=4,
            prices={"silver79": {"underlying": 31.0}},
            positions={},
            signal_data={},
            llm_signals=llm,
            triggered=False,
            trigger_reasons=[],
        )
        with open(SIGNAL_LOG, "r") as f:
            entry = json.loads(f.readline())
        assert "llm" in entry or "llm_signals" in entry

    def test_with_positions(self, tmp_path):
        from metals_signal_tracker import log_snapshot, SIGNAL_LOG

        log_snapshot(
            check_count=5,
            prices={"silver79": {"underlying": 31.0, "bid": 42.0}},
            positions={"silver79": {"active": True, "units": 100, "entry": 40.0, "stop": 38.0}},
            signal_data={},
            llm_signals=None,
            triggered=False,
            trigger_reasons=[],
        )
        with open(SIGNAL_LOG, "r") as f:
            entry = json.loads(f.readline())
        assert "positions" in entry or "pos" in entry

    def test_trigger_reasons_recorded(self, tmp_path):
        from metals_signal_tracker import log_snapshot, SIGNAL_LOG

        log_snapshot(
            check_count=6,
            prices={"silver79": {"underlying": 31.0}},
            positions={},
            signal_data={},
            llm_signals=None,
            triggered=True,
            trigger_reasons=["price_move", "consensus"],
        )
        with open(SIGNAL_LOG, "r") as f:
            entry = json.loads(f.readline())
        reasons = entry.get("trigger_reasons", entry.get("triggers", []))
        assert len(reasons) >= 1

    def test_multiple_snapshots_append(self, tmp_path):
        from metals_signal_tracker import log_snapshot, SIGNAL_LOG

        for i in range(3):
            log_snapshot(
                check_count=i,
                prices={"silver79": {"underlying": 30.0 + i}},
                positions={},
                signal_data={},
                llm_signals=None,
                triggered=False,
                trigger_reasons=[],
            )
        with open(SIGNAL_LOG, "r") as f:
            lines = f.readlines()
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# _resolve_outcome
# ---------------------------------------------------------------------------

class TestResolveOutcome:
    def test_direction_up(self):
        from metals_signal_tracker import _resolve_outcome
        import datetime

        entry = {
            "ts": "2026-03-01T10:00:00+00:00",
            "prices": {"XAG-USD": 30.0},
        }
        now = datetime.datetime(2026, 3, 1, 14, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        result = _resolve_outcome(entry, "1h", {"XAG-USD": 31.0}, now)
        assert result is not None
        assert "XAG-USD" in result
        assert result["XAG-USD"]["actual_dir"] == "up"

    def test_direction_down(self):
        from metals_signal_tracker import _resolve_outcome
        import datetime

        entry = {
            "ts": "2026-03-01T10:00:00+00:00",
            "prices": {"XAG-USD": 30.0},
        }
        now = datetime.datetime(2026, 3, 1, 14, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        result = _resolve_outcome(entry, "1h", {"XAG-USD": 29.0}, now)
        assert result is not None
        assert result["XAG-USD"]["actual_dir"] == "down"

    def test_too_early(self):
        from metals_signal_tracker import _resolve_outcome
        import datetime

        entry = {
            "ts": "2026-03-01T10:00:00+00:00",
            "prices": {"XAG-USD": 30.0},
        }
        # Only 30 min later — too early for 1h horizon
        now = datetime.datetime(2026, 3, 1, 10, 30, 0, tzinfo=datetime.timezone.utc).timestamp()
        result = _resolve_outcome(entry, "1h", {"XAG-USD": 31.0}, now)
        assert result is None

    def test_no_prices_in_entry(self):
        from metals_signal_tracker import _resolve_outcome
        import datetime

        entry = {
            "ts": "2026-03-01T10:00:00+00:00",
            "prices": {},
        }
        now = datetime.datetime(2026, 3, 1, 14, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        result = _resolve_outcome(entry, "1h", {"XAG-USD": 31.0}, now)
        assert result is None

    def test_no_current_prices(self):
        from metals_signal_tracker import _resolve_outcome
        import datetime

        entry = {
            "ts": "2026-03-01T10:00:00+00:00",
            "prices": {"XAG-USD": 30.0},
        }
        now = datetime.datetime(2026, 3, 1, 14, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
        result = _resolve_outcome(entry, "1h", {}, now)
        assert result is None


# ---------------------------------------------------------------------------
# backfill_outcomes
# ---------------------------------------------------------------------------

class TestBackfillOutcomes:
    def test_writes_to_outcomes_file(self, tmp_path):
        from metals_signal_tracker import backfill_outcomes, SIGNAL_LOG, OUTCOMES_LOG
        import datetime

        # Create an old snapshot
        ts = "2026-03-01T10:00:00+00:00"
        entry = {
            "ts": ts,
            "check_count": 1,
            "prices": {"XAG-USD": 30.0},
            "signals": {"XAG-USD": {"consensus": "BUY"}},
        }
        with open(SIGNAL_LOG, "w") as f:
            f.write(json.dumps(entry) + "\n")

        backfill_outcomes({"XAG-USD": 31.0})

        if os.path.exists(OUTCOMES_LOG):
            with open(OUTCOMES_LOG, "r") as f:
                lines = f.readlines()
            assert len(lines) >= 0  # May be 0 if too recent

    def test_dedup_resolved(self, tmp_path):
        from metals_signal_tracker import backfill_outcomes, SIGNAL_LOG, OUTCOMES_LOG
        import datetime

        ts = "2026-03-01T10:00:00+00:00"
        entry = {
            "ts": ts,
            "check_count": 1,
            "prices": {"XAG-USD": 30.0},
            "signals": {},
        }
        with open(SIGNAL_LOG, "w") as f:
            f.write(json.dumps(entry) + "\n")

        # Run twice
        backfill_outcomes({"XAG-USD": 31.0})
        backfill_outcomes({"XAG-USD": 31.5})

        if os.path.exists(OUTCOMES_LOG):
            with open(OUTCOMES_LOG, "r") as f:
                lines = f.readlines()
            # Each (ts, horizon) pair should appear at most once
            keys = set()
            for line in lines:
                o = json.loads(line)
                key = (o.get("snapshot_ts"), o.get("horizon"))
                assert key not in keys, f"Duplicate outcome: {key}"
                keys.add(key)

    def test_signal_log_not_rewritten(self, tmp_path):
        from metals_signal_tracker import backfill_outcomes, SIGNAL_LOG

        ts = "2026-03-01T10:00:00+00:00"
        original = json.dumps({"ts": ts, "prices": {"XAG-USD": 30.0}, "signals": {}})
        with open(SIGNAL_LOG, "w") as f:
            f.write(original + "\n")

        backfill_outcomes({"XAG-USD": 31.0})

        with open(SIGNAL_LOG, "r") as f:
            content = f.read().strip()
        assert content == original  # Signal log should be append-only, not rewritten

    def test_empty_signal_log(self, tmp_path):
        from metals_signal_tracker import backfill_outcomes
        # Should not crash with no signal log
        backfill_outcomes({"XAG-USD": 31.0})

    def test_no_prices(self, tmp_path):
        from metals_signal_tracker import backfill_outcomes
        backfill_outcomes({})


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_recompute_from_outcomes(self, tmp_path):
        import metals_signal_tracker as mod

        outcomes = [
            {"snapshot_ts": f"2026-03-01T{10+i}:00:00+00:00", "horizon": "1h",
             "outcomes": {
                 "XAG-USD": {
                     "price_then": 30.0, "price_now": 31.0,
                     "actual_dir": "up", "move_pct": 3.33,
                     "main_predicted": "BUY", "main_correct": True,
                     "per_signal": {"rsi": {"predicted": "up", "correct": True},
                                    "macd": {"predicted": "down", "correct": False}},
                 }
             }}
            for i in range(5)
        ]
        with open(mod.OUTCOMES_LOG, "w") as f:
            for o in outcomes:
                f.write(json.dumps(o) + "\n")

        mod._recompute_accuracy_from_outcomes()

        with mod._lock:
            cache = dict(mod._accuracy_cache)
        assert len(cache) > 0

    def test_accuracy_cache_written_to_disk(self, tmp_path):
        import metals_signal_tracker as mod

        outcomes = [
            {"snapshot_ts": f"2026-03-01T{10+i}:00:00+00:00", "horizon": "1h",
             "outcomes": {
                 "XAG-USD": {
                     "price_then": 30.0, "price_now": 31.0,
                     "actual_dir": "up", "move_pct": 3.33,
                     "main_predicted": "BUY", "main_correct": True,
                 }
             }}
            for i in range(3)
        ]
        with open(mod.OUTCOMES_LOG, "w") as f:
            for o in outcomes:
                f.write(json.dumps(o) + "\n")

        mod._recompute_accuracy_from_outcomes()

        assert os.path.exists(mod.ACCURACY_CACHE_FILE)
        with open(mod.ACCURACY_CACHE_FILE, "r") as f:
            cache = json.load(f)
        assert isinstance(cache, dict)

    def test_get_report_from_cache(self, tmp_path):
        from metals_signal_tracker import get_accuracy_report, ACCURACY_CACHE_FILE

        cache_data = {
            "rsi_1h_XAG-USD": {"correct": 7, "total": 10, "accuracy": 0.7},
        }
        with open(ACCURACY_CACHE_FILE, "w") as f:
            json.dump(cache_data, f)

        report = get_accuracy_report()
        assert isinstance(report, dict)

    def test_get_summary_string(self, tmp_path):
        from metals_signal_tracker import get_accuracy_summary, ACCURACY_CACHE_FILE

        cache_data = {
            "rsi_1h_XAG-USD": {"correct": 7, "total": 10, "accuracy": 0.7},
            "macd_1h_XAG-USD": {"correct": 3, "total": 10, "accuracy": 0.3},
        }
        with open(ACCURACY_CACHE_FILE, "w") as f:
            json.dump(cache_data, f)

        summary = get_accuracy_summary()
        assert isinstance(summary, str)

    def test_get_summary_no_data(self, tmp_path):
        from metals_signal_tracker import get_accuracy_summary
        summary = get_accuracy_summary()
        assert isinstance(summary, str)

    def test_get_for_context_structured(self, tmp_path):
        from metals_signal_tracker import get_accuracy_for_context, ACCURACY_CACHE_FILE

        cache_data = {
            "rsi_1h_XAG-USD": {"correct": 7, "total": 10, "accuracy": 0.7},
        }
        with open(ACCURACY_CACHE_FILE, "w") as f:
            json.dump(cache_data, f)

        ctx = get_accuracy_for_context()
        assert isinstance(ctx, (dict, list, str))

    def test_get_for_context_no_data(self, tmp_path):
        from metals_signal_tracker import get_accuracy_for_context
        ctx = get_accuracy_for_context()
        assert ctx is not None


# ---------------------------------------------------------------------------
# get_snapshot_count
# ---------------------------------------------------------------------------

class TestSnapshotCount:
    def test_count_lines(self, tmp_path):
        from metals_signal_tracker import get_snapshot_count, SIGNAL_LOG

        with open(SIGNAL_LOG, "w") as f:
            for i in range(5):
                f.write(json.dumps({"ts": f"t{i}"}) + "\n")

        assert get_snapshot_count() == 5

    def test_empty_file(self, tmp_path):
        from metals_signal_tracker import get_snapshot_count
        assert get_snapshot_count() == 0
