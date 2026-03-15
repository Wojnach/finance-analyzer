"""Tests for portfolio.fin_evolve — self-improvement engine for /fin commands."""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio import fin_evolve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_files(tmp_path, monkeypatch):
    """Redirect all file paths to tmp_path for test isolation."""
    monkeypatch.setattr(fin_evolve, "_LOG_FILE", tmp_path / "fin_command_log.jsonl")
    monkeypatch.setattr(fin_evolve, "_PRICE_FILE", tmp_path / "price_snapshots_hourly.jsonl")
    monkeypatch.setattr(fin_evolve, "_LESSONS_FILE", tmp_path / "fin_command_lessons.json")
    monkeypatch.setattr(fin_evolve, "_EVOLVE_STATE_FILE", tmp_path / "fin_evolve_state.json")
    monkeypatch.setattr(fin_evolve, "_DATA_DIR", tmp_path)


def _write_jsonl(path, entries):
    """Write a list of dicts as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    """Read a JSONL file into list of dicts."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _read_json(path):
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _make_verdict(
    ts_offset_hours=-48,
    command="fin-silver",
    ticker="XAG-USD",
    price_usd=85.0,
    verdict_1_3d="bullish",
    verdict_1_3d_conf=0.7,
    verdict_1_4w="bullish",
    verdict_1_4w_conf=0.6,
    regime="trending-up",
    rsi=55.0,
    gs_ratio=60.0,
    dxy=97.0,
    chronos_accuracy=75.0,
    prob_1d=65.0,
    **extra,
):
    """Create a verdict entry with sensible defaults."""
    ts = datetime.now(timezone.utc) + timedelta(hours=ts_offset_hours)
    entry = {
        "ts": ts.isoformat(),
        "command": command,
        "ticker": ticker,
        "price_usd": price_usd,
        "verdict_1_3d": verdict_1_3d,
        "verdict_1_3d_conf": verdict_1_3d_conf,
        "verdict_1_4w": verdict_1_4w,
        "verdict_1_4w_conf": verdict_1_4w_conf,
        "regime": regime,
        "rsi": rsi,
        "gs_ratio": gs_ratio,
        "dxy": dxy,
        "chronos_accuracy": chronos_accuracy,
        "prob_1d": prob_1d,
        "signal_consensus": "BUY",
        "vote_breakdown": "5B/1S/4H",
        "weighted_confidence": 0.6,
    }
    entry.update(extra)
    return entry


def _make_price_snap(ts, prices):
    """Create a price snapshot entry."""
    return {"ts": ts.isoformat(), "prices": prices}


# ---------------------------------------------------------------------------
# Tests: _parse_iso
# ---------------------------------------------------------------------------

class TestParseIso:
    def test_valid_iso(self):
        result = fin_evolve._parse_iso("2026-03-10T12:00:00+00:00")
        assert result is not None
        assert result.year == 2026

    def test_none(self):
        assert fin_evolve._parse_iso(None) is None

    def test_empty_string(self):
        assert fin_evolve._parse_iso("") is None

    def test_invalid(self):
        assert fin_evolve._parse_iso("not-a-date") is None


# ---------------------------------------------------------------------------
# Tests: _check_verdict
# ---------------------------------------------------------------------------

class TestCheckVerdict:
    def test_bullish_correct(self):
        assert fin_evolve._check_verdict("bullish", 2.5) is True

    def test_bullish_wrong(self):
        assert fin_evolve._check_verdict("bullish", -1.0) is False

    def test_bearish_correct(self):
        assert fin_evolve._check_verdict("bearish", -3.0) is True

    def test_bearish_wrong(self):
        assert fin_evolve._check_verdict("bearish", 0.5) is False

    def test_neutral_returns_none(self):
        assert fin_evolve._check_verdict("neutral", 5.0) is None

    def test_none_verdict(self):
        assert fin_evolve._check_verdict(None, 2.0) is None

    def test_unknown_verdict(self):
        assert fin_evolve._check_verdict("sideways", 1.0) is None

    def test_zero_outcome_bearish(self):
        # Price unchanged: bearish verdict says "price goes down", 0 is not < 0
        assert fin_evolve._check_verdict("bearish", 0.0) is False

    def test_zero_outcome_bullish(self):
        # Price unchanged: bullish verdict says "price goes up", 0 is not > 0
        assert fin_evolve._check_verdict("bullish", 0.0) is False


# ---------------------------------------------------------------------------
# Tests: _find_price_at
# ---------------------------------------------------------------------------

class TestFindPriceAt:
    def test_exact_match(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
        history = [
            {"_parsed_ts": ts, "prices": {"XAG-USD": 85.5}},
        ]
        result = fin_evolve._find_price_at(history, "XAG-USD", ts)
        assert result == 85.5

    def test_close_match(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
        snap_ts = datetime(2026, 3, 10, 13, 0, tzinfo=timezone.utc)
        history = [
            {"_parsed_ts": snap_ts, "prices": {"XAG-USD": 86.0}},
        ]
        result = fin_evolve._find_price_at(history, "XAG-USD", ts)
        assert result == 86.0

    def test_too_far(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
        snap_ts = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)  # 24h away
        history = [
            {"_parsed_ts": snap_ts, "prices": {"XAG-USD": 86.0}},
        ]
        result = fin_evolve._find_price_at(history, "XAG-USD", ts)
        assert result is None

    def test_ticker_not_in_snapshot(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
        history = [
            {"_parsed_ts": ts, "prices": {"BTC-USD": 70000.0}},
        ]
        result = fin_evolve._find_price_at(history, "XAG-USD", ts)
        assert result is None

    def test_empty_history(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
        assert fin_evolve._find_price_at([], "XAG-USD", ts) is None

    def test_none_history(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
        assert fin_evolve._find_price_at(None, "XAG-USD", ts) is None

    def test_closest_wins(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=timezone.utc)
        history = [
            {"_parsed_ts": datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc),
             "prices": {"XAG-USD": 84.0}},
            {"_parsed_ts": datetime(2026, 3, 10, 11, 30, tzinfo=timezone.utc),
             "prices": {"XAG-USD": 85.0}},
            {"_parsed_ts": datetime(2026, 3, 10, 14, 0, tzinfo=timezone.utc),
             "prices": {"XAG-USD": 86.0}},
        ]
        # 11:30 is closest to 12:00
        result = fin_evolve._find_price_at(history, "XAG-USD", ts)
        assert result == 85.0


# ---------------------------------------------------------------------------
# Tests: backfill_outcomes
# ---------------------------------------------------------------------------

class TestBackfillOutcomes:
    def test_backfill_1d(self, tmp_path):
        # Verdict from 30 hours ago
        now = datetime.now(timezone.utc)
        verdict_ts = now - timedelta(hours=30)
        verdict = _make_verdict(ts_offset_hours=-30, price_usd=85.0)

        # Price 24h after verdict: went up
        target_ts = verdict_ts + timedelta(days=1)
        snap = _make_price_snap(target_ts, {"XAG-USD": 87.0})

        _write_jsonl(fin_evolve._LOG_FILE, [verdict])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap])

        n = fin_evolve.backfill_outcomes()
        assert n >= 1

        entries = _read_jsonl(fin_evolve._LOG_FILE)
        assert "outcome_1d_pct" in entries[0]
        assert entries[0]["outcome_1d_pct"] == pytest.approx(2.353, abs=0.01)
        assert entries[0]["verdict_correct_1d"] is True  # bullish + up

    def test_backfill_3d(self, tmp_path):
        now = datetime.now(timezone.utc)
        verdict_ts = now - timedelta(hours=80)
        verdict = _make_verdict(ts_offset_hours=-80, price_usd=85.0)

        # Price snapshots at 1d and 3d
        snap_1d = _make_price_snap(
            verdict_ts + timedelta(days=1), {"XAG-USD": 86.0}
        )
        snap_3d = _make_price_snap(
            verdict_ts + timedelta(days=3), {"XAG-USD": 83.0}
        )

        _write_jsonl(fin_evolve._LOG_FILE, [verdict])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap_1d, snap_3d])

        n = fin_evolve.backfill_outcomes()
        assert n >= 2

        entries = _read_jsonl(fin_evolve._LOG_FILE)
        assert "outcome_3d_pct" in entries[0]
        # Price went from 85 to 83: -2.353%
        assert entries[0]["outcome_3d_pct"] == pytest.approx(-2.353, abs=0.01)
        assert entries[0]["verdict_correct_3d"] is False  # bullish but down

    def test_backfill_7d(self, tmp_path):
        now = datetime.now(timezone.utc)
        verdict_ts = now - timedelta(hours=180)
        verdict = _make_verdict(ts_offset_hours=-180, price_usd=85.0)

        snap_1d = _make_price_snap(
            verdict_ts + timedelta(days=1), {"XAG-USD": 86.0}
        )
        snap_3d = _make_price_snap(
            verdict_ts + timedelta(days=3), {"XAG-USD": 84.0}
        )
        snap_7d = _make_price_snap(
            verdict_ts + timedelta(days=7), {"XAG-USD": 90.0}
        )

        _write_jsonl(fin_evolve._LOG_FILE, [verdict])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap_1d, snap_3d, snap_7d])

        n = fin_evolve.backfill_outcomes()
        assert n >= 3

        entries = _read_jsonl(fin_evolve._LOG_FILE)
        assert "outcome_7d_pct" in entries[0]
        assert entries[0]["outcome_7d_pct"] == pytest.approx(5.882, abs=0.01)
        assert entries[0]["verdict_correct_7d"] is True  # bullish + up

    def test_no_double_backfill(self, tmp_path):
        """Entries already backfilled should not be updated again."""
        verdict = _make_verdict(
            ts_offset_hours=-30,
            price_usd=85.0,
            outcome_1d_pct=2.0,
            verdict_correct_1d=True,
        )

        now = datetime.now(timezone.utc)
        verdict_ts = now - timedelta(hours=30)
        snap = _make_price_snap(
            verdict_ts + timedelta(days=1), {"XAG-USD": 999.0}
        )

        _write_jsonl(fin_evolve._LOG_FILE, [verdict])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap])

        n = fin_evolve.backfill_outcomes()
        assert n == 0

        entries = _read_jsonl(fin_evolve._LOG_FILE)
        assert entries[0]["outcome_1d_pct"] == 2.0  # unchanged

    def test_no_entries(self, tmp_path):
        n = fin_evolve.backfill_outcomes()
        assert n == 0

    def test_no_price_data(self, tmp_path):
        verdict = _make_verdict(ts_offset_hours=-30, price_usd=85.0)
        _write_jsonl(fin_evolve._LOG_FILE, [verdict])
        # No price file
        n = fin_evolve.backfill_outcomes()
        assert n == 0

    def test_bearish_verdict_backfill(self, tmp_path):
        now = datetime.now(timezone.utc)
        verdict_ts = now - timedelta(hours=30)
        verdict = _make_verdict(
            ts_offset_hours=-30,
            price_usd=85.0,
            verdict_1_3d="bearish",
        )

        snap = _make_price_snap(
            verdict_ts + timedelta(days=1), {"XAG-USD": 83.0}
        )

        _write_jsonl(fin_evolve._LOG_FILE, [verdict])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap])

        fin_evolve.backfill_outcomes()

        entries = _read_jsonl(fin_evolve._LOG_FILE)
        assert entries[0]["verdict_correct_1d"] is True  # bearish + price down


# ---------------------------------------------------------------------------
# Tests: _analyze_by_field
# ---------------------------------------------------------------------------

class TestAnalyzeByField:
    def test_basic_regime_analysis(self):
        scored = [
            {"regime": "trending-up", "verdict_correct_3d": True},
            {"regime": "trending-up", "verdict_correct_3d": True},
            {"regime": "trending-up", "verdict_correct_3d": False},
            {"regime": "range-bound", "verdict_correct_3d": False},
            {"regime": "range-bound", "verdict_correct_3d": False},
            {"regime": "range-bound", "verdict_correct_3d": True},
        ]
        result = fin_evolve._analyze_by_field(scored, "regime")
        assert "trending-up" in result
        assert result["trending-up"]["accuracy"] == pytest.approx(0.667, abs=0.01)
        assert result["trending-up"]["n"] == 3
        assert "range-bound" in result
        assert result["range-bound"]["accuracy"] == pytest.approx(0.333, abs=0.01)

    def test_skips_small_groups(self):
        scored = [
            {"regime": "high-vol", "verdict_correct_3d": True},
            {"regime": "high-vol", "verdict_correct_3d": False},
            # Only 2 samples, below _MIN_SAMPLES_LESSON=3
        ]
        result = fin_evolve._analyze_by_field(scored, "regime")
        assert "high-vol" not in result

    def test_neutral_excluded(self):
        scored = [
            {"regime": "trending-up", "verdict_correct_3d": None},
            {"regime": "trending-up", "verdict_correct_3d": None},
            {"regime": "trending-up", "verdict_correct_3d": None},
        ]
        result = fin_evolve._analyze_by_field(scored, "regime")
        assert "trending-up" not in result  # 0 evaluable


# ---------------------------------------------------------------------------
# Tests: _analyze_by_confidence
# ---------------------------------------------------------------------------

class TestAnalyzeByConfidence:
    def test_confidence_buckets(self):
        scored = [
            {"verdict_1_3d_conf": 0.8, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.9, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.85, "verdict_correct_3d": False},
            {"verdict_1_3d_conf": 0.5, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.6, "verdict_correct_3d": False},
            {"verdict_1_3d_conf": 0.55, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.2, "verdict_correct_3d": False},
            {"verdict_1_3d_conf": 0.1, "verdict_correct_3d": False},
            {"verdict_1_3d_conf": 0.3, "verdict_correct_3d": True},
        ]
        result = fin_evolve._analyze_by_confidence(scored)
        assert "high (>0.7)" in result
        assert result["high (>0.7)"]["n"] == 3
        assert "medium (0.4-0.7)" in result
        assert result["medium (0.4-0.7)"]["n"] == 3
        assert "low (<0.4)" in result
        assert result["low (<0.4)"]["n"] == 3


# ---------------------------------------------------------------------------
# Tests: _find_anti_patterns
# ---------------------------------------------------------------------------

class TestFindAntiPatterns:
    def test_bearish_low_rsi_anti_pattern(self):
        scored = [
            {"verdict_1_3d": "bearish", "rsi": 25, "verdict_correct_3d": False},
            {"verdict_1_3d": "bearish", "rsi": 30, "verdict_correct_3d": False},
            {"verdict_1_3d": "bearish", "rsi": 28, "verdict_correct_3d": False},
        ]
        patterns = fin_evolve._find_anti_patterns(scored)
        assert any("RSI<35" in p for p in patterns)

    def test_no_anti_pattern_when_accurate(self):
        scored = [
            {"verdict_1_3d": "bearish", "rsi": 25, "verdict_correct_3d": True},
            {"verdict_1_3d": "bearish", "rsi": 30, "verdict_correct_3d": True},
            {"verdict_1_3d": "bearish", "rsi": 28, "verdict_correct_3d": True},
        ]
        patterns = fin_evolve._find_anti_patterns(scored)
        assert not any("RSI<35" in p for p in patterns)

    def test_dxy_anti_pattern(self):
        scored = [
            {"verdict_1_3d": "bullish", "dxy": 103, "verdict_correct_3d": False},
            {"verdict_1_3d": "bullish", "dxy": 104, "verdict_correct_3d": False},
            {"verdict_1_3d": "bullish", "dxy": 105, "verdict_correct_3d": False},
        ]
        patterns = fin_evolve._find_anti_patterns(scored)
        assert any("DXY>102" in p for p in patterns)

    def test_high_confidence_anti_pattern(self):
        scored = [
            {"verdict_1_3d_conf": 0.9, "verdict_correct_3d": False},
            {"verdict_1_3d_conf": 0.85, "verdict_correct_3d": False},
            {"verdict_1_3d_conf": 0.95, "verdict_correct_3d": False},
        ]
        patterns = fin_evolve._find_anti_patterns(scored)
        assert any("overconfidence" in p.lower() for p in patterns)


# ---------------------------------------------------------------------------
# Tests: _find_confirmed_patterns
# ---------------------------------------------------------------------------

class TestFindConfirmedPatterns:
    def test_gs_ratio_pattern(self):
        scored = [
            {
                "verdict_1_3d": "bullish",
                "ticker": "XAG-USD",
                "gs_ratio": 70,
                "verdict_correct_3d": True,
            },
            {
                "verdict_1_3d": "bullish",
                "ticker": "XAG-USD",
                "gs_ratio": 68,
                "verdict_correct_3d": True,
            },
            {
                "verdict_1_3d": "bullish",
                "ticker": "XAG-USD",
                "gs_ratio": 72,
                "verdict_correct_3d": True,
            },
        ]
        patterns = fin_evolve._find_confirmed_patterns(scored)
        assert any("G/S ratio" in p for p in patterns)

    def test_no_pattern_with_wrong_ticker(self):
        scored = [
            {
                "verdict_1_3d": "bullish",
                "ticker": "XAU-USD",  # Gold, not silver
                "gs_ratio": 70,
                "verdict_correct_3d": True,
            },
            {
                "verdict_1_3d": "bullish",
                "ticker": "XAU-USD",
                "gs_ratio": 68,
                "verdict_correct_3d": True,
            },
            {
                "verdict_1_3d": "bullish",
                "ticker": "XAU-USD",
                "gs_ratio": 72,
                "verdict_correct_3d": True,
            },
        ]
        patterns = fin_evolve._find_confirmed_patterns(scored)
        # G/S pattern is XAG-only
        assert not any("G/S ratio" in p for p in patterns)


# ---------------------------------------------------------------------------
# Tests: _compute_calibration_advice
# ---------------------------------------------------------------------------

class TestComputeCalibrationAdvice:
    def test_overconfident(self):
        scored = [
            {"verdict_1_3d_conf": 0.9, "verdict_correct_3d": False},
            {"verdict_1_3d_conf": 0.85, "verdict_correct_3d": False},
            {"verdict_1_3d_conf": 0.8, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.9, "verdict_correct_3d": False},
            {"verdict_1_3d_conf": 0.85, "verdict_correct_3d": False},
        ]
        advice = fin_evolve._compute_calibration_advice(scored)
        assert "OVERCONFIDENT" in advice

    def test_underconfident(self):
        scored = [
            {"verdict_1_3d_conf": 0.3, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.25, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.2, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.35, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.3, "verdict_correct_3d": True},
        ]
        advice = fin_evolve._compute_calibration_advice(scored)
        assert "UNDERCONFIDENT" in advice

    def test_well_calibrated(self):
        scored = [
            {"verdict_1_3d_conf": 0.6, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.5, "verdict_correct_3d": False},
            {"verdict_1_3d_conf": 0.7, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.6, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.5, "verdict_correct_3d": False},
        ]
        advice = fin_evolve._compute_calibration_advice(scored)
        assert "WELL CALIBRATED" in advice

    def test_not_enough_data(self):
        scored = [
            {"verdict_1_3d_conf": 0.6, "verdict_correct_3d": True},
        ]
        advice = fin_evolve._compute_calibration_advice(scored)
        assert "Not enough data" in advice


# ---------------------------------------------------------------------------
# Tests: evolve (full pipeline)
# ---------------------------------------------------------------------------

class TestEvolve:
    def test_evolve_with_enough_data(self, tmp_path):
        """Full evolve pipeline with 6 scored entries."""
        scored = []
        for i in range(6):
            v = _make_verdict(
                ts_offset_hours=-(100 + i * 24),
                price_usd=85.0 + i,
                verdict_1_3d="bullish",
                verdict_1_3d_conf=0.7,
                regime="trending-up",
            )
            v["outcome_3d_pct"] = 2.0 if i % 2 == 0 else -1.0
            v["verdict_correct_3d"] = i % 2 == 0  # alternating correct/wrong
            scored.append(v)

        _write_jsonl(fin_evolve._LOG_FILE, scored)

        result = fin_evolve.evolve()
        assert result is not None
        assert result["total_verdicts"] == 6
        assert "by_command" in result
        assert "by_regime" in result
        assert "by_confidence" in result
        assert "calibration_advice" in result

        # Check the lessons file was written
        lessons = _read_json(fin_evolve._LESSONS_FILE)
        assert lessons["total_verdicts"] == 6

    def test_evolve_not_enough_data(self, tmp_path):
        """Should return None if fewer than 5 scored verdicts."""
        scored = []
        for i in range(3):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24))
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)

        _write_jsonl(fin_evolve._LOG_FILE, scored)

        result = fin_evolve.evolve()
        assert result is None

    def test_evolve_empty_log(self, tmp_path):
        result = fin_evolve.evolve()
        assert result is None

    def test_evolve_per_ticker_stats(self, tmp_path):
        """Verify per-ticker accuracy is computed."""
        scored = []
        for i in range(3):
            v = _make_verdict(
                ts_offset_hours=-(100 + i * 24),
                ticker="XAG-USD",
                command="fin-silver",
            )
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)
        for i in range(3):
            v = _make_verdict(
                ts_offset_hours=-(100 + i * 24),
                ticker="XAU-USD",
                command="fin-gold",
            )
            v["outcome_3d_pct"] = -1.0
            v["verdict_correct_3d"] = False
            scored.append(v)

        _write_jsonl(fin_evolve._LOG_FILE, scored)
        result = fin_evolve.evolve()
        assert result is not None
        assert "by_ticker" in result
        assert result["by_ticker"]["XAG-USD"]["accuracy_3d"] == 1.0
        assert result["by_ticker"]["XAU-USD"]["accuracy_3d"] == 0.0

    def test_evolve_by_command(self, tmp_path):
        """Verify per-command accuracy."""
        scored = []
        for i in range(3):
            v = _make_verdict(
                ts_offset_hours=-(100 + i * 24),
                command="fin-silver",
            )
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)
        for i in range(3):
            v = _make_verdict(
                ts_offset_hours=-(100 + i * 24),
                command="fin-gold",
                ticker="XAU-USD",
            )
            v["outcome_3d_pct"] = -1.0
            v["verdict_correct_3d"] = False
            scored.append(v)

        _write_jsonl(fin_evolve._LOG_FILE, scored)
        result = fin_evolve.evolve()
        assert result["by_command"]["fin-silver"]["accuracy_3d"] == 1.0
        assert result["by_command"]["fin-gold"]["accuracy_3d"] == 0.0

    def test_evolve_execution_stats(self, tmp_path):
        """Verify execution time stats are computed."""
        scored = []
        for i in range(5):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24))
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            v["execution_time_sec"] = 10.0 + i * 5
            scored.append(v)

        _write_jsonl(fin_evolve._LOG_FILE, scored)
        result = fin_evolve.evolve()
        assert "execution_stats" in result
        assert result["execution_stats"]["min_sec"] == 10.0
        assert result["execution_stats"]["max_sec"] == 30.0
        assert result["execution_stats"]["n"] == 5


# ---------------------------------------------------------------------------
# Tests: maybe_evolve (throttling)
# ---------------------------------------------------------------------------

class TestMaybeEvolve:
    def test_runs_on_first_call(self, tmp_path):
        """Should run if no state file exists."""
        # Create enough data
        scored = []
        now = datetime.now(timezone.utc)
        for i in range(6):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24))
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)

        _write_jsonl(fin_evolve._LOG_FILE, scored)

        result = fin_evolve.maybe_evolve()
        assert result is not None

        # State file should exist
        state = _read_json(fin_evolve._EVOLVE_STATE_FILE)
        assert state["status"] == "ok"

    def test_throttled_on_second_call(self, tmp_path):
        """Should not run if called again within the interval."""
        scored = []
        for i in range(6):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24))
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, scored)

        # First call
        fin_evolve.maybe_evolve()

        # Second call should be throttled
        result = fin_evolve.maybe_evolve()
        assert result is None

    def test_runs_after_interval(self, tmp_path, monkeypatch):
        """Should run if enough time has passed."""
        # Write a state with old timestamp
        state = {
            "last_run_epoch": time.time() - fin_evolve._EVOLVE_INTERVAL_SEC - 100,
            "status": "ok",
        }
        fin_evolve._atomic_write_json(fin_evolve._EVOLVE_STATE_FILE, state)

        # Create data
        scored = []
        for i in range(6):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24))
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, scored)

        result = fin_evolve.maybe_evolve()
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: _compute_signal_trust
# ---------------------------------------------------------------------------

class TestComputeSignalTrust:
    def test_basic_trust(self):
        scored = [
            {"signal_consensus": "BUY", "verdict_correct_3d": True},
            {"signal_consensus": "BUY", "verdict_correct_3d": True},
            {"signal_consensus": "BUY", "verdict_correct_3d": False},
            {"signal_consensus": "SELL", "verdict_correct_3d": True},
            {"signal_consensus": "SELL", "verdict_correct_3d": False},
            {"signal_consensus": "SELL", "verdict_correct_3d": False},
        ]
        result = fin_evolve._compute_signal_trust(scored)
        assert "BUY" in result
        assert result["BUY"]["accuracy"] == pytest.approx(0.667, abs=0.01)
        assert "SELL" in result
        assert result["SELL"]["accuracy"] == pytest.approx(0.333, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: _bucket_midpoint
# ---------------------------------------------------------------------------

class TestBucketMidpoint:
    def test_high(self):
        assert fin_evolve._bucket_midpoint("high (>0.7)") == 0.85

    def test_medium(self):
        assert fin_evolve._bucket_midpoint("medium (0.4-0.7)") == 0.55

    def test_low(self):
        assert fin_evolve._bucket_midpoint("low (<0.4)") == 0.25


# ---------------------------------------------------------------------------
# Tests: end-to-end (backfill + evolve)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        """Test complete backfill + evolve flow."""
        now = datetime.now(timezone.utc)

        # Create 6 verdicts from 4 days ago
        verdicts = []
        for i in range(6):
            v = _make_verdict(
                ts_offset_hours=-(96 + i * 2),
                price_usd=85.0 + i * 0.5,
                verdict_1_3d="bullish" if i < 4 else "bearish",
                verdict_1_3d_conf=0.6 + i * 0.05,
                regime="trending-up" if i < 3 else "range-bound",
            )
            verdicts.append(v)

        _write_jsonl(fin_evolve._LOG_FILE, verdicts)

        # Create price snapshots at 1d and 3d after each verdict
        price_snaps = []
        for v in verdicts:
            v_ts = fin_evolve._parse_iso(v["ts"])
            price_at_verdict = v["price_usd"]
            # Price goes up: bullish verdicts correct, bearish wrong
            price_snaps.append(
                _make_price_snap(
                    v_ts + timedelta(days=1),
                    {"XAG-USD": price_at_verdict + 1.0},
                )
            )
            price_snaps.append(
                _make_price_snap(
                    v_ts + timedelta(days=3),
                    {"XAG-USD": price_at_verdict + 2.0},
                )
            )

        _write_jsonl(fin_evolve._PRICE_FILE, price_snaps)

        # Run backfill
        n = fin_evolve.backfill_outcomes()
        assert n > 0

        # Verify outcomes were written
        entries = _read_jsonl(fin_evolve._LOG_FILE)
        for e in entries:
            assert "outcome_1d_pct" in e
            assert "outcome_3d_pct" in e

        # Run evolve
        result = fin_evolve.evolve()
        assert result is not None
        assert result["total_verdicts"] >= 5

        # Bullish verdicts should be correct (price went up)
        # Bearish verdicts should be wrong (price went up)
        assert "calibration_advice" in result
        assert "by_command" in result
        assert "fin-silver" in result["by_command"]
