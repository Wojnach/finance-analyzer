"""Tests for portfolio.fin_evolve — system-wide self-improvement engine."""

import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

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
    monkeypatch.setattr(fin_evolve, "_JOURNAL_FILE", tmp_path / "layer2_journal.jsonl")
    monkeypatch.setattr(fin_evolve, "_JOURNAL_OUTCOMES_FILE", tmp_path / "journal_outcomes.jsonl")
    monkeypatch.setattr(fin_evolve, "_PRICE_FILE", tmp_path / "price_snapshots_hourly.jsonl")
    monkeypatch.setattr(fin_evolve, "_LESSONS_FILE", tmp_path / "system_lessons.json")
    monkeypatch.setattr(fin_evolve, "_LEGACY_LESSONS_FILE", tmp_path / "fin_command_lessons.json")
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
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except FileNotFoundError:
        pass
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
    ts = datetime.now(UTC) + timedelta(hours=ts_offset_hours)
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


def _make_journal_entry(
    ts_offset_hours=-48,
    regime="trending-up",
    trigger="consensus",
    tickers=None,
    prices=None,
):
    """Create a Layer 2 journal entry."""
    ts = datetime.now(UTC) + timedelta(hours=ts_offset_hours)
    if tickers is None:
        tickers = {
            "BTC-USD": {"outlook": "bullish", "thesis": "test", "conviction": 0.6, "levels": []},
            "XAG-USD": {"outlook": "bullish", "thesis": "test", "conviction": 0.7, "levels": []},
        }
    if prices is None:
        prices = {"BTC-USD": 67000.0, "XAG-USD": 85.0}
    return {
        "ts": ts.isoformat(),
        "trigger": trigger,
        "regime": regime,
        "decisions": {"patient": {"action": "HOLD"}, "bold": {"action": "HOLD"}},
        "tickers": tickers,
        "watchlist": [],
        "prices": prices,
    }


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
        assert fin_evolve._check_verdict("bearish", 0.0) is False

    def test_zero_outcome_bullish(self):
        assert fin_evolve._check_verdict("bullish", 0.0) is False


# ---------------------------------------------------------------------------
# Tests: _find_price_at
# ---------------------------------------------------------------------------

class TestFindPriceAt:
    def test_exact_match(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
        history = [
            {"_parsed_ts": ts, "prices": {"XAG-USD": 85.5}},
        ]
        result = fin_evolve._find_price_at(history, "XAG-USD", ts)
        assert result == 85.5

    def test_close_match(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
        snap_ts = datetime(2026, 3, 10, 13, 0, tzinfo=UTC)
        history = [
            {"_parsed_ts": snap_ts, "prices": {"XAG-USD": 86.0}},
        ]
        result = fin_evolve._find_price_at(history, "XAG-USD", ts)
        assert result == 86.0

    def test_too_far(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
        snap_ts = datetime(2026, 3, 11, 12, 0, tzinfo=UTC)
        history = [
            {"_parsed_ts": snap_ts, "prices": {"XAG-USD": 86.0}},
        ]
        result = fin_evolve._find_price_at(history, "XAG-USD", ts)
        assert result is None

    def test_ticker_not_in_snapshot(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
        history = [
            {"_parsed_ts": ts, "prices": {"BTC-USD": 70000.0}},
        ]
        result = fin_evolve._find_price_at(history, "XAG-USD", ts)
        assert result is None

    def test_empty_history(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
        assert fin_evolve._find_price_at([], "XAG-USD", ts) is None

    def test_none_history(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
        assert fin_evolve._find_price_at(None, "XAG-USD", ts) is None

    def test_closest_wins(self):
        ts = datetime(2026, 3, 10, 12, 0, tzinfo=UTC)
        history = [
            {"_parsed_ts": datetime(2026, 3, 10, 10, 0, tzinfo=UTC),
             "prices": {"XAG-USD": 84.0}},
            {"_parsed_ts": datetime(2026, 3, 10, 11, 30, tzinfo=UTC),
             "prices": {"XAG-USD": 85.0}},
            {"_parsed_ts": datetime(2026, 3, 10, 14, 0, tzinfo=UTC),
             "prices": {"XAG-USD": 86.0}},
        ]
        result = fin_evolve._find_price_at(history, "XAG-USD", ts)
        assert result == 85.0


# ---------------------------------------------------------------------------
# Tests: backfill_outcomes (fin_command_log)
# ---------------------------------------------------------------------------

class TestBackfillOutcomes:
    def test_backfill_1d(self, tmp_path):
        now = datetime.now(UTC)
        verdict_ts = now - timedelta(hours=30)
        verdict = _make_verdict(ts_offset_hours=-30, price_usd=85.0)
        target_ts = verdict_ts + timedelta(days=1)
        snap = _make_price_snap(target_ts, {"XAG-USD": 87.0})
        _write_jsonl(fin_evolve._LOG_FILE, [verdict])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap])
        n = fin_evolve.backfill_outcomes()
        assert n >= 1
        entries = _read_jsonl(fin_evolve._LOG_FILE)
        assert "outcome_1d_pct" in entries[0]
        assert entries[0]["outcome_1d_pct"] == pytest.approx(2.353, abs=0.01)
        assert entries[0]["verdict_correct_1d"] is True

    def test_backfill_3d(self, tmp_path):
        now = datetime.now(UTC)
        verdict_ts = now - timedelta(hours=80)
        verdict = _make_verdict(ts_offset_hours=-80, price_usd=85.0)
        snap_1d = _make_price_snap(verdict_ts + timedelta(days=1), {"XAG-USD": 86.0})
        snap_3d = _make_price_snap(verdict_ts + timedelta(days=3), {"XAG-USD": 83.0})
        _write_jsonl(fin_evolve._LOG_FILE, [verdict])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap_1d, snap_3d])
        n = fin_evolve.backfill_outcomes()
        assert n >= 2
        entries = _read_jsonl(fin_evolve._LOG_FILE)
        assert entries[0]["outcome_3d_pct"] == pytest.approx(-2.353, abs=0.01)
        assert entries[0]["verdict_correct_3d"] is False

    def test_backfill_7d(self, tmp_path):
        now = datetime.now(UTC)
        verdict_ts = now - timedelta(hours=180)
        verdict = _make_verdict(ts_offset_hours=-180, price_usd=85.0)
        snap_1d = _make_price_snap(verdict_ts + timedelta(days=1), {"XAG-USD": 86.0})
        snap_3d = _make_price_snap(verdict_ts + timedelta(days=3), {"XAG-USD": 84.0})
        snap_7d = _make_price_snap(verdict_ts + timedelta(days=7), {"XAG-USD": 90.0})
        _write_jsonl(fin_evolve._LOG_FILE, [verdict])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap_1d, snap_3d, snap_7d])
        n = fin_evolve.backfill_outcomes()
        assert n >= 3
        entries = _read_jsonl(fin_evolve._LOG_FILE)
        assert entries[0]["outcome_7d_pct"] == pytest.approx(5.882, abs=0.01)
        assert entries[0]["verdict_correct_7d"] is True

    def test_no_double_backfill(self, tmp_path):
        verdict = _make_verdict(ts_offset_hours=-30, price_usd=85.0,
                                outcome_1d_pct=2.0, verdict_correct_1d=True)
        now = datetime.now(UTC)
        verdict_ts = now - timedelta(hours=30)
        snap = _make_price_snap(verdict_ts + timedelta(days=1), {"XAG-USD": 999.0})
        _write_jsonl(fin_evolve._LOG_FILE, [verdict])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap])
        n = fin_evolve.backfill_outcomes()
        assert n == 0
        entries = _read_jsonl(fin_evolve._LOG_FILE)
        assert entries[0]["outcome_1d_pct"] == 2.0

    def test_no_entries(self, tmp_path):
        assert fin_evolve.backfill_outcomes() == 0

    def test_no_price_data(self, tmp_path):
        verdict = _make_verdict(ts_offset_hours=-30, price_usd=85.0)
        _write_jsonl(fin_evolve._LOG_FILE, [verdict])
        assert fin_evolve.backfill_outcomes() == 0

    def test_bearish_verdict_backfill(self, tmp_path):
        now = datetime.now(UTC)
        verdict_ts = now - timedelta(hours=30)
        verdict = _make_verdict(ts_offset_hours=-30, price_usd=85.0, verdict_1_3d="bearish")
        snap = _make_price_snap(verdict_ts + timedelta(days=1), {"XAG-USD": 83.0})
        _write_jsonl(fin_evolve._LOG_FILE, [verdict])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap])
        fin_evolve.backfill_outcomes()
        entries = _read_jsonl(fin_evolve._LOG_FILE)
        assert entries[0]["verdict_correct_1d"] is True


# ---------------------------------------------------------------------------
# Tests: backfill_journal_outcomes (NEW)
# ---------------------------------------------------------------------------

class TestBackfillJournalOutcomes:
    def test_scores_non_neutral_outlooks(self, tmp_path):
        now = datetime.now(UTC)
        entry_ts = now - timedelta(hours=30)
        journal_entry = _make_journal_entry(ts_offset_hours=-30)
        snap = _make_price_snap(entry_ts + timedelta(days=1),
                                {"BTC-USD": 68000.0, "XAG-USD": 87.0})
        _write_jsonl(fin_evolve._JOURNAL_FILE, [journal_entry])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap])
        n = fin_evolve.backfill_journal_outcomes()
        assert n == 2
        outcomes = _read_jsonl(fin_evolve._JOURNAL_OUTCOMES_FILE)
        assert len(outcomes) == 2
        btc = [o for o in outcomes if o["ticker"] == "BTC-USD"][0]
        assert btc["source"] == "layer2"
        assert btc["outlook"] == "bullish"
        assert btc["conviction"] == 0.6
        assert btc["price_at_verdict"] == 67000.0
        assert btc["correct_1d"] is True

    def test_skips_neutral_outlooks(self, tmp_path):
        journal_entry = _make_journal_entry(
            ts_offset_hours=-30,
            tickers={"BTC-USD": {"outlook": "neutral", "conviction": 0.0}},
            prices={"BTC-USD": 67000.0},
        )
        _write_jsonl(fin_evolve._JOURNAL_FILE, [journal_entry])
        assert fin_evolve.backfill_journal_outcomes() == 0

    def test_skips_recent_entries(self, tmp_path):
        journal_entry = _make_journal_entry(ts_offset_hours=-10)
        _write_jsonl(fin_evolve._JOURNAL_FILE, [journal_entry])
        assert fin_evolve.backfill_journal_outcomes() == 0

    def test_deduplication(self, tmp_path):
        now = datetime.now(UTC)
        entry_ts = now - timedelta(hours=30)
        journal_entry = _make_journal_entry(ts_offset_hours=-30)
        snap = _make_price_snap(entry_ts + timedelta(days=1),
                                {"BTC-USD": 68000.0, "XAG-USD": 87.0})
        _write_jsonl(fin_evolve._JOURNAL_FILE, [journal_entry])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap])
        n1 = fin_evolve.backfill_journal_outcomes()
        assert n1 == 2
        n2 = fin_evolve.backfill_journal_outcomes()
        assert n2 == 0
        outcomes = _read_jsonl(fin_evolve._JOURNAL_OUTCOMES_FILE)
        assert len(outcomes) == 2

    def test_scores_3d_when_old_enough(self, tmp_path):
        now = datetime.now(UTC)
        entry_ts = now - timedelta(hours=80)
        journal_entry = _make_journal_entry(ts_offset_hours=-80)
        snap_1d = _make_price_snap(entry_ts + timedelta(days=1),
                                   {"BTC-USD": 68000.0, "XAG-USD": 87.0})
        snap_3d = _make_price_snap(entry_ts + timedelta(days=3),
                                   {"BTC-USD": 65000.0, "XAG-USD": 83.0})
        _write_jsonl(fin_evolve._JOURNAL_FILE, [journal_entry])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap_1d, snap_3d])
        fin_evolve.backfill_journal_outcomes()
        outcomes = _read_jsonl(fin_evolve._JOURNAL_OUTCOMES_FILE)
        btc = [o for o in outcomes if o["ticker"] == "BTC-USD"][0]
        assert "outcome_1d_pct" in btc
        assert "outcome_3d_pct" in btc
        assert btc["correct_1d"] is True
        assert btc["correct_3d"] is False

    def test_skips_tickers_without_prices(self, tmp_path):
        journal_entry = _make_journal_entry(
            ts_offset_hours=-30,
            tickers={
                "BTC-USD": {"outlook": "bullish", "conviction": 0.5},
                "UNKNOWN": {"outlook": "bullish", "conviction": 0.3},
            },
            prices={"BTC-USD": 67000.0},
        )
        snap = _make_price_snap(datetime.now(UTC) - timedelta(hours=6),
                                {"BTC-USD": 68000.0})
        _write_jsonl(fin_evolve._JOURNAL_FILE, [journal_entry])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap])
        assert fin_evolve.backfill_journal_outcomes() == 1

    def test_no_journal_file(self, tmp_path):
        assert fin_evolve.backfill_journal_outcomes() == 0

    def test_no_price_file(self, tmp_path):
        _write_jsonl(fin_evolve._JOURNAL_FILE, [_make_journal_entry(ts_offset_hours=-30)])
        assert fin_evolve.backfill_journal_outcomes() == 0

    def test_handles_missing_conviction(self, tmp_path):
        now = datetime.now(UTC)
        entry_ts = now - timedelta(hours=30)
        journal_entry = _make_journal_entry(
            ts_offset_hours=-30,
            tickers={"BTC-USD": {"outlook": "bearish", "thesis": "test"}},
            prices={"BTC-USD": 67000.0},
        )
        snap = _make_price_snap(entry_ts + timedelta(days=1), {"BTC-USD": 65000.0})
        _write_jsonl(fin_evolve._JOURNAL_FILE, [journal_entry])
        _write_jsonl(fin_evolve._PRICE_FILE, [snap])
        assert fin_evolve.backfill_journal_outcomes() == 1
        outcomes = _read_jsonl(fin_evolve._JOURNAL_OUTCOMES_FILE)
        assert outcomes[0]["conviction"] == 0
        assert outcomes[0]["correct_1d"] is True


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
        assert "trending-up" not in result


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

    def test_layer2_bearish_bias(self):
        scored = [
            {"source": "layer2", "verdict_1_3d": "bearish", "verdict_correct_3d": False},
            {"source": "layer2", "verdict_1_3d": "bearish", "verdict_correct_3d": False},
            {"source": "layer2", "verdict_1_3d": "bearish", "verdict_correct_3d": False},
        ]
        patterns = fin_evolve._find_anti_patterns(scored)
        assert any("bearish bias" in p.lower() for p in patterns)


# ---------------------------------------------------------------------------
# Tests: _find_confirmed_patterns
# ---------------------------------------------------------------------------

class TestFindConfirmedPatterns:
    def test_gs_ratio_pattern(self):
        scored = [
            {"verdict_1_3d": "bullish", "ticker": "XAG-USD", "gs_ratio": 70,
             "verdict_correct_3d": True},
            {"verdict_1_3d": "bullish", "ticker": "XAG-USD", "gs_ratio": 68,
             "verdict_correct_3d": True},
            {"verdict_1_3d": "bullish", "ticker": "XAG-USD", "gs_ratio": 72,
             "verdict_correct_3d": True},
        ]
        patterns = fin_evolve._find_confirmed_patterns(scored)
        assert any("G/S ratio" in p for p in patterns)

    def test_no_pattern_with_wrong_ticker(self):
        scored = [
            {"verdict_1_3d": "bullish", "ticker": "XAU-USD", "gs_ratio": 70,
             "verdict_correct_3d": True},
            {"verdict_1_3d": "bullish", "ticker": "XAU-USD", "gs_ratio": 68,
             "verdict_correct_3d": True},
            {"verdict_1_3d": "bullish", "ticker": "XAU-USD", "gs_ratio": 72,
             "verdict_correct_3d": True},
        ]
        patterns = fin_evolve._find_confirmed_patterns(scored)
        assert not any("G/S ratio" in p for p in patterns)

    def test_layer2_bullish_pattern(self):
        scored = [
            {"source": "layer2", "verdict_1_3d": "bullish", "verdict_correct_3d": True},
            {"source": "layer2", "verdict_1_3d": "bullish", "verdict_correct_3d": True},
            {"source": "layer2", "verdict_1_3d": "bullish", "verdict_correct_3d": True},
        ]
        patterns = fin_evolve._find_confirmed_patterns(scored)
        assert any("bullish" in p.lower() and "layer 2" in p.lower() for p in patterns)


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
        assert "OVERCONFIDENT" in fin_evolve._compute_calibration_advice(scored)

    def test_underconfident(self):
        scored = [
            {"verdict_1_3d_conf": 0.3, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.25, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.2, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.35, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.3, "verdict_correct_3d": True},
        ]
        assert "UNDERCONFIDENT" in fin_evolve._compute_calibration_advice(scored)

    def test_well_calibrated(self):
        scored = [
            {"verdict_1_3d_conf": 0.6, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.5, "verdict_correct_3d": False},
            {"verdict_1_3d_conf": 0.7, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.6, "verdict_correct_3d": True},
            {"verdict_1_3d_conf": 0.5, "verdict_correct_3d": False},
        ]
        assert "WELL CALIBRATED" in fin_evolve._compute_calibration_advice(scored)

    def test_not_enough_data(self):
        scored = [{"verdict_1_3d_conf": 0.6, "verdict_correct_3d": True}]
        assert "Not enough data" in fin_evolve._compute_calibration_advice(scored)


# ---------------------------------------------------------------------------
# Tests: _compute_cross_asset (NEW)
# ---------------------------------------------------------------------------

class TestComputeCrossAsset:
    def test_gold_silver_correlation(self):
        result_scored = []
        for i in range(3):
            ts_i = f"2026-03-{10+i}T12:00:00+00:00"
            result_scored.append(
                {"journal_ts": ts_i, "ticker": "XAU-USD", "verdict_1_3d": "bullish",
                 "verdict_correct_3d": True})
            result_scored.append(
                {"journal_ts": ts_i, "ticker": "XAG-USD", "verdict_1_3d": "bullish",
                 "verdict_correct_3d": True})
        result = fin_evolve._compute_cross_asset(result_scored)
        assert "gold_bullish_silver_follows" in result
        assert result["gold_bullish_silver_follows"]["pct"] == 1.0

    def test_btc_eth_correlation(self):
        result_scored = []
        for i in range(3):
            ts_i = f"2026-03-{10+i}T12:00:00+00:00"
            result_scored.append(
                {"journal_ts": ts_i, "ticker": "BTC-USD", "verdict_1_3d": "bullish",
                 "verdict_correct_3d": True})
            result_scored.append(
                {"journal_ts": ts_i, "ticker": "ETH-USD", "verdict_1_3d": "bullish",
                 "verdict_correct_3d": False})
        result = fin_evolve._compute_cross_asset(result_scored)
        assert "btc_bullish_eth_follows" in result
        assert result["btc_bullish_eth_follows"]["pct"] == 0.0

    def test_no_cross_asset_with_small_sample(self):
        scored = [
            {"journal_ts": "2026-03-10T12:00:00+00:00", "ticker": "XAU-USD",
             "verdict_1_3d": "bullish", "verdict_correct_3d": True},
            {"journal_ts": "2026-03-10T12:00:00+00:00", "ticker": "XAG-USD",
             "verdict_1_3d": "bullish", "verdict_correct_3d": True},
        ]
        result = fin_evolve._compute_cross_asset(scored)
        assert "gold_bullish_silver_follows" not in result


# ---------------------------------------------------------------------------
# Tests: _normalize_scored (NEW)
# ---------------------------------------------------------------------------

class TestNormalizeScored:
    def test_normalizes_fin_command(self):
        fin = [{"command": "fin-silver", "ticker": "XAG-USD",
                "verdict_1_3d": "bullish", "verdict_1_3d_conf": 0.7,
                "verdict_correct_3d": True, "regime": "trending-up",
                "outcome_3d_pct": 2.0}]
        result = fin_evolve._normalize_scored(fin, [])
        assert len(result) == 1
        assert result[0]["source"] == "fin-silver"

    def test_normalizes_journal(self):
        journal = [{"ticker": "BTC-USD", "outlook": "bearish",
                    "conviction": 0.5, "correct_3d": False,
                    "regime": "range-bound", "outcome_3d_pct": 1.0,
                    "journal_ts": "2026-03-10T12:00:00+00:00"}]
        result = fin_evolve._normalize_scored([], journal)
        assert len(result) == 1
        assert result[0]["source"] == "layer2"
        assert result[0]["verdict_1_3d"] == "bearish"
        assert result[0]["verdict_1_3d_conf"] == 0.5

    def test_merges_both(self):
        fin = [{"command": "fin-silver", "ticker": "XAG-USD",
                "verdict_1_3d": "bullish", "verdict_correct_3d": True}]
        journal = [{"ticker": "BTC-USD", "outlook": "bearish",
                    "conviction": 0.5, "correct_3d": False,
                    "journal_ts": "ts"}]
        result = fin_evolve._normalize_scored(fin, journal)
        assert len(result) == 2
        sources = {r["source"] for r in result}
        assert "fin-silver" in sources
        assert "layer2" in sources


# ---------------------------------------------------------------------------
# Tests: _compute_layer2_patterns (NEW)
# ---------------------------------------------------------------------------

class TestComputeLayer2Patterns:
    def test_by_trigger(self):
        scored = [
            {"trigger": "consensus BUY", "correct_3d": True},
            {"trigger": "consensus SELL", "correct_3d": True},
            {"trigger": "consensus BUY", "correct_3d": False},
            {"trigger": "price_move 2.5%", "correct_3d": True},
            {"trigger": "price_move 3.1%", "correct_3d": False},
            {"trigger": "price_move 4.0%", "correct_3d": True},
        ]
        result = fin_evolve._compute_layer2_patterns(scored)
        assert result is not None
        assert "by_trigger" in result
        assert "consensus" in result["by_trigger"]
        assert "price_move" in result["by_trigger"]

    def test_by_conviction(self):
        scored = [
            {"conviction": 0.8, "correct_3d": True},
            {"conviction": 0.9, "correct_3d": True},
            {"conviction": 0.85, "correct_3d": False},
            {"conviction": 0.5, "correct_3d": True},
            {"conviction": 0.6, "correct_3d": False},
            {"conviction": 0.55, "correct_3d": True},
        ]
        result = fin_evolve._compute_layer2_patterns(scored)
        assert result is not None
        assert "by_conviction" in result

    def test_not_enough_data(self):
        scored = [{"trigger": "consensus", "correct_3d": True}]
        assert fin_evolve._compute_layer2_patterns(scored) is None


# ---------------------------------------------------------------------------
# Tests: evolve (full pipeline)
# ---------------------------------------------------------------------------

class TestEvolve:
    def test_evolve_with_enough_data(self, tmp_path):
        scored = []
        for i in range(6):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24), price_usd=85.0 + i,
                              verdict_1_3d="bullish", verdict_1_3d_conf=0.7,
                              regime="trending-up")
            v["outcome_3d_pct"] = 2.0 if i % 2 == 0 else -1.0
            v["verdict_correct_3d"] = i % 2 == 0
            scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, scored)
        result = fin_evolve.evolve()
        assert result is not None
        assert result["total_verdicts"] == 6
        assert result["total_fin_command"] == 6
        assert result["total_journal"] == 0
        assert "by_command" in result
        assert "by_source" in result
        assert "by_regime" in result
        assert "by_ticker" in result
        assert "cross_asset" in result
        lessons = _read_json(fin_evolve._LESSONS_FILE)
        assert lessons["total_verdicts"] == 6
        legacy = _read_json(fin_evolve._LEGACY_LESSONS_FILE)
        assert legacy["total_verdicts"] == 6

    def test_evolve_not_enough_data(self, tmp_path):
        scored = []
        for i in range(3):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24))
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, scored)
        assert fin_evolve.evolve() is None

    def test_evolve_empty_log(self, tmp_path):
        assert fin_evolve.evolve() is None

    def test_evolve_per_ticker_stats(self, tmp_path):
        scored = []
        for i in range(3):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24), ticker="XAG-USD",
                              command="fin-silver")
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)
        for i in range(3):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24), ticker="XAU-USD",
                              command="fin-gold")
            v["outcome_3d_pct"] = -1.0
            v["verdict_correct_3d"] = False
            scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, scored)
        result = fin_evolve.evolve()
        assert result["by_ticker"]["XAG-USD"]["accuracy_3d"] == 1.0
        assert result["by_ticker"]["XAU-USD"]["accuracy_3d"] == 0.0

    def test_evolve_by_command(self, tmp_path):
        scored = []
        for i in range(3):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24), command="fin-silver")
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)
        for i in range(3):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24), command="fin-gold",
                              ticker="XAU-USD")
            v["outcome_3d_pct"] = -1.0
            v["verdict_correct_3d"] = False
            scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, scored)
        result = fin_evolve.evolve()
        assert result["by_command"]["fin-silver"]["accuracy_3d"] == 1.0
        assert result["by_command"]["fin-gold"]["accuracy_3d"] == 0.0

    def test_evolve_execution_stats(self, tmp_path):
        scored = []
        for i in range(5):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24))
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            v["execution_time_sec"] = 10.0 + i * 5
            scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, scored)
        result = fin_evolve.evolve()
        assert result["execution_stats"]["min_sec"] == 10.0
        assert result["execution_stats"]["max_sec"] == 30.0
        assert result["execution_stats"]["n"] == 5

    def test_evolve_with_journal_only(self, tmp_path):
        outcomes = []
        for i in range(6):
            outcomes.append({
                "journal_ts": (datetime.now(UTC) - timedelta(hours=100 + i * 24)).isoformat(),
                "ticker": "BTC-USD", "source": "layer2", "outlook": "bullish",
                "conviction": 0.6, "price_at_verdict": 67000 + i * 100,
                "regime": "trending-up",
                "outcome_3d_pct": 2.0 if i % 2 == 0 else -1.0,
                "correct_3d": i % 2 == 0,
            })
        _write_jsonl(fin_evolve._JOURNAL_OUTCOMES_FILE, outcomes)
        result = fin_evolve.evolve()
        assert result is not None
        assert result["total_journal"] == 6
        assert result["total_fin_command"] == 0
        assert "layer2" in result["by_source"]
        assert "BTC-USD" in result["by_ticker"]

    def test_evolve_merges_both_sources(self, tmp_path):
        fin_scored = []
        for i in range(3):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24), ticker="XAG-USD")
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            fin_scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, fin_scored)
        journal = []
        for i in range(3):
            journal.append({
                "journal_ts": (datetime.now(UTC) - timedelta(hours=100 + i * 24)).isoformat(),
                "ticker": "BTC-USD", "source": "layer2", "outlook": "bullish",
                "conviction": 0.5, "price_at_verdict": 67000,
                "regime": "trending-up", "outcome_3d_pct": 1.5, "correct_3d": True,
            })
        _write_jsonl(fin_evolve._JOURNAL_OUTCOMES_FILE, journal)
        result = fin_evolve.evolve()
        assert result["total_verdicts"] == 6
        assert result["total_fin_command"] == 3
        assert result["total_journal"] == 3
        assert "fin-silver" in result["by_source"]
        assert "layer2" in result["by_source"]

    def test_evolve_by_source(self, tmp_path):
        fin_scored = []
        for i in range(3):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24), command="fin-silver")
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            fin_scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, fin_scored)
        journal = []
        for i in range(3):
            journal.append({
                "journal_ts": (datetime.now(UTC) - timedelta(hours=100 + i * 24)).isoformat(),
                "ticker": "BTC-USD", "source": "layer2", "outlook": "bullish",
                "conviction": 0.5, "regime": "trending-up",
                "outcome_3d_pct": -1.0, "correct_3d": False,
            })
        _write_jsonl(fin_evolve._JOURNAL_OUTCOMES_FILE, journal)
        result = fin_evolve.evolve()
        assert result["by_source"]["fin-silver"]["accuracy_3d"] == 1.0
        assert result["by_source"]["layer2"]["accuracy_3d"] == 0.0

    def test_evolve_all_tickers(self, tmp_path):
        scored = []
        for ticker in ("XAG-USD", "XAU-USD", "BTC-USD", "ETH-USD", "NVDA"):
            v = _make_verdict(ts_offset_hours=-100, ticker=ticker, command="fin-silver")
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, scored)
        result = fin_evolve.evolve()
        assert "BTC-USD" in result["by_ticker"]
        assert "ETH-USD" in result["by_ticker"]
        assert "NVDA" in result["by_ticker"]


# ---------------------------------------------------------------------------
# Tests: maybe_evolve (throttling)
# ---------------------------------------------------------------------------

class TestMaybeEvolve:
    def test_runs_on_first_call(self, tmp_path):
        scored = []
        for i in range(6):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24))
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, scored)
        result = fin_evolve.maybe_evolve()
        assert result is not None
        state = _read_json(fin_evolve._EVOLVE_STATE_FILE)
        assert state["status"] == "ok"
        assert "journal_outcomes_scored" in state

    def test_throttled_on_second_call(self, tmp_path):
        scored = []
        for i in range(6):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24))
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, scored)
        fin_evolve.maybe_evolve()
        assert fin_evolve.maybe_evolve() is None

    def test_runs_after_interval(self, tmp_path, monkeypatch):
        state = {
            "last_run_epoch": time.time() - fin_evolve._EVOLVE_INTERVAL_SEC - 100,
            "status": "ok",
        }
        fin_evolve._atomic_write_json(fin_evolve._EVOLVE_STATE_FILE, state)
        scored = []
        for i in range(6):
            v = _make_verdict(ts_offset_hours=-(100 + i * 24))
            v["outcome_3d_pct"] = 2.0
            v["verdict_correct_3d"] = True
            scored.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, scored)
        assert fin_evolve.maybe_evolve() is not None

    def test_4h_interval(self, tmp_path):
        assert fin_evolve._EVOLVE_INTERVAL_SEC == 4 * 3600


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
        assert result["BUY"]["accuracy"] == pytest.approx(0.667, abs=0.01)
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
        now = datetime.now(UTC)
        verdicts = []
        for i in range(6):
            v = _make_verdict(ts_offset_hours=-(96 + i * 2), price_usd=85.0 + i * 0.5,
                              verdict_1_3d="bullish" if i < 4 else "bearish",
                              verdict_1_3d_conf=0.6 + i * 0.05,
                              regime="trending-up" if i < 3 else "range-bound")
            verdicts.append(v)
        _write_jsonl(fin_evolve._LOG_FILE, verdicts)
        price_snaps = []
        for v in verdicts:
            v_ts = fin_evolve._parse_iso(v["ts"])
            price_at_verdict = v["price_usd"]
            price_snaps.append(_make_price_snap(
                v_ts + timedelta(days=1), {"XAG-USD": price_at_verdict + 1.0}))
            price_snaps.append(_make_price_snap(
                v_ts + timedelta(days=3), {"XAG-USD": price_at_verdict + 2.0}))
        _write_jsonl(fin_evolve._PRICE_FILE, price_snaps)
        n = fin_evolve.backfill_outcomes()
        assert n > 0
        entries = _read_jsonl(fin_evolve._LOG_FILE)
        for e in entries:
            assert "outcome_1d_pct" in e
            assert "outcome_3d_pct" in e
        result = fin_evolve.evolve()
        assert result is not None
        assert result["total_verdicts"] >= 5
        assert "calibration_advice" in result
        assert "by_command" in result
        assert "fin-silver" in result["by_command"]

    def test_full_pipeline_with_journal(self, tmp_path):
        now = datetime.now(UTC)
        journal_entries = []
        for i in range(6):
            journal_entries.append(_make_journal_entry(
                ts_offset_hours=-(96 + i * 2),
                regime="trending-up" if i < 3 else "range-bound",
                tickers={"BTC-USD": {
                    "outlook": "bullish" if i < 4 else "bearish",
                    "conviction": 0.6 + i * 0.05,
                }},
                prices={"BTC-USD": 67000.0 + i * 100},
            ))
        _write_jsonl(fin_evolve._JOURNAL_FILE, journal_entries)
        price_snaps = []
        for entry in journal_entries:
            entry_ts = fin_evolve._parse_iso(entry["ts"])
            btc_price = entry["prices"]["BTC-USD"]
            price_snaps.append(_make_price_snap(
                entry_ts + timedelta(days=1), {"BTC-USD": btc_price + 500}))
            price_snaps.append(_make_price_snap(
                entry_ts + timedelta(days=3), {"BTC-USD": btc_price + 1000}))
        _write_jsonl(fin_evolve._PRICE_FILE, price_snaps)
        n_journal = fin_evolve.backfill_journal_outcomes()
        assert n_journal == 6
        result = fin_evolve.evolve()
        assert result is not None
        assert result["total_journal"] == 6
        assert "BTC-USD" in result["by_ticker"]
        assert "layer2" in result["by_source"]
