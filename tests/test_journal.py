import json
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch

from portfolio.journal import (
    load_recent,
    _is_all_hold,
    _non_neutral_tickers,
    build_context,
)


def make_entry(
    age_hours=0.5,
    trigger="check_in",
    regime="range-bound",
    patient_action="HOLD",
    patient_reason="waiting",
    bold_action="HOLD",
    bold_reason="no setup",
    tickers=None,
    watchlist=None,
    prices=None,
    reflection=None,
    continues=None,
    now=None,
):
    if now is None:
        now = datetime.now(timezone.utc)
    ts = now - timedelta(hours=age_hours)
    entry = {
        "ts": ts.isoformat(),
        "trigger": trigger,
        "regime": regime,
        "decisions": {
            "patient": {"action": patient_action, "reasoning": patient_reason},
            "bold": {"action": bold_action, "reasoning": bold_reason},
        },
        "tickers": tickers
        or {
            "BTC-USD": {"outlook": "neutral", "thesis": "", "levels": []},
            "ETH-USD": {"outlook": "neutral", "thesis": "", "levels": []},
        },
        "watchlist": watchlist or [],
        "prices": prices or {"BTC-USD": 66800.0, "ETH-USD": 1952.0},
    }
    if reflection is not None:
        entry["reflection"] = reflection
    if continues is not None:
        entry["continues"] = continues
    return entry


def write_journal(tmp_path, entries):
    jf = tmp_path / "layer2_journal.jsonl"
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    jf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jf


class TestLoadRecent:
    def test_empty_file(self, tmp_path):
        jf = tmp_path / "layer2_journal.jsonl"
        jf.write_text("", encoding="utf-8")
        with patch("portfolio.journal.JOURNAL_FILE", jf):
            assert load_recent() == []

    def test_missing_file(self, tmp_path):
        jf = tmp_path / "nonexistent.jsonl"
        with patch("portfolio.journal.JOURNAL_FILE", jf):
            assert load_recent() == []

    def test_age_filtering(self, tmp_path):
        now = datetime.now(timezone.utc)
        entries = [
            make_entry(age_hours=10, now=now),
            make_entry(age_hours=1, now=now),
        ]
        jf = write_journal(tmp_path, entries)
        with patch("portfolio.journal.JOURNAL_FILE", jf):
            result = load_recent(max_age_hours=8)
        assert len(result) == 1

    def test_max_entries(self, tmp_path):
        now = datetime.now(timezone.utc)
        entries = [make_entry(age_hours=i * 0.1, now=now) for i in range(20)]
        jf = write_journal(tmp_path, entries)
        with patch("portfolio.journal.JOURNAL_FILE", jf):
            result = load_recent(max_entries=5)
        assert len(result) == 5

    def test_malformed_lines_skipped(self, tmp_path):
        now = datetime.now(timezone.utc)
        good = make_entry(age_hours=0.5, now=now)
        jf = tmp_path / "layer2_journal.jsonl"
        jf.write_text("not json\n" + json.dumps(good) + "\n{}\n", encoding="utf-8")
        with patch("portfolio.journal.JOURNAL_FILE", jf):
            result = load_recent()
        assert len(result) == 1


class TestIsAllHold:
    def test_both_hold(self):
        assert _is_all_hold(make_entry()) is True

    def test_one_buy(self):
        assert _is_all_hold(make_entry(patient_action="BUY BTC-USD")) is False

    def test_missing_decisions(self):
        assert _is_all_hold({}) is True


class TestNonNeutralTickers:
    def test_filters_neutral(self):
        e = make_entry(
            tickers={
                "BTC-USD": {"outlook": "bullish", "thesis": "up"},
                "ETH-USD": {"outlook": "neutral", "thesis": ""},
            }
        )
        result = _non_neutral_tickers(e)
        assert "BTC-USD" in result
        assert "ETH-USD" not in result

    def test_keeps_bearish(self):
        e = make_entry(tickers={"BTC-USD": {"outlook": "bearish", "thesis": "down"}})
        assert "BTC-USD" in _non_neutral_tickers(e)


class TestBuildContextBackwardCompat:
    def test_empty_entries_fresh_start(self):
        assert "Fresh start" in build_context([])

    def test_hold_compression(self):
        now = datetime.now(timezone.utc)
        entries = [make_entry(age_hours=i * 0.3, now=now) for i in range(3)]
        assert "3x HOLD" in build_context(entries)

    def test_single_hold_not_compressed(self):
        now = datetime.now(timezone.utc)
        md = build_context([make_entry(age_hours=0.5, now=now)])
        assert "1x HOLD" not in md
        assert "trigger:" in md

    def test_watchlist_rendered(self):
        now = datetime.now(timezone.utc)
        md = build_context(
            [make_entry(age_hours=0.5, now=now, watchlist=["BTC breakout above 70K"])]
        )
        assert "BTC breakout above 70K" in md

    def test_prices_rendered(self):
        now = datetime.now(timezone.utc)
        md = build_context(
            [make_entry(age_hours=0.5, now=now, prices={"BTC-USD": 66800.0})]
        )
        assert "$66,800.00" in md

    def test_non_hold_not_compressed(self):
        now = datetime.now(timezone.utc)
        entries = [
            make_entry(age_hours=1.5, now=now),
            make_entry(age_hours=1.0, now=now, patient_action="BUY BTC-USD"),
            make_entry(age_hours=0.5, now=now),
        ]
        md = build_context(entries)
        assert "BUY BTC-USD" in md
        assert "3x HOLD" not in md

    def test_regime_displayed(self):
        now = datetime.now(timezone.utc)
        md = build_context([make_entry(age_hours=0.5, now=now, regime="trending-up")])
        assert "trending-up" in md
