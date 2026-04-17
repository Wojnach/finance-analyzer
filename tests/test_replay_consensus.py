"""Smoke tests for scripts/replay_consensus.py (P1-E adversarial-review fix).

The replay script produced the +3.80pp validation number that justified
shipping the accuracy-gating reconfiguration. Without tests, a bug in the
replay loop could silently produce misleading numbers. These tests pin:

  - _load_entries parses a synthetic JSONL fixture and counts parse errors
  - _verdict_correct direction semantics
  - replay() end-to-end on a 2-entry synthetic signal_log fixture
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def synthetic_signal_log(tmp_path, monkeypatch):
    """Write a 2-entry signal_log.jsonl to tmp_path + point LOG_FILE at it."""
    log = tmp_path / "signal_log.jsonl"
    entries = [
        {
            "ts": "2026-04-16T12:00:00+00:00",
            "tickers": {
                "BTC-USD": {
                    "consensus": "BUY",
                    "regime": "trending-up",
                    "signals": {"rsi": "BUY", "ema": "BUY", "macd": "BUY", "volume": "BUY"},
                },
            },
            "outcomes": {
                "BTC-USD": {
                    "1d": {"change_pct": 1.5},
                    "3h": {"change_pct": 0.4},
                },
            },
        },
        {
            "ts": "2026-04-16T13:00:00+00:00",
            "tickers": {
                "BTC-USD": {
                    "consensus": "SELL",
                    "regime": "ranging",
                    "signals": {"rsi": "SELL", "ema": "SELL", "macd": "SELL"},
                },
            },
            "outcomes": {
                "BTC-USD": {
                    "1d": {"change_pct": 1.2},
                    "3h": {"change_pct": -0.3},
                },
            },
        },
    ]
    with log.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")
        fh.write("{not-valid-json}\n")
        fh.write("\n")

    from scripts import replay_consensus as rc
    monkeypatch.setattr(rc, "LOG_FILE", log)
    return log


class TestLoadEntries:
    """P1-E: _load_entries parses cleanly + counts malformed rows."""

    def test_returns_tuple_of_entries_and_parse_count(self, synthetic_signal_log):
        from scripts.replay_consensus import _load_entries
        entries, parse_errors = _load_entries(days=365)
        assert len(entries) == 2
        assert parse_errors == 1


class TestVerdictCorrect:
    """Direction semantics of the scoring predicate."""

    def test_buy_positive_change_correct(self):
        from scripts.replay_consensus import _verdict_correct
        assert _verdict_correct("BUY", 1.0) is True

    def test_buy_negative_change_wrong(self):
        from scripts.replay_consensus import _verdict_correct
        assert _verdict_correct("BUY", -1.0) is False

    def test_sell_positive_change_wrong(self):
        from scripts.replay_consensus import _verdict_correct
        assert _verdict_correct("SELL", 1.0) is False

    def test_sell_negative_change_correct(self):
        from scripts.replay_consensus import _verdict_correct
        assert _verdict_correct("SELL", -1.0) is True

    def test_hold_returns_none(self):
        from scripts.replay_consensus import _verdict_correct
        assert _verdict_correct("HOLD", 1.0) is None

    def test_missing_change_returns_none(self):
        from scripts.replay_consensus import _verdict_correct
        assert _verdict_correct("BUY", None) is None


class TestReplayRaisesOnEmptyCache:
    """P3-5: replay must raise (not silently return 0) on empty cache."""

    def test_replay_raises_on_empty_cache(self, tmp_path, monkeypatch):
        log = tmp_path / "signal_log.jsonl"
        log.write_text(json.dumps({
            "ts": "2026-04-16T12:00:00+00:00",
            "tickers": {"BTC-USD": {"consensus": "BUY", "regime": "ranging", "signals": {}}},
            "outcomes": {"BTC-USD": {"1d": {"change_pct": 1.0}}},
        }) + "\n")

        from scripts import replay_consensus as rc
        monkeypatch.setattr(rc, "LOG_FILE", log)
        import portfolio.accuracy_stats as acc_stats
        monkeypatch.setattr(acc_stats, "load_cached_accuracy",
                             lambda *a, **kw: None)

        with pytest.raises(RuntimeError, match="accuracy cache empty"):
            rc.replay(days=365, horizon="1d")
