"""Tests for SQLite signal log storage (portfolio/signal_db.py).

Covers:
- Schema creation
- Insert + load round-trip
- Idempotent inserts (duplicate ts)
- Outcome updates
- Accuracy queries (signal, consensus, per-ticker)
- Missing outcome detection
- JSONL ↔ SQLite format compatibility
"""

import json
import pytest
from pathlib import Path

from portfolio.signal_db import SignalDB


def _make_entry(ts="2026-02-21T12:00:00+00:00", tickers=None, outcomes=None):
    """Build a minimal signal_log entry."""
    if tickers is None:
        tickers = {
            "BTC-USD": {
                "price_usd": 67000.0,
                "consensus": "BUY",
                "buy_count": 4,
                "sell_count": 1,
                "total_voters": 5,
                "signals": {
                    "rsi": "BUY", "macd": "BUY", "ema": "BUY",
                    "bb": "HOLD", "fear_greed": "BUY", "sentiment": "HOLD",
                    "ministral": "HOLD", "ml": "HOLD", "funding": "HOLD",
                    "volume": "SELL",
                    "trend": "HOLD", "momentum": "HOLD", "volume_flow": "HOLD",
                    "volatility_sig": "HOLD", "candlestick": "HOLD",
                    "structure": "HOLD", "fibonacci": "HOLD", "smart_money": "HOLD",
                    "oscillators": "HOLD", "heikin_ashi": "HOLD",
                    "mean_reversion": "HOLD", "calendar": "HOLD",
                    "macro_regime": "HOLD", "momentum_factors": "HOLD",
                },
            },
        }
    return {
        "ts": ts,
        "trigger_reasons": ["signal_consensus"],
        "fx_rate": 10.85,
        "tickers": tickers,
        "outcomes": outcomes or {},
    }


@pytest.fixture
def db(tmp_path):
    """Create a SignalDB in a temp directory."""
    d = SignalDB(tmp_path / "test.db")
    yield d
    d.close()


class TestSchema:
    def test_creates_tables(self, db):
        conn = db._get_conn()
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "snapshots" in tables
        assert "ticker_signals" in tables
        assert "outcomes" in tables

    def test_creates_indexes(self, db):
        conn = db._get_conn()
        indexes = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()]
        assert "idx_snapshots_ts" in indexes


class TestInsertAndLoad:
    def test_round_trip(self, db):
        entry = _make_entry()
        db.insert_snapshot(entry)

        entries = db.load_entries()
        assert len(entries) == 1
        e = entries[0]
        assert e["ts"] == "2026-02-21T12:00:00+00:00"
        assert e["fx_rate"] == 10.85
        assert e["trigger_reasons"] == ["signal_consensus"]
        assert "BTC-USD" in e["tickers"]
        assert e["tickers"]["BTC-USD"]["price_usd"] == 67000.0
        assert e["tickers"]["BTC-USD"]["consensus"] == "BUY"
        assert e["tickers"]["BTC-USD"]["signals"]["rsi"] == "BUY"

    def test_multiple_tickers(self, db):
        tickers = {
            "BTC-USD": {
                "price_usd": 67000.0, "consensus": "BUY",
                "buy_count": 3, "sell_count": 1, "total_voters": 4,
                "signals": {"rsi": "BUY"},
            },
            "ETH-USD": {
                "price_usd": 2000.0, "consensus": "SELL",
                "buy_count": 1, "sell_count": 3, "total_voters": 4,
                "signals": {"rsi": "SELL"},
            },
        }
        db.insert_snapshot(_make_entry(tickers=tickers))
        entries = db.load_entries()
        assert len(entries[0]["tickers"]) == 2
        assert entries[0]["tickers"]["ETH-USD"]["consensus"] == "SELL"

    def test_with_outcomes(self, db):
        outcomes = {
            "BTC-USD": {
                "1d": {"price_usd": 68000.0, "change_pct": 1.49, "ts": "2026-02-22T12:00:00+00:00"},
            }
        }
        db.insert_snapshot(_make_entry(outcomes=outcomes))
        entries = db.load_entries()
        assert entries[0]["outcomes"]["BTC-USD"]["1d"]["change_pct"] == 1.49

    def test_idempotent_insert(self, db):
        entry = _make_entry()
        db.insert_snapshot(entry)
        db.insert_snapshot(entry)  # same ts — should be silently skipped
        assert db.snapshot_count() == 1


class TestSnapshotCount:
    def test_empty(self, db):
        assert db.snapshot_count() == 0

    def test_after_inserts(self, db):
        db.insert_snapshot(_make_entry(ts="2026-02-21T12:00:00+00:00"))
        db.insert_snapshot(_make_entry(ts="2026-02-21T13:00:00+00:00"))
        assert db.snapshot_count() == 2


class TestUpdateOutcome:
    def test_updates_existing_snapshot(self, db):
        db.insert_snapshot(_make_entry())
        result = db.update_outcome(
            "2026-02-21T12:00:00+00:00", "BTC-USD", "1d",
            68000.0, 1.49, "2026-02-22T12:00:00+00:00",
        )
        assert result is True
        entries = db.load_entries()
        assert entries[0]["outcomes"]["BTC-USD"]["1d"]["change_pct"] == 1.49

    def test_nonexistent_snapshot_returns_false(self, db):
        result = db.update_outcome(
            "1999-01-01T00:00:00+00:00", "BTC-USD", "1d",
            68000.0, 1.49, "1999-01-02T00:00:00+00:00",
        )
        assert result is False


class TestEntriesMissingOutcomes:
    def test_finds_missing(self, db):
        db.insert_snapshot(_make_entry())
        missing = db.entries_missing_outcomes("1d")
        assert len(missing) == 1
        assert missing[0][0] == "2026-02-21T12:00:00+00:00"
        assert missing[0][1] == "BTC-USD"

    def test_none_missing_when_filled(self, db):
        outcomes = {
            "BTC-USD": {
                "1d": {"price_usd": 68000.0, "change_pct": 1.49, "ts": "t"},
            }
        }
        db.insert_snapshot(_make_entry(outcomes=outcomes))
        missing = db.entries_missing_outcomes("1d")
        assert len(missing) == 0


class TestSignalAccuracy:
    def test_basic_accuracy(self, db):
        # BTC goes up 2% → BUY signals correct, SELL signals wrong
        entry = _make_entry()
        entry["outcomes"] = {
            "BTC-USD": {
                "1d": {"price_usd": 68340.0, "change_pct": 2.0, "ts": "t"},
            }
        }
        db.insert_snapshot(entry)

        acc = db.signal_accuracy("1d")
        # rsi=BUY, macd=BUY, ema=BUY, fear_greed=BUY → all correct (price went up)
        assert acc["rsi"]["correct"] == 1
        assert acc["rsi"]["total"] == 1
        assert acc["rsi"]["accuracy"] == 1.0
        # volume=SELL → wrong (price went up)
        assert acc["volume"]["correct"] == 0
        assert acc["volume"]["total"] == 1

    def test_empty_db(self, db):
        acc = db.signal_accuracy("1d")
        assert acc["rsi"]["total"] == 0


class TestConsensusAccuracy:
    def test_correct_consensus(self, db):
        entry = _make_entry()
        entry["outcomes"] = {
            "BTC-USD": {
                "1d": {"price_usd": 68000.0, "change_pct": 1.5, "ts": "t"},
            }
        }
        db.insert_snapshot(entry)
        ca = db.consensus_accuracy("1d")
        assert ca["correct"] == 1
        assert ca["total"] == 1

    def test_wrong_consensus(self, db):
        entry = _make_entry()
        entry["tickers"]["BTC-USD"]["consensus"] = "SELL"
        entry["outcomes"] = {
            "BTC-USD": {
                "1d": {"price_usd": 68000.0, "change_pct": 1.5, "ts": "t"},
            }
        }
        db.insert_snapshot(entry)
        ca = db.consensus_accuracy("1d")
        assert ca["correct"] == 0
        assert ca["total"] == 1


class TestPerTickerAccuracy:
    def test_per_ticker(self, db):
        tickers = {
            "BTC-USD": {
                "price_usd": 67000.0, "consensus": "BUY",
                "buy_count": 3, "sell_count": 1, "total_voters": 4,
                "signals": {"rsi": "BUY"},
            },
            "ETH-USD": {
                "price_usd": 2000.0, "consensus": "SELL",
                "buy_count": 1, "sell_count": 3, "total_voters": 4,
                "signals": {"rsi": "SELL"},
            },
        }
        entry = _make_entry(tickers=tickers)
        entry["outcomes"] = {
            "BTC-USD": {"1d": {"price_usd": 68000.0, "change_pct": 1.5, "ts": "t"}},
            "ETH-USD": {"1d": {"price_usd": 1900.0, "change_pct": -5.0, "ts": "t"}},
        }
        db.insert_snapshot(entry)
        ta = db.per_ticker_accuracy("1d")
        assert ta["BTC-USD"]["correct"] == 1  # BUY + up
        assert ta["ETH-USD"]["correct"] == 1  # SELL + down


class TestJsonlCompatibility:
    def test_format_matches_jsonl(self, db):
        """Verify that load_entries() output matches JSONL entry format."""
        original = _make_entry()
        original["outcomes"] = {
            "BTC-USD": {
                "1d": {"price_usd": 68000.0, "change_pct": 1.49, "ts": "2026-02-22T12:00:00+00:00"},
            }
        }
        db.insert_snapshot(original)
        loaded = db.load_entries()[0]

        # Key fields must match
        assert loaded["ts"] == original["ts"]
        assert loaded["fx_rate"] == original["fx_rate"]
        assert loaded["trigger_reasons"] == original["trigger_reasons"]
        assert loaded["tickers"]["BTC-USD"]["price_usd"] == original["tickers"]["BTC-USD"]["price_usd"]
        assert loaded["tickers"]["BTC-USD"]["consensus"] == original["tickers"]["BTC-USD"]["consensus"]
        assert loaded["outcomes"]["BTC-USD"]["1d"]["change_pct"] == 1.49
