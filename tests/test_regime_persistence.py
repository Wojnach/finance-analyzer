"""Tests for regime persistence in signal log (Task 1).

Verifies that:
- log_signal_snapshot stores regime per ticker from extra['_regime']
- Missing _regime defaults to "unknown"
- SignalDB schema includes regime column in ticker_signals
- insert_snapshot stores regime in the DB
- load_entries reconstructs regime in ticker dict
- Migration: ALTER TABLE ADD COLUMN handles existing DBs gracefully
"""

from unittest.mock import patch

import pytest

from portfolio.outcome_tracker import log_signal_snapshot
from portfolio.signal_db import SignalDB

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signals_dict(regime=None, ticker="BTC-USD"):
    """Build a minimal signals_dict for log_signal_snapshot."""
    extra = {}
    if regime is not None:
        extra["_regime"] = regime
    return {
        ticker: {
            "indicators": {"rsi": 45},
            "extra": extra,
            "action": "HOLD",
        }
    }


def _make_entry(ts="2026-02-21T12:00:00+00:00", regime="trending-up"):
    """Build a minimal signal_log entry with regime set."""
    return {
        "ts": ts,
        "trigger_reasons": ["test"],
        "fx_rate": 10.85,
        "tickers": {
            "BTC-USD": {
                "price_usd": 67000.0,
                "consensus": "BUY",
                "buy_count": 2,
                "sell_count": 0,
                "total_voters": 2,
                "signals": {"rsi": "BUY", "macd": "HOLD"},
                "regime": regime,
            }
        },
        "outcomes": {},
    }


@pytest.fixture
def db(tmp_path):
    """Create a SignalDB in a temp directory."""
    d = SignalDB(tmp_path / "test.db")
    yield d
    d.close()


# ---------------------------------------------------------------------------
# log_signal_snapshot — regime extraction
# ---------------------------------------------------------------------------


class TestLogSignalSnapshotRegime:
    """log_signal_snapshot must persist regime from extra['_regime']."""

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_regime_stored_in_entry(self, mock_db_cls, mock_append):
        """Regime from _regime key must appear in ticker dict of returned entry."""
        mock_db_cls.side_effect = ImportError("no db")

        signals_dict = _make_signals_dict(regime="trending-up")
        entry = log_signal_snapshot(signals_dict, {"BTC-USD": 67000.0}, 10.5, ["test"])

        assert entry["tickers"]["BTC-USD"]["regime"] == "trending-up"

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_regime_defaults_to_unknown_when_missing(self, mock_db_cls, mock_append):
        """When _regime key is absent from extra, regime should default to 'unknown'."""
        mock_db_cls.side_effect = ImportError("no db")

        signals_dict = _make_signals_dict(regime=None)  # no _regime in extra
        entry = log_signal_snapshot(signals_dict, {"BTC-USD": 67000.0}, 10.5, ["test"])

        assert entry["tickers"]["BTC-USD"]["regime"] == "unknown"

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_all_regime_values_stored(self, mock_db_cls, mock_append):
        """All four regime values are stored faithfully."""
        mock_db_cls.side_effect = ImportError("no db")

        for regime_val in ["trending-up", "trending-down", "ranging", "high-vol"]:
            signals_dict = _make_signals_dict(regime=regime_val, ticker="BTC-USD")
            entry = log_signal_snapshot(signals_dict, {"BTC-USD": 67000.0}, 10.5, ["test"])
            assert entry["tickers"]["BTC-USD"]["regime"] == regime_val, (
                f"Expected regime={regime_val!r}, got "
                f"{entry['tickers']['BTC-USD'].get('regime')!r}"
            )

    @patch("portfolio.outcome_tracker.atomic_append_jsonl")
    @patch("portfolio.outcome_tracker.SignalDB", create=True)
    def test_regime_per_ticker_independent(self, mock_db_cls, mock_append):
        """Each ticker's regime is extracted independently."""
        mock_db_cls.side_effect = ImportError("no db")

        signals_dict = {
            "BTC-USD": {
                "indicators": {},
                "extra": {"_regime": "trending-up"},
                "action": "BUY",
            },
            "ETH-USD": {
                "indicators": {},
                "extra": {"_regime": "ranging"},
                "action": "HOLD",
            },
            "XAG-USD": {
                "indicators": {},
                "extra": {},  # missing _regime
                "action": "SELL",
            },
        }
        entry = log_signal_snapshot(
            signals_dict,
            {"BTC-USD": 67000.0, "ETH-USD": 2000.0, "XAG-USD": 30.0},
            10.5,
            ["test"],
        )

        assert entry["tickers"]["BTC-USD"]["regime"] == "trending-up"
        assert entry["tickers"]["ETH-USD"]["regime"] == "ranging"
        assert entry["tickers"]["XAG-USD"]["regime"] == "unknown"


# ---------------------------------------------------------------------------
# SignalDB — schema has regime column
# ---------------------------------------------------------------------------


class TestSignalDBSchema:
    def test_ticker_signals_has_regime_column(self, db):
        """ticker_signals table must have a 'regime' column."""
        conn = db._get_conn()
        pragma = conn.execute("PRAGMA table_info(ticker_signals)").fetchall()
        col_names = [row[1] for row in pragma]
        assert "regime" in col_names

    def test_regime_column_default_is_unknown(self, db):
        """The regime column must have a default value of 'unknown'."""
        conn = db._get_conn()
        pragma = conn.execute("PRAGMA table_info(ticker_signals)").fetchall()
        # PRAGMA table_info columns: cid, name, type, notnull, dflt_value, pk
        for row in pragma:
            if row[1] == "regime":
                # Default value should be 'unknown' (stored as "'unknown'" in SQLite)
                assert row[4] is not None, "regime column should have a default value"
                assert "unknown" in str(row[4])
                return
        pytest.fail("regime column not found in ticker_signals")


# ---------------------------------------------------------------------------
# SignalDB — insert_snapshot stores regime
# ---------------------------------------------------------------------------


class TestSignalDBInsertRegime:
    def test_insert_stores_regime(self, db):
        """insert_snapshot must store regime in the DB row."""
        entry = _make_entry(regime="trending-down")
        db.insert_snapshot(entry)

        conn = db._get_conn()
        row = conn.execute(
            "SELECT regime FROM ticker_signals WHERE ticker = 'BTC-USD'"
        ).fetchone()
        assert row is not None
        assert row[0] == "trending-down"

    def test_insert_regime_unknown_when_missing(self, db):
        """When regime key is absent from ticker dict, DB should store 'unknown'."""
        entry = {
            "ts": "2026-02-22T12:00:00+00:00",
            "trigger_reasons": [],
            "fx_rate": 10.85,
            "tickers": {
                "ETH-USD": {
                    "price_usd": 2000.0,
                    "consensus": "HOLD",
                    "buy_count": 0,
                    "sell_count": 0,
                    "total_voters": 0,
                    "signals": {},
                    # no 'regime' key
                }
            },
            "outcomes": {},
        }
        db.insert_snapshot(entry)

        conn = db._get_conn()
        row = conn.execute(
            "SELECT regime FROM ticker_signals WHERE ticker = 'ETH-USD'"
        ).fetchone()
        assert row is not None
        assert row[0] == "unknown"

    def test_insert_all_regime_values(self, db):
        """All four regime types round-trip through DB correctly."""
        for i, regime_val in enumerate(["trending-up", "trending-down", "ranging", "high-vol"]):
            ts = f"2026-02-2{i + 1}T12:00:00+00:00"
            entry = _make_entry(ts=ts, regime=regime_val)
            db.insert_snapshot(entry)

        conn = db._get_conn()
        rows = conn.execute(
            "SELECT regime FROM ticker_signals ORDER BY rowid"
        ).fetchall()
        regimes = [r[0] for r in rows]
        assert "trending-up" in regimes
        assert "trending-down" in regimes
        assert "ranging" in regimes
        assert "high-vol" in regimes


# ---------------------------------------------------------------------------
# SignalDB — load_entries reconstructs regime
# ---------------------------------------------------------------------------


class TestSignalDBLoadRegime:
    def test_load_entries_includes_regime(self, db):
        """load_entries() must include regime in each ticker dict."""
        entry = _make_entry(regime="high-vol")
        db.insert_snapshot(entry)

        entries = db.load_entries()
        assert len(entries) == 1
        assert entries[0]["tickers"]["BTC-USD"]["regime"] == "high-vol"

    def test_load_entries_regime_defaults_to_unknown(self, db):
        """If regime is NULL or missing in DB, load_entries returns 'unknown'."""
        # Insert without regime key in ticker dict
        entry = {
            "ts": "2026-02-25T10:00:00+00:00",
            "trigger_reasons": [],
            "fx_rate": 10.5,
            "tickers": {
                "XAU-USD": {
                    "price_usd": 2200.0,
                    "consensus": "BUY",
                    "buy_count": 3,
                    "sell_count": 1,
                    "total_voters": 4,
                    "signals": {},
                    # no 'regime'
                }
            },
            "outcomes": {},
        }
        db.insert_snapshot(entry)

        entries = db.load_entries()
        assert len(entries) == 1
        assert entries[0]["tickers"]["XAU-USD"]["regime"] == "unknown"

    def test_load_entries_round_trip_multiple_regimes(self, db):
        """Multiple snapshots with different regimes all load correctly."""
        for i, (ticker, regime_val) in enumerate([
            ("BTC-USD", "trending-up"),
            ("ETH-USD", "ranging"),
        ]):
            ts = f"2026-02-2{i + 1}T12:00:00+00:00"
            entry = {
                "ts": ts,
                "trigger_reasons": [],
                "fx_rate": 10.5,
                "tickers": {
                    ticker: {
                        "price_usd": 1000.0,
                        "consensus": "HOLD",
                        "buy_count": 0,
                        "sell_count": 0,
                        "total_voters": 0,
                        "signals": {},
                        "regime": regime_val,
                    }
                },
                "outcomes": {},
            }
            db.insert_snapshot(entry)

        entries = db.load_entries()
        assert len(entries) == 2

        regimes_by_ts = {e["ts"]: list(e["tickers"].values())[0]["regime"] for e in entries}
        assert regimes_by_ts["2026-02-21T12:00:00+00:00"] == "trending-up"
        assert regimes_by_ts["2026-02-22T12:00:00+00:00"] == "ranging"


# ---------------------------------------------------------------------------
# Migration: existing DB without regime column
# ---------------------------------------------------------------------------


class TestSignalDBMigration:
    def test_migration_adds_regime_to_existing_db(self, tmp_path):
        """Opening a DB that lacks regime column must add it via ALTER TABLE."""
        import sqlite3

        db_path = tmp_path / "old.db"

        # Create an old-style DB without the regime column
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL UNIQUE,
                trigger_reasons TEXT,
                fx_rate REAL
            );
            CREATE TABLE ticker_signals (
                snapshot_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                price_usd REAL,
                consensus TEXT,
                buy_count INTEGER,
                sell_count INTEGER,
                total_voters INTEGER,
                signals TEXT,
                PRIMARY KEY (snapshot_id, ticker),
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
            );
            CREATE TABLE outcomes (
                snapshot_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                horizon TEXT NOT NULL,
                price_usd REAL,
                change_pct REAL,
                outcome_ts TEXT,
                PRIMARY KEY (snapshot_id, ticker, horizon),
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
            );
        """)
        conn.commit()
        conn.close()

        # Opening via SignalDB should migrate the schema
        db = SignalDB(db_path)

        # Verify the column was added
        pragma = db._get_conn().execute("PRAGMA table_info(ticker_signals)").fetchall()
        col_names = [row[1] for row in pragma]
        assert "regime" in col_names
        db.close()

    def test_migration_preserves_existing_rows(self, tmp_path):
        """Migration must not destroy existing rows; old rows get regime='unknown'."""
        import sqlite3

        db_path = tmp_path / "old_with_data.db"

        # Create old-style DB with an existing row
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL UNIQUE,
                trigger_reasons TEXT,
                fx_rate REAL
            );
            CREATE TABLE ticker_signals (
                snapshot_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                price_usd REAL,
                consensus TEXT,
                buy_count INTEGER,
                sell_count INTEGER,
                total_voters INTEGER,
                signals TEXT,
                PRIMARY KEY (snapshot_id, ticker),
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
            );
            CREATE TABLE outcomes (
                snapshot_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                horizon TEXT NOT NULL,
                price_usd REAL,
                change_pct REAL,
                outcome_ts TEXT,
                PRIMARY KEY (snapshot_id, ticker, horizon),
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
            );
            INSERT INTO snapshots (ts, trigger_reasons, fx_rate)
                VALUES ('2026-01-01T00:00:00+00:00', '[]', 10.5);
            INSERT INTO ticker_signals
                (snapshot_id, ticker, price_usd, consensus, buy_count, sell_count, total_voters, signals)
                VALUES (1, 'BTC-USD', 60000.0, 'BUY', 3, 1, 4, '{}');
        """)
        conn.commit()
        conn.close()

        # Open and migrate
        db = SignalDB(db_path)

        # Existing row should now have regime = 'unknown' (the column default)
        row = db._get_conn().execute(
            "SELECT regime FROM ticker_signals WHERE ticker = 'BTC-USD'"
        ).fetchone()
        assert row is not None
        assert row[0] == "unknown"
        db.close()
