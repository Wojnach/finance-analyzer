"""SQLite storage for signal snapshots — replaces signal_log.jsonl for reads.

Schema:
- snapshots: one row per invocation (ts, trigger_reasons, fx_rate)
- ticker_signals: one row per ticker per snapshot (price, consensus, votes, signals JSON)
- outcomes: one row per ticker per horizon per snapshot (backfilled prices + change_pct)

Usage:
    from portfolio.signal_db import SignalDB
    db = SignalDB()           # uses default path data/signal_log.db
    db.insert_snapshot(entry) # dict in same format as signal_log.jsonl line
    entries = db.load_entries()  # returns list[dict] matching JSONL format
"""

import json
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "signal_log.db"


class SignalDB:
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._ensure_schema()

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), timeout=10)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL UNIQUE,
                trigger_reasons TEXT,
                fx_rate REAL
            );

            CREATE TABLE IF NOT EXISTS ticker_signals (
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

            CREATE TABLE IF NOT EXISTS outcomes (
                snapshot_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                horizon TEXT NOT NULL,
                price_usd REAL,
                change_pct REAL,
                outcome_ts TEXT,
                PRIMARY KEY (snapshot_id, ticker, horizon),
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(ts);
            CREATE INDEX IF NOT EXISTS idx_ticker_signals_ticker ON ticker_signals(ticker);
            CREATE INDEX IF NOT EXISTS idx_outcomes_horizon ON outcomes(horizon);
        """)
        conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # --- Write ---

    def insert_snapshot(self, entry):
        """Insert a signal_log entry (same dict format as JSONL line).

        Skips silently if ts already exists (idempotent for migration).
        """
        conn = self._get_conn()
        ts = entry["ts"]
        trigger_reasons = json.dumps(entry.get("trigger_reasons", []))
        fx_rate = entry.get("fx_rate")

        try:
            cur = conn.execute(
                "INSERT INTO snapshots (ts, trigger_reasons, fx_rate) VALUES (?, ?, ?)",
                (ts, trigger_reasons, fx_rate),
            )
        except sqlite3.IntegrityError:
            return  # duplicate ts, skip

        snapshot_id = cur.lastrowid

        tickers = entry.get("tickers", {})
        for ticker, tdata in tickers.items():
            conn.execute(
                """INSERT INTO ticker_signals
                   (snapshot_id, ticker, price_usd, consensus, buy_count, sell_count, total_voters, signals)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot_id,
                    ticker,
                    tdata.get("price_usd"),
                    tdata.get("consensus"),
                    tdata.get("buy_count"),
                    tdata.get("sell_count"),
                    tdata.get("total_voters"),
                    json.dumps(tdata.get("signals", {})),
                ),
            )

        outcomes = entry.get("outcomes", {})
        for ticker, horizons in outcomes.items():
            if not isinstance(horizons, dict):
                continue
            for horizon, odata in horizons.items():
                if odata is None:
                    continue
                conn.execute(
                    """INSERT OR REPLACE INTO outcomes
                       (snapshot_id, ticker, horizon, price_usd, change_pct, outcome_ts)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        snapshot_id,
                        ticker,
                        horizon,
                        odata.get("price_usd"),
                        odata.get("change_pct"),
                        odata.get("ts"),
                    ),
                )

        conn.commit()

    def update_outcome(self, ts, ticker, horizon, price_usd, change_pct, outcome_ts):
        """Update a single outcome cell. Used by backfill."""
        conn = self._get_conn()
        row = conn.execute("SELECT id FROM snapshots WHERE ts = ?", (ts,)).fetchone()
        if not row:
            return False
        snapshot_id = row["id"]
        conn.execute(
            """INSERT OR REPLACE INTO outcomes
               (snapshot_id, ticker, horizon, price_usd, change_pct, outcome_ts)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (snapshot_id, ticker, horizon, price_usd, change_pct, outcome_ts),
        )
        conn.commit()
        return True

    # --- Read ---

    def load_entries(self):
        """Load all snapshots as list[dict] matching JSONL format.

        Compatible drop-in replacement for accuracy_stats.load_entries().
        """
        conn = self._get_conn()
        snapshots = conn.execute("SELECT * FROM snapshots ORDER BY ts").fetchall()

        entries = []
        for snap in snapshots:
            sid = snap["id"]

            tickers = {}
            for row in conn.execute(
                "SELECT * FROM ticker_signals WHERE snapshot_id = ?", (sid,)
            ):
                tickers[row["ticker"]] = {
                    "price_usd": row["price_usd"],
                    "consensus": row["consensus"],
                    "buy_count": row["buy_count"],
                    "sell_count": row["sell_count"],
                    "total_voters": row["total_voters"],
                    "signals": json.loads(row["signals"]) if row["signals"] else {},
                }

            outcomes = {}
            for row in conn.execute(
                "SELECT * FROM outcomes WHERE snapshot_id = ?", (sid,)
            ):
                if row["ticker"] not in outcomes:
                    outcomes[row["ticker"]] = {}
                outcomes[row["ticker"]][row["horizon"]] = {
                    "price_usd": row["price_usd"],
                    "change_pct": row["change_pct"],
                    "ts": row["outcome_ts"],
                }

            entries.append({
                "ts": snap["ts"],
                "trigger_reasons": json.loads(snap["trigger_reasons"]) if snap["trigger_reasons"] else [],
                "fx_rate": snap["fx_rate"],
                "tickers": tickers,
                "outcomes": outcomes,
            })

        return entries

    def snapshot_count(self):
        """Return total number of snapshots."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM snapshots").fetchone()
        return row["cnt"]

    def entries_missing_outcomes(self, horizon):
        """Find snapshot timestamps that are missing a specific horizon outcome.

        Returns list of (ts, ticker, price_usd) tuples for entries needing backfill.
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT s.ts, ts2.ticker, ts2.price_usd
               FROM snapshots s
               JOIN ticker_signals ts2 ON s.id = ts2.snapshot_id
               LEFT JOIN outcomes o ON s.id = o.snapshot_id
                   AND ts2.ticker = o.ticker AND o.horizon = ?
               WHERE o.snapshot_id IS NULL
               ORDER BY s.ts""",
            (horizon,),
        ).fetchall()
        return [(r["ts"], r["ticker"], r["price_usd"]) for r in rows]

    def signal_accuracy(self, horizon="1d"):
        """Compute per-signal accuracy directly via SQL.

        Returns dict matching accuracy_stats.signal_accuracy() format.
        """
        conn = self._get_conn()
        from portfolio.tickers import SIGNAL_NAMES

        stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}

        rows = conn.execute(
            """SELECT ts2.signals, o.change_pct
               FROM ticker_signals ts2
               JOIN outcomes o ON ts2.snapshot_id = o.snapshot_id AND ts2.ticker = o.ticker
               WHERE o.horizon = ? AND o.change_pct IS NOT NULL""",
            (horizon,),
        ).fetchall()

        for row in rows:
            signals = json.loads(row["signals"]) if row["signals"] else {}
            change_pct = row["change_pct"]
            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                stats[sig_name]["total"] += 1
                if (vote == "BUY" and change_pct > 0) or (vote == "SELL" and change_pct < 0):
                    stats[sig_name]["correct"] += 1

        result = {}
        for sig_name in SIGNAL_NAMES:
            s = stats[sig_name]
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
            result[sig_name] = {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": acc,
                "pct": round(acc * 100, 1),
            }
        return result

    def consensus_accuracy(self, horizon="1d"):
        """Compute consensus accuracy directly via SQL."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT ts2.consensus, o.change_pct
               FROM ticker_signals ts2
               JOIN outcomes o ON ts2.snapshot_id = o.snapshot_id AND ts2.ticker = o.ticker
               WHERE o.horizon = ? AND o.change_pct IS NOT NULL
                 AND ts2.consensus != 'HOLD'""",
            (horizon,),
        ).fetchall()

        correct = 0
        total = 0
        for row in rows:
            total += 1
            if (row["consensus"] == "BUY" and row["change_pct"] > 0) or \
               (row["consensus"] == "SELL" and row["change_pct"] < 0):
                correct += 1

        acc = correct / total if total > 0 else 0.0
        return {
            "correct": correct,
            "total": total,
            "accuracy": acc,
            "pct": round(acc * 100, 1),
        }

    def per_ticker_accuracy(self, horizon="1d"):
        """Compute per-ticker consensus accuracy via SQL."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT ts2.ticker, ts2.consensus, o.change_pct
               FROM ticker_signals ts2
               JOIN outcomes o ON ts2.snapshot_id = o.snapshot_id AND ts2.ticker = o.ticker
               WHERE o.horizon = ? AND o.change_pct IS NOT NULL
                 AND ts2.consensus != 'HOLD'""",
            (horizon,),
        ).fetchall()

        from collections import defaultdict
        stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for row in rows:
            stats[row["ticker"]]["total"] += 1
            if (row["consensus"] == "BUY" and row["change_pct"] > 0) or \
               (row["consensus"] == "SELL" and row["change_pct"] < 0):
                stats[row["ticker"]]["correct"] += 1

        result = {}
        for ticker, s in stats.items():
            acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
            result[ticker] = {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": acc,
                "pct": round(acc * 100, 1),
            }
        return result

    def ticker_signal_accuracy(self, horizon="1d", min_samples=0):
        """Per-ticker per-signal accuracy cross-tabulation via SQL.

        Returns: {ticker: {signal_name: {correct, total, accuracy, pct}}}
        """
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT ts2.ticker, ts2.signals, o.change_pct
               FROM ticker_signals ts2
               JOIN outcomes o ON ts2.snapshot_id = o.snapshot_id AND ts2.ticker = o.ticker
               WHERE o.horizon = ? AND o.change_pct IS NOT NULL""",
            (horizon,),
        ).fetchall()

        from collections import defaultdict
        stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

        for row in rows:
            signals = json.loads(row["signals"]) if row["signals"] else {}
            change_pct = row["change_pct"]
            ticker = row["ticker"]
            for sig_name, vote in signals.items():
                if vote == "HOLD":
                    continue
                stats[ticker][sig_name]["total"] += 1
                if (vote == "BUY" and change_pct > 0) or (vote == "SELL" and change_pct < 0):
                    stats[ticker][sig_name]["correct"] += 1

        result = {}
        for ticker, sig_stats in stats.items():
            ticker_result = {}
            for sig_name, s in sig_stats.items():
                if s["total"] < min_samples:
                    continue
                acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
                ticker_result[sig_name] = {
                    "correct": s["correct"],
                    "total": s["total"],
                    "accuracy": acc,
                    "pct": round(acc * 100, 1),
                }
            if ticker_result:
                result[ticker] = ticker_result
        return result
