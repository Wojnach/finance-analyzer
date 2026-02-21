"""One-time migration: signal_log.jsonl → signal_log.db (SQLite).

Usage:
    python portfolio/migrate_signal_log.py

Reads every line from data/signal_log.jsonl and inserts into SQLite.
Idempotent — duplicate timestamps are silently skipped.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from portfolio.signal_db import SignalDB

DATA_DIR = BASE_DIR / "data"
SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"


def migrate():
    if not SIGNAL_LOG.exists():
        print("No signal_log.jsonl found — nothing to migrate.")
        return 0

    db = SignalDB()
    count = 0
    errors = 0

    with open(SIGNAL_LOG, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                db.insert_snapshot(entry)
                count += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Line {i}: {e}")

    db.close()
    print(f"Migrated {count} entries to {db.db_path}")
    if errors:
        print(f"  ({errors} lines failed)")
    return count


if __name__ == "__main__":
    migrate()
