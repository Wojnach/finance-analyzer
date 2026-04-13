"""Surface unresolved critical errors from data/critical_errors.jsonl.

Invoked at the start of every Claude Code session in this project (via
the STARTUP CHECK block in CLAUDE.md). Prints a compact summary of
unresolved entries from the last 7 days and exits non-zero if any are
found. The non-zero exit makes this suitable for future hook wiring.

Design notes:

* Append-only journal. Resolutions are recorded as follow-up entries with
  a ``resolves_ts`` reference rather than mutating earlier entries.
* 7-day lookback is arbitrary but long enough to span a weekend + a
  trading week. Tune via ``--days N``.
* Zero-error output stays silent to avoid adding noise when the system
  is healthy; non-zero output is compact enough to fit in a session's
  preamble without crowding the user's actual task.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_JOURNAL = Path(__file__).resolve().parent.parent / "data" / "critical_errors.jsonl"
DEFAULT_DAYS = 7


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _load_entries(journal: Path) -> list[dict]:
    if not journal.exists():
        return []
    entries = []
    for line in journal.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def find_unresolved(entries: list[dict], *, days: int, now: datetime | None = None) -> list[dict]:
    """Return entries with resolution=None from the last `days` days.

    A later entry with ``resolves_ts`` pointing at an earlier entry's ``ts``
    retroactively resolves that earlier entry.
    """
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(days=days)

    resolved_ts: set[str] = set()
    for e in entries:
        rts = e.get("resolves_ts")
        if rts:
            resolved_ts.add(rts)

    unresolved = []
    for e in entries:
        if e.get("resolution") is not None:
            continue
        if e.get("ts") in resolved_ts:
            continue
        parsed = _parse_ts(e.get("ts", ""))
        if parsed is None or parsed < cutoff:
            continue
        unresolved.append(e)
    return unresolved


def format_entry(entry: dict) -> str:
    ts = entry.get("ts", "?")
    category = entry.get("category", "?")
    caller = entry.get("caller", "?")
    msg = entry.get("message", "")
    if len(msg) > 180:
        msg = msg[:177] + "..."
    return f"[{ts}] {category} caller={caller} :: {msg}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL,
                        help="Path to critical_errors.jsonl")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help="Lookback window in days")
    parser.add_argument("--json", action="store_true",
                        help="Emit raw JSON entries instead of formatted lines")
    args = parser.parse_args(argv)

    entries = _load_entries(args.journal)
    unresolved = find_unresolved(entries, days=args.days)

    if not unresolved:
        return 0

    if args.json:
        for e in unresolved:
            print(json.dumps(e, ensure_ascii=False))
    else:
        print(f"{len(unresolved)} unresolved critical error(s) in last {args.days} days:")
        print("Journal: " + str(args.journal))
        for e in unresolved:
            print("  " + format_entry(e))
        print()
        print("Surface these to the user before continuing. To resolve, append a "
              "follow-up entry with resolves_ts set to the original ts.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
