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
        parsed = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    # 2026-05-28: coerce naive timestamps to UTC. A hand-authored or
    # agent-appended resolution line that omits the tz offset (e.g. a bare
    # datetime.now().isoformat()) yields a naive datetime; comparing it against
    # the aware `cutoff` in find_unresolved raises "can't compare offset-naive
    # and offset-aware datetimes", which — with no try/except around the
    # comparison — crashes the entire session-start check and hides ALL
    # unresolved critical errors. session_start_bottle.py already does this.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


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


def _auto_resolve_stale_categories(
    entries: list[dict], *, stale_days: int = 3, now: datetime | None = None,
) -> set[str]:
    """Return set of category names that haven't fired in stale_days.

    A category is auto-resolvable when its most recent critical entry is
    older than `stale_days` AND at least one resolution or info entry for
    that category exists after the last critical fire.
    """
    now = now or datetime.now(UTC)
    stale_cutoff = now - timedelta(days=stale_days)

    last_critical_by_cat: dict[str, datetime] = {}
    latest_fix_by_cat: dict[str, datetime] = {}

    for e in entries:
        cat = e.get("category", "")
        if not cat:
            continue
        parsed = _parse_ts(e.get("ts", ""))
        if parsed is None:
            continue
        if e.get("level") == "critical" and e.get("resolution") is None:
            existing = last_critical_by_cat.get(cat)
            if existing is None or parsed > existing:
                last_critical_by_cat[cat] = parsed
        if e.get("resolution") is not None or e.get("level") == "info":
            existing_fix = latest_fix_by_cat.get(cat)
            if existing_fix is None or parsed > existing_fix:
                latest_fix_by_cat[cat] = parsed

    stale_cats = set()
    for cat, last_ts in last_critical_by_cat.items():
        fix_ts = latest_fix_by_cat.get(cat)
        if last_ts < stale_cutoff and fix_ts is not None and fix_ts > last_ts:
            stale_cats.add(cat)
    return stale_cats


def find_unresolved(entries: list[dict], *, days: int, now: datetime | None = None) -> list[dict]:
    """Return entries with resolution=None from the last `days` days.

    A later entry with ``resolves_ts`` pointing at an earlier entry's ``ts``
    retroactively resolves that earlier entry. Categories that haven't fired
    in 3+ days AND have a post-dated resolution/info entry are auto-resolved.
    """
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(days=days)

    resolved_ts: set[str] = set()
    for e in entries:
        rts = e.get("resolves_ts")
        if rts:
            resolved_ts.add(rts)

    stale_cats = _auto_resolve_stale_categories(entries, now=now)

    unresolved = []
    for e in entries:
        if e.get("level") != "critical":
            continue
        if e.get("resolution") is not None:
            continue
        if e.get("ts") in resolved_ts:
            continue
        if e.get("category", "") in stale_cats:
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
    # Force UTF-8 stdout: violation messages contain `→` (U+2192) which Windows'
    # default cp1252 codec can't encode. Without this, the script crashes mid-print
    # and the user sees only the count line, not the entries — meaning unresolved
    # accuracy_degradation rows were silently invisible to the STARTUP CHECK
    # documented in CLAUDE.md (caught 2026-04-28 during contract-alert diagnosis).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
