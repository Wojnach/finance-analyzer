"""One-shot backfill for missing daily accuracy snapshots.

Caught 2026-04-28 during the contract-alert root-cause investigation:
``data/accuracy_snapshots.jsonl`` had only 4 entries — Feb 20 (pre-feature),
Apr 19, Apr 20, Apr 21 — despite the snapshot writer being scheduled
daily. The state file said today was already done, so the natural daily
writer had been short-circuiting since Apr 21 → 7 days of missing
baselines, forcing the degradation detector to compare against an
ever-staler snapshot.

This script regenerates the missing daily snapshots by replaying
``signal_accuracy`` and ``consensus_accuracy`` against historical
``signal_log.jsonl`` cuts. It writes only the **recent-window** scopes
that the detector actually compares against (signals_recent,
per_ticker_recent, consensus_recent). Lifetime/cached scopes
(per_ticker, forecast) are skipped — the detector's recent-window
diff doesn't read them, and they require infra that doesn't accept a
historical "now" cleanly.

Usage:

    .venv/Scripts/python.exe scripts/backfill_accuracy_snapshots.py \\
        --start 2026-04-22 --end 2026-04-27 \\
        --signal-log data/signal_log.jsonl \\
        --output-jsonl data/accuracy_snapshots.jsonl \\
        --hour-utc 6

The script appends in chronological order. Pre-existing entries with
the same date are *not* overwritten (idempotent re-runs print a SKIP).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

# Allow running both as a script and from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio.accuracy_stats import (  # noqa: E402
    consensus_accuracy,
    signal_accuracy,
)
from portfolio.accuracy_degradation import _per_ticker_recent  # noqa: E402


def _load_signal_log_entries(path: Path) -> list[dict]:
    """Load signal_log entries from JSONL — returns list[dict]."""
    if not path.exists():
        raise FileNotFoundError(f"Signal log not found: {path}")
    entries: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _filter_entries_by_cutoff(
    entries: list[dict], cutoff: datetime,
) -> list[dict]:
    """Return entries whose ts is at or before ``cutoff`` (UTC)."""
    cutoff_iso = cutoff.isoformat()
    return [e for e in entries if e.get("ts", "") <= cutoff_iso]


def _filter_entries_window(
    entries: list[dict], lower: datetime, upper: datetime,
) -> list[dict]:
    """Return entries with lower < ts <= upper."""
    lo, hi = lower.isoformat(), upper.isoformat()
    return [e for e in entries if lo < e.get("ts", "") <= hi]


def _compact_per_ticker(per_ticker: dict) -> dict:
    """Mirror the production _compact_per_ticker helper output shape.

    The snapshot stores per_ticker_recent as
    {ticker: {signal: {accuracy, total}}} — strip extra fields like
    "correct" if present, keep only accuracy + total.
    """
    out: dict[str, dict[str, dict]] = {}
    for ticker, signals in (per_ticker or {}).items():
        out[ticker] = {}
        for signal_name, stats in (signals or {}).items():
            if not isinstance(stats, dict):
                continue
            out[ticker][signal_name] = {
                "accuracy": stats.get("accuracy", 0.0),
                "total": stats.get("total", 0),
            }
    return out


def build_historical_snapshot(
    *, all_entries: list[dict], target_dt: datetime, days: int = 7,
    forecast_recent_template: dict | None = None,
) -> dict:
    """Build a partial snapshot dict matching the live writer's shape.

    Includes signals (lifetime as of target_dt), signals_recent,
    consensus, consensus_recent, per_ticker_recent.

    ``forecast_recent`` is filled from ``forecast_recent_template`` (a
    snapshot block from the nearest live snapshot) when provided. This
    is a Codex P2 fix (round 1 2026-04-28): without forecast_recent on
    backfilled days, the detector silently skips forecast comparisons
    when one of those days becomes the 7-day baseline. Copying the
    nearest live forecast_recent value avoids the blind spot — at the
    cost of treating forecast as flat across the backfill window. New
    forecast degradation in that 6-day window is unrecoverable from
    JSONL alone.

    Lifetime per_ticker is omitted — the detector's recent-window
    diff doesn't read it.
    """
    cutoff = target_dt
    lower = cutoff - timedelta(days=days)

    historical_all = _filter_entries_by_cutoff(all_entries, cutoff)
    historical_recent = _filter_entries_window(all_entries, lower, cutoff)

    snapshot: dict = {
        "ts": target_dt.isoformat(),
        "_backfilled": True,
        "_backfilled_note": (
            "Recreated 2026-04-28 from signal_log replay; "
            "per_ticker lifetime + forecast omitted."
        ),
    }

    # Per-signal lifetime and recent
    try:
        lifetime = signal_accuracy("1d", entries=historical_all)
        snapshot["signals"] = {
            name: {"accuracy": data["accuracy"], "total": data["total"]}
            for name, data in lifetime.items()
        }
    except Exception as e:
        print(f"  warn: signals lifetime failed: {e}", file=sys.stderr)
        snapshot["signals"] = {}

    try:
        recent = signal_accuracy("1d", entries=historical_recent)
        snapshot["signals_recent"] = {
            name: {"accuracy": data["accuracy"], "total": data["total"]}
            for name, data in recent.items()
        }
    except Exception as e:
        print(f"  warn: signals_recent failed: {e}", file=sys.stderr)
        snapshot["signals_recent"] = {}

    # Per-ticker recent
    try:
        per_ticker_recent = _per_ticker_recent(
            "1d", days=days, entries=historical_recent,
        )
        snapshot["per_ticker_recent"] = _compact_per_ticker(per_ticker_recent)
    except Exception as e:
        print(f"  warn: per_ticker_recent failed: {e}", file=sys.stderr)
        snapshot["per_ticker_recent"] = {}

    # Consensus lifetime + recent
    try:
        snapshot["consensus"] = consensus_accuracy(
            "1d", entries=historical_all,
        )
    except Exception as e:
        print(f"  warn: consensus lifetime failed: {e}", file=sys.stderr)

    try:
        snapshot["consensus_recent"] = consensus_accuracy(
            "1d", entries=historical_recent,
        )
    except Exception as e:
        print(f"  warn: consensus_recent failed: {e}", file=sys.stderr)

    if forecast_recent_template:
        snapshot["forecast_recent"] = dict(forecast_recent_template)

    return snapshot


def _existing_dates(jsonl_path: Path) -> set[date]:
    """Set of dates already present in the snapshots JSONL."""
    if not jsonl_path.exists():
        return set()
    dates: set[date] = set()
    with jsonl_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ts = obj.get("ts", "")
                dt = datetime.fromisoformat(ts)
                dates.add(dt.date())
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
    return dates


def _load_existing_snapshots(jsonl_path: Path) -> list[dict]:
    """Load all existing snapshot dicts from JSONL (preserves file order)."""
    if not jsonl_path.exists():
        return []
    out: list[dict] = []
    with jsonl_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _newest_forecast_recent(snapshots: list[dict]) -> dict | None:
    """Return the most recent non-empty forecast_recent block.

    Used as a placeholder for backfilled days that lack forecast data —
    keeps the detector's forecast scope from going blind on those days.
    """
    for snap in reversed(snapshots):
        fr = snap.get("forecast_recent")
        if isinstance(fr, dict) and fr:
            return fr
    return None


def _sort_jsonl_chronologically(jsonl_path: Path) -> int:
    """Re-write ``jsonl_path`` with all entries sorted by ts (oldest first).

    Codex P2 fix (round 1 2026-04-28): atomic_append_jsonl writes new
    entries at the end regardless of timestamp. If the file already
    contains a newer row (e.g. today's manual snapshot), append-only
    backfill leaves the older snapshots after the newer one; the daily
    summary path uses ``snapshots[-1]`` as "latest" and reads stale
    data. Sorting once at the end of backfill keeps file order
    monotonic in time.

    Returns the number of entries written.
    """
    snaps = _load_existing_snapshots(jsonl_path)

    def _key(snap: dict) -> str:
        return snap.get("ts", "")

    snaps.sort(key=_key)

    from portfolio.file_utils import atomic_write_jsonl
    atomic_write_jsonl(jsonl_path, snaps)
    return len(snaps)


def _daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur = cur + timedelta(days=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True,
                        help="First missing date to backfill (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, required=True,
                        help="Last missing date to backfill (YYYY-MM-DD)")
    parser.add_argument("--signal-log", type=Path,
                        default=Path("data/signal_log.jsonl"))
    parser.add_argument("--output-jsonl", type=Path,
                        default=Path("data/accuracy_snapshots.jsonl"))
    parser.add_argument("--hour-utc", type=int, default=6,
                        help="Hour-of-day to stamp on backfilled snapshots")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print snapshot summaries; don't write")
    args = parser.parse_args(argv)

    print(f"Loading signal_log from {args.signal_log} ...", file=sys.stderr)
    entries = _load_signal_log_entries(args.signal_log)
    print(f"  loaded {len(entries):,} entries", file=sys.stderr)

    existing_snaps = _load_existing_snapshots(args.output_jsonl)
    existing = _existing_dates(args.output_jsonl)
    print(f"Existing snapshot dates in {args.output_jsonl}: "
          f"{sorted(existing)[-5:]}", file=sys.stderr)
    forecast_template = _newest_forecast_recent(existing_snaps)
    if forecast_template:
        print(
            f"  forecast_recent template: {len(forecast_template)} model(s) "
            f"(copied from nearest live snapshot)", file=sys.stderr,
        )

    from portfolio.file_utils import atomic_append_jsonl

    written = 0
    for d in _daterange(args.start, args.end):
        if d in existing:
            print(f"  SKIP {d.isoformat()} — snapshot already present",
                  file=sys.stderr)
            continue

        target_dt = datetime.combine(
            d, time(hour=args.hour_utc), tzinfo=UTC,
        )
        print(f"  building snapshot for {target_dt.isoformat()}",
              file=sys.stderr)
        snap = build_historical_snapshot(
            all_entries=entries, target_dt=target_dt,
            forecast_recent_template=forecast_template,
        )

        n_signals = len(snap.get("signals_recent", {}))
        sentiment = snap.get("signals_recent", {}).get("sentiment", {})
        consensus_recent = snap.get("consensus_recent", {})
        print(
            f"    signals_recent={n_signals} "
            f"sentiment_recent="
            f"{sentiment.get('accuracy', 0)*100:.1f}%/N={sentiment.get('total', 0)} "
            f"consensus_recent="
            f"{consensus_recent.get('accuracy', 0)*100:.1f}%/"
            f"N={consensus_recent.get('total', 0)}",
            file=sys.stderr,
        )

        if not args.dry_run:
            atomic_append_jsonl(args.output_jsonl, snap)
            written += 1

    if args.dry_run:
        print(f"DRY RUN: would have written {len(list(_daterange(args.start, args.end))) - len(existing & set(_daterange(args.start, args.end)))} snapshots",
              file=sys.stderr)
    else:
        print(f"Wrote {written} new snapshots to {args.output_jsonl}",
              file=sys.stderr)
        if written > 0:
            total = _sort_jsonl_chronologically(args.output_jsonl)
            print(f"Sorted {total} entries chronologically", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
