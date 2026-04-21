"""CLI: backfill outcomes for data/llm_probability_log.jsonl.

Usage:
    python scripts/backfill_llm_outcomes.py [--max-rows N]

Safe to rerun (idempotent). Rows whose horizon hasn't elapsed are skipped
for a future run. Rows already backfilled are detected via
`(ts, signal, ticker, horizon)` key matching.

Typical schedule: run every hour (matches the hourly price snapshot
cadence). Daily also works — the only cost of rare runs is a staleness
delay before shadow-model accuracy shows up in the daily LLM report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from portfolio.llm_outcome_backfill import backfill  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Stop after N outcomes written (diagnostic runs).",
    )
    parser.add_argument(
        "--min-age-hours", type=int, default=0,
        help="Skip rows whose horizon elapsed fewer than N hours ago "
             "(gives snapshot writers a buffer).",
    )
    args = parser.parse_args(argv)

    stats = backfill(max_rows=args.max_rows, min_age_hours=args.min_age_hours)
    print(
        f"Backfill complete: {stats['written']} written, "
        f"{stats['skipped_already_present']} already-present, "
        f"{stats['skipped_too_recent']} too-recent, "
        f"{stats['skipped_missing_price']} missing-price, "
        f"{stats['skipped_bad_row']} bad-row, "
        f"{stats['processed']} total processed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
