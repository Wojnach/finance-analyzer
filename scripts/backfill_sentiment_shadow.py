"""CLI: backfill outcomes for data/sentiment_ab_log.jsonl — shadow sentiment models.

This is what makes FinGPT / FinBERT / CryptoBERT / Trading-Hero-LLM accuracy
versus actual market outcomes computable. Without it, the A/B log captures
verdicts but we never learn whether the shadow models were right.

Usage:
    python scripts/backfill_sentiment_shadow.py [--horizon 1d] [--max-rows N]

Safe to rerun — idempotent per `(ts, ticker, model, horizon)` key.

Recommended schedule: daily for horizon=1d. Rerun with `--horizon 3d`
weekly to accumulate longer-horizon evidence.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from portfolio.sentiment_shadow_backfill import (  # noqa: E402
    backfill,
    compute_model_accuracy,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--horizon", default="1d",
        choices=("3h", "4h", "12h", "1d", "3d", "5d", "10d"),
        help="Horizon to evaluate outcomes at.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Stop after N outcomes written (diagnostic runs).",
    )
    parser.add_argument(
        "--min-age-hours", type=int, default=0,
        help="Skip rows whose horizon elapsed fewer than N hours ago "
             "(buffer for snapshot-writer race).",
    )
    parser.add_argument(
        "--show-accuracy", action="store_true",
        help="After backfill, print per-model accuracy summary.",
    )
    args = parser.parse_args(argv)

    stats = backfill(
        horizon=args.horizon,
        max_rows=args.max_rows,
        min_age_hours=args.min_age_hours,
    )
    print(
        f"Sentiment shadow backfill ({args.horizon}): "
        f"{stats['outcomes_written']} written, "
        f"{stats['skipped_already_present']} already-present, "
        f"{stats['skipped_too_recent']} too-recent, "
        f"{stats['skipped_missing_price']} missing-price, "
        f"{stats['skipped_bad_row']} bad-row, "
        f"{stats['rows_read']} A/B rows read."
    )

    if args.show_accuracy:
        summary = compute_model_accuracy(horizon=args.horizon, days=30)
        if not summary:
            print("\n(no outcomes yet — run backfill first)")
            return 0
        print("\nPer-model accuracy (30d):")
        for model, s in sorted(summary.items()):
            acc_s = f"{s['accuracy']*100:.1f}%" if s["accuracy"] is not None else "—"
            agr_s = (
                f"{s['agreement_with_primary']*100:.1f}%"
                if s["agreement_with_primary"] is not None else "—"
            )
            print(f"  [{s['kind']:>7}] {model:<35} "
                  f"acc {acc_s:>7}   agr {agr_s:>7}   n={s['samples']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
