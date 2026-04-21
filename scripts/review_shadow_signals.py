"""CLI: flag shadow signals older than N days without a resolution.

Usage:
    python scripts/review_shadow_signals.py [--stale-days N] [--seed]

Options:
    --stale-days N   Threshold for "stale" classification (default 30).
    --seed           Run seed_defaults() before reporting. Safe to re-run —
                     never overwrites existing entries.

Exit code:
    0  — no stale shadows (or only resolved ones).
    1  — one or more shadows are stale. Useful in CI/scheduled-task checks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/review_shadow_signals.py` from the repo root
# without requiring an install. Matches the pattern used by other scripts/ files.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from portfolio.shadow_registry import (  # noqa: E402
    days_in_shadow,
    load_registry,
    seed_defaults,
    stale_shadows,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stale-days", type=int, default=30)
    parser.add_argument(
        "--seed", action="store_true",
        help="Run seed_defaults() before reporting (idempotent).",
    )
    args = parser.parse_args(argv)

    if args.seed:
        seed_defaults()

    reg = load_registry()
    if not reg["shadows"]:
        print("(registry empty — run with --seed to populate defaults)")
        return 0

    stale = stale_shadows(stale_days=args.stale_days)
    total = len(reg["shadows"])
    resolved = sum(
        1 for e in reg["shadows"].values() if e.get("status") != "shadow"
    )

    print(f"Shadow registry: {total} total, {resolved} resolved, "
          f"{len(stale)} stale (>{args.stale_days}d in shadow).")
    print()

    for sig, entry in sorted(reg["shadows"].items()):
        status = entry.get("status", "?")
        notes = entry.get("notes", "")
        if status == "shadow":
            age = days_in_shadow(sig)
            if age is not None:
                marker = "STALE " if age >= args.stale_days else "active"
                print(f"  [{status:>9}] {sig:<22} {marker}  "
                      f"{age:>5.1f}d   {notes[:60]}")
            else:
                print(f"  [{status:>9}] {sig:<22}   ?      "
                      f"?d   {notes[:60]}")
        else:
            print(f"  [{status:>9}] {sig:<22}                    "
                  f"{notes[:60]}")

    return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
