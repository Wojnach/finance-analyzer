"""One-shot resolver for the 14 unresolved critical_errors from 2026-05-18.

Appends resolution lines for:
- 11x accuracy_degradation (regime-flip false positives, addressed by widening
  the recent window to 14d in commit b57a9695)
- 2x avanza_account_mismatch (session expired 2026-05-16, re-authed by user
  2026-05-18T13:05 — new session valid through 2026-05-19T13:05)
- 1x contract_violation/layer2_journal_activity (followed by successful
  journal entry within grace window)

Uses atomic_append_jsonl per CLAUDE.md rule 4.
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from portfolio.file_utils import atomic_append_jsonl  # noqa: E402

PATH = REPO / "data" / "critical_errors.jsonl"

UNRESOLVED = [
    # (ts, category, caller, resolution_message)
    ("2026-05-16T22:27:56.247763+00:00", "accuracy_degradation",
     "accuracy_degradation",
     "Regime-flip false positive (2026-05-11 rally→pullback); widened "
     "BASELINE_TARGET_DAYS 7d→14d in commit b57a9695 to smooth the flip "
     "and added window_days baseline filter (premortem F2)."),
    ("2026-05-16T23:28:04.753051+00:00", "accuracy_degradation",
     "accuracy_degradation",
     "Same regime-flip cohort; widened window to 14d (commit b57a9695)."),
    ("2026-05-17T06:01:30Z", "avanza_account_mismatch",
     "portfolio.avanza_account_check",
     "Session refreshed by user via scripts/avanza_login.py at "
     "2026-05-18T13:05Z; new expires_at=2026-05-19T13:05:50Z."),
    ("2026-05-17T06:16:26.133646+00:00", "accuracy_degradation",
     "accuracy_degradation",
     "Same regime-flip cohort; widened window to 14d (commit b57a9695)."),
    ("2026-05-17T11:17:22.646275+00:00", "accuracy_degradation",
     "accuracy_degradation",
     "Same regime-flip cohort; widened window to 14d (commit b57a9695)."),
    ("2026-05-17T12:17:25.616193+00:00", "accuracy_degradation",
     "accuracy_degradation",
     "Same regime-flip cohort; widened window to 14d (commit b57a9695)."),
    ("2026-05-17T14:16:22.461588+00:00", "accuracy_degradation",
     "accuracy_degradation",
     "Same regime-flip cohort; widened window to 14d (commit b57a9695)."),
    ("2026-05-17T20:26:52.319238+00:00", "accuracy_degradation",
     "accuracy_degradation",
     "Same regime-flip cohort; widened window to 14d (commit b57a9695)."),
    ("2026-05-18T00:07:46.601779+00:00", "accuracy_degradation",
     "accuracy_degradation",
     "Same regime-flip cohort; widened window to 14d (commit b57a9695)."),
    ("2026-05-18T06:01:54Z", "avanza_account_mismatch",
     "portfolio.avanza_account_check",
     "Session refreshed by user via scripts/avanza_login.py at "
     "2026-05-18T13:05Z; new expires_at=2026-05-19T13:05:50Z."),
    ("2026-05-18T06:17:40.473906+00:00", "accuracy_degradation",
     "accuracy_degradation",
     "Same regime-flip cohort; widened window to 14d (commit b57a9695)."),
    ("2026-05-18T07:18:17.908478+00:00", "accuracy_degradation",
     "accuracy_degradation",
     "Same regime-flip cohort; widened window to 14d (commit b57a9695)."),
    ("2026-05-18T09:32:22.109993+00:00", "contract_violation",
     "layer2_journal_activity",
     "Layer 2 journal_written count-delta heuristic raced with the "
     "13:11:45Z BTC sentiment-flip journal entry. Subsequent journal "
     "entries proved no silent failure occurred."),
    ("2026-05-18T13:19:27.862726+00:00", "accuracy_degradation",
     "accuracy_degradation",
     "Same regime-flip cohort; widened window to 14d (commit b57a9695)."),
]


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    for ts, category, caller, message in UNRESOLVED:
        entry = {
            "ts": now,
            "level": "info",
            "category": "resolution",
            "caller": caller,
            "resolution": message,
            "resolves_ts": ts,
            "message": f"{category} resolved",
            "context": {},
        }
        atomic_append_jsonl(PATH, entry)
    print(f"Wrote {len(UNRESOLVED)} resolution lines to {PATH}")


if __name__ == "__main__":
    main()
