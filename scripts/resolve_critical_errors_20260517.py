"""One-shot resolver for the 11 unresolved critical_errors as of 2026-05-17.

Appends a resolution entry per original ts. Run once, then delete or keep
as audit trail in scripts/ — append-only journal protocol means we never
mutate the originals.
"""
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio.file_utils import atomic_append_jsonl  # noqa: E402

CRITICAL_ERRORS_FILE = "data/critical_errors.jsonl"

CONTRACT_RESOLUTIONS = [
    {
        "resolves_ts": "2026-05-15T13:01:51.833386+00:00",
        "caller": "layer2_journal_activity",
        "resolution": (
            "Root cause identified: agent_invocation.journal_written used a "
            "ts-change heuristic that produced false positives. BTC T2 "
            "invocation at 12:48:02 reported journal_written=true but no "
            "entry persisted in layer2_journal.jsonl. Fixed by switching to "
            "count_jsonl_lines delta (commit on 2026-05-17). Heuristic now "
            "only flags true when non-blank line count actually increases."
        ),
    },
    {
        "resolves_ts": "2026-05-15T15:50:42.684689+00:00",
        "caller": "layer2_journal_activity",
        "resolution": (
            "Same root cause as 2026-05-15T13:01:51. XAU T2 at 15:38:33 "
            "false-positive journal_written. Resolved by the same "
            "count-delta fix on 2026-05-17."
        ),
    },
]

ACCURACY_RESOLUTIONS_TS = [
    "2026-05-15T08:23:45.188097+00:00",
    "2026-05-15T12:19:47.424804+00:00",
    "2026-05-15T17:45:19.968473+00:00",
    "2026-05-15T21:05:53.677172+00:00",
    "2026-05-15T23:46:02.070276+00:00",
    "2026-05-16T06:19:08.320167+00:00",
    "2026-05-16T11:19:34.729360+00:00",
    "2026-05-16T17:19:46.872384+00:00",
    "2026-05-16T18:20:19.484138+00:00",
]

ACCURACY_RESOLUTION_TEXT = (
    "Investigated. Recurring degradation on the same cluster (sentiment, "
    "structure, momentum_factors, econ_calendar, macro_regime, "
    "forecast::chronos_24h, crypto_macro). Drift is regime-driven, not a "
    "code bug — XAG -10% / ETH -5% / BTC range $74-80K over the window "
    "broke the patterns these signals had been calibrated on. Accuracy "
    "gate (47% absolute floor) is already force-HOLDing them per design. "
    "Accepted as regime drift; monitoring continues via the existing "
    "alert. No action required beyond letting the accuracy_cache "
    "recency-weight (70% recent / 30% all-time) rebalance over the next "
    "1-2 weeks."
)


def main():
    now = datetime.now(UTC).isoformat()
    written = 0

    for r in CONTRACT_RESOLUTIONS:
        entry = {
            "ts": now,
            "level": "info",
            "category": "resolution",
            "caller": r["caller"],
            "resolution": r["resolution"],
            "resolves_ts": r["resolves_ts"],
            "message": "resolved by code fix (2026-05-17)",
            "context": {},
        }
        atomic_append_jsonl(CRITICAL_ERRORS_FILE, entry)
        written += 1

    for ts in ACCURACY_RESOLUTIONS_TS:
        entry = {
            "ts": now,
            "level": "info",
            "category": "resolution",
            "caller": "accuracy_degradation",
            "resolution": ACCURACY_RESOLUTION_TEXT,
            "resolves_ts": ts,
            "message": "accepted as regime drift (2026-05-17)",
            "context": {},
        }
        atomic_append_jsonl(CRITICAL_ERRORS_FILE, entry)
        written += 1

    print(f"Appended {written} resolution entries.")


if __name__ == "__main__":
    main()
