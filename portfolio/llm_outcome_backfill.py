"""Outcome backfill for `data/llm_probability_log.jsonl`.

Reads the probability log, pairs each row with a realized price return at
its target horizon, classifies the realized move into BUY / HOLD / SELL,
and writes the outcome into a companion `data/llm_probability_outcomes.jsonl`
keyed on `(ts, signal, ticker, horizon)`. The original probability log stays
append-only and untouched.

Why a separate file instead of mutating rows
--------------------------------------------
Appending to the existing log is atomic; modifying earlier rows requires a
full rewrite, which invites lost writes on crash and invalidates any tailers.
The outcome file is write-append-only too, and the join happens in
`llm_calibration.compute_metrics()` at read time.

Rerun semantics
---------------
Safe to rerun — the script tracks which (ts, signal, ticker, horizon) keys
it has already written in the outcomes file and skips duplicates. This
means the job can be scheduled hourly / daily without needing state.

Source of truth for prices
--------------------------
Reuses `portfolio.forecast_accuracy._lookup_price_at_time` which reads the
hourly price snapshot jsonl — the same path the existing outcome_tracker
uses. Keeps us on a single retrieval codepath.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl
from portfolio.forecast_accuracy import _lookup_price_at_time
from portfolio.llm_calibration import outcome_from_return

logger = logging.getLogger("portfolio.llm_outcome_backfill")

_BASE_DIR = Path(__file__).resolve().parent.parent
_PROB_LOG = _BASE_DIR / "data" / "llm_probability_log.jsonl"
_OUTCOMES = _BASE_DIR / "data" / "llm_probability_outcomes.jsonl"

_HORIZON_HOURS = {
    "3h": 3,
    "4h": 4,
    "12h": 12,
    "1d": 24,
    "3d": 72,
    "5d": 120,
    "10d": 240,
}


def _row_key(row: dict) -> tuple[str, str, str, str]:
    """Compute the dedup key for a probability/outcome row.

    2026-04-28: horizon is normalized via `or "1d"` to match the write
    path at line 144 (`horizon = row.get("horizon") or "1d"`). The earlier
    implementation used dict.get(default), which only returns the default
    when the key is ABSENT — not when its value is explicitly null. Early
    pre-ed13e608 production rows had `horizon: null`, so on the read side
    their key was (..., None) but the write side stored (..., "1d"). The
    next backfill run never matched, re-wrote the same outcome, and we
    accumulated 91 duplicates of every null-horizon row over 7 days.
    Empty-string fallback for the other fields stays — `signal`, `ticker`,
    `ts` cannot be null in any historical row (verified across 15k rows).
    """
    return (
        row.get("ts", ""),
        row.get("signal") or "",
        row.get("ticker") or "",
        row.get("horizon") or "1d",
    )


def _load_existing_keys(outcomes_path: Path) -> set[tuple[str, str, str, str]]:
    keys: set[tuple[str, str, str, str]] = set()
    if not outcomes_path.exists():
        return keys
    for line in outcomes_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            keys.add(_row_key(row))
        except json.JSONDecodeError:
            continue
    return keys


def backfill(
    *,
    log_path: Path | str | None = None,
    outcomes_path: Path | str | None = None,
    snapshot_path: Path | str | None = None,
    min_age_hours: int = 0,
    max_rows: int | None = None,
) -> dict:
    """Read the probability log and append outcomes for every row whose
    target horizon has already passed and has a price snapshot available.

    Args:
      log_path: override for the input jsonl (tests).
      outcomes_path: override for the output jsonl.
      snapshot_path: override for the price snapshot source.
      min_age_hours: additional buffer before we trust a price lookup.
        Default 0 — we rely on the horizon already elapsing. Set to a small
        positive number to avoid racing with the snapshot writer.
      max_rows: stop after N new outcomes written (diagnostic runs).

    Returns a stats dict:
      `{processed, written, skipped_already_present, skipped_too_recent,
        skipped_missing_price, skipped_bad_row}`.
    """
    log_path = Path(log_path) if log_path else _PROB_LOG
    outcomes_path = Path(outcomes_path) if outcomes_path else _OUTCOMES

    stats = {
        "processed": 0,
        "written": 0,
        "skipped_already_present": 0,
        "skipped_too_recent": 0,
        "skipped_missing_price": 0,
        "skipped_bad_row": 0,
    }

    if not log_path.exists():
        logger.info("No probability log at %s; nothing to backfill", log_path)
        return stats

    existing = _load_existing_keys(outcomes_path)
    now = datetime.now(UTC)

    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            stats["skipped_bad_row"] += 1
            continue

        stats["processed"] += 1
        key = _row_key(row)
        if key in existing:
            stats["skipped_already_present"] += 1
            continue

        # `row.get("horizon", "1d")` doesn't cover rows where horizon exists
        # but is explicitly null. Early production rows (pre-horizon-default
        # fix ed13e608) had `"horizon": null` — still backfill-eligible at
        # the argmax-accuracy 1d default.
        horizon = row.get("horizon") or "1d"
        hours = _HORIZON_HOURS.get(horizon)
        if hours is None:
            stats["skipped_bad_row"] += 1
            continue

        try:
            entry_time = datetime.fromisoformat(row["ts"])
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=UTC)
        except (KeyError, TypeError, ValueError):
            stats["skipped_bad_row"] += 1
            continue

        target_time = entry_time + timedelta(hours=hours)
        if now < target_time + timedelta(hours=min_age_hours):
            stats["skipped_too_recent"] += 1
            continue

        ticker = row.get("ticker", "")
        if not ticker:
            stats["skipped_bad_row"] += 1
            continue

        entry_price = _lookup_price_at_time(ticker, entry_time, snapshot_file=snapshot_path)
        target_price = _lookup_price_at_time(ticker, target_time, snapshot_file=snapshot_path)
        if entry_price is None or target_price is None or entry_price == 0:
            stats["skipped_missing_price"] += 1
            continue

        pct = (target_price - entry_price) / entry_price
        outcome = outcome_from_return(pct)

        outcome_row = {
            "ts": row["ts"],
            "signal": row.get("signal"),
            "ticker": ticker,
            "horizon": horizon,
            "entry_price": entry_price,
            "target_price": target_price,
            "pct_change": pct,
            "outcome": outcome,
            "backfilled_at": now.isoformat(),
        }
        try:
            atomic_append_jsonl(outcomes_path, outcome_row)
        except Exception as e:
            logger.warning("outcome append failed: %s", e)
            continue

        existing.add(key)
        stats["written"] += 1
        if max_rows is not None and stats["written"] >= max_rows:
            break

    return stats


def outcome_lookup(outcomes_path: Path | str | None = None):
    """Return a callable `(ts, ticker, horizon) → Optional[str]` backed by
    the outcomes file. Suitable for passing to
    `llm_calibration.compute_metrics`.

    The lookup reads the outcomes file once on first call and caches the
    dict in a closure — re-call returns the stale cache unless you pass a
    fresh `outcomes_path`.
    """
    path = Path(outcomes_path) if outcomes_path else _OUTCOMES
    cache: dict[tuple[str, str, str], str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                k = (row.get("ts", ""), row.get("ticker", ""), row.get("horizon", ""))
                outcome = row.get("outcome")
                if k[0] and k[1] and outcome:
                    cache[k] = outcome
            except json.JSONDecodeError:
                continue

    def _lookup(ts: str, ticker: str, horizon: str) -> str | None:
        return cache.get((ts, ticker, horizon))

    return _lookup
