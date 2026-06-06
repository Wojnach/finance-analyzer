"""Prophecy step 4 — score matured predictions vs realized price (ZERO tokens).

For each journalled prediction whose horizon has matured (now >= ts + horizon),
fetch the realized price and score directional hit, target error and Brier on the
up-probability. Idempotent: already-scored (run_id, instrument, horizon) keys are
skipped, so it can run every day and only scores newly-matured horizons.

Outputs:
- ``accuracy.jsonl`` — one append per scored prediction (audit trail),
- ``accuracy.json``  — rolled up per instrument x horizon (n, dir_hit_rate,
  target_mae, brier) for the dashboard.

Run: ``python -m prophecy.outcomes [--max-records N]``
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import UTC, datetime

from portfolio.file_utils import (
    atomic_append_jsonl,
    atomic_write_json,
    load_jsonl,
    load_jsonl_tail,
)

from prophecy import config as pcfg
from prophecy.schema import HORIZON_DELTAS, HORIZONS

logger = logging.getLogger("prophecy.outcomes")

# Realized move within +/- this fraction counts as "flat".
_FLAT_BAND = 0.003


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _realized_direction(spot: float, realized: float) -> str:
    pct = (realized - spot) / spot if spot else 0.0
    if pct > _FLAT_BAND:
        return "up"
    if pct < -_FLAT_BAND:
        return "down"
    return "flat"


def _scored_keys() -> set[str]:
    keys: set[str] = set()
    for row in (load_jsonl(pcfg.ACCURACY_JSONL) or []):
        if isinstance(row, dict) and row.get("key"):
            keys.add(row["key"])
    return keys


def score(max_records: int | None = None) -> int:
    pcfg.ensure_dirs()
    journal = load_jsonl(pcfg.JOURNAL_FILE) or []
    if max_records:
        journal = journal[-max_records:]
    if not journal:
        print("no journal rows to score")
        return 0

    try:
        from portfolio.outcome_tracker import _fetch_historical_price
    except Exception as exc:  # pragma: no cover
        logger.error("cannot import price backfill helper: %r", exc)
        return 1

    already = _scored_keys()
    now = datetime.now(UTC)
    newly_scored = 0
    unscorable = 0

    for rec in journal:
        if not isinstance(rec, dict):
            continue
        inst = rec.get("instrument")
        run_id = rec.get("run_id")
        spot = rec.get("spot_at_prediction")
        pred_ts = _parse_ts(rec.get("ts"))
        if not (inst and run_id and pred_ts) or spot is None:
            continue

        for h, pred in (rec.get("horizons") or {}).items():
            if h not in HORIZON_DELTAS or not isinstance(pred, dict):
                continue
            key = f"{run_id}|{inst}|{h}"
            if key in already:
                continue
            target_dt = pred_ts + HORIZON_DELTAS[h]
            if now < target_dt:
                continue  # not matured yet

            realized = _fetch_historical_price(inst, target_dt.timestamp())
            if realized is None:
                unscorable += 1
                continue

            realized_dir = _realized_direction(float(spot), float(realized))
            predicted_dir = str(pred.get("direction", "")).lower()
            dir_hit = int(predicted_dir == realized_dir)
            target = pred.get("target")
            target_err = abs(float(target) - realized) / realized if (target and realized) else None
            prob_up = pred.get("prob_up")
            brier = (float(prob_up) - (1.0 if realized_dir == "up" else 0.0)) ** 2 if prob_up is not None else None

            row = {
                "key": key,
                "scored_at": now.isoformat(),
                "run_id": run_id,
                "instrument": inst,
                "horizon": h,
                "predicted_dir": predicted_dir,
                "realized_dir": realized_dir,
                "dir_hit": dir_hit,
                "spot_at_prediction": spot,
                "target": target,
                "realized": realized,
                "target_error": target_err,
                "prob_up": prob_up,
                "brier": brier,
                "confidence": pred.get("confidence"),
            }
            atomic_append_jsonl(pcfg.ACCURACY_JSONL, row)
            already.add(key)
            newly_scored += 1

    _rollup()
    print(f"scored {newly_scored} newly-matured | unscorable(no feed) {unscorable}")
    return 0


def _rollup() -> None:
    """Roll accuracy.jsonl into per-instrument x per-horizon summary."""
    rows = load_jsonl_tail(pcfg.ACCURACY_JSONL, max_entries=100_000) or []
    agg: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(
        lambda: {"n": 0, "hits": 0, "target_err_sum": 0.0, "target_err_n": 0,
                 "brier_sum": 0.0, "brier_n": 0}))
    for r in rows:
        if not isinstance(r, dict):
            continue
        inst, h = r.get("instrument"), r.get("horizon")
        if not inst or h not in HORIZONS:
            continue
        cell = agg[inst][h]
        cell["n"] += 1
        cell["hits"] += int(r.get("dir_hit") or 0)
        if r.get("target_error") is not None:
            cell["target_err_sum"] += float(r["target_error"])
            cell["target_err_n"] += 1
        if r.get("brier") is not None:
            cell["brier_sum"] += float(r["brier"])
            cell["brier_n"] += 1

    summary: dict[str, dict] = {}
    for inst, horizons in agg.items():
        summary[inst] = {}
        for h, c in horizons.items():
            summary[inst][h] = {
                "n": c["n"],
                "dir_hit_rate": round(c["hits"] / c["n"], 4) if c["n"] else None,
                "target_mae": round(c["target_err_sum"] / c["target_err_n"], 4) if c["target_err_n"] else None,
                "brier": round(c["brier_sum"] / c["brier_n"], 4) if c["brier_n"] else None,
            }
    atomic_write_json(pcfg.ACCURACY_FILE, {
        "updated_at": datetime.now(UTC).isoformat(),
        "instruments": summary,
    })


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Prophecy outcome scoring (zero tokens)")
    ap.add_argument("--max-records", type=int, default=None)
    args = ap.parse_args(argv)
    return score(args.max_records)


if __name__ == "__main__":
    raise SystemExit(main())
