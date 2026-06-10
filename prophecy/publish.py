"""Prophecy step 3 — validate Claude output + publish journal/snapshot (ZERO tokens).

Reads ``raw_<date>.json`` (written by the Claude run), validates each instrument
record, stamps run metadata + ``spot_at_prediction`` (from the prep context),
appends to the append-only ``prediction_journal.jsonl`` and writes the dashboard
snapshot ``latest.json`` with a ``coverage_summary`` (the gap list).

Anti-stale guards (premortem #4): if the raw file is missing, older than the
context (stale), torn, or yields zero valid records, we DO NOT append phantom
journal rows and DO NOT silently overwrite a good snapshot — we mark
``latest.json.stale = true`` and raise a critical error.

Run: ``python -m prophecy.publish [--date YYYY-MM-DD]``
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
from prophecy import config as pcfg
from prophecy.alerts import log_critical
from prophecy.schema import _safe_float, validate_record

logger = logging.getLogger("prophecy.publish")

# 2026-06-11 (audit batch 3): sanity band for the Claude-self-reported
# spot_at_prediction fallback (used only when prep had no live price). Same
# band as outcomes._SPOT_SANITY_BAND — keep in sync.
_SPOT_SANITY_BAND = 0.20


def _validated_fallback_spot(inst: str, raw_spot) -> tuple[float | None, str | None]:
    """Validate a Claude-self-reported spot. Returns (spot, rejection_note).

    The fallback fires only when prep's live_price was None, so the reference
    is a fresh fetch through prep's own price routing. No reference available
    (the usual case for no-feed instruments) -> accept a positive finite spot
    but stamp it claude_self_reported so outcomes re-validates before scoring.
    Band breach -> spot dropped to None (+ rate-limited critical) instead of
    journaling a value that would poison every future accuracy row.
    """
    spot = _safe_float(raw_spot)
    if raw_spot is None:
        return None, None
    if spot is None or spot <= 0:
        return None, f"claude spot_at_prediction not a positive number ({raw_spot!r}) — dropped"
    reference = None
    try:
        from prophecy.prep import _fetch_price
        reference, _src = _fetch_price(inst)
    except Exception as exc:
        logger.warning("reference price fetch failed for %s: %r", inst, exc)
    if reference is not None and reference > 0 and abs(spot / reference - 1.0) > _SPOT_SANITY_BAND:
        log_critical(
            "prophecy_spot_invalid",
            f"{inst}: claude-supplied spot_at_prediction {spot} breaches "
            f"+/-{_SPOT_SANITY_BAND:.0%} band vs live reference {reference} — dropped",
            caller="publish.spot_band",
            context={"instrument": inst, "spot": spot, "reference": reference},
        )
        return None, f"claude spot {spot} breached sanity band vs live {reference} — dropped"
    return spot, None


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _extract_records(raw) -> list:
    """Tolerate a few shapes Claude might emit; return a list of record dicts."""
    if isinstance(raw, dict):
        if isinstance(raw.get("instruments"), dict):
            return [v for v in raw["instruments"].values() if isinstance(v, dict)]
        if isinstance(raw.get("predictions"), list):
            return [v for v in raw["predictions"] if isinstance(v, dict)]
        # bare {INSTR: record} mapping
        vals = [v for v in raw.values() if isinstance(v, dict) and "horizons" in v]
        if vals:
            return vals
    if isinstance(raw, list):
        return [v for v in raw if isinstance(v, dict)]
    return []


def _mark_stale(date: str, reason: str, caller: str) -> int:
    """Flag the existing snapshot stale (keep prior data) + raise critical error."""
    latest = load_json(pcfg.LATEST_FILE, default={}) or {}
    latest["stale"] = True
    latest["stale_reason"] = reason
    latest["stale_marked_at"] = datetime.now(UTC).isoformat()
    atomic_write_json(pcfg.LATEST_FILE, latest)
    log_critical("prophecy_publish", reason, caller=caller, context={"date": date})
    print(f"PUBLISH ABORTED (stale): {reason}")
    return 1


def publish(date: str | None = None) -> int:
    date = date or _today()
    pcfg.ensure_dirs()

    raw_path = pcfg.raw_file(date)
    ctx_path = pcfg.context_file(date)

    if not raw_path.exists():
        return _mark_stale(date, f"raw output missing: {raw_path.name}", "publish.missing_raw")

    # Stale check: raw must be at least as new as the context bundle (i.e. the
    # Claude run actually produced fresh output this cycle).
    if ctx_path.exists() and raw_path.stat().st_mtime < ctx_path.stat().st_mtime:
        return _mark_stale(date, "raw output older than context (Claude likely didn't write)",
                           "publish.stale_raw")

    raw = load_json(raw_path, default=None)
    if raw is None:
        return _mark_stale(date, f"raw output unparseable/torn: {raw_path.name}", "publish.torn_raw")

    records = _extract_records(raw)
    if not records:
        return _mark_stale(date, "raw output had zero records after extraction", "publish.empty_raw")

    ctx = load_json(ctx_path, default={}) or {}
    ctx_instr = ctx.get("instruments", {})
    run_id = f"prophecy-{date}T{datetime.now(UTC).strftime('%H%M%SZ')}"
    model = ctx.get("model") or pcfg.model()
    now_iso = datetime.now(UTC).isoformat()
    # 2026-06-11 (audit batch 3): reconcile against the enabled set. Before
    # this, a hallucinated instrument (e.g. DOGE-USD) was journaled forever and
    # a partial run (3 of 13 instruments) published stale=false with no alert.
    enabled = set(pcfg.enabled_instruments())

    published: dict[str, dict] = {}
    quarantined: list[dict] = []
    total_errors = 0

    for raw_rec in records:
        clean, errors = validate_record(raw_rec)
        total_errors += len(errors)
        if clean is None:
            quarantined.append({"raw": raw_rec, "errors": errors})
            continue

        inst = clean["instrument"]
        if inst not in enabled:
            quarantined.append({"raw": raw_rec, "errors": errors + [
                f"{inst}: not an enabled instrument (hallucinated?) — rejected"]})
            log_critical(
                "prophecy_publish_unknown_instrument",
                f"claude emitted a record for {inst!r}, which is not in the "
                f"enabled instrument set — quarantined, not journaled",
                caller="publish.unknown_instrument",
                context={"date": date, "instrument": inst},
            )
            continue

        ctx_block = ctx_instr.get(inst, {})
        # spot_at_prediction: trust prep's live price; fall back to the
        # record's own claim only after sanity validation (audit batch 3 —
        # an unvalidated claude spot used to crash/poison outcome scoring).
        spot = ctx_block.get("live_price")
        spot_source = ctx_block.get("price_source")
        if spot is None:
            spot, rejection = _validated_fallback_spot(inst, raw_rec.get("spot_at_prediction"))
            spot_source = "claude_self_reported" if spot is not None else spot_source
            if rejection:
                errors.append(rejection)
        clean.update({
            "run_id": run_id,
            "ts": now_iso,
            "date": date,
            "model": clean.get("model") or model,
            "spot_at_prediction": spot,
            "spot_source": spot_source,
        })
        # If prep already flagged a gap, keep it flagged (never clear it here).
        seed = ctx_block.get("coverage_seed") or {}
        if seed.get("needs_work") and not clean["coverage"].get("needs_work"):
            clean["coverage"]["needs_work"] = True
            clean["coverage"].setdefault("note", seed.get("note", ""))
        if errors:
            clean["validation_notes"] = errors[:20]

        atomic_append_jsonl(pcfg.JOURNAL_FILE, clean)
        published[inst] = clean

    if quarantined:
        atomic_write_json(pcfg.QUARANTINE_DIR / f"quarantine_{date}.json", quarantined)

    if not published:
        return _mark_stale(date, f"all {len(records)} records failed validation", "publish.all_invalid")

    # 2026-06-11 (audit batch 3): enabled instruments absent from this run are
    # flagged in the publish record + alerted — partial coverage must never
    # look healthy on the dashboard. The run still publishes (partial data
    # beats none); stale stays false.
    missing = sorted(enabled - set(published))
    if missing:
        log_critical(
            "prophecy_publish_missing_instruments",
            f"run for {date} published {len(published)}/{len(enabled)} enabled "
            f"instruments; missing: {missing}",
            caller="publish.missing_instruments",
            context={"date": date, "missing": missing},
        )

    needing_work = [i for i, r in published.items() if r["coverage"].get("needs_work")]
    latest = {
        "date": date,
        "generated_at": now_iso,
        "run_id": run_id,
        "model": model,
        "stale": False,
        "instruments": published,
        "coverage_summary": {
            "instruments_needing_work": needing_work,
            "count_needing_work": len(needing_work),
            "total_instruments": len(published),
            "missing_instruments": missing,
            "enabled_instruments": len(enabled),
        },
        "quarantined_count": len(quarantined),
        # cost_summary is filled in by cost.py after the run is priced.
        "cost_summary": load_json(pcfg.LATEST_FILE, default={}).get("cost_summary"),
    }
    atomic_write_json(pcfg.LATEST_FILE, latest)

    print(f"published {len(published)} | quarantined {len(quarantined)} | "
          f"missing {len(missing)} -> {missing} | "
          f"needs_work {len(needing_work)} -> {needing_work} | repairs {total_errors}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Prophecy publish (zero tokens)")
    ap.add_argument("--date", default=None)
    args = ap.parse_args(argv)
    return publish(args.date)


if __name__ == "__main__":
    raise SystemExit(main())
