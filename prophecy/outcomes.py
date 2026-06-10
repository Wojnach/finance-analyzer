"""Prophecy step 4 — score matured predictions vs realized price (ZERO tokens).

For each journalled prediction whose horizon has matured (now >= ts + horizon),
fetch the realized price and score directional hit, target error and Brier on the
up-probability. Idempotent: already-scored (date, instrument, horizon) keys are
skipped, so it can run every day and only scores newly-matured horizons.

Scoring price sources (2026-06-11, audit batch 3 — previously only the 5
outcome_tracker-mapped tickers were scoreable and the other 8 enabled
instruments were silently skipped forever):

- Binance-mapped 24/7 instruments (BTC/ETH spot, XAU/XAG FAPI): exact-timestamp
  1m kline via portfolio.outcome_tracker._fetch_historical_price.
- Daily-bar instruments (MSTR, CL=F, BZ=F): yfinance daily close of the FIRST
  session ON OR AFTER the horizon target date ("next session close", matching
  the schema's 1d semantics). The previous last-close-BEFORE-target resolution
  scored Friday-prediction weekend horizons against the same Friday close — a
  ~4h window graded as a 1-2 day move. If the target session hasn't printed
  yet, the horizon stays pending and is retried next run.
- Everything else (warrants, Tier-2 Swedish equities): no historical feed.
  Must be flagged ``scoreable: false`` in prophecy_config.json — enabled
  instruments that are neither scoreable nor flagged raise a (rate-limited)
  critical instead of silence.

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
from datetime import UTC, datetime, timedelta

from portfolio.file_utils import (
    atomic_append_jsonl,
    atomic_write_json,
    load_jsonl,
    load_jsonl_tail,
)
from prophecy import config as pcfg
from prophecy.alerts import log_critical
from prophecy.schema import HORIZON_DELTAS, HORIZONS

logger = logging.getLogger("prophecy.outcomes")

# Realized move within +/- this fraction counts as "flat".
_FLAT_BAND = 0.003

# 2026-06-11 (audit batch 3): sanity band for Claude-self-reported
# spot_at_prediction. A spot more than +/-20% away from the scoring source's
# own price at prediction time is rejected (record skipped + critical) instead
# of poisoning dir_hit/Brier for every horizon of that record.
_SPOT_SANITY_BAND = 0.20

# Instruments scored from yfinance DAILY bars (non-24/7 markets). MSTR moved
# here from the outcome_tracker YF path to fix the weekend-window bug; oil
# added to close the 8/13-unscoreable gap (prep already prices CL=F/BZ=F via
# price_source, so predictions carry a real spot).
_DAILY_BAR_YF: dict[str, str] = {"MSTR": "MSTR", "CL=F": "CL=F", "BZ=F": "BZ=F"}


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


def _scoring_source(inst: str) -> str | None:
    """Which historical price path can score this instrument (None = none)."""
    try:
        from portfolio.tickers import BINANCE_FAPI_MAP, BINANCE_SPOT_MAP
    except Exception:  # pragma: no cover - import guard
        return None
    if inst in BINANCE_FAPI_MAP or inst in BINANCE_SPOT_MAP:
        return "binance"
    if inst in _DAILY_BAR_YF:
        return "daily_bar"
    return None


def _fetch_daily_bar_close(inst: str, target_dt: datetime) -> tuple[float | None, str | None]:
    """Close + ISO date of the first daily bar ON OR AFTER ``target_dt``.

    Returns (None, None) when that session hasn't printed yet (weekend/holiday
    target with the next session still in the future) — callers treat this as
    pending, not unscorable. Window is target-1d..target+8d: 8 calendar days
    always contain at least one NYSE/NYMEX session.
    """
    import yfinance as yf

    from portfolio.outcome_tracker import _yfinance_limiter

    _yfinance_limiter.wait()
    start = (target_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (target_dt + timedelta(days=8)).strftime("%Y-%m-%d")
    # auto_adjust=False for the same split-only-adjustment reasons documented
    # in portfolio/outcome_tracker.py (2026-05-28).
    h = yf.Ticker(_DAILY_BAR_YF[inst]).history(start=start, end=end, auto_adjust=False)
    if h.empty:
        return None, None
    # Compare dates in the bars' own exchange timezone (outcome_tracker
    # 2026-05-28 lesson: UTC-date vs NY-date off-by-one in early UTC hours).
    idx_tz = getattr(h.index, "tz", None)
    target_local = target_dt.astimezone(idx_tz) if idx_tz is not None else target_dt
    candidates = h[h.index.date >= target_local.date()]
    if candidates.empty:
        return None, None
    return float(candidates["Close"].iloc[0]), candidates.index[0].date().isoformat()


def _safe_positive_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or abs(out) == float("inf") or out <= 0:
        return None
    return out


def _untrusted_spot_source(source) -> bool:
    """True when spot_at_prediction did NOT come from a live prep price feed.

    publish.py stamps "claude_self_reported" on the fallback path since
    2026-06-11; rows published before that carry None/"no_feed"/"*_error" in
    the same situation.
    """
    if not source or not isinstance(source, str):
        return True
    return source in ("no_feed", "claude_self_reported") or source.endswith("_error")


def _validate_spot(inst: str, source: str | None, spot: float,
                   pred_ts: datetime, fetch_historical) -> bool:
    """Sanity-band check for untrusted spots vs the scoring source's own price
    at prediction time. True = usable. False = rejected (caller skips record).
    """
    reference = None
    try:
        if source == "binance":
            reference = fetch_historical(inst, pred_ts.timestamp())
        elif source == "daily_bar":
            reference, _ = _fetch_daily_bar_close(inst, pred_ts)
    except Exception as exc:
        logger.warning("spot reference fetch failed for %s: %r", inst, exc)
    if reference is None or reference <= 0:
        return True  # no reference available — cannot validate, proceed
    if abs(spot / float(reference) - 1.0) > _SPOT_SANITY_BAND:
        log_critical(
            "prophecy_spot_invalid",
            f"{inst}: spot_at_prediction {spot} breaches +/-{_SPOT_SANITY_BAND:.0%} "
            f"band vs scoring-source price {reference} at prediction time — record skipped",
            caller="outcomes.spot_band",
            context={"instrument": inst, "spot": spot, "reference": reference,
                     "pred_ts": pred_ts.isoformat()},
        )
        return False
    return True


def score(max_records: int | None = None) -> int:
    pcfg.ensure_dirs()
    cfg = pcfg.load_config()

    # --- coverage reconcile (audit batch 3, 2026-06-11; MUST stay ahead of
    # the per-record loop — premortem hook 13 ordering) -----------------------
    # Every enabled instrument must either have a scoring source or be
    # explicitly flagged scoreable=false in prophecy_config.json. Flagged ones
    # get ONE warning per run (not per record, not silence); unflagged gaps
    # are a config/coverage bug and raise a rate-limited critical.
    enabled = pcfg.enabled_instruments(cfg)
    flagged_unscoreable = set(pcfg.unscoreable_instruments(cfg))
    coverage_gaps = [i for i in enabled
                     if _scoring_source(i) is None and i not in flagged_unscoreable]
    if coverage_gaps:
        log_critical(
            "prophecy_scoring_coverage",
            f"enabled instruments with no scoring price source and no "
            f"scoreable=false flag: {coverage_gaps} — their accuracy will "
            f"silently never exist until mapped or flagged",
            caller="outcomes.coverage_reconcile",
            context={"coverage_gaps": coverage_gaps},
        )
    if flagged_unscoreable:
        logger.warning(
            "unscoreable by config (scoreable=false, no historical feed): %s — "
            "predictions journaled but never outcome-scored",
            sorted(flagged_unscoreable))

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
    pending_bar = 0
    skipped_flagged = 0
    rejected_spot = 0
    record_errors = 0

    for rec in journal:
        # Per-record isolation (audit batch 3): one poisoned journal row used
        # to abort the whole loop at the same row on every future run. Now it
        # is alerted (rate-limited) and skipped.
        try:
            if not isinstance(rec, dict):
                continue
            inst = rec.get("instrument")
            run_id = rec.get("run_id")
            spot = rec.get("spot_at_prediction")
            pred_ts = _parse_ts(rec.get("ts"))
            if not (inst and run_id and pred_ts) or spot is None:
                continue

            if inst in flagged_unscoreable:
                skipped_flagged += 1
                continue
            source = _scoring_source(inst)
            if source is None:
                # Not enabled-and-unflagged (that already raised above) — a
                # historic/hallucinated journal instrument. Count, don't spam.
                unscorable += 1
                continue

            spot_f = _safe_positive_float(spot)
            if spot_f is None:
                rejected_spot += 1
                log_critical(
                    "prophecy_spot_invalid",
                    f"{inst}: spot_at_prediction not a positive finite number "
                    f"({spot!r}) — record skipped",
                    caller="outcomes.spot_invalid",
                    context={"instrument": inst, "run_id": run_id, "spot": repr(spot)},
                )
                continue
            if (_untrusted_spot_source(rec.get("spot_source"))
                    and not _validate_spot(inst, source, spot_f, pred_ts,
                                           _fetch_historical_price)):
                rejected_spot += 1
                continue

            # Dedup identity is (date, instrument, horizon) — STABLE across publish
            # re-runs. run_id embeds a per-invocation timestamp, so a retry/re-run of
            # publish appends fresh journal rows with a new run_id for the SAME
            # prediction; keying on run_id would score both and inflate the rollup n
            # (review P1). Keying on date scores each (date, inst, horizon) once.
            day = rec.get("date") or pred_ts.strftime("%Y-%m-%d")
            for h, pred in (rec.get("horizons") or {}).items():
                if h not in HORIZON_DELTAS or not isinstance(pred, dict):
                    continue
                key = f"{day}|{inst}|{h}"
                if key in already:
                    continue
                target_dt = pred_ts + HORIZON_DELTAS[h]
                if now < target_dt:
                    continue  # not matured yet

                if source == "binance":
                    realized = _fetch_historical_price(inst, target_dt.timestamp())
                    realized_bar_ts = target_dt.isoformat()
                    if realized is None:
                        unscorable += 1
                        continue
                else:  # daily_bar — next-session-close semantics (weekend fix)
                    realized, realized_bar_ts = _fetch_daily_bar_close(inst, target_dt)
                    if realized is None:
                        pending_bar += 1  # session not printed yet; retry next run
                        continue

                realized_dir = _realized_direction(spot_f, float(realized))
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
                    "spot_at_prediction": spot_f,
                    "target": target,
                    "realized": realized,
                    # Auditability (audit batch 3): the bar actually used —
                    # for daily-bar instruments this can be later than the
                    # nominal horizon date (next session close).
                    "realized_bar_ts": realized_bar_ts,
                    "target_error": target_err,
                    "prob_up": prob_up,
                    "brier": brier,
                    "confidence": pred.get("confidence"),
                }
                atomic_append_jsonl(pcfg.ACCURACY_JSONL, row)
                already.add(key)
                newly_scored += 1
        except Exception as exc:
            record_errors += 1
            log_critical(
                "prophecy_scoring_error",
                f"scoring crashed on a journal row (instrument="
                f"{rec.get('instrument') if isinstance(rec, dict) else '?'}): {exc!r} "
                f"— row skipped, loop continues",
                caller="outcomes.record_error",
                context={"error": repr(exc)},
            )
            continue

    _rollup()
    print(f"scored {newly_scored} newly-matured | unscorable(no feed) {unscorable} | "
          f"pending(bar not printed) {pending_bar} | skipped(scoreable=false) {skipped_flagged} | "
          f"rejected(bad spot) {rejected_spot} | record errors {record_errors}")
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
