"""Prophecy step 5 — token/cost tracking (ZERO tokens).

Parses the ``claude -p --output-format json`` result written to
``run_<date>.json`` (``total_cost_usd`` + ``usage`` + ``num_turns`` +
``is_error``), appends a row to ``cost_log.jsonl`` and rolls a ``cost_summary``
into ``latest.json``.

"Go unhinged to begin with, control later" — so this NEVER blocks a run; it just
measures, and raises a critical-error ALERT when a run exceeds the (optional)
``budget_usd_soft_cap`` or the Claude result reports an error.

Run: ``python -m prophecy.cost [--date YYYY-MM-DD]``
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json, load_jsonl_tail
from prophecy import config as pcfg
from prophecy.alerts import log_critical

logger = logging.getLogger("prophecy.cost")


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _parse_result(path) -> dict | None:
    """Extract the claude `result` object from a json or stream-json file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = text.strip()
    if not text:
        return None
    # Fast path: a single JSON object/array.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):  # stream emitted as a JSON array
            for item in reversed(obj):
                if isinstance(item, dict) and (item.get("type") == "result" or "total_cost_usd" in item):
                    return item
    except json.JSONDecodeError:
        pass
    # Stream-json: scan lines, take the last result-bearing object.
    result = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (obj.get("type") == "result" or "total_cost_usd" in obj):
            result = obj
    return result


def _usage_tokens(result: dict) -> dict:
    usage = result.get("usage") or {}
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "cache_read_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_tokens": usage.get("cache_creation_input_tokens"),
    }


def record_cost(date: str | None = None) -> int:
    date = date or _today()
    pcfg.ensure_dirs()
    run_path = pcfg.run_file(date)

    result = _parse_result(run_path)
    if result is None:
        log_critical("prophecy_cost", f"no parseable claude result in {run_path.name}",
                     caller="cost.no_result", context={"date": date})
        print(f"no parseable result: {run_path}")
        return 1

    is_error = bool(result.get("is_error"))
    total_usd = result.get("total_cost_usd")
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "date": date,
        # 2026-06-11 (audit batch 3): prophecy-daily.bat now resolves its
        # --model from the SAME config key (prophecy_config.json "model"), so
        # this stamp matches the actually-used model. Prefer the model the
        # claude result self-reports when present (belt and braces against a
        # config edit landing mid-run).
        "model": result.get("model") or pcfg.model(),
        "total_cost_usd": total_usd,
        "num_turns": result.get("num_turns"),
        "duration_ms": result.get("duration_ms"),
        "is_error": is_error,
        "subtype": result.get("subtype"),
        "session_id": result.get("session_id"),
        **_usage_tokens(result),
    }
    atomic_append_jsonl(pcfg.COST_LOG, entry)

    # 30-day cumulative for the dashboard.
    # 2026-06-11 (audit batch 3): previously summed the last 30 JSONL *entries*
    # — same-day retries inflated the figure and skipped days silently widened
    # the window past 30 days. Now: a real ts >= now-30d window, deduped to the
    # LAST entry per (date, session_id). Rationale: re-parsing the same
    # run_<date>.json (manual `python -m prophecy.cost` re-run) appends a
    # duplicate row for the SAME claude session = same spend, counted once;
    # a genuine same-day re-run (Task Scheduler retry that re-invoked claude)
    # has a new session_id = new real spend, still counted. The only consumer
    # is the dashboard cost_summary tile (the soft-cap gate below uses the
    # per-run total only), so display semantics are the constraint.
    rows = load_jsonl_tail(pcfg.COST_LOG, max_entries=1000) or []
    cutoff = datetime.now(UTC) - timedelta(days=30)
    last_per_run: dict[tuple, dict] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            ts = datetime.fromisoformat(r.get("ts"))
        except (TypeError, ValueError):
            continue
        if ts < cutoff:
            continue
        last_per_run[(r.get("date"), r.get("session_id"))] = r
    cumulative = sum(r.get("total_cost_usd") or 0.0 for r in last_per_run.values())

    # latest.json is MULTI-WRITER: publish.py writes the predictions snapshot,
    # then this re-reads it and sets only cost_summary. The .bat enforces the
    # order publish -> outcomes -> cost, so cost always layers on top of a fresh
    # snapshot (review P2). We re-read immediately before writing to pick up
    # publish's version; only cost_summary is mutated here.
    latest = load_json(pcfg.LATEST_FILE, default={}) or {}
    latest["cost_summary"] = {
        "last_run_usd": total_usd,
        "cumulative_30d_usd": round(cumulative, 4),
        "num_turns": entry["num_turns"],
        "is_error": is_error,
        "date": date,
    }
    atomic_write_json(pcfg.LATEST_FILE, latest)

    # Alerts (never blocks).
    if is_error:
        log_critical("prophecy_cost", f"claude run reported error: {result.get('subtype')}",
                     caller="cost.run_error", context={"date": date})
    soft_cap = pcfg.budget_soft_cap()
    if soft_cap is not None and isinstance(total_usd, (int, float)) and total_usd > soft_cap:
        # level="warning" is INTENTIONAL and intentionally non-surfacing:
        # check_critical_errors.py + the fix-agent dispatcher only act on
        # level == "critical". The soft cap is a "measure, don't block" budget
        # tripwire ("go unhinged to begin with") — it lands in the journal for
        # cost archaeology but must not page the session-start check or spawn
        # fix agents. Escalate to the default (critical) only when a HARD
        # budget policy is decided. (audit batch 3, 2026-06-11)
        log_critical("prophecy_cost",
                     f"run cost ${total_usd:.2f} exceeded soft cap ${soft_cap:.2f}",
                     caller="cost.over_budget", level="warning",
                     context={"date": date, "total_usd": total_usd, "soft_cap": soft_cap})

    print(f"cost: ${total_usd} | turns={entry['num_turns']} | err={is_error} | "
          f"30d=${cumulative:.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Prophecy cost tracking (zero tokens)")
    ap.add_argument("--date", default=None)
    args = ap.parse_args(argv)
    return record_cost(args.date)


if __name__ == "__main__":
    raise SystemExit(main())
