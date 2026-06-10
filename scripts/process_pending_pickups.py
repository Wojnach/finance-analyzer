"""Process due entries in `data/pending_pickups.json`.

Designed to run from a daily Windows scheduled task (PF-PendingPickups,
08:00 CET). Each pickup has a `due_ts`; when that's <= now AND the
status is "pending", the corresponding handler module under
``scripts.pickups`` is invoked, its result recorded into the pickup's
history, the pickup is marked completed, an optional Telegram alert is
sent, and a brief entry is appended to ``docs/SESSION_PROGRESS.md`` so
the next interactive Claude session sees the verdict.

Idempotent: re-running on the same day is a no-op once a pickup is
completed. Every pickup is a one-shot job — there is no recurrence
mechanism. (2026-06-10: this docstring used to claim status="recurring"
pickups were flipped back to pending after completion; no such code ever
existed and no pickup uses the field, so the claim was removed rather
than implementing a dead feature.)

Failure handling (2026-06-10): a handler error keeps the pickup at
status="pending" with an ``attempts`` counter so the next daily run
retries transient failures. After ``_MAX_ATTEMPTS`` (3) failed attempts
the pickup is parked at status="error" and a critical_errors.jsonl
entry (category="pickup_failed") is appended so the session-start check
and the fix-agent dispatcher see it.

CLI:
    python scripts/process_pending_pickups.py            # process due
    python scripts/process_pending_pickups.py --dry-run  # show plan only
    python scripts/process_pending_pickups.py --force ID # ignore due_ts

Exit code 0 if any work happened OR no work was due. Exit code 1 if a
handler returned verdict=error so the cron logs surface it. Exit code 2
if --force names an unknown pickup id.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from portfolio.file_utils import atomic_write_json, load_json  # noqa: E402

# Static handler whitelist. Adding a new pickup type requires editing this
# dict so that a malicious or corrupted `data/pending_pickups.json` cannot
# import arbitrary modules via a `handler` field. (Semgrep WARNING
# CWE-706 -- dynamic importlib.import_module with untrusted input.)
from scripts.pickups import llm_cryptotrader_72h as _llm_cryptotrader_72h  # noqa: E402

_HANDLERS = {
    "llm_cryptotrader_72h": _llm_cryptotrader_72h,
}

_PICKUPS_PATH = _REPO_ROOT / "data" / "pending_pickups.json"
_SESSION_PROGRESS = _REPO_ROOT / "docs" / "SESSION_PROGRESS.md"

# 2026-06-10: bounded retry for failed handlers. A single transient error
# (data file briefly missing, backfill behind) used to park the pickup at
# status="error" forever — never retried, invisible to the session-start
# bottle. We retry up to this many attempts, then park permanently and
# escalate via critical_errors.jsonl.
_MAX_ATTEMPTS = 3


def _now_utc() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _parse_iso(raw: str | None) -> _dt.datetime | None:
    if not raw:
        return None
    try:
        dt = _dt.datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.UTC)
        return dt
    except (TypeError, ValueError):
        return None


def _load_pickups() -> dict:
    data = load_json(str(_PICKUPS_PATH))
    if not isinstance(data, dict) or "pickups" not in data:
        return {"pickups": []}
    return data


def _save_pickups(data: dict) -> None:
    atomic_write_json(str(_PICKUPS_PATH), data)


def _send_telegram(lines: list[str]) -> None:
    """Best-effort Telegram notification. Never raises."""
    if not lines:
        return
    try:
        # 2026-06-10: was `from portfolio import config` — a module that has
        # never existed, so the ImportError was swallowed below and every
        # verdict alert silently no-oped since this script shipped. The real
        # loader is portfolio.api_utils.load_config (same pattern as
        # portfolio/main.py).
        from portfolio.api_utils import load_config
        from portfolio.telegram_notifications import send_telegram

        send_telegram("\n".join(lines), load_config())
    except Exception as e:
        # Telegram failures must never abort the pickup pipeline, but print
        # the cause so a future regression is visible in the task log
        # (2026-06-10 — the old bare `pass` hid the broken import above).
        print(f"[pickup] telegram send failed: {e!r}")


def _append_session_progress(pickup: dict, result: dict) -> None:
    """Prepend a short pickup-completed block to docs/SESSION_PROGRESS.md."""
    if not _SESSION_PROGRESS.exists():
        return
    ts = _now_utc().date().isoformat()
    title = pickup.get("title") or pickup.get("id")
    verdict = result.get("verdict", "?")
    summary = result.get("summary", "")
    block = (
        f"## {ts} -- Scheduled pickup auto-run: {pickup.get('id')} "
        f"(verdict: {verdict.upper()})\n\n"
        f"**Title:** {title}\n\n"
        f"**Summary:** {summary}\n\n"
        "Stats:\n```json\n"
        + json.dumps(result.get("details", {}), indent=2, default=str)[:2000]
        + "\n```\n\n"
        "Source: `scripts/process_pending_pickups.py` -- see "
        "`data/pending_pickups.json` for full history.\n\n---\n\n"
    )
    try:
        original = _SESSION_PROGRESS.read_text(encoding="utf-8")
    except OSError:
        return
    if original.startswith("# Session Progress\n"):
        head, _, rest = original.partition("\n")
        new = head + "\n\n" + block + rest.lstrip("\n")
    else:
        new = block + original
    _SESSION_PROGRESS.write_text(new, encoding="utf-8")


def _dispatch(pickup: dict) -> dict:
    handler_name = pickup.get("handler")
    if not handler_name or not isinstance(handler_name, str):
        return {
            "verdict": "error",
            "summary": "Pickup missing 'handler' field.",
            "details": {},
            "telegram_lines": [],
        }
    # Whitelist lookup -- never `importlib.import_module(user_input)` because
    # the handler name is read from a JSON file that an attacker with disk
    # write could craft to load arbitrary code (CWE-706). New handler types
    # require editing `_HANDLERS` at module top, which is git-tracked.
    mod = _HANDLERS.get(handler_name)
    if mod is None:
        return {
            "verdict": "error",
            "summary": (
                f"Handler {handler_name!r} not in whitelist. Add to _HANDLERS "
                "in scripts/process_pending_pickups.py before scheduling."
            ),
            "details": {"allowed_handlers": sorted(_HANDLERS.keys())},
            "telegram_lines": [],
        }
    if not hasattr(mod, "run"):
        return {
            "verdict": "error",
            "summary": f"Handler {handler_name} has no run() function.",
            "details": {},
            "telegram_lines": [],
        }
    # 2026-06-10: a raising handler used to propagate out of main() — the
    # script crashed mid-loop, _save_pickups never ran, and the scheduled
    # task surfaced nothing. Convert to a verdict=error result so the
    # bounded-retry logic in main() handles it like any other failure.
    try:
        return mod.run(pickup, _REPO_ROOT)
    except Exception as e:  # noqa: BLE001 — handler code is arbitrary
        return {
            "verdict": "error",
            "summary": f"Handler {handler_name} raised {type(e).__name__}: {e}",
            "details": {"exception_type": type(e).__name__},
            "telegram_lines": [],
        }


def _record_pickup_failure(pickup: dict, result: dict) -> None:
    """Append a critical_errors.jsonl entry when a pickup is parked at
    status="error" after exhausting retries. Best-effort: journaling
    failures must never abort the pickup pipeline."""
    try:
        from portfolio.claude_gate import record_critical_error

        record_critical_error(
            category="pickup_failed",
            caller="process_pending_pickups",
            message=(
                f"Pickup {pickup.get('id')} failed {_MAX_ATTEMPTS} attempts; "
                f"parked at status=error. Last: {result.get('summary', '')[:200]}"
            ),
            context={
                "pickup_id": pickup.get("id"),
                "handler": pickup.get("handler"),
                "attempts": pickup.get("attempts"),
            },
        )
    except Exception as e:  # noqa: BLE001
        print(f"[pickup] could not record pickup_failed critical entry: {e!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would run; mutate nothing.",
    )
    parser.add_argument(
        "--force", default=None,
        help="Force-run a specific pickup id regardless of due_ts.",
    )
    args = parser.parse_args(argv)

    data = _load_pickups()
    pickups = data.get("pickups", [])
    now = _now_utc()
    exit_code = 0
    any_due = False

    for pickup in pickups:
        pid = pickup.get("id")
        if not pid:
            continue
        status = pickup.get("status", "pending")
        due = _parse_iso(pickup.get("due_ts"))
        if args.force:
            if pid != args.force:
                continue
            due = _now_utc() - _dt.timedelta(seconds=1)
        else:
            if status != "pending":
                continue
            if due is None or due > now:
                continue

        any_due = True
        print(f"[pickup] {pid} -- dispatching {pickup.get('handler')}")
        if args.dry_run:
            print(f"  DRY: would run handler={pickup.get('handler')}")
            continue

        result = _dispatch(pickup)
        verdict = result.get("verdict", "error")
        print(f"  verdict={verdict} summary={result.get('summary', '')[:200]}")

        history = pickup.setdefault("history", [])
        history.append({
            "ran_at": now.isoformat(),
            "verdict": verdict,
            "summary": result.get("summary", ""),
            "details": result.get("details", {}),
        })
        if verdict != "error":
            pickup["status"] = "completed"
        else:
            # 2026-06-10: bounded retry. status="error" used to be terminal
            # after a single failure AND invisible to session_start_bottle —
            # a transient failure silently killed the pickup forever. Keep
            # it pending (the daily task retries) until _MAX_ATTEMPTS, then
            # park at error and escalate via critical_errors.jsonl.
            attempts = int(pickup.get("attempts", 0) or 0) + 1
            pickup["attempts"] = attempts
            if attempts >= _MAX_ATTEMPTS:
                pickup["status"] = "error"
                _record_pickup_failure(pickup, result)
            else:
                pickup["status"] = "pending"
                print(
                    f"  attempt {attempts}/{_MAX_ATTEMPTS} failed -- "
                    "left pending for retry on next run"
                )
        pickup["last_run_ts"] = now.isoformat()

        _send_telegram(result.get("telegram_lines") or [])
        _append_session_progress(pickup, result)

        if verdict == "error":
            exit_code = 1

    if not any_due:
        if args.force:
            # 2026-06-10: a typo'd --force ID used to print the routine
            # "no due pickups" and exit 0 — a success-looking no-op on the
            # documented manual recovery path (CLAUDE.md / bottle hook).
            valid = sorted(str(p.get("id")) for p in pickups if p.get("id"))
            print(
                f"[pickup] ERROR: no pickup with id {args.force!r}. "
                f"Known ids: {', '.join(valid) if valid else '(none)'}"
            )
            return 2
        print("[pickup] no due pickups")

    if not args.dry_run:
        _save_pickups(data)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
