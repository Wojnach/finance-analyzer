"""SessionStart hook: surface pending pickups as a bottle-from-the-ocean message.

Runs at the start of every Claude Code session in this repo (wired up
via `.claude/settings.json`). Reads `data/pending_pickups.json` and
emits a JSON blob in the SessionStart hook format whose
`additionalContext` is injected into the session prompt, so the
session opens already aware of any due / overdue verification jobs
even if neither the user nor the previous session-end remembered to
mention them.

Output contract: a single line of JSON shaped like

    {"hookSpecificOutput": {
      "hookEventName": "SessionStart",
      "additionalContext": "..."
    }}

The hook never writes files, never raises (any error path prints a
minimal payload), and is silent when no pickups need attention.

The bottle is also a sentinel for the next session to act:
- DUE TODAY or OVERDUE + status=pending  -> caller should consider
  running ``.venv/Scripts/python.exe scripts/process_pending_pickups.py``
  to fire the cron path manually, OR open the More -> Pickups dashboard
  tile.
- COMPLETED yesterday or earlier -> caller should read the latest
  history entry in pending_pickups.json (or skim `docs/SESSION_PROGRESS.md`
  top) and decide whether the verdict needs human action.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path


def _quiet_payload(text: str) -> None:
    """Emit the SessionStart hook envelope with `text` as additionalContext."""
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": text,
        }
    }))


def main() -> int:
    try:
        repo_root = Path(__file__).resolve().parent.parent
        pickups_path = repo_root / "data" / "pending_pickups.json"
        if not pickups_path.exists():
            return 0
        try:
            data = json.loads(pickups_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        if not isinstance(data, dict):
            return 0
        rows = data.get("pickups") or []
        if not rows:
            return 0

        now = _dt.datetime.now(_dt.UTC)
        lines: list[str] = []
        recent_completions: list[str] = []
        errored: list[str] = []
        for p in rows:
            if not isinstance(p, dict):
                continue
            pid = p.get("id") or "?"
            title = p.get("title") or ""
            status = p.get("status", "pending")
            due_raw = p.get("due_ts")
            due_dt = None
            if isinstance(due_raw, str):
                try:
                    due_dt = _dt.datetime.fromisoformat(due_raw)
                    if due_dt.tzinfo is None:
                        due_dt = due_dt.replace(tzinfo=_dt.UTC)
                except ValueError:
                    due_dt = None
            days_until = None
            if due_dt is not None:
                days_until = (due_dt - now).total_seconds() / 86400.0

            if status == "pending" and days_until is not None and days_until <= 14.0:
                # Surface anything pending within 2 weeks. Severity tag
                # signals whether action is needed NOW or just FYI.
                if days_until < 0:
                    tag = "OVERDUE"
                elif days_until <= 1.0:
                    tag = "DUE TODAY"
                elif days_until <= 7.0:
                    tag = f"DUE in {days_until:.1f}d"
                else:
                    tag = f"pending ({days_until:.1f}d)"
                lines.append(
                    f"  - [{tag}] {pid}: {title}"
                )
            elif status == "completed" and p.get("last_run_ts"):
                last_run = p.get("last_run_ts")
                try:
                    lr_dt = _dt.datetime.fromisoformat(last_run)
                    if lr_dt.tzinfo is None:
                        lr_dt = lr_dt.replace(tzinfo=_dt.UTC)
                    age_h = (now - lr_dt).total_seconds() / 3600.0
                except (TypeError, ValueError):
                    age_h = None
                if age_h is not None and age_h <= 48.0:
                    history = p.get("history") or []
                    last_verdict = None
                    if history and isinstance(history[-1], dict):
                        last_verdict = history[-1].get("verdict")
                    summary = ""
                    if history and isinstance(history[-1], dict):
                        summary = (history[-1].get("summary") or "")[:120]
                    recent_completions.append(
                        f"  - {pid}: verdict={last_verdict} -- {summary}"
                    )
            elif status == "error":
                # 2026-06-10: errored pickups used to be invisible here —
                # the exact "work scheduled for a future session" this hook
                # exists to never forget vanished silently after the handler
                # exhausted its retries. Surface them until a human acts.
                history = p.get("history") or []
                last_summary = ""
                if history and isinstance(history[-1], dict):
                    last_summary = (history[-1].get("summary") or "")[:120]
                attempts = p.get("attempts")
                errored.append(
                    f"  - [ERRORED] {pid}: {title} -- handler failed"
                    f"{f' {attempts}x' if attempts else ''}; "
                    f"last: {last_summary}"
                )

        if not lines and not recent_completions and not errored:
            return 0

        parts = [
            "BOTTLE FROM PRIOR SESSION -- scheduled pickups:",
        ]
        if lines:
            parts.append("Pending verifications (auto-run by PF-PendingPickups daily 08:00 CET on due-date):")
            parts.extend(lines)
            parts.append(
                "Force-run a pickup now: "
                "`.venv/Scripts/python.exe scripts/process_pending_pickups.py --force <ID>`"
            )
        if errored:
            parts.append(
                "Errored pickups (handler gave up after retries -- needs human "
                "attention; history in data/pending_pickups.json):"
            )
            parts.extend(errored)
        if recent_completions:
            parts.append("Recently completed (last 48h -- read history before next action):")
            parts.extend(recent_completions)
        parts.append(
            "Source: `data/pending_pickups.json`. Dispatcher: "
            "`scripts/process_pending_pickups.py`. Docs: "
            "`docs/IMPROVEMENT_BACKLOG.md`."
        )

        _quiet_payload("\n".join(parts))
        return 0
    except Exception as e:  # noqa: BLE001 — never let the hook crash a session
        _quiet_payload(f"(pickup hook error: {e!r})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
