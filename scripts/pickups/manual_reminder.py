"""Pickup handler: surface a reminder that only a human can close.

Some pickups are not computations — they are standing instructions to the
operator (rotate the leaked credentials, revisit the storage book). There is
nothing to automate: the handler's whole job is to re-state the reminder and
its age so it keeps showing up until the human acts.

That maps onto verdict ``defer``, the one verdict ``main()`` leaves at
status="pending" (see ``process_pending_pickups.py`` near the 2026-08-04
note). ``completed`` would retire a reminder nobody acted on; ``error`` would
burn the bounded-retry budget and park the pickup at status="error" after
three days, which is how `pf-pickups.service` came to sit in ``failed`` before
this module existed.

The reminder therefore fires forever by design, so the details block carries
the instruction for closing it out.

Never raises: a malformed pickup must still produce a reminder rather than
taking the dispatcher down.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

_CLOSE_HINT = (
    'set "status" to "completed" for this id in data/pending_pickups.json '
    "once the action is done"
)

# Context keys worth putting in front of the operator, in reading order.
# Anything else still lands in details["context"].
_HEADLINE_KEYS = ("action", "decision_standing", "note")


def _days_overdue(due_ts):
    """Whole days between due_ts and now, or None if unparseable."""
    if not isinstance(due_ts, str) or not due_ts.strip():
        return None
    raw = due_ts.strip().replace("Z", "+00:00")
    try:
        due = _dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if due.tzinfo is None:
        due = due.replace(tzinfo=_dt.UTC)
    return (_dt.datetime.now(_dt.UTC) - due).days


def run(pickup: dict, repo_root: Path) -> dict:
    """Re-state a human-only reminder; never raises, always defers."""
    try:
        pickup = pickup if isinstance(pickup, dict) else {}
        ctx = pickup.get("context")
        ctx = ctx if isinstance(ctx, dict) else {}

        pid = pickup.get("id") or "(no id)"
        title = pickup.get("title") or "(untitled reminder)"
        priority = ctx.get("priority")
        overdue = _days_overdue(pickup.get("due_ts"))

        age = f", {overdue}d overdue" if overdue and overdue > 0 else ""
        summary = f"{pid}: {title} — needs a human{age}. Left pending."

        lines = [f"Reminder {pid}{age}", title]
        if priority:
            lines.append(f"Priority: {priority}")
        for key in _HEADLINE_KEYS:
            val = ctx.get(key)
            if isinstance(val, str) and val.strip():
                lines.append(f"{key.replace('_', ' ').capitalize()}: {val}")
        lines.append(f"To close: {_CLOSE_HINT}")

        return {
            "verdict": "defer",
            "summary": summary,
            "details": {
                "requires_human": True,
                "days_overdue": overdue,
                "priority": priority,
                "how_to_close": _CLOSE_HINT,
                "context": ctx,
            },
            "telegram_lines": lines,
        }
    except Exception as e:  # noqa: BLE001 — a reminder must never red the unit
        return {
            "verdict": "defer",
            "summary": f"Manual reminder could not be rendered ({type(e).__name__}: {e}). Left pending.",
            "details": {"requires_human": True, "how_to_close": _CLOSE_HINT},
            "telegram_lines": ["Manual reminder — see data/pending_pickups.json"],
        }
