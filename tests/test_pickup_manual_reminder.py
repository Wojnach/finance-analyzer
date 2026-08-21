"""Tests for `scripts/pickups/manual_reminder.py`.

Three pickups (`ROTATE-LEAKED-CREDS`, `TRIM-SNDK-WDC-NBIS`,
`STORAGE-REVISIT-AUG10`) name handler `manual_reminder`. Until 2026-08-21 no
such module existed and it was absent from the dispatcher whitelist, so
`pf-pickups.service` failed every morning at 08:42 with

    Handler 'manual_reminder' not in whitelist.

and the systemd unit sat in `failed`.

The handler automates nothing by design — the whole point is that a human
must act. So it must return verdict `defer`, which is the one verdict
`main()` treats as "leave this pickup pending" (see the 2026-08-04 note at
`process_pending_pickups.py:277`). Anything else either closes a reminder
nobody acted on (`completed`) or re-reds the unit (`error`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _pickup(context=None, **over):
    p = {
        "id": "ROTATE-LEAKED-CREDS",
        "title": "Rotate credentials exposed in early git history",
        "handler": "manual_reminder",
        "due_ts": "2026-07-31T00:00:00+02:00",
        "context": {} if context is None else context,
    }
    p.update(over)
    return p


def _run(pickup):
    from scripts.pickups import manual_reminder

    return manual_reminder.run(pickup, _REPO_ROOT)


# --------------------------------------------------------------------------
# The verdict contract — this is the actual bug being fixed
# --------------------------------------------------------------------------


def test_verdict_is_defer_so_the_reminder_stays_pending():
    assert _run(_pickup())["verdict"] == "defer"


def test_verdict_is_defer_even_when_context_is_rich():
    """A fully-specified reminder is still a human action, never auto-closed."""
    result = _run(
        _pickup(
            {
                "requires_human": True,
                "priority": "high",
                "action": "SELL 2 of 5 STX shares (~15K SEK). Keep 3.",
            }
        )
    )
    assert result["verdict"] == "defer"


def test_result_has_the_four_keys_the_dispatcher_reads():
    result = _run(_pickup())
    assert set(result) >= {"verdict", "summary", "details", "telegram_lines"}
    assert isinstance(result["summary"], str) and result["summary"]
    assert isinstance(result["details"], dict)
    assert isinstance(result["telegram_lines"], list)


# --------------------------------------------------------------------------
# The reminder has to actually surface its content
# --------------------------------------------------------------------------


def test_summary_carries_the_pickup_title():
    result = _run(_pickup())
    assert "Rotate credentials exposed in early git history" in result["summary"]


def test_summary_marks_the_reminder_as_needing_a_human():
    result = _run(_pickup({"requires_human": True}))
    assert "human" in result["summary"].lower()


def test_action_from_context_reaches_the_telegram_lines():
    action = "SELL 2 of 5 STX shares (~15K SEK). Keep 3."
    lines = _run(_pickup({"action": action, "priority": "high"}))["telegram_lines"]
    assert any(action in ln for ln in lines)


def test_priority_reaches_the_telegram_lines():
    lines = _run(_pickup({"priority": "high"}))["telegram_lines"]
    assert any("high" in ln.lower() for ln in lines)


def test_days_overdue_is_reported():
    """A reminder 3 weeks past due must not read the same as one due today."""
    details = _run(_pickup(due_ts="2026-07-31T00:00:00+02:00"))["details"]
    assert details.get("days_overdue") is not None
    assert details["days_overdue"] > 0


def test_details_echo_the_context_so_the_history_entry_is_self_contained():
    ctx = {"priority": "high", "note": "detail lives in the operator's memory dir"}
    details = _run(_pickup(ctx))["details"]
    assert details.get("context") == ctx


def test_details_name_how_to_close_the_reminder():
    """Defer means it fires forever — the operator needs the exit instruction."""
    details = _run(_pickup())["details"]
    blob = " ".join(str(v) for v in details.values())
    assert "status" in blob and "pending_pickups.json" in blob


# --------------------------------------------------------------------------
# Never raises — a reminder crash must not red the unit
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pickup",
    [
        {},
        {"id": "X"},
        {"id": "X", "context": None},
        {"id": "X", "context": "not-a-dict"},
        {"id": "X", "due_ts": "not-a-timestamp"},
        {"id": "X", "due_ts": None, "title": None},
    ],
)
def test_never_raises_on_malformed_pickups(pickup):
    result = _run(pickup)
    assert result["verdict"] == "defer"
    assert isinstance(result["summary"], str)


# --------------------------------------------------------------------------
# Dispatcher wiring — the whitelist is the thing that was actually broken
# --------------------------------------------------------------------------


def test_handler_is_in_the_dispatcher_whitelist():
    import scripts.process_pending_pickups as proc

    assert "manual_reminder" in proc._HANDLERS


def test_dispatch_of_a_real_manual_reminder_pickup_does_not_error():
    import scripts.process_pending_pickups as proc

    result = proc._dispatch(_pickup({"requires_human": True, "priority": "high"}))
    assert result["verdict"] == "defer"


def test_every_handler_named_in_pending_pickups_is_whitelisted():
    """Guards the class of bug, not just this instance: a pickup whose handler
    is not whitelisted fails silently-forever at 08:42 each morning."""
    import scripts.process_pending_pickups as proc
    from portfolio.file_utils import load_json

    data = load_json(str(_REPO_ROOT / "data" / "pending_pickups.json")) or {}
    named = {p.get("handler") for p in (data.get("pickups") or []) if p.get("handler")}
    missing = sorted(named - set(proc._HANDLERS))
    assert not missing, f"pickup handlers absent from _HANDLERS: {missing}"
