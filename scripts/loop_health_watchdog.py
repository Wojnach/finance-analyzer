"""Periodic loop-health watchdog — telegram alerts on stale/missing heartbeats.

Designed for a scheduled task that fires every 30 minutes. Reads the
loop health rollup via `portfolio.loop_health.read_loop_health()`. If
any loop is unhealthy, sends a single consolidated telegram message
(not per-loop spam).

Per-loop cooldown prevents the same dead loop from spamming every 30
minutes — default cooldown is 4 hours. State persisted in
`data/loop_health_watchdog_state.json`.

Usage:
    .venv/Scripts/python.exe scripts/loop_health_watchdog.py
"""
from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.loop_health import read_loop_health

logger = logging.getLogger("loop_health_watchdog")

STATE_FILE = REPO / "data" / "loop_health_watchdog_state.json"
COOLDOWN_HOURS = 4


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _load_state() -> dict[str, Any]:
    return load_json(str(STATE_FILE)) or {"last_alert_per_loop": {}}


def _save_state(state: dict[str, Any]) -> None:
    try:
        atomic_write_json(str(STATE_FILE), state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("save state failed: %s", exc)


def _is_in_cooldown(loop_name: str, state: dict[str, Any],
                     now: datetime.datetime,
                     cooldown_hours: int = COOLDOWN_HOURS) -> bool:
    """Return True if we recently alerted on this loop."""
    last_iso = (state.get("last_alert_per_loop") or {}).get(loop_name)
    if not last_iso:
        return False
    try:
        last = datetime.datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
        if last.tzinfo is None:
            last = last.replace(tzinfo=datetime.UTC)
    except ValueError:
        return False
    return (now - last).total_seconds() < cooldown_hours * 3600


def build_alert(rollup: dict[str, Any], state: dict[str, Any],
                 now: datetime.datetime) -> tuple[str | None, list[str]]:
    """Construct an alert message for the loops worth alerting on.

    Returns (message, alerted_loops). message is None when nothing is
    worth alerting (either all fresh, or all unhealthy loops are in
    cooldown).
    """
    alerted = []
    lines = []
    for name in rollup.get("unhealthy", []):
        if _is_in_cooldown(name, state, now):
            continue
        loop_status = rollup["loops"].get(name) or {}
        loop_state = loop_status.get("state", "?")
        age = loop_status.get("age_seconds")
        if loop_state == "stale":
            lines.append(f"  • {name}: STALE (age={age:.0f}s)")
        elif loop_state == "missing":
            lines.append(f"  • {name}: NO HEARTBEAT — task likely never started")
        elif loop_state == "unparseable":
            err = loop_status.get("error", "?")
            lines.append(f"  • {name}: heartbeat unparseable ({err})")
        else:
            lines.append(f"  • {name}: {loop_state}")
        alerted.append(name)

    if not lines:
        return None, []

    message = (
        "⚠️ Loop health watchdog\n\n"
        f"{len(alerted)} loop(s) need attention:\n"
        + "\n".join(lines)
        + f"\n\nChecked at {now.isoformat()}."
        + f"\nNext alert per loop in ≥{COOLDOWN_HOURS}h."
    )
    return message, alerted


def send_telegram(message: str) -> bool:
    """Best-effort telegram send.

    Returns True only when the underlying send_telegram() actually
    delivered the message. Returns False when:
      - config.json missing or has no telegram.token
      - send_telegram() returned False (typically because
        telegram.mute_all=true or telegram.layer1_messages=false)
      - send_telegram() raised an exception

    2026-05-02 codex P2: previously returned True unconditionally after
    calling _send(), which masked the muted/suppressed cases and let
    the cooldown gate respect a delivery that never happened.
    """
    try:
        from portfolio.telegram_notifications import send_telegram as _send
        cfg = load_json(str(REPO / "config.json")) or {}
        if not cfg.get("telegram", {}).get("token"):
            logger.info("telegram not configured — printing message")
            print(message)
            return False
        result = _send(message, cfg)
        # send_telegram() returns True/False to indicate actual delivery.
        # None or other falsy → treat as "did not actually send".
        if result is False:
            logger.info("telegram send suppressed (mute_all or layer1_messages)")
            print(message)
            return False
        return bool(result) if result is not None else True
    except Exception as exc:  # noqa: BLE001
        logger.warning("telegram send failed: %s — printing", exc)
        print(message)
        return False


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    now = _now_utc()
    state = _load_state()
    rollup = read_loop_health()

    if not rollup["any_unhealthy"]:
        logger.info("all loops healthy: %s",
                    {n: rollup["loops"][n]["state"] for n in rollup["loops"]})
        return 0

    message, alerted = build_alert(rollup, state, now)
    if message is None:
        logger.info("unhealthy loops in cooldown: %s — skipping alert",
                    rollup["unhealthy"])
        return 0

    sent = send_telegram(message)
    # 2026-05-02 codex P2: only stamp the cooldown when the alert
    # actually went out. If telegram is muted/down/missing config, we
    # still WANT the next watchdog tick to retry rather than wait 4h.
    # The trade-off: when mute_all is on, the watchdog will print to
    # stdout every 30 min instead of every 4h. That's louder in the
    # log but correct: an unhealthy loop the operator can't see is
    # worse than a noisy log.
    if sent:
        state.setdefault("last_alert_per_loop", {})
        for name in alerted:
            state["last_alert_per_loop"][name] = now.isoformat()
        _save_state(state)
        logger.info("alerted on %s (telegram delivered)", alerted)
    else:
        logger.warning("alerted on %s but telegram NOT delivered — "
                        "cooldown not set, will retry next tick", alerted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
