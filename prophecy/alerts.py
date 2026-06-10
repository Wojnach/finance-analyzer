"""Critical-error surfacing for the Prophecy pipeline.

Appends to the same ``data/critical_errors.jsonl`` the rest of the system uses
(read by scripts/check_critical_errors.py + the fix-agent dispatcher), so a
silent prophecy failure (empty/stale output, cost blowout) gets surfaced to the
session-start check instead of dying quietly. Append-only; never mutates.

2026-06-11 (audit batch 3): default level changed "error" -> "critical".
check_critical_errors.find_unresolved() and the fix-agent dispatcher only
surface entries with level == "critical"; the old default made every prophecy
alert invisible to both consumers (P1, IMPROVEMENT_AUDIT_2026-06-10).

Rate limiting (premortem hook 13, BINDING): a structurally-failing day (e.g. a
whole class of instruments unscoreable) must not flood critical_errors.jsonl or
burn the fix-agent dispatcher's exponential backoff to disabled-on-noise. Max
ONE critical journal append per (category, UTC day); suppressed emissions still
hit the module logger so the .bat / Task Scheduler log retains every event.
State file: data/prophecy_runs/alert_ratelimit.json (atomic I/O).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
from prophecy import config as pcfg

logger = logging.getLogger("prophecy.alerts")

_RATELIMIT_FILENAME = "alert_ratelimit.json"


def _ratelimit_path():
    # Resolved at call time (not import time) so tests that monkeypatch
    # pcfg.PROPHECY_DIR get the redirected location.
    return pcfg.PROPHECY_DIR / _RATELIMIT_FILENAME


def _ratelimit_allows(category: str, today: str) -> bool:
    """True if no critical for ``category`` has been journaled today.

    Records the emission optimistically (before the journal append) — losing
    one alert to a crash between mark and append is preferable to a flood if
    the append itself is what keeps failing. Fail-open on unreadable state
    (over-emitting once beats silently suppressing), but the write path calls
    ensure_dirs() so a missing prophecy_runs/ dir doesn't disable limiting.
    """
    try:
        pcfg.ensure_dirs()
        state = load_json(_ratelimit_path(), default={}) or {}
        if not isinstance(state, dict):
            state = {}
        if state.get(category) == today:
            return False
        state[category] = today
        atomic_write_json(_ratelimit_path(), state)
        return True
    except Exception as exc:  # never let rate limiting crash alerting
        logger.error("alert rate-limit state unavailable: %r (emitting anyway)", exc)
        return True


def log_critical(category: str, message: str, *, caller: str, context: dict | None = None,
                 level: str = "critical") -> None:
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "level": level,
        "category": category,
        "caller": caller,
        "message": message,
        "context": context or {},
    }
    # Rate limit applies to surfacing-level entries only (1/category/day).
    # Sub-critical levels (e.g. cost.py's intentionally non-surfacing soft-cap
    # "warning") are informational rows; they pass through unlimited.
    if level == "critical":
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if not _ratelimit_allows(category, today):
            logger.error("[RATE-LIMITED] [%s] %s: %s (1/day/category cap hit — "
                         "journal append suppressed)", level.upper(), category, message)
            return
    try:
        atomic_append_jsonl(pcfg.DATA_DIR / "critical_errors.jsonl", entry)
    except Exception as exc:  # never let alerting crash the pipeline
        logger.error("failed to write critical error: %r (%s)", exc, message)
    logger.error("[%s] %s: %s", level.upper(), category, message)
