"""Critical-error surfacing for the Prophecy pipeline.

Appends to the same ``data/critical_errors.jsonl`` the rest of the system uses
(read by scripts/check_critical_errors.py + the fix-agent dispatcher), so a
silent prophecy failure (empty/stale output, cost blowout) gets surfaced to the
session-start check instead of dying quietly. Append-only; never mutates.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from portfolio.file_utils import atomic_append_jsonl

from prophecy import config as pcfg

logger = logging.getLogger("prophecy.alerts")


def log_critical(category: str, message: str, *, caller: str, context: dict | None = None,
                 level: str = "error") -> None:
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "level": level,
        "category": category,
        "caller": caller,
        "message": message,
        "context": context or {},
    }
    try:
        atomic_append_jsonl(pcfg.DATA_DIR / "critical_errors.jsonl", entry)
    except Exception as exc:  # never let alerting crash the pipeline
        logger.error("failed to write critical error: %r (%s)", exc, message)
    logger.error("[%s] %s: %s", level.upper(), category, message)
