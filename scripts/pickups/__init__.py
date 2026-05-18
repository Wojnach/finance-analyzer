"""Pickup handlers for `scripts/process_pending_pickups.py`.

Each handler is a module with a top-level `run(pickup: dict, repo_root: Path)
-> dict` function that:

* reads the pickup's `context` block,
* performs the verification / decision work,
* returns a dict shaped like
  `{"verdict": "promote"|"retire"|"defer", "summary": "...",
    "details": {...}, "telegram_lines": [...]}`,
* MUST NOT raise. Wrap exceptions and return verdict="error" with
  the trace in `details`.

The orchestrator (`process_pending_pickups.py`) writes the handler's
return value into the pickup's `history[]`, marks the pickup
`status="completed"`, sends Telegram, and appends to SESSION_PROGRESS.md
so the next interactive Claude session picks up the verdict.

Add a new pickup type:
1. Drop a new module `scripts/pickups/<id_lowercased>.py` exposing
   `run(pickup, repo_root)`.
2. Reference its filename stem in the pickup entry's `handler` field.
3. Add an integration test in `tests/test_pickup_<id>.py`.
"""

from __future__ import annotations
