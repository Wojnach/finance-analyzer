"""Loop heartbeat rollup — single source of truth for loop liveness.

The crypto and oil swing loops write `data/{name}_loop.heartbeat` JSON
files each successful cycle (per the 2026-05-01 hardening in commit
e7a1ec47). This module reads them all and returns a per-loop status
dict that the dashboard, watchdog, and any future operator script can
consume.

Status semantics:

| State    | Meaning |
|----------|---------|
| `fresh`  | heartbeat exists AND age <= STALE_THRESHOLD_SECONDS |
| `stale`  | heartbeat exists but age > STALE_THRESHOLD_SECONDS |
| `missing`| heartbeat file does not exist |
| `unparseable` | file exists but is malformed JSON or has no ts |

The `stale` and `missing` states are what the watchdog alerts on. The
`unparseable` state should be rare — usually means a half-written file
during cycle end and clears on the next cycle.

NOT a replacement for the per-loop scorecards. Scorecards report on
trade quality; this module reports on whether the loop is even running.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("loop_health")

# Default stale threshold — loops cycle every 60s, so anything older than
# 5 minutes is definitely a problem. Scripts can override per-loop if
# they want tighter or looser bounds.
STALE_THRESHOLD_SECONDS = 300

# Map of loop_name -> heartbeat file path (relative to repo root). The
# main loop (PF-DataLoop) is intentionally NOT here — it has its own
# liveness mechanism via data/health_state.json + loop_contract.py
# invariants. Duplicating coverage would risk alert-fatigue on the same
# incident. To extend: add the new loop's heartbeat write at the end of
# its cycle and register the path here.
DEFAULT_HEARTBEAT_FILES: dict[str, str] = {
    "crypto": "data/crypto_loop.heartbeat",
    "oil": "data/oil_loop.heartbeat",
    "mstr": "data/mstr_loop.heartbeat",
    "metals": "data/metals_loop.heartbeat",
    "golddigger": "data/golddigger_loop.heartbeat",
}


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def read_loop_status(
    name: str,
    path: str | Path,
    now: datetime.datetime | None = None,
    stale_threshold_seconds: int = STALE_THRESHOLD_SECONDS,
) -> dict[str, Any]:
    """Read a single loop's heartbeat and classify its state.

    Returns a dict with: name, path, state, age_seconds, payload, error.
    `state` is one of: "fresh", "stale", "missing", "unparseable".
    """
    now = now or _now_utc()
    path = Path(path)
    out: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "state": "missing",
        "age_seconds": None,
        "payload": None,
        "error": None,
    }

    if not path.exists():
        return out

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        out["state"] = "unparseable"
        out["error"] = f"read failed: {exc}"
        return out

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        out["state"] = "unparseable"
        out["error"] = f"json decode: {exc}"
        return out

    out["payload"] = payload
    ts_str = payload.get("ts") if isinstance(payload, dict) else None
    if not ts_str:
        out["state"] = "unparseable"
        out["error"] = "no ts field"
        return out

    # 2026-05-02 codex P3: a heartbeat file can be valid JSON but have
    # a non-string ts (number, object, list). Without this guard,
    # ts_str.replace() raises AttributeError and crashes the rollup.
    if not isinstance(ts_str, str):
        out["state"] = "unparseable"
        out["error"] = f"ts not a string (got {type(ts_str).__name__})"
        return out

    try:
        ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.UTC)
    except (ValueError, TypeError, AttributeError) as exc:
        out["state"] = "unparseable"
        out["error"] = f"ts parse: {exc}"
        return out

    age_seconds = (now - ts).total_seconds()
    out["age_seconds"] = round(age_seconds, 2)
    out["state"] = "fresh" if age_seconds <= stale_threshold_seconds else "stale"
    return out


def read_loop_health(
    repo_root: str | Path | None = None,
    files: dict[str, str] | None = None,
    now: datetime.datetime | None = None,
    stale_threshold_seconds: int = STALE_THRESHOLD_SECONDS,
) -> dict[str, Any]:
    """Read all configured loop heartbeats and return a rollup.

    Args:
        repo_root: Defaults to the repo root inferred from this file's
            location. Pass an explicit path in tests.
        files: Map of loop_name -> heartbeat file path (relative to
            repo_root). Defaults to DEFAULT_HEARTBEAT_FILES. Tests pass
            their own dict.
        now: For deterministic tests. Defaults to UTC now.
        stale_threshold_seconds: Override the default 300s threshold.

    Returns:
        {
          "checked_at": ISO timestamp,
          "stale_threshold_seconds": int,
          "loops": {name: {state, age_seconds, payload, error, path}, ...},
          "any_unhealthy": bool,
          "unhealthy": [name, ...],   # loops in stale/missing/unparseable
        }
    """
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
    files = files or DEFAULT_HEARTBEAT_FILES
    now = now or _now_utc()

    loops: dict[str, dict[str, Any]] = {}
    unhealthy: list[str] = []
    for name, rel_path in files.items():
        full_path = repo_root / rel_path
        status = read_loop_status(
            name, full_path,
            now=now, stale_threshold_seconds=stale_threshold_seconds,
        )
        loops[name] = status
        if status["state"] != "fresh":
            unhealthy.append(name)

    return {
        "checked_at": now.isoformat(),
        "stale_threshold_seconds": stale_threshold_seconds,
        "loops": loops,
        "any_unhealthy": bool(unhealthy),
        "unhealthy": unhealthy,
    }


def write_heartbeat(
    path: str | Path,
    cycle: int,
    *,
    ok: bool = True,
    n_positions: int = 0,
    extra: dict[str, Any] | None = None,
    now: datetime.datetime | None = None,
) -> bool:
    """Write a watchdog-compatible heartbeat file.

    Centralised so loops don't reinvent the schema. Best-effort by design:
    swallows all exceptions and returns False — the caller (a live
    trading loop) must never crash because telemetry failed.

    Args:
        path: Destination heartbeat path (typically `data/{name}_loop.heartbeat`).
        cycle: Monotonic cycle counter from the loop. Operator-facing.
        ok: Whether the cycle ran cleanly. Operator-facing.
        n_positions: Currently-open position count. Operator-facing.
        extra: Additional fields merged into the payload (e.g. phase).
        now: Override timestamp (for tests). Defaults to UTC now.

    Returns:
        True if the file was written, False if anything failed.

    NOTE: New loops should call this helper. Existing crypto_loop /
    oil_loop / mstr_loop ship their own private wrappers (predate this
    function); migrating them is a no-op refactor for a future PR.
    """
    try:
        from portfolio.file_utils import atomic_write_json
        ts = (now or _now_utc()).isoformat()
        payload: dict[str, Any] = {
            "ts": ts,
            "status": "ok" if ok else "degraded",
            "cycle": cycle,
            "ok": ok,
            "n_positions": n_positions,
        }
        if extra:
            payload.update(extra)
        atomic_write_json(str(path), payload)
        return True
    except Exception:
        # Best-effort: never crash the loop on telemetry failure.
        logger.debug("write_heartbeat failed for %s", path, exc_info=True)
        return False


__all__ = [
    "STALE_THRESHOLD_SECONDS",
    "DEFAULT_HEARTBEAT_FILES",
    "read_loop_status",
    "read_loop_health",
    "write_heartbeat",
]
