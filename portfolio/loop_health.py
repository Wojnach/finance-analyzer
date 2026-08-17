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
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("loop_health")

# Default stale threshold — loops cycle every 60s, so anything older than
# 5 minutes is definitely a problem. Scripts can override per-loop if
# they want tighter or looser bounds.
STALE_THRESHOLD_SECONDS = 300


# --- suspend-aware heartbeat (data/heartbeat.txt) --------------------------
#
# This host is a Steam Deck and deep-suspends constantly: measured 2026-08-17,
# it had spent 48 of its last 96 uptime hours suspended. Wall-clock staleness
# therefore says nothing about loop health — main.py's cadence is anchored to
# time.monotonic(), which excludes suspend, so the loop stays on schedule while
# a wall-clock reader screams that it died.
#
# Naively subtracting total suspend-since-boot is NOT safe: after an 8h
# overnight suspend it would happily mask a genuine 30-minute stall the next
# morning. Instead each heartbeat records the monotonic clock at write time,
# so age can be measured in awake seconds exactly. The boot_id scopes that
# anchor — monotonic restarts at zero on reboot, so a heartbeat from a previous
# boot has no comparable reading and reports awake age as unknown rather than
# inventing one.

_BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"


def _awake_seconds() -> float:
    """Seconds of wall time the host has been AWAKE since boot.

    CLOCK_MONOTONIC on Linux excludes time spent suspended, which is the
    property this whole module hangs on. Falls back to time.monotonic() on
    platforms without the named clock.
    """
    try:
        return time.clock_gettime(time.CLOCK_MONOTONIC)
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        return time.monotonic()


def _boot_id() -> str:
    """Current host boot identifier, or "" when unavailable (non-Linux)."""
    try:
        with open(_BOOT_ID_PATH, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:  # pragma: no cover - non-Linux
        return ""


def write_wall_heartbeat(path, now: datetime.datetime | None = None) -> bool:
    """Write Layer 1's own heartbeat with a suspend-proof age anchor.

    Best-effort like write_heartbeat(): a telemetry failure must never take
    down the loop.
    """
    stamp = now or datetime.datetime.now(datetime.UTC)
    payload = {
        "ts": stamp.isoformat(),
        "awake_s": float(_awake_seconds()),
        "boot_id": _boot_id(),
    }
    try:
        from portfolio.file_utils import atomic_write_text

        atomic_write_text(Path(path), json.dumps(payload))
        return True
    except Exception:
        logger.warning("heartbeat write failed for %s", path, exc_info=True)
        return False


def read_heartbeat_age(path) -> dict[str, Any]:
    """Age of a heartbeat written by write_wall_heartbeat().

    Returns ``{"ts", "wall_age_s", "awake_age_s", "same_boot"}``.

    ``awake_age_s`` is None whenever it cannot be known honestly — a legacy
    bare-ISO file, a heartbeat from a previous boot, or an unreadable file.
    Callers must fall back to ``wall_age_s`` and accept its suspend blindness
    rather than treat a missing anchor as zero age.
    """
    out: dict[str, Any] = {
        "ts": None,
        "wall_age_s": None,
        "awake_age_s": None,
        "same_boot": False,
        # has_anchor distinguishes "written before this feature existed"
        # (legacy bare-ISO, no anchor) from "written by a previous boot"
        # (anchor present, boot_id differs). Both leave awake_age_s None, but
        # only the second is evidence of a reboot — conflating them logged a
        # reboot for a heartbeat written 0 minutes earlier.
        "has_anchor": False,
    }
    try:
        raw = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return out
    if not raw:
        return out

    stamp_text, awake_s, boot_id = raw, None, None
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
            stamp_text = payload.get("ts") or ""
            awake_s = payload.get("awake_s")
            boot_id = payload.get("boot_id")
        except (ValueError, AttributeError):
            return out

    try:
        stamp = datetime.datetime.fromisoformat(stamp_text)
    except (ValueError, TypeError):
        return out
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=datetime.UTC)

    out["ts"] = stamp.isoformat()
    out["wall_age_s"] = (datetime.datetime.now(datetime.UTC) - stamp).total_seconds()

    out["has_anchor"] = isinstance(awake_s, (int, float)) and bool(boot_id)
    if out["has_anchor"] and boot_id == _boot_id():
        out["same_boot"] = True
        out["awake_age_s"] = max(0.0, _awake_seconds() - float(awake_s))
    return out


def heartbeat_is_stale(path, threshold_s: float) -> bool:
    """True when a heartbeat is older than threshold_s of AWAKE time.

    Prefers the suspend-proof anchor and falls back to wall clock only when
    there is none. A missing/unreadable heartbeat is not reported stale here —
    "absent" is a different condition and callers handle it separately.
    """
    info = read_heartbeat_age(path)
    age = info["awake_age_s"]
    if age is None:
        age = info["wall_age_s"]
    if age is None:
        return False
    return age > threshold_s


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
            name,
            full_path,
            now=now,
            stale_threshold_seconds=stale_threshold_seconds,
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
