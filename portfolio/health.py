"""Health monitoring for the finance-analyzer Layer 1 loop."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from portfolio.file_utils import atomic_write_json

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEALTH_FILE = DATA_DIR / "health_state.json"


def update_health(cycle_count: int, signals_ok: int, signals_failed: int,
                  last_trigger_reason: str = None, error: str = None):
    """Called at end of each Layer 1 cycle to update health state."""
    state = load_health()
    state["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
    state["cycle_count"] = cycle_count
    state["signals_ok"] = signals_ok
    state["signals_failed"] = signals_failed
    state["uptime_seconds"] = time.time() - state.get("start_time", time.time())
    if last_trigger_reason:
        state["last_trigger_reason"] = last_trigger_reason
        state["last_trigger_time"] = datetime.now(timezone.utc).isoformat()
        # Cache the invocation timestamp so check_agent_silence() can avoid
        # re-parsing invocations.jsonl on every call.
        state["last_invocation_ts"] = state["last_trigger_time"]
    if error:
        state["errors"] = state.get("errors", [])[-19:] + [
            {"ts": datetime.now(timezone.utc).isoformat(), "error": error}
        ]
        state["error_count"] = state.get("error_count", 0) + 1
    atomic_write_json(HEALTH_FILE, state)


def load_health() -> dict:
    """Load current health state."""
    if HEALTH_FILE.exists():
        try:
            return json.loads(HEALTH_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"start_time": time.time(), "cycle_count": 0, "error_count": 0, "errors": []}


def check_staleness(max_age_seconds: int = 300) -> tuple:
    """Check if the loop heartbeat is stale.
    Returns (is_stale: bool, age_seconds: float, state: dict)
    """
    state = load_health()
    hb = state.get("last_heartbeat")
    if not hb:
        return True, float("inf"), state
    last = datetime.fromisoformat(hb)
    age = (datetime.now(timezone.utc) - last).total_seconds()
    return age > max_age_seconds, age, state


def check_agent_silence(max_market_seconds: int = 7200,
                        max_offhours_seconds: int = 14400) -> dict:
    """Detect silent Layer 2 agent (no invocation for too long).

    Args:
        max_market_seconds: Max allowed silence during market hours (default 2h).
        max_offhours_seconds: Max allowed silence outside market hours (default 4h).

    Returns:
        dict with keys: silent (bool), age_seconds (float), threshold (int), market_open (bool)
    """
    # Try cached timestamp from health_state first (avoids re-parsing invocations.jsonl)
    last_ts = None
    state = load_health()
    last_ts = state.get("last_invocation_ts")

    # Fall back to parsing invocations.jsonl if health_state doesn't have the timestamp
    if not last_ts:
        invocations_file = DATA_DIR / "invocations.jsonl"
        if not invocations_file.exists():
            return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}

        with open(invocations_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    ts = entry.get("ts")
                    if ts:
                        last_ts = ts
                except json.JSONDecodeError:
                    continue

    if not last_ts:
        return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}

    last = datetime.fromisoformat(last_ts)
    now = datetime.now(timezone.utc)
    age = (now - last).total_seconds()

    # DST-aware market hours check
    from portfolio.market_timing import get_market_state
    market_state, _, _ = get_market_state()
    market_open = (market_state == "open")
    threshold = max_market_seconds if market_open else max_offhours_seconds

    return {
        "silent": age > threshold,
        "age_seconds": round(age, 1),
        "threshold": threshold,
        "market_open": market_open,
    }


def get_health_summary() -> dict:
    """Return a summary dict suitable for API/dashboard consumption."""
    state = load_health()
    is_stale, age, _ = check_staleness()
    agent_silence = check_agent_silence()
    return {
        "status": "stale" if is_stale else "healthy",
        "heartbeat_age_seconds": round(age, 1),
        "cycle_count": state.get("cycle_count", 0),
        "error_count": state.get("error_count", 0),
        "last_trigger": state.get("last_trigger_reason"),
        "last_trigger_time": state.get("last_trigger_time"),
        "recent_errors": state.get("errors", [])[-5:],
        "signals_ok": state.get("signals_ok", 0),
        "signals_failed": state.get("signals_failed", 0),
        "agent_silent": agent_silence["silent"],
        "agent_silence_seconds": agent_silence["age_seconds"],
    }
