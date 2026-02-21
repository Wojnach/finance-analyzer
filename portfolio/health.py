"""Health monitoring for the finance-analyzer Layer 1 loop."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

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
    if error:
        state["errors"] = state.get("errors", [])[-19:] + [
            {"ts": datetime.now(timezone.utc).isoformat(), "error": error}
        ]
        state["error_count"] = state.get("error_count", 0) + 1
    # Atomic write
    tmp = HEALTH_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(HEALTH_FILE)


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


def get_health_summary() -> dict:
    """Return a summary dict suitable for API/dashboard consumption."""
    state = load_health()
    is_stale, age, _ = check_staleness()
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
    }
