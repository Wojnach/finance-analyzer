"""Health monitoring for the finance-analyzer Layer 1 loop."""

import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, last_jsonl_entry, load_json, load_jsonl_tail

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEALTH_FILE = DATA_DIR / "health_state.json"

# C10/H17: Protect all read-modify-write sequences in health.py.
_health_lock = threading.Lock()


def update_health(cycle_count: int, signals_ok: int, signals_failed: int,
                  last_trigger_reason: str = None, error: str = None):
    """Called at end of each Layer 1 cycle to update health state."""
    with _health_lock:
        state = load_health()
        state["last_heartbeat"] = datetime.now(UTC).isoformat()
        state["cycle_count"] = cycle_count
        state["signals_ok"] = signals_ok
        state["signals_failed"] = signals_failed
        state["uptime_seconds"] = time.time() - state.get("start_time", time.time())
        if last_trigger_reason:
            state["last_trigger_reason"] = last_trigger_reason
            state["last_trigger_time"] = datetime.now(UTC).isoformat()
            # Cache the invocation timestamp so check_agent_silence() can avoid
            # re-parsing invocations.jsonl on every call.
            state["last_invocation_ts"] = state["last_trigger_time"]
        if error:
            state["errors"] = state.get("errors", [])[-19:] + [
                {"ts": datetime.now(UTC).isoformat(), "error": error}
            ]
            state["error_count"] = state.get("error_count", 0) + 1
        atomic_write_json(HEALTH_FILE, state)


def load_health() -> dict:
    """Load current health state. Returns defaults if missing or corrupt."""
    state = load_json(HEALTH_FILE)
    if state is not None:
        return state
    return {"start_time": time.time(), "cycle_count": 0, "error_count": 0, "errors": []}


def reset_session_start():
    """Reset start_time to current time — call at loop startup.

    Prevents uptime_seconds from inheriting a stale start_time
    from a previous session's health_state.json.
    """
    with _health_lock:
        state = load_health()
        state["start_time"] = time.time()
        atomic_write_json(HEALTH_FILE, state)


def check_staleness(max_age_seconds: int = 300) -> tuple:
    """Check if the loop heartbeat is stale.
    Returns (is_stale: bool, age_seconds: float, state: dict)
    """
    state = load_health()
    hb = state.get("last_heartbeat")
    if not hb:
        return True, float("inf"), state
    last = datetime.fromisoformat(hb)
    age = (datetime.now(UTC) - last).total_seconds()
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

    # Fall back to parsing invocations.jsonl if health_state doesn't have the timestamp.
    if not last_ts:
        invocations_file = DATA_DIR / "invocations.jsonl"
        last_ts = last_jsonl_entry(invocations_file, field="ts")
        if last_ts is None:
            return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}

    if not last_ts:
        return {"silent": True, "age_seconds": float("inf"), "threshold": max_market_seconds, "market_open": False}

    last = datetime.fromisoformat(last_ts)
    now = datetime.now(UTC)
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


def update_module_failures(failures: list):
    """Record which reporting modules failed in the current cycle.

    Called by reporting.py after generating the agent summary.
    Persists module names + timestamp in health_state.json so the dashboard
    and monitoring scripts can see per-module status without parsing logs.
    """
    if not failures:
        return
    with _health_lock:
        state = load_health()
        state["last_module_failures"] = {
            "ts": datetime.now(UTC).isoformat(),
            "modules": list(failures),
        }
        atomic_write_json(HEALTH_FILE, state)


def update_signal_health(signal_name: str, success: bool):
    """Record a single signal execution result.

    For batch updates (multiple signals per cycle), prefer
    update_signal_health_batch() to avoid repeated disk writes.
    """
    update_signal_health_batch({signal_name: success})


def update_signal_health_batch(results: dict):
    """Record multiple signal execution results in a single disk write.

    Args:
        results: dict of {signal_name: bool} where True=success, False=failure.
    """
    if not results:
        return
    with _health_lock:
        state = load_health()
        sh = state.setdefault("signal_health", {})
        now = datetime.now(UTC).isoformat()

        for signal_name, success in results.items():
            entry = sh.setdefault(signal_name, {
                "total_calls": 0,
                "total_failures": 0,
                "last_success": None,
                "last_failure": None,
                "recent_results": [],
            })
            entry["total_calls"] = entry.get("total_calls", 0) + 1
            if success:
                entry["last_success"] = now
            else:
                entry["total_failures"] = entry.get("total_failures", 0) + 1
                entry["last_failure"] = now

            # Rolling window: keep last 50 results for recent success rate
            recent = entry.get("recent_results", [])
            recent.append(success)
            if len(recent) > 50:
                recent = recent[-50:]
            entry["recent_results"] = recent

        atomic_write_json(HEALTH_FILE, state)


def get_signal_health(signal_name: str = None) -> dict:
    """Get signal health data.

    If signal_name is given, returns that signal's health dict.
    Otherwise returns the full signal_health dict for all signals.
    """
    state = load_health()
    sh = state.get("signal_health", {})
    if signal_name:
        return sh.get(signal_name, {})
    return sh


def get_signal_health_summary() -> dict:
    """Compact signal health summary for reporting.

    Returns dict of signal_name -> {success_rate, total_calls, total_failures,
    last_failure} for signals with at least 1 call.
    """
    sh = get_signal_health()
    summary = {}
    for sig_name, data in sh.items():
        total = data.get("total_calls", 0)
        if total == 0:
            continue
        failures = data.get("total_failures", 0)
        recent = data.get("recent_results", [])
        recent_rate = (sum(1 for r in recent if r) / len(recent) * 100) if recent else 0
        summary[sig_name] = {
            "success_rate_pct": round(recent_rate, 1),
            "total_calls": total,
            "total_failures": failures,
            "last_failure": data.get("last_failure"),
        }
    return summary


def get_health_summary() -> dict:
    """Return a summary dict suitable for API/dashboard consumption."""
    state = load_health()
    is_stale, age, _ = check_staleness()
    agent_silence = check_agent_silence()
    summary = {
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
        "module_failures": state.get("last_module_failures"),
        "signal_health": get_signal_health_summary(),
    }
    # Include circuit breaker status if data_collector has been imported
    try:
        from portfolio.data_collector import alpaca_cb, binance_fapi_cb, binance_spot_cb
        summary["circuit_breakers"] = {
            "binance_spot": binance_spot_cb.get_status(),
            "binance_fapi": binance_fapi_cb.get_status(),
            "alpaca": alpaca_cb.get_status(),
        }
    except Exception as e:
        logger.warning("Circuit breaker status unavailable: %s", e)
    return summary


def check_outcome_staleness(max_age_hours: int = 36) -> dict:
    """Check if outcome backfill is stale (no recent outcomes in signal_log).

    Returns dict with: stale (bool), newest_outcome_age_hours (float),
    entries_without_outcomes (int).
    """
    signal_log = DATA_DIR / "signal_log.jsonl"

    now = time.time()
    newest_outcome_ts = 0
    missing_count = 0
    # BUG-122: Use load_jsonl_tail instead of reading the entire 68MB file
    entries = load_jsonl_tail(signal_log, max_entries=50)
    if not entries:
        return {"stale": True, "newest_outcome_age_hours": float("inf"),
                "entries_without_outcomes": 0}

    try:
        for entry in entries:
            outcomes = entry.get("outcomes", {})
            has_any = any(
                outcomes.get(t, {}).get("1d") is not None
                for t in outcomes
            )
            if has_any:
                # Parse outcome timestamps to find newest
                for t_outcomes in outcomes.values():
                    for h_data in t_outcomes.values():
                        if isinstance(h_data, dict) and h_data.get("ts"):
                            try:
                                ots = datetime.fromisoformat(h_data["ts"]).timestamp()
                                newest_outcome_ts = max(newest_outcome_ts, ots)
                            except (ValueError, TypeError):
                                pass
            else:
                missing_count += 1
    except Exception as exc:
        logger.warning("check_outcome_staleness error: %s", exc)
        return {"stale": True, "newest_outcome_age_hours": float("inf"),
                "entries_without_outcomes": 0}

    if newest_outcome_ts == 0:
        age_hours = float("inf")
    else:
        age_hours = (now - newest_outcome_ts) / 3600

    return {
        "stale": age_hours > max_age_hours,
        "newest_outcome_age_hours": round(age_hours, 1),
        "entries_without_outcomes": missing_count,
    }


def check_dead_signals(recent_entries: int = 20) -> list[str]:
    """Detect signals that voted HOLD on every ticker in the last N entries.

    Returns list of signal names that are effectively dead (100% HOLD).
    """
    signal_log = DATA_DIR / "signal_log.jsonl"

    # BUG-122: Use load_jsonl_tail instead of reading the entire 68MB file
    entries = load_jsonl_tail(signal_log, max_entries=recent_entries)
    if not entries:
        return []

    # Collect vote counts per signal
    from collections import defaultdict
    vote_counts = defaultdict(lambda: {"total": 0, "non_hold": 0})

    for entry in entries:
        for _ticker, tdata in entry.get("tickers", {}).items():
            for sig_name, vote in tdata.get("signals", {}).items():
                vote_counts[sig_name]["total"] += 1
                if vote in ("BUY", "SELL"):
                    vote_counts[sig_name]["non_hold"] += 1

    # Signals with >0 total votes but 0 non-HOLD votes are dead
    dead = []
    for sig_name, counts in vote_counts.items():
        if counts["total"] >= recent_entries and counts["non_hold"] == 0:
            dead.append(sig_name)
    return sorted(dead)
