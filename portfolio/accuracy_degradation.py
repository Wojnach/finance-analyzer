"""Signal accuracy degradation tracker.

BUG-178/W15-W16 follow-up (2026-04-16): the W15/W16 Tier-1 1d consensus
collapse from 52-56% to 36-41% (memory/project_accuracy_degradation_20260416.md)
went undetected because the existing 11 main-loop runtime contracts
(portfolio/loop_contract.py) check execution health, not decision quality.
This module is the missing piece: daily snapshots of recent-7d accuracy
across four scopes (per-signal global, per-ticker per-signal, forecast
models, aggregate consensus), an hourly comparison against the snapshot
from 7 days ago, and severity classification that plugs into the existing
Violation framework.

Public surface used by other modules:

* save_full_accuracy_snapshot()        — once-per-day snapshot writer
* check_degradation()                  — hourly violation producer
* maybe_save_daily_snapshot(config)    — guarded daily snapshot driver
* maybe_send_degradation_summary(cfg)  — guarded daily Telegram summary
* build_daily_summary(...)             — Telegram body builder
* DEGRADATION_INVARIANT                — invariant name string

Codex pre-impl review (2026-04-16) flagged 4 design problems addressed
in commit history of docs/plans/2026-04-16-accuracy-degradation-tracker.md:
  P1#1 — recent-window source, not lifetime aggregate
  P1#2 — throttle replays cached violations so ViolationTracker keeps
         escalation counts (do NOT return [] when throttled)
  P2#3 — backward FOMC/CPI window via econ_dates.recent_high_impact_events
  P2#4 — Ministral/Qwen3 from signal_log, only Chronos/Kronos from
         forecast_predictions.jsonl
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from portfolio.file_utils import atomic_write_json, load_json
from portfolio.tickers import SIGNAL_NAMES

logger = logging.getLogger("portfolio.accuracy_degradation")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ALERT_STATE_FILE = DATA_DIR / "degradation_alert_state.json"
SNAPSHOT_STATE_FILE = DATA_DIR / "accuracy_snapshot_state.json"

# --- Public constants exposed to callers (loop_contract integration) ---

DEGRADATION_INVARIANT = "accuracy_degradation"

# --- Detection tuning ---

# 15pp drop AND <50% absolute. The dual gate keeps strong signals dropping
# from 75% to 60% from triggering (still strong) while catching 58%->42%.
DROP_THRESHOLD_PP = 15.0
ABSOLUTE_FLOOR_PCT = 50.0
RISE_THRESHOLD_PP = 10.0  # symmetric for the daily summary "improved" list

# Anti-noise gates
MIN_SAMPLES_HISTORICAL = 100
MIN_SAMPLES_CURRENT = 100
MIN_SNAPSHOT_AGE_DAYS = 6.0   # don't alert without a real baseline
BASELINE_TARGET_DAYS = 7.0    # find snapshot near now-7d
BASELINE_MAX_DELTA_HOURS = 36.0  # tolerance when picking the baseline snapshot
COOLDOWN_PER_SIGNAL_S = 24 * 3600   # Telegram re-emission cooldown
HOURLY_THROTTLE_S = 55 * 60         # don't recompute more than once per ~hour
ECON_BLACKOUT_HOURS = 24.0
ECON_BLACKOUT_IMPACTS = ("high",)

# Severity classification
CRITICAL_MIN_SIGNAL_COUNT = 3   # ≥3 simultaneous drops = CRITICAL
SEVERITY_WARNING = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"

# Daily summary
SUMMARY_HOUR_UTC_DEFAULT = 6
TOP_DROPS_IN_SUMMARY = 8
TOP_GAINS_IN_SUMMARY = 5


# --- State helpers ---

def _load_alert_state() -> dict:
    state = load_json(ALERT_STATE_FILE, default={})
    if not isinstance(state, dict):
        return {}
    state.setdefault("last_full_check_time", 0.0)
    state.setdefault("last_full_check_violations", [])
    state.setdefault("last_alert_per_signal", {})
    state.setdefault("last_summary_send_time", 0.0)
    return state


def _save_alert_state(state: dict) -> None:
    atomic_write_json(ALERT_STATE_FILE, state)


def _load_snapshot_state() -> dict:
    state = load_json(SNAPSHOT_STATE_FILE, default={})
    if not isinstance(state, dict):
        return {}
    state.setdefault("last_snapshot_date_utc", "")
    return state


def _save_snapshot_state(state: dict) -> None:
    atomic_write_json(SNAPSHOT_STATE_FILE, state)


# --- Snapshot writer ---

def save_full_accuracy_snapshot(*, days: int = 7) -> dict[str, Any]:
    """Compute the full four-scope snapshot and append to accuracy_snapshots.jsonl.

    Returns the snapshot dict for inspection. Safe to call repeatedly —
    each call appends a new line. Caller is responsible for once-per-day
    gating (see maybe_save_daily_snapshot).

    BUG-178/W15-W16 review (2026-04-16): loads the signal log ONCE and
    threads pre-loaded `entries` through every per-signal/per-ticker/
    consensus call. Without the share, _per_ticker_recent alone re-scans
    the 50,000-row file 41 times, blowing snapshot wall time.
    """
    from datetime import timedelta as _td

    from portfolio.accuracy_stats import (
        accuracy_by_ticker_signal_cached,
        consensus_accuracy,
        load_entries,
        save_accuracy_snapshot,
        signal_accuracy,
    )
    from portfolio.forecast_accuracy import cached_forecast_accuracy

    extras: dict[str, Any] = {}

    cutoff = (datetime.now(UTC) - _td(days=days)).isoformat()
    all_entries = load_entries()
    recent_entries = [e for e in all_entries if e.get("ts", "") >= cutoff]

    # Recent-window per-signal accuracy (Codex P1#1)
    try:
        recent = signal_accuracy("1d", entries=recent_entries)
        extras["signals_recent"] = {
            name: {"accuracy": data["accuracy"], "total": data["total"]}
            for name, data in recent.items()
        }
    except Exception as e:
        logger.warning("Recent signal_accuracy snapshot failed: %s", e)

    # Per-ticker per-signal — lifetime via cached helper (1h TTL inside),
    # recent via shared-entries scan
    try:
        per_ticker_lifetime = accuracy_by_ticker_signal_cached("1d")
        extras["per_ticker"] = _compact_per_ticker(per_ticker_lifetime)
    except Exception as e:
        logger.warning("Lifetime per-ticker accuracy snapshot failed: %s", e)

    try:
        per_ticker_recent = _per_ticker_recent(
            "1d", days=days, entries=recent_entries,
        )
        extras["per_ticker_recent"] = _compact_per_ticker(per_ticker_recent)
    except Exception as e:
        logger.warning("Recent per-ticker accuracy snapshot failed: %s", e)

    # Forecast (Chronos/Kronos) — Codex P2#4 split. Forecast uses its
    # own JSONL (forecast_predictions), not signal_log, so we can't share
    # entries here.
    try:
        forecast_recent = cached_forecast_accuracy(
            horizon="24h", days=days, use_raw_sub_signals=True,
        )
        extras["forecast_recent"] = {
            name: {"accuracy": stats["accuracy"], "total": stats["total"]}
            for name, stats in forecast_recent.items()
        }
    except Exception as e:
        logger.warning("Forecast accuracy snapshot failed: %s", e)

    # Aggregate consensus — lifetime over all entries, recent over shared list
    try:
        extras["consensus"] = consensus_accuracy("1d", entries=all_entries)
        extras["consensus_recent"] = consensus_accuracy(
            "1d", entries=recent_entries,
        )
    except Exception as e:
        logger.warning("Consensus accuracy snapshot failed: %s", e)

    # save_accuracy_snapshot() also writes the lifetime `signals` block.
    # That helper still does its own load_entries() — acceptable cost
    # (one extra scan / day).
    return save_accuracy_snapshot(extras=extras)


def _per_ticker_recent(horizon: str, days: int, *, entries=None) -> dict:
    """Per-ticker per-signal accuracy on a recent-N-day window.

    accuracy_by_ticker_signal_cached only exposes the lifetime aggregate;
    for the recent-window variant we compute it inline from
    accuracy_by_signal_ticker(name, horizon, days=days), inverting the
    indexing so the result is keyed by ticker first.

    BUG-178/W15-W16 follow-up review (2026-04-16): pre-loaded `entries`
    is mandatory for the hot path. Without it, this function calls
    accuracy_by_signal_ticker once per signal (~41 entries in
    SIGNAL_NAMES today), each one re-scanning the entire 50,000-row
    signal log. That ~290s of redundant compute would blow the
    180s MAX_CYCLE_DURATION_S contract every time the degradation
    check runs from verify_contract.
    """
    from portfolio.accuracy_stats import accuracy_by_signal_ticker, load_entries

    if entries is None:
        entries = load_entries()

    result: dict[str, dict[str, dict]] = {}
    for sig_name in SIGNAL_NAMES:
        per_ticker = accuracy_by_signal_ticker(
            sig_name, horizon=horizon, days=days, entries=entries,
        )
        for ticker, stats in per_ticker.items():
            samples = stats.get("samples", stats.get("total", 0))
            if samples <= 0:
                continue
            result.setdefault(ticker, {})[sig_name] = {
                "accuracy": stats.get("accuracy", 0.0),
                "total": samples,
            }
    return result


def _compact_per_ticker(per_ticker: dict) -> dict:
    """Strip per-ticker per-signal blocks to just (accuracy, total).

    Snapshots are append-only forever; keep them small. accuracy_by_ticker_signal
    returns extra fields like buy_accuracy/sell_accuracy that the degradation
    detector doesn't use.
    """
    compact: dict[str, dict[str, dict]] = {}
    for ticker, sigs in (per_ticker or {}).items():
        per_sig: dict[str, dict] = {}
        for sig_name, stats in (sigs or {}).items():
            total = stats.get("total", stats.get("samples", 0))
            if total <= 0:
                continue
            per_sig[sig_name] = {
                "accuracy": stats.get("accuracy", 0.0),
                "total": total,
            }
        if per_sig:
            compact[ticker] = per_sig
    return compact


# --- Snapshot loader (delegates to accuracy_stats for the JSONL parse) ---

def _load_snapshots() -> list[dict]:
    from portfolio.accuracy_stats import _load_accuracy_snapshots
    return _load_accuracy_snapshots()


def _find_baseline_snapshot(snapshots: list[dict], now: datetime) -> dict | None:
    from portfolio.accuracy_stats import _find_snapshot_near
    target = now - timedelta(days=BASELINE_TARGET_DAYS)
    return _find_snapshot_near(snapshots, target,
                               max_delta_hours=BASELINE_MAX_DELTA_HOURS)


def _snapshot_age_days(snapshot: dict, now: datetime) -> float:
    try:
        snap_ts = datetime.fromisoformat(snapshot["ts"])
    except (KeyError, ValueError, TypeError):
        return 0.0
    return (now - snap_ts).total_seconds() / 86400.0


# --- Blackout helpers ---

def _is_econ_blackout() -> bool:
    """True when high-impact econ event is within ±ECON_BLACKOUT_HOURS."""
    try:
        from portfolio.econ_dates import (
            events_within_hours,
            recent_high_impact_events,
        )
    except Exception:
        return False
    try:
        forward = events_within_hours(ECON_BLACKOUT_HOURS)
        if any(e.get("impact") in ECON_BLACKOUT_IMPACTS for e in forward):
            return True
        backward = recent_high_impact_events(
            ECON_BLACKOUT_HOURS, impact_filter=ECON_BLACKOUT_IMPACTS,
        )
        return bool(backward)
    except Exception as e:
        logger.debug("econ blackout check failed: %s", e)
        return False


# --- Core check ---

def _make_violation(invariant: str, severity: str, message: str,
                    details: dict | None = None):
    """Build a loop_contract.Violation without a hard import dependency.

    loop_contract imports from many subsystems; importing it here at
    module-import time would risk a cycle. Lazy-import inside the call
    keeps accuracy_degradation a leaf module.
    """
    from portfolio.loop_contract import Violation
    return Violation(
        invariant=invariant,
        severity=severity,
        message=message,
        details=details or {},
    )


def check_degradation(now: datetime | None = None,
                      *,
                      drop_threshold_pp: float = DROP_THRESHOLD_PP,
                      absolute_floor_pct: float = ABSOLUTE_FLOOR_PCT,
                      min_samples_historical: int = MIN_SAMPLES_HISTORICAL,
                      min_samples_current: int = MIN_SAMPLES_CURRENT,
                      throttle_seconds: float = HOURLY_THROTTLE_S) -> list:
    """Compare recent-7d accuracy to the snapshot from 7 days ago.

    Returns a list of loop_contract.Violation objects. Codex P1#2: when
    throttled, returns the cached violations from the last full check
    rather than [] so ViolationTracker can keep the consecutive-fire
    count alive across cycles.

    Returns [] when:
    - No baseline snapshot exists.
    - Baseline is younger than MIN_SNAPSHOT_AGE_DAYS.
    - High-impact econ event is within ±24h.
    """
    now = now or datetime.now(UTC)
    state = _load_alert_state()

    # Hourly throttle — replay cached result instead of returning empty
    elapsed = time.time() - float(state.get("last_full_check_time", 0.0) or 0.0)
    if elapsed < throttle_seconds:
        return _hydrate_cached_violations(state)

    # FOMC/CPI/NFP blackout (forward + backward).
    # BUG-178/W15-W16 review (2026-04-16, P2#4): return [] rather than
    # replaying cached violations. Otherwise a 24h+ blackout (FOMC week,
    # NFP-on-Friday-then-CPI-on-Monday) keeps replaying alerts from
    # before the blackout, and ViolationTracker would happily escalate
    # the same stale alert to CRITICAL after 3 cycles.
    if _is_econ_blackout():
        logger.info("Degradation check skipped: econ blackout window active")
        # Clear the cached violation list so post-blackout checks start
        # from a known-fresh slate.
        if state.get("last_full_check_violations"):
            state["last_full_check_violations"] = []
            _save_alert_state(state)
        return []

    snapshots = _load_snapshots()
    baseline = _find_baseline_snapshot(snapshots, now) if snapshots else None
    age_days = _snapshot_age_days(baseline, now) if baseline else 0.0

    # Compute violations (empty list on the no-baseline / too-young paths)
    if baseline and age_days >= MIN_SNAPSHOT_AGE_DAYS:
        alerts = _diff_against_baseline(
            baseline=baseline,
            now=now,
            drop_threshold_pp=drop_threshold_pp,
            absolute_floor_pct=absolute_floor_pct,
            min_samples_historical=min_samples_historical,
            min_samples_current=min_samples_current,
        )
        violations = _alerts_to_violations(alerts, age_days=age_days)
    else:
        violations = []

    # Always update last_full_check_time after passing the throttle + blackout
    # gates — even when the no-baseline / too-young branches produced []. Skipping
    # the state write would let the throttle re-fire every cycle, defeating the
    # whole point. The cached violations list is also written so the next
    # throttled cycle replays exactly what verify_contract saw this time.
    state["last_full_check_time"] = time.time()
    state["last_full_check_violations"] = [_violation_to_dict(v) for v in violations]
    _save_alert_state(state)

    return violations


def _hydrate_cached_violations(state: dict) -> list:
    """Re-build Violation objects from the cached state JSON.

    This is the throttle-replay path. Crucial for ViolationTracker
    continuity — see Codex P1#2 in the plan doc.
    """
    cached = state.get("last_full_check_violations") or []
    return [_dict_to_violation(c) for c in cached]


def _violation_to_dict(violation) -> dict:
    return {
        "invariant": violation.invariant,
        "severity": violation.severity,
        "message": violation.message,
        "details": dict(violation.details or {}),
    }


def _dict_to_violation(payload: dict):
    return _make_violation(
        invariant=payload.get("invariant", DEGRADATION_INVARIANT),
        severity=payload.get("severity", SEVERITY_WARNING),
        message=payload.get("message", ""),
        details=payload.get("details", {}),
    )


# --- Diff engine ---

def _diff_against_baseline(*, baseline: dict, now: datetime,
                           drop_threshold_pp: float,
                           absolute_floor_pct: float,
                           min_samples_historical: int,
                           min_samples_current: int) -> list[dict]:
    """Return alert dicts for each scope where the degradation gate fires."""
    # BUG-178/W15-W16 review (2026-04-16): load entries ONCE and share
    # across the per-signal / per-ticker / consensus diffs. Without
    # entry-sharing _per_ticker_recent re-scans the 50,000-entry SQLite
    # file once per signal name (~41 scans), blowing the 180s cycle
    # budget every time the degradation check runs.
    from datetime import timedelta as _td

    from portfolio.accuracy_stats import (
        consensus_accuracy,
        load_entries,
        signal_accuracy,
    )
    from portfolio.forecast_accuracy import cached_forecast_accuracy
    cutoff = (now - _td(days=int(BASELINE_TARGET_DAYS))).isoformat()
    all_entries = load_entries()
    recent_entries = [e for e in all_entries if e.get("ts", "") >= cutoff]

    alerts: list[dict] = []

    # 1) Per-signal global (recent-window now vs recent-window in baseline)
    try:
        old_signals = baseline.get("signals_recent") or {}
        new_signals = signal_accuracy("1d", entries=recent_entries)
        for sig_name, new_data in new_signals.items():
            old_data = old_signals.get(sig_name)
            alert = _maybe_alert(
                key=sig_name,
                scope="signal",
                old=old_data,
                new={
                    "accuracy": new_data.get("accuracy", 0.0),
                    "total": new_data.get("total", 0),
                },
                drop_threshold_pp=drop_threshold_pp,
                absolute_floor_pct=absolute_floor_pct,
                min_samples_historical=min_samples_historical,
                min_samples_current=min_samples_current,
            )
            if alert:
                alerts.append(alert)
    except Exception as e:
        logger.warning("Per-signal degradation diff failed: %s", e)

    # 2) Per-ticker per-signal — share entries to avoid 41x re-scan
    try:
        old_per = baseline.get("per_ticker_recent") or {}
        new_per = _per_ticker_recent(
            "1d", days=int(BASELINE_TARGET_DAYS), entries=recent_entries,
        )
        for ticker, sigs in new_per.items():
            old_for_ticker = old_per.get(ticker, {}) or {}
            for sig_name, new_data in sigs.items():
                old_data = old_for_ticker.get(sig_name)
                alert = _maybe_alert(
                    key=f"{ticker}::{sig_name}",
                    scope="per_ticker",
                    old=old_data,
                    new=new_data,
                    drop_threshold_pp=drop_threshold_pp,
                    absolute_floor_pct=absolute_floor_pct,
                    min_samples_historical=min_samples_historical,
                    min_samples_current=min_samples_current,
                )
                if alert:
                    alerts.append(alert)
    except Exception as e:
        logger.warning("Per-ticker degradation diff failed: %s", e)

    # 3) Forecast models (Chronos/Kronos only — Ministral/Qwen3 are in scope #1)
    try:
        old_forecast = baseline.get("forecast_recent") or {}
        new_forecast = cached_forecast_accuracy(
            horizon="24h", days=int(BASELINE_TARGET_DAYS),
            use_raw_sub_signals=True,
        )
        for sub_name, stats in new_forecast.items():
            old_data = old_forecast.get(sub_name)
            new_data = {
                "accuracy": stats.get("accuracy", 0.0),
                "total": stats.get("total", 0),
            }
            alert = _maybe_alert(
                key=f"forecast::{sub_name}",
                scope="forecast",
                old=old_data,
                new=new_data,
                drop_threshold_pp=drop_threshold_pp,
                absolute_floor_pct=absolute_floor_pct,
                min_samples_historical=min_samples_historical,
                min_samples_current=min_samples_current,
            )
            if alert:
                alerts.append(alert)
    except Exception as e:
        logger.warning("Forecast degradation diff failed: %s", e)

    # 4) Aggregate consensus — share entries (consensus_accuracy honors
    # `entries` and skips both load_entries() and the days filter)
    try:
        old_consensus = baseline.get("consensus_recent")
        new_consensus = consensus_accuracy("1d", entries=recent_entries)
        alert = _maybe_alert(
            key="consensus",
            scope="consensus",
            old=old_consensus,
            new={
                "accuracy": new_consensus.get("accuracy", 0.0),
                "total": new_consensus.get("total", 0),
            },
            drop_threshold_pp=drop_threshold_pp,
            absolute_floor_pct=absolute_floor_pct,
            min_samples_historical=min_samples_historical,
            min_samples_current=min_samples_current,
        )
        if alert:
            alerts.append(alert)
    except Exception as e:
        logger.warning("Consensus degradation diff failed: %s", e)

    return alerts


def _maybe_alert(*, key: str, scope: str, old, new,
                 drop_threshold_pp: float, absolute_floor_pct: float,
                 min_samples_historical: int, min_samples_current: int) -> dict | None:
    """Apply the dual gate (drop AND absolute floor) to one (key, scope)."""
    if not old or not new:
        return None
    try:
        old_acc = float(old.get("accuracy", 0.0))
        old_total = int(old.get("total", 0))
        new_acc = float(new.get("accuracy", 0.0))
        new_total = int(new.get("total", 0))
    except (TypeError, ValueError):
        return None
    if old_total < min_samples_historical or new_total < min_samples_current:
        return None
    drop_pp = (old_acc - new_acc) * 100.0
    new_pct = new_acc * 100.0
    if drop_pp < drop_threshold_pp:
        return None
    if new_pct >= absolute_floor_pct:
        return None
    return {
        "key": key,
        "scope": scope,
        "old_accuracy_pct": round(old_acc * 100.0, 1),
        "new_accuracy_pct": round(new_pct, 1),
        "drop_pp": round(drop_pp, 1),
        "old_samples": old_total,
        "new_samples": new_total,
    }


def _classify_severity(alerts: list[dict]) -> str:
    """≥3 simultaneous drops OR consensus drop ⇒ CRITICAL; else WARNING."""
    if not alerts:
        return SEVERITY_WARNING
    if any(a.get("scope") == "consensus" for a in alerts):
        return SEVERITY_CRITICAL
    if len(alerts) >= CRITICAL_MIN_SIGNAL_COUNT:
        return SEVERITY_CRITICAL
    return SEVERITY_WARNING


def _alerts_to_violations(alerts: list[dict], *, age_days: float) -> list:
    if not alerts:
        return []

    severity = _classify_severity(alerts)
    summary_parts = []
    for a in alerts[:8]:  # cap message length
        summary_parts.append(
            f"{a['key']} {a['old_accuracy_pct']}%→{a['new_accuracy_pct']}%"
        )
    overflow = len(alerts) - 8
    summary_str = ", ".join(summary_parts)
    if overflow > 0:
        summary_str += f" (+{overflow} more)"

    message = (
        f"{len(alerts)} signal(s) dropped >{DROP_THRESHOLD_PP:.0f}pp vs "
        f"{int(BASELINE_TARGET_DAYS)}d baseline AND below "
        f"{ABSOLUTE_FLOOR_PCT:.0f}% absolute: {summary_str}"
    )

    details = {
        "alert_count": len(alerts),
        "baseline_age_days": round(age_days, 1),
        "alerts": alerts,
    }

    return [_make_violation(DEGRADATION_INVARIANT, severity, message, details)]


# --- Cooldown filter for the Telegram path ---

def filter_alerts_by_cooldown(alerts: list[dict],
                              now_ts: float | None = None,
                              cooldown_s: float = COOLDOWN_PER_SIGNAL_S) -> list[dict]:
    """Drop alerts that fired for the same key within `cooldown_s` seconds.

    Codex P1#2: the underlying Violation list still includes ALL alerts
    (so ViolationTracker can escalate), but Telegram and the daily
    summary should not re-shout. Updates state with new last-alert
    timestamps for keys that pass.
    """
    now_ts = now_ts if now_ts is not None else time.time()
    state = _load_alert_state()
    last_per_key = state.get("last_alert_per_signal") or {}

    fresh: list[dict] = []
    for a in alerts:
        key = a.get("key", "")
        last_ts = float(last_per_key.get(key, 0.0) or 0.0)
        if now_ts - last_ts < cooldown_s:
            continue
        fresh.append(a)
        last_per_key[key] = now_ts

    state["last_alert_per_signal"] = last_per_key
    _save_alert_state(state)
    return fresh


# --- Daily snapshot + summary drivers (called from main.py post-cycle) ---

def maybe_save_daily_snapshot(config: dict | None = None,
                              now: datetime | None = None) -> bool:
    """Write a snapshot iff today's snapshot hasn't been written yet.

    Returns True when a snapshot was written this call. Driven by main.py
    in the post-cycle path, gated by configurable hour-of-day so the
    snapshot lands after the daily PF-OutcomeCheck backfill runs.

    BUG-178/W15-W16 review (2026-04-16, P2#3): on success this also
    invalidates the degradation check throttle (last_full_check_time=0)
    so the next contract cycle re-runs the full check against the
    freshly-written snapshot rather than replaying the cached
    violation list that compared against the previous baseline.
    """
    now = now or datetime.now(UTC)
    cfg_section = (config or {}).get("notification", {}) if config else {}
    target_hour = int(cfg_section.get(
        "accuracy_snapshot_hour_utc", SUMMARY_HOUR_UTC_DEFAULT,
    ))
    if now.hour < target_hour:
        return False

    state = _load_snapshot_state()
    today_str = now.date().isoformat()
    if state.get("last_snapshot_date_utc") == today_str:
        return False

    try:
        save_full_accuracy_snapshot()
    except Exception as e:
        logger.warning("Daily accuracy snapshot failed: %s", e)
        return False

    state["last_snapshot_date_utc"] = today_str
    _save_snapshot_state(state)

    # Force the next contract cycle to re-check against the new snapshot.
    alert_state = _load_alert_state()
    alert_state["last_full_check_time"] = 0.0
    alert_state["last_full_check_violations"] = []
    _save_alert_state(alert_state)
    return True


def maybe_send_degradation_summary(config: dict | None = None,
                                   now: datetime | None = None) -> bool:
    """Send the daily Telegram summary iff today's hasn't been sent yet.

    BUG-178/W15-W16 review (2026-04-16, P1#2): refuses to build a
    summary if the latest snapshot is older than today's expected
    snapshot. Without this guard a silent snapshot failure would
    cause the summary to ship yesterday's data labeled as today's
    "Δ vs prev 7d", and on a recurring failure the same stale data
    would shout day after day.
    """
    now = now or datetime.now(UTC)
    cfg_section = (config or {}).get("notification", {}) if config else {}
    target_hour = int(cfg_section.get(
        "accuracy_snapshot_hour_utc", SUMMARY_HOUR_UTC_DEFAULT,
    ))
    if now.hour < target_hour:
        return False

    state = _load_alert_state()
    last_send_ts = float(state.get("last_summary_send_time", 0.0) or 0.0)
    last_send_dt = datetime.fromtimestamp(last_send_ts, tz=UTC) if last_send_ts else None
    if last_send_dt and last_send_dt.date() == now.date():
        return False

    try:
        snapshots = _load_snapshots()
        if not snapshots:
            return False
        latest = snapshots[-1]
        # Refuse to ship stale data: today's snapshot must already exist.
        try:
            latest_ts = datetime.fromisoformat(latest.get("ts", ""))
        except (ValueError, TypeError):
            latest_ts = None
        if latest_ts is None or latest_ts.date() != now.date():
            logger.warning(
                "Daily summary skipped: latest snapshot %s is not from today (%s). "
                "Snapshot writer likely failed; check accuracy_snapshot_state.json.",
                latest.get("ts"), now.date().isoformat(),
            )
            return False
        baseline = _find_baseline_snapshot(snapshots, now)
        body = build_daily_summary(latest=latest, baseline=baseline, now=now)
    except Exception as e:
        logger.warning("Daily summary build failed: %s", e)
        return False

    try:
        from portfolio.message_store import send_or_store
        # send_or_store(msg, config, category=...) — config is REQUIRED
        # (needs telegram.token + telegram.chat_id). Production bug found
        # 2026-04-16 first cycle after the merge: passing only `category=`
        # raised TypeError, the wrapper try/except caught it but the daily
        # summary never went out. Pass the loop's config dict through.
        send_or_store(body, config or {}, category="daily_digest")
    except Exception as e:
        logger.warning("Daily summary send failed: %s", e)
        return False

    state["last_summary_send_time"] = time.time()
    _save_alert_state(state)
    return True


# --- Telegram body builder ---

def build_daily_summary(*, latest: dict, baseline: dict | None,
                        now: datetime | None = None) -> str:
    """Build the Telegram body for the *ACCURACY DAILY* summary."""
    now = now or datetime.now(UTC)
    lines = [f"*ACCURACY DAILY* · {now.date().isoformat()}"]

    consensus_recent = latest.get("consensus_recent") or {}
    consensus_acc = float(consensus_recent.get("accuracy", 0.0)) * 100.0
    consensus_total = int(consensus_recent.get("total", 0))
    delta_str = ""
    if baseline:
        b_consensus = baseline.get("consensus_recent") or {}
        try:
            b_acc = float(b_consensus.get("accuracy", 0.0)) * 100.0
            delta = consensus_acc - b_acc
            delta_str = f" (Δ {delta:+.1f}pp vs prev 7d)"
        except (TypeError, ValueError):
            delta_str = ""
    lines.append(
        f"`Consensus: {consensus_acc:.0f}% recent7d{delta_str} · "
        f"{consensus_total} sam`"
    )

    # Forecast block: explicit allowlist instead of Ministral/Qwen3
    # exclusion, so a future renamed model doesn't silently misclassify
    # (review P3#7).
    forecast = latest.get("forecast_recent") or {}
    forecast_pairs = [
        (k, v.get("accuracy", 0.0)) for k, v in forecast.items()
        if k.startswith("chronos") or k.startswith("kronos")
    ]
    if forecast_pairs:
        rendered = " · ".join(
            f"{name.replace('_24h','').replace('_1h','')} {acc*100:.0f}%"
            for name, acc in sorted(forecast_pairs)
        )
        lines.append(f"`Forecast:  {rendered}`")

    signals_recent = latest.get("signals_recent") or {}
    llm_pairs = [
        (k, v.get("accuracy", 0.0)) for k, v in signals_recent.items()
        if k in ("ministral", "qwen3")
    ]
    if llm_pairs:
        rendered = " · ".join(
            f"{name} {acc*100:.0f}%"
            for name, acc in sorted(llm_pairs)
        )
        lines.append(f"`LLM:       {rendered}`")

    drops, gains = _summary_diffs(latest=latest, baseline=baseline)
    if drops:
        lines.append("")
        lines.append(
            f"*Degraded (>{DROP_THRESHOLD_PP:.0f}pp drop vs prev 7d, "
            f"<{ABSOLUTE_FLOOR_PCT:.0f}% recent abs)*"
        )
        for d in drops[:TOP_DROPS_IN_SUMMARY]:
            lines.append(_format_summary_row(d))

    if gains:
        lines.append("")
        lines.append(f"*Improved (>{RISE_THRESHOLD_PP:.0f}pp gain vs prev 7d)*")
        for g in gains[:TOP_GAINS_IN_SUMMARY]:
            lines.append(_format_summary_row(g))

    snap_age = "?"
    try:
        snap_age_days = _snapshot_age_days(baseline, now) if baseline else 0.0
        snap_age = f"{snap_age_days:.1f}d"
    except Exception:
        pass

    sig_count = len(signals_recent or {})
    lines.append("")
    lines.append(
        f"`Snapshot age: {snap_age} · {sig_count} signals tracked · window: recent-7d`"
    )

    return "\n".join(lines)


def _summary_diffs(*, latest: dict,
                   baseline: dict | None) -> tuple[list[dict], list[dict]]:
    drops: list[dict] = []
    gains: list[dict] = []
    if not baseline:
        return drops, gains

    new_signals = latest.get("signals_recent") or {}
    old_signals = baseline.get("signals_recent") or {}
    for name, new_data in new_signals.items():
        old_data = old_signals.get(name)
        if not old_data:
            continue
        try:
            old_acc = float(old_data.get("accuracy", 0.0))
            new_acc = float(new_data.get("accuracy", 0.0))
            samples = int(new_data.get("total", 0))
        except (TypeError, ValueError):
            continue
        change_pp = (new_acc - old_acc) * 100.0
        if change_pp <= -DROP_THRESHOLD_PP and new_acc * 100.0 < ABSOLUTE_FLOOR_PCT:
            drops.append({
                "key": name,
                "old_accuracy_pct": round(old_acc * 100.0, 1),
                "new_accuracy_pct": round(new_acc * 100.0, 1),
                "drop_pp": round(-change_pp, 1),
                "samples": samples,
            })
        elif change_pp >= RISE_THRESHOLD_PP:
            gains.append({
                "key": name,
                "old_accuracy_pct": round(old_acc * 100.0, 1),
                "new_accuracy_pct": round(new_acc * 100.0, 1),
                "drop_pp": round(-change_pp, 1),  # negative = gain
                "samples": samples,
            })

    drops.sort(key=lambda x: x["drop_pp"], reverse=True)
    gains.sort(key=lambda x: x["drop_pp"])  # most-negative first (biggest gain)
    return drops, gains


def _format_summary_row(item: dict) -> str:
    name = item["key"]
    return (
        f"`{name:<10} {item['old_accuracy_pct']:>4.0f}% -> "
        f"{item['new_accuracy_pct']:>4.0f}% "
        f"({-item['drop_pp']:+.0f}pp, {item['samples']:>5} sam)`"
    )
