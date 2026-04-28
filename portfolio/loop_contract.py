"""Loop Contract — runtime invariant verification for all system loops.

After every cycle, verify functions check that critical operations
actually happened. Violations are logged, alerted, and optionally
trigger a self-healing Claude Code session.

Supports: main loop, metals loop, GoldDigger, Elongir.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.file_utils import (
    atomic_append_jsonl,
    atomic_write_json,
    last_jsonl_entry,
    load_json,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONTRACT_STATE_FILE = DATA_DIR / "contract_state.json"
CONTRACT_LOG_FILE = DATA_DIR / "contract_violations.jsonl"
CONFIG_FILE = BASE_DIR / "config.json"
HEALTH_STATE_FILE = DATA_DIR / "health_state.json"
LAYER2_JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"

# 2026-04-28: Per-invariant Telegram alert cooldown. Background: the
# accuracy_degradation invariant uses a throttled-replay design (replays
# cached violations every cycle to keep ViolationTracker.consecutive
# alive). Without per-alert dedup _alert_violations shipped one Telegram
# per cycle for 192 cycles in a row before we noticed. The cooldown only
# suppresses *exact* replays — same invariant + same message text within
# the window. Any text change (a new degraded signal joining the alert
# list, a different trigger reason on layer2_journal_activity) bypasses
# the cooldown and re-fires immediately. Configurable via
# notification.contract_alert_cooldown_s; defaults to 4 h, which is short
# enough that a stuck regression still pages the user a few times per day
# but long enough that the same-text replay is rate-limited 24x.
DEFAULT_CONTRACT_ALERT_COOLDOWN_S = 4 * 3600

# 2026-04-28 (Codex P2): TTL for the critical_errors.jsonl dedup. After
# this many seconds, a same-text degradation replay re-emits a fresh
# critical_errors row so the auto-fix-agent dispatcher
# (PF-FixAgentDispatcher, 24 h lookback in scripts/fix_agent_dispatcher.py)
# keeps seeing the incident as long as it persists. 6 h gives the
# dispatcher 4 fresh entries per dispatcher-day, well inside its
# lookback window, while still rate-limiting the same-issue noise 4x
# vs the per-cycle pattern that prompted this fix.
DEFAULT_CRITICAL_ERRORS_DEDUP_TTL_S = 6 * 3600

# 2026-04-28 (Codex P2): invariants whose CRITICAL violations get routed
# to critical_errors.jsonl after ViolationTracker has had a chance to
# escalate. Kept as a set rather than a hard-coded "if accuracy_degradation"
# branch so adding another auto-fix-agent-friendly invariant is a one-
# liner. Note: ``layer2_journal_activity`` is intentionally NOT in this
# set — it already calls record_critical_error inline on the original
# (pre-tracker) violation; routing it again post-tracker would double-
# write.
CRITICAL_ERROR_DISPATCH_INVARIANTS = frozenset({"accuracy_degradation"})

# 2026-04-28 (Codex P2): how many recent message-hashes to remember per
# invariant in both the Telegram cooldown and critical_errors dedup
# tables. The earlier one-hash-per-invariant design lost the original
# A on an A -> B -> A flap inside the dedup window, re-firing what was
# really the same incident as a fresh alert. 8 entries bounds storage
# while covering essentially every plausible flap pattern (a single
# alert text rarely oscillates more than ~3 ways inside 6 h).
MAX_RECENT_HASHES_PER_INVARIANT = 8

# 2026-04-28 (Codex P2): the contract Telegram alert category. Kept as a
# constant so _alert_violations and the mute-check helper agree. Don't
# change this without updating message_store.SEND_CATEGORIES too.
CONTRACT_ALERT_CATEGORY = "error"
# Global Claude CLI log — written by claude_gate for ALL callers
# (claude_fundamental, bigbet, iskbets, self-heal, etc.). Used here only
# for *enriching* violation context with last_invocation_caller.
CLAUDE_INVOCATIONS_FILE = DATA_DIR / "claude_invocations.jsonl"
# Layer-2-specific invocation log — written by agent_invocation._log_trigger
# and check_agent_completion. Has entries with tier=1/2/3 + status. Used for
# the in-flight suppression check because it ONLY contains Layer 2 events
# and won't false-positive suppress on an unrelated claude_fundamental run
# (Codex P1 2026-04-17).
LAYER2_INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"

# Thresholds
# 2026-04-09: cadence bumped to 600s, _TICKER_POOL_TIMEOUT bumped to 360s.
# Contract limit updated from 180s (60s-cadence era) to match the pool timeout
# so normal 200–265s cycles don't false-positive. Any cycle exceeding 360s
# means the pool hit its timeout — that's the genuine-hang threshold.
MAX_CYCLE_DURATION_S = 360
MIN_SUCCESS_RATE = 0.5
SIGNAL_DROP_THRESHOLD = 0.3  # >30% drop in voter count = warning
ESCALATION_THRESHOLD = 3     # consecutive warnings → CRITICAL
SELF_HEAL_COOLDOWN_S = 1800  # 30 minutes between sessions

# Layer 2 journal-activity contract (2026-04-13). Motivated by the 3-week
# silent outage 2026-03-27 → 2026-04-13 where --bare broke OAuth auth and
# every Layer 2 invocation exited 0 without writing a journal entry. The
# contract fires when triggers happen but journal stays empty — a class-
# of-failure check that would catch future silent-stall bugs, not just
# this specific auth case.
LAYER2_TRIGGER_LOOKBACK_S = 6 * 3600   # only complain if trigger was recent
# Tightened 2026-04-16 from 60m → 18m. T3 (Full) timeout is 900s (15m); the
# longest real Layer 2 invocation we ever observe finishes well under that.
# A 60m grace let three consecutive overnight auth-silent outages pass
# undetected (Apr 14–16). 18m = T3 cap + 3m slack for subprocess startup,
# Telegram delivery, and journal flush — enough that a healthy slow session
# still passes, short enough that an all-nighter of silent failures gets
# caught before markets open.
#
# Retained as a legacy single-value default; the active code path uses
# LAYER2_JOURNAL_GRACE_S_BY_TIER + _get_layer2_grace_s() below (2026-04-17).
LAYER2_JOURNAL_GRACE_S = 18 * 60       # grace period post-trigger for agent to journal

# 2026-04-17: Per-tier dynamic grace. Background: two overnight entries at
# 2026-04-16T23:11 and 2026-04-17T05:19 both showed last_invocation_status
# = "timeout" (not auth_failure). When T3 (900s = 15m) itself times out and
# respawns, the wall-clock gap trigger→journal can exceed the flat 18m
# grace because the *next* invocation has barely started. The fix is
# two-pronged: (1) derive grace from the actual tier of the most recent
# invocation (T3 needs more slack than T1), and (2) additionally suppress
# the alert while an invocation is demonstrably in flight (precondition 4).
#
# Each grace value = tier timeout + ~5min slack for subprocess startup,
# context load, first Claude round-trip, journal flush, and Telegram send.
# Default (when tier is absent from health_state) is T3 — fail-safe in the
# sense that the longer window won't mask auth failures (those still
# produce status != "invoked" entries that precondition 4 evaluates), and
# it won't delay detection past market open (US open is 13:30 UTC ≈
# 15:30 CET, so a 20m grace from an overnight trigger at 02:00 is well
# inside the 7+ hour buffer).
LAYER2_JOURNAL_GRACE_S_BY_TIER = {
    1: 3 * 60,    # T1 timeout 120s + 3m slack
    2: 12 * 60,   # T2 timeout 600s + 2m slack
    3: 20 * 60,   # T3 timeout 900s + 5m slack
}
# Default grace when health_state has no last_invocation_tier recorded.
# Uses T3 grace as the conservative choice (longer window) since silent
# auth failures still get caught via precondition 4 (which inspects the
# most recent invocation status rather than age alone).
LAYER2_JOURNAL_GRACE_S_DEFAULT = LAYER2_JOURNAL_GRACE_S_BY_TIER[3]


def _get_layer2_grace_s(health: dict | None) -> int:
    """Return the per-tier grace window in seconds.

    Reads ``last_invocation_tier`` from health_state and maps it to the
    per-tier table. Falls back to ``LAYER2_JOURNAL_GRACE_S_DEFAULT`` (T3
    grace) when the key is absent, unreadable, or not 1/2/3 — a fail-safe
    that prefers false-negatives (under-alerting) over false-positives
    during the narrow window while agent_invocation.py is populating the
    new field. See 2026-04-17 design note above.
    """
    if not health:
        return LAYER2_JOURNAL_GRACE_S_DEFAULT
    tier = health.get("last_invocation_tier")
    if not isinstance(tier, int):
        return LAYER2_JOURNAL_GRACE_S_DEFAULT
    return LAYER2_JOURNAL_GRACE_S_BY_TIER.get(tier, LAYER2_JOURNAL_GRACE_S_DEFAULT)


@dataclass
class CycleReport:
    """Populated during run() to track what actually happened this cycle."""

    cycle_id: int
    active_tickers: set = field(default_factory=set)
    signals_ok: int = 0
    signals_failed: int = 0
    signals: dict = field(default_factory=dict)
    cycle_start: float = 0.0
    cycle_end: float = 0.0
    llm_batch_flushed: bool = False
    health_updated: bool = False
    heartbeat_updated: bool = False
    summary_written: bool = False
    post_cycle_results: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    @property
    def cycle_duration_s(self) -> float:
        if self.cycle_end and self.cycle_start:
            return self.cycle_end - self.cycle_start
        return 0.0


@dataclass
class Violation:
    """A single contract invariant violation."""

    invariant: str
    severity: str  # "CRITICAL" or "WARNING"
    message: str
    details: dict = field(default_factory=dict)


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def check_layer2_journal_activity(now: datetime | None = None) -> list[Violation]:
    """Contract: if Layer 2 is enabled and a trigger fired in the last 6h,
    the Layer 2 agent must have written a journal entry within 1h of that
    trigger.

    Returns ``[]`` when the contract passes OR when preconditions don't
    apply (Layer 2 disabled, no recent trigger, missing state files).
    Never raises — all file-read failures degrade to "contract passes"
    because we cannot distinguish a healthy-but-uninstrumented system
    from a silently-broken one at this level, and false-positive alerts
    would erode trust in the contract framework.
    """
    now = now or datetime.now(UTC)

    # Precondition 1: Layer 2 must be enabled.
    cfg = load_json(CONFIG_FILE, default={}) or {}
    if not cfg.get("layer2", {}).get("enabled", True):
        return []

    # Precondition 2: a trigger must have fired recently.
    health = load_json(HEALTH_STATE_FILE)
    if not health:
        return []
    last_trigger = _parse_iso(health.get("last_trigger_time"))
    if last_trigger is None:
        return []
    trigger_age_s = (now - last_trigger).total_seconds()
    if trigger_age_s > LAYER2_TRIGGER_LOOKBACK_S or trigger_age_s < 0:
        return []

    # Precondition 3: the trigger must be old enough that the agent has
    # had its grace window to actually journal. Complaining before the
    # grace window elapses would spam on every cycle immediately after a
    # fresh trigger. 2026-04-17: grace is now per-tier — see
    # _get_layer2_grace_s() above for rationale.
    grace_s = _get_layer2_grace_s(health)
    if trigger_age_s < grace_s:
        return []

    # Precondition 4 (2026-04-17): suppress the alert while an invocation
    # is demonstrably in flight. When T3 (900s) times out and respawns,
    # the prior "timeout" entry is already written, but the NEW invocation
    # is a fresh "invoked" line that hasn't reached completion yet. The
    # journal can't possibly be written until that invocation finishes,
    # so firing the contract now would be a false-positive race. We treat
    # a most-recent "invoked" entry younger than the tier grace as "still
    # working" and wait for it to complete. Any non-"invoked" terminal
    # status (success/incomplete/failed/auth_error/timeout) falls through
    # to the real check — those are the cases the contract is meant to
    # catch.
    #
    # IMPORTANT (Codex P1 2026-04-17): use LAYER2_INVOCATIONS_FILE (the
    # Layer 2-specific log from agent_invocation), NOT the global
    # claude_invocations.jsonl. The latter is written by claude_gate for
    # unrelated callers (claude_fundamental, bigbet, iskbets, self-heal)
    # whose "invoked" entries would wrongly suppress the L2 journal alert
    # and mask genuine Layer 2 silent failures.
    latest_l2_inv = last_jsonl_entry(LAYER2_INVOCATIONS_FILE)
    if latest_l2_inv and latest_l2_inv.get("status") == "invoked":
        inv_ts = _parse_iso(
            latest_l2_inv.get("timestamp") or latest_l2_inv.get("ts")
        )
        if inv_ts is not None:
            inv_age_s = (now - inv_ts).total_seconds()
            if 0 <= inv_age_s < grace_s:
                return []

    # Precondition 4b (2026-04-18): suppress the alert when the most recent
    # L2 invocation was SKIPPED for a legitimate reason — but only for
    # skips that represent intentional non-runs, NOT for skips that hide
    # agent failures.
    #
    # skipped_offhours           — market closed, intentional non-run ✓
    # skipped_gate               — perception gate decided not to invoke ✓
    # skipped_stack_overflow     — pre-flight guard before spawning ✓
    # skipped_test               — test-mode guard (CI / unit tests) ✓
    # skipped_busy               — EXCLUDED. main.py logs skipped_busy
    #   whenever invoke_agent() returns False, including real failure
    #   paths ("couldn't kill old agent", "no agent binary"). Suppressing
    #   those would mask silent failures (Codex P1 2026-04-18).
    #
    # Only suppress for skip-statuses newer than the trigger that would
    # otherwise fire the violation. Uses a 2s tolerance on the comparison:
    # _log_trigger() and update_health() both call datetime.now()
    # independently, so the invocation ts is typically 25-40ms before
    # last_trigger_time. A strict ``inv_ts >= last_trigger`` would always
    # lose that race. 2s covers it by 50× while still rejecting stale
    # skips from prior cycles (those are ≥60s old).
    _LEGITIMATE_SKIP_STATUSES = frozenset({
        "skipped_offhours",
        "skipped_gate",
        "skipped_stack_overflow",
        "skipped_test",
    })
    if latest_l2_inv and latest_l2_inv.get("status") in _LEGITIMATE_SKIP_STATUSES:
        inv_ts = _parse_iso(
            latest_l2_inv.get("timestamp") or latest_l2_inv.get("ts")
        )
        if inv_ts is not None and inv_ts >= last_trigger - timedelta(seconds=2):
            return []

    # For violation context only (non-blocking): read the global claude
    # log to surface last_invocation_caller in the alert message. This is
    # informational — it does NOT gate the alert.
    latest_inv = last_jsonl_entry(CLAUDE_INVOCATIONS_FILE)

    # Check: journal entry since the trigger?
    latest_journal_entry = last_jsonl_entry(LAYER2_JOURNAL_FILE)
    journal_ts = None
    if latest_journal_entry:
        journal_ts = _parse_iso(
            latest_journal_entry.get("timestamp")
            or latest_journal_entry.get("ts")
        )
    if journal_ts is not None and journal_ts >= last_trigger:
        return []  # Journal was written after the trigger. Contract passes.

    # Precondition 5 (2026-04-18): violation dedup. The contract runs every
    # loop cycle once grace elapses, so without dedup the same trigger
    # fires the same violation every cycle (observed on 2026-04-17 where a
    # single XAU-USD trigger at 22:42 fired violations at 22:53, 23:03,
    # 23:13 etc — same reason, age +10m each). Track the last
    # (trigger_time, triggered_reason) we already fired for; only fire a
    # NEW violation when the trigger itself is new or when a journal entry
    # has been written since the last violation (which would reset the
    # dedup state implicitly next cycle via the earlier journal check).
    contract_state = load_json(CONTRACT_STATE_FILE, default={}) or {}
    last_fired_trigger_ts = contract_state.get("layer2_last_violation_trigger_ts")
    current_trigger_iso = health.get("last_trigger_time")
    if last_fired_trigger_ts and current_trigger_iso == last_fired_trigger_ts:
        # Same trigger already alerted on. Stay silent until a new trigger
        # fires OR until a journal entry catches up (journal_ts >= last_trigger
        # check above will return [] next cycle and reset the dedup on the
        # subsequent new trigger).
        return []

    # Violation. Try to enrich the message with whether a recent auth
    # failure was logged — it's the most common root cause, and telling
    # the user "auth_error was already recorded" saves them investigation.
    # (latest_inv was already fetched for precondition 4.)
    inv_context = {}
    if latest_inv:
        inv_context = {
            "last_invocation_status": latest_inv.get("status"),
            "last_invocation_caller": latest_inv.get("caller"),
            "last_invocation_ts": latest_inv.get("timestamp"),
        }

    violation = Violation(
        invariant="layer2_journal_activity",
        severity="CRITICAL",
        message=(
            f"Layer 2 trigger fired {trigger_age_s / 60:.0f}m ago "
            f"({health.get('last_trigger_reason', '?')}) but no journal "
            f"entry has been written since. Agent may be failing silently. "
            f"Check data/agent.log and data/critical_errors.jsonl."
        ),
        details={
            "trigger_time": health.get("last_trigger_time"),
            "trigger_age_s": round(trigger_age_s),
            "trigger_reason": health.get("last_trigger_reason"),
            "last_journal_ts": (
                latest_journal_entry.get("timestamp") or latest_journal_entry.get("ts")
                if latest_journal_entry else None
            ),
            # 2026-04-17: include effective grace + tier so postmortems can
            # tell at a glance which tier's budget was in play.
            "grace_s": grace_s,
            "last_invocation_tier": health.get("last_invocation_tier"),
            **inv_context,
        },
    )

    # Also record to critical_errors.jsonl so the CLAUDE.md STARTUP CHECK
    # surfaces this to every future Claude session. The contract's own
    # Telegram alerting is per-cycle; critical_errors is persistent until
    # explicitly resolved.
    try:
        from portfolio.claude_gate import record_critical_error
        record_critical_error(
            category="contract_violation",
            caller="layer2_journal_activity",
            message=violation.message,
            context=violation.details,
        )
    except Exception as e:
        # Never let record_critical_error failures break the contract check.
        logger.warning("record_critical_error failed in contract check: %s", e)

    # Persist the dedup marker so we don't re-fire on this same trigger.
    # Wrapped in best-effort: a failed state write reverts to today's
    # per-cycle firing (harmless noise), but won't break the contract.
    try:
        contract_state["layer2_last_violation_trigger_ts"] = current_trigger_iso
        atomic_write_json(CONTRACT_STATE_FILE, contract_state)
    except Exception as e:
        logger.warning("contract dedup state write failed: %s", e)

    return [violation]


def verify_contract(report: CycleReport, previous_signal_counts: dict | None = None) -> list[Violation]:
    """Check all loop contract invariants against a cycle report.

    Args:
        report: The completed CycleReport from this cycle.
        previous_signal_counts: Dict of ticker -> voter count from previous cycle
            (for signal stability check). None skips the check.

    Returns:
        List of Violation objects (empty = all invariants passed).
    """
    violations = []
    n_active = len(report.active_tickers)

    # 1. All tickers processed
    total_processed = report.signals_ok + report.signals_failed
    if n_active > 0 and total_processed != n_active:
        violations.append(Violation(
            invariant="all_tickers_processed",
            severity="CRITICAL",
            message=(
                f"Ticker count mismatch: processed {total_processed} "
                f"but {n_active} were active. "
                f"{n_active - total_processed} ticker(s) silently vanished."
            ),
            details={
                "active": n_active,
                "processed": total_processed,
                "ok": report.signals_ok,
                "failed": report.signals_failed,
            },
        ))

    # 2. Minimum success rate
    if n_active > 0:
        success_rate = report.signals_ok / n_active
        if success_rate < MIN_SUCCESS_RATE:
            violations.append(Violation(
                invariant="min_success_rate",
                severity="CRITICAL",
                message=(
                    f"Signal success rate {success_rate:.0%} below "
                    f"{MIN_SUCCESS_RATE:.0%} threshold. "
                    f"{report.signals_failed}/{n_active} tickers failed."
                ),
                details={
                    "success_rate": success_rate,
                    "threshold": MIN_SUCCESS_RATE,
                    "ok": report.signals_ok,
                    "failed": report.signals_failed,
                },
            ))

    # 3. Cycle duration
    duration = report.cycle_duration_s
    if duration > MAX_CYCLE_DURATION_S:
        violations.append(Violation(
            invariant="cycle_duration",
            severity="WARNING",
            message=(
                f"Cycle took {duration:.1f}s "
                f"(limit: {MAX_CYCLE_DURATION_S}s). "
                f"Something may be hanging."
            ),
            details={
                "duration_s": duration,
                "limit_s": MAX_CYCLE_DURATION_S,
            },
        ))

    # 4. LLM batch flushed
    if not report.llm_batch_flushed:
        violations.append(Violation(
            invariant="llm_batch_flushed",
            severity="WARNING",
            message="LLM batch flush did not complete. LLM signal results may be stale.",
            details={"flushed": False},
        ))

    # 5. Valid signals — each successful ticker must have action + confidence
    invalid_signals = []
    for ticker, sig in report.signals.items():
        if not isinstance(sig, dict):
            invalid_signals.append((ticker, "not a dict"))
            continue
        if "action" not in sig:
            invalid_signals.append((ticker, "missing action"))
        elif sig["action"] is None:
            invalid_signals.append((ticker, "action is None"))
        if "confidence" not in sig:
            invalid_signals.append((ticker, "missing confidence"))
    if invalid_signals:
        violations.append(Violation(
            invariant="valid_signals",
            severity="CRITICAL",
            message=(
                f"{len(invalid_signals)} ticker(s) have invalid signals: "
                f"{', '.join(f'{t}({r})' for t, r in invalid_signals[:5])}"
            ),
            details={"invalid": invalid_signals},
        ))

    # 6. Health updated
    if not report.health_updated:
        violations.append(Violation(
            invariant="health_updated",
            severity="WARNING",
            message="Health state was not updated this cycle.",
            details={"updated": False},
        ))

    # 7. Summary written
    if not report.summary_written:
        violations.append(Violation(
            invariant="summary_written",
            severity="WARNING",
            message="Agent summary was not written this cycle.",
            details={"written": False},
        ))

    # 8. Signal count stability
    if previous_signal_counts:
        dropped = []
        for ticker, sig in report.signals.items():
            prev_count = previous_signal_counts.get(ticker)
            if prev_count is None or prev_count == 0:
                continue
            extra = sig.get("extra", {}) if isinstance(sig, dict) else {}
            current_count = extra.get("active_voters", prev_count)
            drop_ratio = (prev_count - current_count) / prev_count
            if drop_ratio > SIGNAL_DROP_THRESHOLD:
                dropped.append({
                    "ticker": ticker,
                    "previous": prev_count,
                    "current": current_count,
                    "drop_pct": f"{drop_ratio:.0%}",
                })
        if dropped:
            violations.append(Violation(
                invariant="signal_count_stable",
                severity="WARNING",
                message=(
                    f"Signal voter count dropped >30% for "
                    f"{len(dropped)} ticker(s): "
                    f"{', '.join(d['ticker'] for d in dropped[:5])}"
                ),
                details={"dropped": dropped},
            ))

    # 9. Heartbeat updated
    if not report.heartbeat_updated:
        violations.append(Violation(
            invariant="heartbeat_updated",
            severity="WARNING",
            message="Heartbeat file was not updated this cycle.",
            details={"updated": False},
        ))

    # 10. Post-cycle tasks
    failed_tasks = [
        name for name, ok in report.post_cycle_results.items() if not ok
    ]
    if failed_tasks:
        violations.append(Violation(
            invariant="post_cycle_complete",
            severity="WARNING",
            message=(
                f"{len(failed_tasks)} post-cycle task(s) failed: "
                f"{', '.join(failed_tasks[:5])}"
            ),
            details={"failed_tasks": failed_tasks},
        ))

    # 11. Layer 2 journal activity — stateful file-read check, independent
    # of the cycle report. Catches the "trigger fired but agent silently
    # failed" pattern. See check_layer2_journal_activity() for details.
    violations.extend(check_layer2_journal_activity())

    # 12. Signal accuracy degradation (BUG-178/W15-W16 follow-up, 2026-04-16).
    # Stateful file-read check that compares the recent-7d accuracy across
    # signals/per-ticker/forecast/consensus to a snapshot from 7 days ago.
    # Internally hourly-throttled (replays cached violations on cycles in
    # between full re-checks so the ViolationTracker consecutive-fire count
    # is preserved — Codex P1#2). Try/except wrapped: if anything in the
    # accuracy stack is broken we'd rather miss a degradation alert than
    # take down the entire main loop's contract framework.
    violations.extend(check_signal_accuracy_degradation_safe())

    return violations


def check_signal_accuracy_degradation_safe() -> list[Violation]:
    """Wrapped accuracy degradation check that never raises.

    The contract framework calls us every cycle. If the accuracy stack
    is in a bad state (cache corruption, missing snapshot file, malformed
    JSON entry), returning [] keeps the rest of the framework working
    while a separate WARNING gets logged so the next session sees what
    happened.

    The wire to critical_errors.jsonl lives in verify_and_act (post
    ViolationTracker) so that warnings escalated to CRITICAL by the
    tracker also reach the auto-fix-agent dispatcher (Codex P2
    2026-04-28).
    """
    try:
        from portfolio.accuracy_degradation import check_degradation
        return check_degradation()
    except Exception as e:
        logger.warning("signal accuracy degradation check failed: %s", e)
        return []


def _has_unresolved_critical_entry(
    *,
    category: str,
    message_hash: str,
    ttl_s: float,
    now: float,
) -> bool:
    """Return True iff critical_errors.jsonl has an unresolved row in the
    last ``ttl_s`` seconds whose category matches and whose message
    identity hash matches.

    Used to gate the dispatch-time dedup so a resolved-then-recurring
    incident still produces a fresh row. Resolution semantics mirror
    scripts/fix_agent_dispatcher._find_unresolved: a row is resolved if
    (a) its own ``resolution`` field is non-null, or (b) a later entry
    has ``resolves_ts`` pointing at its ``ts``.

    Best-effort: any I/O or parse failure returns False so the dispatch
    path falls through to a (safer) re-write rather than silencing on
    a transient read error.
    """
    try:
        from portfolio.claude_gate import CRITICAL_ERRORS_LOG
    except Exception:
        return False
    if not CRITICAL_ERRORS_LOG.exists():
        return False
    cutoff_dt = datetime.fromtimestamp(now - ttl_s, tz=UTC)
    cutoff_iso = cutoff_dt.isoformat()

    resolved_ts: set[str] = set()
    candidates: list[dict] = []
    try:
        import json
        with open(CRITICAL_ERRORS_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except (ValueError, TypeError):
                    continue
                rts = entry.get("resolves_ts")
                if rts:
                    resolved_ts.add(rts)
                if entry.get("category") != category:
                    continue
                ts = entry.get("ts")
                if not ts or ts < cutoff_iso:
                    continue
                if entry.get("resolution") is not None:
                    continue
                if entry.get("level") != "critical":
                    continue
                candidates.append(entry)
    except Exception as e:
        logger.debug(
            "could not scan critical_errors.jsonl for resolved-row "
            "check (%s); falling through to re-write", e,
        )
        return False

    for entry in candidates:
        if entry.get("ts") in resolved_ts:
            continue
        # Re-derive identity from the stored message + invariant fields.
        # We approximate the identity by comparing messages; the trigger
        # ts disambiguator only matters for layer2_journal_activity which
        # already has a separate inline write path, so message-text
        # match is sufficient here.
        stored_hash = _hash_violation_message(entry.get("message", ""))
        if stored_hash == message_hash:
            return True
    return False


def _dispatch_critical_errors_for_degradation(
    violations: list[Violation],
    *,
    invariants: frozenset[str] = CRITICAL_ERROR_DISPATCH_INVARIANTS,
    ttl_s: float = DEFAULT_CRITICAL_ERRORS_DEDUP_TTL_S,
) -> None:
    """Append a deduplicated critical_errors.jsonl row per CRITICAL violation
    whose invariant is in ``invariants``.

    Called from verify_and_act AFTER ViolationTracker.update, so that
    WARNINGs escalated to CRITICAL by the tracker (3x consecutive) also
    reach the auto-fix-agent dispatcher (Codex P2 2026-04-28). Without
    this hook running post-tracker, a persistent low-cardinality drift
    that escalates to CRITICAL paged Telegram + self-heal but never
    reached PF-FixAgentDispatcher.

    Dedup state lives in contract_state.json under
    ``critical_error_dispatch``. Keys: (invariant, sha1(message)). On
    same-hash replay we re-emit if the prior entry is older than
    ``ttl_s`` seconds (Codex P2 2026-04-28) so the dispatcher's 24 h
    lookback always sees a fresh row for an issue that is still active.

    Layer 2 journal-activity intentionally bypasses this path because
    check_layer2_journal_activity already records its own
    critical_errors row inline on the pre-tracker violation. Routing
    layer2 here too would double-write.

    Best-effort: any I/O or import failure logs a warning and proceeds —
    the contract pipeline is the priority and a missing critical_errors
    write is recoverable on the next cycle.
    """
    critical = [
        v for v in violations
        if v.severity == "CRITICAL" and v.invariant in invariants
    ]
    if not critical:
        return

    try:
        state = load_json(CONTRACT_STATE_FILE, default={}) or {}
    except Exception as e:
        logger.warning("critical_errors dispatch state read failed: %s", e)
        state = {}
    dispatch_state = state.get("critical_error_dispatch") or {}
    now = time.time()

    state_updates: dict[str, dict] = {}
    pending_recent: dict[str, list[dict]] = {}
    for v in critical:
        msg_hash = _hash_violation_identity(v)
        prior = dispatch_state.get(v.invariant) or {}
        recent = pending_recent.get(v.invariant)
        if recent is None:
            # Trim expired entries up front so we work with a live window.
            recent = [
                r for r in _normalize_recent_hashes(prior)
                if (now - float(r.get("ts", 0) or 0)) < ttl_s
            ]
            pending_recent[v.invariant] = recent
        # Codex P2 round-4 (2026-04-28): the auto-fix-agent dispatcher
        # only acts on UNRESOLVED critical_errors rows. State-only
        # dedup will skip a same-hash recurrence inside the TTL window
        # even when the prior row was resolved — leaving the dispatcher
        # with no fresh unresolved entry. Consult the journal for an
        # unresolved match before claiming dedup.
        if (any(r.get("hash") == msg_hash for r in recent)
                and _has_unresolved_critical_entry(
                    category=v.invariant,
                    message_hash=msg_hash,
                    ttl_s=ttl_s,
                    now=now,
                )):
            continue
        try:
            from portfolio.claude_gate import record_critical_error
            # record_critical_error swallows IO errors internally and
            # returns False — Codex P2 2026-04-28. We must check the
            # boolean: claiming the dedup slot when the row didn't land
            # would silence 6 h of an unrecorded incident.
            wrote = record_critical_error(
                category=v.invariant,
                caller=v.invariant,
                message=v.message,
                context=dict(v.details or {}),
            )
        except Exception as e:
            logger.warning(
                "record_critical_error raised for %s: %s", v.invariant, e,
            )
            continue
        if not wrote:
            logger.warning(
                "record_critical_error reported failure for %s; "
                "skipping dedup-slot claim so next cycle retries",
                v.invariant,
            )
            continue
        recent.append({"hash": msg_hash, "ts": now})
        if len(recent) > MAX_RECENT_HASHES_PER_INVARIANT:
            del recent[: len(recent) - MAX_RECENT_HASHES_PER_INVARIANT]
        state_updates[v.invariant] = {
            "recent_hashes": list(recent),
            # Convenience mirrors for human inspection.
            "last_message_hash": msg_hash,
            "ts": now,
        }

    if not state_updates:
        return
    try:
        existing = load_json(CONTRACT_STATE_FILE, default={}) or {}
        merged = existing.get("critical_error_dispatch") or {}
        merged.update(state_updates)
        existing["critical_error_dispatch"] = merged
        atomic_write_json(CONTRACT_STATE_FILE, existing)
    except Exception as e:
        logger.warning("critical_errors dispatch state write failed: %s", e)


# ---------------------------------------------------------------------------
# Metals Loop Contract
# ---------------------------------------------------------------------------

METALS_MAX_CYCLE_DURATION_S = 120  # metals loop is simpler, tighter budget


@dataclass
class MetalsCycleReport:
    """Populated during the metals loop to track what happened this cycle."""

    cycle_id: int
    cycle_start: float = 0.0
    cycle_end: float = 0.0
    underlying_prices_fetched: bool = False   # Binance FAPI prices obtained
    underlying_tickers_ok: set = field(default_factory=set)  # which underlyings succeeded
    position_prices_updated: bool = False     # active warrant prices fetched
    active_positions: int = 0                 # how many active positions exist
    positions_priced: int = 0                 # how many got valid prices
    holdings_reconciled: bool = False         # holdings diff ran this cycle
    session_alive: bool = True                # Avanza session health
    stops_verified: bool = False              # stop orders in place for active positions
    probability_computed: bool = False        # probability report ran on schedule
    errors: list = field(default_factory=list)

    @property
    def cycle_duration_s(self) -> float:
        if self.cycle_end and self.cycle_start:
            return self.cycle_end - self.cycle_start
        return 0.0


def verify_metals_contract(report: MetalsCycleReport) -> list[Violation]:
    """Check metals loop invariants."""
    violations = []

    # 1. Underlying prices fetched (XAG + XAU minimum)
    required = {"XAG-USD", "XAU-USD"}
    missing = required - report.underlying_tickers_ok
    if missing:
        violations.append(Violation(
            invariant="underlying_prices_fetched",
            severity="CRITICAL",
            message=f"Missing underlying prices: {', '.join(sorted(missing))}",
            details={"missing": sorted(missing), "ok": sorted(report.underlying_tickers_ok)},
        ))

    # 2. Position prices updated
    if report.active_positions > 0 and report.positions_priced < report.active_positions:
        violations.append(Violation(
            invariant="position_prices_updated",
            severity="WARNING",
            message=(
                f"Only {report.positions_priced}/{report.active_positions} "
                f"active positions got price updates"
            ),
            details={
                "active": report.active_positions,
                "priced": report.positions_priced,
            },
        ))

    # 3. Session alive
    if not report.session_alive:
        violations.append(Violation(
            invariant="session_alive",
            severity="CRITICAL",
            message="Avanza session is dead. Trading disabled until session renewed.",
            details={"alive": False},
        ))

    # 4. Cycle duration
    duration = report.cycle_duration_s
    if duration > METALS_MAX_CYCLE_DURATION_S:
        violations.append(Violation(
            invariant="cycle_duration",
            severity="WARNING",
            message=f"Metals cycle took {duration:.1f}s (limit: {METALS_MAX_CYCLE_DURATION_S}s)",
            details={"duration_s": duration, "limit_s": METALS_MAX_CYCLE_DURATION_S},
        ))

    # 5. Stops in place
    if report.active_positions > 0 and not report.stops_verified:
        violations.append(Violation(
            invariant="stops_in_place",
            severity="WARNING",
            message="Active positions exist but stop orders not verified this cycle",
            details={"active_positions": report.active_positions},
        ))

    # 6. Holdings reconciled
    if not report.holdings_reconciled:
        violations.append(Violation(
            invariant="holdings_reconciled",
            severity="WARNING",
            message="Holdings reconciliation did not run this cycle",
            details={"reconciled": False},
        ))

    # 7. Errors recorded
    if report.errors:
        violations.append(Violation(
            invariant="no_critical_errors",
            severity="WARNING",
            message=f"{len(report.errors)} error(s) this cycle: {report.errors[0][0]}",
            details={"errors": report.errors[:5]},
        ))

    return violations


# ---------------------------------------------------------------------------
# Bot Loop Contract (GoldDigger + Elongir)
# ---------------------------------------------------------------------------

BOT_MAX_CYCLE_DURATION_S = 120
BOT_ERROR_WARNING_THRESHOLD = 2  # warn before halt threshold


@dataclass
class BotCycleReport:
    """Populated during GoldDigger/Elongir to track what happened this cycle."""

    cycle_id: int
    bot_name: str = ""                        # "golddigger" or "elongir"
    cycle_start: float = 0.0
    cycle_end: float = 0.0
    snapshot_collected: bool = False           # data snapshot obtained
    bot_step_completed: bool = False           # bot.step() returned
    action_taken: str = ""                     # "BUY", "SELL", "HOLD", ""
    session_alive: bool = True                 # Avanza session (GoldDigger only)
    consecutive_errors: int = 0               # current error streak
    max_consecutive_errors: int = 5           # halt threshold
    report_on_schedule: bool = True           # periodic reporting ran
    errors: list = field(default_factory=list)

    @property
    def cycle_duration_s(self) -> float:
        if self.cycle_end and self.cycle_start:
            return self.cycle_end - self.cycle_start
        return 0.0


def verify_bot_contract(report: BotCycleReport) -> list[Violation]:
    """Check GoldDigger/Elongir bot invariants."""
    violations = []
    name = report.bot_name or "bot"

    # 1. Bot step completed
    if not report.bot_step_completed:
        violations.append(Violation(
            invariant="bot_step_completed",
            severity="CRITICAL",
            message=f"{name}: bot.step() did not complete this cycle",
            details={"completed": False},
        ))

    # 2. Snapshot collected (Elongir) / session alive (GoldDigger)
    if not report.snapshot_collected:
        violations.append(Violation(
            invariant="snapshot_collected",
            severity="CRITICAL",
            message=f"{name}: data snapshot not collected this cycle",
            details={"collected": False},
        ))

    # 3. Session alive (GoldDigger with Playwright)
    if not report.session_alive:
        violations.append(Violation(
            invariant="session_alive",
            severity="CRITICAL",
            message=f"{name}: Avanza session is dead",
            details={"alive": False},
        ))

    # 4. Consecutive errors approaching halt
    if report.consecutive_errors >= BOT_ERROR_WARNING_THRESHOLD:
        remaining = report.max_consecutive_errors - report.consecutive_errors
        severity = "CRITICAL" if remaining <= 1 else "WARNING"
        violations.append(Violation(
            invariant="consecutive_errors",
            severity=severity,
            message=(
                f"{name}: {report.consecutive_errors} consecutive errors "
                f"({remaining} until halt)"
            ),
            details={
                "consecutive": report.consecutive_errors,
                "max": report.max_consecutive_errors,
                "remaining": remaining,
            },
        ))

    # 5. Cycle duration
    duration = report.cycle_duration_s
    if duration > BOT_MAX_CYCLE_DURATION_S:
        violations.append(Violation(
            invariant="cycle_duration",
            severity="WARNING",
            message=f"{name}: cycle took {duration:.1f}s (limit: {BOT_MAX_CYCLE_DURATION_S}s)",
            details={"duration_s": duration, "limit_s": BOT_MAX_CYCLE_DURATION_S},
        ))

    # 6. Report on schedule
    if not report.report_on_schedule:
        violations.append(Violation(
            invariant="report_on_schedule",
            severity="WARNING",
            message=f"{name}: periodic report missed schedule",
            details={"on_schedule": False},
        ))

    return violations


# ---------------------------------------------------------------------------
# Shared Infrastructure
# ---------------------------------------------------------------------------


class ViolationTracker:
    """Tracks consecutive violations per invariant for escalation.

    Persists state to CONTRACT_STATE_FILE so escalation survives restarts.
    """

    def __init__(self, state_file: Path = CONTRACT_STATE_FILE):
        self._state_file = state_file
        state = load_json(state_file, default={})
        self._consecutive: dict[str, int] = state.get("consecutive", {})
        self._last_heal_time: float = state.get("last_heal_time", 0.0)
        self._previous_signal_counts: dict[str, int] = state.get(
            "previous_signal_counts", {}
        )

    @property
    def previous_signal_counts(self) -> dict[str, int]:
        return self._previous_signal_counts

    def update(self, violations: list[Violation], report=None) -> list[Violation]:
        """Process violations: track consecutive counts, escalate warnings.

        Updates previous_signal_counts from the current report (if applicable).
        Returns the final violation list (with escalated severities).
        """
        # Track which invariants violated vs passed
        violated_names = {v.invariant for v in violations}
        # Use all known invariant names (from any loop type)
        all_invariant_names = set(self._consecutive.keys()) | violated_names

        # Reset counts for invariants that passed
        for name in all_invariant_names - violated_names:
            self._consecutive.pop(name, None)

        # Increment counts for violated invariants
        for v in violations:
            self._consecutive[v.invariant] = self._consecutive.get(v.invariant, 0) + 1

        # Escalate warnings that hit the threshold
        escalated = []
        for v in violations:
            count = self._consecutive.get(v.invariant, 0)
            if v.severity == "WARNING" and count >= ESCALATION_THRESHOLD:
                escalated.append(Violation(
                    invariant=v.invariant,
                    severity="CRITICAL",
                    message=f"ESCALATED ({count}x consecutive): {v.message}",
                    details={**v.details, "consecutive": count},
                ))
            else:
                escalated.append(v)

        # Update signal counts for next cycle's stability check (main loop only)
        if report is not None and hasattr(report, "signals"):
            self._update_signal_counts(report)
        self._save()
        return escalated

    def can_self_heal(self) -> bool:
        """Check if enough time has passed since last self-healing session."""
        return (time.time() - self._last_heal_time) > SELF_HEAL_COOLDOWN_S

    def record_heal(self):
        """Record that a self-healing session was triggered."""
        self._last_heal_time = time.time()
        self._save()

    def _update_signal_counts(self, report: CycleReport):
        """Extract per-ticker active voter counts from this cycle's signals."""
        counts = {}
        for ticker, sig in report.signals.items():
            if not isinstance(sig, dict):
                continue
            extra = sig.get("extra", {})
            if isinstance(extra, dict) and "active_voters" in extra:
                counts[ticker] = extra["active_voters"]
        if counts:
            self._previous_signal_counts = counts

    def _save(self):
        # Preserve unknown keys so other writers to CONTRACT_STATE_FILE can
        # coexist (e.g. check_layer2_journal_activity writes
        # layer2_last_violation_trigger_ts here — Codex P1 2026-04-18).
        existing = load_json(self._state_file, default={}) or {}
        existing["consecutive"] = self._consecutive
        existing["last_heal_time"] = self._last_heal_time
        existing["previous_signal_counts"] = self._previous_signal_counts
        atomic_write_json(self._state_file, existing)


_LOOP_FILES = {
    "main": "portfolio/main.py",
    "metals": "data/metals_loop.py",
    "golddigger": "portfolio/golddigger/runner.py",
    "elongir": "portfolio/elongir/runner.py",
}


def _build_heal_prompt(violations: list[Violation], loop_name: str = "main") -> str:
    """Build a diagnostic prompt for the self-healing Claude Code session."""
    loop_file = _LOOP_FILES.get(loop_name, "portfolio/main.py")
    parts = [
        f"LOOP CONTRACT VIOLATION — SELF-HEALING SESSION ({loop_name} loop)",
        "",
        f"The {loop_name} loop's runtime contract has detected critical violations.",
        "Your job: diagnose the root cause and fix it.",
        "",
        "## Violations",
    ]
    for v in violations:
        parts.append(f"- **{v.invariant}** [{v.severity}]: {v.message}")
        if v.details:
            for k, val in v.details.items():
                parts.append(f"  {k}: {val}")
        parts.append("")

    parts.extend([
        "## Instructions",
        "",
        f"1. Read `{loop_file}` — the loop that triggered this.",
        "2. Read `portfolio/loop_contract.py` — the contract that detected this.",
        "3. Check recent errors in `data/health_state.json`.",
        "4. Check `data/contract_violations.jsonl` for violation history.",
        "5. Diagnose the root cause — why did these invariants fail?",
        "6. Fix the issue. Run relevant tests to verify.",
        "7. Do NOT change signal weights, trading logic, or config.json.",
        "",
        "Keep the fix minimal and targeted. If the issue is environmental",
        "(API down, network issue), log your diagnosis but don't change code.",
    ])
    return "\n".join(parts)


def _log_violations(violations: list[Violation], cycle_id: int):
    """Append violations to the JSONL log."""
    ts = datetime.now(UTC).isoformat()
    for v in violations:
        atomic_append_jsonl(CONTRACT_LOG_FILE, {
            "ts": ts,
            "cycle_id": cycle_id,
            "invariant": v.invariant,
            "severity": v.severity,
            "message": v.message,
            "details": v.details,
        })


# Codex P1 round-3 (2026-04-28): the ViolationTracker rewrites promoted
# WARNINGs as "ESCALATED (Nx consecutive): <original>" where N
# increments every cycle. Hashing the rendered text would give a
# different hash each pass and the dedup would never engage on tracker-
# promoted incidents. Strip the prefix so the hash represents the
# underlying incident, not the cycle counter.
_ESCALATED_PREFIX_RE = re.compile(r"^ESCALATED \(\d+x consecutive\): ")


def _hash_violation_message(message: str) -> str:
    """Compatibility shim: hash a bare message string.

    Prefer ``_hash_violation_identity(violation)`` for new code; that
    helper consults per-invariant identity fields (e.g. trigger_time
    for layer2_journal_activity) so two distinct incidents with
    identical rendered text don't collide. This shim is kept for the
    handful of call sites that genuinely have only the message.
    """
    stripped = _ESCALATED_PREFIX_RE.sub("", message or "", count=1)
    return hashlib.sha1(stripped.encode("utf-8")).hexdigest()


def _hash_violation_identity(violation: "Violation") -> str:
    """Stable per-incident identity used for both Telegram cooldown dedup
    and critical_errors.jsonl dedup.

    Strips the ``ESCALATED (Nx consecutive):`` prefix added by
    ViolationTracker, so a tracker-promoted alert dedups against the
    pre-promotion form. For ``layer2_journal_activity`` the message text
    only embeds rounded age + reason — two distinct triggers with the
    same reason can render identically, so we additionally fold in
    ``details['trigger_time']`` (Codex P2 round-4 2026-04-28). Other
    invariants fall back to the message-only hash.

    SHA-1 is a content fingerprint here, not a security primitive.
    """
    msg = _ESCALATED_PREFIX_RE.sub("", violation.message or "", count=1)
    parts = [msg]
    if violation.invariant == "layer2_journal_activity":
        details = violation.details or {}
        trigger = details.get("trigger_time")
        if trigger:
            parts.append(f"trigger_time={trigger}")
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()


def _normalize_recent_hashes(prior: dict) -> list[dict]:
    """Return prior['recent_hashes'] as a list of {'hash', 'ts'} dicts.

    Backward-compat: older state dicts stored only ``last_message_hash``
    + ``last_sent_ts``. We synthesize a one-element history from those so
    the migration is invisible to live state files written before this
    patch shipped (Codex P2-3 2026-04-28).
    """
    raw = prior.get("recent_hashes")
    if isinstance(raw, list) and raw:
        out: list[dict] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            h = entry.get("hash")
            try:
                ts = float(entry.get("ts", 0) or 0)
            except (TypeError, ValueError):
                ts = 0.0
            if isinstance(h, str):
                out.append({"hash": h, "ts": ts})
        return out
    legacy_hash = prior.get("last_message_hash")
    if isinstance(legacy_hash, str):
        try:
            legacy_ts = float(prior.get("last_sent_ts", 0) or 0)
        except (TypeError, ValueError):
            legacy_ts = 0.0
        return [{"hash": legacy_hash, "ts": legacy_ts}]
    return []


def _telegram_will_actually_deliver(config: dict | None,
                                    category: str = CONTRACT_ALERT_CATEGORY) -> bool:
    """Mirror message_store.send_or_store's mute gating to predict whether
    Telegram will actually receive the alert.

    Codex P2 2026-04-28: send_or_store returns True on muted/stored paths
    too, so the cooldown can't trust its return value alone. Peeking at
    the same gates here lets the cooldown logic skip claiming a 4 h
    silence window for a message that was never delivered. Mirrors the
    logic in portfolio/message_store.py:170-219 — keep both in sync.
    """
    tg_cfg = (config or {}).get("telegram", {}) or {}
    muted = set(tg_cfg.get("muted_categories", []) or [])
    if category in muted:
        return False
    if tg_cfg.get("mute_all", False):
        unmuted = set(tg_cfg.get("unmuted_categories", []) or [])
        if category not in unmuted:
            return False
    return True


def _filter_critical_by_cooldown(critical: list[Violation], now: float,
                                 cooldown_s: float):
    """Split CRITICAL violations into (fresh, suppressed) using the per-
    invariant multi-hash cooldown stored in CONTRACT_STATE_FILE.

    A violation is *fresh* when (invariant, sha1(message)) is not present
    in the recent-hashes list for that invariant within the cooldown
    window. The recent-hashes list keeps up to
    ``MAX_RECENT_HASHES_PER_INVARIANT`` entries so an A -> B -> A flap
    inside the window dedups the second A against the original A row,
    not against the "last seen" B (Codex P2-3 2026-04-28).

    Returns the fresh-list plus a state-update dict that the caller
    should persist ONLY after a successful send — a failed Telegram post
    must not claim a dedup slot and silence the next legitimate alert.
    """
    state = load_json(CONTRACT_STATE_FILE, default={}) or {}
    cooldown_state = state.get("telegram_alert_state") or {}

    fresh: list[Violation] = []
    state_updates: dict[str, dict] = {}
    # Per-call cache so multiple violations of the same invariant see
    # each other's tentative additions.
    pending_recent: dict[str, list[dict]] = {}
    for v in critical:
        msg_hash = _hash_violation_identity(v)
        prior = cooldown_state.get(v.invariant) or {}
        recent = pending_recent.get(v.invariant)
        if recent is None:
            # Hydrate the recent-hashes list and trim expired entries up front.
            recent = [
                r for r in _normalize_recent_hashes(prior)
                if (now - float(r.get("ts", 0) or 0)) < cooldown_s
            ]
            pending_recent[v.invariant] = recent
        if any(r.get("hash") == msg_hash for r in recent):
            # Same incident identity seen within the cooldown window — drop.
            continue
        recent.append({"hash": msg_hash, "ts": now})
        # Cap memory; oldest entries fall off first.
        if len(recent) > MAX_RECENT_HASHES_PER_INVARIANT:
            del recent[: len(recent) - MAX_RECENT_HASHES_PER_INVARIANT]
        fresh.append(v)
        state_updates[v.invariant] = {
            "recent_hashes": list(recent),
            # Convenience mirrors for human inspection / debugging tools.
            "last_sent_ts": now,
            "last_message_hash": msg_hash,
        }
    return fresh, state_updates


def _persist_alert_cooldown(state_updates: dict[str, dict]) -> None:
    """Merge state_updates into contract_state.json's telegram_alert_state.

    Best-effort. A failure to persist means the next cycle re-fires the
    same alert; harmless noise compared to crashing the contract pipeline.
    Preserves all unrelated keys (consecutive, last_heal_time, etc.) so
    other writers to CONTRACT_STATE_FILE can coexist — same pattern
    ViolationTracker._save() and check_layer2_journal_activity follow.
    """
    if not state_updates:
        return
    try:
        existing = load_json(CONTRACT_STATE_FILE, default={}) or {}
        cooldown_state = existing.get("telegram_alert_state") or {}
        cooldown_state.update(state_updates)
        existing["telegram_alert_state"] = cooldown_state
        atomic_write_json(CONTRACT_STATE_FILE, existing)
    except Exception as e:
        logger.warning("alert cooldown state write failed: %s", e)


def _alert_violations(violations: list[Violation], config: dict,
                      loop_name: str = "main"):
    """Send Telegram alert for critical violations.

    Per-invariant cooldown (2026-04-28): each (invariant, message_hash)
    fires at most once per cooldown window. See
    DEFAULT_CONTRACT_ALERT_COOLDOWN_S above for rationale. Cooldown logic
    fails open — if it raises, we ship the alert anyway because Telegram
    silence on a real CRITICAL is worse than duplicate noise.
    """
    critical = [v for v in violations if v.severity == "CRITICAL"]
    if not critical:
        return

    cooldown_s = float(
        (config or {}).get("notification", {}).get(
            "contract_alert_cooldown_s",
            DEFAULT_CONTRACT_ALERT_COOLDOWN_S,
        )
    )
    try:
        fresh, state_updates = _filter_critical_by_cooldown(
            critical, time.time(), cooldown_s,
        )
    except Exception as e:
        # Fail-open. Emit the warning so the next operator sees what
        # happened, but never let cooldown bookkeeping silence a real
        # alert.
        logger.warning("alert cooldown filter failed, fail-open: %s", e)
        fresh = critical
        state_updates = {}

    if not fresh:
        return

    lines = [f"*LOOP CONTRACT ({loop_name})* — {len(fresh)} critical violation(s)"]
    for v in fresh:
        lines.append(f"• {v.invariant}: {v.message}")
    msg = "\n".join(lines)
    # Pre-compute whether send_or_store WILL actually deliver, before we
    # call it. The mute gates (muted_categories, mute_all without unmute)
    # cause send_or_store to return True after only storing the message
    # locally — claiming the cooldown for those would silence 4 h of
    # legitimate alerts at the moment the user unmutes (Codex P2-2
    # 2026-04-28).
    will_deliver = _telegram_will_actually_deliver(
        config, category=CONTRACT_ALERT_CATEGORY,
    )

    try:
        from portfolio.message_store import send_or_store
        # Codex P1 2026-04-28: send_or_store returns True on success
        # (sent or intentionally stored) and False on actual delivery
        # failure (missing token/chat_id, non-OK sendMessage response).
        # Honor the boolean — claiming the cooldown after a False return
        # would silence 4 h of legitimate alerts that no operator was
        # paged for.
        sent_ok = send_or_store(msg, config, category=CONTRACT_ALERT_CATEGORY)
    except Exception as e:
        logger.warning("Failed to send contract violation alert: %s", e)
        return

    if not sent_ok:
        logger.warning(
            "send_or_store returned falsy for contract alert; skipping "
            "cooldown persist so next cycle retries (Codex P1 2026-04-28)",
        )
        return

    if not will_deliver:
        logger.debug(
            "Contract alert was muted (category=%s); not persisting "
            "cooldown so next cycle retries after an unmute "
            "(Codex P2-2 2026-04-28)", CONTRACT_ALERT_CATEGORY,
        )
        return

    _persist_alert_cooldown(state_updates)


def _trigger_self_heal(violations: list[Violation], tracker: ViolationTracker,
                       loop_name: str = "main"):
    """Spawn a Claude Code session to diagnose and fix critical violations."""
    if not tracker.can_self_heal():
        logger.info(
            "Self-heal cooldown active, skipping (last: %.0fs ago)",
            time.time() - tracker._last_heal_time,
        )
        return

    critical = [v for v in violations if v.severity == "CRITICAL"]
    if not critical:
        return

    try:
        from portfolio.claude_gate import invoke_claude
        prompt = _build_heal_prompt(critical, loop_name=loop_name)
        logger.info(
            "Triggering self-healing session for %d critical violation(s) [%s]",
            len(critical), loop_name,
        )
        tracker.record_heal()
        # C1: Self-heal is read-only by default — diagnostic only.
        # Previously granted Edit+Bash+Write which allowed unreviewed
        # code modification during critical failures (worst time for
        # autonomous changes). Now restricted to read-only tools.
        invoke_claude(
            prompt=prompt,
            caller=f"loop_contract_{loop_name}",
            model="sonnet",
            max_turns=15,
            timeout=180,
            allowed_tools="Read,Grep,Glob",
        )
    except Exception as e:
        logger.warning("Self-healing invocation failed: %s", e)


def verify_and_act(report, config: dict,
                   tracker: ViolationTracker | None = None,
                   verify_fn=None, loop_name: str = "main"):
    """Full contract verification pipeline: check → track → log → alert → heal.

    Works with any report type (CycleReport, MetalsCycleReport, BotCycleReport).

    Args:
        report: A cycle report with at least cycle_id and signals attributes.
        config: Application config dict (for Telegram alerts).
        tracker: Optional ViolationTracker (created with defaults if None).
        verify_fn: Custom verification function. If None, uses verify_contract
            for main loop (requires CycleReport).
        loop_name: Identifier for this loop ("main", "metals", "golddigger", "elongir").
    """
    if tracker is None:
        tracker = ViolationTracker()

    # Run contract checks
    if verify_fn is not None:
        violations = verify_fn(report)
    else:
        violations = verify_contract(
            report,
            previous_signal_counts=tracker.previous_signal_counts,
        )

    if not violations:
        # All good — still update tracker to clear consecutive counts and save
        tracker.update([], report)
        return

    # Track consecutive counts and escalate
    violations = tracker.update(violations, report)

    # 2026-04-28 (Codex P2): wire dispatcher-tracked invariants into
    # critical_errors.jsonl AFTER the tracker has had its chance to
    # promote consecutive WARNINGs to CRITICAL. Doing it pre-tracker
    # would miss exactly those promoted cases. Layer 2 still writes its
    # own row inline because the layer2 check needs the trigger ts in
    # the resolution context.
    _dispatch_critical_errors_for_degradation(violations)

    # Log all violations
    _log_violations(violations, report.cycle_id)

    # Alert critical violations via Telegram
    _alert_violations(violations, config, loop_name=loop_name)

    # Self-heal on critical violations
    critical = [v for v in violations if v.severity == "CRITICAL"]
    if critical:
        _trigger_self_heal(violations, tracker, loop_name=loop_name)

    logger.warning(
        "Contract violations [%s]: %d total (%d critical) — cycle %d",
        loop_name, len(violations), len(critical), report.cycle_id,
    )
