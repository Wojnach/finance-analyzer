"""Loop Contract — runtime invariant verification for all system loops.

After every cycle, verify functions check that critical operations
actually happened. Violations are logged, alerted, and optionally
trigger a self-healing Claude Code session.

Supports: main loop, metals loop, GoldDigger, Elongir.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CONTRACT_STATE_FILE = DATA_DIR / "contract_state.json"
CONTRACT_LOG_FILE = DATA_DIR / "contract_violations.jsonl"
CONFIG_FILE = BASE_DIR / "config.json"
HEALTH_STATE_FILE = DATA_DIR / "health_state.json"
LAYER2_JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
CLAUDE_INVOCATIONS_FILE = DATA_DIR / "claude_invocations.jsonl"

# Thresholds
MAX_CYCLE_DURATION_S = 180
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
LAYER2_JOURNAL_GRACE_S = 60 * 60       # grace period post-trigger for agent to journal


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


def _read_json(path: Path) -> dict | None:
    """Best-effort JSON read. Returns None on any error — contract checks
    must never fail noisily or block the loop."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _last_jsonl_entry(path: Path) -> dict | None:
    """Return the last parseable JSON line from a JSONL file, or None.
    Reads the whole file — acceptable because these journals are small
    (hundreds of KB at most) and the check runs at cycle cadence (60s)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            last = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue
            return last
    except OSError:
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
    cfg = _read_json(CONFIG_FILE) or {}
    if not cfg.get("layer2", {}).get("enabled", True):
        return []

    # Precondition 2: a trigger must have fired recently.
    health = _read_json(HEALTH_STATE_FILE)
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
    # fresh trigger.
    if trigger_age_s < LAYER2_JOURNAL_GRACE_S:
        return []

    # Check: journal entry since the trigger?
    latest_journal_entry = _last_jsonl_entry(LAYER2_JOURNAL_FILE)
    journal_ts = None
    if latest_journal_entry:
        journal_ts = _parse_iso(
            latest_journal_entry.get("timestamp")
            or latest_journal_entry.get("ts")
        )
    if journal_ts is not None and journal_ts >= last_trigger:
        return []  # Journal was written after the trigger. Contract passes.

    # Violation. Try to enrich the message with whether a recent auth
    # failure was logged — it's the most common root cause, and telling
    # the user "auth_error was already recorded" saves them investigation.
    latest_inv = _last_jsonl_entry(CLAUDE_INVOCATIONS_FILE)
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

    return violations


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
        from portfolio.file_utils import load_json
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
        from portfolio.file_utils import atomic_write_json
        atomic_write_json(self._state_file, {
            "consecutive": self._consecutive,
            "last_heal_time": self._last_heal_time,
            "previous_signal_counts": self._previous_signal_counts,
        })


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
    from portfolio.file_utils import atomic_append_jsonl
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


def _alert_violations(violations: list[Violation], config: dict,
                      loop_name: str = "main"):
    """Send Telegram alert for critical violations."""
    critical = [v for v in violations if v.severity == "CRITICAL"]
    if not critical:
        return
    lines = [f"*LOOP CONTRACT ({loop_name})* — {len(critical)} critical violation(s)"]
    for v in critical:
        lines.append(f"• {v.invariant}: {v.message}")
    msg = "\n".join(lines)
    try:
        from portfolio.message_store import send_or_store
        send_or_store(msg, config, category="error")
    except Exception as e:
        logger.warning("Failed to send contract violation alert: %s", e)


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
