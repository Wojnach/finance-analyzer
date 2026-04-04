"""Loop Contract — runtime invariant verification for the main loop.

After every cycle, verify_contract() checks that critical operations
actually happened. Violations are logged, alerted, and optionally
trigger a self-healing Claude Code session.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CONTRACT_STATE_FILE = DATA_DIR / "contract_state.json"
CONTRACT_LOG_FILE = DATA_DIR / "contract_violations.jsonl"

# Thresholds
MAX_CYCLE_DURATION_S = 180
MIN_SUCCESS_RATE = 0.5
SIGNAL_DROP_THRESHOLD = 0.3  # >30% drop in voter count = warning
ESCALATION_THRESHOLD = 3     # consecutive warnings → CRITICAL
SELF_HEAL_COOLDOWN_S = 1800  # 30 minutes between sessions


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

    return violations


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

    def update(self, violations: list[Violation], report: CycleReport) -> list[Violation]:
        """Process violations: track consecutive counts, escalate warnings.

        Updates previous_signal_counts from the current report.
        Returns the final violation list (with escalated severities).
        """
        # Track which invariants passed this cycle
        violated_names = {v.invariant for v in violations}
        all_invariant_names = {
            "all_tickers_processed", "min_success_rate", "cycle_duration",
            "llm_batch_flushed", "valid_signals", "health_updated",
            "summary_written", "signal_count_stable", "heartbeat_updated",
            "post_cycle_complete",
        }

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

        # Update signal counts for next cycle's stability check
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


def _build_heal_prompt(violations: list[Violation]) -> str:
    """Build a diagnostic prompt for the self-healing Claude Code session."""
    parts = [
        "LOOP CONTRACT VIOLATION — SELF-HEALING SESSION",
        "",
        "The main loop's runtime contract has detected critical violations.",
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
        "1. Read `portfolio/main.py` — the main loop and `run()` function.",
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


def _alert_violations(violations: list[Violation], config: dict):
    """Send Telegram alert for critical violations."""
    critical = [v for v in violations if v.severity == "CRITICAL"]
    if not critical:
        return
    lines = [f"*LOOP CONTRACT* — {len(critical)} critical violation(s)"]
    for v in critical:
        lines.append(f"• {v.invariant}: {v.message}")
    msg = "\n".join(lines)
    try:
        from portfolio.message_store import send_or_store
        send_or_store(msg, config, category="error")
    except Exception as e:
        logger.warning("Failed to send contract violation alert: %s", e)


def _trigger_self_heal(violations: list[Violation], tracker: ViolationTracker):
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
        prompt = _build_heal_prompt(critical)
        logger.info(
            "Triggering self-healing session for %d critical violation(s)",
            len(critical),
        )
        tracker.record_heal()
        invoke_claude(
            prompt=prompt,
            caller="loop_contract",
            model="sonnet",
            max_turns=15,
            timeout=180,
        )
    except Exception as e:
        logger.warning("Self-healing invocation failed: %s", e)


def verify_and_act(report: CycleReport, config: dict,
                   tracker: ViolationTracker | None = None):
    """Full contract verification pipeline: check → track → log → alert → heal.

    This is the single entry point called from the main loop after each cycle.
    """
    if tracker is None:
        tracker = ViolationTracker()

    # Run contract checks
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
    _alert_violations(violations, config)

    # Self-heal on critical violations
    critical = [v for v in violations if v.severity == "CRITICAL"]
    if critical:
        _trigger_self_heal(violations, tracker)

    logger.warning(
        "Contract violations: %d total (%d critical) — cycle %d",
        len(violations),
        len(critical),
        report.cycle_id,
    )
