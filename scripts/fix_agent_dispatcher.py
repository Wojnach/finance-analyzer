"""Auto-spawn a Claude Code fix agent when critical errors accumulate.

Runs as a separate scheduled task (``PF-FixAgentDispatcher``, every 10m)
rather than inline in the main loop — failure of the dispatcher must never
destabilise trading.

Source of truth: ``data/critical_errors.jsonl``. The dispatcher reads
unresolved entries, respects cooldown and kill-switch, and invokes
``portfolio.claude_gate.invoke_claude`` with a fix-agent prompt. The
agent is instructed to append resolution lines back into the journal;
the next dispatcher run sees the resolution and stops re-firing.

Design docs: ``docs/plans/2026-04-13-auto-spawn-fix-agent.md``.

Exits 0 on every healthy code path — non-zero only on unexpected errors
(so the scheduled task's "last result" surfaces real breakage, not
routine "nothing to do" runs).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Ensure Q:\finance-analyzer is importable when invoked from a scheduled task
# (which may cwd elsewhere). This mirrors the pattern in other scripts/.
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
CRITICAL_ERRORS_LOG = DATA_DIR / "critical_errors.jsonl"
STATE_FILE = DATA_DIR / "fix_agent_state.json"
KILL_SWITCH = DATA_DIR / "fix_agent.disabled"

# --- Tunables ---
# Same constant the loop contract uses so behavior is coherent.
SELF_HEAL_COOLDOWN_S = 1800  # 30 min between attempts per category
# Exponential backoff on consecutive failures.
BACKOFF_SCHEDULE_S = [1800, 7200, 43200]  # 30m → 2h → 12h, then disabled
# Recursion guard: dispatcher must refuse to fire if invoked from within
# another fix-agent subprocess (env flag propagates to child Claude).
RECURSION_ENV = "PF_FIX_AGENT_DEPTH"
MAX_RECURSION_DEPTH = 1
# Look-back window for unresolved errors (24h). Older issues are stale.
LOOKBACK_H = 24
# Per-attempt budget (seconds, Opus 30 turns is typically <15 min).
AGENT_TIMEOUT_S = 900
AGENT_MAX_TURNS = 30
AGENT_MODEL = "opus"
AGENT_ALLOWED_TOOLS = "Read,Edit,Bash"

logger = logging.getLogger("fix_agent_dispatcher")


# ---------------------------------------------------------------------------
# Journal I/O — tolerant of malformed / missing files (never raises)
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(UTC)


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _read_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {"by_category": {}, "recursion_counter": 0}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"by_category": {}, "recursion_counter": 0}


def _save_state(state: dict) -> None:
    """Atomic state-file write — tmp + rename. A mid-write crash must
    never leave a corrupt JSON that would break the next dispatcher run."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    # os.replace is atomic on Windows + POSIX when src/dest are on the same volume.
    os.replace(tmp, STATE_FILE)


def _find_unresolved(entries: list[dict], lookback_h: int) -> list[dict]:
    """Return unresolved entries from the last `lookback_h` hours.

    An entry is resolved if (a) it has a non-null ``resolution`` field, or
    (b) a later entry has ``resolves_ts`` pointing at its ``ts``.
    """
    cutoff = _now() - timedelta(hours=lookback_h)
    resolved_ts: set[str] = set()
    for e in entries:
        if e.get("resolves_ts"):
            resolved_ts.add(e["resolves_ts"])

    unresolved = []
    for e in entries:
        # Only treat "critical" level entries as actionable; skip info/resolution lines
        if e.get("level") != "critical":
            continue
        if e.get("resolution") is not None:
            continue
        if e.get("ts") in resolved_ts:
            continue
        parsed = _parse_iso(e.get("ts"))
        if parsed is None or parsed < cutoff:
            continue
        unresolved.append(e)
    return unresolved


def _append_critical(entry: dict) -> None:
    """Append a record to critical_errors.jsonl.

    Uses ``portfolio.file_utils.atomic_append_jsonl`` when available —
    the main loop's claude_gate.record_critical_error ALSO writes to this
    file concurrently, and plain ``open("a")`` can interleave mid-line on
    Windows NTFS, corrupting the JSONL. We fall back to the simple append
    only if the package can't be imported (e.g. running as a standalone
    script from outside the repo).
    """
    CRITICAL_ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        from portfolio.file_utils import atomic_append_jsonl
        atomic_append_jsonl(CRITICAL_ERRORS_LOG, entry)
    except ImportError:
        with open(CRITICAL_ERRORS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Gating — kill switch, cooldown, recursion
# ---------------------------------------------------------------------------

@dataclass
class GateDecision:
    allowed: bool
    reason: str  # machine-readable short tag, e.g. "cooldown" / "ok"


def check_gates(category: str, state: dict, now: datetime | None = None) -> GateDecision:
    """Decide whether a fix attempt for *category* is permitted right now."""
    now = now or _now()

    if KILL_SWITCH.exists():
        return GateDecision(False, "disabled_by_kill_switch")

    depth = int(os.environ.get(RECURSION_ENV, "0") or "0")
    if depth >= MAX_RECURSION_DEPTH:
        return GateDecision(False, "recursion_depth_exceeded")

    cat_state = state.get("by_category", {}).get(category, {})
    blocked_until = _parse_iso(cat_state.get("blocked_until"))
    if blocked_until and blocked_until > now:
        return GateDecision(False, "cooldown")

    return GateDecision(True, "ok")


def update_state_after_attempt(
    state: dict, category: str, success: bool, now: datetime | None = None,
) -> dict:
    """Bump cooldown + consecutive_failures for the category."""
    now = now or _now()
    cats = state.setdefault("by_category", {})
    entry = cats.setdefault(category, {"consecutive_failures": 0})

    entry["last_attempt_ts"] = now.isoformat()
    entry["last_attempt_success"] = success

    if success:
        entry["consecutive_failures"] = 0
        entry["blocked_until"] = (now + timedelta(seconds=SELF_HEAL_COOLDOWN_S)).isoformat()
    else:
        prev = entry.get("consecutive_failures", 0)
        new_count = prev + 1
        entry["consecutive_failures"] = new_count
        idx = min(new_count - 1, len(BACKOFF_SCHEDULE_S) - 1)
        if new_count > len(BACKOFF_SCHEDULE_S):
            # Beyond the schedule: effectively disabled for 10 years. User
            # must manually reset by editing state file or adding a
            # resolution line.
            entry["blocked_until"] = (now + timedelta(days=3650)).isoformat()
        else:
            entry["blocked_until"] = (now + timedelta(seconds=BACKOFF_SCHEDULE_S[idx])).isoformat()
    return state


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_fix_prompt(category: str, entries: list[dict]) -> str:
    """Context-complete prompt for the fix agent — the agent runs in a
    fresh conversation and has no memory of how the errors were recorded."""
    bullet_points = []
    for e in entries[:10]:  # cap at 10 to bound prompt size
        bullet_points.append(
            f"- [{e.get('ts','?')}] {e.get('category','?')} "
            f"caller={e.get('caller','?')} :: {e.get('message','')}"
        )
    bullets = "\n".join(bullet_points) or "(no details available)"

    return (
        "You are the Layer 2 fix agent for finance-analyzer. A critical error "
        "was recorded in data/critical_errors.jsonl and has not been "
        "auto-resolved. Your job is to diagnose and either fix it or document "
        "why it requires human attention.\n\n"
        f"## Unresolved critical errors (category: {category})\n\n"
        f"{bullets}\n\n"
        "## Your instructions\n\n"
        "1. Read CLAUDE.md and any source files referenced by the error messages.\n"
        "2. Identify the root cause.\n"
        "3. Either:\n"
        "   a. Make the fix directly using Edit (preferred for simple regressions).\n"
        "   b. Write a fix proposal to data/proposed_fixes/<timestamp>.md when the\n"
        "      fix is risky, out of scope, or requires user decisions.\n"
        "4. When done, append a resolution line to data/critical_errors.jsonl:\n"
        '   {"ts":"<ISO UTC now>","level":"info","category":"resolution",\n'
        '    "caller":"fix_agent","resolves_ts":"<original ts>",\n'
        '    "resolution":"<short description>","message":"<details>","context":{}}\n\n'
        "DO NOT:\n"
        "- Modify files outside portfolio/, scripts/, tests/, docs/.\n"
        "- Kill processes or restart any loop.\n"
        "- Edit config.json, .env, or anything in ~/.claude.\n"
        "- Commit or push.\n\n"
        "If you cannot safely fix it, still append a resolution line explaining\n"
        "what you investigated and why the fix requires human action — this\n"
        "stops the dispatcher re-firing on the same error.\n"
    )


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def run(
    dry_run: bool = False,
    lookback_h: int = LOOKBACK_H,
    invoke_claude_fn=None,
) -> int:
    """Dispatcher entry point. Returns 0 on success (including no-op).

    ``invoke_claude_fn`` is dependency-injected for tests; production
    passes None and we import at call time.
    """
    entries = _read_journal(CRITICAL_ERRORS_LOG)
    unresolved = _find_unresolved(entries, lookback_h)
    if not unresolved:
        logger.info("No unresolved critical errors — exiting")
        return 0

    # Group by category so one agent handles related entries
    by_category: dict[str, list[dict]] = defaultdict(list)
    for e in unresolved:
        by_category[e.get("category", "unknown")].append(e)

    state = _load_state()
    any_spawned = False

    # Snapshot the recursion depth ONCE at startup. Each category's spawn
    # temporarily bumps os.environ[RECURSION_ENV] so the child Claude
    # subprocess inherits depth+1, but we reset afterwards so successive
    # categories in the same dispatcher run are treated as siblings, not
    # recursion.
    caller_recursion_depth = int(os.environ.get(RECURSION_ENV, "0") or "0")

    for category, cat_entries in by_category.items():
        decision = check_gates(category, state)
        if not decision.allowed:
            logger.info("Skipping category=%s (%s)", category, decision.reason)
            _append_critical({
                "ts": _now().isoformat(),
                "level": "info",
                "category": "fix_attempt_skipped",
                "caller": "fix_agent_dispatcher",
                "resolution": None,
                "message": f"Skipped fix attempt for {category}: {decision.reason}",
                "context": {"skipped_category": category, "reason": decision.reason,
                            "unresolved_count": len(cat_entries)},
            })
            continue

        _append_critical({
            "ts": _now().isoformat(),
            "level": "info",
            "category": "fix_attempt_started",
            "caller": "fix_agent_dispatcher",
            "resolution": None,
            "message": f"Spawning fix agent for category={category}",
            "context": {"target_category": category, "entry_count": len(cat_entries),
                        "dry_run": dry_run},
        })

        if dry_run:
            logger.info("DRY RUN — would spawn fix agent for category=%s", category)
            any_spawned = True
            continue

        prompt = build_fix_prompt(category, cat_entries)

        if invoke_claude_fn is None:
            from portfolio.claude_gate import invoke_claude as invoke_claude_fn  # type: ignore

        # Set the recursion env flag only for the duration of the invoke_claude
        # call so the child Claude subprocess inherits depth+1 (blocking any
        # transitive dispatcher re-entry). Restore afterwards so subsequent
        # categories in this same run aren't mistaken for recursion.
        prior_env = os.environ.get(RECURSION_ENV)
        os.environ[RECURSION_ENV] = str(caller_recursion_depth + 1)
        try:
            success, exit_code = invoke_claude_fn(
                prompt=prompt,
                caller=f"fix_agent_{category}",
                model=AGENT_MODEL,
                max_turns=AGENT_MAX_TURNS,
                allowed_tools=AGENT_ALLOWED_TOOLS,
                timeout=AGENT_TIMEOUT_S,
            )
        except Exception as e:
            logger.exception("Fix agent invocation raised: %s", e)
            success, exit_code = False, -1
        finally:
            if prior_env is None:
                os.environ.pop(RECURSION_ENV, None)
            else:
                os.environ[RECURSION_ENV] = prior_env

        any_spawned = True
        state = update_state_after_attempt(state, category, success)
        _append_critical({
            "ts": _now().isoformat(),
            "level": "info" if success else "critical",
            "category": "fix_attempt_completed" if success else "fix_agent_failed",
            "caller": "fix_agent_dispatcher",
            "resolution": None,  # agent will write its own resolution line
            "message": (
                f"Fix agent for {category} {'succeeded' if success else 'FAILED'} "
                f"(exit={exit_code}). Check journal for the agent's resolution line."
            ),
            "context": {"target_category": category, "success": success,
                        "exit_code": exit_code,
                        "consecutive_failures":
                            state["by_category"][category]["consecutive_failures"]},
        })

    _save_state(state)
    logger.info("Dispatcher run complete (spawned=%s)", any_spawned)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually spawn the agent — just log what would happen")
    parser.add_argument("--lookback-h", type=int, default=LOOKBACK_H,
                        help=f"Hours of critical_errors.jsonl to inspect (default {LOOKBACK_H})")
    args = parser.parse_args(argv)

    try:
        return run(dry_run=args.dry_run, lookback_h=args.lookback_h)
    except Exception as e:
        logger.exception("Dispatcher crashed: %s", e)
        # Leave an explicit marker so the user knows the dispatcher itself
        # broke — not just the underlying fix attempts.
        try:
            _append_critical({
                "ts": _now().isoformat(),
                "level": "critical",
                "category": "fix_agent_dispatcher_crashed",
                "caller": "fix_agent_dispatcher",
                "resolution": None,
                "message": f"Dispatcher crashed: {e}",
                "context": {"exception_type": type(e).__name__},
            })
        except Exception:
            pass
        return 2


if __name__ == "__main__":
    sys.exit(main())
