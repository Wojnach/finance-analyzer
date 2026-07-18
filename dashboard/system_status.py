"""System-health rollup for the dashboard home page.

Pure aggregator. Reads the same jsonl/json files the rest of the system
already writes — adds no new instrumentation. Returns a single payload
shape that the home view consumes via ``/api/system_status``.

Why this exists: the previous home page led with simulated-portfolio
P&L, which the user explicitly deprioritised in favour of "is the
system actually working" indicators. See
``docs/PLAN.md`` 2026-05-04 entry and the plan file
``/root/.claude/plans/merry-tinkering-cake.md`` for the full design.

Side-effects: never invokes Avanza, never writes a file. One deliberate
exception to the old "no network socket" rule (2026-07-18): the voters
section calls llama_server.remote_llm_available(), which health-pings the
herc2 llama-server (3s connect timeout, 60s process-level cache) — the
dashboard must report the LIVE remote-LLM gate state, not a stale proxy.
Still safe to call from a 30s polling loop.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import re
import subprocess
import time

from portfolio.file_utils import load_json, load_jsonl, load_jsonl_tail
from portfolio.loop_contract import violation_identity_payload

# Mirror loop_contract._ESCALATED_PREFIX_RE so the dashboard strips the
# tracker's "ESCALATED (Nx consecutive): " prefix before computing the
# identity payload — without this, a tracker-promoted CV row hashes
# differently from its critical_errors counterpart and the cross-stream
# resolution check fails to match. (Claude review of a85a646f, P1-2.)
_ESCALATED_PREFIX_RE = re.compile(r"^ESCALATED \(\d+x consecutive\): ")

# Repo data dir. Resolved relative to this file so the module works in
# both the main checkout and a worktree without further config.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# config.json lives at repo root, a sibling of ``data/`` — NOT inside it,
# so it can't be derived from a caller-supplied ``dd`` the way every other
# section's files are (``dd`` is a test tmp_path in most callers, and its
# parent is an untrusted shared pytest tmp root, not the repo). Fixed like
# DATA_DIR; tests override via _avanza_status's ``config_path`` kwarg.
_CONFIG_FILE = DATA_DIR.parent / "config.json"

# Severity thresholds — tune freely, no schema migration needed.
HEARTBEAT_GREEN_S = 120
HEARTBEAT_YELLOW_S = 600
ERRORS_YELLOW_MAX = 3
# Lookback window for counting unresolved criticals on the home hero. Matches
# scripts/check_critical_errors.py DEFAULT_DAYS so the dashboard and the CLI
# gate agree on what "unresolved" means. 2026-06-06: before this, the dashboard
# scanned all-time with no auto-resolve and pinned the hero RED on ancient
# quiescent categories (4 May-2026 avanza session-expiry alerts the 7d gate
# already ignored). See docs/SESSION_PROGRESS.md 2026-06-06.
CRITICAL_ERRORS_LOOKBACK_DAYS = 7
LLM_GREEN_PCT = 95.0
LLM_YELLOW_PCT = 80.0
LAYER2_GREEN_PCT = 85.0
LAYER2_YELLOW_PCT = 60.0
LAYER2_MIN_TRIGGERS_FOR_GATE = 3

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def compute(data_dir: Path | None = None) -> dict[str, Any]:
    """Build the full system_status payload.

    Each section catches its own exceptions and surfaces an ``error``
    string in-band so a single bad jsonl line doesn't blank the hero.
    Mirrors the per-section error envelope used by ``/api/avanza_account``.
    """
    dd = Path(data_dir) if data_dir else DATA_DIR
    out: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "heartbeat": _heartbeat(dd),
        "errors": _errors_unresolved(dd),
        "contract_violations": _violations_recent(dd),
        "llm_inference": _llm_inference(dd),
        "layer2": _layer2_24h(dd),
        "signal_aggregate": _signal_aggregate(dd),
        "pnl_footer": _pnl_footer(dd),
        "sources": _sources(dd),
        "layer1": _layer1(dd),
        "voters": _voters(dd),
        "avanza": _avanza_status(dd),
    }
    # 2026-06-06: attach Claude-gate state into the layer2 section so the home
    # page shows when Layer 2 / Claude trading is intentionally FROZEN
    # (token-saving) instead of looking like a silent outage. Nested under
    # layer2 (not a top-level key) so the existing layer2-activity-card renders
    # the badge with no home.js rewiring. See _claude_gate + SESSION_PROGRESS.
    if isinstance(out.get("layer2"), dict):
        out["layer2"]["gate"] = _claude_gate(dd)
    if isinstance(out.get("avanza"), dict) and isinstance(out.get("errors"), dict):
        out["avanza"]["unresolved_errors"] = out["errors"].get("avanza_unresolved", 0)
    overall, reasons = _color(out)
    out["overall"] = overall
    out["reasons"] = reasons
    return out


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _claude_gate(dd: Path) -> dict[str, Any]:
    """Layer 2 / Claude invocation-gate state (ACTIVE vs FROZEN).

    Surfaces the three independent kill switches so the home page shows when
    Claude trading is *intentionally* frozen (token-saving) rather than
    looking like a silent outage:
      - config.json layer2.enabled            — runtime gate for the
        crypto/MSTR Layer 2 agent + every portfolio.claude_gate caller
      - portfolio.claude_gate.CLAUDE_ENABLED  — master module switch
      - data/metals_loop.py CLAUDE_ENABLED    — metals loop claude_proc spawn

    ``enabled`` is False if ANY switch is off (label FROZEN), True only when
    none are off (label ACTIVE), and None/UNKNOWN if nothing could be read.
    See docs/SESSION_PROGRESS.md 2026-06-06 (token-conservation freeze).
    """
    out: dict[str, Any] = {
        "config_layer2_enabled": None,
        "claude_gate_enabled": None,
        "metals_claude_enabled": None,
        "enabled": None,
        "label": "UNKNOWN",
    }
    try:
        import portfolio.claude_gate as _cg

        out["claude_gate_enabled"] = bool(_cg.CLAUDE_ENABLED)
        try:
            out["config_layer2_enabled"] = bool(_cg._load_config_layer2_enabled())
        except Exception:
            pass
    except Exception as e:  # pragma: no cover - import guard
        out["error"] = f"claude_gate: {type(e).__name__}: {e}"

    out["metals_claude_enabled"] = _parse_metals_claude_enabled(dd / "metals_loop.py")

    flags = (
        out["config_layer2_enabled"],
        out["claude_gate_enabled"],
        out["metals_claude_enabled"],
    )
    if any(f is not None for f in flags):
        any_off = any(f is False for f in flags)
        out["enabled"] = not any_off
        out["label"] = "ACTIVE" if out["enabled"] else "FROZEN"
    return out


def _parse_metals_claude_enabled(path: Path) -> bool | None:
    """Read the top-level ``CLAUDE_ENABLED`` constant from metals_loop.py
    WITHOUT importing it (the module pulls in heavy LLM deps + import-time
    side effects). Returns None if the file/constant can't be read.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    m = re.search(r"(?m)^CLAUDE_ENABLED\s*=\s*(True|False)\b", text)
    if not m:
        return None
    return m.group(1) == "True"


def _heartbeat(dd: Path) -> dict[str, Any]:
    """Codex P2 follow-up: outer try/except so any parse / I/O failure
    surfaces in-band rather than 500'ing the whole endpoint."""
    try:
        health = load_json(dd / "health_state.json", default={}) or {}
        if not isinstance(health, dict):
            return _hb_default(error="health_state.json is not a JSON object")
        last = health.get("last_heartbeat")
        age_s: float | None = None
        err: str | None = None
        if last:
            try:
                age_s = (datetime.now(UTC) - _parse_ts(str(last))).total_seconds()
            except Exception as e:
                err = f"heartbeat parse: {type(e).__name__}: {e}"
        out: dict[str, Any] = {
            "age_seconds": age_s,
            "last_ts": last,
            "cycle_count": health.get("cycle_count", 0),
            "error_count": health.get("error_count", 0),
        }
        if err:
            out["error"] = err
        return out
    except Exception as e:
        return _hb_default(error=f"heartbeat: {type(e).__name__}: {e}")


def _hb_default(error: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "age_seconds": None,
        "last_ts": None,
        "cycle_count": 0,
        "error_count": 0,
    }
    if error:
        out["error"] = error
    return out


# Avanza-session categories the home page's "system" errors panel excludes
# in favour of a dedicated Avanza chip (2026-07-18 — 267+ consecutive
# avanza_session_consecutive_failures entries alone were drowning out real
# system errors). Prefix match only. ``auth_failure`` is deliberately NOT
# included: the 2026-07-18 redesign plan's recon miscategorised it as
# Avanza noise, but it's raised by portfolio/claude_gate.py on `claude` CLI
# OAuth login failures — exactly the silent-outage class CLAUDE.md's
# startup check exists to catch (the March-April 2026 3-week Layer 2 auth
# outage). Hiding it under "Avanza" would bury a real Claude-auth failure.
def _is_avanza_category(category: Any) -> bool:
    return isinstance(category, str) and category.startswith("avanza_")


def _error_row(e: dict) -> dict[str, Any]:
    return {
        "ts": e.get("ts"),
        "category": e.get("category"),
        "caller": e.get("caller"),
        "message": (e.get("message") or "")[:200],
    }


def _errors_unresolved(dd: Path) -> dict[str, Any]:
    """Count unresolved critical errors using the SAME semantics as
    scripts/check_critical_errors.py (the CLI startup gate), so the home
    hero and the gate can never disagree:

      - ``CRITICAL_ERRORS_LOOKBACK_DAYS`` window (criticals older than this
        no longer pin the hero — they are the gate's responsibility while
        fresh, and stale noise after);
      - explicit ``resolves_ts`` back-references resolve an earlier entry;
      - auto-resolution of stale categories (a category quiet for >=3 days
        that has a later info/resolution row).

    The full file is still scanned (Codex P1 finding 2026-05-04: never a
    fixed tail — newer info/resolution rows must not hide older unresolved
    criticals). The file is small and this sits behind the 30s TTL cache.

    2026-07-18: also splits the unresolved set by Avanza-vs-system category
    (see ``_is_avanza_category``) so the home page can show a dedicated
    Avanza chip without it drowning out real system errors. ``recent``
    stays as-is (top 5 overall, back-compat); ``recent_system`` is the same
    shape filtered to non-Avanza rows, computed from the FULL unresolved
    list (not the pre-capped ``recent``) so a system error doesn't get
    hidden just because Avanza noise is more recent.
    """
    try:
        entries = load_jsonl(dd / "critical_errors.jsonl")
    except Exception as e:
        # unresolved=None (NOT 0): a degraded reader must never fabricate a
        # clean count — _color surfaces None/error as YELLOW so a broken
        # critical-errors check can't silently flip the hero GREEN.
        return {"unresolved": None, "recent": [], "error": f"errors load: {type(e).__name__}: {e}"}
    try:
        # Reuse the canonical gate logic (window + auto-resolve-stale) so there
        # is a single source of truth. Lazy import keeps the dashboard's other
        # sections working even if scripts/ is somehow unavailable.
        from scripts.check_critical_errors import find_unresolved

        clean = [e for e in entries if isinstance(e, dict)]
        unresolved = find_unresolved(clean, days=CRITICAL_ERRORS_LOOKBACK_DAYS)
        unresolved.sort(key=lambda x: x.get("ts", ""), reverse=True)
        recent = [_error_row(e) for e in unresolved[:5]]
        recent_system = [
            _error_row(e) for e in unresolved if not _is_avanza_category(e.get("category"))
        ][:5]
        avanza_unresolved = sum(1 for e in unresolved if _is_avanza_category(e.get("category")))
        return {
            "unresolved": len(unresolved),
            "recent": recent,
            "recent_system": recent_system,
            "avanza_unresolved": avanza_unresolved,
            "system_unresolved": len(unresolved) - avanza_unresolved,
        }
    except Exception as e:
        # unresolved=None (NOT 0): if the gate import/logic fails, do NOT report
        # a clean zero — that is exactly the "silently flipped to GREEN" class
        # the Codex P1 2026-05-04 note guards against. _color turns this YELLOW.
        return {"unresolved": None, "recent": [], "error": f"errors aggregate: {type(e).__name__}: {e}"}


def _violations_recent(dd: Path) -> dict[str, Any]:
    """Last-24h CRITICAL contract violations, filtered to *unresolved* rows.

    Uses adaptive tail growth so we don't undercount on high-volume
    days. ``contract_violations.jsonl`` can grow unbounded; we read
    the tail and re-pull more if the oldest entry is still inside the
    24h window.

    Resolution-aware (added 2026-05-04). A row is treated as resolved when:

    1. ``layer2_journal_activity`` — ``layer2_journal.jsonl`` has at least
       one entry with ``ts >= violation.details.trigger_time``. The
       journal entry IS the resolution; the contract check itself returns
       early on this condition the next cycle.
    2. The same incident is already represented by an *unresolved* row in
       ``critical_errors.jsonl`` (same ``invariant`` + per-invariant
       identity hash). The errors panel will surface that row; showing
       it again under "violations" would be cosmetic noise.
       *Resolved* critical_errors rows do NOT hide the violation — that
       hand-off must come through path 1 or path 3.
    3. ``critical_errors.jsonl`` has a resolution row whose
       ``resolves_ts`` matches the timestamp of a critical_errors row that
       in turn matches our violation's identity. (Production ``resolves_ts``
       points at the critical_errors row, not the contract_violations row,
       so the match must go via the critical_errors row's ``ts``.)

    Cross-row dedup uses per-invariant identity hashing — the same
    ``_hash_violation_identity`` keys that ``loop_contract`` uses for
    Telegram cooldown — so distinct incidents with similar message text
    don't collapse into one row.

    Without these filters the panel surfaced cleared-but-stale events for
    up to 24h after the underlying issue was fixed (observed 2026-05-04
    when 6 CV rows were shown despite Layer 2 working and accuracy
    regression already disposition'd by the 2026-05-03 research session).
    """
    try:
        entries = _load_last_n_hours(
            dd / "contract_violations.jsonl", hours=24, ts_field="ts"
        )
    except Exception as e:
        return {"unresolved": 0, "recent": [], "error": f"violations load: {type(e).__name__}: {e}"}
    try:
        crit_idx = _critical_errors_index(dd)
        latest_l2_journal_ts = _latest_layer2_journal_ts(dd)

        # Pass 1: severity filter + per-row resolution check.
        kept: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("severity") != "CRITICAL":
                continue
            if _violation_resolved(entry, crit_idx, latest_l2_journal_ts):
                continue
            kept.append(entry)

        # Pass 2: cross-row dedup using per-invariant identity (mirrors
        # loop_contract._hash_violation_identity so two distinct incidents
        # that happen to share the same first 200 chars don't collapse
        # into one row, and two layer2_journal_activity violations on
        # different triggers stay separate even when the rendered text
        # rounds to the same minute count).
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for entry in sorted(kept, key=lambda e: e.get("ts", ""), reverse=True):
            key = _violation_identity_key(entry)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(entry)

        recent = [
            {
                "ts": e.get("ts"),
                "invariant": e.get("invariant"),
                "severity": e.get("severity"),
                "message": (e.get("message") or "")[:200],
            }
            for e in deduped
        ]
        return {"unresolved": len(recent), "recent": recent[:5]}
    except Exception as e:
        return {"unresolved": 0, "recent": [], "error": f"violations aggregate: {type(e).__name__}: {e}"}


def _critical_errors_index(dd: Path) -> dict[str, Any]:
    """Index of critical_errors.jsonl for cross-stream resolution checks.

    Returns a dict with:

    - ``unresolved_keys``: set of ``(invariant, identity_key)`` tuples for
      *unresolved* critical-level entries. A contract_violations row whose
      identity matches one of these is already represented in the errors
      panel, so we hide it under violations to avoid double-counting.
    - ``resolved_keys``: set of ``(invariant, identity_key)`` tuples for
      critical-level entries that have been retroactively resolved (a
      later row pointed at them via ``resolves_ts``). Matching contract
      violations are treated as resolved.

    Resolution rows themselves carry ``level == 'info'`` and a
    ``resolves_ts`` pointing at the original critical row's ``ts`` —
    same protocol as ``check_critical_errors.py``.
    """
    try:
        entries = load_jsonl(dd / "critical_errors.jsonl")
    except Exception:
        return {"unresolved_keys": set(), "resolved_keys": set()}

    by_ts: dict[str, dict] = {}
    resolved_ts: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("category") == "resolution" and entry.get("resolves_ts"):
            resolved_ts.add(str(entry["resolves_ts"]))
            continue
        if entry.get("level") and entry.get("level") != "critical":
            continue
        ts = entry.get("ts")
        if ts:
            by_ts[str(ts)] = entry

    unresolved_keys: set[tuple[str, str]] = set()
    resolved_keys: set[tuple[str, str]] = set()
    for ts, entry in by_ts.items():
        # critical_errors rows record the invariant in either ``caller``
        # or ``category`` (the dispatcher uses the invariant name for
        # both). Prefer caller; fall back to category.
        invariant = (
            entry.get("caller") or entry.get("category") or ""
        )
        key = (str(invariant), _identity_key_for_dict(entry))
        if ts in resolved_ts:
            resolved_keys.add(key)
        else:
            unresolved_keys.add(key)

    return {
        "unresolved_keys": unresolved_keys,
        "resolved_keys": resolved_keys,
    }


def _latest_layer2_journal_ts(dd: Path) -> str | None:
    """Newest ts from ``layer2_journal.jsonl``, or None if empty."""
    try:
        tail = load_jsonl_tail(dd / "layer2_journal.jsonl", max_entries=20)
    except Exception:
        return None
    best: str | None = None
    for e in tail:
        if not isinstance(e, dict):
            continue
        ts = e.get("ts") or e.get("timestamp")
        if ts and (best is None or str(ts) > best):
            best = str(ts)
    return best


def _violation_identity_key(entry: dict) -> str:
    """Per-invariant identity payload for cross-stream resolution checks.

    Delegates to ``portfolio.loop_contract.violation_identity_payload`` —
    the *same* function the source uses for Telegram cooldown / dedup
    state — so the two sides cannot drift.
    """
    return _identity_key_for_dict(entry)


def _identity_key_for_dict(entry: dict) -> str:
    # contract_violations.jsonl uses ``invariant``, critical_errors.jsonl
    # uses ``caller`` (or ``category`` as fallback). The dispatcher writes
    # the invariant name into both fields for the periodic-violation path
    # (loop_contract:992-997), but the inline layer2 path writes
    # category="contract_violation" + caller=invariant_name
    # (loop_contract:471-476) — prefer caller, fall back to category.
    invariant = (
        entry.get("invariant")
        or entry.get("caller")
        or entry.get("category")
        or ""
    )
    raw_msg = entry.get("message") or ""
    # ViolationTracker promotes warnings by prepending
    # "ESCALATED (Nx consecutive): " — the source strips this before
    # hashing; mirror that here so escalated CV rows match their
    # pre-escalation form and their critical_errors counterpart.
    msg = _ESCALATED_PREFIX_RE.sub("", raw_msg, count=1)[:200]

    # ``record_critical_error`` writes the payload under ``context``;
    # ``_log_violations`` writes it under ``details``. Accept either so
    # the cross-stream identity match works on both sides without a
    # wire-format change. (Claude review of a85a646f, P1-1.)
    payload_dict = entry.get("details") or entry.get("context") or {}
    if not isinstance(payload_dict, dict):
        payload_dict = {}

    return violation_identity_payload(invariant, msg, payload_dict)


def _violation_resolved(
    entry: dict,
    crit_idx: dict[str, Any],
    latest_l2_journal_ts: str | None,
) -> bool:
    # Path 1: layer2_journal_activity is implicitly resolved by a later
    # journal entry — same condition the contract check itself uses next
    # cycle.
    if entry.get("invariant") == "layer2_journal_activity":
        details = entry.get("details") or {}
        trig = details.get("trigger_time")
        if trig and latest_l2_journal_ts and str(latest_l2_journal_ts) >= str(trig):
            return True

    # Paths 2 + 3: cross-stream dedup via critical_errors.jsonl. A
    # contract violation that matches any unresolved or resolved
    # critical_errors entry is already accounted for: unresolved -> the
    # errors panel will surface it; resolved -> it's been cleared.
    key = (
        str(entry.get("invariant") or ""),
        _identity_key_for_dict(entry),
    )
    if key in crit_idx.get("resolved_keys", set()):
        return True
    if key in crit_idx.get("unresolved_keys", set()):
        return True
    return False


def _llm_inference(dd: Path) -> dict[str, Any]:
    """Per-LLM inference success rate.

    Sources:
      * ``local_llm_report_latest.json -> health.{chronos,kronos}`` —
        forecast models, ``ok``/``fail`` counts over the report window.
      * ``health_state.json -> signal_health[*]`` — for signal-level LLM
        signals (``claude_fundamental``, ``forecast``). ``total_calls``
        and ``total_failures`` are lifetime counters maintained by
        ``portfolio.health.record_signal_call``.

    Ministral and Qwen3 don't yet have inference-success counters in
    either source; their accuracy lives in ``local_llm_report_latest``
    but that is a different metric (was the BUY/SELL right at horizon).
    Kept out of this view rather than mislabelled.

    Codex P2 follow-up: every numeric field is parsed defensively so
    a malformed ``{"ok": "oops"}`` row no longer crashes the whole
    payload — that model is skipped with no other side-effects.
    """
    try:
        health = load_json(dd / "health_state.json", default={}) or {}
        sh = health.get("signal_health", {}) if isinstance(health, dict) else {}
        if not isinstance(sh, dict):
            sh = {}
        llm_report = load_json(dd / "local_llm_report_latest.json", default={}) or {}
        llm_health = llm_report.get("health", {}) if isinstance(llm_report, dict) else {}
        if not isinstance(llm_health, dict):
            llm_health = {}

        models: list[dict[str, Any]] = []

        for key, label in (("chronos", "Chronos-2"), ("kronos", "Kronos")):
            h = llm_health.get(key)
            if not isinstance(h, dict):
                continue
            ok = _safe_int(h.get("ok"))
            fail = _safe_int(h.get("fail"))
            if ok is None or fail is None:
                continue
            total = ok + fail
            if total == 0:
                continue
            models.append(
                {
                    "name": label,
                    "key": key,
                    "total": total,
                    "failures": fail,
                    "success_pct": round(100.0 * ok / total, 1),
                }
            )

        for key, label in (
            ("claude_fundamental", "Claude Fundamental"),
            ("forecast", "Forecast voter"),
        ):
            h = sh.get(key)
            if not isinstance(h, dict):
                continue
            total = _safe_int(h.get("total_calls"))
            fail = _safe_int(h.get("total_failures"))
            if total is None or fail is None or total <= 0:
                continue
            models.append(
                {
                    "name": label,
                    "key": key,
                    "total": total,
                    "failures": fail,
                    "success_pct": round(100.0 * (total - fail) / total, 1),
                    "last_failure_ts": h.get("last_failure"),
                }
            )

        overall_pct: float | None = None
        if models:
            weight = sum(m["total"] for m in models)
            if weight > 0:
                overall_pct = round(
                    sum(m["success_pct"] * m["total"] for m in models) / weight, 1
                )

        return {"models": models, "overall_pct": overall_pct}
    except Exception as e:
        return {"models": [], "overall_pct": None,
                "error": f"llm_inference: {type(e).__name__}: {e}"}


def _layer2_24h(dd: Path) -> dict[str, Any]:
    """Layer 2 trigger frequency + success rate over the last 24h.

    A trigger fires whenever Layer 1 spawns a Claude CLI subprocess and
    appends to ``claude_invocations.jsonl``. ``status`` is one of
    ``invoked`` (success), ``timeout``, ``error``, ``blocked``.

    Codex P2 follow-up: uses adaptive tail growth so a high-volume day
    (>2000 invocations in 24h) doesn't silently undercount.
    """
    try:
        entries = _load_last_n_hours(
            dd / "claude_invocations.jsonl", hours=24, ts_field="timestamp"
        )
    except Exception as e:
        return {
            "triggers_24h": 0,
            "success_24h": 0,
            "success_pct": None,
            "latest": None,
            "spark_24h": [0] * 24,
            "cost_usd_24h": 0.0,
            "input_tokens_24h": 0,
            "output_tokens_24h": 0,
            "cache_read_tokens_24h": 0,
            "cache_creation_tokens_24h": 0,
            "parsed_24h": 0,
            "error": f"invocations load: {type(e).__name__}: {e}",
        }
    try:
        cutoff = datetime.now(UTC) - timedelta(hours=24)
        triggers: list[tuple[datetime, dict]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ts_raw = entry.get("timestamp")
            if not ts_raw:
                continue
            try:
                ts = _parse_ts(str(ts_raw))
            except Exception:
                continue
            if ts < cutoff:
                continue
            triggers.append((ts, entry))

        triggers.sort(key=lambda x: x[0])
        success = sum(1 for _, e in triggers if e.get("status") == "invoked")
        pct = round(100.0 * success / len(triggers), 1) if triggers else None
        latest_entry = triggers[-1][1] if triggers else None

        # Cost + token rollup. Only parse_ok rows have meaningful numbers;
        # rows without ``cost_usd`` contribute nothing. Mirrors the logic in
        # scripts/claude_cost_report.summarise so dashboard and CLI agree.
        cost_usd = 0.0
        input_tokens = 0
        output_tokens = 0
        cache_read_tokens = 0
        cache_creation_tokens = 0
        parsed = 0
        for _, e in triggers:
            if not e.get("parse_ok"):
                continue
            parsed += 1
            try:
                cost_usd += float(e.get("cost_usd") or 0)
                input_tokens += int(e.get("input_tokens") or 0)
                output_tokens += int(e.get("output_tokens") or 0)
                cache_read_tokens += int(e.get("cache_read_tokens") or 0)
                cache_creation_tokens += int(e.get("cache_creation_tokens") or 0)
            except (TypeError, ValueError):
                continue

        now = datetime.now(UTC)
        buckets = [0] * 24
        for ts, _ in triggers:
            hours_ago = int((now - ts).total_seconds() // 3600)
            if 0 <= hours_ago < 24:
                buckets[23 - hours_ago] += 1

        latest_payload: dict[str, Any] | None = None
        if latest_entry is not None:
            latest_payload = {
                "ts": latest_entry.get("timestamp"),
                "caller": latest_entry.get("caller"),
                "status": latest_entry.get("status"),
                "duration_seconds": latest_entry.get("duration_seconds"),
                "model": latest_entry.get("model"),
            }

        return {
            "triggers_24h": len(triggers),
            "success_24h": success,
            "success_pct": pct,
            "latest": latest_payload,
            "spark_24h": buckets,
            "cost_usd_24h": round(cost_usd, 4),
            "input_tokens_24h": input_tokens,
            "output_tokens_24h": output_tokens,
            "cache_read_tokens_24h": cache_read_tokens,
            "cache_creation_tokens_24h": cache_creation_tokens,
            "parsed_24h": parsed,
        }
    except Exception as e:
        return {
            "triggers_24h": 0,
            "success_24h": 0,
            "success_pct": None,
            "latest": None,
            "spark_24h": [0] * 24,
            "cost_usd_24h": 0.0,
            "input_tokens_24h": 0,
            "output_tokens_24h": 0,
            "cache_read_tokens_24h": 0,
            "cache_creation_tokens_24h": 0,
            "parsed_24h": 0,
            "error": f"layer2 aggregate: {type(e).__name__}: {e}",
        }


def _signal_aggregate(dd: Path) -> dict[str, Any]:
    """Latest signal_log entry collapsed into per-ticker counts.

    ``total_voters`` in the source is BUY+SELL only (active voters);
    every other signal in the dict counts as HOLD/abstain. We expose
    both ``hold`` (literal vote count) and ``abstain`` (alias) so the
    UI can word the row either way.

    Codex P2 follow-up: tolerate the latest entry being a non-dict
    (e.g. an empty list or null) — the section reports an in-band
    error instead of bubbling AttributeError up to the route.
    """
    try:
        entries = load_jsonl_tail(dd / "signal_log.jsonl", max_entries=5)
    except Exception as e:
        return {"tickers": [], "error": f"signal_log load: {type(e).__name__}: {e}"}
    if not entries:
        return {"tickers": []}
    last = entries[-1]
    if not isinstance(last, dict):
        return {"tickers": [], "error": "signal_log: last entry is not a JSON object"}
    try:
        tickers_dict = last.get("tickers", {}) or {}
        if not isinstance(tickers_dict, dict):
            return {"ts": last.get("ts"), "tickers": [],
                    "error": "signal_log.tickers is not a JSON object"}
        tickers: list[dict[str, Any]] = []
        for sym, data in tickers_dict.items():
            if not isinstance(data, dict):
                continue
            signals = data.get("signals", {}) or {}
            if not isinstance(signals, dict):
                continue
            total = len(signals)
            buy = sum(1 for v in signals.values() if v == "BUY")
            sell = sum(1 for v in signals.values() if v == "SELL")
            hold = total - buy - sell
            active = buy + sell
            confidence = (active / total) if total else 0.0
            tickers.append(
                {
                    "ticker": sym,
                    "consensus": data.get("consensus", "HOLD"),
                    "buy": buy,
                    "sell": sell,
                    "hold": hold,
                    "abstain": hold,
                    "total": total,
                    "confidence": round(confidence, 3),
                    "regime": data.get("regime"),
                    # 2026-07-18: vote target horizon persisted by
                    # outcome_tracker.log_signal_snapshot; None on rows
                    # written before the field existed → UI falls back
                    # to "1d (default)".
                    "horizon": data.get("horizon"),
                }
            )
        return {"ts": last.get("ts"), "tickers": tickers}
    except Exception as e:
        return {"tickers": [], "error": f"signal_aggregate: {type(e).__name__}: {e}"}


def _pnl_footer(dd: Path) -> dict[str, Any]:
    """Single-line P&L for the deprioritised portfolio footer."""
    try:
        ps = load_json(dd / "portfolio_state.json", default={}) or {}
        pb = load_json(dd / "portfolio_state_bold.json", default={}) or {}
        if not isinstance(ps, dict):
            ps = {}
        if not isinstance(pb, dict):
            pb = {}
        return {
            "patient_value_sek": ps.get("portfolio_value", ps.get("equity_sek")),
            "bold_value_sek": pb.get("portfolio_value", pb.get("equity_sek")),
            "patient_starting_sek": ps.get("starting_capital", 500_000.0),
            "bold_starting_sek": pb.get("starting_capital", 500_000.0),
        }
    except Exception as e:
        return {"error": f"pnl load: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Truth & freshness layer (2026-07-18) — sources / layer1 / voters
# ---------------------------------------------------------------------------

# Expected write cadence per backing file, in seconds. ``frozen`` in
# _source_freshness() fires at 2x this. Loop heartbeats are 60s except
# heartbeat.txt (pf-dataloop / Layer 1's own heartbeat, 600s main-loop
# cycle — see portfolio/main.py). health_state.json and signal_log.jsonl
# are written every Layer 1 cycle (600s); local_llm_report_latest.json is
# refreshed hourly by a separate reporting job.
_SOURCE_CADENCE_SEC: dict[str, int] = {
    "health_state.json": 600,
    "signal_log.jsonl": 600,
    "local_llm_report_latest.json": 3600,
    "crypto_loop.heartbeat": 60,
    "metals_loop.heartbeat": 60,
    "oil_loop.heartbeat": 60,
    "mstr_loop.heartbeat": 60,
    "golddigger_loop.heartbeat": 60,
    "heartbeat.txt": 600,
}

# Error/violation logs have no steady cadence — a long gap just means
# nothing went wrong, not that the writer died. Never mark them frozen.
_SOURCE_NEVER_FROZEN: tuple[str, ...] = (
    "critical_errors.jsonl",
    "contract_violations.jsonl",
)

_SOURCE_FILES: tuple[str, ...] = tuple(_SOURCE_CADENCE_SEC) + _SOURCE_NEVER_FROZEN


def _source_freshness(dd: Path, filename: str) -> dict[str, Any]:
    """mtime/age/frozen for one backing file. Missing file -> all-null,
    not frozen (nothing to compare a cadence against yet)."""
    try:
        mtime = (dd / filename).stat().st_mtime
    except OSError:
        return {"mtime": None, "age_sec": None, "frozen": False}
    age_sec = int(time.time() - mtime)
    if filename in _SOURCE_NEVER_FROZEN:
        frozen = False
    else:
        frozen = age_sec > 2 * _SOURCE_CADENCE_SEC.get(filename, 600)
    return {"mtime": mtime, "age_sec": age_sec, "frozen": frozen}


def _sources(dd: Path) -> dict[str, Any]:
    """Freshness of every file the home page's other sections read from.

    Lets the dashboard say "signal data frozen since <ts>" instead of
    quietly rendering stale numbers as if they were live (2026-07-17
    pf-dataloop pause went unnoticed for a full day this way).
    """
    try:
        return {name: _source_freshness(dd, name) for name in _SOURCE_FILES}
    except Exception as e:
        return {"error": f"sources: {type(e).__name__}: {e}"}


_LAYER1_UNIT = "pf-dataloop"


def _systemctl_user_query(unit: str, verb: str) -> str | None:
    """Run `systemctl --user <verb> <unit>` and return stripped stdout, or
    None on any failure (binary missing, timeout, non-systemd host). Never
    raises — this feeds a dashboard section that must not 500."""
    try:
        r = subprocess.run(
            ["systemctl", "--user", verb, unit],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip()
    except Exception:
        return None


def _layer1(dd: Path) -> dict[str, Any]:
    """pf-dataloop (Layer 1 main loop) systemd state + last cycle ts.

    2026-07-17: the dashboard showed "Claude Fundamental 100%" (a lifetime
    counter) as if it were live health while pf-dataloop sat disabled for
    a full day. This section is the fix — surface the actual unit state
    instead of inferring it from stale signal data.
    """
    try:
        active = _systemctl_user_query(_LAYER1_UNIT, "is-active") == "active"
        enabled = _systemctl_user_query(_LAYER1_UNIT, "is-enabled") == "enabled"
        try:
            last_cycle_ts = (dd / "heartbeat.txt").read_text(encoding="utf-8").strip() or None
        except OSError:
            last_cycle_ts = None
        return {
            "unit": _LAYER1_UNIT,
            "active": active,
            "enabled": enabled,
            "last_cycle_ts": last_cycle_ts,
        }
    except Exception as e:
        return {
            "unit": _LAYER1_UNIT,
            "active": False,
            "enabled": False,
            "last_cycle_ts": None,
            "error": f"layer1: {type(e).__name__}: {e}",
        }


def _avanza_status(dd: Path, *, config_path: Path | None = None) -> dict[str, Any]:
    """Avanza credential/session truth for the home page's Avanza chip.

    Config-only (no Avanza network calls — that's the expensive
    Playwright-backed /api/avanza_account path, unsuitable for a 30s home
    poll). Never exposes credential VALUES, only whether they're populated.
    ``unresolved_errors`` is filled in by ``compute()`` from the errors
    section's ``avanza_unresolved`` count (kept out of here to avoid this
    function depending on ``_errors_unresolved``'s return shape).

    ``config_path`` defaults to the real repo config.json (via
    ``_CONFIG_FILE``, a sibling of ``data/``); tests override it explicitly
    so they never read/depend on a shared pytest tmp root.
    """
    cfg_file = config_path if config_path is not None else _CONFIG_FILE
    try:
        cfg = load_json(cfg_file, default={}) or {}
        av_cfg = cfg.get("avanza") if isinstance(cfg, dict) else None
        if not isinstance(av_cfg, dict):
            av_cfg = {}
        creds_configured = bool(
            av_cfg.get("username") and av_cfg.get("password") and av_cfg.get("totp_secret")
        )
        return {"creds_configured": creds_configured}
    except Exception as e:
        return {"creds_configured": None, "error": f"avanza status: {type(e).__name__}: {e}"}


# Curated voter watch-list — the signals whose "green" health has
# historically meant "force-HOLD, not actually voting" (claude_fundamental,
# forecast) or whose availability flips on a live remote-GPU gate
# (phi4_mini). Not the full 89-signal registry — see Phase 4 plan for that.
_VOTER_NAMES: tuple[str, ...] = (
    "claude_fundamental",
    "forecast",
    "phi4_mini",
    "ministral",
    "qwen3",
    "ml",
    "sentiment",
)

# Signals whose voting (when rescued per-ticker) depends on local GPU
# inference being unpaused. Checked against data/local_llm.disabled.
# NOTE: "ml" is a joblib/sklearn classifier, not an LLM — it doesn't
# actually depend on local_llm_gate today, but is included here per the
# Phase 1 spec; flagged as a discrepancy for the Phase 4 registry cleanup.
_LLM_FLAG_GATED_SIGNALS: frozenset[str] = frozenset({"ministral", "qwen3", "sentiment", "ml"})


def _voter_state(
    name: str,
    *,
    registry: Any,
    overrides: Any,
    signal_health: dict[str, Any],
    shadows: dict[str, Any],
    llm_flag_paused: bool,
) -> dict[str, Any]:
    h = signal_health.get(name)
    last_activity_ts = h.get("last_success") if isinstance(h, dict) else None

    if not registry.is_globally_disabled(name, ticker=None):
        # None of the curated names are expected to be globally enabled
        # today; stay truthful if that ever changes rather than mislabel.
        return {"state": "VOTING", "reason": None, "last_activity_ts": last_activity_ts}

    rescued_tickers = sorted(t for (s, t) in overrides if s == name)

    if name == "phi4_mini" and rescued_tickers:
        try:
            from portfolio.llama_server import remote_llm_available

            available = remote_llm_available()
        except Exception:
            available = False
        tickers_str = ", ".join(rescued_tickers)
        if available:
            return {
                "state": "VOTING",
                "reason": f"force-HOLD globally; remote LLM reachable, voting for: {tickers_str}",
                "last_activity_ts": last_activity_ts,
            }
        return {
            "state": "GATED_REMOTE_DOWN",
            "reason": f"force-HOLD globally; remote LLM (herc2) unreachable, would vote for: {tickers_str}",
            "last_activity_ts": last_activity_ts,
        }

    if rescued_tickers:
        if llm_flag_paused and name in _LLM_FLAG_GATED_SIGNALS:
            return {
                "state": "PAUSED_LLM_FLAG",
                "reason": (
                    f"rescued per-ticker for {', '.join(rescued_tickers)}, but local "
                    "LLM inference is paused (data/local_llm.disabled)"
                ),
                "last_activity_ts": last_activity_ts,
            }
        return {
            "state": "VOTING",
            "reason": f"force-HOLD globally; re-enabled per-ticker for: {', '.join(rescued_tickers)}",
            "last_activity_ts": last_activity_ts,
        }

    reason = registry.disabled_reason(name)
    shadow = shadows.get(name)
    if isinstance(shadow, dict) and shadow.get("status"):
        note = f"shadow_registry status={shadow['status']}"
        reason = f"{reason}; {note}" if reason else note

    return {"state": "DISABLED", "reason": reason, "last_activity_ts": last_activity_ts}


def _get_registry() -> Any:
    """Indirection so tests can inject a synthetic registry stub without
    needing a full portfolio.registry_defaults SIGNALS table — see
    tests/test_dashboard_system_status.py TestVoters.
    """
    from portfolio.component_registry import get_registry

    return get_registry()


def _voter_overrides() -> Any:
    """(signal, ticker) pairs rescued from a global disable.

    Same source component_registry.py reads (portfolio.registry_defaults),
    imported separately here because ComponentRegistry exposes this as a
    plain module-level table, not through an instance method.
    """
    from portfolio.registry_defaults import DISABLED_SIGNAL_OVERRIDES

    return DISABLED_SIGNAL_OVERRIDES


def _voters(dd: Path) -> dict[str, Any]:
    """Per-signal voting truth for the curated LLM/ML watch-list.

    Kills the "100% green but force-HOLD" confusion: a signal's lifetime
    call-success counter (health_state.json) says nothing about whether
    it's actually in the consensus vote today. Wrapped whole so one bad
    import/accessor (e.g. component_registry failing to load) can't blank
    the rest of the hero.

    Base enabled/disabled state + the global disabled-reason string come
    from the Phase 4.1 component registry (``_get_registry()`` /
    ``_voter_overrides()`` — both a thin indirection so tests can
    substitute a synthetic registry instead of the real
    portfolio.registry_defaults tables). Everything else here stays a
    live, dynamic read that the registry deliberately does not model:
    remote-LLM gate (remote_llm_available), the local_llm.disabled flag,
    shadow_registry status, and health_state.json activity timestamps.
    """
    try:
        registry = _get_registry()
        overrides = _voter_overrides()

        health = load_json(dd / "health_state.json", default={}) or {}
        signal_health = health.get("signal_health", {}) if isinstance(health, dict) else {}
        if not isinstance(signal_health, dict):
            signal_health = {}

        shadow_registry = load_json(dd / "shadow_registry.json", default={}) or {}
        shadows = shadow_registry.get("shadows", {}) if isinstance(shadow_registry, dict) else {}
        if not isinstance(shadows, dict):
            shadows = {}

        llm_flag_paused = (dd / "local_llm.disabled").exists()

        return {
            name: _voter_state(
                name,
                registry=registry,
                overrides=overrides,
                signal_health=signal_health,
                shadows=shadows,
                llm_flag_paused=llm_flag_paused,
            )
            for name in _VOTER_NAMES
        }
    except Exception as e:
        return {"error": f"voters: {type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Severity rollup
# ---------------------------------------------------------------------------


def _errors_reason(n: int, total: int) -> str:
    """Hero reason string for the errors count.

    Shows the Avanza-excluded system count (``n``), with the raw
    (Avanza-included) ``total`` in parens when the two differ — e.g.
    "15 system (66 total)". Falls back to the old "N unresolved error(s)"
    phrasing when they're equal (no Avanza noise to subtract).
    """
    if total != n:
        return f"{n} system ({total} total)"
    return f"{n} unresolved error{'s' if n != 1 else ''}"


def _color(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """Compute overall GREEN/YELLOW/RED + a list of reasons.

    YELLOW means "look at this when you next glance at the dashboard."
    RED means "something is actually broken right now."
    """
    severity = "GREEN"
    reasons: list[str] = []

    def bump(level: str) -> None:
        nonlocal severity
        rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}
        if rank[level] > rank[severity]:
            severity = level

    hb = payload.get("heartbeat") or {}
    age = hb.get("age_seconds")
    if age is None:
        bump("RED")
        reasons.append("loop heartbeat: unknown")
    elif age > HEARTBEAT_YELLOW_S:
        bump("RED")
        reasons.append(f"loop silent {int(age / 60)}m")
    elif age > HEARTBEAT_GREEN_S:
        bump("YELLOW")
        reasons.append(f"loop heartbeat {int(age)}s ago")

    err = payload.get("errors") or {}
    if err.get("error") or err.get("unresolved") is None:
        # Degraded reader (load/import/logic failure). NEVER silently GREEN —
        # an unknown count must be visible. YELLOW = "look at this" (Codex P1
        # 2026-05-04: a fabricated 0 once hid real unresolved criticals).
        bump("YELLOW")
        reasons.append("critical-errors check degraded")
    else:
        # 2026-07-18: the hero used to count ALL unresolved criticals,
        # including the Avanza auth-retry noise (see the dedicated Avanza
        # chip) — a BankID re-auth backlog alone pinned the hero RED even
        # when everything else was fine. Drive severity off
        # system_unresolved (Avanza categories excluded); fall back to the
        # raw unresolved count for older cached payloads that predate the
        # split. _errors_reason keeps the raw total visible in parens.
        total = err.get("unresolved", 0) or 0
        n = err.get("system_unresolved")
        if n is None:
            n = total
        if n > ERRORS_YELLOW_MAX:
            bump("RED")
            reasons.append(_errors_reason(n, total))
        elif n > 0:
            bump("YELLOW")
            reasons.append(_errors_reason(n, total))

    cv = payload.get("contract_violations") or {}
    vn = cv.get("unresolved", 0) or 0
    if vn > 5:
        bump("RED")
        reasons.append(f"{vn} contract violations 24h")
    elif vn > 0:
        bump("YELLOW")
        reasons.append(f"{vn} contract violation{'s' if vn != 1 else ''} 24h")

    llm = payload.get("llm_inference") or {}
    pct = llm.get("overall_pct")
    if pct is not None:
        if pct < LLM_YELLOW_PCT:
            bump("RED")
            reasons.append(f"LLM inference {pct}%")
        elif pct < LLM_GREEN_PCT:
            bump("YELLOW")
            reasons.append(f"LLM inference {pct}%")

    l2 = payload.get("layer2") or {}
    l2pct = l2.get("success_pct")
    if (
        l2pct is not None
        and (l2.get("triggers_24h", 0) or 0) >= LAYER2_MIN_TRIGGERS_FOR_GATE
    ):
        if l2pct < LAYER2_YELLOW_PCT:
            bump("RED")
            reasons.append(f"Layer 2 success {l2pct}%")
        elif l2pct < LAYER2_GREEN_PCT:
            bump("YELLOW")
            reasons.append(f"Layer 2 success {l2pct}%")

    gate = (payload.get("layer2") or {}).get("gate") or {}
    if gate.get("enabled") is False:
        # Intentional token-saving freeze — surface it on the hero but do NOT
        # bump severity: the system is healthy, just not spending Claude tokens.
        # ASCII only: this string is logged/printed on a cp1252 Windows console;
        # a non-ASCII glyph here raises UnicodeEncodeError. The browser pill
        # (layer2-activity-card.js) carries the visual icon instead.
        reasons.append("Layer 2 frozen (token-saving)")

    if not reasons:
        reasons = ["all systems nominal"]
    return severity, reasons


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ts(s: str) -> datetime:
    """Parse an ISO timestamp tolerating trailing Z."""
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def _safe_int(v: Any) -> int | None:
    """``int(v)`` that returns None instead of raising on bad input.

    Used so a ``{"ok": "oops"}`` row inside local_llm_report skips the
    affected model rather than 500'ing the whole endpoint (codex P2
    finding 2026-05-04).
    """
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _load_last_n_hours(path: Path, *, hours: int, ts_field: str) -> list[dict]:
    """Tail-read a jsonl file growing the window until the oldest entry
    is older than the cutoff or we've fully scanned the file.

    Used by ``_layer2_24h`` and ``_violations_recent`` to make the
    "last 24h" claim authoritative even when activity spikes (codex
    P2 finding 2026-05-04: previous tail of 2000 lines silently
    undercounted on high-volume days).

    The returned list is in file order (oldest first). Every step
    doubles the tail size up to a 50K cap; if the cap is hit we fall
    back to a full ``load_jsonl`` so the count remains correct.
    """
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    cutoff_iso = cutoff.isoformat()
    for max_entries in (500, 2_000, 10_000, 50_000):
        rows = load_jsonl_tail(
            path, max_entries=max_entries, tail_bytes=max_entries * 512
        )
        if not rows:
            return []
        oldest = next(
            (r.get(ts_field) for r in rows if isinstance(r, dict) and r.get(ts_field)),
            None,
        )
        if not oldest or str(oldest) < cutoff_iso:
            return [r for r in rows if isinstance(r, dict)
                    and (r.get(ts_field) or "") >= cutoff_iso]
    rows = load_jsonl(path)
    return [r for r in rows if isinstance(r, dict)
            and (r.get(ts_field) or "") >= cutoff_iso]
