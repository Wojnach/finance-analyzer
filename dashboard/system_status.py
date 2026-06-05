"""System-health rollup for the dashboard home page.

Pure aggregator. Reads the same jsonl/json files the rest of the system
already writes — adds no new instrumentation. Returns a single payload
shape that the home view consumes via ``/api/system_status``.

Why this exists: the previous home page led with simulated-portfolio
P&L, which the user explicitly deprioritised in favour of "is the
system actually working" indicators. See
``docs/PLAN.md`` 2026-05-04 entry and the plan file
``/root/.claude/plans/merry-tinkering-cake.md`` for the full design.

Side-effect-free: never opens a network socket, never invokes Avanza,
never writes a file. Safe to call from a 30s polling loop.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import re

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

# Severity thresholds — tune freely, no schema migration needed.
HEARTBEAT_GREEN_S = 120
HEARTBEAT_YELLOW_S = 600
ERRORS_YELLOW_MAX = 3
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
    }
    # 2026-06-06: attach Claude-gate state into the layer2 section so the home
    # page shows when Layer 2 / Claude trading is intentionally FROZEN
    # (token-saving) instead of looking like a silent outage. Nested under
    # layer2 (not a top-level key) so the existing layer2-activity-card renders
    # the badge with no home.js rewiring. See _claude_gate + SESSION_PROGRESS.
    if isinstance(out.get("layer2"), dict):
        out["layer2"]["gate"] = _claude_gate(dd)
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


def _errors_unresolved(dd: Path) -> dict[str, Any]:
    """Walk critical_errors.jsonl. An entry with category="resolution"
    and ``resolves_ts`` pointing at an earlier entry resolves it.

    Codex P1 finding 2026-05-04: this MUST scan the whole file, not a
    fixed tail. If 500 newer info/resolution rows came after older
    unresolved criticals, the older ones disappeared from the count
    and the home page silently flipped to GREEN. critical_errors.jsonl
    is small (~120 KB at the time of writing); we accept the full scan
    behind the 30s TTL cache.
    """
    try:
        entries = load_jsonl(dd / "critical_errors.jsonl")
    except Exception as e:
        return {"unresolved": 0, "recent": [], "error": f"errors load: {type(e).__name__}: {e}"}
    try:
        resolved_ts: set[str] = set()
        by_ts: dict[str, dict] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            ts = entry.get("ts")
            if entry.get("category") == "resolution" and entry.get("resolves_ts"):
                resolved_ts.add(str(entry["resolves_ts"]))
                continue
            # Skip non-critical levels (resolution rows arrive as "info").
            if entry.get("level") and entry.get("level") != "critical":
                continue
            if ts:
                by_ts[ts] = entry
        unresolved = [e for ts, e in by_ts.items() if ts not in resolved_ts]
        unresolved.sort(key=lambda x: x.get("ts", ""), reverse=True)
        recent = [
            {
                "ts": e.get("ts"),
                "category": e.get("category"),
                "caller": e.get("caller"),
                "message": (e.get("message") or "")[:200],
            }
            for e in unresolved[:5]
        ]
        return {"unresolved": len(unresolved), "recent": recent}
    except Exception as e:
        return {"unresolved": 0, "recent": [], "error": f"errors aggregate: {type(e).__name__}: {e}"}


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
# Severity rollup
# ---------------------------------------------------------------------------


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
    n = err.get("unresolved", 0) or 0
    if n > ERRORS_YELLOW_MAX:
        bump("RED")
        reasons.append(f"{n} unresolved errors")
    elif n > 0:
        bump("YELLOW")
        reasons.append(f"{n} unresolved error{'s' if n != 1 else ''}")

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
