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

from portfolio.file_utils import load_json, load_jsonl_tail

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
    overall, reasons = _color(out)
    out["overall"] = overall
    out["reasons"] = reasons
    return out


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _heartbeat(dd: Path) -> dict[str, Any]:
    health = load_json(dd / "health_state.json", default={}) or {}
    last = health.get("last_heartbeat")
    age_s: float | None = None
    err: str | None = None
    if last:
        try:
            age_s = (datetime.now(UTC) - _parse_ts(last)).total_seconds()
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


def _errors_unresolved(dd: Path) -> dict[str, Any]:
    """Walk critical_errors.jsonl. An entry with category="resolution"
    and ``resolves_ts`` pointing at an earlier entry resolves it.
    """
    try:
        entries = load_jsonl_tail(dd / "critical_errors.jsonl", max_entries=500)
    except Exception as e:
        return {"unresolved": 0, "recent": [], "error": f"errors load: {type(e).__name__}: {e}"}
    resolved_ts: set[str] = set()
    by_ts: dict[str, dict] = {}
    for entry in entries:
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


def _violations_recent(dd: Path) -> dict[str, Any]:
    """Last-24h CRITICAL contract violations. WARNINGs are noise."""
    try:
        entries = load_jsonl_tail(dd / "contract_violations.jsonl", max_entries=500)
    except Exception as e:
        return {"unresolved": 0, "recent": [], "error": f"violations load: {type(e).__name__}: {e}"}
    cutoff = datetime.now(UTC) - timedelta(hours=24)
    recent: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("severity") != "CRITICAL":
            continue
        ts = entry.get("ts")
        try:
            if ts and _parse_ts(ts) < cutoff:
                continue
        except Exception:
            continue
        recent.append(
            {
                "ts": ts,
                "invariant": entry.get("invariant"),
                "severity": entry.get("severity"),
                "message": (entry.get("message") or "")[:200],
            }
        )
    recent.sort(key=lambda x: x.get("ts", ""), reverse=True)
    return {"unresolved": len(recent), "recent": recent[:5]}


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
    """
    health = load_json(dd / "health_state.json", default={}) or {}
    sh = health.get("signal_health", {}) or {}
    llm_report = load_json(dd / "local_llm_report_latest.json", default={}) or {}
    llm_health = llm_report.get("health", {}) or {}

    models: list[dict[str, Any]] = []

    for key, label in (("chronos", "Chronos-2"), ("kronos", "Kronos")):
        h = llm_health.get(key)
        if not isinstance(h, dict):
            continue
        ok = int(h.get("ok", 0))
        fail = int(h.get("fail", 0))
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
        total = int(h.get("total_calls", 0))
        fail = int(h.get("total_failures", 0))
        if total == 0:
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


def _layer2_24h(dd: Path) -> dict[str, Any]:
    """Layer 2 trigger frequency + success rate over the last 24h.

    A trigger fires whenever Layer 1 spawns a Claude CLI subprocess and
    appends to ``claude_invocations.jsonl``. ``status`` is one of
    ``invoked`` (success), ``timeout``, ``error``, ``blocked``.
    """
    try:
        entries = load_jsonl_tail(dd / "claude_invocations.jsonl", max_entries=2000)
    except Exception as e:
        return {
            "triggers_24h": 0,
            "success_24h": 0,
            "success_pct": None,
            "latest": None,
            "spark_24h": [0] * 24,
            "error": f"invocations load: {type(e).__name__}: {e}",
        }

    cutoff = datetime.now(UTC) - timedelta(hours=24)
    triggers: list[tuple[datetime, dict]] = []
    for entry in entries:
        ts_raw = entry.get("timestamp")
        if not ts_raw:
            continue
        try:
            ts = _parse_ts(ts_raw)
        except Exception:
            continue
        if ts < cutoff:
            continue
        triggers.append((ts, entry))

    triggers.sort(key=lambda x: x[0])
    success = sum(1 for _, e in triggers if e.get("status") == "invoked")
    pct = round(100.0 * success / len(triggers), 1) if triggers else None
    latest_entry = triggers[-1][1] if triggers else None

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
    }


def _signal_aggregate(dd: Path) -> dict[str, Any]:
    """Latest signal_log entry collapsed into per-ticker counts.

    ``total_voters`` in the source is BUY+SELL only (active voters);
    every other signal in the dict counts as HOLD/abstain. We expose
    both ``hold`` (literal vote count) and ``abstain`` (alias) so the
    UI can word the row either way.
    """
    try:
        entries = load_jsonl_tail(dd / "signal_log.jsonl", max_entries=5)
    except Exception as e:
        return {"tickers": [], "error": f"signal_log load: {type(e).__name__}: {e}"}
    if not entries:
        return {"tickers": []}

    last = entries[-1]
    tickers: list[dict[str, Any]] = []
    for sym, data in (last.get("tickers", {}) or {}).items():
        signals = data.get("signals", {}) or {}
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


def _pnl_footer(dd: Path) -> dict[str, Any]:
    """Single-line P&L for the deprioritised portfolio footer."""
    try:
        ps = load_json(dd / "portfolio_state.json", default={}) or {}
        pb = load_json(dd / "portfolio_state_bold.json", default={}) or {}
    except Exception as e:
        return {"error": f"pnl load: {type(e).__name__}: {e}"}
    return {
        "patient_value_sek": ps.get("portfolio_value", ps.get("equity_sek")),
        "bold_value_sek": pb.get("portfolio_value", pb.get("equity_sek")),
        "patient_starting_sek": ps.get("starting_capital", 500_000.0),
        "bold_starting_sek": pb.get("starting_capital", 500_000.0),
    }


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
