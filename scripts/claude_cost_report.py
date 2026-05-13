#!/usr/bin/env python3
"""Weekly Claude Code cost + invocation rollup.

Reads ``data/claude_invocations.jsonl`` (written by ``portfolio/claude_gate.py``
and ``data/metals_loop.py``) and ``data/invocations.jsonl`` (written by
``portfolio/agent_invocation.py``) and produces a per-day / per-caller /
per-model table of tokens, cost, duration, and outcomes.

Usage::

    .venv/Scripts/python.exe scripts/claude_cost_report.py [--days N] [--json]

``--days`` defaults to 7. ``--json`` emits a machine-readable summary instead
of the human-readable table.

Token + cost rows only exist for invocations that ran with ``--output-format
json`` and parsed successfully. Older rows (and any ``parse_ok=False`` row)
contribute to call counts and duration but not token/cost totals — those are
flagged separately so you can see how much of the picture is missing.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
GATE_LOG = DATA / "claude_invocations.jsonl"
LAYER2_LOG = DATA / "invocations.jsonl"


def _parse_ts(s: str) -> dt.datetime | None:
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        t = dt.datetime.fromisoformat(s)
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t
    except (ValueError, TypeError):
        return None


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
    return out


def collect(days: int) -> dict:
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    gate_rows = []
    for r in _load(GATE_LOG):
        t = _parse_ts(r.get("timestamp") or r.get("ts"))
        if t and t >= cutoff:
            r["_ts"] = t
            gate_rows.append(r)

    layer2_rows = []
    for r in _load(LAYER2_LOG):
        t = _parse_ts(r.get("ts") or r.get("timestamp"))
        if t and t >= cutoff:
            r["_ts"] = t
            layer2_rows.append(r)

    return {
        "cutoff": cutoff,
        "gate_rows": gate_rows,
        "layer2_rows": layer2_rows,
    }


def summarise(bundle: dict) -> dict:
    gate = bundle["gate_rows"]
    layer2 = bundle["layer2_rows"]

    # --- gate (claude_invocations.jsonl) totals
    parsed_rows = [r for r in gate if r.get("parse_ok")]
    by_caller = collections.defaultdict(
        lambda: dict(calls=0, input_tokens=0, output_tokens=0,
                     cache_read=0, cache_creation=0, cost_usd=0.0,
                     duration_s=0.0, parsed=0)
    )
    by_model = collections.defaultdict(
        lambda: dict(calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0,
                     duration_s=0.0)
    )
    by_day = collections.defaultdict(
        lambda: dict(calls=0, input_tokens=0, output_tokens=0, cost_usd=0.0,
                     duration_s=0.0)
    )
    by_status = collections.Counter()
    total_cost = 0.0
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_duration = 0.0

    for r in gate:
        caller = r.get("caller", "?")
        model = r.get("model", "?")
        day = r["_ts"].date().isoformat()
        status = r.get("status", "?")
        by_status[status] += 1
        by_caller[caller]["calls"] += 1
        by_model[model]["calls"] += 1
        by_day[day]["calls"] += 1
        dur = float(r.get("duration_seconds") or 0)
        by_caller[caller]["duration_s"] += dur
        by_model[model]["duration_s"] += dur
        by_day[day]["duration_s"] += dur
        total_duration += dur
        if r.get("parse_ok"):
            by_caller[caller]["parsed"] += 1
            ip = int(r.get("input_tokens") or 0)
            op = int(r.get("output_tokens") or 0)
            cr = int(r.get("cache_read_tokens") or 0)
            cc = int(r.get("cache_creation_tokens") or 0)
            cost = float(r.get("cost_usd") or 0)
            by_caller[caller]["input_tokens"] += ip
            by_caller[caller]["output_tokens"] += op
            by_caller[caller]["cache_read"] += cr
            by_caller[caller]["cache_creation"] += cc
            by_caller[caller]["cost_usd"] += cost
            by_model[model]["input_tokens"] += ip
            by_model[model]["output_tokens"] += op
            by_model[model]["cost_usd"] += cost
            by_day[day]["input_tokens"] += ip
            by_day[day]["output_tokens"] += op
            by_day[day]["cost_usd"] += cost
            total_cost += cost
            total_input += ip
            total_output += op
            total_cache_read += cr

    # --- layer2 (invocations.jsonl) totals (no tokens; tier/duration only)
    layer2_by_tier = collections.defaultdict(
        lambda: dict(calls=0, ran=0, duration_s=0.0)
    )
    layer2_by_status = collections.Counter()
    for r in layer2:
        tier = r.get("tier", "?")
        status = r.get("status", "?")
        layer2_by_status[status] += 1
        layer2_by_tier[tier]["calls"] += 1
        if r.get("duration_s") is not None:
            layer2_by_tier[tier]["ran"] += 1
            layer2_by_tier[tier]["duration_s"] += float(r["duration_s"])

    return {
        "totals": {
            "gate_rows": len(gate),
            "parsed_rows": len(parsed_rows),
            "parse_coverage_pct": round(100 * len(parsed_rows) / max(1, len(gate)), 1),
            "cost_usd": round(total_cost, 4),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "cache_read_tokens": total_cache_read,
            "duration_minutes": round(total_duration / 60, 1),
            "layer2_rows": len(layer2),
        },
        "by_caller": dict(by_caller),
        "by_model": dict(by_model),
        "by_day": dict(sorted(by_day.items())),
        "by_status": dict(by_status),
        "layer2_by_tier": dict(layer2_by_tier),
        "layer2_by_status": dict(layer2_by_status),
    }


def render_text(summary: dict, days: int) -> str:
    out = [f"=== Claude cost report — last {days}d ===\n"]
    t = summary["totals"]
    out.append(
        f"Gate rows: {t['gate_rows']} (parsed {t['parsed_rows']} = {t['parse_coverage_pct']}%)\n"
        f"Cost: ${t['cost_usd']:.4f} | Tokens in/out: {t['input_tokens']:,}/{t['output_tokens']:,}\n"
        f"Cache reads: {t['cache_read_tokens']:,} | Wall time: {t['duration_minutes']} min\n"
        f"Layer 2 rows: {t['layer2_rows']}\n"
    )

    out.append("\n--- By day ---")
    out.append(f"{'date':<12} {'calls':>6} {'cost($)':>10} {'in_tok':>10} {'out_tok':>10} {'wall(s)':>10}")
    for day, v in summary["by_day"].items():
        out.append(f"{day:<12} {v['calls']:>6} {v['cost_usd']:>10.4f} {v['input_tokens']:>10,} {v['output_tokens']:>10,} {v['duration_s']:>10.0f}")

    out.append("\n--- By caller ---")
    out.append(f"{'caller':<40} {'calls':>6} {'parsed':>7} {'cost($)':>10} {'in_tok':>10} {'out_tok':>10} {'wall(s)':>9}")
    by_caller_sorted = sorted(summary["by_caller"].items(), key=lambda kv: kv[1]["cost_usd"], reverse=True)
    for caller, v in by_caller_sorted:
        out.append(f"{caller[:40]:<40} {v['calls']:>6} {v['parsed']:>7} {v['cost_usd']:>10.4f} {v['input_tokens']:>10,} {v['output_tokens']:>10,} {v['duration_s']:>9.0f}")

    out.append("\n--- By model ---")
    out.append(f"{'model':<12} {'calls':>6} {'cost($)':>10} {'in_tok':>10} {'out_tok':>10} {'wall(s)':>10}")
    for model, v in sorted(summary["by_model"].items(), key=lambda kv: kv[1]["cost_usd"], reverse=True):
        out.append(f"{model:<12} {v['calls']:>6} {v['cost_usd']:>10.4f} {v['input_tokens']:>10,} {v['output_tokens']:>10,} {v['duration_s']:>10.0f}")

    out.append("\n--- Gate status counts ---")
    for s, c in sorted(summary["by_status"].items(), key=lambda kv: -kv[1]):
        out.append(f"  {s}: {c}")

    out.append("\n--- Layer 2 invocations by tier ---")
    out.append(f"{'tier':<6} {'calls':>6} {'ran':>6} {'wall(s)':>10}")
    for tier, v in sorted(summary["layer2_by_tier"].items()):
        out.append(f"T{tier!s:<5} {v['calls']:>6} {v['ran']:>6} {v['duration_s']:>10.0f}")

    out.append("\n--- Layer 2 status counts ---")
    for s, c in sorted(summary["layer2_by_status"].items(), key=lambda kv: -kv[1]):
        out.append(f"  {s}: {c}")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = ap.parse_args()

    bundle = collect(args.days)
    summary = summarise(bundle)

    if args.json:
        print(json.dumps(summary, default=str, indent=2))
    else:
        print(render_text(summary, args.days))
    return 0


if __name__ == "__main__":
    sys.exit(main())
