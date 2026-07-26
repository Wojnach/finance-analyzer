#!/usr/bin/env python3
"""Propose per-instrument component enablement from measured accuracy.

The component registry (portfolio/component_registry.py, live since
2026-07-18) makes per-(ticker, signal, horizon) enablement *data*: an
overlay at data/control/registry_overrides.json. This script reads the
measured per-ticker accuracy caches and proposes what that overlay should
contain for one instrument — the "research what an instrument benefits
most from, enable only those" step.

It NEVER writes the live overlay by itself. Default output is a proposal
(stdout table + optional JSON); --write requires --yes and writes the
overlay atomically after printing the same table. Every decision carries
the numbers it was made from, so the diff is auditable.

Statistics
----------
A raw hit-rate is not evidence: shannon_entropy showed 93.8% on n=16 for
XAG-USD 1d on 2026-07-19, which is 12 coin flips away from meaningless.
Decisions therefore use the **Wilson score interval lower bound** at 95%,
matching the LLM keep-policy already in force ("Wilson CI-low >= 60%"):

  ENABLE   ci_low >= keep_bar (default 60%)   — earns a vote
  DISABLE  ci_high < gate      (default 47%)  — provably worse than the gate
  KEEP     otherwise, currently enabled       — inconclusive, leave alone
  WATCH    otherwise, currently disabled      — inconclusive, stay off

"Inconclusive" is the common case with small n and is deliberately a
no-op: this tool only moves a component when the interval clears a bar,
so it cannot curve-fit a decision out of noise.

Usage
-----
    python scripts/tune_instrument.py --ticker XAG-USD
    python scripts/tune_instrument.py --ticker XAG-USD --min-samples 50
    python scripts/tune_instrument.py --ticker XAG-USD --json proposal.json
    python scripts/tune_instrument.py --ticker XAG-USD --write --yes

Data sources (read-only):
  data/ticker_signal_accuracy_cache.json  per-ticker per-signal per-horizon
  portfolio.component_registry            current enablement + reasons
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from portfolio.component_registry import get_registry  # noqa: E402
from portfolio.file_utils import atomic_write_json, load_json  # noqa: E402

TICKER_ACC_FILE = _REPO / "data" / "ticker_signal_accuracy_cache.json"
OVERLAY_FILE = _REPO / "data" / "control" / "registry_overrides.json"

DEFAULT_KEEP_BAR_PCT = 60.0
DEFAULT_GATE_PCT = 47.0
DEFAULT_MIN_SAMPLES = 30
Z_95 = 1.959963985

ENABLE, DISABLE, KEEP, WATCH, SKIP = "ENABLE", "DISABLE", "KEEP", "WATCH", "SKIP"


def wilson_interval(correct: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion, in percent.

    Wilson (not normal-approximation) because n is often <50 here and the
    normal interval misbehaves badly near 0/1 and at small n.
    """
    if total <= 0:
        return (0.0, 100.0)
    p = correct / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))) / denom
    lo = max(0.0, center - margin) * 100.0
    hi = min(1.0, center + margin) * 100.0
    return (lo, hi)


def data_span_days() -> float | None:
    """Span of the snapshot history in days, or None if unreadable.

    signal_log.db is authoritative (the JSONL is rotated to ~80 lines). The
    span bounds how many INDEPENDENT outcome windows can exist per horizon —
    see effective_windows().
    """
    db = _REPO / "data" / "signal_log.db"
    if not db.exists():
        return None
    try:
        import sqlite3
        from datetime import datetime

        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT MIN(ts), MAX(ts) FROM snapshots").fetchone()
        finally:
            con.close()
        if not row or not row[0] or not row[1]:
            return None
        lo = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
        hi = datetime.fromisoformat(str(row[1]).replace("Z", "+00:00"))
        return max(0.0, (hi - lo).total_seconds() / 86400.0)
    except Exception:
        return None


def _horizons(cache: dict) -> list[str]:
    return sorted(k for k, v in cache.items() if isinstance(v, dict) and k != "time")


HORIZON_DAYS = {
    "3h": 0.125,
    "4h": 0.1667,
    "12h": 0.5,
    "1d": 1.0,
    "3d": 3.0,
    "5d": 5.0,
    "10d": 10.0,
}

# Minimum NON-OVERLAPPING outcome windows required before a decision is
# allowed. See effective_windows() for why raw n lies.
DEFAULT_MIN_WINDOWS = 8.0


def effective_windows(horizon: str, span_days: float | None) -> float | None:
    """How many independent outcome windows the data span can actually hold.

    THE TRAP THIS CLOSES (found 2026-07-26): the accuracy caches counted 338
    "3d" outcomes inside a 6.7-day span. That is impossible as independent
    evidence — only ~2 non-overlapping 3-day windows fit. Snapshots are 600s
    apart and each looks `horizon` ahead, so consecutive rows share ~99.8% of
    their outcome window. Wilson assumes INDEPENDENT trials, so on such data
    it reported CI[0.0, 5.1] for drift_regime_gate@3d — "provably worse than
    chance" — when the truth was one market move counted 71 times.

    Same failure class as the LLM backtest's "1d candles + 8h step => 3
    identical prompts/day => effective n ~= 1/3 nominal".

    Returns None when the span is unknown (then the caller must not rely on
    the overlap gate). Windows, not samples: a fractional value < 1 means the
    data cannot even contain one complete outcome window.
    """
    if span_days is None:
        return None
    hz_days = HORIZON_DAYS.get(horizon)
    if not hz_days:
        return None
    return span_days / hz_days


def classify(
    correct: int,
    total: int,
    currently_enabled: bool,
    *,
    keep_bar: float,
    gate: float,
    min_samples: int,
    windows: float | None = None,
    min_windows: float = DEFAULT_MIN_WINDOWS,
) -> tuple[str, str]:
    """Return (decision, reason) for one (signal, horizon) cell."""
    if total < min_samples:
        state = "enabled" if currently_enabled else "disabled"
        return (
            SKIP,
            f"n={total} < {min_samples} — no evidence either way, leaving {state}",
        )
    # Overlap gate BEFORE the interval: raw n is meaningless when the data
    # span cannot hold enough independent outcome windows.
    if windows is not None and windows < min_windows:
        state = "enabled" if currently_enabled else "disabled"
        return (
            SKIP,
            f"n={total} but only ~{windows:.1f} independent outcome windows fit "
            f"the data span (need {min_windows:.0f}) — overlapping samples, "
            f"not evidence; leaving {state}",
        )
    # Interval on the EFFECTIVE sample size, not the raw one. With 71 samples
    # spread over ~54 independent windows, ~17 of them are re-measurements of
    # a window already counted; crediting all 71 to the interval overstates
    # confidence. Downscale (keeping the observed rate) so the CI reflects
    # independent evidence only.
    pct = 100.0 * correct / total
    n_eff, c_eff = total, correct
    if windows is not None and windows < total:
        n_eff = max(1, int(windows))
        c_eff = int(round(n_eff * correct / total))
    lo, hi = wilson_interval(c_eff, n_eff)
    win = (
        f", ~{windows:.0f} indep windows → n_eff={n_eff}"
        if windows is not None and n_eff != total
        else (f", ~{windows:.0f} indep windows" if windows is not None else "")
    )
    stat = f"{pct:.1f}% (n={total}{win}, CI {lo:.1f}-{hi:.1f})"
    if lo >= keep_bar:
        return (
            (KEEP if currently_enabled else ENABLE),
            f"{stat} — CI-low clears {keep_bar:.0f}% keep-bar",
        )
    if hi < gate:
        return (
            (DISABLE if currently_enabled else WATCH),
            f"{stat} — CI-high below {gate:.0f}% gate, provably worse than chance-gate",
        )
    return (
        (KEEP if currently_enabled else WATCH),
        f"{stat} — inconclusive (CI straddles the bars), no change",
    )


def analyse(
    ticker: str,
    *,
    keep_bar: float = DEFAULT_KEEP_BAR_PCT,
    gate: float = DEFAULT_GATE_PCT,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_windows: float = DEFAULT_MIN_WINDOWS,
    acc_file: Path | None = None,
    span_days: float | None = None,
) -> dict[str, Any]:
    """Build the full proposal for one ticker. Pure: reads, never writes."""
    if span_days is None:
        span_days = data_span_days()
    cache = load_json(acc_file or TICKER_ACC_FILE, default={}) or {}
    if not isinstance(cache, dict):
        cache = {}
    reg = get_registry()
    horizons = _horizons(cache)

    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        per_ticker = cache.get(horizon) or {}
        signals = per_ticker.get(ticker) or {}
        if not isinstance(signals, dict):
            continue
        for signal, stats in sorted(signals.items()):
            if not isinstance(stats, dict):
                continue
            correct = int(stats.get("correct") or 0)
            total = int(stats.get("total") or 0)
            enabled = reg.is_enabled(signal, ticker, horizon)
            windows = effective_windows(horizon, span_days)
            decision, reason = classify(
                correct,
                total,
                enabled,
                keep_bar=keep_bar,
                gate=gate,
                min_samples=min_samples,
                windows=windows,
                min_windows=min_windows,
            )
            # Same effective-n downscale the decision used, so the grid's CI
            # columns can never disagree with the decision's stated reason.
            if total:
                n_eff, c_eff = total, correct
                if windows is not None and windows < total:
                    n_eff = max(1, int(windows))
                    c_eff = int(round(n_eff * correct / total))
                lo, hi = wilson_interval(c_eff, n_eff)
            else:
                lo, hi, n_eff = 0.0, 100.0, 0
            rows.append(
                {
                    "signal": signal,
                    "horizon": horizon,
                    "correct": correct,
                    "total": total,
                    "pct": round(100.0 * correct / total, 1) if total else None,
                    "ci_low": round(lo, 1),
                    "ci_high": round(hi, 1),
                    "indep_windows": (
                        round(windows, 2) if windows is not None else None
                    ),
                    "n_effective": n_eff,
                    "currently_enabled": enabled,
                    "decision": decision,
                    "reason": reason,
                }
            )

    changes = [r for r in rows if r["decision"] in (ENABLE, DISABLE)]
    return {
        "ticker": ticker,
        "generated_ts": time.time(),
        "data_ts": (
            cache.get("time") if isinstance(cache.get("time"), (int, float)) else None
        ),
        "thresholds": {
            "keep_bar_pct": keep_bar,
            "gate_pct": gate,
            "min_samples": min_samples,
            "min_indep_windows": min_windows,
            "interval": "wilson-95",
        },
        "data_span_days": (round(span_days, 2) if span_days is not None else None),
        "horizons": horizons,
        "rows": rows,
        "changes": changes,
        "overlay_patch": build_overlay_patch(ticker, changes),
    }


def build_overlay_patch(ticker: str, changes: list[dict]) -> dict[str, Any]:
    """Translate ENABLE/DISABLE rows into the registry overlay schema.

    Per-horizon decisions land in ``horizons`` (the registry's per-horizon
    key), never the blunt top-level ``enabled`` — a signal good at 3h and
    bad at 1d must stay split, which is the whole point of per-instrument
    tuning.
    """
    patch: dict[str, Any] = {}
    for row in changes:
        entry = patch.setdefault(row["signal"], {"horizons": {}, "reason": ""})
        entry["horizons"][row["horizon"]] = row["decision"] == ENABLE
        note = f"{row['horizon']}: {row['reason']}"
        entry["reason"] = f"{entry['reason']}; {note}" if entry["reason"] else note
    for entry in patch.values():
        entry["reason"] = f"tune_instrument.py — {entry['reason']}"
    return {ticker: patch} if patch else {}


def format_table(proposal: dict) -> str:
    """Human-readable report — changes first, then the full evidence grid."""
    out: list[str] = []
    t = proposal["thresholds"]
    age = ""
    if proposal.get("data_ts"):
        age = f", data {(time.time() - proposal['data_ts']) / 3600:.1f}h old"
    span = proposal.get("data_span_days")
    span_s = f", span {span:.1f}d" if span is not None else ", span UNKNOWN"
    out.append(
        f"=== {proposal['ticker']} component tuning proposal "
        f"(keep>={t['keep_bar_pct']:.0f}% gate<{t['gate_pct']:.0f}% "
        f"min_n={t['min_samples']} min_windows={t.get('min_indep_windows', '?')}, "
        f"{t['interval']}{age}{span_s}) ==="
    )
    if span is not None:
        starved = sorted(
            {
                r["horizon"]
                for r in proposal["rows"]
                if r.get("indep_windows") is not None
                and r["indep_windows"] < t.get("min_indep_windows", 0)
            }
        )
        if starved:
            out.append(
                f"    NOTE: horizons {', '.join(starved)} have too few independent "
                f"windows in a {span:.1f}d span — their samples overlap, so no "
                f"decision is taken there (raw n is NOT evidence)."
            )
    changes = proposal["changes"]
    out.append("")
    if not changes:
        out.append("NO CHANGES PROPOSED — nothing clears a bar with enough samples.")
    else:
        out.append(f"PROPOSED CHANGES ({len(changes)}):")
        for r in changes:
            out.append(
                f"  {r['decision']:7} {r['signal']:28} @{r['horizon']:3} "
                f"{r['pct']:>5.1f}% n={r['total']:<5} CI[{r['ci_low']:.1f},{r['ci_high']:.1f}]"
            )
            out.append(f"          └ {r['reason']}")
    out.append("")
    out.append("FULL EVIDENCE GRID:")
    hdr = (
        f"  {'signal':28} {'hz':>3} {'acc':>6} {'n':>5} "
        f"{'CI-low':>7} {'CI-high':>8} {'now':>8}  decision"
    )
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))
    for r in sorted(proposal["rows"], key=lambda x: (-(x["total"] or 0), x["signal"])):
        pct = f"{r['pct']:.1f}%" if r["pct"] is not None else "—"
        out.append(
            f"  {r['signal'][:28]:28} {r['horizon']:>3} {pct:>6} {r['total']:>5} "
            f"{r['ci_low']:>7.1f} {r['ci_high']:>8.1f} "
            f"{'enabled' if r['currently_enabled'] else 'disabled':>8}  {r['decision']}"
        )
    counts: dict[str, int] = {}
    for r in proposal["rows"]:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    out.append("")
    out.append("  totals: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(out)


def merge_into_overlay(patch: dict, *, overlay_file: Path | None = None) -> dict:
    """Merge a patch into the existing overlay (per-signal horizons union).

    Existing operator entries for other signals/tickers are preserved —
    this is a merge, never a clobber.
    """
    path = overlay_file or OVERLAY_FILE
    current = load_json(path, default={}) or {}
    if not isinstance(current, dict):
        current = {}
    for ticker, signals in patch.items():
        tdict = current.setdefault(ticker, {})
        if not isinstance(tdict, dict):
            tdict = {}
            current[ticker] = tdict
        for signal, entry in signals.items():
            existing = tdict.setdefault(signal, {})
            if not isinstance(existing, dict):
                existing = {}
                tdict[signal] = existing
            hz = existing.setdefault("horizons", {})
            if not isinstance(hz, dict):
                hz = {}
                existing["horizons"] = hz
            hz.update(entry.get("horizons") or {})
            existing["reason"] = entry.get("reason") or existing.get("reason") or ""
    return current


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ticker", required=True, help="e.g. XAG-USD")
    ap.add_argument("--keep-bar", type=float, default=DEFAULT_KEEP_BAR_PCT)
    ap.add_argument("--gate", type=float, default=DEFAULT_GATE_PCT)
    ap.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    ap.add_argument(
        "--min-windows",
        type=float,
        default=DEFAULT_MIN_WINDOWS,
        help="minimum independent (non-overlapping) outcome windows required "
        "before a decision is allowed; guards against overlapping-sample "
        "overconfidence on long horizons",
    )
    ap.add_argument("--json", metavar="PATH", help="write the full proposal as JSON")
    ap.add_argument(
        "--write",
        action="store_true",
        help="merge the overlay patch into data/control/registry_overrides.json",
    )
    ap.add_argument("--yes", action="store_true", help="required with --write")
    args = ap.parse_args(argv)

    proposal = analyse(
        args.ticker,
        keep_bar=args.keep_bar,
        gate=args.gate,
        min_samples=args.min_samples,
        min_windows=args.min_windows,
    )
    print(format_table(proposal))

    if args.json:
        atomic_write_json(Path(args.json), proposal)
        print(f"\nproposal JSON -> {args.json}")

    if args.write:
        if not args.yes:
            print(
                "\nREFUSING to write: --write requires --yes (this changes live "
                "engine behavior via the registry overlay).",
                file=sys.stderr,
            )
            return 2
        patch = proposal["overlay_patch"]
        if not patch:
            print("\nnothing to write — no changes proposed.")
            return 0
        merged = merge_into_overlay(patch)
        atomic_write_json(OVERLAY_FILE, merged)
        print(
            f"\noverlay updated -> {OVERLAY_FILE} ({len(proposal['changes'])} changes)"
        )
        print("The engine picks this up live (mtime-cached overlay reload).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
