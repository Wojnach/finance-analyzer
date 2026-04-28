"""One-shot audit of which "dropped" signals are real degradation vs noise.

Reads the latest accuracy_degradation alert (from the live state file or
a freshly computed snapshot) and for each flagged signal/scope reports:
- Lifetime accuracy + N
- Most-recent 7-day accuracy + N
- Standard error of the difference
- A verdict: REAL (drop > 2*SE below lifetime) vs NOISE (within 2*SE)

The intent is to separate signals that genuinely regressed from those
that just happen to have a lucky-then-unlucky 7-day window.

Output: markdown table to ``docs/accuracy_audit_<YYYYMMDD>.md`` under the
data dir.

Usage:
    .venv/Scripts/python.exe scripts/audit_accuracy_drops.py \\
        [--data-dir PATH] [--output PATH]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from portfolio.accuracy_stats import (  # noqa: E402
    accuracy_by_ticker_signal_cached,
    consensus_accuracy,
    signal_accuracy,
)
from portfolio.accuracy_degradation import _per_ticker_recent  # noqa: E402


def _se_diff_pp(p1: float, n1: int, p2: float, n2: int) -> float:
    if n1 < 1 or n2 < 1:
        return 0.0
    var = p1 * (1.0 - p1) / n1 + p2 * (1.0 - p2) / n2
    if var <= 0.0:
        return 0.0
    return math.sqrt(var) * 100.0


def _load_signal_log(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _filter_window(entries: list[dict], lower: datetime, upper: datetime):
    lo, hi = lower.isoformat(), upper.isoformat()
    return [e for e in entries if lo < e.get("ts", "") <= hi]


def _filter_lifetime(entries: list[dict], cutoff: datetime):
    hi = cutoff.isoformat()
    return [e for e in entries if e.get("ts", "") <= hi]


def _alert_keys_from_state(state_path: Path) -> list[dict]:
    """Pull the most recent flagged alerts from degradation_alert_state.json."""
    if not state_path.exists():
        return []
    try:
        d = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return d.get("last_full_check_violations", [])[0:1] and \
        d["last_full_check_violations"][0].get("details", {}).get("alerts", []) or []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data",
                        help="Directory holding signal_log.jsonl + state files")
    parser.add_argument("--output", type=Path, default=None,
                        help="Markdown output path (default: docs/accuracy_audit_<date>.md)")
    args = parser.parse_args(argv)

    repo_data = args.data_dir
    signal_log = repo_data / "signal_log.jsonl"
    alert_state = repo_data / "degradation_alert_state.json"

    print(f"Loading signal_log from {signal_log}", file=sys.stderr)
    entries = _load_signal_log(signal_log)
    print(f"  {len(entries):,} entries", file=sys.stderr)

    now = datetime.now(UTC)
    last_7d_lower = now - timedelta(days=7)

    print("Computing lifetime + recent-7d accuracy...", file=sys.stderr)
    recent_entries = _filter_window(entries, last_7d_lower, now)
    lifetime_entries = _filter_lifetime(entries, now)

    # NOTE on "lifetime" semantics: ``entries`` here is whatever the
    # JSONL contains — the file is rotated/pruned, so this is closer to
    # "all available recent history" than a true unbounded lifetime.
    # The verdict is still meaningful — both arms (lifetime and
    # recent-7d) are computed against the same dataset, so the
    # difference and SE are internally consistent.
    lifetime_signals = signal_accuracy("1d", entries=lifetime_entries)
    recent_signals = signal_accuracy("1d", entries=recent_entries)
    recent_per_ticker = _per_ticker_recent("1d", days=7, entries=recent_entries)
    # Codex P3 fix (2026-04-28): per_ticker lifetime so MSTR::sentiment
    # compares against MSTR's own lifetime sentiment, not the global one.
    # Mixing populations (per-ticker recent vs global lifetime) inflates
    # or deflates the verdict for ticker-specific drops.
    # We compute lifetime per-ticker from ``lifetime_entries`` directly
    # (same dataset as signal_accuracy) so both arms are commensurable.
    # accuracy_by_ticker_signal_cached uses the default backend which is
    # often empty for one-shot script runs.
    try:
        lifetime_per_ticker = _per_ticker_recent(
            "1d", days=10_000, entries=lifetime_entries,
        )
    except Exception as e:
        print(f"  warn: per-ticker lifetime fetch failed: {e}", file=sys.stderr)
        try:
            lifetime_per_ticker = accuracy_by_ticker_signal_cached("1d")
        except Exception:
            lifetime_per_ticker = {}

    flagged = _alert_keys_from_state(alert_state)
    if not flagged:
        print("No flagged alerts in degradation_alert_state — auditing all "
              "signals with N>=200 and recent < 50%.", file=sys.stderr)
        flagged = []
        for name, stats in recent_signals.items():
            if (stats.get("total", 0) >= 200
                    and stats.get("accuracy", 0.0) < 0.50):
                flagged.append({"key": name, "scope": "signal"})

    rows: list[dict] = []
    for entry in flagged:
        key = entry.get("key", "?")
        scope = entry.get("scope", "?")

        if scope == "signal":
            life = lifetime_signals.get(key, {})
            rec = recent_signals.get(key, {})
        elif scope == "per_ticker":
            # key = "TICKER::signal_name". Codex P3 fix: lifetime must be
            # the per-ticker lifetime, not the global signal lifetime.
            ticker, _, signal_name = key.partition("::")
            life = (lifetime_per_ticker.get(ticker, {}) or {}).get(
                signal_name, {},
            )
            rec = (recent_per_ticker.get(ticker, {}) or {}).get(signal_name, {})
        elif scope == "consensus":
            life = consensus_accuracy("1d", entries=lifetime_entries)
            rec = consensus_accuracy("1d", entries=recent_entries)
        elif scope == "forecast":
            # Codex P3 fix round 1 (2026-04-28): forecast alerts have
            # keys like "chronos_24h" / "kronos_24h"; route through the
            # cached_forecast_accuracy infra the live writer uses.
            #
            # Codex round 2 P2 fix: degradation_alert_state.json stores
            # forecast keys with a "forecast::" prefix (per the live
            # check at accuracy_degradation.py:518), but
            # cached_forecast_accuracy returns dicts keyed by the bare
            # model name. Strip the prefix before lookup.
            from portfolio.forecast_accuracy import cached_forecast_accuracy
            bare_key = key
            if bare_key.startswith("forecast::"):
                bare_key = bare_key[len("forecast::"):]
            try:
                forecast_recent = cached_forecast_accuracy(
                    horizon="24h", days=7, use_raw_sub_signals=True,
                )
                forecast_lifetime = cached_forecast_accuracy(
                    horizon="24h", days=365, use_raw_sub_signals=True,
                )
            except Exception as e:
                print(f"  warn: forecast accuracy fetch failed: {e}",
                      file=sys.stderr)
                forecast_recent, forecast_lifetime = {}, {}
            life = forecast_lifetime.get(bare_key, {})
            rec = forecast_recent.get(bare_key, {})
        else:
            life, rec = {}, {}

        life_acc = float(life.get("accuracy", 0.0) or 0.0)
        life_n = int(life.get("total", 0) or 0)
        rec_acc = float(rec.get("accuracy", 0.0) or 0.0)
        rec_n = int(rec.get("total", 0) or 0)

        drop_vs_lifetime_pp = (life_acc - rec_acc) * 100.0
        se_pp = _se_diff_pp(life_acc, life_n, rec_acc, rec_n)
        is_real = drop_vs_lifetime_pp >= 2.0 * se_pp and drop_vs_lifetime_pp > 0
        verdict = "REAL" if is_real else "NOISE"

        rows.append({
            "key": key,
            "scope": scope,
            "lifetime_pct": round(life_acc * 100.0, 1),
            "lifetime_n": life_n,
            "recent_pct": round(rec_acc * 100.0, 1),
            "recent_n": rec_n,
            "drop_vs_lifetime_pp": round(drop_vs_lifetime_pp, 1),
            "se_pp": round(se_pp, 2),
            "verdict": verdict,
        })

    rows.sort(key=lambda r: -r["drop_vs_lifetime_pp"])

    out_path = args.output or (
        REPO_ROOT / "docs" / f"accuracy_audit_{now.strftime('%Y%m%d')}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# Accuracy audit — {now.date().isoformat()}\n\n")
        f.write(
            "Generated by `scripts/audit_accuracy_drops.py`. "
            "Compares each flagged signal's recent-7d accuracy to its "
            "lifetime accuracy, then classifies the drop as REAL (≥ 2 SE "
            "below lifetime) or NOISE (within 2 SE).\n\n"
        )
        f.write(
            "| Signal/Scope | Lifetime | N (life) | Recent 7d | N (recent) "
            "| Drop vs lifetime | SE | Verdict |\n"
        )
        f.write(
            "|---|---:|---:|---:|---:|---:|---:|---|\n"
        )
        for r in rows:
            f.write(
                f"| {r['scope']}::{r['key']} "
                f"| {r['lifetime_pct']}% | {r['lifetime_n']} "
                f"| {r['recent_pct']}% | {r['recent_n']} "
                f"| {r['drop_vs_lifetime_pp']}pp | {r['se_pp']}pp "
                f"| **{r['verdict']}** |\n"
            )
        f.write("\n## Summary\n\n")
        real = [r for r in rows if r["verdict"] == "REAL"]
        noise = [r for r in rows if r["verdict"] == "NOISE"]
        f.write(f"- **{len(real)} REAL** (≥ 2 SE below lifetime)\n")
        f.write(f"- **{len(noise)} NOISE** (within 2 SE)\n")
        if real:
            f.write(
                "\nReal degraders worth follow-up gating decisions:\n"
            )
            for r in real:
                f.write(
                    f"  - `{r['scope']}::{r['key']}` lifetime "
                    f"{r['lifetime_pct']}% → recent {r['recent_pct']}% "
                    f"(drop {r['drop_vs_lifetime_pp']}pp, SE {r['se_pp']}pp)\n"
                )

    print(f"Wrote {out_path}", file=sys.stderr)
    print(f"  {len(rows)} signals audited: "
          f"{len([r for r in rows if r['verdict'] == 'REAL'])} REAL, "
          f"{len([r for r in rows if r['verdict'] == 'NOISE'])} NOISE",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
