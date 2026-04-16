"""Counterfactual consensus replay — validate Batch 1-4 gating reconfig.

Replays signal_log.jsonl with the new gating logic and compares the implied
consensus against the logged historical consensus. Quantifies the expected
accuracy improvement before the fix goes live.

Usage:
    python scripts/replay_consensus.py --days 30 --horizon 1d
    python scripts/replay_consensus.py --days 14 --horizon 3h --out data/consensus_replay.json

Output: summary to stdout; full JSON report to --out path.

Notes:
- The replay is a LOWER-BOUND estimate. The new code paths (per-horizon
  blacklist, circuit breaker relaxation) activate on specific conditions
  that aren't always met in historical data. Expect the LIVE improvement
  to be at least as large as the replay estimate.
- HOLD consensus is counted but doesn't factor into accuracy — only
  BUY/SELL are judged against the price outcome.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure project root on sys.path when invoked as a script from anywhere.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from portfolio.file_utils import atomic_write_json  # noqa: E402

LOG_FILE = _ROOT / "data" / "signal_log.jsonl"


def _load_entries(days: int):
    """Stream signal_log entries from the last N days."""
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Missing signal log: {LOG_FILE}")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    entries = []
    with LOG_FILE.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = e.get("ts", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
            if dt < cutoff:
                continue
            entries.append(e)
    return entries


def _verdict_correct(consensus: str, change_pct: float) -> bool | None:
    """Return True if consensus direction matches sign of change_pct.

    Returns None for HOLD or missing change_pct — these rows are not counted
    in accuracy computations.
    """
    if consensus not in ("BUY", "SELL"):
        return None
    if change_pct is None:
        return None
    if consensus == "BUY":
        return change_pct > 0
    return change_pct < 0


def _simulate_consensus_with_new_rules(tickers_snapshot: dict, horizon: str) -> dict:
    """Rebuild consensus under new rules for each ticker in the snapshot.

    Uses the NEW _weighted_consensus path with horizon and ticker parameters —
    exercises Batches 1 (constants/MSTR trim), 2 (circuit breaker), and 4
    (horizon-specific blacklist). Batch 3's per-ticker override was already
    in place pre-reconfig, so no simulation delta expected from it.
    """
    # Import inside function so the module-load cost isn't paid for --help.
    from portfolio.accuracy_stats import (
        blend_accuracy_data,
        load_cached_accuracy,
    )
    from portfolio.signal_engine import _weighted_consensus

    # 2026-04-16 review (silent-failure-hunter P1): narrow the exception window
    # and surface failures. The previous bare `except Exception` silently
    # defaulted the cache to {}, which would make every simulated signal pass
    # the accuracy gate with a 0.5 default — producing a misleadingly optimistic
    # replay that looks great because nothing is gated. Now the cache-load
    # failure raises RuntimeError so the caller records it as a fatal error
    # instead of quietly proceeding.
    try:
        alltime = load_cached_accuracy(horizon) or {}
        recent = load_cached_accuracy(f"{horizon}_recent") or {}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"replay_consensus: failed to load accuracy cache for {horizon!r} - {exc}. "
            "Aborting replay rather than returning bogus numbers.",
        ) from exc
    acc_data_global = blend_accuracy_data(alltime, recent)

    result = {}
    for ticker, tdata in tickers_snapshot.items():
        votes = dict(tdata.get("signals", {}))
        regime = tdata.get("regime", "unknown")
        if not votes:
            result[ticker] = {"consensus": "HOLD", "conf": 0.0}
            continue
        # 2026-04-16 review (silent-failure-hunter P1): per-ticker failures now
        # record an explicit error row so the caller can count them and so
        # _verdict_correct won't silently drop the row. Upstream `replay()`
        # surfaces the error count in the summary output.
        try:
            action, conf = _weighted_consensus(
                votes,
                acc_data_global,
                regime,
                horizon=horizon,
                ticker=ticker,
            )
        except (KeyError, ValueError, TypeError) as exc:
            result[ticker] = {"consensus": "ERROR", "conf": 0.0, "error": str(exc)}
            continue
        result[ticker] = {"consensus": action, "conf": round(conf, 4)}
    return result


def replay(days: int, horizon: str) -> dict:
    """Run the replay and return a summary dict."""
    entries = _load_entries(days)
    actual_buckets = defaultdict(lambda: [0, 0])  # ticker -> [hits, total]
    simulated_buckets = defaultdict(lambda: [0, 0])
    agree_count = 0
    disagree_count = 0
    total_scored = 0
    # 2026-04-16 review: count per-ticker simulate errors so the summary can
    # surface them. Previously ERROR consensus rows silently dropped out.
    sim_error_count = 0
    sim_error_samples: list[tuple[str, str]] = []  # up to 5 (ticker, error) pairs

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})
        if not outcomes or not tickers:
            continue
        simulated = _simulate_consensus_with_new_rules(tickers, horizon)
        for ticker, tdata in tickers.items():
            actual = tdata.get("consensus", "HOLD")
            sim_entry = simulated.get(ticker, {})
            sim = sim_entry.get("consensus", "HOLD")
            if sim == "ERROR":
                sim_error_count += 1
                if len(sim_error_samples) < 5:
                    sim_error_samples.append((ticker, sim_entry.get("error", "")))

            outcome = outcomes.get(ticker, {}).get(horizon)
            if not isinstance(outcome, dict):
                continue
            change_pct = outcome.get("change_pct")
            if change_pct is None:
                continue

            actual_correct = _verdict_correct(actual, change_pct)
            sim_correct = _verdict_correct(sim, change_pct)

            if actual_correct is not None:
                actual_buckets[ticker][1] += 1
                if actual_correct:
                    actual_buckets[ticker][0] += 1
            if sim_correct is not None:
                simulated_buckets[ticker][1] += 1
                if sim_correct:
                    simulated_buckets[ticker][0] += 1

            if sim == "ERROR":
                # Don't score disagreement against a broken simulation row.
                continue
            if actual == sim:
                agree_count += 1
            else:
                disagree_count += 1
            total_scored += 1

    def _rate(bucket):
        out = {}
        for ticker, (hits, total) in bucket.items():
            out[ticker] = {
                "hits": hits,
                "total": total,
                "accuracy_pct": round(hits / total * 100, 2) if total else None,
            }
        return out

    actual_total_hits = sum(h for h, _ in actual_buckets.values())
    actual_total = sum(t for _, t in actual_buckets.values())
    sim_total_hits = sum(h for h, _ in simulated_buckets.values())
    sim_total = sum(t for _, t in simulated_buckets.values())

    summary = {
        "replayed_at": datetime.now(timezone.utc).isoformat(),
        "window_days": days,
        "horizon": horizon,
        "entries_considered": len(entries),
        "rows_scored": total_scored,
        "simulation_errors": {
            "count": sim_error_count,
            "samples": [{"ticker": t, "error": e} for t, e in sim_error_samples],
        },
        "consensus_agreement": {
            "agree": agree_count,
            "disagree": disagree_count,
            "agree_pct": round(agree_count / total_scored * 100, 2)
                if total_scored else None,
        },
        "actual_per_ticker": _rate(actual_buckets),
        "simulated_per_ticker": _rate(simulated_buckets),
        "aggregate_accuracy": {
            "actual": {
                "hits": actual_total_hits,
                "total": actual_total,
                "pct": round(actual_total_hits / actual_total * 100, 2)
                    if actual_total else None,
            },
            "simulated": {
                "hits": sim_total_hits,
                "total": sim_total,
                "pct": round(sim_total_hits / sim_total * 100, 2)
                    if sim_total else None,
            },
        },
    }
    return summary


def _print_report(summary: dict) -> None:
    print(f"Replay window: {summary['window_days']}d @ horizon {summary['horizon']}")
    print(f"Entries considered: {summary['entries_considered']}")
    print(f"Rows scored: {summary['rows_scored']}")
    sim_err = summary.get("simulation_errors", {})
    err_count = sim_err.get("count", 0)
    if err_count:
        samples = sim_err.get("samples", [])
        parts = [f"{s.get('ticker', '?')}:{(s.get('error') or '')[:40]}" for s in samples]
        print(f"[WARNING] Simulation errors: {err_count} "
              f"(first {len(samples)}: {', '.join(parts)}) "
              "- numbers below UNDER-REPORT the true simulated state.")
    agree = summary["consensus_agreement"]
    print(f"Agreement (actual vs simulated): "
          f"{agree['agree']}/{agree['agree'] + agree['disagree']} "
          f"({agree['agree_pct']}%)")
    agg = summary["aggregate_accuracy"]
    actual_pct = agg["actual"]["pct"]
    sim_pct = agg["simulated"]["pct"]
    if actual_pct is not None and sim_pct is not None:
        delta = round(sim_pct - actual_pct, 2)
        arrow = "+" if delta >= 0 else ""
        print(f"Aggregate accuracy:  actual={actual_pct}%  "
              f"simulated={sim_pct}%  delta={arrow}{delta}pp")
    print()
    print("Per-ticker (actual | simulated | delta):")
    all_tickers = sorted(set(summary["actual_per_ticker"].keys())
                          | set(summary["simulated_per_ticker"].keys()))
    for t in all_tickers:
        a = summary["actual_per_ticker"].get(t, {})
        s = summary["simulated_per_ticker"].get(t, {})
        a_pct = a.get("accuracy_pct")
        s_pct = s.get("accuracy_pct")
        if a_pct is None or s_pct is None:
            print(f"  {t:<12} insufficient data")
            continue
        delta = round(s_pct - a_pct, 2)
        arrow = "+" if delta >= 0 else ""
        print(f"  {t:<12} "
              f"actual={a_pct:>6.2f}% (n={a.get('total', 0):>4})  "
              f"simulated={s_pct:>6.2f}% (n={s.get('total', 0):>4})  "
              f"delta={arrow}{delta}pp")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=30,
                    help="Lookback window in days (default: 30)")
    ap.add_argument("--horizon", type=str, default="1d",
                    choices=["3h", "4h", "12h", "1d", "3d", "5d", "10d"],
                    help="Outcome horizon (default: 1d)")
    ap.add_argument("--out", type=str, default=None,
                    help="Write full JSON summary to this path")
    args = ap.parse_args(argv)

    summary = replay(days=args.days, horizon=args.horizon)
    _print_report(summary)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(out_path), summary)
        print(f"\nFull report written: {out_path}")


if __name__ == "__main__":
    main()
