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
    """Stream signal_log entries from the last N days.

    P3-8 (2026-04-17 adversarial review): return (entries, parse_error_count)
    so callers can surface malformed-row prevalence. A corrupted log
    previously silently produced fewer rows while the summary still claimed
    "30d window".
    """
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"Missing signal log: {LOG_FILE}")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    entries = []
    parse_errors = 0
    with LOG_FILE.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            ts = e.get("ts", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (ValueError, AttributeError, TypeError):
                parse_errors += 1
                continue
            if dt < cutoff:
                continue
            entries.append(e)
    return entries, parse_errors


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
    # P3-5/P3-4 (2026-04-17 adversarial review): when both caches are missing
    # (load_cached_accuracy returns None not raises), the previous narrow-
    # except branch silently produced an empty blend. Every simulated signal
    # would then default to 0.5 accuracy and vote freely - misleading but
    # silent. Raise explicitly so the caller knows to run the loop first.
    if not alltime and not recent:
        raise RuntimeError(
            f"replay_consensus: accuracy cache empty/missing for {horizon!r}. "
            "Run the loop at least once to populate the cache before replay, "
            "or point LOG_FILE at a workspace with populated accuracy_cache.json."
        )
    acc_data_global = blend_accuracy_data(alltime, recent)

    result = {}
    for ticker, tdata in tickers_snapshot.items():
        votes = dict(tdata.get("signals", {}))
        regime = tdata.get("regime", "unknown")
        if not votes:
            result[ticker] = {"consensus": "HOLD", "conf": 0.0}
            continue
        # P2-J (2026-04-17 adversarial review): broaden exception catch.
        # Previously narrow (KeyError/ValueError/TypeError) let AttributeError,
        # RuntimeError, etc. escape and crash the full replay loop. For a
        # per-ticker simulation error we want to record and continue; only
        # the outer RuntimeError on cache-load stays fail-fast.
        try:
            action, conf = _weighted_consensus(
                votes,
                acc_data_global,
                regime,
                horizon=horizon,
                ticker=ticker,
            )
        except Exception as exc:
            result[ticker] = {"consensus": "ERROR", "conf": 0.0, "error": str(exc)}
            continue
        result[ticker] = {"consensus": action, "conf": round(conf, 4)}
    return result


def replay(days: int, horizon: str) -> dict:
    """Run the replay and return a summary dict."""
    entries, parse_errors = _load_entries(days)
    actual_buckets = defaultdict(lambda: [0, 0])  # ticker -> [hits, total]
    simulated_buckets = defaultdict(lambda: [0, 0])
    agree_count = 0
    disagree_count = 0
    total_scored = 0
    # 2026-04-16 review: count per-ticker simulate errors so the summary can
    # surface them. Previously ERROR consensus rows silently dropped out.
    sim_error_count = 0
    sim_error_samples: list[tuple[str, str]] = []  # up to 5 (ticker, error) pairs
    # P1-A (2026-04-17 adversarial review): surface how often the regime
    # field was missing from signal_log. When most rows use "unknown"
    # regime, the circuit breaker's trending/high-vol paths are not
    # exercised - the replay result is a lower-bound estimate for those
    # paths.
    regime_counter = defaultdict(int)

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
                # Don't score disagreement against a broken simulation row,
                # and (Codex round 11 P3) don't count the regime either -
                # ERROR rows contribute to sim_error_count but not to
                # rows_scored, so they shouldn't inflate regime_distribution.
                continue

            # Codex round 11 P3: regime counter increments only for rows
            # that actually contribute to rows_scored (past outcome validation
            # AND past ERROR check). Previously an ERROR-simulation row still
            # inflated the regime distribution, false-negativing the
            # "under-exercised regime" warning in exactly the runs where
            # replay was already degraded.
            regime_counter[tdata.get("regime", "unknown")] += 1

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
        "parse_errors": parse_errors,
        "regime_distribution": dict(regime_counter),
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
    if summary.get("parse_errors", 0):
        print(f"[WARNING] Parse errors (skipped rows): {summary['parse_errors']}")
    print(f"Rows scored: {summary['rows_scored']}")
    # P1-A surface: show regime distribution so the reader knows which
    # circuit-breaker paths were exercised. "unknown" regime uses the
    # strictest quorum (5) - if that dominates, the trending/high-vol
    # recovery paths weren't tested by this replay.
    regime_dist = summary.get("regime_distribution", {})
    if regime_dist:
        total_regime = sum(regime_dist.values())
        top = sorted(regime_dist.items(), key=lambda x: -x[1])[:4]
        parts = [f"{r}={n}({n*100//max(total_regime,1)}%)" for r, n in top]
        print(f"Regime distribution: {', '.join(parts)}")
        unknown_pct = regime_dist.get("unknown", 0) * 100 // max(total_regime, 1)
        if unknown_pct > 50:
            print(f"[WARNING] {unknown_pct}% of rows have regime='unknown' — "
                  "the circuit-breaker's trending/high-vol recovery paths "
                  "are under-exercised. Replay result is a lower-bound "
                  "estimate for those regimes.")
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
