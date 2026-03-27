"""Backtest: replay historical signal snapshots through old vs new consensus.

Usage:
    .venv/Scripts/python.exe -m portfolio.backtester [--horizon 1d] [--days 30]

Compares:
  OLD: simple majority voting (no accuracy weighting)
  NEW: _weighted_consensus with accuracy gate + EWMA + regime + utility + MWU
"""

import argparse
import logging
from datetime import UTC, datetime, timedelta

logger = logging.getLogger("portfolio.backtester")

HORIZONS = ["3h", "4h", "12h", "1d", "3d", "5d", "10d"]


def _old_consensus(votes):
    """Original simple majority voting (no weighting)."""
    buy = sum(1 for v in votes.values() if v == "BUY")
    sell = sum(1 for v in votes.values() if v == "SELL")
    if buy > sell:
        return "BUY"
    elif sell > buy:
        return "SELL"
    return "HOLD"


def _build_accuracy_data(horizon="1d"):
    """Build blended accuracy_data dict matching signal_engine's EWMA blend.

    ARCH-23: Uses shared blend_accuracy_data() from accuracy_stats to avoid
    duplicating the blending logic.
    """
    from portfolio.accuracy_stats import (
        blend_accuracy_data,
        load_cached_accuracy,
        signal_accuracy,
        signal_accuracy_recent,
        write_accuracy_cache,
    )

    acc_horizon = horizon if horizon in ("3h", "4h", "12h") else "1d"

    alltime = load_cached_accuracy(acc_horizon)
    if not alltime:
        alltime = signal_accuracy(acc_horizon)
        if alltime:
            write_accuracy_cache(acc_horizon, alltime)

    recent = load_cached_accuracy(f"{acc_horizon}_recent")
    if not recent:
        recent = signal_accuracy_recent(acc_horizon, days=7)
        if recent:
            write_accuracy_cache(f"{acc_horizon}_recent", recent)

    return blend_accuracy_data(alltime, recent)


def run_backtest(horizon="1d", days=None):
    """Run the full backtest comparing old vs new consensus.

    Args:
        horizon: Outcome horizon to evaluate.
        days: Lookback window in days (None = all available data).

    Returns:
        dict with per-horizon and per-signal comparison results.
    """
    from portfolio.accuracy_stats import _vote_correct, load_entries
    from portfolio.signal_engine import (
        ACCURACY_GATE_THRESHOLD,
        _weighted_consensus,
    )
    from portfolio.tickers import DISABLED_SIGNALS, SIGNAL_NAMES

    entries = load_entries()

    # Apply time filter if requested
    cutoff = None
    if days is not None:
        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

    if cutoff:
        entries = [e for e in entries if e.get("ts", "") >= cutoff]

    if not entries:
        return {"error": "No entries found", "entries": 0}

    # Pre-build accuracy_data for new consensus (based on the target horizon)
    accuracy_data = _build_accuracy_data(horizon)

    # Collect results per horizon and per signal (1d only for signal breakdown)
    # Structure: {horizon: {old_correct, old_total, new_correct, new_total}}
    hz_results = {
        h: {"old_correct": 0, "old_total": 0, "new_correct": 0, "new_total": 0}
        for h in HORIZONS
    }

    # Per-signal breakdown (1d horizon only for brevity)
    sig_results = {
        s: {"old_correct": 0, "old_total": 0, "new_correct": 0, "new_total": 0}
        for s in SIGNAL_NAMES
        if s not in DISABLED_SIGNALS
    }

    # Track date range
    min_ts = None
    max_ts = None

    for entry in entries:
        ts = entry.get("ts", "")
        if ts:
            if min_ts is None or ts < min_ts:
                min_ts = ts
            if max_ts is None or ts > max_ts:
                max_ts = ts

        outcomes = entry.get("outcomes", {})
        tickers_data = entry.get("tickers", {})

        for ticker, tdata in tickers_data.items():
            votes = tdata.get("signals", {})
            if not votes:
                continue

            # Filter out disabled signals for old consensus too
            active_votes = {k: v for k, v in votes.items() if k not in DISABLED_SIGNALS}
            if not active_votes:
                continue

            # Regime for new consensus (may not exist in old entries)
            regime = tdata.get("regime", "unknown")

            # Compute old consensus (simple majority)
            old_action = _old_consensus(active_votes)

            # Compute new consensus (weighted)
            new_action, _ = _weighted_consensus(
                active_votes,
                accuracy_data,
                regime,
                activation_rates=None,
                accuracy_gate=ACCURACY_GATE_THRESHOLD,
            )

            # Evaluate against each horizon
            for h in HORIZONS:
                outcome = outcomes.get(ticker, {}).get(h)
                if not outcome:
                    continue
                change_pct = outcome.get("change_pct", 0)

                if old_action != "HOLD":
                    result_val = _vote_correct(old_action, change_pct)
                    if result_val is not None:
                        hz_results[h]["old_total"] += 1
                        if result_val:
                            hz_results[h]["old_correct"] += 1

                if new_action != "HOLD":
                    result_val = _vote_correct(new_action, change_pct)
                    if result_val is not None:
                        hz_results[h]["new_total"] += 1
                        if result_val:
                            hz_results[h]["new_correct"] += 1

            # Per-signal accuracy on target horizon
            target_outcome = outcomes.get(ticker, {}).get(horizon)
            if target_outcome:
                change_pct = target_outcome.get("change_pct", 0)
                for sig_name in sig_results:
                    vote = votes.get(sig_name, "HOLD")
                    if vote == "HOLD":
                        continue
                    result_val = _vote_correct(vote, change_pct)
                    if result_val is None:
                        continue

                    # Old: always counts this signal's vote
                    sig_results[sig_name]["old_total"] += 1
                    if result_val:
                        sig_results[sig_name]["old_correct"] += 1

                    # New: only counts if signal is not gated
                    sig_data = accuracy_data.get(sig_name, {})
                    sig_acc = sig_data.get("accuracy", 0.5)
                    sig_samples = sig_data.get("total", 0)
                    from portfolio.signal_engine import ACCURACY_GATE_MIN_SAMPLES
                    if sig_samples >= ACCURACY_GATE_MIN_SAMPLES and sig_acc < ACCURACY_GATE_THRESHOLD:
                        pass  # gated out — doesn't count in new system
                    else:
                        sig_results[sig_name]["new_total"] += 1
                        if result_val:
                            sig_results[sig_name]["new_correct"] += 1

    return {
        "entries": len(entries),
        "min_ts": min_ts,
        "max_ts": max_ts,
        "horizon_results": hz_results,
        "signal_results": sig_results,
    }


def _safe_div(a, b, default=0.0):
    return a / b if b > 0 else default


def print_report(results, target_horizon="1d"):
    """Print a formatted comparison report."""
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    entries = results["entries"]
    min_ts = results.get("min_ts", "?")
    max_ts = results.get("max_ts", "?")

    # Parse date range
    try:
        d1 = datetime.fromisoformat(min_ts).date()
        d2 = datetime.fromisoformat(max_ts).date()
        days_span = (d2 - d1).days
        period_str = f"{d1} to {d2} ({days_span} days, {entries} entries)"
    except Exception:
        period_str = f"{entries} entries"

    print("=== BACKTEST: Old vs New Consensus ===")
    print(f"Period: {period_str}")
    print()

    # Per-horizon table
    hz_results = results["horizon_results"]
    print(f"{'Horizon':<10} {'Old Acc':>9} {'New Acc':>9} {'Delta':>8} {'Old N':>8} {'New N':>8}")
    print(f"{'-------':<10} {'-------':>9} {'-------':>9} {'-----':>8} {'-----':>8} {'-----':>8}")
    for h in HORIZONS:
        r = hz_results[h]
        old_n = r["old_total"]
        new_n = r["new_total"]
        if old_n == 0 and new_n == 0:
            continue
        old_acc = _safe_div(r["old_correct"], old_n)
        new_acc = _safe_div(r["new_correct"], new_n)
        delta = new_acc - old_acc
        marker = " *" if h == target_horizon else ""
        print(
            f"{h + marker:<10} {old_acc*100:>8.1f}% {new_acc*100:>8.1f}% {delta*100:>+7.1f}% {old_n:>8,} {new_n:>8,}"
        )

    print()
    print(f"Per-Signal Accuracy ({target_horizon} horizon):")
    print(f"{'Signal':<22} {'Old':>8} {'New':>8} {'Delta':>8} {'Old N':>7} {'New N':>7}")
    print(f"{'------':<22} {'---':>8} {'---':>8} {'-----':>8} {'-----':>7} {'-----':>7}")

    sig_results = results["signal_results"]
    # Sort by new accuracy desc
    sig_sorted = sorted(
        sig_results.items(),
        key=lambda x: _safe_div(x[1]["new_correct"], x[1]["new_total"]),
        reverse=True,
    )
    for sig_name, r in sig_sorted:
        old_n = r["old_total"]
        new_n = r["new_total"]
        if old_n < 5 and new_n < 5:
            continue  # skip signals with no data
        old_acc = _safe_div(r["old_correct"], old_n)
        # Signal gated in new system when it had old data but new_n == 0
        if old_n >= 5 and new_n == 0:
            new_acc_str = "  GATED"
            delta_str = "       "
        else:
            new_acc = _safe_div(r["new_correct"], new_n)
            delta = new_acc - old_acc
            new_acc_str = f"{new_acc*100:>7.1f}%"
            delta_str = f"{delta*100:>+7.1f}%"
        print(
            f"{sig_name:<22} {old_acc*100:>7.1f}% {new_acc_str} {delta_str} {old_n:>7,} {new_n:>7,}"
        )

    # Summary
    print()
    r_target = hz_results.get(target_horizon, {})
    old_acc_t = _safe_div(r_target.get("old_correct", 0), r_target.get("old_total", 0))
    new_acc_t = _safe_div(r_target.get("new_correct", 0), r_target.get("new_total", 0))
    delta_t = new_acc_t - old_acc_t
    print(f"=== Summary ({target_horizon}) ===")
    print(f"Old system accuracy: {old_acc_t*100:.1f}% ({r_target.get('old_total', 0):,} samples)")
    print(f"New system accuracy: {new_acc_t*100:.1f}% ({r_target.get('new_total', 0):,} samples)")
    print(f"Improvement: {delta_t*100:+.1f}pp")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(
        description="Backtest old vs new consensus systems on historical signal data."
    )
    parser.add_argument(
        "--horizon",
        default="1d",
        choices=HORIZONS,
        help="Primary horizon to evaluate (default: 1d)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Lookback window in days (default: all available data)",
    )
    args = parser.parse_args()

    results = run_backtest(horizon=args.horizon, days=args.days)
    print_report(results, target_horizon=args.horizon)
