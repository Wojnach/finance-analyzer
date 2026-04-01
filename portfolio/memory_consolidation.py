"""Daily memory consolidation — autoDream pattern for trading insights.

Runs daily, reads the last N days of signal outcomes and trade decisions,
then produces a compact markdown file (<200 lines) that Layer 2 reads
at the start of each invocation.

Pattern: Orient -> Gather -> Consolidate -> Prune.
"""

import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.file_utils import load_jsonl_tail

logger = logging.getLogger("portfolio.memory_consolidation")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

DEFAULT_SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"
DEFAULT_JOURNAL = DATA_DIR / "layer2_journal.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "trading_insights.md"

MAX_OUTPUT_LINES = 200


def _load_recent_entries(path, days=7):
    """Load JSONL entries from the last N days, parsing ISO timestamps.

    Args:
        path: Path to a JSONL file with ``ts`` fields.
        days: Number of days to look back.

    Returns:
        list of dicts within the time window, chronological order.
    """
    path = Path(path)
    cutoff = datetime.now(UTC) - timedelta(days=days)

    # Read a generous tail — 7 days of signal logs at ~1 entry/min is ~10K
    entries = load_jsonl_tail(path, max_entries=50_000, tail_bytes=50_000_000)

    recent = []
    for entry in entries:
        ts_str = entry.get("ts", "")
        if not ts_str:
            continue
        try:
            dt = datetime.fromisoformat(ts_str)
            # Ensure timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            continue
        if dt >= cutoff:
            recent.append(entry)

    return recent


def _compute_signal_accuracy(entries):
    """Compute per-signal accuracy from entries that have outcomes.

    For each signal that voted BUY or SELL, checks whether the 1d price
    outcome moved in the predicted direction.

    Returns:
        dict mapping signal_name -> {"correct": int, "total": int, "accuracy": float}
    """
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, ticker_data in tickers.items():
            signals = ticker_data.get("signals", {})
            outcome = outcomes.get(ticker, {})

            # Prefer 1d outcome for accuracy
            horizon = outcome.get("1d", {})
            change_pct = horizon.get("change_pct")
            if change_pct is None:
                continue

            for signal_name, vote in signals.items():
                if vote == "HOLD":
                    continue

                stats[signal_name]["total"] += 1
                # BUY is correct if price went up, SELL if price went down
                if (vote == "BUY" and change_pct > 0) or (
                    vote == "SELL" and change_pct < 0
                ):
                    stats[signal_name]["correct"] += 1

    # Compute accuracy percentages
    result = {}
    for name, s in stats.items():
        total = s["total"]
        correct = s["correct"]
        accuracy = (correct / total * 100) if total > 0 else 0.0
        result[name] = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 1),
        }

    return result


def _compute_regime_stats(entries):
    """Count regime occurrences across all entries and tickers.

    Looks for ``regime`` in per-ticker data and at the top level.

    Returns:
        dict mapping regime_name -> {"count": int, "pct": float}
    """
    regime_counter = Counter()

    for entry in entries:
        # Top-level regime (journal entries)
        top_regime = entry.get("regime")
        if top_regime:
            regime_counter[top_regime] += 1

        # Per-ticker regime (signal log entries)
        tickers = entry.get("tickers", {})
        for _ticker, ticker_data in tickers.items():
            if not isinstance(ticker_data, dict):
                continue
            regime = ticker_data.get("regime")
            if regime:
                regime_counter[regime] += 1

    total = sum(regime_counter.values())
    result = {}
    for regime, count in regime_counter.most_common():
        pct = (count / total * 100) if total > 0 else 0.0
        result[regime] = {"count": count, "pct": round(pct, 1)}

    return result


def _compute_trade_stats(journal_entries):
    """Summarize trade decisions from journal entries.

    Returns:
        dict with keys: total_decisions, action_counts (Counter),
        strategy_counts (dict of strategy -> Counter).
    """
    action_counts = Counter()
    strategy_counts = defaultdict(Counter)

    for entry in journal_entries:
        decisions = entry.get("decisions", {})
        for strategy, decision in decisions.items():
            if not isinstance(decision, dict):
                continue
            action = decision.get("action", "UNKNOWN")
            action_counts[action] += 1
            strategy_counts[strategy][action] += 1

    return {
        "total_decisions": sum(action_counts.values()),
        "action_counts": dict(action_counts),
        "strategy_counts": {k: dict(v) for k, v in strategy_counts.items()},
    }


def _format_markdown(
    days, signal_accuracy, regime_stats, trade_stats,
    signal_entries_count, journal_entries_count,
):
    """Format the consolidated insights as markdown, enforcing max lines."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    lines.append(f"# Trading Insights -- Last {days} Days")
    lines.append("")
    lines.append(f"*Auto-generated: {now}*")
    lines.append(f"*Signal entries: {signal_entries_count} | "
                 f"Journal entries: {journal_entries_count}*")
    lines.append("")

    # --- Signal Performance ---
    lines.append("## Signal Performance")
    lines.append("")

    if signal_accuracy:
        # Best signals: >55% accuracy with 5+ samples
        best = sorted(
            [(n, s) for n, s in signal_accuracy.items()
             if s["accuracy"] > 55 and s["total"] >= 5],
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        )
        # Worst signals: <45% accuracy with 5+ samples
        worst = sorted(
            [(n, s) for n, s in signal_accuracy.items()
             if s["accuracy"] < 45 and s["total"] >= 5],
            key=lambda x: x[1]["accuracy"],
        )

        if best:
            lines.append("### Best Signals (>55%, 5+ samples)")
            lines.append("")
            lines.append("| Signal | Accuracy | Samples |")
            lines.append("|--------|----------|---------|")
            for name, s in best[:10]:
                lines.append(f"| {name} | {s['accuracy']}% | {s['total']} |")
            lines.append("")

        if worst:
            lines.append("### Worst Signals (<45%, 5+ samples)")
            lines.append("")
            lines.append("| Signal | Accuracy | Samples |")
            lines.append("|--------|----------|---------|")
            for name, s in worst[:10]:
                lines.append(f"| {name} | {s['accuracy']}% | {s['total']} |")
            lines.append("")

        if not best and not worst:
            lines.append("No signals with 5+ samples in the accuracy threshold range.")
            lines.append("")
    else:
        lines.append("No signal accuracy data available (no outcomes in log).")
        lines.append("")

    # --- Regime Summary ---
    lines.append("## Regime Summary")
    lines.append("")

    if regime_stats:
        lines.append("| Regime | Count | Share |")
        lines.append("|--------|-------|-------|")
        for regime, s in regime_stats.items():
            lines.append(f"| {regime} | {s['count']} | {s['pct']}% |")
        lines.append("")
    else:
        lines.append("No regime data available.")
        lines.append("")

    # --- Trade Decisions ---
    lines.append("## Trade Decisions")
    lines.append("")

    if trade_stats["total_decisions"] > 0:
        lines.append(f"Total decisions: {trade_stats['total_decisions']}")
        lines.append("")
        lines.append("| Action | Count |")
        lines.append("|--------|-------|")
        for action, count in sorted(
            trade_stats["action_counts"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            lines.append(f"| {action} | {count} |")
        lines.append("")

        # Per-strategy breakdown
        for strategy, counts in trade_stats["strategy_counts"].items():
            lines.append(f"**{strategy}**: "
                         + ", ".join(f"{a} {c}" for a, c in sorted(counts.items())))
        lines.append("")
    else:
        lines.append("No trade decisions in the period.")
        lines.append("")

    # --- Key Takeaways ---
    lines.append("## Key Takeaways")
    lines.append("")

    takeaways = []

    if signal_accuracy:
        best_all = sorted(
            [(n, s) for n, s in signal_accuracy.items() if s["total"] >= 5],
            key=lambda x: x[1]["accuracy"],
            reverse=True,
        )
        worst_all = sorted(
            [(n, s) for n, s in signal_accuracy.items() if s["total"] >= 5],
            key=lambda x: x[1]["accuracy"],
        )
        if best_all:
            b = best_all[0]
            takeaways.append(
                f"- Best signal: **{b[0]}** ({b[1]['accuracy']}% over {b[1]['total']} samples)"
            )
        if worst_all:
            w = worst_all[0]
            takeaways.append(
                f"- Worst signal: **{w[0]}** ({w[1]['accuracy']}% over {w[1]['total']} samples)"
            )

    if regime_stats:
        dominant = next(iter(regime_stats))
        takeaways.append(
            f"- Dominant regime: **{dominant}** ({regime_stats[dominant]['pct']}%)"
        )

    if trade_stats["total_decisions"] > 0:
        takeaways.append(
            f"- Total decisions: {trade_stats['total_decisions']} "
            f"({trade_stats['action_counts']})"
        )

    if not takeaways:
        takeaways.append("- Insufficient data for takeaways.")

    lines.extend(takeaways)
    lines.append("")

    # Prune to MAX_OUTPUT_LINES
    if len(lines) > MAX_OUTPUT_LINES:
        lines = lines[:MAX_OUTPUT_LINES - 2]
        lines.append("")
        lines.append("*(truncated to 200 lines)*")

    return "\n".join(lines)


def consolidate_insights(
    signal_log_path=None,
    journal_path=None,
    output_path=None,
    days=7,
):
    """Main entry point: Orient -> Gather -> Consolidate -> Prune.

    Args:
        signal_log_path: Path to signal_log.jsonl.
        journal_path: Path to layer2_journal.jsonl.
        output_path: Path to write trading_insights.md.
        days: Number of days to look back.

    Returns:
        dict with: best_signals, worst_signals, dominant_regime,
        total_decisions, entries_processed.
    """
    signal_log_path = Path(signal_log_path or DEFAULT_SIGNAL_LOG)
    journal_path = Path(journal_path or DEFAULT_JOURNAL)
    output_path = Path(output_path or DEFAULT_OUTPUT)

    # --- Orient: load recent data ---
    logger.info("Memory consolidation: loading last %d days of data", days)
    signal_entries = _load_recent_entries(signal_log_path, days=days)
    journal_entries = _load_recent_entries(journal_path, days=days)

    # --- Gather: compute stats ---
    signal_accuracy = _compute_signal_accuracy(signal_entries)
    regime_stats = _compute_regime_stats(signal_entries + journal_entries)
    trade_stats = _compute_trade_stats(journal_entries)

    # --- Consolidate: format markdown ---
    md = _format_markdown(
        days=days,
        signal_accuracy=signal_accuracy,
        regime_stats=regime_stats,
        trade_stats=trade_stats,
        signal_entries_count=len(signal_entries),
        journal_entries_count=len(journal_entries),
    )

    # --- Prune: write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    logger.info("Trading insights written to %s (%d lines)",
                output_path, md.count("\n") + 1)

    # Build summary return
    best_signals = sorted(
        [(n, s) for n, s in signal_accuracy.items()
         if s["accuracy"] > 55 and s["total"] >= 5],
        key=lambda x: x[1]["accuracy"],
        reverse=True,
    )
    worst_signals = sorted(
        [(n, s) for n, s in signal_accuracy.items()
         if s["accuracy"] < 45 and s["total"] >= 5],
        key=lambda x: x[1]["accuracy"],
    )
    dominant_regime = next(iter(regime_stats), None) if regime_stats else None

    return {
        "best_signals": [n for n, _ in best_signals],
        "worst_signals": [n for n, _ in worst_signals],
        "dominant_regime": dominant_regime,
        "total_decisions": trade_stats["total_decisions"],
        "entries_processed": len(signal_entries) + len(journal_entries),
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    result = consolidate_insights(days=days)

    print(f"\nConsolidation complete:")
    print(f"  Entries processed: {result['entries_processed']}")
    print(f"  Total decisions:   {result['total_decisions']}")
    print(f"  Dominant regime:   {result['dominant_regime']}")
    print(f"  Best signals:      {result['best_signals']}")
    print(f"  Worst signals:     {result['worst_signals']}")
