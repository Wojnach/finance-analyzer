"""Signal postmortem — analyze WHY signals fail by regime, ticker, and time.

Reads accuracy data broken down by regime and identifies patterns:
- Which signals work in which regimes (and fail in others)
- Which tickers are unpredictable for specific signals
- Signal correlation clusters (vote agreement rates)

Output goes to data/signal_postmortem.json for Layer 2 context and
periodic review by the after-hours research agent.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.signal_postmortem")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POSTMORTEM_FILE = DATA_DIR / "signal_postmortem.json"

# Minimum samples for reliable analysis
MIN_SAMPLES = 15

# Thresholds for classification
STRONG_THRESHOLD = 0.60  # >=60% = signal works
WEAK_THRESHOLD = 0.45    # <45% = signal is noise/harmful
DIVERGENCE_THRESHOLD = 0.15  # 15pp regime divergence = regime-dependent


def compute_regime_insights(regime_accuracy: dict) -> list[dict]:
    """Identify signals that perform very differently across regimes.

    Finds signals where accuracy in one regime is >15pp different from another.
    These are regime-dependent signals that should be gated or boosted.

    Args:
        regime_accuracy: Output of accuracy_stats.signal_accuracy_by_regime()

    Returns:
        List of insight dicts with signal, best_regime, worst_regime, spread.
    """
    if not regime_accuracy:
        return []

    # Collect per-signal accuracy across regimes
    signal_regimes: dict[str, dict[str, dict]] = {}
    for regime, sig_map in regime_accuracy.items():
        for sig, stats in sig_map.items():
            if stats.get("total", 0) < MIN_SAMPLES:
                continue
            signal_regimes.setdefault(sig, {})[regime] = stats

    insights = []
    for sig, regimes in signal_regimes.items():
        if len(regimes) < 2:
            continue

        accs = {r: s["accuracy"] for r, s in regimes.items()}
        best_regime = max(accs, key=accs.get)
        worst_regime = min(accs, key=accs.get)
        spread = accs[best_regime] - accs[worst_regime]

        if spread >= DIVERGENCE_THRESHOLD:
            insights.append({
                "signal": sig,
                "type": "regime_dependent",
                "best_regime": best_regime,
                "best_accuracy": round(accs[best_regime] * 100, 1),
                "best_samples": regimes[best_regime]["total"],
                "worst_regime": worst_regime,
                "worst_accuracy": round(accs[worst_regime] * 100, 1),
                "worst_samples": regimes[worst_regime]["total"],
                "spread_pp": round(spread * 100, 1),
                "recommendation": (
                    f"Gate {sig} in {worst_regime} regime "
                    f"({accs[worst_regime]*100:.0f}%) — "
                    f"it works in {best_regime} ({accs[best_regime]*100:.0f}%)"
                ),
            })

    insights.sort(key=lambda x: x["spread_pp"], reverse=True)
    return insights


def compute_signal_health_report(accuracy_data: dict) -> list[dict]:
    """Classify signals into strong, weak, and marginal categories.

    Args:
        accuracy_data: Standard accuracy dict {signal: {accuracy, total, ...}}

    Returns:
        List of signal health dicts sorted by accuracy.
    """
    report = []
    for sig, stats in accuracy_data.items():
        total = stats.get("total", 0)
        if total < MIN_SAMPLES:
            continue

        acc = stats.get("accuracy", 0.5)
        if acc >= STRONG_THRESHOLD:
            category = "strong"
        elif acc < WEAK_THRESHOLD:
            category = "weak"
        else:
            category = "marginal"

        report.append({
            "signal": sig,
            "accuracy_pct": round(acc * 100, 1),
            "samples": total,
            "category": category,
        })

    report.sort(key=lambda x: x["accuracy_pct"], reverse=True)
    return report


def compute_vote_correlation(entries: list[dict] | None = None) -> list[dict]:
    """Compute pairwise signal vote agreement rates.

    Analyzes signal_log entries to find which signals frequently agree.
    High agreement (>80%) suggests redundancy — one signal adds no
    information beyond what the other provides.

    Args:
        entries: Pre-loaded signal_log entries. If None, loads from disk.

    Returns:
        List of correlated pairs sorted by agreement rate.
    """
    if entries is None:
        try:
            from portfolio.accuracy_stats import load_entries
            entries = load_entries()
        except Exception:
            logger.warning("Could not load signal_log entries for correlation analysis")
            return []

    if not entries:
        return []

    # Count pairwise agreement
    from collections import defaultdict
    pair_agree = defaultdict(int)
    pair_total = defaultdict(int)

    for entry in entries:
        tickers = entry.get("tickers", {})
        for _ticker, tdata in tickers.items():
            signals = tdata.get("signals", {})
            # Only count signals that are actually voting (non-HOLD)
            active = {s: v for s, v in signals.items() if v != "HOLD"}
            active_names = sorted(active.keys())

            for i, s1 in enumerate(active_names):
                for s2 in active_names[i + 1:]:
                    pair = (s1, s2)
                    pair_total[pair] += 1
                    if active[s1] == active[s2]:
                        pair_agree[pair] += 1

    # Compute agreement rates
    correlations = []
    for pair, total in pair_total.items():
        if total < 30:  # need enough co-occurrences
            continue
        agree = pair_agree.get(pair, 0)
        rate = agree / total
        if rate >= 0.70:  # only report high correlations
            correlations.append({
                "signal_a": pair[0],
                "signal_b": pair[1],
                "agreement_rate": round(rate, 3),
                "co_occurrences": total,
                "agrees": agree,
            })

    correlations.sort(key=lambda x: x["agreement_rate"], reverse=True)
    return correlations


def generate_postmortem() -> dict:
    """Generate a complete signal postmortem report.

    Combines regime insights, health classification, and correlation analysis.
    Writes to data/signal_postmortem.json.
    """
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "regime_insights": [],
        "signal_health": [],
        "correlations": [],
        "summary": {},
    }

    # Regime-dependent analysis
    try:
        from portfolio.accuracy_stats import (
            load_cached_accuracy,
            load_cached_regime_accuracy,
            signal_accuracy,
            signal_accuracy_by_regime,
        )

        # Overall accuracy
        acc = load_cached_accuracy("1d")
        if not acc:
            acc = signal_accuracy("1d")
        if acc:
            report["signal_health"] = compute_signal_health_report(acc)

        # Regime breakdown
        regime_acc = load_cached_regime_accuracy("1d")
        if not regime_acc:
            regime_acc = signal_accuracy_by_regime("1d")
        if regime_acc:
            report["regime_insights"] = compute_regime_insights(regime_acc)

    except Exception:
        logger.warning("Accuracy data unavailable for postmortem", exc_info=True)

    # Correlation analysis
    try:
        report["correlations"] = compute_vote_correlation()
    except Exception:
        logger.warning("Correlation analysis failed", exc_info=True)

    # Summary
    strong = [s for s in report["signal_health"] if s["category"] == "strong"]
    weak = [s for s in report["signal_health"] if s["category"] == "weak"]
    report["summary"] = {
        "strong_signals": len(strong),
        "weak_signals": len(weak),
        "regime_dependent": len(report["regime_insights"]),
        "correlated_pairs": len(report["correlations"]),
        "top_3_strong": [s["signal"] for s in strong[:3]],
        "top_3_weak": [s["signal"] for s in weak[:3]],
    }

    atomic_write_json(POSTMORTEM_FILE, report)
    logger.info(
        "Signal postmortem: %d strong, %d weak, %d regime-dependent, %d correlated pairs",
        len(strong), len(weak), len(report["regime_insights"]), len(report["correlations"]),
    )

    return report


def get_postmortem_context() -> dict | None:
    """Load cached postmortem for inclusion in agent_summary.

    Returns compact version suitable for Layer 2 context.
    """
    data = load_json(POSTMORTEM_FILE)
    if not data:
        return None

    # Return compact version — just summary + top insights
    return {
        "summary": data.get("summary", {}),
        "top_regime_insights": data.get("regime_insights", [])[:5],
        "top_correlations": data.get("correlations", [])[:5],
    }
