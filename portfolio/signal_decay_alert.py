"""Signal decay alerting — detects when signal accuracy degrades significantly.

Compares recent (7d) accuracy against all-time accuracy for each signal.
Alerts when a signal degrades by >10pp with 50+ recent samples.

Run as part of --check-outcomes or standalone via check_signal_decay().

Added 2026-04-30 after-hours research session. Prevents silent accuracy
erosion between manual audit sessions.
"""

import json
import logging
from datetime import UTC, datetime

logger = logging.getLogger("portfolio.signal_decay_alert")


# Minimum absolute accuracy drop (percentage points) to trigger an alert.
_DECAY_THRESHOLD_PP = 10.0
# Minimum recent samples to be confident in degradation.
_MIN_RECENT_SAMPLES = 50
# Minimum all-time samples to have a reliable baseline.
_MIN_ALLTIME_SAMPLES = 100


def check_signal_decay(accuracy_cache_path="data/accuracy_cache.json"):
    """Check for signal accuracy decay and return a list of decay alerts.

    Returns:
        list[dict]: Each dict has keys: signal, horizon, alltime_acc, recent_acc,
        drop_pp, recent_samples, severity.
    """
    try:
        with open(accuracy_cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Cannot load accuracy cache: %s", e)
        return []

    alerts = []
    horizon_pairs = [
        ("3h", "3h_recent"),
        ("1d", "1d_recent"),
    ]

    for alltime_key, recent_key in horizon_pairs:
        alltime_data = cache.get(alltime_key, {})
        recent_data = cache.get(recent_key, {})

        if not isinstance(alltime_data, dict) or not isinstance(recent_data, dict):
            continue

        for signal in recent_data:
            recent = recent_data[signal]
            alltime = alltime_data.get(signal, {})

            if not isinstance(recent, dict) or not isinstance(alltime, dict):
                continue

            recent_acc = recent.get("accuracy", 0)
            recent_total = recent.get("total", 0)
            alltime_acc = alltime.get("accuracy", 0)
            alltime_total = alltime.get("total", 0)

            if recent_total < _MIN_RECENT_SAMPLES:
                continue
            if alltime_total < _MIN_ALLTIME_SAMPLES:
                continue

            drop_pp = (alltime_acc - recent_acc) * 100

            if drop_pp >= _DECAY_THRESHOLD_PP:
                severity = "critical" if drop_pp >= 20 else "warning"
                alerts.append({
                    "signal": signal,
                    "horizon": alltime_key,
                    "alltime_acc": round(alltime_acc * 100, 1),
                    "recent_acc": round(recent_acc * 100, 1),
                    "drop_pp": round(drop_pp, 1),
                    "recent_samples": recent_total,
                    "alltime_samples": alltime_total,
                    "severity": severity,
                })

    # Sort by drop magnitude (worst first)
    alerts.sort(key=lambda a: -a["drop_pp"])
    return alerts


def format_decay_report(alerts):
    """Format decay alerts as a human-readable string.

    Returns:
        str: Formatted report, or empty string if no alerts.
    """
    if not alerts:
        return ""

    critical = [a for a in alerts if a["severity"] == "critical"]
    warnings = [a for a in alerts if a["severity"] == "warning"]

    lines = [f"SIGNAL DECAY ALERT — {len(alerts)} signals degrading"]
    lines.append("")

    if critical:
        lines.append(f"CRITICAL ({len(critical)} signals, >20pp drop):")
        for a in critical:
            lines.append(
                f"  {a['signal']:25s} {a['horizon']:>3s}: "
                f"{a['alltime_acc']:.1f}% -> {a['recent_acc']:.1f}% "
                f"({a['drop_pp']:+.1f}pp, {a['recent_samples']} sam)"
            )
        lines.append("")

    if warnings:
        lines.append(f"WARNING ({len(warnings)} signals, >10pp drop):")
        for a in warnings:
            lines.append(
                f"  {a['signal']:25s} {a['horizon']:>3s}: "
                f"{a['alltime_acc']:.1f}% -> {a['recent_acc']:.1f}% "
                f"({a['drop_pp']:+.1f}pp, {a['recent_samples']} sam)"
            )

    return "\n".join(lines)


def log_decay_alerts(alerts):
    """Log decay alerts and write to data/signal_decay_alerts.jsonl."""
    if not alerts:
        logger.info("Signal decay check: no degradation detected")
        return

    report = format_decay_report(alerts)
    logger.warning("Signal decay detected:\n%s", report)

    # Append to JSONL log
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "alert_count": len(alerts),
        "critical_count": sum(1 for a in alerts if a["severity"] == "critical"),
        "alerts": alerts,
    }
    try:
        from portfolio.file_utils import atomic_append_jsonl
        atomic_append_jsonl("data/signal_decay_alerts.jsonl", entry)
    except Exception:
        logger.debug("Could not write signal_decay_alerts.jsonl", exc_info=True)


def run_decay_check():
    """Run a full decay check and log results. Called from main.py --check-outcomes."""
    alerts = check_signal_decay()
    log_decay_alerts(alerts)
    return alerts
