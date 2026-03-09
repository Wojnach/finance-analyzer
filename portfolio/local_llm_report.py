"""Local LLM accuracy, health, and gating report."""

import json
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from portfolio.accuracy_stats import accuracy_by_signal_ticker
from portfolio.api_utils import load_config
from portfolio.forecast_accuracy import compute_forecast_accuracy, load_health_stats

logger = logging.getLogger("portfolio.local_llm_report")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PREDICTIONS_FILE = DATA_DIR / "forecast_predictions.jsonl"
CONFIG_EXAMPLE_FILE = BASE_DIR / "config.example.json"


def _load_prediction_entries(predictions_file=None, days=None):
    path = predictions_file or PREDICTIONS_FILE
    if not path.exists():
        return []

    cutoff = None
    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        if cutoff is not None:
            try:
                ts = datetime.fromisoformat(entry.get("ts", ""))
            except (TypeError, ValueError):
                continue
            if ts < cutoff:
                continue

        entries.append(entry)
    return entries


def _aggregate_accuracy(by_ticker):
    correct = sum(v.get("correct", 0) for v in by_ticker.values())
    samples = sum(v.get("samples", 0) for v in by_ticker.values())
    accuracy = (correct / samples) if samples else 0.0
    return {
        "accuracy": round(accuracy, 3),
        "correct": correct,
        "samples": samples,
    }


def _count_forecast_gating(entries):
    forecast_counts = Counter()
    subsignal_counts = {}

    for entry in entries:
        gating = entry.get("gating_action")
        if gating:
            forecast_counts[gating] += 1

        subsignal_gating = entry.get("subsignal_gating") or {}
        for name, meta in subsignal_gating.items():
            counter = subsignal_counts.setdefault(name, Counter())
            counter[meta.get("gating", "unknown")] += 1

    return {
        "forecast": dict(forecast_counts),
        "subsignals": {
            name: dict(counter)
            for name, counter in sorted(subsignal_counts.items())
        },
    }


def _build_recommendations(report):
    recommendations = []

    chronos_model = (
        ((report.get("config") or {}).get("forecast") or {}).get("chronos_model")
        or "amazon/chronos-t5-small"
    )
    if chronos_model.startswith("amazon/chronos-t5"):
        recommendations.append(
            "Benchmark `amazon/chronos-bolt-small` against the current T5 Chronos model before any live swap."
        )

    raw_1h = report.get("forecast", {}).get("raw", {}).get("1h", {})
    raw_24h = report.get("forecast", {}).get("raw", {}).get("24h", {})
    if not raw_1h and not raw_24h:
        recommendations.append(
            "Run `--forecast-outcomes` or `--check-outcomes` regularly so raw Chronos/Kronos accuracy stops showing as unknown."
        )

    kronos = report.get("health", {}).get("kronos", {})
    if kronos.get("total", 0) and kronos.get("success_rate", 1.0) < 0.9:
        recommendations.append(
            "Keep Kronos gated or shadow-only until subprocess reliability is consistently above 90%."
        )

    ministral_cfg = (((report.get("config") or {}).get("local_models") or {}).get("ministral") or {})
    hold_threshold = ministral_cfg.get("hold_threshold", 0.55)
    ministral = report.get("ministral", {}).get("overall", {})
    if ministral.get("samples", 0) and ministral.get("accuracy", 0.0) < hold_threshold:
        recommendations.append(
            "Keep Ministral in abstention mode; its measured 1d accuracy is below the configured hold threshold."
        )
    elif ministral.get("samples", 0):
        recommendations.append(
            "Ministral is above the current hold threshold; keep it ticker-gated and benchmark a newer Mistral small model in shadow mode."
        )

    return recommendations


def _load_report_config(config=None):
    if config is not None:
        return config

    try:
        return load_config()
    except FileNotFoundError:
        if CONFIG_EXAMPLE_FILE.exists():
            return json.loads(CONFIG_EXAMPLE_FILE.read_text(encoding="utf-8"))
        logger.warning("config.json not found for local LLM report; using empty config")
        return {}


def build_local_llm_report(days=30, config=None, predictions_file=None, health_file=None):
    cfg = _load_report_config(config=config)

    entries = _load_prediction_entries(predictions_file=predictions_file, days=days)
    ministral_by_ticker = accuracy_by_signal_ticker("ministral", horizon="1d", days=days)

    forecast_raw = {
        "1h": compute_forecast_accuracy(
            horizon="1h",
            days=days,
            predictions_file=predictions_file,
            use_raw_sub_signals=True,
        ),
        "24h": compute_forecast_accuracy(
            horizon="24h",
            days=days,
            predictions_file=predictions_file,
            use_raw_sub_signals=True,
        ),
    }
    forecast_effective = {
        "1h": compute_forecast_accuracy(
            horizon="1h",
            days=days,
            predictions_file=predictions_file,
        ),
        "24h": compute_forecast_accuracy(
            horizon="24h",
            days=days,
            predictions_file=predictions_file,
        ),
    }

    report = {
        "days": days,
        "config": cfg,
        "health": load_health_stats(health_file=health_file),
        "ministral": {
            "overall": _aggregate_accuracy(ministral_by_ticker),
            "by_ticker": ministral_by_ticker,
        },
        "forecast": {
            "raw": forecast_raw,
            "effective": forecast_effective,
        },
        "gating_counts": _count_forecast_gating(entries),
    }
    report["recommendations"] = _build_recommendations(report)
    return report


def print_local_llm_report(days=30):
    report = build_local_llm_report(days=days)

    print(f"=== Local LLM Report ({days}d) ===")

    ministral = report["ministral"]["overall"]
    print("\nMinistral (1d)")
    print(
        f"  Overall: {ministral['accuracy']*100:.1f}% ({ministral['correct']}/{ministral['samples']})"
    )
    for ticker, stats in sorted(report["ministral"]["by_ticker"].items()):
        print(
            f"  {ticker}: {stats['accuracy']*100:.1f}% ({stats['correct']}/{stats['samples']})"
        )

    print("\nForecast Health")
    if not report["health"]:
        print("  No health data available.")
    else:
        for model, stats in sorted(report["health"].items()):
            print(
                f"  {model}: {stats['success_rate']*100:.1f}% success ({stats['ok']}/{stats['total']})"
            )

    print("\nForecast Raw Accuracy")
    for horizon in ("1h", "24h"):
        print(f"  {horizon}:")
        stats = report["forecast"]["raw"][horizon]
        if not stats:
            print("    no outcome data")
            continue
        for sub_name, sub_stats in sorted(stats.items()):
            print(
                f"    {sub_name}: {sub_stats['accuracy']*100:.1f}% ({sub_stats['correct']}/{sub_stats['total']})"
            )

    print("\nForecast Effective Accuracy")
    for horizon in ("1h", "24h"):
        print(f"  {horizon}:")
        stats = report["forecast"]["effective"][horizon]
        if not stats:
            print("    no outcome data")
            continue
        for sub_name, sub_stats in sorted(stats.items()):
            print(
                f"    {sub_name}: {sub_stats['accuracy']*100:.1f}% ({sub_stats['correct']}/{sub_stats['total']})"
            )

    print("\nForecast Gating")
    gating = report["gating_counts"]["forecast"]
    if not gating:
        print("  No gating data logged yet.")
    else:
        for name, count in sorted(gating.items()):
            print(f"  {name}: {count}")

    print("\nRecommendations")
    if not report["recommendations"]:
        print("  No recommendations.")
    else:
        for rec in report["recommendations"]:
            print(f"  - {rec}")
