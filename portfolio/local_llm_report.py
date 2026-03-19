"""Local LLM accuracy, health, and gating report."""

import json
import logging
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from portfolio.accuracy_stats import accuracy_by_signal_ticker
from portfolio.api_utils import load_config
from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json, prune_jsonl
from portfolio.forecast_accuracy import compute_forecast_accuracy, load_health_stats

logger = logging.getLogger("portfolio.local_llm_report")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PREDICTIONS_FILE = DATA_DIR / "forecast_predictions.jsonl"
CONFIG_EXAMPLE_FILE = BASE_DIR / "config.example.json"
LATEST_REPORT_FILE = DATA_DIR / "local_llm_report_latest.json"
HISTORY_FILE = DATA_DIR / "local_llm_report_history.jsonl"
EXPORT_STATE_FILE = DATA_DIR / "local_llm_report_export_state.json"
DEFAULT_REPORT_DAYS = 30
DEFAULT_HISTORY_MAX_ENTRIES = 366


def _load_prediction_entries(predictions_file=None, days=None):
    path = predictions_file or PREDICTIONS_FILE
    if not path.exists():
        return []

    cutoff = None
    if days is not None:
        cutoff = datetime.now(UTC) - timedelta(days=days)

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


def _int_or_default(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_report_config(config=None):
    if config is not None:
        return config

    try:
        return load_config()
    except FileNotFoundError:
        result = load_json(CONFIG_EXAMPLE_FILE)
        if result is not None:
            return result
        logger.warning("config.json not found for local LLM report; using empty config")
        return {}


def _report_config_view(config):
    forecast = (config.get("forecast") or {})
    ministral = ((config.get("local_models") or {}).get("ministral") or {})
    report_cfg = config.get("local_llm_report") or {}

    view = {
        "forecast": {
            key: forecast[key]
            for key in (
                "chronos_model",
                "kronos_enabled",
                "kronos_interval",
                "kronos_periods",
                "kronos_temperature",
                "kronos_top_p",
                "kronos_samples",
                "hold_threshold",
                "min_samples",
                "subsignal_hold_threshold",
                "subsignal_min_samples",
                "subsignal_accuracy_days",
            )
            if key in forecast
        },
        "local_models": {
            "ministral": {
                key: ministral[key]
                for key in ("hold_threshold", "min_samples", "accuracy_days")
                if key in ministral
            }
        },
        "local_llm_report": {
            "daily_export_enabled": report_cfg.get("daily_export_enabled", True),
            "report_days": _int_or_default(report_cfg.get("report_days"), DEFAULT_REPORT_DAYS),
            "history_max_entries": _int_or_default(
                report_cfg.get("history_max_entries"), DEFAULT_HISTORY_MAX_ENTRIES
            ),
        },
    }

    if not view["local_models"]["ministral"]:
        view["local_models"] = {}
    return view


def build_local_llm_report(days=DEFAULT_REPORT_DAYS, config=None, predictions_file=None, health_file=None):
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
        "config": _report_config_view(cfg),
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


def _build_export_entry(report, exported_at):
    return {
        "date": exported_at[:10],
        "exported_at": exported_at,
        "days": report["days"],
        "config": report["config"],
        "health": report["health"],
        "ministral": report["ministral"],
        "forecast": report["forecast"],
        "gating_counts": report["gating_counts"],
        "recommendations": report["recommendations"],
    }


def export_local_llm_report(
    days=DEFAULT_REPORT_DAYS,
    config=None,
    now=None,
    predictions_file=None,
    health_file=None,
    latest_file=None,
    history_file=None,
    state_file=None,
    max_entries=DEFAULT_HISTORY_MAX_ENTRIES,
):
    timestamp = now or datetime.now(UTC)
    exported_at = timestamp.isoformat()
    cfg = _load_report_config(config=config)
    report = build_local_llm_report(
        days=days,
        config=cfg,
        predictions_file=predictions_file,
        health_file=health_file,
    )
    entry = _build_export_entry(report, exported_at)

    latest_path = latest_file or LATEST_REPORT_FILE
    history_path = history_file or HISTORY_FILE
    state_path = state_file or EXPORT_STATE_FILE

    atomic_write_json(latest_path, entry)
    atomic_append_jsonl(history_path, entry)
    prune_jsonl(history_path, max_entries=max_entries)
    atomic_write_json(
        state_path,
        {
            "last_export_date": entry["date"],
            "last_exported_at": exported_at,
            "days": days,
        },
    )
    return entry


def maybe_export_local_llm_report(
    config=None,
    now=None,
    days=None,
    predictions_file=None,
    health_file=None,
    latest_file=None,
    history_file=None,
    state_file=None,
):
    cfg = _load_report_config(config=config)
    report_cfg = cfg.get("local_llm_report") or {}
    if not report_cfg.get("daily_export_enabled", True):
        return None

    timestamp = now or datetime.now(UTC)
    export_date = timestamp.date().isoformat()
    state_path = state_file or EXPORT_STATE_FILE
    state = load_json(state_path, default={}) or {}
    if state.get("last_export_date") == export_date:
        return None

    export_days = days
    if export_days is None:
        export_days = _int_or_default(report_cfg.get("report_days"), DEFAULT_REPORT_DAYS)
    max_entries = _int_or_default(
        report_cfg.get("history_max_entries"), DEFAULT_HISTORY_MAX_ENTRIES
    )
    return export_local_llm_report(
        days=export_days,
        config=cfg,
        now=timestamp,
        predictions_file=predictions_file,
        health_file=health_file,
        latest_file=latest_file,
        history_file=history_file,
        state_file=state_path,
        max_entries=max_entries,
    )


def print_local_llm_report(days=DEFAULT_REPORT_DAYS):
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
