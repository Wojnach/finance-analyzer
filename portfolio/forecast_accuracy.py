"""Forecast sub-signal accuracy tracker.

Reads forecast_predictions.jsonl, backfills actual prices at 1h/24h horizons,
and computes per-model per-ticker per-horizon accuracy statistics.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger("portfolio.forecast_accuracy")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PREDICTIONS_FILE = DATA_DIR / "forecast_predictions.jsonl"
HEALTH_FILE = DATA_DIR / "forecast_health.jsonl"


def load_predictions(predictions_file=None):
    """Load all forecast predictions from JSONL file."""
    path = predictions_file or PREDICTIONS_FILE
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def load_health_stats(health_file=None):
    """Load forecast health stats (success/failure rates per model)."""
    path = health_file or HEALTH_FILE
    if not path.exists():
        return {}
    stats = defaultdict(lambda: {"ok": 0, "fail": 0, "total": 0})
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            model = entry.get("model", "unknown")
            if entry.get("ok"):
                stats[model]["ok"] += 1
            else:
                stats[model]["fail"] += 1
            stats[model]["total"] += 1
        except json.JSONDecodeError:
            continue

    result = {}
    for model, s in stats.items():
        result[model] = {
            "ok": s["ok"],
            "fail": s["fail"],
            "total": s["total"],
            "success_rate": round(s["ok"] / s["total"], 3) if s["total"] else 0.0,
        }
    return result


def compute_forecast_accuracy(ticker=None, horizon="24h", days=None,
                              predictions_file=None):
    """Compute accuracy of forecast sub-signals.

    For each prediction entry that has an actual outcome (backfilled),
    check if the predicted direction matched actual price movement.

    Args:
        ticker: Filter to specific ticker (None = all).
        horizon: Which horizon to evaluate ("1h" or "24h").
        days: Only include entries from last N days (None = all).
        predictions_file: Override predictions file path (for testing).

    Returns:
        dict: {
            model_name: {
                "accuracy": float,
                "correct": int,
                "total": int,
                "by_ticker": {ticker: {"accuracy": float, "correct": int, "total": int}}
            }
        }
    """
    entries = load_predictions(predictions_file)

    cutoff = None
    if days is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    # Track per-model stats
    model_stats = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for entry in entries:
        if cutoff and entry.get("ts", "") < cutoff:
            continue

        entry_ticker = entry.get("ticker", "")
        if ticker and entry_ticker != ticker:
            continue

        # Need actual outcome
        outcome = entry.get("outcome", {}).get(horizon)
        if outcome is None:
            continue

        actual_change = outcome.get("change_pct", 0)

        # Check each sub-signal
        sub_signals = entry.get("sub_signals", {})
        for sub_name, vote in sub_signals.items():
            if vote == "HOLD":
                continue

            # Only check sub-signals matching the requested horizon
            if "_" in sub_name:
                sub_horizon = sub_name.split("_", 1)[1]
                if sub_horizon != horizon:
                    continue

            predicted_up = vote == "BUY"
            actual_up = actual_change > 0

            correct = (predicted_up and actual_up) or (not predicted_up and not actual_up)

            model_stats[sub_name][entry_ticker]["total"] += 1
            if correct:
                model_stats[sub_name][entry_ticker]["correct"] += 1

    # Aggregate
    result = {}
    for sub_name, ticker_stats in model_stats.items():
        total_correct = 0
        total_count = 0
        by_ticker = {}

        for t, s in ticker_stats.items():
            total_correct += s["correct"]
            total_count += s["total"]
            if s["total"] > 0:
                by_ticker[t] = {
                    "accuracy": round(s["correct"] / s["total"], 3),
                    "correct": s["correct"],
                    "total": s["total"],
                }

        result[sub_name] = {
            "accuracy": round(total_correct / total_count, 3) if total_count else 0.0,
            "correct": total_correct,
            "total": total_count,
            "by_ticker": by_ticker,
        }

    return result


def backfill_forecast_outcomes(max_entries=500, predictions_file=None,
                               snapshot_file=None):
    """Backfill actual price outcomes into forecast predictions.

    For each prediction without an outcome, check if enough time has
    passed for the horizon, then look up the actual price and compute
    the change percentage.

    Returns number of entries updated.
    """
    path = predictions_file or PREDICTIONS_FILE
    entries = load_predictions(path)
    if not entries:
        return 0

    updated = 0
    modified_entries = []

    for entry in entries:
        if "outcome" not in entry:
            entry["outcome"] = {}

        ts_str = entry.get("ts", "")
        if not ts_str:
            modified_entries.append(entry)
            continue

        try:
            entry_time = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            modified_entries.append(entry)
            continue

        current_price = entry.get("current_price", 0)
        if not current_price:
            modified_entries.append(entry)
            continue

        entry_ticker = entry.get("ticker", "")
        if not entry_ticker:
            modified_entries.append(entry)
            continue

        now = datetime.now(timezone.utc)

        for horizon_key, hours in [("1h", 1), ("24h", 24)]:
            if horizon_key in entry["outcome"]:
                continue  # already backfilled

            horizon_time = entry_time + timedelta(hours=hours)
            if now < horizon_time:
                continue  # not enough time passed

            # Look up actual price at horizon time
            actual_price = _lookup_price_at_time(
                entry_ticker, horizon_time, snapshot_file=snapshot_file
            )
            if actual_price is not None:
                change_pct = (actual_price - current_price) / current_price * 100
                entry["outcome"][horizon_key] = {
                    "actual_price": round(actual_price, 6),
                    "change_pct": round(change_pct, 4),
                    "backfilled_at": now.isoformat(),
                }
                updated += 1

        modified_entries.append(entry)

        if updated >= max_entries:
            break

    # Write back
    if updated > 0:
        _write_predictions(modified_entries, path)

    return updated


def _lookup_price_at_time(ticker, target_time, snapshot_file=None):
    """Look up the actual price for a ticker at a specific time.

    Uses hourly price snapshots from data/price_snapshots_hourly.jsonl
    and finds the closest entry within 2 hours of target_time.
    """
    path = snapshot_file or (DATA_DIR / "price_snapshots_hourly.jsonl")
    if not path.exists():
        return None

    best_price = None
    best_delta = timedelta(hours=2)  # max 2h tolerance

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            snap = json.loads(line)
            snap_time = datetime.fromisoformat(snap.get("ts", ""))
            delta = abs(snap_time - target_time)
            if delta < best_delta:
                prices = snap.get("prices", {})
                if ticker in prices:
                    best_price = prices[ticker]
                    best_delta = delta
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    return best_price


def _write_predictions(entries, predictions_file=None):
    """Write predictions back to JSONL file."""
    path = predictions_file or PREDICTIONS_FILE
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def get_forecast_accuracy_summary(focus_tickers=None, days=7):
    """Get a compact accuracy summary for Layer 2 consumption.

    Args:
        focus_tickers: List of tickers to include (None = all with data).
        days: Lookback window.

    Returns:
        dict: {
            "health": {model: success_rate},
            "accuracy": {sub_signal: {accuracy, samples, by_ticker}},
        }
    """
    health = load_health_stats()
    accuracy = compute_forecast_accuracy(days=days)

    result = {
        "health": health,
        "accuracy": {},
    }

    for sub_name, stats in accuracy.items():
        entry = {
            "accuracy": stats["accuracy"],
            "samples": stats["total"],
        }
        if focus_tickers:
            by_ticker = {t: stats["by_ticker"][t]
                         for t in focus_tickers
                         if t in stats["by_ticker"]}
            if by_ticker:
                entry["by_ticker"] = by_ticker
        else:
            if stats["by_ticker"]:
                entry["by_ticker"] = stats["by_ticker"]
        result["accuracy"][sub_name] = entry

    return result


def print_forecast_accuracy_report():
    """Print a human-readable forecast accuracy report."""
    health = load_health_stats()

    print("=== Forecast Model Health ===")
    if not health:
        print("  No health data available yet.")
    else:
        for model, stats in sorted(health.items()):
            rate = stats["success_rate"] * 100
            print(f"  {model:10s}: {stats['ok']}/{stats['total']} ({rate:.1f}% success)")

    print("\n=== Forecast Sub-Signal Accuracy ===")
    accuracy = compute_forecast_accuracy()

    if not accuracy:
        print("  No outcome data available yet. Run --forecast-outcomes to backfill.")
        return

    for sub_name, stats in sorted(accuracy.items()):
        if stats["total"] == 0:
            continue
        acc = stats["accuracy"] * 100
        print(f"\n  {sub_name}:")
        print(f"    Overall: {acc:.1f}% ({stats['correct']}/{stats['total']})")
        for t, t_stats in sorted(stats["by_ticker"].items()):
            t_acc = t_stats["accuracy"] * 100
            print(f"    {t:10s}: {t_acc:.1f}% ({t_stats['correct']}/{t_stats['total']})")
