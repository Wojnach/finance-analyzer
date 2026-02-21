import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from portfolio.tickers import SIGNAL_NAMES

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"
ACCURACY_CACHE_FILE = DATA_DIR / "accuracy_cache.json"
ACCURACY_CACHE_TTL = 3600
HORIZONS = ["1d", "3d", "5d", "10d"]


def _atomic_write_json(path, data):
    """Write JSON atomically using tempfile + os.replace to prevent corruption on crash."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, str(path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_entries():
    """Load signal log entries. Prefers SQLite if available, falls back to JSONL."""
    try:
        from portfolio.signal_db import SignalDB
        db = SignalDB()
        count = db.snapshot_count()
        if count > 0:
            entries = db.load_entries()
            db.close()
            return entries
        db.close()
    except Exception:
        pass
    # Fallback to JSONL
    if not SIGNAL_LOG.exists():
        return []
    entries = []
    with open(SIGNAL_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def _vote_correct(vote, change_pct):
    if vote == "BUY" and change_pct > 0:
        return True
    if vote == "SELL" and change_pct < 0:
        return True
    return False


def signal_accuracy(horizon="1d"):
    entries = load_entries()
    stats = {s: {"correct": 0, "total": 0} for s in SIGNAL_NAMES}

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            signals = tdata.get("signals", {})

            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name, "HOLD")
                if vote == "HOLD":
                    continue
                stats[sig_name]["total"] += 1
                if _vote_correct(vote, change_pct):
                    stats[sig_name]["correct"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        result[sig_name] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": acc,
            "pct": round(acc * 100, 1),
        }
    return result


def consensus_accuracy(horizon="1d"):
    entries = load_entries()
    correct = 0
    total = 0

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            consensus = tdata.get("consensus", "HOLD")
            if consensus == "HOLD":
                continue

            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            total += 1
            if _vote_correct(consensus, change_pct):
                correct += 1

    acc = correct / total if total > 0 else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": acc,
        "pct": round(acc * 100, 1),
    }


def per_ticker_accuracy(horizon="1d"):
    entries = load_entries()
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for entry in entries:
        outcomes = entry.get("outcomes", {})
        tickers = entry.get("tickers", {})

        for ticker, tdata in tickers.items():
            consensus = tdata.get("consensus", "HOLD")
            if consensus == "HOLD":
                continue

            outcome = outcomes.get(ticker, {}).get(horizon)
            if not outcome:
                continue

            change_pct = outcome.get("change_pct", 0)
            stats[ticker]["total"] += 1
            if _vote_correct(consensus, change_pct):
                stats[ticker]["correct"] += 1

    result = {}
    for ticker, s in stats.items():
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        result[ticker] = {
            "correct": s["correct"],
            "total": s["total"],
            "accuracy": acc,
            "pct": round(acc * 100, 1),
        }
    return result


def best_worst_signals(horizon="1d"):
    acc = signal_accuracy(horizon)
    qualified = {k: v for k, v in acc.items() if v["total"] >= 5}
    if not qualified:
        return {"best": None, "worst": None}

    best_name = max(qualified, key=lambda k: qualified[k]["accuracy"])
    worst_name = min(qualified, key=lambda k: qualified[k]["accuracy"])
    return {
        "best": (best_name, qualified[best_name]["accuracy"]),
        "worst": (worst_name, qualified[worst_name]["accuracy"]),
    }


def signal_activation_rates():
    """Compute per-signal activation rates (how often each signal votes non-HOLD).

    Returns dict: {signal_name: {activation_rate, buy_rate, sell_rate, bias, samples}}
    - activation_rate: fraction of votes that are BUY or SELL (0.0 to 1.0)
    - bias: directional bias = abs(buy_rate - sell_rate) / activation_rate (0=balanced, 1=all one side)
    - rarity_weight: log(1 + 1/activation_rate) — rare signals get higher weight
    - bias_penalty: 1 - bias (minimum 0.1 floor) — directional signals get penalized
    - normalized_weight: rarity_weight * bias_penalty — the final multiplier
    """
    import math

    entries = load_entries()
    stats = {s: {"buy": 0, "sell": 0, "total": 0} for s in SIGNAL_NAMES}

    for entry in entries:
        tickers = entry.get("tickers", {})
        for ticker, tdata in tickers.items():
            signals = tdata.get("signals", {})
            for sig_name in SIGNAL_NAMES:
                vote = signals.get(sig_name)
                if vote is None:
                    continue
                stats[sig_name]["total"] += 1
                if vote == "BUY":
                    stats[sig_name]["buy"] += 1
                elif vote == "SELL":
                    stats[sig_name]["sell"] += 1

    result = {}
    for sig_name in SIGNAL_NAMES:
        s = stats[sig_name]
        total = s["total"]
        if total == 0:
            result[sig_name] = {
                "activation_rate": 0.0, "buy_rate": 0.0, "sell_rate": 0.0,
                "bias": 1.0, "rarity_weight": 1.0, "bias_penalty": 0.1,
                "normalized_weight": 0.1, "samples": 0,
            }
            continue

        buy_rate = s["buy"] / total
        sell_rate = s["sell"] / total
        activation_rate = buy_rate + sell_rate

        # Rarity: IDF-style weight — rare signals get more weight when they vote
        if activation_rate > 0.01:
            rarity_weight = math.log(1 + 1 / activation_rate)
        else:
            rarity_weight = math.log(1 + 100)  # cap for near-zero activation

        # Bias: penalize signals that always vote one direction
        if activation_rate > 0:
            bias = abs(buy_rate - sell_rate) / activation_rate
        else:
            bias = 1.0
        bias_penalty = max(1.0 - bias, 0.1)  # floor at 0.1

        result[sig_name] = {
            "activation_rate": round(activation_rate, 4),
            "buy_rate": round(buy_rate, 4),
            "sell_rate": round(sell_rate, 4),
            "bias": round(bias, 4),
            "rarity_weight": round(rarity_weight, 4),
            "bias_penalty": round(bias_penalty, 4),
            "normalized_weight": round(rarity_weight * bias_penalty, 4),
            "samples": total,
        }

    return result


ACTIVATION_CACHE_TTL = 3600  # recompute hourly


def load_cached_activation_rates():
    """Load cached activation rates, recomputing if stale."""
    cache_file = DATA_DIR / "activation_cache.json"
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
            if time.time() - cache.get("time", 0) < ACTIVATION_CACHE_TTL:
                return cache.get("rates", {})
        except (json.JSONDecodeError, KeyError):
            pass
    rates = signal_activation_rates()
    try:
        _atomic_write_json(cache_file, {"rates": rates, "time": time.time()})
    except Exception:
        pass
    return rates


def load_cached_accuracy(horizon="1d"):
    if ACCURACY_CACHE_FILE.exists():
        try:
            cache = json.loads(ACCURACY_CACHE_FILE.read_text(encoding="utf-8"))
            if time.time() - cache.get("time", 0) < ACCURACY_CACHE_TTL:
                cached = cache.get(horizon)
                if cached:
                    return cached
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def write_accuracy_cache(horizon, data):
    cache = {}
    if ACCURACY_CACHE_FILE.exists():
        try:
            cache = json.loads(ACCURACY_CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError):
            pass
    cache[horizon] = data
    cache["time"] = time.time()
    _atomic_write_json(ACCURACY_CACHE_FILE, cache)


def _count_entries_with_outcomes(entries, horizon):
    count = 0
    for entry in entries:
        outcomes = entry.get("outcomes", {})
        for ticker, horizons in outcomes.items():
            if horizons.get(horizon):
                count += 1
                break
    return count


def print_accuracy_report():
    entries = load_entries()
    if not entries:
        print("No signal log data found.")
        return

    horizon_counts = {h: _count_entries_with_outcomes(entries, h) for h in HORIZONS}
    counts_str = ", ".join(f"{horizon_counts[h]} with {h} outcomes" for h in HORIZONS)

    print("=== Signal Accuracy Report ===")
    print()
    print(f"Entries: {len(entries)} total, {counts_str}")

    for h in HORIZONS:
        if horizon_counts[h] == 0:
            continue

        print()
        print(f"--- {h} Horizon ({horizon_counts[h]} entries with outcomes) ---")
        print()

        sig_acc = signal_accuracy(h)
        sorted_sigs = sorted(
            SIGNAL_NAMES, key=lambda s: sig_acc[s]["accuracy"], reverse=True
        )

        print(f"{'Signal':<16}{'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
        print(f"{'------':<16}{'-------':>7}  {'-----':>5}  {'--------':>8}")

        for sig_name in sorted_sigs:
            s = sig_acc[sig_name]
            if s["total"] == 0:
                continue
            print(
                f"{sig_name:<16}{s['correct']:>7}  {s['total']:>5}  {s['accuracy']*100:>7.1f}%"
            )

        cons = consensus_accuracy(h)
        print()
        if cons["total"] > 0:
            print(
                f"{'Consensus':<16}{cons['correct']:>7}  {cons['total']:>5}  {cons['accuracy']*100:>7.1f}%"
            )

        ticker_acc = per_ticker_accuracy(h)
        if ticker_acc:
            print()
            print("Per-Ticker:")
            sorted_tickers = sorted(
                ticker_acc.keys(), key=lambda t: ticker_acc[t]["accuracy"], reverse=True
            )
            for ticker in sorted_tickers:
                s = ticker_acc[ticker]
                print(
                    f"{ticker:<16}{s['correct']:>7}  {s['total']:>5}  {s['accuracy']*100:>7.1f}%"
                )


ACCURACY_SNAPSHOTS_FILE = DATA_DIR / "accuracy_snapshots.jsonl"


def save_accuracy_snapshot():
    """Save current per-signal accuracy as a timestamped snapshot.

    Appends one JSON line to accuracy_snapshots.jsonl with the current
    accuracy for each signal at the 1d horizon. Used by check_accuracy_changes()
    to detect significant shifts over time.
    """
    from datetime import datetime, timezone

    acc = signal_accuracy("1d")
    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "signals": {
            name: {"accuracy": data["accuracy"], "total": data["total"]}
            for name, data in acc.items()
        },
    }
    DATA_DIR.mkdir(exist_ok=True)
    with open(ACCURACY_SNAPSHOTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
    return snapshot


def _load_accuracy_snapshots():
    """Load all accuracy snapshots from JSONL file."""
    if not ACCURACY_SNAPSHOTS_FILE.exists():
        return []
    entries = []
    for line in ACCURACY_SNAPSHOTS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _find_snapshot_near(snapshots, target_ts, max_delta_hours=36):
    """Find the snapshot closest to target_ts within max_delta_hours.

    Args:
        snapshots: List of snapshot dicts with 'ts' field.
        target_ts: datetime to search near.
        max_delta_hours: Maximum allowed time difference in hours.

    Returns:
        dict or None: The closest snapshot, or None if none within range.
    """
    from datetime import datetime

    best = None
    best_delta = None
    for snap in snapshots:
        try:
            snap_ts = datetime.fromisoformat(snap["ts"])
            delta = abs((snap_ts - target_ts).total_seconds()) / 3600
            if delta <= max_delta_hours:
                if best_delta is None or delta < best_delta:
                    best = snap
                    best_delta = delta
        except (ValueError, TypeError, KeyError):
            continue
    return best


def check_accuracy_changes(threshold_drop=0.1, threshold_rise=0.1):
    """Check for significant accuracy changes vs 7 days ago.

    Compares current signal accuracy against the snapshot closest to 7 days ago.
    Returns alerts for signals whose accuracy changed by more than the thresholds.

    Args:
        threshold_drop: Minimum accuracy drop (as fraction, e.g. 0.1 = 10pp) to alert.
        threshold_rise: Minimum accuracy rise (as fraction, e.g. 0.1 = 10pp) to alert.

    Returns:
        list[dict]: List of alert dicts with keys:
            signal, old_accuracy, new_accuracy, change, direction ("dropped"/"rose"),
            old_samples, new_samples.
        Empty list if no significant changes or no historical snapshot available.
    """
    from datetime import datetime, timedelta, timezone

    snapshots = _load_accuracy_snapshots()
    if not snapshots:
        return []

    now = datetime.now(timezone.utc)
    target = now - timedelta(days=7)
    old_snapshot = _find_snapshot_near(snapshots, target)

    if old_snapshot is None:
        return []

    # Compute current accuracy
    current_acc = signal_accuracy("1d")
    old_signals = old_snapshot.get("signals", {})

    alerts = []
    for sig_name in SIGNAL_NAMES:
        old_data = old_signals.get(sig_name)
        new_data = current_acc.get(sig_name)

        if not old_data or not new_data:
            continue

        # Require minimum samples for meaningful comparison
        if old_data.get("total", 0) < 10 or new_data.get("total", 0) < 10:
            continue

        old_acc = old_data.get("accuracy", 0.0)
        new_acc = new_data.get("accuracy", 0.0)
        change = new_acc - old_acc

        if change <= -threshold_drop:
            alerts.append({
                "signal": sig_name,
                "old_accuracy": round(old_acc * 100, 1),
                "new_accuracy": round(new_acc * 100, 1),
                "change": round(change * 100, 1),
                "direction": "dropped",
                "old_samples": old_data.get("total", 0),
                "new_samples": new_data.get("total", 0),
            })
        elif change >= threshold_rise:
            alerts.append({
                "signal": sig_name,
                "old_accuracy": round(old_acc * 100, 1),
                "new_accuracy": round(new_acc * 100, 1),
                "change": round(change * 100, 1),
                "direction": "rose",
                "old_samples": old_data.get("total", 0),
                "new_samples": new_data.get("total", 0),
            })

    # Sort by absolute change magnitude, largest first
    alerts.sort(key=lambda a: abs(a["change"]), reverse=True)
    return alerts


def format_accuracy_alerts(alerts):
    """Format accuracy change alerts as human-readable strings.

    Args:
        alerts: List of alert dicts from check_accuracy_changes().

    Returns:
        list[str]: Formatted alert strings.
    """
    lines = []
    for a in alerts:
        lines.append(
            f"{a['signal']} accuracy {a['direction']} from "
            f"{a['old_accuracy']}% to {a['new_accuracy']}% "
            f"({a['change']:+.1f}pp, {a['new_samples']} samples)"
        )
    return lines


if __name__ == "__main__":
    print_accuracy_report()

    # Also show accuracy changes if snapshots exist
    alerts = check_accuracy_changes()
    if alerts:
        print()
        print("=== Accuracy Changes (vs 7 days ago) ===")
        print()
        for line in format_accuracy_alerts(alerts):
            print(f"  {line}")
    else:
        print()
        print("No significant accuracy changes detected (or no 7-day snapshot available).")
