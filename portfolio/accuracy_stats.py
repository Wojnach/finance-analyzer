import json
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"

SIGNAL_NAMES = [
    "rsi",
    "macd",
    "ema",
    "bb",
    "fear_greed",
    "sentiment",
    "ministral",
    "ml",
    "funding",
    "volume",
    "custom_lora",
]
HORIZONS = ["1d", "3d", "5d", "10d"]


def load_entries():
    if not SIGNAL_LOG.exists():
        return []
    entries = []
    with open(SIGNAL_LOG) as f:
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
    return {"correct": correct, "total": total, "accuracy": acc}


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
        result[ticker] = {"correct": s["correct"], "total": s["total"], "accuracy": acc}
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


if __name__ == "__main__":
    print_accuracy_report()
