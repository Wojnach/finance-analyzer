import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INVOCATIONS_FILE = DATA_DIR / "invocations.jsonl"
TELEGRAMS_FILE = DATA_DIR / "telegram_messages.jsonl"


def load_jsonl(path):
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def invocation_stats():
    entries = load_jsonl(INVOCATIONS_FILE)
    if not entries:
        print("No invocations recorded yet.")
        return

    by_date = defaultdict(list)
    reason_counts = Counter()
    for e in entries:
        dt = datetime.fromisoformat(e["ts"])
        day = dt.strftime("%Y-%m-%d")
        by_date[day].append(e)
        for r in e.get("reasons", []):
            tag = r.split(" ")[0] if " " in r else r
            reason_counts[tag] += 1

    print("=== Agent Invocations ===\n")
    print(f"{'Date':<14} {'Count':>6}")
    print("-" * 22)
    total = 0
    for day in sorted(by_date.keys()):
        n = len(by_date[day])
        total += n
        print(f"{day:<14} {n:>6}")
    print("-" * 22)
    print(f"{'Total':<14} {total:>6}")

    days = len(by_date)
    if days > 0:
        print(f"{'Avg/day':<14} {total / days:>6.1f}")

    print(f"\n{'Trigger Reason':<24} {'Count':>6}")
    print("-" * 32)
    for reason, count in reason_counts.most_common():
        print(f"{reason:<24} {count:>6}")


def telegram_stats():
    entries = load_jsonl(TELEGRAMS_FILE)
    if not entries:
        print("\nNo Telegram messages recorded yet.")
        return

    by_date = defaultdict(list)
    decisions = Counter()
    for e in entries:
        dt = datetime.fromisoformat(e["ts"])
        day = dt.strftime("%Y-%m-%d")
        by_date[day].append(e)
        text = e.get("text", "")
        if text.startswith("*HOLD*"):
            decisions["HOLD"] += 1
        elif text.startswith("*BUY"):
            decisions["BUY"] += 1
        elif text.startswith("*SELL"):
            decisions["SELL"] += 1
        else:
            decisions["OTHER"] += 1

    print("\n=== Telegram Messages ===\n")
    print(f"{'Date':<14} {'Sent':>6}")
    print("-" * 22)
    total = 0
    for day in sorted(by_date.keys()):
        n = len(by_date[day])
        total += n
        print(f"{day:<14} {n:>6}")
    print("-" * 22)
    print(f"{'Total':<14} {total:>6}")

    print(f"\n{'Decision':<14} {'Count':>6} {'%':>7}")
    print("-" * 29)
    for dec, count in decisions.most_common():
        pct = count / total * 100 if total > 0 else 0
        print(f"{dec:<14} {count:>6} {pct:>6.1f}%")


def full_report():
    invocation_stats()
    telegram_stats()

    invocations = load_jsonl(INVOCATIONS_FILE)
    telegrams = load_jsonl(TELEGRAMS_FILE)
    missed = len(invocations) - len(telegrams)
    if missed > 0:
        print(
            f"\nWARNING: {missed} invocation(s) without Telegram response (agent may have crashed)"
        )
    elif missed < 0:
        print(f"\nNOTE: {-missed} extra Telegram(s) beyond invocations (manual runs)")


if __name__ == "__main__":
    full_report()
