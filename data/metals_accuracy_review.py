"""
Metals trading accuracy review script.
Reads metals_decisions.jsonl and evaluates:
1. Prediction accuracy (direction predictions vs actual moves)
2. Decision quality (HOLD/SELL actions vs subsequent price moves)
3. Summary statistics (win rate, avg P&L per decision, etc.)

Run: .venv/Scripts/python.exe data/metals_accuracy_review.py
"""
import json, os, sys, datetime
os.chdir(r"Q:/finance-analyzer")

DECISIONS_FILE = "data/metals_decisions.jsonl"
POSITIONS = {
    "gold": {"entry": 972.4},
    "silver79": {"entry": 65.13},
    "silver301": {"entry": 20.70},
}

def load_decisions():
    """Load all decisions from JSONL file."""
    if not os.path.exists(DECISIONS_FILE):
        print(f"No decisions file found at {DECISIONS_FILE}")
        return []
    entries = []
    with open(DECISIONS_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: malformed line {line_num}: {e}")
    return entries

def evaluate_predictions(decisions):
    """Compare each prediction with the NEXT decision's actual prices."""
    if len(decisions) < 2:
        print("Need at least 2 decisions to evaluate predictions")
        return []

    results = []
    for i in range(len(decisions) - 1):
        current = decisions[i]
        next_d = decisions[i + 1]

        pred = current.get("prediction", {})
        if not pred:
            continue

        direction = pred.get("direction", "flat")
        confidence = pred.get("confidence", 0)

        # Get silver prices at prediction time and at next check
        cur_silver = current.get("underlying", {}).get("silver_usd", 0)
        next_silver = next_d.get("underlying", {}).get("silver_usd", 0)

        if cur_silver and next_silver:
            actual_move_pct = ((next_silver - cur_silver) / cur_silver) * 100
            predicted_up = direction == "up"
            predicted_down = direction == "down"
            actual_up = actual_move_pct > 0.05  # >0.05% threshold for "up"
            actual_down = actual_move_pct < -0.05

            if direction == "flat":
                correct = abs(actual_move_pct) < 0.5  # flat = less than 0.5% move
            elif predicted_up:
                correct = actual_up
            elif predicted_down:
                correct = actual_down
            else:
                correct = None

            results.append({
                "ts": current.get("ts", "?"),
                "direction": direction,
                "confidence": confidence,
                "silver_at_prediction": cur_silver,
                "silver_at_check": next_silver,
                "actual_move_pct": round(actual_move_pct, 3),
                "correct": correct,
                "horizon": pred.get("horizon", "?"),
            })

    return results

def evaluate_hold_sells(decisions):
    """Evaluate whether HOLD/SELL decisions were correct in hindsight."""
    if len(decisions) < 2:
        return {}

    position_stats = {}
    for key in POSITIONS:
        holds_correct = 0
        holds_total = 0
        sells_correct = 0
        sells_total = 0

        for i in range(len(decisions) - 1):
            current = decisions[i]
            next_d = decisions[i + 1]

            pos_data = current.get("positions", {}).get(key, {})
            next_pos = next_d.get("positions", {}).get(key, {})

            if not pos_data or not next_pos:
                continue

            action = pos_data.get("action", "HOLD")
            cur_bid = pos_data.get("bid", 0)
            next_bid = next_pos.get("bid", 0)

            if cur_bid and next_bid:
                move_pct = ((next_bid - cur_bid) / cur_bid) * 100

                if action == "HOLD":
                    holds_total += 1
                    # HOLD was correct if price didn't drop significantly
                    if move_pct >= -1.0:  # didn't drop >1%
                        holds_correct += 1
                elif action == "SELL":
                    sells_total += 1
                    # SELL was correct if price dropped after selling
                    if move_pct < 0:
                        sells_correct += 1

        position_stats[key] = {
            "holds_total": holds_total,
            "holds_correct": holds_correct,
            "holds_accuracy": (holds_correct / holds_total * 100) if holds_total > 0 else 0,
            "sells_total": sells_total,
            "sells_correct": sells_correct,
            "sells_accuracy": (sells_correct / sells_total * 100) if sells_total > 0 else 0,
        }

    return position_stats

def print_report(decisions):
    """Print comprehensive accuracy report."""
    print(f"\n{'='*60}")
    print(f"  METALS TRADING ACCURACY REPORT")
    print(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}\n")

    if not decisions:
        print("  No decisions recorded yet. Run the trading loop first.")
        return

    print(f"  Total decisions logged: {len(decisions)}")

    # Time range
    first_ts = decisions[0].get("ts", "?")
    last_ts = decisions[-1].get("ts", "?")
    print(f"  Period: {first_ts[:19]} to {last_ts[:19]}")

    # Tier breakdown
    tier_counts = {}
    for d in decisions:
        t = d.get("tier", "?")
        tier_counts[t] = tier_counts.get(t, 0) + 1
    print(f"  Tiers: {', '.join(f'T{k}={v}' for k, v in sorted(tier_counts.items()))}")

    # Trigger breakdown
    trigger_counts = {}
    for d in decisions:
        t = d.get("trigger", "?")
        trigger_counts[t] = trigger_counts.get(t, 0) + 1
    print(f"\n  Triggers:")
    for trigger, count in sorted(trigger_counts.items(), key=lambda x: -x[1]):
        print(f"    {trigger}: {count}x")

    # Prediction accuracy
    print(f"\n--- PREDICTION ACCURACY ---\n")
    pred_results = evaluate_predictions(decisions)
    if pred_results:
        correct = sum(1 for r in pred_results if r["correct"] is True)
        total = sum(1 for r in pred_results if r["correct"] is not None)
        accuracy = (correct / total * 100) if total > 0 else 0

        print(f"  Direction predictions: {correct}/{total} correct ({accuracy:.1f}%)")

        # By confidence bucket
        high_conf = [r for r in pred_results if r["confidence"] >= 0.6]
        med_conf = [r for r in pred_results if 0.3 <= r["confidence"] < 0.6]
        low_conf = [r for r in pred_results if r["confidence"] < 0.3]

        for label, bucket in [("High (≥0.6)", high_conf), ("Med (0.3-0.6)", med_conf), ("Low (<0.3)", low_conf)]:
            if bucket:
                bc = sum(1 for r in bucket if r["correct"] is True)
                bt = sum(1 for r in bucket if r["correct"] is not None)
                ba = (bc / bt * 100) if bt > 0 else 0
                print(f"    {label}: {bc}/{bt} ({ba:.1f}%)")

        # Recent 5 predictions
        print(f"\n  Last 5 predictions:")
        for r in pred_results[-5:]:
            mark = "✓" if r["correct"] else "✗" if r["correct"] is not None else "?"
            print(f"    [{mark}] {r['ts'][:16]} pred={r['direction']} conf={r['confidence']:.1f} "
                  f"silver ${r['silver_at_prediction']:.2f}→${r['silver_at_check']:.2f} ({r['actual_move_pct']:+.2f}%)")
    else:
        print("  No predictions to evaluate yet")

    # Per-position HOLD/SELL accuracy
    print(f"\n--- PER-POSITION DECISION ACCURACY ---\n")
    pos_stats = evaluate_hold_sells(decisions)
    for key, stats in pos_stats.items():
        entry = POSITIONS.get(key, {}).get("entry", 0)
        last_bid = 0
        for d in reversed(decisions):
            p = d.get("positions", {}).get(key, {})
            if p.get("bid"):
                last_bid = p["bid"]
                break
        last_pnl = ((last_bid - entry) / entry * 100) if entry and last_bid else 0

        print(f"  {key} (entry={entry}, last={last_bid}, pnl={last_pnl:+.2f}%):")
        print(f"    HOLD: {stats['holds_correct']}/{stats['holds_total']} correct ({stats['holds_accuracy']:.0f}%)")
        print(f"    SELL: {stats['sells_correct']}/{stats['sells_total']} correct ({stats['sells_accuracy']:.0f}%)")

    # Reflections summary
    print(f"\n--- REFLECTION HIGHLIGHTS ---\n")
    reflections = [d.get("reflection", "") for d in decisions if d.get("reflection")]
    if reflections:
        for r in reflections[-5:]:
            print(f"  • {r[:120]}")
    else:
        print("  No reflections recorded yet")

    print(f"\n{'='*60}")

def main():
    decisions = load_decisions()
    print_report(decisions)

if __name__ == "__main__":
    main()
