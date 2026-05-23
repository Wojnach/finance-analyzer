"""Quick backtest for autotune_adaptive_cycle signal."""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worktrees", "signal-research-2026-05-23"))

from portfolio.data_collector import yfinance_klines
from portfolio.signals.autotune_adaptive_cycle import compute_autotune_adaptive_cycle_signal

target_assets = ["BTC-USD", "ETH-USD", "GC=F", "SI=F", "MSTR"]
DISPLAY = {"GC=F": "XAU-USD", "SI=F": "XAG-USD"}
results = {}

for ticker in target_assets:
    display = DISPLAY.get(ticker, ticker)
    print(f"Backtesting {display}...")
    try:
        df = yfinance_klines(ticker, "1d", limit=365)
    except Exception as e:
        print(f"  Failed to fetch data: {e}")
        continue

    if df is None or len(df) < 70:
        print(f"  Insufficient data: {len(df) if df is not None else 0} rows")
        continue

    correct_1d, correct_3d, correct_5d = 0, 0, 0
    total = 0
    buy_count, sell_count = 0, 0

    for i in range(60, len(df) - 5):
        window = df.iloc[:i + 1].copy()
        sig = compute_autotune_adaptive_cycle_signal(window)
        if sig["action"] == "HOLD":
            continue

        total += 1
        if sig["action"] == "BUY":
            buy_count += 1
        else:
            sell_count += 1

        future_1d = df.iloc[i + 1]["close"] / df.iloc[i]["close"] - 1
        future_3d = df.iloc[min(i + 3, len(df) - 1)]["close"] / df.iloc[i]["close"] - 1
        future_5d = df.iloc[min(i + 5, len(df) - 1)]["close"] / df.iloc[i]["close"] - 1

        if sig["action"] == "BUY":
            if future_1d > 0.0005:
                correct_1d += 1
            if future_3d > 0.0005:
                correct_3d += 1
            if future_5d > 0.0005:
                correct_5d += 1
        elif sig["action"] == "SELL":
            if future_1d < -0.0005:
                correct_1d += 1
            if future_3d < -0.0005:
                correct_3d += 1
            if future_5d < -0.0005:
                correct_5d += 1

    if total > 0:
        results[ticker] = {
            "total_signals": total,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "accuracy_1d": round(correct_1d / total, 4),
            "accuracy_3d": round(correct_3d / total, 4),
            "accuracy_5d": round(correct_5d / total, 4),
            "activation_rate": round(total / (len(df) - 65), 4),
        }
        print(f"  {total} signals ({buy_count}B/{sell_count}S), 1d={correct_1d/total:.1%}, 3d={correct_3d/total:.1%}, 5d={correct_5d/total:.1%}")
    else:
        print(f"  No signals generated")

print("\n=== SUMMARY ===")
for ticker, r in results.items():
    print(f"{ticker}: {r['total_signals']} signals, 1d={r['accuracy_1d']:.1%}, 3d={r['accuracy_3d']:.1%}, 5d={r['accuracy_5d']:.1%}, activation={r['activation_rate']:.1%}")

if results:
    avg_1d = sum(r["accuracy_1d"] for r in results.values()) / len(results)
    print(f"\nAverage 1d accuracy: {avg_1d:.1%}")
    print(f"Assets tested: {len(results)}/{len(target_assets)}")
