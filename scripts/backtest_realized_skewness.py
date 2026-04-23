"""Quick backtest for realized_skewness_directional signal."""
import json
import sys
import os

# Add worktree to path so we import the new signal
sys.path.insert(0, os.path.join("worktrees", "signal-research-2026-04-23"))

from portfolio.data_collector import yfinance_klines
from portfolio.signals.realized_skewness import compute_realized_skewness_signal

target_assets = ["BTC-USD", "ETH-USD", "XAU-USD", "XAG-USD", "MSTR"]
results = {}

for ticker in target_assets:
    print(f"Backtesting {ticker}...")
    try:
        df = yfinance_klines(ticker, "1d", limit=365)
    except Exception as e:
        print(f"  Skipped: {e}")
        continue
    if df is None or len(df) < 80:
        print(f"  Skipped: insufficient data ({len(df) if df is not None else 0} rows)")
        continue

    correct_1d, correct_3d, correct_5d = 0, 0, 0
    total = 0

    for i in range(80, len(df) - 5):
        window = df.iloc[: i + 1].copy()
        sig = compute_realized_skewness_signal(window)
        if sig["action"] == "HOLD":
            continue

        total += 1
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
            "accuracy_1d": round(correct_1d / total, 4),
            "accuracy_3d": round(correct_3d / total, 4),
            "accuracy_5d": round(correct_5d / total, 4),
        }
        print(
            f"  Signals: {total}, "
            f"1d: {correct_1d / total:.1%}, "
            f"3d: {correct_3d / total:.1%}, "
            f"5d: {correct_5d / total:.1%}"
        )
    else:
        print(f"  No signals generated (all HOLD)")

print("\n=== BACKTEST RESULTS ===")
print(json.dumps(results, indent=2))

if results:
    best_acc = max(r.get("accuracy_1d", 0) for r in results.values())
    print(f"\nBest 1d accuracy: {best_acc:.1%}")
    if best_acc < 0.45:
        print("WARNING: 1d accuracy < 45% on all assets")
