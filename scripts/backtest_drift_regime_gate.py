"""Quick backtest for drift_regime_gate signal on all Tier-1 tickers.

Fetches 1 year of daily data from yfinance, computes signal at each bar,
then measures 1d/3d/5d forward accuracy.
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio.signals.drift_regime_gate import (
    MIN_ROWS,
    compute_drift_regime_gate_signal,
)

TICKERS_YF = {
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "XAU-USD": "GC=F",
    "XAG-USD": "SI=F",
    "MSTR": "MSTR",
}

HORIZONS = [1, 3, 5]


def fetch_data(yf_symbol: str, days: int = 400) -> pd.DataFrame:
    """Fetch daily OHLCV data from yfinance."""
    import yfinance as yf
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(yf_symbol, start=start.strftime("%Y-%m-%d"),
                     end=end.strftime("%Y-%m-%d"), progress=False)
    if df.empty:
        return pd.DataFrame()
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            return pd.DataFrame()
    return df[["open", "high", "low", "close", "volume"]].copy()


def backtest_ticker(ticker: str, yf_sym: str) -> dict:
    """Run signal on rolling windows and measure forward accuracy."""
    df = fetch_data(yf_sym)
    if len(df) < MIN_ROWS + max(HORIZONS) + 10:
        return {"ticker": ticker, "error": f"insufficient data ({len(df)} rows)"}

    results = {h: {"correct": 0, "total": 0, "buy_correct": 0, "buy_total": 0,
                    "sell_correct": 0, "sell_total": 0} for h in HORIZONS}

    for i in range(MIN_ROWS, len(df) - max(HORIZONS)):
        window = df.iloc[:i+1].copy()
        sig = compute_drift_regime_gate_signal(window)
        action = sig["action"]
        if action == "HOLD":
            continue

        current_close = df["close"].iloc[i]
        for h in HORIZONS:
            future_close = df["close"].iloc[i + h]
            fwd_return = (future_close - current_close) / current_close

            if action == "BUY":
                correct = fwd_return > 0
                results[h]["buy_total"] += 1
                if correct:
                    results[h]["buy_correct"] += 1
            else:
                correct = fwd_return < 0
                results[h]["sell_total"] += 1
                if correct:
                    results[h]["sell_correct"] += 1

            results[h]["total"] += 1
            if correct:
                results[h]["correct"] += 1

    summary = {"ticker": ticker, "bars": len(df)}
    for h in HORIZONS:
        r = results[h]
        total = r["total"]
        if total > 0:
            summary[f"{h}d_accuracy"] = round(r["correct"] / total * 100, 1)
            summary[f"{h}d_samples"] = total
            summary[f"{h}d_buy_acc"] = round(r["buy_correct"] / r["buy_total"] * 100, 1) if r["buy_total"] > 0 else None
            summary[f"{h}d_sell_acc"] = round(r["sell_correct"] / r["sell_total"] * 100, 1) if r["sell_total"] > 0 else None
            summary[f"{h}d_buy_n"] = r["buy_total"]
            summary[f"{h}d_sell_n"] = r["sell_total"]
        else:
            summary[f"{h}d_accuracy"] = None
            summary[f"{h}d_samples"] = 0
    return summary


def main():
    print("=" * 70)
    print("DRIFT REGIME GATE — BACKTEST (1yr daily, all Tier-1 tickers)")
    print("=" * 70)

    all_results = []
    for ticker, yf_sym in TICKERS_YF.items():
        print(f"\n--- {ticker} ({yf_sym}) ---")
        try:
            result = backtest_ticker(ticker, yf_sym)
        except Exception as e:
            result = {"ticker": ticker, "error": str(e)}
        all_results.append(result)
        if "error" not in result:
            for h in HORIZONS:
                acc = result.get(f"{h}d_accuracy")
                n = result.get(f"{h}d_samples", 0)
                if acc is not None:
                    print(f"  {h}d: {acc}% ({n} samples)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in all_results:
        if "error" in r:
            print(f"  {r['ticker']}: ERROR — {r['error']}")
            continue
        parts = [f"{r['ticker']} ({r['bars']} bars):"]
        for h in HORIZONS:
            acc = r.get(f"{h}d_accuracy")
            n = r.get(f"{h}d_samples", 0)
            if acc is not None:
                parts.append(f"{h}d={acc}% ({n})")
        print("  " + "  ".join(parts))

    out_path = Path(__file__).resolve().parent.parent / "data" / "drift_regime_gate_backtest.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
