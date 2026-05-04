"""Backtest MIN_BUY_CONFIDENCE 0.56 vs 0.60 (and a sweep) on the historical
metals_signal_log.

Method
------
1. Walk metals_signal_log.jsonl for XAG-USD and XAU-USD entries.
2. Treat consecutive same-direction signals as a single trade event (the
   SwingTrader has a cooldown + "already have a position" gate, so back-to-
   back BUYs from the same regime would not fire as new trades).
3. Fetch 1h underlying bars from yfinance for the date range.
4. For each first-BUY event whose confidence ≥ threshold, compute forward
   returns at +1h, +3h, +1d on the underlying. (Warrants amplify by ~5x
   on direction; we report underlying numbers and let the leverage scale
   linearly — direction quality is what changes between thresholds.)

Output
------
- per-threshold table: trade count, win rate, avg return, median return,
  std dev, expectancy
- saved CSV at data/backtest_conf_threshold/results.csv
- saved per-trade detail at data/backtest_conf_threshold/trades.csv

Run: .venv/Scripts/python.exe scripts/perf/backtest_conf_threshold.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median, pstdev

import pandas as pd
import yfinance as yf

REPO = Path(__file__).resolve().parents[2]
SIGNAL_LOG = REPO / "data" / "metals_signal_log.jsonl"
OUTDIR = REPO / "data" / "backtest_conf_threshold"
OUTDIR.mkdir(parents=True, exist_ok=True)

# yfinance symbols for underlyings the swing trader watches
YF_SYMBOLS = {"XAG-USD": "SI=F", "XAU-USD": "GC=F"}

THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]
HORIZONS_HOURS = [1, 3, 24]


def load_first_buy_events() -> list[dict]:
    """Return list of first-BUY events: each transition from non-BUY to BUY."""
    events: list[dict] = []
    last_action: dict[str, str] = {"XAG-USD": "HOLD", "XAU-USD": "HOLD"}
    with SIGNAL_LOG.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts_str = rec.get("ts")
            if not ts_str:
                continue
            ts = datetime.fromisoformat(ts_str)
            for ticker in ("XAG-USD", "XAU-USD"):
                sig = rec.get("signals", {}).get(ticker, {}) or {}
                action = sig.get("action", "HOLD")
                conf = float(sig.get("confidence") or 0.0)
                # First-BUY: previous was NOT BUY, this one IS BUY
                if action == "BUY" and last_action[ticker] != "BUY":
                    events.append({
                        "ts": ts,
                        "ticker": ticker,
                        "confidence": conf,
                        "rsi": sig.get("rsi"),
                        "regime": sig.get("regime"),
                        "buy_count": sig.get("buy_count"),
                        "sell_count": sig.get("sell_count"),
                        "voters": sig.get("voters"),
                    })
                last_action[ticker] = action
    return events


def fetch_prices(start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """Fetch 1h bars for SI=F and GC=F covering [start, end]."""
    out: dict[str, pd.DataFrame] = {}
    # Pad +2 days at end for the +24h horizon to land
    yf_end = end + timedelta(days=3)
    yf_start = start - timedelta(hours=2)
    for ticker, sym in YF_SYMBOLS.items():
        df = yf.download(
            sym,
            start=yf_start.strftime("%Y-%m-%d"),
            end=yf_end.strftime("%Y-%m-%d"),
            interval="1h",
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            print(f"WARN: no yfinance data for {sym}", file=sys.stderr)
            continue
        # Normalize tz to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        # Flatten multi-index columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out[ticker] = df[["Close"]].copy()
    return out


def lookup_price(df: pd.DataFrame, ts: datetime) -> float | None:
    """Return the close price of the nearest hourly bar on/after ts."""
    if df is None or df.empty:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    # Find the first index >= ts
    idx = df.index.searchsorted(ts)
    if idx >= len(df):
        return None
    return float(df["Close"].iloc[idx])


def compute_returns(events: list[dict], prices: dict[str, pd.DataFrame]) -> list[dict]:
    """Compute forward-return columns ret_1h, ret_3h, ret_1d for each event."""
    enriched: list[dict] = []
    for ev in events:
        ticker = ev["ticker"]
        df = prices.get(ticker)
        if df is None:
            continue
        p0 = lookup_price(df, ev["ts"])
        if p0 is None or p0 <= 0:
            continue
        ret = {"entry_price": p0}
        for h in HORIZONS_HOURS:
            p1 = lookup_price(df, ev["ts"] + timedelta(hours=h))
            ret[f"ret_{h}h"] = (p1 / p0 - 1.0) * 100.0 if p1 else None
        enriched.append({**ev, **ret})
    return enriched


def summarize(trades: list[dict], threshold: float, horizon_key: str) -> dict:
    """Return metrics for trades passing the confidence threshold."""
    rets = [t[horizon_key] for t in trades if t["confidence"] >= threshold and t.get(horizon_key) is not None]
    if not rets:
        return {
            "threshold": threshold,
            "horizon": horizon_key,
            "n": 0,
            "wins": 0,
            "win_rate": 0.0,
            "avg_ret_pct": 0.0,
            "med_ret_pct": 0.0,
            "stdev_ret_pct": 0.0,
            "expectancy_pct": 0.0,
        }
    wins = sum(1 for r in rets if r > 0)
    return {
        "threshold": threshold,
        "horizon": horizon_key,
        "n": len(rets),
        "wins": wins,
        "win_rate": wins / len(rets),
        "avg_ret_pct": mean(rets),
        "med_ret_pct": median(rets),
        "stdev_ret_pct": pstdev(rets) if len(rets) > 1 else 0.0,
        "expectancy_pct": mean(rets),  # avg return = expectancy per trade
    }


def main() -> int:
    print("Loading first-BUY events from metals_signal_log…")
    events = load_first_buy_events()
    if not events:
        print("No events found.", file=sys.stderr)
        return 1
    print(f"  {len(events)} first-BUY events across XAG-USD + XAU-USD")
    by_ticker: dict[str, int] = {}
    for ev in events:
        by_ticker[ev["ticker"]] = by_ticker.get(ev["ticker"], 0) + 1
    print(f"  per-ticker: {by_ticker}")

    start = min(e["ts"] for e in events)
    end = max(e["ts"] for e in events)
    print(f"  date range: {start.date()} -> {end.date()}")

    print("Fetching 1h price data from yfinance (SI=F, GC=F)…")
    prices = fetch_prices(start, end)
    for k, df in prices.items():
        print(f"  {k}: {len(df)} bars, {df.index.min()} -> {df.index.max()}")

    print("Computing forward returns…")
    trades = compute_returns(events, prices)
    print(f"  {len(trades)} trades with valid forward prices")

    # Save per-trade detail
    if trades:
        df_trades = pd.DataFrame(trades)
        df_trades["ts"] = df_trades["ts"].astype(str)
        df_trades.to_csv(OUTDIR / "trades.csv", index=False)
        print(f"  detail saved -> {OUTDIR / 'trades.csv'}")

    # Build summary across thresholds × horizons
    rows: list[dict] = []
    for thresh in THRESHOLDS:
        for h in HORIZONS_HOURS:
            rows.append(summarize(trades, thresh, f"ret_{h}h"))

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(OUTDIR / "results.csv", index=False)

    # Pretty print: pivot threshold × horizon, showing n / winrate / avg_ret
    print("\n=== Backtest results ===")
    print(f"  Underlying-only (no warrant leverage); apply ~5x for cert P&L\n")
    print(f"{'threshold':>10} {'horizon':>8} {'n':>5} {'wins':>5} {'winrate':>8} {'avg_ret_%':>10} {'med_%':>8} {'stdev_%':>8}")
    print("-" * 72)
    for r in rows:
        print(f"{r['threshold']:>10.2f} {r['horizon']:>8} {r['n']:>5d} {r['wins']:>5d} "
              f"{r['win_rate']:>7.1%} {r['avg_ret_pct']:>10.3f} "
              f"{r['med_ret_pct']:>8.3f} {r['stdev_ret_pct']:>8.3f}")
    print()
    print(f"Saved summary -> {OUTDIR / 'results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
