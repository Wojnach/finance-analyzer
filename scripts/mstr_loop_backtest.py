"""Backtest the MSTR Loop momentum_rider strategy against historical signals.

Reads `data/signal_log.jsonl` for MSTR snapshots (contains per-cycle votes
but NOT per-cycle RSI). Reconstructs RSI from yfinance MSTR 1d OHLCV,
aligns with signal snapshots by timestamp, then replays the strategy rules
against synthesized bundles to measure how many entries would have fired
and what the cert-P&L would have been under the v1/v2 thresholds.

Two modes:
  --mode logic-only   — count signal-gate entries/exits; no price outcomes.
  --mode full         — also fetch MSTR OHLCV + compute realized P&L.

Outputs `data/mstr_loop_backtest_results.json` with per-threshold sweep.

Usage:
    .venv/Scripts/python.exe scripts/mstr_loop_backtest.py --mode full
    .venv/Scripts/python.exe scripts/mstr_loop_backtest.py \\
        --buy-threshold 0.50 0.55 0.60 --trail-pct 1.5 2.0 2.5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
SIGNAL_LOG = DATA_DIR / "signal_log.jsonl"
OUTPUT = DATA_DIR / "mstr_loop_backtest_results.json"

sys.path.insert(0, str(REPO_ROOT))
from portfolio.mstr_loop import config
from portfolio.mstr_loop.data_provider import (
    _compute_weighted_scores, _parse_vote_detail,
)


@dataclass
class Trade:
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str


def load_mstr_signal_entries(
    path: Path = SIGNAL_LOG, ticker: str = "MSTR",
) -> list[dict[str, Any]]:
    """Stream signal_log.jsonl, return only entries containing a MSTR block.

    signal_log.jsonl entries have the shape
    ``{"ts": "...", "tickers": {"MSTR": {...}, "BTC-USD": {...}, ...}}``
    We flatten to a list of MSTR-only snapshots with the parent ts attached.
    """
    out = []
    if not path.exists():
        print(f"no signal_log at {path}", file=sys.stderr)
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tickers = row.get("tickers") or {}
            mstr = tickers.get(ticker)
            if not mstr:
                continue
            record = dict(mstr)
            record["_ts"] = row.get("ts") or row.get("timestamp")
            out.append(record)
    return out


def reconstruct_votes(entry: dict[str, Any]) -> dict[str, str]:
    """Extract per-signal votes from a signal_log entry.

    signal_log entries store `signals` as {signal_name: {"vote": "BUY"|...}}
    or as a flat {signal_name: "BUY"} map. Normalize both shapes.
    """
    sigs = entry.get("signals") or {}
    votes = {}
    for name, payload in sigs.items():
        if isinstance(payload, dict):
            v = payload.get("vote") or payload.get("action")
        else:
            v = payload
        if v in ("BUY", "SELL", "HOLD"):
            votes[name] = v
    # Fallback: try _vote_detail in extras
    if not votes:
        vd = entry.get("extra", {}).get("_vote_detail") or entry.get("_vote_detail")
        if vd:
            votes = _parse_vote_detail(vd)
    return votes


def compute_rsi_series(closes: list[float], period: int = 14) -> list[float]:
    """Simple RSI(period) — Wilder smoothing, minimal impl for backtest."""
    if len(closes) < period + 1:
        return [50.0] * len(closes)
    rsi = [50.0] * period
    gains, losses = [], []
    for i in range(1, period + 1):
        chg = closes[i] - closes[i - 1]
        gains.append(max(0.0, chg))
        losses.append(max(0.0, -chg))
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    rs = avg_g / avg_l if avg_l > 0 else 99.0
    rsi.append(100 - (100 / (1 + rs)))
    for i in range(period + 1, len(closes)):
        chg = closes[i] - closes[i - 1]
        g = max(0.0, chg)
        l = max(0.0, -chg)
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
        rs = avg_g / avg_l if avg_l > 0 else 99.0
        rsi.append(100 - (100 / (1 + rs)))
    return rsi


def fetch_mstr_closes_yfinance(period: str = "3mo") -> dict[str, float]:
    """Fetch MSTR 1d closes from yfinance. Returns {YYYY-MM-DD: close}."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance unavailable — cannot run full-P&L mode", file=sys.stderr)
        return {}
    t = yf.Ticker("MSTR")
    hist = t.history(period=period, interval="1d")
    return {str(idx.date()): float(close) for idx, close in zip(hist.index, hist["Close"])}


def simulate(
    entries: list[dict[str, Any]],
    buy_threshold: float,
    trail_activation_pct: float,
    trail_distance_pct: float,
    hard_stop_pct: float,
    rsi_min: float,
    rsi_max: float,
    closes_by_date: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Replay the momentum_rider entry/exit logic against historical entries."""
    trades: list[Trade] = []
    open_entry_ts: str | None = None
    open_entry_price: float = 0.0
    open_peak: float = 0.0
    open_trail_active: bool = False

    for i, e in enumerate(entries):
        votes = reconstruct_votes(e)
        long_s, short_s = _compute_weighted_scores(votes)
        price = float(e.get("price_usd") or 0)
        if price <= 0 and closes_by_date:
            date = (e.get("_ts") or "")[:10]
            price = closes_by_date.get(date, 0.0)
        if price <= 0:
            continue

        # Estimate RSI from recent closes if available
        if closes_by_date:
            # Use closes BEFORE current timestamp for RSI
            cur_date = (e.get("_ts") or "")[:10]
            sorted_dates = sorted([d for d in closes_by_date if d <= cur_date])
            closes_for_rsi = [closes_by_date[d] for d in sorted_dates[-30:]]
            rsi_series = compute_rsi_series(closes_for_rsi)
            current_rsi = rsi_series[-1] if rsi_series else 50.0
        else:
            current_rsi = 50.0  # no RSI available → gate passes (optimistic)

        if open_entry_ts is None:
            # Consider entry
            if long_s < buy_threshold:
                continue
            if not (rsi_min <= current_rsi <= rsi_max):
                continue
            open_entry_ts = str(e.get("_ts") or "")
            open_entry_price = price
            open_peak = price
            open_trail_active = False
            continue

        # In position — consider exit
        pnl_pct = (price - open_entry_price) / open_entry_price * 100
        if price > open_peak:
            open_peak = price
        if pnl_pct >= trail_activation_pct:
            open_trail_active = True

        exit_reason = None
        if pnl_pct <= -hard_stop_pct:
            exit_reason = "stop"
        elif short_s >= buy_threshold:
            exit_reason = "signal_flip"
        elif open_trail_active and open_peak > 0:
            pullback = (open_peak - price) / open_peak * 100
            if pullback >= trail_distance_pct:
                exit_reason = "trail"
        if exit_reason is not None:
            trades.append(Trade(
                entry_ts=open_entry_ts, exit_ts=str(e.get("_ts") or ""),
                entry_price=open_entry_price, exit_price=price,
                pnl_pct=pnl_pct, exit_reason=exit_reason,
            ))
            open_entry_ts = None

    n = len(trades)
    wins = [t for t in trades if t.pnl_pct > 0]
    losses = [t for t in trades if t.pnl_pct <= 0]
    total_pnl_pct = sum(t.pnl_pct for t in trades)
    avg_win = sum(t.pnl_pct for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) if losses else 0.0
    win_rate = (len(wins) / n * 100) if n else 0.0
    expectancy_pct = avg_win * (win_rate / 100) + avg_loss * (1 - win_rate / 100)
    return {
        "params": {
            "buy_threshold": buy_threshold,
            "trail_activation_pct": trail_activation_pct,
            "trail_distance_pct": trail_distance_pct,
            "hard_stop_pct": hard_stop_pct,
            "rsi_min": rsi_min, "rsi_max": rsi_max,
        },
        "n_trades": n,
        "wins": len(wins), "losses": len(losses),
        "win_rate_pct": round(win_rate, 2),
        "avg_win_pct": round(avg_win, 3),
        "avg_loss_pct": round(avg_loss, 3),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "expectancy_pct": round(expectancy_pct, 3),
        "exit_reasons": {
            "stop": sum(1 for t in trades if t.exit_reason == "stop"),
            "signal_flip": sum(1 for t in trades if t.exit_reason == "signal_flip"),
            "trail": sum(1 for t in trades if t.exit_reason == "trail"),
        },
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["logic-only", "full"], default="full")
    p.add_argument("--buy-threshold", nargs="+", type=float,
                   default=[0.45, 0.50, 0.55, 0.60, 0.65])
    p.add_argument("--trail-pct", nargs="+", type=float, default=[1.5, 2.0, 2.5])
    p.add_argument("--hard-stop-pct", type=float, default=2.0)
    p.add_argument("--trail-activation-pct", type=float, default=1.5)
    p.add_argument("--rsi-min", type=float, default=40)
    p.add_argument("--rsi-max", type=float, default=78)
    args = p.parse_args()

    entries = load_mstr_signal_entries()
    print(f"Loaded {len(entries)} MSTR signal snapshots")
    if not entries:
        return 1

    closes = {}
    if args.mode == "full":
        closes = fetch_mstr_closes_yfinance("3mo")
        print(f"Fetched {len(closes)} MSTR 1d closes from yfinance")

    results = []
    for bt in args.buy_threshold:
        for tp in args.trail_pct:
            r = simulate(
                entries=entries,
                buy_threshold=bt,
                trail_activation_pct=args.trail_activation_pct,
                trail_distance_pct=tp,
                hard_stop_pct=args.hard_stop_pct,
                rsi_min=args.rsi_min, rsi_max=args.rsi_max,
                closes_by_date=closes,
            )
            results.append(r)
            print(f"buy≥{bt:.2f}  trail {tp:.1f}%:  "
                  f"n={r['n_trades']:<3} win={r['win_rate_pct']:>5.1f}%  "
                  f"tot={r['total_pnl_pct']:+6.2f}%  "
                  f"exp={r['expectancy_pct']:+5.2f}%")

    # Rank by expectancy
    results.sort(key=lambda x: x["expectancy_pct"], reverse=True)
    summary = {
        "universe": "MSTR",
        "n_signal_snapshots": len(entries),
        "n_parameter_combos": len(results),
        "top_5_by_expectancy": results[:5],
        "bottom_5_by_expectancy": results[-5:],
        "all_results": results,
    }
    try:
        OUTPUT.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {OUTPUT}")
    except OSError as e:
        print(f"write failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
