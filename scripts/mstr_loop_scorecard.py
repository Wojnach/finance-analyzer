"""Score Phase B shadow decisions against realized outcomes.

Reads data/mstr_loop_shadow.jsonl, pairs SHADOW_BUY with matching
SHADOW_SELL, computes per-strategy win rate, expectancy, and equity curve.

Usage:
    .venv/Scripts/python.exe scripts/mstr_loop_scorecard.py

Output goes to stdout; machine-readable summary written to
data/mstr_loop_scorecard.json.
"""

from __future__ import annotations

import collections
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SHADOW_LOG = DATA_DIR / "mstr_loop_shadow.jsonl"
OUTPUT = DATA_DIR / "mstr_loop_scorecard.json"


def load_shadow(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"no shadow log at {path}")
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def pair_trades(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Pair SHADOW_BUY with matching SHADOW_SELL, grouped by strategy_key.

    A trade is a BUY followed by the next SELL for the same strategy.
    Unclosed BUYs (no matching SELL) are tracked as open positions.
    """
    trades: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    open_positions: dict[str, dict[str, Any]] = {}
    for e in events:
        key = e.get("strategy_key", "unknown")
        event = e.get("event")
        if event == "SHADOW_BUY":
            open_positions[key] = e
        elif event == "SHADOW_SELL":
            opened = open_positions.pop(key, None)
            if opened is None:
                continue
            trades[key].append({
                "entry_ts": opened["ts"],
                "exit_ts": e["ts"],
                "direction": e.get("direction", opened.get("direction", "?")),
                "entry_underlying": opened.get("underlying_price"),
                "exit_underlying": e.get("underlying_price"),
                "entry_cert": opened.get("cert_price"),
                "exit_cert": e.get("cert_price"),
                "units": e.get("units", opened.get("units")),
                "pnl_sek": e.get("pnl_sek", 0),
                "exit_reason": e.get("exit_reason"),
                "confidence_at_entry": opened.get("confidence"),
                "weighted_long_at_entry": opened.get("weighted_score_long"),
            })
    return trades


def score_strategy(trades: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0}
    pnls = [float(t.get("pnl_sek", 0)) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n if n else 0.0
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = statistics.mean(losses) if losses else 0.0
    total_pnl = sum(pnls)
    # Expectancy = avg_win × win_rate - |avg_loss| × (1 - win_rate)
    expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)
    # Max drawdown on cumulative P&L
    equity = []
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        running += p
        equity.append(running)
        peak = max(peak, running)
        dd = peak - running
        if dd > max_dd:
            max_dd = dd

    # Per-underlying-% slice
    under_pcts = []
    for t in trades:
        e_u = t.get("entry_underlying") or 0
        x_u = t.get("exit_underlying") or 0
        if e_u > 0:
            pct = (x_u - e_u) / e_u * 100
            if t.get("direction") == "SHORT":
                pct = -pct
            under_pcts.append(pct)
    return {
        "n_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate * 100, 2),
        "avg_win_sek": round(avg_win, 2),
        "avg_loss_sek": round(avg_loss, 2),
        "total_pnl_sek": round(total_pnl, 2),
        "expectancy_sek_per_trade": round(expectancy, 2),
        "max_drawdown_sek": round(max_dd, 2),
        "avg_underlying_move_pct": (
            round(statistics.mean(under_pcts), 3) if under_pcts else 0.0
        ),
        "median_underlying_move_pct": (
            round(statistics.median(under_pcts), 3) if under_pcts else 0.0
        ),
    }


def main() -> int:
    events = load_shadow(SHADOW_LOG)
    if not events:
        return 0
    trades_by_strategy = pair_trades(events)

    print(f"Shadow log: {len(events)} events, {sum(len(v) for v in trades_by_strategy.values())} closed trades")
    print()
    summary: dict[str, Any] = {"strategies": {}, "total_events": len(events)}
    for key, trades in trades_by_strategy.items():
        stats = score_strategy(trades)
        summary["strategies"][key] = stats
        print(f"=== {key} ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        # Phase-B graduation gate hints
        if stats.get("n_trades", 0) >= 25:
            wr = stats.get("win_rate_pct", 0)
            ex = stats.get("expectancy_sek_per_trade", 0)
            print(f"  GRADUATION: {'PASS' if wr >= 55 and ex > 0 else 'WAIT'} "
                  f"(need ≥55% win rate AND positive expectancy; have {wr}% / {ex})")
        else:
            print(f"  GRADUATION: need {25 - stats['n_trades']} more trades")
        print()

    try:
        OUTPUT.write_text(json.dumps(summary, indent=2))
        print(f"Machine-readable summary → {OUTPUT}")
    except OSError as e:
        print(f"write summary failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
