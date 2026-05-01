"""Score oil swing-trader DRY_RUN decisions against realized outcomes.

Mirrors `scripts/mstr_loop_scorecard.py` for the oil subsystem. Reads
`data/oil_swing_decisions.jsonl` + `data/oil_swing_trades.jsonl`, pairs
BUY with matching SELL, and computes per-instrument win rate, expectancy,
and equity curve.

The oil loop ships in DRY_RUN=True (per data/oil_swing_config.py). Until
that flag is flipped, the trades log will be empty and this scorecard
emits a "no closed trades yet" notice.

Usage:
    .venv/Scripts/python.exe scripts/oil_loop_scorecard.py

Output goes to stdout; machine-readable summary written to
data/oil_loop_scorecard.json.
"""

from __future__ import annotations

import collections
import datetime
import json
import statistics
import sys
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DECISIONS_LOG = DATA_DIR / "oil_swing_decisions.jsonl"
TRADES_LOG = DATA_DIR / "oil_swing_trades.jsonl"
OUTPUT = DATA_DIR / "oil_loop_scorecard.json"

# Live-flip readiness gates — analogous to MSTR's Phase A but with shorter
# horizon (oil's fundamentals shift faster than MSTR's NAV-tracking).
LIVE_MIN_DAYS = 30
LIVE_MIN_TRADES = 15
LIVE_MIN_WIN_RATE = 55.0  # percent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
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


def _ticker_of(event: dict[str, Any]) -> str:
    """Extract ticker. Per oil_swing_trader log format, the ticker lives
    inside `pos.ticker` for BUY-side events and `pos_id` (TICKER_<ts>)
    for SELL-side events. Falls back to top-level for forward-compat."""
    pos = event.get("pos") or {}
    if pos.get("ticker"):
        return pos["ticker"]
    pos_id = event.get("pos_id") or ""
    if "_" in pos_id:
        return pos_id.split("_", 1)[0]
    return event.get("ticker", "OIL-USD")


def _is_buy(action: str) -> bool:
    return action in ("BUY", "BUY_DRY_RUN")


def _is_sell(action: str) -> bool:
    return action in ("SELL", "SELL_DRY_RUN", "STOP", "EXIT")


def pair_trades(decisions: list[dict[str, Any]],
                 trades: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Pair BUY decisions with matching SELL trades by ticker.

    oil_swing_trader logs:
      Decisions: {"action":"BUY_DRY_RUN", "pos":{ticker,entry_underlying_price,..},
                  "warrant":{...}, "underlying_price":..., "ts":...}
      Decisions: {"action":"SELL_DRY_RUN", "pos_id":"TICKER_<ts>",
                  "underlying_price":..., "warrant_bid":..., "reason":..., "ts":...}
      Trades:    {"action":"SELL", "pos_id":..., "reason":..., "underlying_pct":...,
                  "exit_underlying":..., "exit_warrant_bid":..., "dry_run":True, "ts":...}

    P&L is reported as `underlying_pct` (signed for SHORT) on the SELL
    side. Trades log entries are authoritative for closed round-trips;
    decisions log entries flesh out the entry context.
    """
    paired: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    open_positions: dict[str, dict[str, Any]] = {}

    by_ts = sorted(decisions + trades,
                    key=lambda e: e.get("ts") or "")
    for e in by_ts:
        action = (e.get("action") or e.get("type") or "").upper()
        ticker = _ticker_of(e)
        if _is_buy(action):
            open_positions[ticker] = e
        elif _is_sell(action):
            opened = open_positions.pop(ticker, None)
            if opened is None:
                continue
            opened_pos = opened.get("pos") or {}
            entry_price = (opened_pos.get("entry_underlying_price")
                           or opened.get("underlying_price")
                           or opened.get("price"))
            exit_price = (e.get("exit_underlying")
                          or e.get("underlying_price")
                          or e.get("price"))
            paired[ticker].append({
                "entry_ts": opened.get("ts"),
                "exit_ts": e.get("ts"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "underlying_pct": e.get("underlying_pct"),
                "pnl_sek": e.get("pnl_sek", 0),
                "exit_reason": e.get("reason") or e.get("exit_reason"),
                "direction": opened_pos.get("direction"),
                "confidence_at_entry": (opened_pos.get("signal_context") or {})
                                       .get("confidence"),
            })
    return paired


def _pnl_of(trade: dict[str, Any]) -> float:
    """Return the trade's P&L. In DRY_RUN, oil_swing_trader records only
    `underlying_pct` (signed); we treat that as the P&L unit. In live
    mode, `pnl_sek` is the canonical field."""
    if trade.get("pnl_sek"):
        return float(trade["pnl_sek"])
    if trade.get("underlying_pct") is not None:
        return float(trade["underlying_pct"])
    return 0.0


def score(trades: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(trades)
    if n == 0:
        return {"n_trades": 0}
    pnls = [_pnl_of(t) for t in trades]
    # Distinguish dry-run (underlying_pct units) from live (sek units) for
    # the output labels — keeps the scorecard readable in either mode.
    pnl_unit = "sek" if any(t.get("pnl_sek") for t in trades) else "underlying_pct"
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n if n else 0.0
    avg_win = statistics.mean(wins) if wins else 0.0
    avg_loss = statistics.mean(losses) if losses else 0.0
    total_pnl = sum(pnls)
    expectancy = avg_win * win_rate + avg_loss * (1 - win_rate)
    running, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:
        running += p
        peak = max(peak, running)
        max_dd = max(max_dd, peak - running)
    return {
        "n_trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate * 100, 2),
        f"avg_win_{pnl_unit}": round(avg_win, 2),
        f"avg_loss_{pnl_unit}": round(avg_loss, 2),
        f"total_pnl_{pnl_unit}": round(total_pnl, 2),
        f"expectancy_per_trade_{pnl_unit}": round(expectancy, 2),
        f"max_drawdown_{pnl_unit}": round(max_dd, 2),
        "pnl_unit": pnl_unit,
    }


def compute_observation_window(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"days_observed": 0, "days_remaining": LIVE_MIN_DAYS,
                "min_days": LIVE_MIN_DAYS, "earliest_event_ts": None}
    timestamps = []
    for e in events:
        ts_str = e.get("ts")
        if not ts_str:
            continue
        try:
            ts = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.UTC)
            timestamps.append(ts)
        except ValueError:
            continue
    if not timestamps:
        return {"days_observed": 0, "days_remaining": LIVE_MIN_DAYS,
                "min_days": LIVE_MIN_DAYS, "earliest_event_ts": None}
    earliest = min(timestamps)
    now = datetime.datetime.now(datetime.UTC)
    days_observed = (now - earliest).total_seconds() / 86400.0
    return {
        "days_observed": round(days_observed, 2),
        "days_remaining": max(0, round(LIVE_MIN_DAYS - days_observed, 2)),
        "min_days": LIVE_MIN_DAYS,
        "earliest_event_ts": earliest.isoformat(),
    }


def main() -> int:
    decisions = load_jsonl(DECISIONS_LOG)
    trades = load_jsonl(TRADES_LOG)
    events = decisions + trades
    if not events:
        print("No decision/trade events yet — oil loop ships in DRY_RUN.")
        print("Once the loop runs, this scorecard will populate from")
        print(f"  {DECISIONS_LOG}")
        print(f"  {TRADES_LOG}")
        return 0

    window = compute_observation_window(events)
    paired = pair_trades(decisions, trades)
    n_paired = sum(len(v) for v in paired.values())

    print(f"Decisions: {len(decisions)} events. Trades: {len(trades)}. "
          f"Paired round-trips: {n_paired}.")
    print(f"Observation window: {window['days_observed']} / "
          f"{window['min_days']} days "
          f"({window['days_remaining']} remaining to live-flip readiness)")
    print()

    summary: dict[str, Any] = {
        "instruments": {},
        "total_events": len(events),
        "observation_window": window,
    }
    for ticker, ts in paired.items():
        stats = score(ts)
        n = stats.get("n_trades", 0)
        wr = stats.get("win_rate_pct", 0)
        ex = stats.get("expectancy_sek_per_trade", 0)
        gates = {
            "trades": n >= LIVE_MIN_TRADES,
            "days": window["days_observed"] >= LIVE_MIN_DAYS,
            "win_rate": wr >= LIVE_MIN_WIN_RATE,
            "expectancy": ex > 0,
        }
        stats["live_flip_gates"] = gates
        if all(gates.values()):
            stats["live_flip_status"] = "PASS"
        else:
            stats["live_flip_status"] = "WAIT"
        summary["instruments"][ticker] = stats
        print(f"=== {ticker} ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print()

    try:
        OUTPUT.write_text(json.dumps(summary, indent=2))
        print(f"Machine-readable summary → {OUTPUT}")
    except OSError as e:
        print(f"write summary failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
