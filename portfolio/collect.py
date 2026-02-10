#!/usr/bin/env python3
"""Collect current market data and signals into a summary JSON for the Claude agent."""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from portfolio.main import (
    SYMBOLS,
    CONFIG_FILE,
    load_state,
    fetch_usd_sek,
    compute_indicators,
    generate_signal,
    collect_timeframes,
    portfolio_value,
)
from portfolio.fear_greed import get_fear_greed


def collect():
    config = json.loads(CONFIG_FILE.read_text())
    state = load_state()
    fx = fetch_usd_sek()
    prices = {}
    signals = {}
    timeframes = {}
    fear_greeds = {}

    for name, symbol in SYMBOLS.items():
        tfs = collect_timeframes(symbol)
        now_entry = tfs[0][1] if tfs else None
        if now_entry and "indicators" in now_entry:
            ind = now_entry["indicators"]
        else:
            continue
        price = ind["close"]
        prices[name] = price
        action, conf, extra = generate_signal(ind, ticker=name, config=config)
        signals[name] = {
            "action": action,
            "confidence": conf,
            "price_usd": price,
            "rsi": round(ind["rsi"], 1),
            "macd_hist": round(ind["macd_hist"], 2),
            "bb_position": ind["price_vs_bb"],
            "extra": extra,
        }
        try:
            fear_greeds[name] = get_fear_greed(ticker=name)
        except Exception:
            pass
        tf_summary = []
        for label, entry in tfs:
            if "error" in entry:
                tf_summary.append({"horizon": label, "error": entry["error"]})
            else:
                ei = entry["indicators"]
                tf_summary.append(
                    {
                        "horizon": label,
                        "action": entry["action"] if label != "Now" else action,
                        "confidence": entry["confidence"] if label != "Now" else conf,
                        "rsi": round(ei["rsi"], 1),
                        "macd_hist": round(ei["macd_hist"], 2),
                        "ema_bullish": ei["ema9"] > ei["ema21"],
                        "bb_position": ei["price_vs_bb"],
                    }
                )
        timeframes[name] = tf_summary

    total = portfolio_value(state, prices, fx)
    pnl_pct = ((total - state["initial_value_sek"]) / state["initial_value_sek"]) * 100

    summary = {
        "portfolio": {
            "total_sek": round(total),
            "pnl_pct": round(pnl_pct, 2),
            "cash_sek": round(state["cash_sek"]),
            "holdings": state.get("holdings", {}),
            "num_transactions": len(state.get("transactions", [])),
        },
        "fx_rate": round(fx, 2),
        "fear_greed": fear_greeds,
        "signals": signals,
        "timeframes": timeframes,
    }

    out = BASE_DIR / "data" / "agent_summary.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    collect()
