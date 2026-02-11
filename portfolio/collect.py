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
    generate_signal,
    collect_timeframes,
    write_agent_summary,
)


def collect():
    config = json.loads(CONFIG_FILE.read_text())
    state = load_state()
    fx = fetch_usd_sek()
    prices = {}
    signals = {}
    tf_data = {}

    for name, source in SYMBOLS.items():
        tfs = collect_timeframes(source)
        tf_data[name] = tfs
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
            "indicators": ind,
            "extra": extra,
        }

    summary = write_agent_summary(signals, prices, fx, state, tf_data)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    collect()
