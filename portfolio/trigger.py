"""Smart trigger system — detects meaningful market changes to reduce noise.

Sustained signal filter: signal flips only trigger if the new direction was
already present in the previous check cycle (held for 2 consecutive checks).
This filters out noise (BUY→HOLD→BUY chattering) while adding at most 60s
latency since the loop runs every 60s.
"""

import json
import time
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "trigger_state.json"
COOLDOWN_SECONDS = 7200  # 2 hours max silence
PRICE_THRESHOLD = 0.02  # 2% move
FG_THRESHOLDS = (20, 80)  # extreme fear / extreme greed boundaries


def _load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def check_triggers(signals, prices_usd, fear_greeds, sentiments):
    state = _load_state()
    prev = state.get("last", {})
    prev_cycle = state.get("prev_cycle_signals", {})
    reasons = []

    # 1. Signal flip — only if sustained (same direction for 2 consecutive checks)
    prev_triggered = prev.get("signals", {})
    for ticker, sig in signals.items():
        triggered_action = prev_triggered.get(ticker, {}).get("action")
        prev_cycle_action = prev_cycle.get(ticker, {}).get("action")
        current_action = sig["action"]
        if (
            triggered_action
            and current_action != triggered_action
            and current_action == prev_cycle_action
        ):
            reasons.append(
                f"{ticker} flipped {triggered_action}->{current_action} (sustained)"
            )

    # 2. Price move >2% since last trigger
    prev_prices = prev.get("prices", {})
    for ticker, price in prices_usd.items():
        old_price = prev_prices.get(ticker)
        if old_price and old_price > 0:
            pct = abs(price - old_price) / old_price
            if pct >= PRICE_THRESHOLD:
                direction = "up" if price > old_price else "down"
                reasons.append(f"{ticker} moved {pct:.1%} {direction}")

    # 3. Fear & Greed crossed threshold
    prev_fg = prev.get("fear_greeds", {})
    for ticker, fg in fear_greeds.items():
        val = fg.get("value", 50) if isinstance(fg, dict) else 50
        old_val = (
            prev_fg.get(ticker, {}).get("value", 50)
            if isinstance(prev_fg.get(ticker), dict)
            else 50
        )
        for threshold in FG_THRESHOLDS:
            if (old_val > threshold) != (val > threshold):
                reasons.append(f"F&G crossed {threshold} ({old_val}->{val})")
                break

    # 4. Sentiment reversal
    prev_sent = prev.get("sentiments", {})
    for ticker, sent in sentiments.items():
        old_sent = prev_sent.get(ticker)
        if (
            old_sent
            and old_sent != sent
            and sent != "neutral"
            and old_sent != "neutral"
        ):
            reasons.append(f"{ticker} sentiment {old_sent}->{sent}")

    # 5. Cooldown expired
    last_trigger_time = state.get("last_trigger_time", 0)
    if time.time() - last_trigger_time > COOLDOWN_SECONDS:
        reasons.append(f"cooldown ({COOLDOWN_SECONDS // 3600}h)")

    triggered = len(reasons) > 0

    if triggered:
        state["last_trigger_time"] = time.time()
        state["last"] = {
            "signals": {
                t: {"action": s["action"], "confidence": s["confidence"]}
                for t, s in signals.items()
            },
            "prices": dict(prices_usd),
            "fear_greeds": {
                t: fg if isinstance(fg, dict) else {} for t, fg in fear_greeds.items()
            },
            "sentiments": dict(sentiments),
            "time": time.time(),
        }

    # Always save current signals as prev_cycle for next check
    state["prev_cycle_signals"] = {
        t: {"action": s["action"], "confidence": s["confidence"]}
        for t, s in signals.items()
    }
    _save_state(state)

    return triggered, reasons
