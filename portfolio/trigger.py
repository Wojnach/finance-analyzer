"""Smart trigger system — detects meaningful market changes to reduce noise."""

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
    reasons = []

    # 1. Signal flip — any symbol's action changed
    prev_signals = prev.get("signals", {})
    for ticker, sig in signals.items():
        old_action = prev_signals.get(ticker, {}).get("action")
        if old_action and old_action != sig["action"]:
            reasons.append(f"{ticker} flipped {old_action}->{sig['action']}")

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
    _save_state(state)

    return triggered, reasons
