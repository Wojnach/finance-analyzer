"""Smart trigger system — detects meaningful market changes to reduce noise.

Sustained signal filter: signal flips only trigger after the new direction
holds for SUSTAINED_CHECKS consecutive loop cycles. With a 60s loop interval,
SUSTAINED_CHECKS=3 means a signal must hold for ~3 minutes before triggering.
"""

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "trigger_state.json"
COOLDOWN_SECONDS = 7200  # 2 hours max silence (market hours)
OFFHOURS_COOLDOWN = 21600  # 6 hours (nights/weekends, crypto only)
PRICE_THRESHOLD = 0.02  # 2% move
FG_THRESHOLDS = (20, 80)  # extreme fear / extreme greed boundaries
SUSTAINED_CHECKS = 3  # consecutive cycles a signal must hold before triggering


def _load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_state(state):
    STATE_FILE.parent.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_FILE.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, STATE_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def check_triggers(signals, prices_usd, fear_greeds, sentiments):
    state = _load_state()
    prev = state.get("last", {})
    sustained = state.get("sustained_counts", {})
    reasons = []

    # 1. Signal flip — only if sustained for SUSTAINED_CHECKS consecutive cycles
    prev_triggered = prev.get("signals", {})
    for ticker, sig in signals.items():
        current_action = sig["action"]
        prev_count = sustained.get(ticker, {})
        if prev_count.get("action") == current_action:
            sustained[ticker] = {
                "action": current_action,
                "count": prev_count["count"] + 1,
            }
        else:
            sustained[ticker] = {"action": current_action, "count": 1}

        triggered_action = prev_triggered.get(ticker, {}).get("action")
        if (
            triggered_action
            and current_action != triggered_action
            and sustained[ticker]["count"] >= SUSTAINED_CHECKS
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
    # Market hours (weekdays 07-21 UTC): 2h cooldown for all instruments
    # Off-hours (nights/weekends): 6h cooldown for crypto check-ins
    now_utc = datetime.now(timezone.utc)
    market_open = now_utc.weekday() < 5 and 7 <= now_utc.hour < 21
    last_trigger_time = state.get("last_trigger_time", 0)
    elapsed = time.time() - last_trigger_time
    if market_open and elapsed > COOLDOWN_SECONDS:
        reasons.append(f"cooldown ({COOLDOWN_SECONDS // 3600}h)")
    elif not market_open and elapsed > OFFHOURS_COOLDOWN:
        reasons.append(f"crypto check-in ({OFFHOURS_COOLDOWN // 3600}h)")

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

    state["sustained_counts"] = sustained
    _save_state(state)

    return triggered, reasons
