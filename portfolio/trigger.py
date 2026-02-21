"""Smart trigger system — detects meaningful market changes to reduce noise.

Layer 1 runs every minute during market hours. Layer 2 is invoked when:
- Signal consensus: any ticker has BUY or SELL consensus (signals agree)
- Signal flip sustained for SUSTAINED_CHECKS consecutive cycles (~3 min)
- Price moved >2% since last trigger
- Fear & Greed crossed extreme threshold (20 or 80)
- Sentiment reversal (positive↔negative)
- Cooldown expired (30 min market hours, 1h off-hours)

After a trade (BUY/SELL), the cooldown timer is reset so the agent can
reassess the new portfolio state promptly.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from portfolio.file_utils import atomic_write_json

BASE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = BASE_DIR / "data" / "trigger_state.json"
PORTFOLIO_FILE = BASE_DIR / "data" / "portfolio_state.json"
PORTFOLIO_BOLD_FILE = BASE_DIR / "data" / "portfolio_state_bold.json"

COOLDOWN_SECONDS = 1800  # 30 min max silence (market hours)
OFFHOURS_COOLDOWN = 3600  # 1 hour (nights/weekends, crypto only)
PRICE_THRESHOLD = 0.02  # 2% move
FG_THRESHOLDS = (20, 80)  # extreme fear / extreme greed boundaries
SUSTAINED_CHECKS = 3  # consecutive cycles a signal must hold before triggering


def _load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_state(state):
    atomic_write_json(STATE_FILE, state)


def _check_recent_trade(state):
    """Check if Layer 2 executed a trade since our last trigger.

    If so, reset the cooldown so the agent can reassess promptly.
    Returns True if a recent trade was detected.
    """
    last_trigger = state.get("last_trigger_time", 0)
    last_checked_tx = state.get("last_checked_tx_count", {})

    trade_detected = False
    new_tx_counts = {}

    for label, pf_file in [("patient", PORTFOLIO_FILE), ("bold", PORTFOLIO_BOLD_FILE)]:
        if not pf_file.exists():
            continue
        try:
            pf = json.loads(pf_file.read_text(encoding="utf-8"))
            txs = pf.get("transactions", [])
            current_count = len(txs)
            prev_count = last_checked_tx.get(label, current_count)
            new_tx_counts[label] = current_count

            if current_count > prev_count:
                trade_detected = True
        except (json.JSONDecodeError, KeyError):
            pass

    if new_tx_counts:
        state["last_checked_tx_count"] = new_tx_counts

    return trade_detected


def check_triggers(signals, prices_usd, fear_greeds, sentiments):
    state = _load_state()
    prev = state.get("last", {})
    sustained = state.get("sustained_counts", {})
    reasons = []

    now_utc = datetime.now(timezone.utc)
    from portfolio.market_timing import _market_close_hour_utc
    close_hour = _market_close_hour_utc(now_utc)
    market_open = now_utc.weekday() < 5 and 7 <= now_utc.hour < close_hour

    # 0. Trade reset — if Layer 2 made a trade, reset cooldown
    if _check_recent_trade(state):
        state["last_trigger_time"] = 0  # reset cooldown
        reasons.append("post-trade reassessment")

    # 1. Signal consensus — trigger when a ticker newly reaches BUY or SELL
    #    Only fires when consensus is different from the last triggered state
    prev_triggered = prev.get("signals", {})
    for ticker, sig in signals.items():
        action = sig["action"]
        if action in ("BUY", "SELL"):
            prev_action = prev_triggered.get(ticker, {}).get("action", "HOLD")
            if action != prev_action:
                conf = sig.get("confidence", 0)
                reasons.append(f"{ticker} consensus {action} ({conf:.0%})")

    # 2. Signal flip — only if sustained for SUSTAINED_CHECKS consecutive cycles
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

    # 3. Price move >2% since last trigger
    prev_prices = prev.get("prices", {})
    for ticker, price in prices_usd.items():
        old_price = prev_prices.get(ticker)
        if old_price and old_price > 0:
            pct = abs(price - old_price) / old_price
            if pct >= PRICE_THRESHOLD:
                direction = "up" if price > old_price else "down"
                reasons.append(f"{ticker} moved {pct:.1%} {direction}")

    # 4. Fear & Greed crossed threshold
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

    # 5. Sentiment reversal
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

    # 6. Cooldown expired — safety net to ensure periodic check-ins
    last_trigger_time = state.get("last_trigger_time", 0)
    elapsed = time.time() - last_trigger_time
    if market_open and elapsed > COOLDOWN_SECONDS:
        reasons.append(f"cooldown ({COOLDOWN_SECONDS // 60}min)")
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
