"""Smart trigger system — detects meaningful market changes to reduce noise.

Layer 1 runs on a 10-minute cadence during every market state (see
``portfolio/market_timing.py:INTERVAL_MARKET_OPEN``). Layer 2 is invoked when:
- Signal consensus: any ticker NEWLY reaches BUY or SELL from HOLD
- Signal flip sustained for SUSTAINED_CHECKS consecutive cycles (see below)
- Price moved >2% since last trigger
- Fear & Greed crossed extreme threshold (20 or 80)
- Sentiment reversal: sustained for SUSTAINED_CHECKS cycles (filters oscillation)
- Post-trade reassessment: after a BUY/SELL trade

No periodic cooldown — Layer 2 is only invoked when Layer 1 detects a
meaningful change. The Tier 3 periodic full review (every 2h market / 4h
off-hours) provides the "heartbeat" via classify_tier(), but only when
another trigger has already fired.
"""

import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.trigger")

BASE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = BASE_DIR / "data" / "trigger_state.json"
PORTFOLIO_FILE = BASE_DIR / "data" / "portfolio_state.json"
PORTFOLIO_BOLD_FILE = BASE_DIR / "data" / "portfolio_state_bold.json"

PRICE_THRESHOLD = 0.02  # 2% move
FG_THRESHOLDS = (20, 80)  # extreme fear / extreme greed boundaries
# A signal flip triggers Layer 2 when EITHER of these holds:
#   - SUSTAINED_CHECKS consecutive cycles show the new action, OR
#   - SUSTAINED_DURATION_S seconds of wall-clock time have elapsed since
#     the flip first appeared.
# The count path is the original behavior (unchanged at the 60s cadence).
# The duration path is new (added 2026-04-09 with the cadence bump to 600s);
# at 600s cadence the count path would require ≥30 min of sustained flip
# before triggering, which effectively disables the trigger for fast-moving
# events. The duration gate bounds the worst case to ~1 cycle after flip
# (≈10 min at 600s cadence, ≈2 min at 60s cadence — both unchanged or better
# than the old count-only behavior).
SUSTAINED_CHECKS = 3
SUSTAINED_DURATION_S = 120

# Startup grace period — after a restart, the first loop iteration updates the
# baseline without triggering Layer 2. This prevents spurious T3 full reviews
# every time the loop is restarted for a code update.
_GRACE_PERIOD_KEY = "last_loop_pid"  # stored in trigger_state.json
_startup_grace_active = True  # True until first check_triggers call completes


def _today_str():
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _load_state():
    return load_json(STATE_FILE, default={})


def _save_state(state):
    # Prune triggered_consensus entries for tickers not in current signals
    # to prevent unbounded growth when tickers are removed from tracking
    tc = state.get("triggered_consensus", {})
    current_tickers = state.get("_current_tickers")
    if current_tickers is not None:
        removed = {k for k in tc if k not in current_tickers}
        if removed:
            logger.info("trigger: pruning %d stale ticker(s) from baseline: %s", len(removed), ", ".join(sorted(removed)))
        pruned = {k: v for k, v in tc.items() if k in current_tickers}
        state["triggered_consensus"] = pruned
    state.pop("_current_tickers", None)  # don't persist internal field
    atomic_write_json(STATE_FILE, state)


def _check_recent_trade(state):
    """Check if Layer 2 executed a trade since our last trigger.

    Returns True if a recent trade was detected.
    """
    last_checked_tx = state.get("last_checked_tx_count", {})

    trade_detected = False
    new_tx_counts = {}

    for label, pf_file in [("patient", PORTFOLIO_FILE), ("bold", PORTFOLIO_BOLD_FILE)]:
        try:
            pf = load_json(pf_file, default=None)
            if pf is None:
                continue
            txs = pf.get("transactions", [])
            current_count = len(txs)
            prev_count = last_checked_tx.get(label, current_count)
            new_tx_counts[label] = current_count

            if current_count > prev_count:
                trade_detected = True
        except (KeyError, AttributeError) as exc:
            logger.warning("Failed to parse portfolio file %s: %s", pf_file, exc)

    if new_tx_counts:
        state["last_checked_tx_count"] = new_tx_counts

    return trade_detected


def check_triggers(signals, prices_usd, fear_greeds, sentiments):
    global _startup_grace_active
    state = _load_state()
    state["_current_tickers"] = set(signals.keys())  # for pruning in _save_state

    # Startup grace period: on the first iteration after a restart, update the
    # baseline (prices, signals, consensus) WITHOUT triggering Layer 2.
    # This lets the loop restart for code updates without spurious T3 reviews.
    current_pid = os.getpid()
    saved_pid = state.get(_GRACE_PERIOD_KEY)
    if _startup_grace_active and saved_pid != current_pid:
        import logging
        _logger = logging.getLogger("portfolio.trigger")
        _logger.info(
            "Startup grace period: updating baseline without triggering "
            "(pid %s -> %s)", saved_pid, current_pid,
        )
        state[_GRACE_PERIOD_KEY] = current_pid
        # Update baselines so next iteration compares from NOW
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
        # Update triggered_consensus baseline to current state
        tc = state.get("triggered_consensus", {})
        for ticker, sig in signals.items():
            tc[ticker] = sig["action"]
        state["triggered_consensus"] = tc
        state["today_date"] = _today_str()
        _startup_grace_active = False
        _save_state(state)
        return False, []

    _startup_grace_active = False
    prev = state.get("last", {})
    sustained = state.get("sustained_counts", {})
    reasons = []

    # 0. Trade reset — if Layer 2 made a trade, trigger reassessment
    if _check_recent_trade(state):
        state["last_trigger_time"] = 0
        reasons.append("post-trade reassessment")

    # 1. Signal consensus — trigger ONLY when a ticker first reaches BUY/SELL
    #    from HOLD. BUY↔SELL direction flips are handled by the sustained flip
    #    trigger (#2). Uses persistent triggered_consensus that is NOT wiped
    #    when unrelated triggers (sentiment, etc.) fire.
    triggered_consensus = state.get("triggered_consensus", {})
    for ticker, sig in signals.items():
        action = sig["action"]
        last_tc = triggered_consensus.get(ticker, "HOLD")
        if action in ("BUY", "SELL") and last_tc == "HOLD":
            # New consensus from HOLD — trigger immediately
            conf = sig.get("confidence", 0)
            reasons.append(f"{ticker} consensus {action} ({conf:.0%})")
            triggered_consensus[ticker] = action
        elif action == "HOLD" and last_tc != "HOLD":
            # Consensus cleared — reset so next BUY/SELL is "new"
            triggered_consensus[ticker] = "HOLD"
        elif action in ("BUY", "SELL") and action != last_tc:
            # Direction flip (BUY↔SELL) — update baseline silently,
            # let sustained flip trigger (#2) handle it
            triggered_consensus[ticker] = action
    state["triggered_consensus"] = triggered_consensus

    # 2. Signal flip — triggers when the new action has been seen for
    #    SUSTAINED_CHECKS consecutive cycles OR for SUSTAINED_DURATION_S
    #    wall-clock seconds, whichever comes first. The duration gate was
    #    added 2026-04-09 so the trigger still fires within ~1 cycle at
    #    long cadences (e.g. 600s); at the historical 60s cadence the count
    #    gate still dominates and behavior is unchanged.
    prev_triggered = prev.get("signals", {})
    _flip_now_ts = time.time()
    for ticker, sig in signals.items():
        current_action = sig["action"]
        prev_count = sustained.get(ticker, {})
        if prev_count.get("action") == current_action:
            sustained[ticker] = {
                "action": current_action,
                "count": prev_count["count"] + 1,
                "started_ts": prev_count.get("started_ts", _flip_now_ts),
            }
        else:
            sustained[ticker] = {
                "action": current_action,
                "count": 1,
                "started_ts": _flip_now_ts,
            }

        triggered_action = prev_triggered.get(ticker, {}).get("action")
        if triggered_action and current_action != triggered_action:
            count_ok = sustained[ticker]["count"] >= SUSTAINED_CHECKS
            duration_ok = (
                (_flip_now_ts - sustained[ticker]["started_ts"]) >= SUSTAINED_DURATION_S
            )
            if count_ok or duration_ok:
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

    # 5. Sentiment reversal — same OR-debounce as section 2: SUSTAINED_CHECKS
    #    consecutive cycles OR SUSTAINED_DURATION_S wall-clock seconds.
    sustained_sent = state.get("sustained_sentiment", {})
    stable_sent = state.get("stable_sentiment", {})
    _sent_now_ts = time.time()
    for ticker, sent in sentiments.items():
        prev_sc = sustained_sent.get(ticker, {})
        if prev_sc.get("value") == sent:
            sustained_sent[ticker] = {
                "value": sent,
                "count": prev_sc.get("count", 0) + 1,
                "started_ts": prev_sc.get("started_ts", _sent_now_ts),
            }
        else:
            sustained_sent[ticker] = {
                "value": sent,
                "count": 1,
                "started_ts": _sent_now_ts,
            }

        _sent_count_ok = sustained_sent[ticker]["count"] >= SUSTAINED_CHECKS
        _sent_duration_ok = (
            (_sent_now_ts - sustained_sent[ticker]["started_ts"]) >= SUSTAINED_DURATION_S
        )
        if _sent_count_ok or _sent_duration_ok:
            last_stable = stable_sent.get(ticker)
            if (
                last_stable
                and last_stable != sent
                and sent != "neutral"
                and last_stable != "neutral"
            ):
                reasons.append(
                    f"{ticker} sentiment {last_stable}->{sent} (sustained)"
                )
            stable_sent[ticker] = sent
    state["sustained_sentiment"] = sustained_sent
    state["stable_sentiment"] = stable_sent

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
        # C4/NEW-2: only update last_trigger_date when a real trigger fires, so that
        # classify_tier() can correctly detect the first real trigger of the day.
        state["last_trigger_date"] = _today_str()

    # Track today_date for other purposes
    state["today_date"] = _today_str()

    state["sustained_counts"] = sustained
    _save_state(state)

    return triggered, reasons


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

# Full review interval: 4h during market hours, 4h off-hours (T1 only)
_FULL_REVIEW_MARKET_HOURS = 4
_FULL_REVIEW_OFF_HOURS = 4  # Off-hours caps at T1, not T3


def classify_tier(reasons, state=None):
    """Classify trigger reasons into invocation tier (1=quick, 2=signal, 3=full).

    Tier 3 (Full Review): periodic review, F&G extreme, first of day.
    Tier 2 (Signal Analysis): new consensus, price moves, post-trade, signal flips.
    Tier 1 (Quick Check): sentiment noise, repeated triggers.

    M10/NEW-4: pass state=<dict> to avoid a redundant disk read when the caller
    already has the trigger state loaded. Falls back to loading from file if None.
    """
    if state is None:
        state = _load_state()

    # Tier 3: periodic full review
    last_full = state.get("last_full_review_time", 0)
    hours_since = (time.time() - last_full) / 3600

    now_utc = datetime.now(UTC)
    from portfolio.market_timing import _eu_market_open_hour_utc, _market_close_hour_utc
    close_hour = _market_close_hour_utc(now_utc)
    eu_open = _eu_market_open_hour_utc(now_utc)
    market_open = now_utc.weekday() < 5 and eu_open <= now_utc.hour < close_hour

    # C4/NEW-2: first-of-day T3 check must precede the off-hours periodic cap.
    # An off-hours trigger 4+ hours after the last full review would otherwise
    # return T1 early (line below), skipping the first-of-day T3 entirely.
    if state.get("last_trigger_date") != _today_str():
        return 3  # first real trigger of the day

    if any("F&G crossed" in r for r in reasons):
        return 3

    if market_open and hours_since >= _FULL_REVIEW_MARKET_HOURS:
        return 3
    if not market_open and hours_since >= _FULL_REVIEW_OFF_HOURS:
        return 1  # T1 quick check only — save T3 budget for market hours

    # Tier 2: new actionable signals
    tier2_patterns = ["consensus", "moved", "post-trade", "flipped"]
    if any(p in r for r in reasons for p in tier2_patterns):
        return 2

    # Tier 1: cooldowns, sentiment noise, repeated triggers
    return 1


def update_tier_state(tier, state=None):
    """Update trigger state after a tier classification.

    Called by the main loop after classify_tier() to persist tier-specific state.
    M10/NEW-4: accepts an optional state dict to avoid re-reading trigger_state.json.
    """
    if state is None:
        state = _load_state()
    if tier == 3:
        state["last_full_review_time"] = time.time()
    _save_state(state)
