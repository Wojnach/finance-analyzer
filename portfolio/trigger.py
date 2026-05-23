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
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.trigger")


# ---------------------------------------------------------------------------
# claude_budget config gates (Batch A — items 1, 4, 5 of docs/PLAN.md)
# Added 2026-05-15. All defaults are no-ops; behavior unchanged unless
# config.claude_budget overrides are set.
# ---------------------------------------------------------------------------
_CLAUDE_BUDGET_DEFAULTS = {
    "consensus_min_pct": 0,             # item 1: drop low-confidence consensus
    "sustained_checks_low_density": 5,  # item 4: low-density tickers need 5 cycles
    "sustained_density_threshold": 0.4, # item 4: <40% voter density = low-density
    "min_weighted_confidence": 0.0,     # item 5: confidence floor disabled
    "min_atr_multiple": 0.0,            # item 5: ATR floor disabled
}


def _load_claude_budget():
    """Read claude_budget section from config with safe defaults.

    Cheap to call per check_triggers — load_config() is mtime-cached.
    Falls back to defaults on any config load error (worktrees often
    lack the config.json symlink — see memory/reference_worktree_symlinks).
    """
    cfg = {}
    try:
        from portfolio.api_utils import load_config
        cfg = (load_config() or {}).get("claude_budget", {}) or {}
    except Exception:
        pass
    return {k: cfg.get(k, v) for k, v in _CLAUDE_BUDGET_DEFAULTS.items()}


# Reason types exempt from confidence/ATR floor (item 5). These are
# operational triggers that must always fire.
_FLOOR_EXEMPT_REASON_TYPES = {
    "first_of_day", "periodic_review", "F&G_extreme",
    "post_trade",
    # price_move removed 2026-05-15: a 2% move in a 3% ATR ranging tape
    # is low-quality. atr_mult floor (default 1.5) already protects
    # genuine breakouts (2% / 0.5% ATR = 4x passes).
}


def _reason_type(reason: str) -> str:
    """Classify a reason string to a reason_type for the floor exemption."""
    if "post-trade" in reason:
        return "post_trade"
    if "F&G crossed" in reason:
        return "F&G_extreme"
    if "moved" in reason:
        return "price_move"
    if "consensus" in reason:
        return "consensus"
    if "flipped" in reason:
        return "sustained_flip"
    if "sentiment" in reason:
        return "sentiment"
    return "other"

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
SUSTAINED_DURATION_S = 900

# Per-ticker flip cooldown (2026-05-08): after a sustained flip fires a Layer 2
# trigger, suppress further sustained flip triggers for the SAME ticker for
# FLIP_COOLDOWN_S seconds.  Prevents whiplash where volatile tickers (e.g. MSTR)
# produce 3+ sustained flips in under an hour, each invoking Layer 2 for a HOLD.
# Does NOT suppress consensus triggers (section 1), price moves (section 3), or
# F&G crossings (section 4) — only section-2 sustained flips.
FLIP_COOLDOWN_S = 1800  # 30 min

# Ranging regime dampening (2026-04-22): when a ticker's regime is "ranging",
# require a minimum consensus confidence before triggering Layer 2. In ranging
# markets, consensus oscillates between HOLD and weak BUY/SELL, producing 20+
# Layer 2 invocations per day that all return HOLD — wasting compute and token
# budget. Setting this to 0.0 disables dampening without code change.
RANGING_CONSENSUS_MIN_CONFIDENCE = 0.40

# Startup grace period — after a restart, the first loop iteration updates the
# baseline without triggering Layer 2. This prevents spurious T3 full reviews
# every time the loop is restarted for a code update.
_GRACE_PERIOD_KEY = "last_loop_pid"  # stored in trigger_state.json
_startup_grace_active = True  # True until first check_triggers call completes


def _update_sustained(
    state_dict: dict, key: str, value, now_ts: float
) -> tuple[bool, bool]:
    """Update sustained-debounce state for a key and return gate results.

    Shared by signal flip (section 2) and sentiment reversal (section 5).
    Increments count if value unchanged, resets if changed. Returns
    (count_ok, duration_ok) indicating whether either debounce gate passed.

    Duration tracking uses time.monotonic() internally to avoid NTP-jump
    false negatives. On process restart, monotonic origin resets and the
    duration gate conservatively starts fresh (correct behavior — a
    restart already resets the sustained counter).
    """
    mono_now = time.monotonic()
    prev = state_dict.get(key, {})
    if prev.get("value") == value:
        state_dict[key] = {
            "value": value,
            "count": prev["count"] + 1,
            "_mono_start": prev.get("_mono_start", mono_now),
        }
    else:
        state_dict[key] = {
            "value": value,
            "count": 1,
            "_mono_start": mono_now,
        }
    entry = state_dict[key]
    count_ok = entry["count"] >= SUSTAINED_CHECKS
    duration_ok = (mono_now - entry["_mono_start"]) >= SUSTAINED_DURATION_S
    return count_ok, duration_ok


def _today_str():
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _load_state():
    return load_json(STATE_FILE, default={})


def _save_state(state):
    # Prune triggered_consensus entries for tickers not in current signals
    # to prevent unbounded growth when tickers are removed from tracking.
    # P1.2 guard: skip prune when current_tickers is empty (e.g., BUG-178
    # pool timeout, total fetch failure). Without this, an empty cycle wipes
    # all baselines and the next successful cycle re-fires consensus triggers
    # for every ticker — spawning a Layer 2 invocation storm.
    tc = state.get("triggered_consensus", {})
    current_tickers = state.get("_current_tickers")
    if current_tickers is not None and len(current_tickers) > 0:
        removed = {k for k in tc if k not in current_tickers}
        if removed:
            logger.info("trigger: pruning %d stale ticker(s) from baseline: %s", len(removed), ", ".join(sorted(removed)))
        pruned = {k: v for k, v in tc.items() if k in current_tickers}
        state["triggered_consensus"] = pruned
        fc = state.get("flip_cooldowns", {})
        if fc:
            state["flip_cooldowns"] = {k: v for k, v in fc.items() if k in current_tickers}
        sc = state.get("sustained_counts", {})
        if sc:
            state["sustained_counts"] = {k: v for k, v in sc.items() if k in current_tickers}
    elif current_tickers is not None and len(current_tickers) == 0:
        logger.warning("trigger: empty ticker set — skipping baseline prune to prevent invocation storm")
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

    # claude_budget gates (items 1, 4, 5)
    budget = _load_claude_budget()
    # candidate reasons accumulate as (reason, weighted_conf, atr_mult) tuples
    # for the item-5 floor; we append to `reasons` immediately for exempt
    # reason types and defer floor-eligible ones until after collection.
    _floor_candidates: list[tuple[str, float, float]] = []

    # 0. Trade reset — if Layer 2 made a trade, trigger reassessment
    if _check_recent_trade(state):
        state["last_trigger_time"] = 0
        reasons.append("post-trade reassessment")

    # 1. Signal consensus — trigger ONLY when a ticker first reaches BUY/SELL
    #    from HOLD. BUY↔SELL direction flips are handled by the sustained flip
    #    trigger (#2). Uses persistent triggered_consensus that is NOT wiped
    #    when unrelated triggers (sentiment, etc.) fire.
    #
    #    Ranging regime dampening (2026-04-22): in ranging regime, low-confidence
    #    consensus crossings are noise — require RANGING_CONSENSUS_MIN_CONFIDENCE
    #    to actually fire the trigger. Prevents 20+ HOLD invocations per day.
    triggered_consensus = state.get("triggered_consensus", {})
    for ticker, sig in signals.items():
        action = sig["action"]
        last_tc = triggered_consensus.get(ticker, "HOLD")
        if action in ("BUY", "SELL") and last_tc == "HOLD":
            conf = sig.get("confidence", 0)
            # Ranging regime dampening: skip low-confidence consensus triggers
            ticker_regime = (sig.get("extra") or {}).get("_regime", "unknown")
            if (
                ticker_regime == "ranging"
                and RANGING_CONSENSUS_MIN_CONFIDENCE > 0
                and conf < RANGING_CONSENSUS_MIN_CONFIDENCE
            ):
                logger.info(
                    "Ranging dampening: %s consensus %s (%.0f%%) suppressed "
                    "(min %.0f%%)",
                    ticker, action, conf * 100,
                    RANGING_CONSENSUS_MIN_CONFIDENCE * 100,
                )
                # Still update baseline so we don't re-trigger next cycle
                triggered_consensus[ticker] = action
                continue
            # Item 1 (2026-05-15): claude_budget consensus floor.
            # Drop sub-min_pct consensus crossings entirely. Still update
            # the baseline so we don't re-trigger next cycle when conf
            # eventually crosses the floor (matches existing dampening
            # behavior at line 239).
            min_pct = budget["consensus_min_pct"]
            if min_pct > 0 and (conf * 100) < min_pct:
                logger.info(
                    "claude_budget: %s consensus %s (%.0f%%) suppressed "
                    "(min %d%%)",
                    ticker, action, conf * 100, min_pct,
                )
                triggered_consensus[ticker] = action
                continue
            # New consensus from HOLD — trigger (deferred to floor gate)
            _reason_str = f"{ticker} consensus {action} ({conf:.0%})"
            _floor_candidates.append((_reason_str, conf, 0.0))
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
    flip_cooldowns = state.get("flip_cooldowns", {})
    _flip_now_ts = time.time()
    for ticker, sig in signals.items():
        current_action = sig["action"]
        count_ok, duration_ok = _update_sustained(
            sustained, ticker, current_action, _flip_now_ts,
        )

        # Item 4 (2026-05-15): for low-density tickers (few active voters
        # vs total applicable signals), require MORE consecutive ticks
        # before firing the sustained flip. High-density flips keep the
        # default SUSTAINED_CHECKS. The duration_ok path is unchanged.
        extra = sig.get("extra") or {}
        active_voters = extra.get("_voters_post_filter", extra.get("_voters", 0)) or 0
        total_applicable = extra.get("_total_applicable", 0) or 0
        density = (active_voters / total_applicable) if total_applicable > 0 else 1.0
        density_threshold = budget["sustained_density_threshold"]
        low_density_required = budget["sustained_checks_low_density"]
        if (
            density_threshold > 0
            and density < density_threshold
            and low_density_required > SUSTAINED_CHECKS
        ):
            tick_count = sustained.get(ticker, {}).get("count", 0)
            count_ok = tick_count >= low_density_required

        triggered_action = prev_triggered.get(ticker, {}).get("action")
        if triggered_action and current_action != triggered_action and (count_ok or duration_ok):
            last_flip_ts = flip_cooldowns.get(ticker, 0)
            elapsed = _flip_now_ts - last_flip_ts
            if elapsed < 0:
                flip_cooldowns[ticker] = _flip_now_ts
                elapsed = 0
                logger.warning("Clock skew detected for %s flip cooldown, resetting", ticker)
            if elapsed < FLIP_COOLDOWN_S:
                logger.info(
                    "Flip cooldown: %s %s->%s suppressed (%.0fs remaining)",
                    ticker, triggered_action, current_action,
                    FLIP_COOLDOWN_S - elapsed,
                )
                continue
            flip_cooldowns[ticker] = _flip_now_ts
            # Item 5: route through floor gate with weighted_conf + atr_mult.
            _flip_reason = (
                f"{ticker} flipped {triggered_action}->{current_action} (sustained)"
            )
            _flip_conf = sig.get("confidence", 0) or 0
            _flip_atr_pct = extra.get("atr_pct") or sig.get("atr_pct") or 0
            prev_price = prev.get("prices", {}).get(ticker)
            cur_price = prices_usd.get(ticker)
            _flip_atr_mult = 0.0
            if prev_price and cur_price and _flip_atr_pct and _flip_atr_pct > 0:
                pct_move = abs(cur_price - prev_price) / prev_price * 100
                _flip_atr_mult = pct_move / _flip_atr_pct
            _floor_candidates.append((_flip_reason, _flip_conf, _flip_atr_mult))
    state["flip_cooldowns"] = flip_cooldowns

    # 3. Price move >2% since last trigger
    prev_prices = prev.get("prices", {})
    for ticker, price in prices_usd.items():
        old_price = prev_prices.get(ticker)
        if old_price and old_price > 0:
            pct = abs(price - old_price) / old_price
            if pct >= PRICE_THRESHOLD:
                direction = "up" if price > old_price else "down"
                _pm_reason = f"{ticker} moved {pct:.1%} {direction}"
                # 2026-05-15: route price_move through floor gate.
                # A 2% move in a 3% ATR ranging tape is low-quality;
                # the atr_mult floor (default 1.5) filters it. Floor
                # is OR-gated with weighted_conf so high-conviction
                # moves still pass.
                _pm_sig = signals.get(ticker, {}) or {}
                _pm_extra = _pm_sig.get("extra", {}) or {}
                _pm_conf = _pm_sig.get("confidence", 0) or 0
                _pm_atr_pct = (
                    _pm_extra.get("atr_pct") or _pm_sig.get("atr_pct") or 0
                )
                _pm_atr_mult = 0.0
                if _pm_atr_pct and _pm_atr_pct > 0:
                    _pm_atr_mult = (pct * 100.0) / _pm_atr_pct
                _floor_candidates.append((_pm_reason, _pm_conf, _pm_atr_mult))

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

    # 5. Sentiment reversal — same OR-debounce as section 2.
    sustained_sent = state.get("sustained_sentiment", {})
    stable_sent = state.get("stable_sentiment", {})
    _sent_now_ts = time.time()
    for ticker, sent in sentiments.items():
        count_ok, duration_ok = _update_sustained(
            sustained_sent, ticker, sent, _sent_now_ts,
        )
        if count_ok or duration_ok:
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

    # Item 5 (2026-05-15): apply weighted-conf + ATR floor to deferred
    # candidates (consensus crossings and sustained flips). Emit only if
    # confidence ≥ min_weighted_confidence OR atr_mult ≥ min_atr_multiple
    # OR reason_type is in the exempt set. Defaults are 0.0 (no-op).
    min_conf = budget["min_weighted_confidence"]
    min_atr = budget["min_atr_multiple"]
    for reason_str, weighted_conf, atr_mult in _floor_candidates:
        rt = _reason_type(reason_str)
        if rt in _FLOOR_EXEMPT_REASON_TYPES:
            reasons.append(reason_str)
            continue
        if min_conf <= 0 and min_atr <= 0:
            reasons.append(reason_str)
            continue
        if weighted_conf >= min_conf or atr_mult >= min_atr:
            reasons.append(reason_str)
        else:
            logger.info(
                "claude_budget floor: suppressed '%s' (conf=%.2f < %.2f, "
                "atr_mult=%.2f < %.2f)",
                reason_str, weighted_conf, min_conf, atr_mult, min_atr,
            )

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

# Option P (2026-04-17): confidence-aware tier downshift.
# When every reason in a T2 trigger is either a low-conviction consensus
# crossing (<TIER_DOWNSHIFT_CONFIDENCE) or a fade flip (*->HOLD sustained),
# downshift T2 -> T1 to save Claude token budget. T3 triggers (first-of-day,
# F&G extreme, periodic full review) are NEVER downshifted. Sustained
# direction flips (BUY<->SELL) and non-consensus triggers (post-trade, price
# move, sentiment) block downshift. Setting this to 0.0 disables downshift
# without code change.
TIER_DOWNSHIFT_CONFIDENCE = 0.40

# Precompiled patterns for downshift eligibility analysis on reason strings
# produced by check_triggers(). Reason shape stays stable across releases;
# if the format ever changes, these miss -> downshift fails open (tier
# stays T2, safe over-invocation rather than under-invocation).
#
# Word boundaries (\b) on "consensus" and "flipped" prevent substring
# collisions — e.g. a hypothetical future reason containing "nonconsensus"
# or "preflipped" would NOT accidentally match and trigger a downshift.
# Current check_triggers has no such reasons, but anchoring is cheap
# insurance against future regressions. Added 2026-04-17 after an
# adversarial self-review surfaced the issue.
_CONSENSUS_CONF_RE = re.compile(r'\bconsensus (?:BUY|SELL) \((\d+)%\)')
_FADE_FLIP_RE = re.compile(r'\bflipped (?:BUY|SELL)->HOLD \(sustained\)')


def _reason_is_downshiftable(reason: str, threshold: float) -> bool:
    """Return True if this reason is low-conviction enough to allow T2->T1.

    A reason qualifies if it is either:
      - A consensus crossing with confidence < threshold, or
      - A fade flip (*->HOLD sustained).

    Any other reason type (direction flip, post-trade, price move, F&G,
    sentiment, startup) returns False and blocks downshift for the whole
    reason list.
    """
    m = _CONSENSUS_CONF_RE.search(reason)
    if m:
        conf_pct = int(m.group(1))
        return conf_pct < threshold * 100
    return bool(_FADE_FLIP_RE.search(reason))


def _should_downshift_to_t1(reasons, threshold: float | None = None) -> bool:
    """Decide whether a T2 tier should be downshifted to T1.

    Returns True only when every reason is either a low-conviction consensus
    crossing or a fade flip — i.e. all reasons are individually downshiftable.
    A single high-conviction or non-consensus reason blocks downshift.

    Empty reason list returns False (no downshift). Called only after
    classify_tier() has already chosen T2 — T1 and T3 are never affected.

    threshold=None (default) looks up TIER_DOWNSHIFT_CONFIDENCE at call time,
    allowing runtime overrides via mock.patch or module-attribute reassignment
    (the module-level constant is the single config knob). Passing an explicit
    float overrides for testing.
    """
    if not reasons:
        return False
    effective = TIER_DOWNSHIFT_CONFIDENCE if threshold is None else threshold
    return all(_reason_is_downshiftable(r, effective) for r in reasons)


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
        # Option P (2026-04-17): downshift T2 -> T1 when every reason is
        # low-conviction (consensus <40% confidence or *->HOLD fade flip).
        # Preserves trigger firing + signal/accuracy data; only cuts Claude
        # analysis depth on signals that reliably return HOLD anyway.
        if _should_downshift_to_t1(reasons):
            return 1
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
