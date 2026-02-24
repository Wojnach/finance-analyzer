"""Big Bet Alert — Mean-reversion volatility hunter.

Detects extreme oversold/overbought setups and sends Telegram alerts.
Does NOT trade. User manually trades turbo warrants on Avanza.
"""

import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("portfolio.bigbet")

from portfolio.file_utils import atomic_write_json
from portfolio.message_store import send_or_store

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_FILE = DATA_DIR / "bigbet_state.json"
TOTAL_CONDITIONS = 6  # RSI, BB, F&G, Volume, MACD, 24h price change

# Margin-buffered thresholds — tighter than raw indicator levels to filter
# ephemeral boundary crossings that revert within 1-2 cycles.
RSI_OVERSOLD = 22          # was 25: 3-point buffer
RSI_OVERBOUGHT = 82        # was 80: 2-point buffer
RSI_MACD_OVERSOLD = 33     # was 35: 2-point buffer
RSI_MACD_OVERBOUGHT = 72   # was 70: 2-point buffer
VOL_SPIKE_MIN = 2.5        # was 2.0: 0.5x buffer
FG_EXTREME_FEAR = 12       # was 15: 3-point buffer
FG_EXTREME_GREED = 88      # was 85: 3-point buffer

# Max age for condition streak entries — streaks older than this are stale
MAX_STREAK_AGE_S = 300     # 5 minutes


def _load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"cooldowns": {}, "price_history": {}}


def _save_state(state):
    atomic_write_json(STATE_FILE, state)


EVAL_LOG_FILE = DATA_DIR / "bigbet_gate_log.jsonl"


def _build_eval_prompt(ticker, direction, conditions, signals, tf_data, prices_usd):
    """Build a focused prompt for Claude to evaluate a big bet setup."""
    sig = signals.get(ticker, {})
    extra = sig.get("extra", {})
    ind = sig.get("indicators", {})

    buy_c = extra.get("_buy_count", 0)
    sell_c = extra.get("_sell_count", 0)
    hold_c = extra.get("_total_applicable", 21) - buy_c - sell_c

    rsi = ind.get("rsi", "N/A")
    macd = ind.get("macd_hist", "N/A")
    bb = ind.get("price_vs_bb", "N/A")
    vol = extra.get("volume_ratio", "N/A")
    atr_pct = ind.get("atr_pct", "N/A")

    # Build TF heatmap row
    tf_row = "N/A"
    tf_list = tf_data.get(ticker, [])
    if tf_list:
        labels = []
        actions = []
        for label, td in tf_list:
            labels.append(label)
            a = td.get("action")
            actions.append("B" if a == "BUY" else "S" if a == "SELL" else "H")
        tf_row = "/".join(labels) + ": " + " ".join(actions)

    fg = extra.get("fear_greed", "N/A")

    # Macro context from agent_summary
    dxy = "N/A"
    dxy_trend = ""
    yield_10y = "N/A"
    fomc_days = "N/A"
    try:
        summary_file = DATA_DIR / "agent_summary.json"
        if summary_file.exists():
            summary = json.loads(summary_file.read_text(encoding="utf-8"))
            macro = summary.get("macro", {})
            dxy_info = macro.get("dxy", {})
            dxy = dxy_info.get("value", "N/A")
            dxy_trend = dxy_info.get("change_5d_pct", "")
            yields = macro.get("treasury", {})
            yield_10y = yields.get("10y", "N/A")
            fed_info = macro.get("fed", {})
            fomc_days = fed_info.get("days_until", "N/A")
    except Exception:
        pass

    cond_str = "\n".join(f"- {c}" for c in conditions)

    return (
        f"You are evaluating a BIG BET alert for {ticker} ({direction}).\n\n"
        f"This is a mean-reversion bounce/pullback trade using 5x warrants.\n"
        f"Hold time: 3-5 hours. The user trades BULL warrants for bounces, "
        f"BEAR warrants for pullbacks.\n\n"
        f"{len(conditions)}/{TOTAL_CONDITIONS} conditions triggered:\n"
        f"{cond_str}\n\n"
        f"Signals: {buy_c}B/{sell_c}S/{hold_c}H\n"
        f"RSI {rsi} | MACD {macd} | BB {bb} | Volume {vol}x\n"
        f"ATR: {atr_pct}%\n\n"
        f"Timeframes ({tf_row})\n\n"
        f"F&G: {fg} | DXY: {dxy} ({dxy_trend}) | 10Y: {yield_10y}% | FOMC: {fomc_days}d\n\n"
        f"Respond EXACTLY:\n"
        f"PROBABILITY: X/10\n"
        f"REASONING: 1-2 sentences on why this is or isn't a good setup.\n\n"
        f"Consider: Is this a genuine capitulation/euphoria or just noise? "
        f"Are the TFs aligned for a bounce/pullback? Is volume confirming? "
        f"Any macro headwinds? Rate 1-3 as poor, 4-6 as marginal, 7-8 as good, "
        f"9-10 as excellent."
    )


def _parse_eval_response(output):
    """Parse PROBABILITY: X/10 and REASONING: ... from eval output.

    Returns (probability: int|None, reasoning: str).
    Defaults to (None, "") on parse failure.
    """
    probability = None
    reasoning = ""

    for line in output.strip().splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("PROBABILITY:"):
            val = line.split(":", 1)[1].strip()
            # Extract number from "X/10" or just "X"
            num_str = val.split("/")[0].strip()
            try:
                probability = int(num_str)
                probability = max(1, min(10, probability))
            except (ValueError, IndexError):
                pass
        elif upper.startswith("REASONING:"):
            reasoning = line.split(":", 1)[1].strip()

    return probability, reasoning


def invoke_layer2_eval(ticker, direction, conditions, signals, tf_data, prices_usd, config):
    """Invoke Claude CLI to evaluate a big bet setup.

    Returns (probability: int|None, reasoning: str).
    Never blocks — returns (None, "") on any failure.
    """
    import os

    if os.environ.get("NO_TELEGRAM"):
        return None, ""

    prompt = _build_eval_prompt(ticker, direction, conditions, signals, tf_data, prices_usd)

    t0 = time.time()
    probability = None
    reasoning = ""

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--max-turns", "1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        elapsed = time.time() - t0
        output = result.stdout.strip()

        if result.returncode == 0 and output:
            probability, reasoning = _parse_eval_response(output)
            logger.info("BIG BET L2: %s %s — %s/10 (%.1fs)", ticker, direction, probability, elapsed)
        else:
            logger.warning("BIG BET L2: claude returned code %s", result.returncode)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        logger.warning("BIG BET L2: timeout after %.1fs", elapsed)
    except FileNotFoundError:
        elapsed = time.time() - t0
        logger.warning("BIG BET L2: claude not found in PATH")
    except Exception as e:
        elapsed = time.time() - t0
        logger.warning("BIG BET L2: error — %s", e)

    # Log evaluation
    try:
        EVAL_LOG_FILE.parent.mkdir(exist_ok=True)
        with open(EVAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "ticker": ticker,
                        "direction": direction,
                        "probability": probability,
                        "reasoning": reasoning,
                        "elapsed_s": round(time.time() - t0, 2),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass

    return probability, reasoning


def _evaluate_conditions(ticker, signals, prices_usd, tf_data):
    sig = signals.get(ticker)
    if not sig:
        return [], [], {}

    ind = sig["indicators"]
    extra = sig.get("extra", {})
    price = prices_usd.get(ticker, 0)

    bull_conditions = []
    bear_conditions = []

    # 1. RSI extremes — check "Now" (15m) and "12h" timeframe (1h candles)
    rsi_now = ind.get("rsi", 50)
    rsi_1h = None
    for label, entry in tf_data.get(ticker, []):
        if label == "12h" and "indicators" in entry:
            rsi_1h = entry["indicators"].get("rsi", 50)
            break

    if rsi_now < RSI_OVERSOLD:
        detail = f"RSI {rsi_now:.0f} (oversold) on 15m"
        if rsi_1h is not None and rsi_1h < RSI_OVERSOLD:
            detail += f" + next TF ({rsi_1h:.0f})"
        bull_conditions.append(detail)
    if rsi_now > RSI_OVERBOUGHT:
        detail = f"RSI {rsi_now:.0f} (overbought) on 15m"
        if rsi_1h is not None and rsi_1h > RSI_OVERBOUGHT:
            detail += f" + next TF ({rsi_1h:.0f})"
        bear_conditions.append(detail)

    # 2. Bollinger Bands — price vs band on multiple timeframes
    bb_below_count = 0
    bb_above_count = 0
    bb_below_tfs = []
    bb_above_tfs = []
    for label, entry in tf_data.get(ticker, []):
        if "indicators" not in entry:
            continue
        ei = entry["indicators"]
        pos = ei.get("price_vs_bb", "inside")
        if pos == "below_lower":
            bb_below_count += 1
            bb_below_tfs.append(label)
        elif pos == "above_upper":
            bb_above_count += 1
            bb_above_tfs.append(label)

    if bb_below_count >= 2:
        bull_conditions.append(f"Below lower BB on {', '.join(bb_below_tfs)}")
    if bb_above_count >= 2:
        bear_conditions.append(f"Above upper BB on {', '.join(bb_above_tfs)}")

    # 3. Fear & Greed
    fg = extra.get("fear_greed")
    fg_class = extra.get("fear_greed_class", "")
    if fg is not None:
        if fg <= FG_EXTREME_FEAR:
            bull_conditions.append(f"F&G: {fg} ({fg_class})")
        if fg >= FG_EXTREME_GREED:
            bear_conditions.append(f"F&G: {fg} ({fg_class})")

    # 4. 24h price change (calculated from stored history)
    # This is handled by the caller since it needs state

    # 5. Volume spike
    vol_ratio = extra.get("volume_ratio")
    if (
        vol_ratio is not None
        and isinstance(vol_ratio, (int, float))
        and vol_ratio >= VOL_SPIKE_MIN
    ):
        # Volume spike — direction depends on price action
        vol_detail = f"Volume {vol_ratio:.1f}x avg"
        vol_action = extra.get("volume_action", "HOLD")
        if vol_action == "SELL":
            bull_conditions.append(f"{vol_detail} (capitulation)")
        elif vol_action == "BUY":
            bear_conditions.append(f"{vol_detail} (euphoria)")
        # HOLD = non-directional spike — not evidence for either direction

    # 6. MACD momentum shift — histogram turning while RSI extreme
    macd_hist = ind.get("macd_hist")
    macd_hist_prev = ind.get("macd_hist_prev")
    if macd_hist is not None and macd_hist_prev is not None:
        if macd_hist > macd_hist_prev and rsi_now < RSI_MACD_OVERSOLD:
            bull_conditions.append("MACD turning up while oversold")
        if macd_hist < macd_hist_prev and rsi_now > RSI_MACD_OVERBOUGHT:
            bear_conditions.append("MACD turning down while overbought")

    return bull_conditions, bear_conditions, {"fg": fg, "fg_class": fg_class}


def _format_alert(ticker, direction, conditions, prices_usd, fx_rate, extra_info,
                   probability=None, l2_reasoning=""):
    emoji = "\U0001f535" if direction == "BULL" else "\U0001f534"
    price = prices_usd.get(ticker, 0)
    n = len(conditions)
    total = TOTAL_CONDITIONS
    now = datetime.now(timezone.utc)

    if n >= 5:
        confidence = "HIGH"
    elif n >= 4:
        confidence = "GOOD"
    else:
        confidence = "MODERATE"

    lines = [f"{emoji} *BIG BET: {direction} {ticker}*", ""]
    lines.append(f"Setup: {n}/{total} conditions met")
    for c in conditions:
        lines.append(f"\u2022 {c}")

    lines.append("")
    price_parts = [f"{ticker} ${price:,.2f}"]
    fg = extra_info.get("fg")
    if fg is not None:
        price_parts.append(f"F&G: {fg}")
    lines.append(" | ".join(price_parts))
    lines.append(f"Confidence: {confidence} ({n}/{total})")
    lines.append(f"_Signal time: {now.strftime('%H:%M')} UTC_")

    lines.append("")
    if direction == "BULL":
        lo = price * 1.05
        hi = price * 1.15
        lines.append(f"_Expected bounce: 5-15% (${lo:,.0f}–${hi:,.0f})_")
    else:
        lo = price * 0.90
        hi = price * 0.95
        lines.append(f"_Expected pullback: 5-10% (${hi:,.0f}–${lo:,.0f})_")
    lines.append(f"_Hold: 3-5h max_")

    if probability is not None:
        lines.append(f"_Claude: {probability}/10 — {l2_reasoning}_")

    return "\n".join(lines)


MAX_ACTIVE_BET_SECONDS = 6 * 3600  # 6 hours — auto-expire stale bets


def _format_window_closed(ticker, direction, price_at_trigger, current_price, elapsed_minutes):
    """Format a 'window closed' notification for an expired active bet."""
    if price_at_trigger and price_at_trigger > 0:
        pct = ((current_price - price_at_trigger) / price_at_trigger) * 100
        price_line = (
            f"Entry price: ${price_at_trigger:,.2f} \u2192 Current: "
            f"${current_price:,.2f} ({pct:+.1f}%)"
        )
    else:
        price_line = f"Current: ${current_price:,.2f}"

    return (
        f"\u26aa *BIG BET CLOSED: {direction} {ticker}*\n\n"
        f"Setup expired after {elapsed_minutes:.0f}m. Conditions no longer met.\n"
        f"{price_line}"
    )


def _resolve_cooldown_minutes(bigbet_cfg):
    """Resolve cooldown in minutes from config, with backwards compatibility.

    Checks ``cooldown_minutes`` first (new key, default 10).
    Falls back to ``cooldown_hours`` (legacy key) converted to minutes.
    """
    if "cooldown_minutes" in bigbet_cfg:
        return bigbet_cfg["cooldown_minutes"]
    if "cooldown_hours" in bigbet_cfg:
        return bigbet_cfg["cooldown_hours"] * 60
    return 10  # default: 10 minutes


def _update_streak(condition_streaks, key, met, now):
    """Increment or reset a condition streak. Returns current count.

    Streaks auto-expire if the last update was > MAX_STREAK_AGE_S ago.
    """
    if met:
        entry = condition_streaks.get(key)
        if entry and (now - entry[1]) <= MAX_STREAK_AGE_S:
            count = entry[0] + 1
        else:
            count = 1  # fresh or stale — start at 1
        condition_streaks[key] = [count, now]
        return count
    # Not met — reset
    condition_streaks.pop(key, None)
    return 0


def check_bigbet(signals, prices_usd, fx_rate, tf_data, config):
    bigbet_cfg = config.get("bigbet", {})
    min_conditions = bigbet_cfg.get("min_conditions", 3)
    min_persistence = bigbet_cfg.get("min_persistence", 2)
    min_probability = bigbet_cfg.get("min_probability", 6)
    cooldown_minutes = _resolve_cooldown_minutes(bigbet_cfg)
    cooldown_seconds = cooldown_minutes * 60

    state = _load_state()
    cooldowns = state.get("cooldowns", {})
    price_history = state.get("price_history", {})
    active_bets = state.get("active_bets", {})
    condition_streaks = state.get("condition_streaks", {})
    now = time.time()
    changed = False

    # --- Phase 1: Check existing active bets for expiry ---
    expired_keys = []
    for bet_key, bet_info in list(active_bets.items()):
        triggered_at = bet_info.get("triggered_at", 0)
        elapsed = now - triggered_at

        # Auto-expire after MAX_ACTIVE_BET_SECONDS (6h)
        if elapsed > MAX_ACTIVE_BET_SECONDS:
            expired_keys.append(bet_key)
            # Parse ticker and direction from key
            parts = bet_key.rsplit("_", 1)
            if len(parts) == 2:
                ticker_k, direction_k = parts
            else:
                continue
            current_price = prices_usd.get(ticker_k, 0)
            elapsed_min = elapsed / 60
            msg = _format_window_closed(
                ticker_k, direction_k,
                bet_info.get("price_at_trigger", 0),
                current_price, elapsed_min,
            )
            logger.info("BIG BET EXPIRED (6h): %s", bet_key)
            try:
                _send_telegram(msg, config)
            except Exception as e:
                logger.warning("Big Bet telegram failed: %s", e)
            changed = True
            continue

        # Re-evaluate conditions to see if setup is still active
        parts = bet_key.rsplit("_", 1)
        if len(parts) != 2:
            expired_keys.append(bet_key)
            continue
        ticker_k, direction_k = parts

        if ticker_k not in signals:
            # Ticker no longer in signals — expire
            expired_keys.append(bet_key)
            current_price = prices_usd.get(ticker_k, 0)
            elapsed_min = elapsed / 60
            msg = _format_window_closed(
                ticker_k, direction_k,
                bet_info.get("price_at_trigger", 0),
                current_price, elapsed_min,
            )
            logger.info("BIG BET CLOSED (ticker gone): %s", bet_key)
            try:
                _send_telegram(msg, config)
            except Exception as e:
                logger.warning("Big Bet telegram failed: %s", e)
            changed = True
            continue

        bull_conds, bear_conds, _ = _evaluate_conditions(
            ticker_k, signals, prices_usd, tf_data
        )
        relevant_conds = bull_conds if direction_k == "BULL" else bear_conds

        if len(relevant_conds) < min_conditions:
            # Conditions no longer met — send window closed
            expired_keys.append(bet_key)
            current_price = prices_usd.get(ticker_k, 0)
            elapsed_min = elapsed / 60
            msg = _format_window_closed(
                ticker_k, direction_k,
                bet_info.get("price_at_trigger", 0),
                current_price, elapsed_min,
            )
            logger.info("BIG BET CLOSED (conditions faded): %s after %.0fm", bet_key, elapsed_min)
            try:
                _send_telegram(msg, config)
            except Exception as e:
                logger.warning("Big Bet telegram failed: %s", e)
            changed = True

    for key in expired_keys:
        active_bets.pop(key, None)

    # --- Phase 2: Evaluate new alerts ---
    for ticker in signals:
        price = prices_usd.get(ticker, 0)
        if price <= 0:
            continue

        bull_conds, bear_conds, extra_info = _evaluate_conditions(
            ticker, signals, prices_usd, tf_data
        )

        # 4. 24h price change — from stored history
        hist = price_history.get(ticker, [])
        if hist:
            # Find price ~24h ago (closest entry to 86400s ago)
            target_time = now - 86400
            closest = min(hist, key=lambda h: abs(h["t"] - target_time))
            if abs(closest["t"] - target_time) < 7200:  # within 2h tolerance
                old_price = closest["p"]
                pct_change = ((price - old_price) / old_price) * 100
                if pct_change <= -5:
                    bull_conds.append(
                        f"Price {pct_change:+.1f}% in 24h (${old_price:,.0f}\u2192${price:,.0f})"
                    )
                if pct_change >= 5:
                    bear_conds.append(
                        f"Price {pct_change:+.1f}% in 24h (${old_price:,.0f}\u2192${price:,.0f})"
                    )

        # Update price history — keep last 48h of entries, sample every ~10min
        if not hist or (now - hist[-1]["t"]) >= 600:
            hist.append({"t": now, "p": price})
            # Prune entries older than 48h
            cutoff = now - 172800
            hist = [h for h in hist if h["t"] >= cutoff]
            price_history[ticker] = hist
            changed = True

        # Check BULL and BEAR alerts with persistence + probability gating
        for direction, conds in [("BULL", bull_conds), ("BEAR", bear_conds)]:
            cd_key = f"{ticker}_{direction}"
            met = len(conds) >= min_conditions
            streak = _update_streak(condition_streaks, cd_key, met, now)

            if not met:
                continue

            if streak < min_persistence:
                logger.info(
                    "Big Bet: %s %s (%d/%d) — persistence %d/%d",
                    direction, ticker, len(conds), TOTAL_CONDITIONS,
                    streak, min_persistence,
                )
                changed = True
                continue

            last_alert = cooldowns.get(cd_key, 0)
            if now - last_alert <= cooldown_seconds:
                remaining = cooldown_seconds - (now - last_alert)
                logger.info(
                    "Big Bet: %s %s (%d/%d) — cooldown (%.0fm left)",
                    direction, ticker, len(conds), TOTAL_CONDITIONS,
                    remaining / 60,
                )
                continue

            probability, l2_reasoning = invoke_layer2_eval(
                ticker, direction, conds, signals, tf_data, prices_usd, config
            )

            # Probability gate — require eval to succeed and meet threshold
            if probability is None or probability < min_probability:
                logger.info(
                    "Big Bet: %s %s (%d/%d) — blocked by probability (%s < %d)",
                    direction, ticker, len(conds), TOTAL_CONDITIONS,
                    probability, min_probability,
                )
                continue

            msg = _format_alert(
                ticker, direction, conds, prices_usd, fx_rate, extra_info,
                probability=probability, l2_reasoning=l2_reasoning,
            )
            logger.info(
                "BIG BET ALERT: %s %s (%d/%d conditions, %d/10 prob)",
                direction, ticker, len(conds), TOTAL_CONDITIONS, probability,
            )
            try:
                _send_telegram(msg, config)
            except Exception as e:
                logger.warning("Big Bet telegram failed: %s", e)
            cooldowns[cd_key] = now
            active_bets[cd_key] = {
                "triggered_at": now,
                "conditions": list(conds),
                "price_at_trigger": price,
            }
            changed = True

    if changed:
        state["cooldowns"] = cooldowns
        state["price_history"] = price_history
        state["active_bets"] = active_bets
        state["condition_streaks"] = condition_streaks
        _save_state(state)


def _send_telegram(msg, config):
    send_or_store(msg, config, category="bigbet")
