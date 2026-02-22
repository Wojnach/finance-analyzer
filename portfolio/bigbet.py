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
from portfolio.telegram_notifications import send_telegram as _shared_send_telegram

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_FILE = DATA_DIR / "bigbet_state.json"
TOTAL_CONDITIONS = 6  # RSI, BB, F&G, Volume, MACD, 24h price change


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

    if rsi_now < 25:
        detail = f"RSI {rsi_now:.0f} (oversold) on 15m"
        if rsi_1h is not None and rsi_1h < 25:
            detail += f" + next TF ({rsi_1h:.0f})"
        bull_conditions.append(detail)
    if rsi_now > 80:
        detail = f"RSI {rsi_now:.0f} (overbought) on 15m"
        if rsi_1h is not None and rsi_1h > 80:
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
        if fg <= 15:
            bull_conditions.append(f"F&G: {fg} ({fg_class})")
        if fg >= 85:
            bear_conditions.append(f"F&G: {fg} ({fg_class})")

    # 4. 24h price change (calculated from stored history)
    # This is handled by the caller since it needs state

    # 5. Volume spike
    vol_ratio = extra.get("volume_ratio")
    if (
        vol_ratio is not None
        and isinstance(vol_ratio, (int, float))
        and vol_ratio >= 2.0
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
        if macd_hist > macd_hist_prev and rsi_now < 35:
            bull_conditions.append("MACD turning up while oversold")
        if macd_hist < macd_hist_prev and rsi_now > 70:
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


def check_bigbet(signals, prices_usd, fx_rate, tf_data, config):
    bigbet_cfg = config.get("bigbet", {})
    min_conditions = bigbet_cfg.get("min_conditions", 3)
    cooldown_hours = bigbet_cfg.get("cooldown_hours", 4)

    state = _load_state()
    cooldowns = state.get("cooldowns", {})
    price_history = state.get("price_history", {})
    now = time.time()
    changed = False

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

        # Check BULL alert
        if len(bull_conds) >= min_conditions:
            cd_key = f"{ticker}_BULL"
            last_alert = cooldowns.get(cd_key, 0)
            if now - last_alert > cooldown_hours * 3600:
                probability, l2_reasoning = invoke_layer2_eval(
                    ticker, "BULL", bull_conds, signals, tf_data, prices_usd, config
                )
                msg = _format_alert(
                    ticker, "BULL", bull_conds, prices_usd, fx_rate, extra_info,
                    probability=probability, l2_reasoning=l2_reasoning,
                )
                logger.info("BIG BET ALERT: BULL %s (%d/%d conditions)", ticker, len(bull_conds), TOTAL_CONDITIONS)
                try:
                    _send_telegram(msg, config)
                except Exception as e:
                    logger.warning("Big Bet telegram failed: %s", e)
                cooldowns[cd_key] = now
                changed = True
            else:
                remaining = cooldown_hours * 3600 - (now - last_alert)
                logger.info("Big Bet: BULL %s (%d/%d) — cooldown (%.0fm left)", ticker, len(bull_conds), TOTAL_CONDITIONS, remaining / 60)

        # Check BEAR alert
        if len(bear_conds) >= min_conditions:
            cd_key = f"{ticker}_BEAR"
            last_alert = cooldowns.get(cd_key, 0)
            if now - last_alert > cooldown_hours * 3600:
                probability, l2_reasoning = invoke_layer2_eval(
                    ticker, "BEAR", bear_conds, signals, tf_data, prices_usd, config
                )
                msg = _format_alert(
                    ticker, "BEAR", bear_conds, prices_usd, fx_rate, extra_info,
                    probability=probability, l2_reasoning=l2_reasoning,
                )
                logger.info("BIG BET ALERT: BEAR %s (%d/%d conditions)", ticker, len(bear_conds), TOTAL_CONDITIONS)
                try:
                    _send_telegram(msg, config)
                except Exception as e:
                    logger.warning("Big Bet telegram failed: %s", e)
                cooldowns[cd_key] = now
                changed = True
            else:
                remaining = cooldown_hours * 3600 - (now - last_alert)
                logger.info("Big Bet: BEAR %s (%d/%d) — cooldown (%.0fm left)", ticker, len(bear_conds), TOTAL_CONDITIONS, remaining / 60)

    if changed:
        state["cooldowns"] = cooldowns
        state["price_history"] = price_history
        _save_state(state)


def _send_telegram(msg, config):
    _shared_send_telegram(msg, config)
