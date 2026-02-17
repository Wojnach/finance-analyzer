"""Big Bet Alert — Mean-reversion volatility hunter.

Detects extreme oversold/overbought setups and sends Telegram alerts.
Does NOT trade. User manually trades turbo warrants on Avanza.
"""

import json
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
STATE_FILE = DATA_DIR / "bigbet_state.json"


def _load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"cooldowns": {}, "price_history": {}}


def _save_state(state):
    import os, tempfile

    DATA_DIR.mkdir(exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=DATA_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, str(STATE_FILE))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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
    macd_hist = ind.get("macd_hist", 0)
    macd_hist_prev = ind.get("macd_hist_prev", 0)
    if macd_hist > macd_hist_prev and rsi_now < 35:
        bull_conditions.append(f"MACD turning up while oversold")
    if macd_hist < macd_hist_prev and rsi_now > 70:
        bear_conditions.append(f"MACD turning down while overbought")

    return bull_conditions, bear_conditions, {"fg": fg, "fg_class": fg_class}


def _format_alert(ticker, direction, conditions, prices_usd, fx_rate, extra_info):
    emoji = "\U0001f535" if direction == "BULL" else "\U0001f534"
    price = prices_usd.get(ticker, 0)
    n = len(conditions)
    total = 6

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

    lines.append("")
    if direction == "BULL":
        lines.append(f"_Suggested: BULL warrant, ~10x leverage_")
        lines.append(f"_Expected bounce: 5-15%_")
    else:
        lines.append(f"_Suggested: BEAR warrant if available_")
        lines.append(f"_Expected pullback: 5-10%_")
    lines.append(f"_Hold: hours to 1-2 days max_")

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
                msg = _format_alert(
                    ticker, "BULL", bull_conds, prices_usd, fx_rate, extra_info
                )
                print(
                    f"  BIG BET ALERT: BULL {ticker} ({len(bull_conds)}/{6} conditions)"
                )
                try:
                    _send_telegram(msg, config)
                except Exception as e:
                    print(f"  WARNING: Big Bet telegram failed: {e}")
                cooldowns[cd_key] = now
                changed = True
            else:
                remaining = cooldown_hours * 3600 - (now - last_alert)
                print(
                    f"  Big Bet: BULL {ticker} ({len(bull_conds)}/6) — cooldown ({remaining/60:.0f}m left)"
                )

        # Check BEAR alert
        if len(bear_conds) >= min_conditions:
            cd_key = f"{ticker}_BEAR"
            last_alert = cooldowns.get(cd_key, 0)
            if now - last_alert > cooldown_hours * 3600:
                msg = _format_alert(
                    ticker, "BEAR", bear_conds, prices_usd, fx_rate, extra_info
                )
                print(
                    f"  BIG BET ALERT: BEAR {ticker} ({len(bear_conds)}/{6} conditions)"
                )
                try:
                    _send_telegram(msg, config)
                except Exception as e:
                    print(f"  WARNING: Big Bet telegram failed: {e}")
                cooldowns[cd_key] = now
                changed = True
            else:
                remaining = cooldown_hours * 3600 - (now - last_alert)
                print(
                    f"  Big Bet: BEAR {ticker} ({len(bear_conds)}/6) — cooldown ({remaining/60:.0f}m left)"
                )

    if changed:
        state["cooldowns"] = cooldowns
        state["price_history"] = price_history
        _save_state(state)


def _send_telegram(msg, config):
    token = config["telegram"]["token"]
    chat_id = config["telegram"]["chat_id"]
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
        timeout=30,
    )
    if not r.ok:
        print(f"  Telegram error: {r.status_code} {r.text[:200]}")
