"""Autonomous decision engine for the main portfolio loop.

Replaces _maybe_send_alert() when layer2.enabled=false. Provides:
- Signal-based ticker classification and prediction
- Journal entries (same format as Claude Layer 2)
- Decision log with full signal data
- Rich Telegram messages (Mode A / Mode B)
- Throttling for routine HOLD messages

No trade execution — decisions are logged as recommendations only.
"""

import json
import logging
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl, load_json, load_jsonl
from portfolio.message_store import send_or_store
from portfolio.portfolio_mgr import portfolio_value, load_bold_state, BOLD_STATE_FILE
from portfolio.telegram_notifications import escape_markdown_v1
from portfolio.tickers import SYMBOLS, CRYPTO_SYMBOLS, METALS_SYMBOLS

logger = logging.getLogger("portfolio.autonomous")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"
DECISIONS_FILE = DATA_DIR / "layer2_decisions.jsonl"
THROTTLE_FILE = DATA_DIR / "autonomous_throttle.json"

_HOLD_COOLDOWN_SECONDS = 1800  # 30 minutes between routine HOLD messages
_TF_ORDER = ["Now", "12h", "2d", "7d", "1mo", "3mo", "6mo"]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def autonomous_decision(config, signals, prices_usd, fx_rate, state,
                        reasons, tf_data, tier, triggered_tickers):
    """Main entry — called from main.py when layer2.enabled=false."""
    try:
        _autonomous_decision_inner(
            config, signals, prices_usd, fx_rate, state,
            reasons, tf_data, tier, triggered_tickers,
        )
    except Exception:
        logger.exception("autonomous_decision failed")


def _autonomous_decision_inner(config, signals, prices_usd, fx_rate, state,
                               reasons, tf_data, tier, triggered_tickers):
    now = datetime.now(timezone.utc)

    # Load bold state
    bold_state = _load_bold_state_safe()

    # Previous journal for reflection
    prev_entries = load_jsonl(JOURNAL_FILE, limit=5)
    prev_entry = prev_entries[-1] if prev_entries else None

    # Core analysis
    reflection = _build_reflection(prev_entry, prices_usd)
    regime = _detect_regime(signals)
    actionable, top_hold, hold_count, sell_count = _classify_tickers(
        signals, state, bold_state, tier, triggered_tickers,
    )

    # Per-ticker predictions
    predictions = {}
    for ticker, sig in actionable.items():
        tf_entries = tf_data.get(ticker, [])
        predictions[ticker] = _ticker_prediction(ticker, sig, tf_entries)

    # Decisions (always HOLD — no auto-trading)
    patient_reasoning = _strategy_reasoning(predictions, "patient")
    bold_reasoning = _strategy_reasoning(predictions, "bold")
    decisions = {
        "patient": {"action": "HOLD", "reasoning": patient_reasoning},
        "bold": {"action": "HOLD", "reasoning": bold_reasoning},
    }

    # Build ticker entries for journal
    ticker_entries = {}
    for ticker, pred in predictions.items():
        if pred["outlook"] != "neutral":
            ticker_entries[ticker] = {
                "outlook": pred["outlook"],
                "thesis": pred["thesis"],
                "conviction": pred["conviction"],
                "levels": pred.get("levels", []),
            }

    # Watchlist
    watchlist = _build_watchlist(predictions, reasons)

    # Write journal
    journal_entry = {
        "ts": now.isoformat(),
        "source": "autonomous",
        "trigger": "; ".join(reasons) if reasons else "unknown",
        "regime": regime,
        "reflection": reflection,
        "continues": prev_entry["ts"] if prev_entry else None,
        "decisions": decisions,
        "tickers": ticker_entries,
        "watchlist": watchlist,
        "prices": {t: prices_usd.get(t) for t in signals if prices_usd.get(t) is not None},
    }
    atomic_append_jsonl(JOURNAL_FILE, journal_entry)

    # Write decision log
    decision_log = {
        "ts": now.isoformat(),
        "source": "autonomous",
        "tier": tier,
        "trigger": "; ".join(reasons) if reasons else "unknown",
        "regime": regime,
        "predictions": predictions,
        "hold_count": hold_count,
        "sell_count": sell_count,
        "fx_rate": fx_rate,
    }
    atomic_append_jsonl(DECISIONS_FILE, decision_log)

    # Telegram
    if _should_send(predictions, reasons, tier):
        msg = _build_telegram(
            actionable, hold_count, sell_count, state, bold_state,
            prices_usd, fx_rate, signals, tf_data, predictions,
            config, tier, regime, reflection, reasons,
        )
        try:
            send_or_store(msg, config, category="analysis")
            _update_throttle()
            logger.info("Autonomous message sent (%d chars)", len(msg))
        except Exception:
            logger.exception("Autonomous telegram send failed")
    else:
        logger.info("Autonomous: throttled (routine HOLD)")


# ---------------------------------------------------------------------------
# Ticker classification
# ---------------------------------------------------------------------------

def _classify_tickers(signals, patient_state, bold_state, tier, triggered_tickers):
    """Classify tickers into actionable set based on tier.

    Returns: (actionable_dict, top_hold_list, hold_count, sell_count)
    """
    if not signals:
        return {}, [], 0, 0

    # Held tickers across both portfolios
    held = set()
    for pf in (patient_state, bold_state):
        for t, h in pf.get("holdings", {}).items():
            if h.get("shares", 0) > 0:
                held.add(t)

    actionable = {}
    hold_count = 0
    sell_count = 0

    if tier == 1:
        # T1: only held positions
        for ticker in held:
            if ticker in signals:
                actionable[ticker] = signals[ticker]
        # Count remaining
        for ticker, sig in signals.items():
            if ticker not in actionable:
                if sig["action"] == "SELL":
                    sell_count += 1
                else:
                    hold_count += 1
    elif tier == 2:
        # T2: triggered + held
        for ticker in triggered_tickers | held:
            if ticker in signals:
                actionable[ticker] = signals[ticker]
        for ticker, sig in signals.items():
            if ticker not in actionable:
                if sig["action"] == "SELL":
                    sell_count += 1
                else:
                    hold_count += 1
    else:
        # T3: all BUY/SELL + held
        for ticker, sig in signals.items():
            if sig["action"] in ("BUY", "SELL") or ticker in held:
                actionable[ticker] = sig
            else:
                hold_count += 1
        # Count sells not already in actionable
        for ticker, sig in signals.items():
            if ticker not in actionable and sig["action"] == "SELL":
                sell_count += 1

    # Top hold tickers (for when all are HOLD)
    top_hold = []
    if not actionable:
        scored = []
        for ticker, sig in signals.items():
            extra = sig.get("extra", {})
            b = extra.get("_buy_count", 0)
            s = extra.get("_sell_count", 0)
            scored.append((ticker, b + s, sig))
        scored.sort(key=lambda x: -x[1])
        for ticker, _, sig in scored[:5]:
            top_hold.append(ticker)
            actionable[ticker] = sig
        hold_count = max(0, len(signals) - len(actionable))

    return actionable, top_hold, hold_count, sell_count


# ---------------------------------------------------------------------------
# Per-ticker prediction
# ---------------------------------------------------------------------------

def _ticker_prediction(ticker, sig, tf_entries):
    """Generate prediction from signal data and timeframe alignment."""
    action = sig.get("action", "HOLD")
    extra = sig.get("extra", {})
    ind = sig.get("indicators", {})
    rsi = ind.get("rsi", 50)
    buy_count = extra.get("_buy_count", 0)
    sell_count = extra.get("_sell_count", 0)
    total = extra.get("_total_applicable", 20)
    weighted_conf = extra.get("weighted_confidence", sig.get("confidence", 0.5))

    # Base conviction from signal consensus
    active = buy_count + sell_count
    if active == 0:
        conviction = 0.0
        outlook = "neutral"
        recommendation = "HOLD"
    elif action == "BUY":
        conviction = buy_count / max(active, 1) * 0.7
        outlook = "bullish"
        recommendation = "BUY"
    elif action == "SELL":
        conviction = sell_count / max(active, 1) * 0.7
        outlook = "bearish"
        recommendation = "SELL"
    else:
        conviction = 0.2
        outlook = "neutral"
        recommendation = "HOLD"

    # Timeframe alignment bonus
    tf_buy = 0
    tf_sell = 0
    tf_total = 0
    for _, entry in tf_entries:
        if isinstance(entry, dict) and "action" in entry:
            tf_total += 1
            if entry["action"] == "BUY":
                tf_buy += 1
            elif entry["action"] == "SELL":
                tf_sell += 1

    if tf_total > 0:
        if action == "BUY":
            tf_alignment = tf_buy / tf_total
        elif action == "SELL":
            tf_alignment = tf_sell / tf_total
        else:
            tf_alignment = 0.5
        conviction = conviction * 0.7 + tf_alignment * 0.3

    # RSI adjustment
    if action == "BUY" and rsi > 70:
        conviction *= 0.7  # overbought penalty
    elif action == "BUY" and rsi < 30:
        conviction *= 1.1  # oversold boost
    elif action == "SELL" and rsi < 30:
        conviction *= 0.7  # oversold penalty for sell
    elif action == "SELL" and rsi > 70:
        conviction *= 1.1  # overbought boost for sell

    conviction = max(0.0, min(1.0, conviction))

    # Generate thesis
    parts = []
    if action != "HOLD":
        parts.append(f"{buy_count}B/{sell_count}S")
    if rsi < 30:
        parts.append("RSI oversold")
    elif rsi > 70:
        parts.append("RSI overbought")
    if tf_total > 0 and action == "BUY" and tf_buy >= tf_total * 0.6:
        parts.append(f"{tf_buy}/{tf_total} TFs aligned")
    elif tf_total > 0 and action == "SELL" and tf_sell >= tf_total * 0.6:
        parts.append(f"{tf_sell}/{tf_total} TFs aligned")
    macd = ind.get("macd_hist", 0)
    if abs(macd) > 5:
        parts.append(f"MACD {macd:+.1f}")

    thesis = ", ".join(parts) if parts else "mixed signals"

    return {
        "outlook": outlook,
        "conviction": round(conviction, 3),
        "thesis": thesis,
        "recommendation": recommendation,
        "levels": [],
        "buy_count": buy_count,
        "sell_count": sell_count,
        "rsi": rsi,
    }


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------

def _build_reflection(prev_entry, current_prices):
    """Compare previous thesis with current prices."""
    if not prev_entry:
        return ""
    prev_prices = prev_entry.get("prices", {})
    if not prev_prices:
        return ""
    prev_tickers = prev_entry.get("tickers", {})

    parts = []
    for ticker, info in prev_tickers.items():
        outlook = info.get("outlook", "neutral")
        if outlook == "neutral":
            continue
        prev_p = prev_prices.get(ticker)
        curr_p = current_prices.get(ticker)
        if prev_p and curr_p and prev_p > 0:
            pct = (curr_p - prev_p) / prev_p * 100
            if outlook == "bullish" and pct > 0.5:
                parts.append(f"{ticker} bullish confirmed (+{pct:.1f}%)")
            elif outlook == "bullish" and pct < -0.5:
                parts.append(f"{ticker} bullish wrong ({pct:+.1f}%)")
            elif outlook == "bearish" and pct < -0.5:
                parts.append(f"{ticker} bearish confirmed ({pct:+.1f}%)")
            elif outlook == "bearish" and pct > 0.5:
                parts.append(f"{ticker} bearish wrong (+{pct:.1f}%)")

    return ". ".join(parts[:3]) if parts else ""


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def _detect_regime(signals):
    """Detect dominant market regime from signal extras."""
    regimes = []
    for sig in signals.values():
        r = sig.get("extra", {}).get("regime")
        if r:
            regimes.append(r)
    if not regimes:
        return "range-bound"
    counts = Counter(regimes)
    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Strategy reasoning
# ---------------------------------------------------------------------------

def _strategy_reasoning(predictions, strategy):
    """Build reasoning string for a strategy."""
    buy_tickers = [t for t, p in predictions.items() if p["recommendation"] == "BUY"]
    sell_tickers = [t for t, p in predictions.items() if p["recommendation"] == "SELL"]

    if not buy_tickers and not sell_tickers:
        return "No actionable signals. All positions monitored."

    parts = []
    if buy_tickers:
        for t in buy_tickers[:2]:
            p = predictions[t]
            parts.append(f"Signal: BUY {t} {p['buy_count']}B/{p['sell_count']}S. Manual action recommended.")
    if sell_tickers:
        for t in sell_tickers[:2]:
            p = predictions[t]
            parts.append(f"Signal: SELL {t} {p['buy_count']}B/{p['sell_count']}S. Manual action recommended.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Watchlist
# ---------------------------------------------------------------------------

def _build_watchlist(predictions, reasons):
    """Build watchlist items from predictions."""
    items = []
    for ticker, pred in predictions.items():
        if pred["recommendation"] == "BUY" and pred["conviction"] > 0.4:
            items.append(f"{ticker} BUY setup ({pred['conviction']:.0%} conviction)")
        elif pred["recommendation"] == "SELL" and pred["conviction"] > 0.4:
            items.append(f"{ticker} SELL signal ({pred['conviction']:.0%} conviction)")
    return items[:3]


# ---------------------------------------------------------------------------
# Telegram message builder
# ---------------------------------------------------------------------------

def _format_price(price):
    """Aggressively round price for display."""
    if price >= 10000:
        return f"${price / 1000:.0f}K"
    elif price >= 1000:
        return f"${price:,.0f}"
    elif price >= 100:
        return f"${price:.0f}"
    else:
        return f"${price:.2f}"


def _tf_heatmap(tf_entries):
    """Build 7-char heatmap from timeframe entries (B/S/middle-dot)."""
    heatmap = []
    for _, entry in tf_entries[:7]:
        if isinstance(entry, dict):
            a = entry.get("action", "HOLD")
            if a == "BUY":
                heatmap.append("B")
            elif a == "SELL":
                heatmap.append("S")
            else:
                heatmap.append("\u00b7")  # middle dot
        else:
            heatmap.append("\u00b7")
    # Pad to 7
    while len(heatmap) < 7:
        heatmap.append("\u00b7")
    return "".join(heatmap[:7])


def _build_telegram(actionable, hold_count, sell_count, patient_state, bold_state,
                    prices_usd, fx_rate, signals, tf_data, predictions,
                    config, tier, regime, reflection, reasons):
    """Build Telegram message following CLAUDE.md format."""
    notification_cfg = config.get("notification", {})
    mode = notification_cfg.get("mode", "signals")

    if mode == "probability":
        return _build_telegram_mode_b(
            actionable, hold_count, sell_count, patient_state, bold_state,
            prices_usd, fx_rate, signals, tf_data, predictions,
            config, tier, regime, reflection, reasons,
        )

    return _build_telegram_mode_a(
        actionable, hold_count, sell_count, patient_state, bold_state,
        prices_usd, fx_rate, signals, tf_data, predictions,
        config, tier, regime, reflection, reasons,
    )


def _build_telegram_mode_a(actionable, hold_count, sell_count, patient_state, bold_state,
                           prices_usd, fx_rate, signals, tf_data, predictions,
                           config, tier, regime, reflection, reasons):
    """Mode A: BUY/SELL ticker grid format."""
    lines = []

    # --- First line: Apple Watch glance ---
    has_trade = any(p["recommendation"] in ("BUY", "SELL") for p in predictions.values())
    if has_trade:
        # Find top movers
        top = sorted(
            [(t, p) for t, p in predictions.items() if p["recommendation"] != "HOLD"],
            key=lambda x: -x[1].get("conviction", 0),
        )
        if top:
            t, p = top[0]
            price_str = _format_price(prices_usd.get(t, 0))
            lines.append(f"*AUTO {p['recommendation']} {t}* {price_str}")
        else:
            lines.append("*AUTO HOLD*")
    else:
        # Top movers by vote count
        movers = sorted(
            predictions.items(),
            key=lambda x: x[1].get("buy_count", 0) + x[1].get("sell_count", 0),
            reverse=True,
        )
        mover_parts = []
        for t, p in movers[:2]:
            b = p.get("buy_count", 0)
            s = p.get("sell_count", 0)
            label = f"{b}B" if b > s else f"{s}S"
            mover_parts.append(f"{t} {label}")

        # F&G
        fg_str = ""
        for sig in signals.values():
            fg = sig.get("extra", {}).get("fear_greed")
            if fg is not None:
                fg_str = f" F&G {fg}"
                break

        mover_text = " ".join(mover_parts)
        first = f"*AUTO HOLD* {mover_text}{fg_str}"
        if len(first) > 70:
            first = first[:67] + "..."
        lines.append(first)

    lines.append("")

    # --- Ticker grid ---
    for ticker, sig in actionable.items():
        price = prices_usd.get(ticker, 0)
        p_str = _format_price(price)
        action = sig.get("action", "HOLD")
        extra = sig.get("extra", {})
        b = extra.get("_buy_count", 0)
        s = extra.get("_sell_count", 0)
        total = extra.get("_total_applicable", 20)
        h = max(0, total - b - s)
        tf_entries = tf_data.get(ticker, [])
        heatmap = _tf_heatmap(tf_entries)

        # Truncate ticker to 5 chars for alignment
        t_short = ticker.replace("-USD", "").replace("-", "")[:5]
        lines.append(f"`{t_short:<5} {p_str:>7}  {action:<4} {b}B/{s}S/{h}H {heatmap}`")

    # --- Summary line ---
    summary_parts = []
    if hold_count > 0:
        summary_parts.append(f"+{hold_count} hold")
    if sell_count > 0:
        summary_parts.append(f"{sell_count} sell")
    if summary_parts:
        lines.append(f"_{' \u00b7 '.join(summary_parts)}_")

    # --- Context line ---
    safe_fx = fx_rate if fx_rate > 0 else 1
    p_total = portfolio_value(patient_state, prices_usd, safe_fx)
    p_pnl = ((p_total - patient_state.get("initial_value_sek", 500000))
              / max(patient_state.get("initial_value_sek", 500000), 1)) * 100
    b_total = portfolio_value(bold_state, prices_usd, safe_fx)
    b_pnl = ((b_total - bold_state.get("initial_value_sek", 500000))
              / max(bold_state.get("initial_value_sek", 500000), 1)) * 100

    # Bold holdings summary
    bold_holdings_str = ""
    for t, h in bold_state.get("holdings", {}).items():
        if h.get("shares", 0) > 0:
            bold_holdings_str += f" {t.replace('-USD', '')} {h['shares']:.0f}sh"

    ctx = f"_P:{p_total / 1000:.0f}K({p_pnl:+.0f}%) \u00b7 B:{b_total / 1000:.0f}K({b_pnl:+.0f}%){bold_holdings_str}_"
    lines.append("")
    lines.append(ctx)

    # --- Reasoning ---
    reasoning_parts = []
    buy_tickers = [t for t, p in predictions.items() if p["recommendation"] == "BUY"]
    sell_tickers = [t for t, p in predictions.items() if p["recommendation"] == "SELL"]
    if buy_tickers:
        for t in buy_tickers[:2]:
            p = predictions[t]
            b = p.get("buy_count", 0)
            s = p.get("sell_count", 0)
            reasoning_parts.append(f"{t}: BUY signal {b}B/{s}S — {p.get('thesis', '')}")
    if sell_tickers:
        for t in sell_tickers[:2]:
            p = predictions[t]
            b = p.get("buy_count", 0)
            s = p.get("sell_count", 0)
            reasoning_parts.append(f"{t}: SELL signal {b}B/{s}S — {p.get('thesis', '')}")
    if not reasoning_parts:
        reasoning_parts.append(f"No clean entries. Regime: {regime}.")
    if reflection:
        reasoning_parts.append(f"Ref: {reflection[:100]}")

    lines.append(escape_markdown_v1(" | ".join(reasoning_parts[:3])))

    msg = "\n".join(lines)
    return msg[:4096]


def _build_telegram_mode_b(actionable, hold_count, sell_count, patient_state, bold_state,
                           prices_usd, fx_rate, signals, tf_data, predictions,
                           config, tier, regime, reflection, reasons):
    """Mode B: Probability format for focus instruments."""
    notification_cfg = config.get("notification", {})
    focus_tickers = notification_cfg.get("focus_tickers", ["XAG-USD", "BTC-USD"])
    lines = []

    # Try to load focus_probabilities from compact summary
    compact_file = DATA_DIR / "agent_summary_compact.json"
    focus_probs = {}
    cumulative_gains = {}
    if compact_file.exists():
        try:
            compact = json.loads(compact_file.read_text(encoding="utf-8"))
            focus_probs = compact.get("focus_probabilities", {})
            cumulative_gains = compact.get("cumulative_gains", {})
        except Exception as e:
            logger.debug("Failed to load compact summary for probability mode: %s", e)

    # First line
    focus_parts = []
    for t in focus_tickers[:2]:
        prob = focus_probs.get(t, {})
        p3h = prob.get("3h", {})
        direction = "\u2191" if p3h.get("direction", "up") == "up" else "\u2193"
        pct = p3h.get("probability", 50)
        t_short = t.replace("-USD", "")
        focus_parts.append(f"{t_short} {direction}{pct}% 3h")

    first_line = f"*AUTO PROB* {' \u00b7 '.join(focus_parts)}" if focus_parts else "*AUTO PROB*"
    lines.append(first_line)
    lines.append("")

    # Focus instruments - rich format
    safe_fx = fx_rate if fx_rate > 0 else 1
    for t in focus_tickers:
        price = prices_usd.get(t, 0)
        prob = focus_probs.get(t, {})
        gains = cumulative_gains.get(t, {})
        t_short = t.replace("-USD", "")

        p3h = prob.get("3h", {})
        p1d = prob.get("1d", {})
        p3d = prob.get("3d", {})

        def _prob_str(p_data):
            d = "\u2191" if p_data.get("direction", "up") == "up" else "\u2193"
            return f"{d}{p_data.get('probability', 50)}%"

        lines.append(f"`{t_short}  {_format_price(price)}  {_prob_str(p3h)} 3h  {_prob_str(p1d)} 1d  {_prob_str(p3d)} 3d`")

        # Accuracy + 7d gain
        acc = p1d.get("accuracy", 0)
        samples = p1d.get("samples", 0)
        gain_7d = gains.get("7d", 0)
        lines.append(f"`  acc: {acc:.0f}% 1d ({samples} sam) | 7d: {gain_7d:+.1f}%`")

        # Claude's call
        pred = predictions.get(t)
        if pred:
            lines.append(f"`  Claude: {pred['recommendation']} ({pred['thesis'][:40]})`")

    # Non-focus tickers - compact grid
    for ticker, sig in actionable.items():
        if ticker in focus_tickers:
            continue
        price = prices_usd.get(ticker, 0)
        gains = cumulative_gains.get(ticker, {})
        t_short = ticker.replace("-USD", "").replace("-", "")[:5]
        g7d = gains.get("7d", 0)
        lines.append(f"`{t_short:<5} {_format_price(price):>7} 7d:{g7d:+.1f}%`")

    # Summary
    if hold_count > 0 or sell_count > 0:
        parts = []
        if hold_count > 0:
            parts.append(f"+{hold_count} hold")
        if sell_count > 0:
            parts.append(f"{sell_count} sell")
        lines.append(f"_{' \u00b7 '.join(parts)}_")

    # Context
    p_total = portfolio_value(patient_state, prices_usd, safe_fx)
    b_total = portfolio_value(bold_state, prices_usd, safe_fx)
    lines.append("")
    lines.append(f"_P:{p_total / 1000:.0f}K \u00b7 B:{b_total / 1000:.0f}K_")

    # Reasoning
    reasoning = []
    for t in focus_tickers:
        pred = predictions.get(t)
        if pred and pred["recommendation"] != "HOLD":
            reasoning.append(f"{t.replace('-USD', '')}: {pred['thesis'][:50]}")
    if not reasoning:
        reasoning.append(f"Regime: {regime}. Monitoring.")
    lines.append(escape_markdown_v1(". ".join(reasoning[:2])))

    msg = "\n".join(lines)
    return msg[:4096]


# ---------------------------------------------------------------------------
# Throttle logic
# ---------------------------------------------------------------------------

def _should_send(predictions, reasons, tier):
    """Determine whether to send Telegram for this invocation."""
    # Always send for trades, F&G extremes, T3, post-trade
    has_action = any(p["recommendation"] in ("BUY", "SELL") for p in predictions.values())
    if has_action:
        return True
    if tier >= 3:
        return True
    for r in reasons:
        r_lower = r.lower()
        if "f&g" in r_lower or "fear" in r_lower:
            return True
        if "post-trade" in r_lower:
            return True
        if "consensus" in r_lower and ("buy" in r_lower or "sell" in r_lower):
            return True

    # Routine HOLD: throttle
    data = load_json(THROTTLE_FILE, default={})
    last_send = data.get("last_send")
    if last_send:
        try:
            last_dt = datetime.fromisoformat(last_send)
            age = (datetime.now(timezone.utc) - last_dt).total_seconds()
            if age < _HOLD_COOLDOWN_SECONDS:
                return False
        except (ValueError, TypeError):
            pass
    return True


def _update_throttle():
    """Update throttle timestamp."""
    data = {"last_send": datetime.now(timezone.utc).isoformat()}
    try:
        from portfolio.file_utils import atomic_write_json
        atomic_write_json(THROTTLE_FILE, data)
    except Exception:
        logger.warning("Failed to update throttle file")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_bold_state_safe():
    """Load bold state without crashing."""
    try:
        return load_bold_state()
    except Exception:
        return {
            "cash_sek": 500000,
            "holdings": {},
            "transactions": [],
            "initial_value_sek": 500000,
            "total_fees_sek": 0,
        }
