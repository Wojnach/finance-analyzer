"""Daily digest — morning Telegram with long-term perspective.

Separate from the 4-hour digest (digest.py). Runs at a configurable UTC hour
(default 06:00 = 07:00 CET). Provides focus instrument probabilities, rolling
changes, warrant P&L, and top movers.

Runs in both Mode A and Mode B.
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from portfolio.file_utils import load_json
from portfolio.message_store import send_or_store
from portfolio.portfolio_mgr import load_state, portfolio_value
from portfolio.telegram_notifications import escape_markdown_v1

logger = logging.getLogger("portfolio.daily_digest")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRIGGER_STATE_FILE = DATA_DIR / "trigger_state.json"
AGENT_SUMMARY_FILE = DATA_DIR / "agent_summary.json"
BOLD_STATE_FILE = DATA_DIR / "portfolio_state_bold.json"


def _get_last_daily_digest_time():
    """Get the last daily digest timestamp from trigger state."""
    state = load_json(TRIGGER_STATE_FILE, default={})
    return state.get("last_daily_digest_time", 0)


def _set_last_daily_digest_time(t):
    """Save the last daily digest timestamp."""
    from portfolio.file_utils import atomic_write_json
    state = load_json(TRIGGER_STATE_FILE, default={})
    state["last_daily_digest_time"] = t
    atomic_write_json(TRIGGER_STATE_FILE, state)


def should_send_daily_digest(config):
    """Check if it's time to send the daily digest.

    Sends once per day at the configured hour (default 06:00 UTC = 07:00 CET).

    Args:
        config: Full app config dict.

    Returns:
        True if the daily digest should be sent now.
    """
    notification = config.get("notification", {})
    digest_hour = notification.get("daily_digest_hour_utc", 6)

    now = datetime.now(timezone.utc)
    if now.hour != digest_hour:
        return False

    last = _get_last_daily_digest_time()
    if last:
        # Don't send more than once per 20 hours
        if (time.time() - last) < 72000:
            return False

    return True


def _format_price(price):
    """Format a price for display."""
    if price >= 10000:
        return f"${price / 1000:.0f}K"
    elif price >= 100:
        return f"${price:,.0f}"
    else:
        return f"${price:.1f}"


def build_daily_digest(config):
    """Build the daily digest message.

    Args:
        config: Full app config dict.

    Returns:
        str: Formatted Telegram message.
    """
    summary = load_json(AGENT_SUMMARY_FILE, default={})
    if not summary:
        return None

    fx_rate = summary.get("fx_rate", 10.5)
    signals = summary.get("signals", {})
    prices_usd = {t: s.get("price_usd", 0) for t, s in signals.items() if s.get("price_usd")}

    notification = config.get("notification", {})
    focus_tickers = notification.get("focus_tickers", ["XAG-USD", "BTC-USD"])
    mover_thresholds = notification.get("mover_thresholds", {"3d_pct": 5.0, "7d_pct": 10.0})

    lines = []

    # --- Header with focus instrument rolling changes ---
    cumulative = summary.get("cumulative_gains", {})
    ticker_changes = cumulative.get("ticker_changes", {})

    header_parts = []
    for ticker in focus_tickers:
        changes = ticker_changes.get(ticker, {})
        c7d = changes.get("change_7d")
        if c7d is not None:
            short = ticker.split("-")[0] if "-" in ticker else ticker
            header_parts.append(f"{short} {c7d:+.0f}% 7d")

    if header_parts:
        lines.append(f"*DAILY* · {' · '.join(header_parts)}")
    else:
        lines.append("*DAILY DIGEST*")

    lines.append("")

    # --- Focus instruments section ---
    focus_probs = summary.get("focus_probabilities", {})
    warrant_summary = summary.get("warrant_portfolio", {})
    warrant_positions = warrant_summary.get("positions", {}) if warrant_summary else {}

    if focus_tickers:
        lines.append("*Focus Instruments*")
        for ticker in focus_tickers:
            sig = signals.get(ticker, {})
            price = sig.get("price_usd", 0)
            changes = ticker_changes.get(ticker, {})
            c7d = changes.get("change_7d")
            c7d_str = f"{c7d:+.1f}% 7d" if c7d is not None else ""

            prob = focus_probs.get(ticker, {})
            prob_1d = prob.get("1d", {})
            acc = prob.get("accuracy_1d", 0)
            acc_samples = prob.get("accuracy_samples", 0)

            # Main ticker line
            prob_str = ""
            if prob_1d:
                direction = prob_1d.get("direction", "neutral")
                p = prob_1d.get("probability", 0.5)
                arrow = "↑" if direction == "up" else "↓" if direction == "down" else "→"
                prob_str = f"{arrow}{p*100:.0f}% 1d"

            price_str = _format_price(price) if price else "?"
            short = ticker.split("-")[0] if "-" in ticker else ticker
            parts = [f"`{short}  {price_str}"]
            if c7d_str:
                parts.append(c7d_str)
            if prob_str:
                parts.append(f"{prob_str} (acc {acc*100:.0f}%)")
            lines.append("  ".join(parts) + "`")

            # Warrant line if position held
            for w_key, w_pos in warrant_positions.items():
                if w_pos.get("underlying") == ticker and w_pos.get("pnl"):
                    pnl = w_pos["pnl"]
                    leverage = w_pos.get("leverage", 1)
                    pnl_pct = pnl.get("pnl_pct", 0)
                    pnl_sek = pnl.get("pnl_sek", 0)
                    lines.append(f"`  -> {w_pos.get('name', w_key)} {leverage}x: {pnl_pct:+.0f}% ({pnl_sek:+,.0f} SEK)`")

        lines.append("")

    # --- Movers section (>5% 3d or >10% 7d, excluding focus) ---
    movers = cumulative.get("movers", [])
    non_focus_movers = [m for m in movers if m["ticker"] not in focus_tickers]
    if non_focus_movers:
        lines.append("*Other Movers*")
        for m in non_focus_movers[:5]:
            ticker = m["ticker"]
            price = prices_usd.get(ticker, 0)
            price_str = _format_price(price) if price else "?"
            c7d = m.get("change_7d")
            c3d = m.get("change_3d")
            change_str = f"{c7d:+.1f}% 7d" if c7d else f"{c3d:+.1f}% 3d"
            short = ticker.split("-")[0] if "-" in ticker else ticker
            lines.append(f"`{short:<5} {price_str:<8} {change_str}`")
        lines.append("")

    # --- Portfolio section ---
    lines.append("*Portfolio*")

    state = load_state()
    p_total = portfolio_value(state, prices_usd, fx_rate)
    p_pnl = ((p_total - state["initial_value_sek"]) / state["initial_value_sek"]) * 100
    p_holdings = [t for t, h in state.get("holdings", {}).items() if h.get("shares", 0) > 0]
    p_h_str = " · " + escape_markdown_v1(", ".join(p_holdings)) if p_holdings else ""
    lines.append(f"`Patient: {p_total:,.0f} SEK ({p_pnl:+.1f}%){p_h_str}`")

    if BOLD_STATE_FILE.exists():
        try:
            bold = json.loads(BOLD_STATE_FILE.read_text(encoding="utf-8"))
            b_total = portfolio_value(bold, prices_usd, fx_rate)
            b_pnl = ((b_total - bold["initial_value_sek"]) / bold["initial_value_sek"]) * 100
            b_holdings = [t for t, h in bold.get("holdings", {}).items() if h.get("shares", 0) > 0]
            b_h_str = " · " + escape_markdown_v1(", ".join(b_holdings)) if b_holdings else ""
            lines.append(f"`Bold:    {b_total:,.0f} SEK ({b_pnl:+.1f}%){b_h_str}`")
        except (json.JSONDecodeError, FileNotFoundError):
            pass

    # Warrant total
    if warrant_summary and warrant_summary.get("total_pnl_sek"):
        total_pnl = warrant_summary["total_pnl_sek"]
        lines.append(f"`Warrants: {total_pnl:+,.0f} SEK`")

    lines.append("")

    # --- Accuracy summary for focus instruments ---
    if focus_probs:
        lines.append("*Accuracy (7d window)*")
        acc_parts = []
        for ticker in focus_tickers:
            prob = focus_probs.get(ticker, {})
            acc = prob.get("accuracy_1d", 0)
            samples = prob.get("accuracy_samples", 0)
            short = ticker.split("-")[0] if "-" in ticker else ticker
            acc_parts.append(f"{short}: {acc*100:.0f}% ({samples} sam)")
        lines.append(f"`{' | '.join(acc_parts)}`")

    return "\n".join(lines)


def maybe_send_daily_digest(config):
    """Send daily digest if it's time.

    Args:
        config: Full app config dict.
    """
    if not should_send_daily_digest(config):
        return

    try:
        msg = build_daily_digest(config)
        if msg:
            send_or_store(msg, config, category="daily_digest")
            _set_last_daily_digest_time(time.time())
            logger.info("Daily digest sent")
    except Exception as e:
        logger.warning("Daily digest failed: %s", e)
