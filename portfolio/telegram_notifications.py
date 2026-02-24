"""Telegram notification utilities — send messages, alerts, escape markdown."""

import json
import logging
import os
import re

from portfolio.http_retry import fetch_with_retry
from portfolio.tickers import SYMBOLS

logger = logging.getLogger("portfolio.telegram")

_MD_V1_SPECIAL = re.compile(r'([_*`\[\]])')

from pathlib import Path
BOLD_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "portfolio_state_bold.json"
_COOLDOWN_PREFIXES = ("cooldown", "crypto check-in", "startup")


def escape_markdown_v1(text):
    """Escape special Markdown v1 characters in dynamic content to prevent parse failures.

    Use this on user-facing dynamic strings (ticker names, error messages, reason text)
    that are inserted into Markdown-formatted Telegram messages. Do NOT apply to the
    entire message — it would break intentional formatting like *bold* and _italic_.
    """
    return _MD_V1_SPECIAL.sub(r'\\\1', str(text))


def send_telegram(msg, config):
    if os.environ.get("NO_TELEGRAM"):
        logger.info("[NO_TELEGRAM] Skipping send")
        return True
    # Layer 1 messages disabled — only Layer 2 (Claude Code) sends Telegram
    # via direct requests.post. To re-enable, set telegram.layer1_messages: true.
    if not config.get("telegram", {}).get("layer1_messages", False):
        logger.debug("[layer1_messages=false] Skipping Layer 1 send")
        return True
    token = config["telegram"]["token"]
    chat_id = config["telegram"]["chat_id"]
    r = fetch_with_retry(
        f"https://api.telegram.org/bot{token}/sendMessage",
        method="POST",
        json_body={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
        timeout=30,
    )
    if r is None:
        return False
    if r.ok:
        return True
    # Markdown parse failure (HTTP 400) — retry without parse_mode so the message
    # still arrives (unformatted) rather than being silently lost.
    if r.status_code == 400:
        err_desc = ""
        try:
            err_desc = r.json().get("description", "")
        except Exception:
            pass
        if "parse" in err_desc.lower() or "markdown" in err_desc.lower() or "entity" in err_desc.lower():
            logger.warning("Telegram Markdown parse failed (%s), resending without formatting", err_desc)
            r2 = fetch_with_retry(
                f"https://api.telegram.org/bot{token}/sendMessage",
                method="POST",
                json_body={"chat_id": chat_id, "text": msg},
                timeout=30,
            )
            return r2 is not None and r2.ok
    return False


def _maybe_send_alert(config, signals, prices_usd, fx_rate, state, reasons, tf_data):
    from portfolio.portfolio_mgr import portfolio_value

    significant = [r for r in reasons if not r.startswith(_COOLDOWN_PREFIXES)]
    if not significant:
        return
    headline = escape_markdown_v1(significant[0])
    lines = [f"*ALERT: {headline}*", ""]
    # Actionable-only: show BUY/SELL tickers, compress HOLDs
    hold_count = 0
    for ticker in SYMBOLS:
        sig = signals.get(ticker)
        if not sig:
            continue
        action = sig["action"]
        if action == "HOLD":
            hold_count += 1
            continue
        price = prices_usd.get(ticker, 0)
        extra = sig.get("extra", {})
        b = extra.get("_buy_count", 0)
        s = extra.get("_sell_count", 0)
        total = extra.get("_total_applicable", 0)
        h = max(0, total - b - s)
        if price >= 1000:
            p_str = f"${price:,.0f}"
        else:
            p_str = f"${price:,.2f}"
        lines.append(f"`{ticker:<7} {p_str:>9}  {action:<4} {b}B/{s}S/{h}H`")
    if hold_count > 0:
        lines.append(f"_+ {hold_count} HOLD_")
    fg_val = ""
    for ticker, sig in signals.items():
        extra = sig.get("extra", {})
        if "fear_greed" in extra:
            fg_class = escape_markdown_v1(extra.get("fear_greed_class", ""))
            fg_val = f"{extra['fear_greed']} ({fg_class})"
            break
    patient_total = portfolio_value(state, prices_usd, fx_rate)
    patient_pnl = (
        (patient_total - state["initial_value_sek"]) / state["initial_value_sek"]
    ) * 100
    lines.append("")
    if fg_val:
        lines.append(f"_F&G: {fg_val}_")
    lines.append(f"_Patient: {patient_total:,.0f} SEK ({patient_pnl:+.1f}%)_")
    if BOLD_STATE_FILE.exists():
        bold = json.loads(BOLD_STATE_FILE.read_text(encoding="utf-8"))
        bold_total = portfolio_value(bold, prices_usd, fx_rate)
        bold_pnl = (
            (bold_total - bold["initial_value_sek"]) / bold["initial_value_sek"]
        ) * 100
        lines.append(f"_Bold: {bold_total:,.0f} SEK ({bold_pnl:+.1f}%)_")
    msg = "\n".join(lines)
    try:
        send_telegram(msg, config)
        logger.info("Alert sent: %s", headline)
    except Exception as e:
        logger.warning("alert send failed: %s", e)
