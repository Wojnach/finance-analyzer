"""Central message routing — save all messages to JSONL, send only selected categories to Telegram.

Categories that are ALWAYS sent to Telegram:
  - trade:   simulated BUY/SELL executions (Layer 2)
  - iskbets: intraday entry/exit alerts
  - bigbet:  mean-reversion BIG BET alerts
  - digest:  4-hourly activity report

Categories that are SENT to Telegram:
  - analysis:   HOLD analysis, market commentary (Layer 2 — sole Telegram sender)

Categories that are SAVED ONLY (viewable on dashboard / via file):
  - invocation:  "Layer 2 Tx invoked" notifications
  - regime:      regime shift alerts
  - fx_alert:    FX rate staleness warnings
  - error:       loop crash notifications
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl
from portfolio.http_retry import fetch_with_retry

logger = logging.getLogger("portfolio.message_store")

BASE_DIR = Path(__file__).resolve().parent.parent
MESSAGES_FILE = BASE_DIR / "data" / "telegram_messages.jsonl"

_TELEGRAM_MAX_LENGTH = 4096

# Categories whose messages should be sent to Telegram in addition to being saved.
SEND_CATEGORIES = {"trade", "iskbets", "bigbet", "digest", "daily_digest", "analysis"}


def log_message(text, category="analysis", sent=False):
    """Append a message to the JSONL message log.

    Args:
        text: Message text (may contain Markdown).
        category: Message category (see module docstring for valid values).
        sent: Whether the message was actually sent to Telegram.
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "text": text,
        "category": category,
        "sent": sent,
    }
    atomic_append_jsonl(MESSAGES_FILE, entry)


def _do_send_telegram(msg, config):
    """Actually send a message to Telegram. Returns True on success.

    This is the raw API call — no gating by layer1_messages or category.
    Handles truncation, Markdown fallback on parse errors.
    """
    if os.environ.get("NO_TELEGRAM"):
        logger.info("[NO_TELEGRAM] Skipping send")
        return True

    token = config.get("telegram", {}).get("token")
    chat_id = config.get("telegram", {}).get("chat_id")
    if not token or not chat_id:
        logger.warning("Telegram token/chat_id not configured")
        return False

    # Truncate to Telegram's max message length
    if len(msg) > _TELEGRAM_MAX_LENGTH:
        logger.warning(
            "Telegram message truncated from %d to %d chars",
            len(msg), _TELEGRAM_MAX_LENGTH,
        )
        msg = msg[:_TELEGRAM_MAX_LENGTH - 20] + "\n...(truncated)"

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

    # Markdown parse failure (HTTP 400) — retry without parse_mode
    if r.status_code == 400:
        err_desc = ""
        try:
            err_desc = r.json().get("description", "")
        except Exception:
            pass
        if any(kw in err_desc.lower() for kw in ("parse", "markdown", "entity")):
            logger.warning(
                "Telegram Markdown parse failed (%s), resending without formatting",
                err_desc,
            )
            r2 = fetch_with_retry(
                f"https://api.telegram.org/bot{token}/sendMessage",
                method="POST",
                json_body={"chat_id": chat_id, "text": msg},
                timeout=30,
            )
            return r2 is not None and r2.ok
    return False


def send_or_store(msg, config, category="analysis"):
    """Central routing: save message to JSONL, optionally send to Telegram.

    If category is in SEND_CATEGORIES, the message is sent to Telegram AND logged.
    Otherwise it is only logged (saved to JSONL for dashboard / file reading).

    This function bypasses the ``layer1_messages`` config gate — the category
    determines whether to send, not the global flag.

    Args:
        msg: Message text (may contain Markdown).
        config: Full config dict (needs ``telegram.token`` and ``telegram.chat_id``).
        category: Message category string.

    Returns:
        True if message was sent (or save-only succeeded), False on send failure.
    """
    should_send = category in SEND_CATEGORIES

    if should_send:
        sent_ok = _do_send_telegram(msg, config)
        log_message(msg, category=category, sent=sent_ok)
        if sent_ok:
            logger.info("Message sent [%s]: %.60s...", category, msg.replace("\n", " "))
        else:
            logger.warning("Message send failed [%s]: %.60s...", category, msg.replace("\n", " "))
        return sent_ok
    else:
        log_message(msg, category=category, sent=False)
        logger.debug("Message stored [%s]: %.60s...", category, msg.replace("\n", " "))
        return True
