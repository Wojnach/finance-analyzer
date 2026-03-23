"""Central message routing — save all messages to JSONL, send only selected categories to Telegram.

Categories that are ALWAYS sent to Telegram:
  - trade:   simulated BUY/SELL executions (Layer 2)
  - iskbets: intraday entry/exit alerts
  - bigbet:  mean-reversion BIG BET alerts
  - digest:  4-hourly activity report

Categories that are SENT to Telegram:
  - analysis:   HOLD analysis, market commentary (Layer 2 — sole Telegram sender)

Categories that are ALSO SENT to Telegram:
  - invocation:  "Layer 2 Tx invoked" notifications
  - regime:      regime shift alerts
  - error:       loop crash notifications

Categories that are SAVED ONLY (viewable on dashboard / via file):
  - fx_alert:    FX rate staleness warnings
"""

import logging
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl
from portfolio.http_retry import fetch_with_retry

logger = logging.getLogger("portfolio.message_store")

BASE_DIR = Path(__file__).resolve().parent.parent
MESSAGES_FILE = BASE_DIR / "data" / "telegram_messages.jsonl"

_TELEGRAM_MAX_LENGTH = 4096
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_COMMON_MOJIBAKE_REPLACEMENTS = {
    "Â·": "·",
    "â": "—",
    "â€“": "–",
    "â": "'",
    "â": "'",
    'â': '"',
    'â': '"',
    "â": "→",
    "â": "↑",
    "â": "↓",
    "Â": "",
}

# Categories whose messages should be sent to Telegram in addition to being saved.
SEND_CATEGORIES = {"trade", "iskbets", "bigbet", "digest", "daily_digest", "analysis", "invocation", "regime", "error", "elongir", "crypto_report"}


def _repair_common_mojibake(text):
    repaired = text
    for bad, good in _COMMON_MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(bad, good)
    return repaired


def _normalize_message_whitespace(text):
    lines = []
    for raw_line in text.split("\n"):
        if raw_line.startswith("`") and raw_line.endswith("`"):
            lines.append(raw_line.rstrip())
            continue
        line = raw_line.replace("\t", " ")
        line = re.sub(r" {2,}", " ", line).strip()
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def sanitize_message_text(text):
    """Normalize message text before saving/sending.

    Keeps intended Markdown structure while removing common control-byte and
    mojibake artifacts that make Telegram messages unreadable.
    """
    cleaned = str(text or "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _repair_common_mojibake(cleaned)
    cleaned = _CONTROL_CHAR_RE.sub(" ", cleaned)
    return _normalize_message_whitespace(cleaned)


def log_message(text, category="analysis", sent=False):
    """Append a message to the JSONL message log.

    Args:
        text: Message text (may contain Markdown).
        category: Message category (see module docstring for valid values).
        sent: Whether the message was actually sent to Telegram.
    """
    cleaned = sanitize_message_text(text)
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "text": cleaned,
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

    msg = sanitize_message_text(msg)

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
        except Exception as e:
            logger.debug("Failed to parse Telegram error response: %s", e)
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
    cleaned = sanitize_message_text(msg)
    should_send = category in SEND_CATEGORIES

    # Mute gates: skip Telegram send, still log to JSONL
    tg_cfg = config.get("telegram", {})

    # Per-category blocklist: mute specific categories
    muted = set(tg_cfg.get("muted_categories", []))
    if category in muted:
        log_message(cleaned, category=category, sent=False)
        logger.info("Message muted [%s]: %.60s...", category, cleaned.replace("\n", " "))
        return True

    # Global mute gate: skip Telegram send unless category is whitelisted
    if tg_cfg.get("mute_all", False):
        unmuted = set(tg_cfg.get("unmuted_categories", []))
        if category not in unmuted:
            log_message(cleaned, category=category, sent=False)
            logger.info("Message muted [%s]: %.60s...", category, cleaned.replace("\n", " "))
            return True

    if should_send:
        sent_ok = _do_send_telegram(cleaned, config)
        log_message(cleaned, category=category, sent=sent_ok)
        if sent_ok:
            logger.info("Message sent [%s]: %.60s...", category, cleaned.replace("\n", " "))
        else:
            logger.warning("Message send failed [%s]: %.60s...", category, cleaned.replace("\n", " "))
        return sent_ok
    else:
        log_message(cleaned, category=category, sent=False)
        logger.debug("Message stored [%s]: %.60s...", category, cleaned.replace("\n", " "))
        return True
