"""Message throttle — limits analysis Telegram messages to max 1 per cooldown period.

Trade messages always bypass the throttle.
Analysis messages are queued and consolidated.
"""

import json
import logging
import time
from pathlib import Path

from portfolio.file_utils import atomic_write_json, load_json

logger = logging.getLogger("portfolio.message_throttle")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PENDING_FILE = DATA_DIR / "pending_telegram.json"

# Default cooldown: 3 hours
DEFAULT_COOLDOWN_SECONDS = 10800


def should_send_analysis(config=None):
    """Check if enough time has elapsed since the last analysis message.

    Args:
        config: Full app config dict (optional). Reads cooldown from
                config.notification.analysis_cooldown_seconds.

    Returns:
        True if an analysis message can be sent now.
    """
    cooldown = DEFAULT_COOLDOWN_SECONDS
    if config:
        cooldown = config.get("notification", {}).get(
            "analysis_cooldown_seconds", DEFAULT_COOLDOWN_SECONDS
        )

    state = load_json(PENDING_FILE, default={})
    last_sent = state.get("last_analysis_sent", 0)
    return (time.time() - last_sent) >= cooldown


def queue_analysis(text, config=None):
    """Add an analysis message to the pending queue.

    If cooldown has elapsed, the message is sent immediately.
    Otherwise, it replaces any previously queued message (latest wins).

    Args:
        text: Message text.
        config: Full app config dict.

    Returns:
        "sent" if sent immediately, "queued" if queued for later.
    """
    if should_send_analysis(config):
        return _send_now(text, config)

    # Queue it (latest message replaces previous)
    state = load_json(PENDING_FILE, default={})
    state["pending_text"] = text
    state["pending_ts"] = time.time()
    atomic_write_json(PENDING_FILE, state)
    logger.debug("Analysis message queued (cooldown active)")
    return "queued"


def flush_and_send(config):
    """Send any pending queued message if cooldown has elapsed.

    Called from the main loop each cycle.

    Args:
        config: Full app config dict.

    Returns:
        True if a message was sent, False otherwise.
    """
    if not should_send_analysis(config):
        return False

    state = load_json(PENDING_FILE, default={})
    text = state.get("pending_text")
    if not text:
        return False

    _send_now(text, config)
    return True


def _send_now(text, config):
    """Send the message and update the last-sent timestamp.

    Args:
        text: Message text.
        config: Full app config dict.

    Returns:
        "sent"
    """
    from portfolio.message_store import send_or_store

    if config:
        send_or_store(text, config, category="analysis")

    state = load_json(PENDING_FILE, default={})
    state["last_analysis_sent"] = time.time()
    state.pop("pending_text", None)
    state.pop("pending_ts", None)
    atomic_write_json(PENDING_FILE, state)
    logger.info("Analysis message sent (throttle reset)")
    return "sent"


def mark_trade_sent():
    """Record that a trade message was sent (bypasses throttle).

    Trades always send immediately and don't affect the analysis cooldown.
    This is a no-op — trades go through message_store directly.
    """
    pass
