"""Telegram Poller — Background thread for ISKBETS commands.

Polls getUpdates every 5 seconds. Parses bought/sold/cancel/status commands
and delegates to iskbets.handle_command().
"""

import json
import logging
import threading
import time

from portfolio.http_retry import fetch_with_retry

logger = logging.getLogger("portfolio.telegram_poller")


class TelegramPoller:
    def __init__(self, config, on_command):
        """
        config: full app config dict (with telegram.token, telegram.chat_id)
        on_command: callback(cmd, args, config) -> response_text or None
        """
        self.token = config["telegram"]["token"]
        self.chat_id = str(config["telegram"]["chat_id"])
        self.config = config
        self.on_command = on_command
        self.offset = 0
        self._startup_time = time.time()
        self._thread = None

    def start(self):
        """Start the poller in a daemon thread."""
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self):
        while True:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._handle_update(update)
            except Exception as e:
                logger.warning("Poller error: %s", e)
            time.sleep(5)

    def _get_updates(self):
        """Fetch new updates from Telegram."""
        params = {"timeout": 3, "allowed_updates": ["message"]}
        if self.offset:
            params["offset"] = self.offset

        r = fetch_with_retry(
            f"https://api.telegram.org/bot{self.token}/getUpdates",
            params=params,
            timeout=10,
        )
        if r is None or not r.ok:
            return []

        data = r.json()
        if not data.get("ok"):
            return []

        return data.get("result", [])

    def _handle_update(self, update):
        """Process a single update."""
        update_id = update.get("update_id", 0)
        self.offset = max(self.offset, update_id + 1)

        msg = update.get("message")
        if not msg:
            return

        # Only process messages from our chat_id
        chat = msg.get("chat", {})
        if str(chat.get("id")) != self.chat_id:
            return

        # Ignore messages older than 60s at startup (prevent stale command processing)
        msg_date = msg.get("date", 0)
        if msg_date < self._startup_time - 60:
            return

        text = (msg.get("text") or "").strip()
        if not text:
            return

        # Parse command
        cmd, args = self._parse_command(text)
        if cmd is None:
            return

        # Delegate to handler
        response = self.on_command(cmd, args, self.config)
        if response:
            self._send_reply(response)

    def _parse_command(self, text):
        """Parse ISKBETS commands from message text.

        Returns (cmd, args) or (None, None) for non-commands.
        Recognized: bought, sold, cancel, status
        """
        text_lower = text.lower().strip()
        parts = text.split(None, 1)
        first_word = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        if first_word in ("bought", "sold", "cancel", "status"):
            return first_word, rest

        return None, None

    def _send_reply(self, text):
        """Send a reply to the user."""
        try:
            r = fetch_with_retry(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                method="POST",
                json_body={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=30,
            )
            if r is not None and not r.ok:
                logger.warning("Poller reply error: %s %s", r.status_code, r.text[:200])
        except Exception as e:
            logger.warning("Poller reply failed: %s", e)
