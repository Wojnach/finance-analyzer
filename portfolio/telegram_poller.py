"""Telegram Poller — Background thread for ISKBETS + system commands.

Polls getUpdates every 5 seconds. Parses bought/sold/cancel/status commands
and delegates to iskbets.handle_command(). Also handles /mode command for
switching notification format (signals vs probability).
"""

import json
import logging
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from portfolio.file_utils import atomic_append_jsonl, atomic_write_json, load_json
from portfolio.http_retry import fetch_with_retry

logger = logging.getLogger("portfolio.telegram_poller")

INBOUND_LOG = Path(__file__).resolve().parent.parent / "data" / "telegram_inbound.jsonl"
# 2026-04-28: persisted offset across loop restarts. Without this, every
# `schtasks /run PF-DataLoop` resets self.offset to 0, re-fetches every
# pending getUpdates, and then the stale filter (msg_date < startup-60s)
# silently drops anything the user sent during the restart window. With
# the file present, init reloads the last-acknowledged update_id, and
# _handle_update bypasses the stale filter for ``update_id >
# persisted_offset`` — those are by definition post-restart pending
# updates the user expects to execute (e.g. a ``bought MSTR …``
# confirmation sent while the loop was bouncing).
POLLER_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "telegram_poller_state.json"


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
        # Restore offset from disk so updates acknowledged in a previous
        # process don't get re-fetched (and re-stale-filtered) on restart.
        # ``_initial_offset`` is the value we loaded from disk — the stale
        # filter uses it to recognize "this update arrived during downtime,
        # process don't drop". A fresh install with no state file yields 0,
        # which preserves the original cold-start behavior.
        self._initial_offset = self._load_persisted_offset()
        self.offset = self._initial_offset
        self._has_persisted_offset = self._initial_offset > 0
        self._startup_time = time.time()
        self._thread = None

    @staticmethod
    def _load_persisted_offset() -> int:
        """Read offset from POLLER_STATE_FILE. Returns 0 on any failure
        (missing file, malformed JSON, non-int value) — fail-soft so a
        corrupted state file never prevents the loop from polling."""
        try:
            state = load_json(POLLER_STATE_FILE, default=None)
        except Exception as e:
            logger.warning("poller offset load failed: %s", e)
            return 0
        if not isinstance(state, dict):
            return 0
        try:
            return int(state.get("offset", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _save_offset(self) -> None:
        """Persist current offset atomically. Best-effort: a write failure
        means the next restart re-fetches updates we already acked, but
        that's recoverable (Telegram dedups via the same update_id) so we
        don't crash the poll loop on disk errors."""
        try:
            atomic_write_json(
                POLLER_STATE_FILE,
                {
                    "offset": int(self.offset),
                    "updated_ts": datetime.now(UTC).isoformat(),
                },
            )
        except Exception as e:
            logger.warning("poller offset persist failed: %s", e)

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
        prev_offset = self.offset
        self.offset = max(self.offset, update_id + 1)
        if self.offset > prev_offset:
            # Persist whenever the high-water mark moves so a restart
            # doesn't re-fetch updates we've already acknowledged. We do
            # this *before* parse/dispatch so a crashed callback doesn't
            # cause endless redelivery of the same poison message.
            self._save_offset()

        msg = update.get("message")
        if not msg:
            return

        # Only process messages from our chat_id. Drop others without logging —
        # no point persisting spam from strangers who can't affect state.
        chat = msg.get("chat", {})
        if str(chat.get("id")) != self.chat_id:
            return

        # Accumulate log outcome; single append in finally so we log every
        # inbound message exactly once, even if parse/dispatch raises.
        outcome = {"cmd": None, "processed": False, "drop_reason": None}
        try:
            # Stale filter: ignore messages older than 60s at startup so we
            # don't re-execute commands after a loop restart. Still log them
            # — useful for reconstructing what the user sent during downtime.
            #
            # Bypass when (a) we have a persisted offset and (b) this
            # update_id is past it. Those are post-restart pending updates
            # — by definition arrived during downtime, the user expects
            # them to execute, and the persisted offset proves we're not
            # accidentally re-running a stale getUpdates queue from a long
            # outage. Cold-start (no persisted offset) keeps the original
            # protection because we can't distinguish "user sent during
            # restart" from "Telegram re-delivering 2-week-old updates"
            # without that prior.
            msg_date = msg.get("date", 0)
            # update_id can EQUAL self._initial_offset legitimately: the
            # persisted value uses next-offset semantics (last_acked + 1)
            # so the first genuinely-new update after restart has
            # update_id == self._initial_offset, not strictly greater.
            # `>=` covers the single-message-during-restart case that was
            # the whole reason for adding persistence (Codex P1
            # 2026-04-28).
            is_post_restart_pending = (
                self._has_persisted_offset and update_id >= self._initial_offset
            )
            if msg_date < self._startup_time - 60 and not is_post_restart_pending:
                outcome["drop_reason"] = "stale_at_startup"
                return

            text = (msg.get("text") or "").strip()
            if not text:
                outcome["drop_reason"] = "empty_text"
                return

            cmd, args = self._parse_command(text)
            outcome["cmd"] = cmd
            if cmd is None:
                outcome["drop_reason"] = "unrecognized"
                return

            # Dispatch can raise (Avanza session, volume math, network) — we
            # want processed=True to mean "dispatch completed", not "dispatch
            # was attempted". On raise, tag drop_reason with the exception
            # type so the audit log reflects the actual outcome, then re-raise
            # to preserve the old error-propagation behavior.
            try:
                if cmd == "mode":
                    response = self._handle_mode_command(args)
                else:
                    response = self.on_command(cmd, args, self.config)
                if response:
                    self._send_reply(response)
                outcome["processed"] = True
            except Exception as exc:
                outcome["drop_reason"] = f"raised:{type(exc).__name__}"
                raise
        finally:
            self._log_inbound(update, msg, **outcome)

    def _log_inbound(self, update, msg, cmd, processed, drop_reason):
        """Persist one inbound message to data/telegram_inbound.jsonl.

        Rotation registered in portfolio/log_rotation.py (90d / 20 MB).
        """
        try:
            sender = msg.get("from") or {}
            entry = {
                "ts": datetime.now(UTC).isoformat(),
                "direction": "inbound",
                "update_id": update.get("update_id"),
                "message_id": msg.get("message_id"),
                "msg_date": msg.get("date"),
                "from": {
                    "id": sender.get("id"),
                    "username": sender.get("username"),
                },
                "text": msg.get("text") or "",
                "cmd": cmd,
                "processed": processed,
                "drop_reason": drop_reason,
            }
            atomic_append_jsonl(INBOUND_LOG, entry)
        except Exception as e:
            logger.warning("Inbound log write failed: %s", e)

    def _parse_command(self, text):
        """Parse ISKBETS and system commands from message text.

        Returns (cmd, args) or (None, None) for non-commands.
        Recognized: bought, sold, cancel, status, /mode
        """
        parts = text.split(None, 1)
        first_word = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        if first_word in ("bought", "sold", "cancel", "status"):
            return first_word, rest

        # /mode command — switch notification format
        if first_word in ("/mode", "mode"):
            return "mode", rest.strip().lower()

        return None, None

    def _handle_mode_command(self, mode_arg):
        """Handle /mode command — switch notification format.

        Args:
            mode_arg: "signals" or "probability" (or empty to query current mode)

        Returns:
            Reply text for the user.
        """
        from pathlib import Path

        from portfolio.file_utils import atomic_write_json

        config_path = Path(__file__).resolve().parent.parent / "config.json"

        if not mode_arg:
            # Query current mode
            current = self.config.get("notification", {}).get("mode", "signals")
            return f"Current notification mode: *{current}*"

        if mode_arg not in ("signals", "probability"):
            return "Usage: `/mode signals` or `/mode probability`"

        # Update config.json
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            cfg = {}

        # BUG-210: Guard against writing suspiciously small config.
        # If config.json was momentarily unreadable (symlink, AV lock, fs
        # glitch), cfg={} and the write below would destroy all API keys.
        if len(cfg) < 5:
            logger.error(
                "Refusing to write config — loaded config has only %d keys "
                "(expected 5+, possible transient read failure)", len(cfg)
            )
            return "Error: config file appears corrupt or unreadable. Try again."

        if "notification" not in cfg:
            cfg["notification"] = {}
        cfg["notification"]["mode"] = mode_arg

        atomic_write_json(config_path, cfg)

        # Update in-memory config
        if "notification" not in self.config:
            self.config["notification"] = {}
        self.config["notification"]["mode"] = mode_arg

        logger.info("Notification mode changed to: %s", mode_arg)
        return f"Notification mode set to *{mode_arg}*"

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
