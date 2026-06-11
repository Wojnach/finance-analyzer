"""Telegram Poller — Background thread for ISKBETS + system commands.

Polls getUpdates every 5 seconds. Parses bought/sold/cancel/status commands
and delegates to iskbets.handle_command(). Also handles /mode command for
switching notification format (signals vs probability).
"""

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
# _handle_update bypasses the stale filter for post-restart pending
# updates (those the user expects to execute, e.g. a ``bought MSTR …``
# confirmation sent while the loop was bouncing) UP TO a bounded age:
# see RESTART_BYPASS_MAX_AGE_S below.
POLLER_STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "telegram_poller_state.json"

# Codex P1 round-4 (2026-04-28): cap the post-restart bypass to 1 hour.
# A bot that was down for days could otherwise execute every queued
# 'bought MSTR …' confirmation on next start, even though the user has
# since traded manually. 1 h is generous enough to cover any realistic
# restart window (schtasks rerun + loop boot < 5 min in practice) while
# still rejecting commands that are old enough that the user almost
# certainly resolved them out-of-band. Beyond this window the original
# 60 s stale filter applies.
RESTART_BYPASS_MAX_AGE_S = 60 * 60

# 2026-06-11 (audit batch 9, finding "offset-holdback retry is illusory"):
# The old design left the persisted offset un-advanced on a dispatch raise
# so a *restart* would re-fetch and retry. But Telegram acks an update as
# soon as the next getUpdates is called with offset > update_id, so the
# in-process 5s poll permanently consumes the failed update server-side —
# the persisted-offset replay only fired in the narrow window where the
# process died within that single poll. Real-money bookkeeping commands
# ("bought MSTR …") were silently lost on any transient handler failure.
#
# New design — ack-after-success with a bounded retry:
#   * On dispatch raise, increment a persisted per-update attempt count and
#     do NOT advance the persisted offset (the in-memory offset still
#     advances within this poll to avoid spinning, but the next poll resets
#     it to the persisted value so the failed update is re-fetched).
#   * After MAX_DISPATCH_ATTEMPTS failures the update is given up on: the
#     offset advances past it (so it stops blocking the queue) AND an
#     explicit "command dropped" reply is sent to the user (premortem hook
#     12 — never a silent drop; the user may have sent a halt command).
MAX_DISPATCH_ATTEMPTS = 3


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
        # The last offset durably written to disk. The in-memory ``self.offset``
        # may run ahead within a poll to avoid spinning on a poison update, but
        # a dispatch failure leaves ``_persisted_offset`` behind so the next
        # poll re-fetches the failed update (ack-after-success, 2026-06-11).
        self._persisted_offset = self._initial_offset
        self._has_persisted_offset = self._initial_offset > 0
        # Per-update_id dispatch attempt counts, persisted alongside the offset
        # so a restart does not reset the bounded-retry counter (otherwise a
        # reliably-failing command would retry forever across restarts).
        self._attempts = self._load_persisted_attempts()
        self._startup_time = time.time()
        self._thread = None

    @staticmethod
    def _load_persisted_offset() -> int:
        """Read offset from POLLER_STATE_FILE. Returns 0 on any failure
        (missing file, malformed JSON, non-int value, or negative
        integer) — fail-soft so a corrupted state file never prevents
        the loop from polling. Negative values are explicitly rejected
        because Telegram's getUpdates treats negative offsets as a
        backward count from the latest update, not as cold-start
        behavior (Codex P3 round-3 2026-04-28)."""
        try:
            state = load_json(POLLER_STATE_FILE, default=None)
        except Exception as e:
            logger.warning("poller offset load failed: %s", e)
            return 0
        if not isinstance(state, dict):
            return 0
        try:
            offset = int(state.get("offset", 0) or 0)
        except (TypeError, ValueError):
            return 0
        if offset < 0:
            logger.warning(
                "poller offset state had negative value %d; clamping to 0",
                offset,
            )
            return 0
        return offset

    @staticmethod
    def _load_persisted_attempts() -> dict:
        """Read the per-update dispatch attempt counts from POLLER_STATE_FILE.

        Returns a ``{update_id_str: int}`` dict. Fail-soft to ``{}`` on any
        problem (missing file, malformed JSON, wrong shape) — a lost counter
        just means a failing command gets a fresh set of retries, which is
        strictly safer than crashing the poll loop."""
        try:
            state = load_json(POLLER_STATE_FILE, default=None)
        except Exception as e:
            logger.warning("poller attempts load failed: %s", e)
            return {}
        if not isinstance(state, dict):
            return {}
        raw = state.get("attempts")
        if not isinstance(raw, dict):
            return {}
        out = {}
        for k, v in raw.items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return out

    def _save_offset(self) -> None:
        """Persist the *durable* offset + attempt counts atomically.

        2026-06-11: persists ``self._persisted_offset`` (the ack-after-success
        watermark), NOT the in-memory ``self.offset`` which may run ahead of a
        still-failing update. Best-effort: a write failure means the next
        restart re-fetches updates we already acked, recoverable via Telegram's
        update_id dedup, so we never crash the poll loop on disk errors."""
        try:
            atomic_write_json(
                POLLER_STATE_FILE,
                {
                    "offset": int(self._persisted_offset),
                    "attempts": {str(k): int(v) for k, v in self._attempts.items()},
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
                # 2026-06-11: start each poll from the durable watermark so a
                # dispatch that failed last poll (and therefore did not advance
                # _persisted_offset) is re-fetched and retried, up to
                # MAX_DISPATCH_ATTEMPTS. The in-memory offset may run ahead
                # within a single poll to avoid spinning on a poison update.
                self.offset = self._persisted_offset
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

    # Drop reasons that represent a *settled* outcome — the message was
    # examined and intentionally not acted on (stale, empty, unrecognized,
    # or no message body / wrong chat). Re-fetching these on a restart
    # would just settle them the same way, so we ack the offset.
    # Excluded: ``raised:*`` outcomes — those represent a transient
    # dispatch failure where the user's command is genuinely at risk of
    # being lost if we ack the offset before it succeeds (Codex P1
    # round-7 2026-04-28).
    _SETTLED_DROP_REASONS = frozenset({
        "stale_at_startup",
        "empty_text",
        "unrecognized",
    })

    def _ack_update(self, update_id) -> None:
        """Durably ack *update_id*: advance the persisted offset past it and
        clear any retry counter, then persist. The in-memory ``self.offset``
        already advanced within the poll; this commits it to disk."""
        self._persisted_offset = max(self._persisted_offset, update_id + 1)
        self._attempts.pop(str(update_id), None)
        self._save_offset()

    def _notify_dropped(self, cmd, text) -> None:
        """Tell the user a command was dropped after exhausting retries.

        Premortem hook 12: a dropped command must NEVER be silent — the user
        may have sent a halt/position-update command.

        DEVIATION from the literal "via telegram_notifications" directive:
        telegram_notifications.send_telegram is gated by the ``layer1_messages``
        config flag (default OFF) and returns True *without sending* when the
        gate is off — routing the drop notice through it would silently swallow
        exactly the message hook 12 says must never be silent. The poller's own
        ``_send_reply`` posts directly to the user's chat, is ungated, and is
        already the path every successful command reply uses, so we use it for
        guaranteed delivery. Best-effort; ``_send_reply`` already swallows its
        own send errors and never raises."""
        label = cmd or (text or "")[:60]
        self._send_reply(
            f"⚠ command dropped after {MAX_DISPATCH_ATTEMPTS} failed "
            f"attempts: {label}"
        )

    def _handle_update(self, update):
        """Process a single update."""
        update_id = update.get("update_id", 0)
        self.offset = max(self.offset, update_id + 1)
        # In-memory offset advances unconditionally so a single poison update
        # doesn't loop the in-process poll. Durable persistence is decided by
        # the ack-after-success accounting at the end of this method
        # (2026-06-11): a settled outcome acks; a raised dispatch leaves the
        # durable offset behind for a bounded retry.
        offset_settled = False

        msg = update.get("message")
        if not msg:
            offset_settled = True

        # Only process messages from our chat_id. Drop others without logging —
        # no point persisting spam from strangers who can't affect state.
        # We DO still ack the offset on chat-mismatch so the bot's
        # getUpdates queue doesn't accumulate stranger spam over time.
        if msg is not None:
            chat = msg.get("chat", {})
            if str(chat.get("id")) != self.chat_id:
                offset_settled = True
                msg = None  # short-circuit out of the rest of the body

        if msg is None:
            if offset_settled:
                self._ack_update(update_id)
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
            #
            # Codex P1 round-4 (2026-04-28): bound the bypass to
            # RESTART_BYPASS_MAX_AGE_S so a multi-day outage doesn't
            # execute every queued command on next start.
            is_post_restart_pending = (
                self._has_persisted_offset
                and update_id >= self._initial_offset
                and msg_date >= self._startup_time - RESTART_BYPASS_MAX_AGE_S
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
            # was attempted". 2026-06-11: the exception is now caught locally
            # (no re-raise) so the finally block can run the ack-after-success
            # bookkeeping; _poll_loop's own try/except no longer needs to see
            # it. The raise:* drop_reason is still logged for the audit trail.
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
                logger.warning("Poller dispatch failed for %r: %s", cmd, exc)
        finally:
            self._log_inbound(update, msg, **outcome)
            # Offset/ack accounting (2026-06-11 ack-after-success):
            #   * processed OR intentional settled-drop -> ack (advance the
            #     durable offset, clear any attempt counter).
            #   * raised dispatch -> bounded retry: count the attempt; if we
            #     have exhausted MAX_DISPATCH_ATTEMPTS, give up (ack so it
            #     stops blocking the queue) and send the user an explicit
            #     "command dropped" reply (premortem hook 12 — never silent).
            #     Otherwise leave the durable offset behind so the next poll
            #     re-fetches and retries.
            if outcome["processed"] or outcome["drop_reason"] in self._SETTLED_DROP_REASONS:
                self._ack_update(update_id)
            elif outcome["drop_reason"] and outcome["drop_reason"].startswith("raised:"):
                key = str(update_id)
                attempts = self._attempts.get(key, 0) + 1
                self._attempts[key] = attempts
                if attempts >= MAX_DISPATCH_ATTEMPTS:
                    logger.error(
                        "Poller giving up on update %s (cmd=%r) after %d failed "
                        "attempts — dropping", update_id, outcome["cmd"], attempts,
                    )
                    self._notify_dropped(outcome["cmd"], text)
                    self._ack_update(update_id)  # also clears the counter
                else:
                    # Persist the incremented counter without advancing the
                    # durable offset, so the retry budget survives a restart.
                    self._save_offset()

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

        # Adversarial review 04-29 IN-P1-3 (2026-05-02): use the
        # file_utils helpers (load_json + atomic_write_json) rather than
        # raw open()/json.load(). Two reasons:
        #   1. CLAUDE.md rule 4: "Atomic I/O only".
        #   2. config.json is a symlink to an external file; raw open() can
        #      race against an external atomic_write_json rename mid-read on
        #      Windows (we've seen partial-byte reads in agent.log). load_json
        #      handles the same edge cases (missing/corrupt → default) as
        #      every other consumer in the codebase.
        config_path = Path(__file__).resolve().parent.parent / "config.json"

        if not mode_arg:
            # Query current mode
            current = self.config.get("notification", {}).get("mode", "signals")
            return f"Current notification mode: *{current}*"

        if mode_arg not in ("signals", "probability"):
            return "Usage: `/mode signals` or `/mode probability`"

        # Update config.json — load_json returns {} for missing/corrupt files
        # without raising, so the BUG-210 size guard below catches both the
        # genuine-corrupt case and the transient-unreadable case.
        cfg = load_json(config_path, default={})

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
