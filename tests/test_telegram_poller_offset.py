"""Tests for Telegram poller offset persistence.

Background (2026-04-28): TelegramPoller.offset is in-memory only. After every
loop restart it resets to 0 and re-fetches all pending updates, then drops
anything older than `startup_time - 60s` via the stale filter. Combined,
that means real user commands sent during a restart window are silently
dropped.

This test pins the expected behavior:
  - On instantiation, the poller loads the persisted offset from disk.
  - After processing an update, it persists the new offset atomically.
  - The stale filter is bypassed for messages whose update_id is greater
    than the persisted offset (those are post-restart pending updates that
    arrived during downtime, and the user expects them to execute).
"""

import json
import time

import pytest


@pytest.fixture()
def poller_paths(tmp_path, monkeypatch):
    """Redirect the poller's state and inbound log files to tmp."""
    state_file = tmp_path / "telegram_poller_state.json"
    inbound_file = tmp_path / "telegram_inbound.jsonl"
    monkeypatch.setattr(
        "portfolio.telegram_poller.POLLER_STATE_FILE", state_file,
        raising=False,
    )
    monkeypatch.setattr(
        "portfolio.telegram_poller.INBOUND_LOG", inbound_file,
    )
    return state_file, inbound_file


@pytest.fixture()
def fake_config():
    return {"telegram": {"token": "fake-token", "chat_id": "12345"}}


def _build_update(update_id, msg_date, text="bought MSTR 130 100000",
                  chat_id=12345, from_id=999, message_id=1001):
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id,
            "date": msg_date,
            "chat": {"id": chat_id},
            "from": {"id": from_id, "username": "trader"},
            "text": text,
        },
    }


def _read_jsonl(path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


class TestPollerOffsetPersistence:
    """Offset survives across TelegramPoller instances."""

    def test_fresh_install_starts_at_zero(self, poller_paths, fake_config):
        from portfolio.telegram_poller import TelegramPoller

        state_file, _ = poller_paths
        assert not state_file.exists()
        poller = TelegramPoller(fake_config, on_command=lambda *a: None)
        assert poller.offset == 0

    def test_offset_advance_persists_to_disk(self, poller_paths, fake_config):
        from portfolio.telegram_poller import TelegramPoller

        state_file, _inbound = poller_paths
        poller = TelegramPoller(fake_config, on_command=lambda *a: None)

        # Process an update — offset advances to update_id + 1.
        update = _build_update(update_id=42, msg_date=int(time.time()))
        poller._handle_update(update)

        assert poller.offset == 43
        # And it landed on disk.
        assert state_file.exists()
        on_disk = json.loads(state_file.read_text())
        assert on_disk["offset"] == 43

    def test_new_instance_loads_offset_from_disk(self, poller_paths, fake_config):
        from portfolio.telegram_poller import TelegramPoller

        state_file, _inbound = poller_paths
        # Pre-seed disk with an offset from a previous "session".
        state_file.write_text(json.dumps({"offset": 1000, "updated_ts": "x"}))

        poller = TelegramPoller(fake_config, on_command=lambda *a: None)
        assert poller.offset == 1000

    def test_first_update_at_persisted_boundary_bypasses_stale_filter(
        self, poller_paths, fake_config,
    ):
        """Codex P1 follow-up: persisted offset uses next-offset semantics
        (update_id + 1), so the first genuinely-new update after restart
        has update_id == self._initial_offset, not strictly greater. The
        bypass must accept ``>=``, otherwise a single command sent during
        the restart window still trips the stale filter — defeating the
        whole reason we persist."""
        from portfolio.telegram_poller import TelegramPoller

        state_file, inbound_file = poller_paths
        state_file.write_text(json.dumps({"offset": 1001, "updated_ts": "x"}))

        called = {}

        def on_command(cmd, args, _config):
            called["cmd"] = cmd
            return None

        poller = TelegramPoller(fake_config, on_command=on_command)
        # update_id == persisted_offset is the very first new update.
        update = _build_update(
            update_id=1001,
            msg_date=int(time.time()) - 5 * 60,
            text="bought MSTR 130 100000",
        )
        poller._handle_update(update)

        assert called.get("cmd") == "bought"
        rows = _read_jsonl(inbound_file)
        assert len(rows) == 1
        assert rows[0]["processed"] is True
        assert rows[0]["drop_reason"] is None

    def test_post_restart_pending_update_bypasses_stale_filter(
        self, poller_paths, fake_config,
    ):
        """Update arriving with id > persisted_offset should be processed
        even if msg_date < startup - 60. That's the "user sent a command
        while we were restarting" case the persistence is meant to fix."""
        from portfolio.telegram_poller import TelegramPoller

        state_file, inbound_file = poller_paths
        # User had previously processed updates up to id=1000.
        state_file.write_text(json.dumps({"offset": 1001, "updated_ts": "x"}))

        called = {}

        def on_command(cmd, args, _config):
            called["cmd"] = cmd
            called["args"] = args
            return None

        poller = TelegramPoller(fake_config, on_command=on_command)
        # An update sent 5 minutes before the poller started — that's older
        # than the 60s stale filter window, but it has update_id 1500, which
        # is past the persisted offset, so it must be processed not dropped.
        update = _build_update(
            update_id=1500,
            msg_date=int(time.time()) - 5 * 60,
            text="bought MSTR 130 100000",
        )
        poller._handle_update(update)

        assert called.get("cmd") == "bought"
        rows = _read_jsonl(inbound_file)
        assert len(rows) == 1
        assert rows[0]["processed"] is True
        assert rows[0]["drop_reason"] is None

    def test_cold_start_still_drops_truly_old_messages(self, poller_paths, fake_config):
        """When there's no persisted offset (fresh install or wiped state),
        the stale filter must still drop ancient messages — that's the
        original protection against re-running stale commands after a long
        outage."""
        from portfolio.telegram_poller import TelegramPoller

        _state_file, inbound_file = poller_paths
        poller = TelegramPoller(fake_config, on_command=lambda *a: None)

        # update_id is small (< default offset 0 is N/A) AND msg_date is ancient.
        # Without a persisted offset the poller can't know if this is a replay
        # from a 2-week-old getUpdates queue, so the stale filter still trips.
        update = _build_update(
            update_id=1,
            msg_date=int(time.time()) - 7 * 24 * 3600,
        )
        poller._handle_update(update)

        rows = _read_jsonl(inbound_file)
        assert len(rows) == 1
        assert rows[0]["drop_reason"] == "stale_at_startup"
        assert rows[0]["processed"] is False

    def test_persisted_offset_does_not_decrease(self, poller_paths, fake_config):
        """If a stale getUpdates returns an older update_id (Telegram does
        not guarantee ordering across reconnects), the persisted offset
        must not regress."""
        from portfolio.telegram_poller import TelegramPoller

        state_file, _inbound = poller_paths
        state_file.write_text(json.dumps({"offset": 1500, "updated_ts": "x"}))

        poller = TelegramPoller(fake_config, on_command=lambda *a: None)
        poller._handle_update(_build_update(update_id=200, msg_date=int(time.time())))

        # Offset is unchanged — the in-memory and on-disk both stay at 1500
        # (or higher), never below it.
        assert poller.offset >= 1500
        on_disk = json.loads(state_file.read_text())
        assert on_disk["offset"] >= 1500

    def test_corrupt_state_file_falls_back_to_zero(self, poller_paths, fake_config):
        """A garbled state file must not crash the poller at startup."""
        from portfolio.telegram_poller import TelegramPoller

        state_file, _inbound = poller_paths
        state_file.write_text("not-json{")

        poller = TelegramPoller(fake_config, on_command=lambda *a: None)
        assert poller.offset == 0
