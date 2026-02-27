"""Tests for portfolio.message_throttle — analysis message rate limiting."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest


class TestShouldSendAnalysis:
    """Tests for should_send_analysis()."""

    @patch("portfolio.message_throttle.load_json")
    def test_no_prior_sends(self, mock_load):
        from portfolio.message_throttle import should_send_analysis
        mock_load.return_value = {}

        assert should_send_analysis() is True

    @patch("portfolio.message_throttle.load_json")
    def test_recent_send_blocks(self, mock_load):
        from portfolio.message_throttle import should_send_analysis
        mock_load.return_value = {"last_analysis_sent": time.time() - 60}  # 1 min ago

        assert should_send_analysis() is False

    @patch("portfolio.message_throttle.load_json")
    def test_old_send_allows(self, mock_load):
        from portfolio.message_throttle import should_send_analysis
        mock_load.return_value = {"last_analysis_sent": time.time() - 20000}  # >3h ago

        assert should_send_analysis() is True

    @patch("portfolio.message_throttle.load_json")
    def test_custom_cooldown(self, mock_load):
        from portfolio.message_throttle import should_send_analysis
        mock_load.return_value = {"last_analysis_sent": time.time() - 600}  # 10 min ago

        # 5 min cooldown → should allow
        config = {"notification": {"analysis_cooldown_seconds": 300}}
        assert should_send_analysis(config) is True

        # 15 min cooldown → should block
        config = {"notification": {"analysis_cooldown_seconds": 900}}
        assert should_send_analysis(config) is False


class TestQueueAnalysis:
    """Tests for queue_analysis()."""

    @patch("portfolio.message_throttle._send_now")
    @patch("portfolio.message_throttle.should_send_analysis")
    def test_sends_immediately_when_cooldown_elapsed(self, mock_should, mock_send):
        from portfolio.message_throttle import queue_analysis
        mock_should.return_value = True
        mock_send.return_value = "sent"

        result = queue_analysis("test message", {})
        assert result == "sent"
        mock_send.assert_called_once()

    @patch("portfolio.message_throttle.atomic_write_json")
    @patch("portfolio.message_throttle.load_json")
    @patch("portfolio.message_throttle.should_send_analysis")
    def test_queues_when_cooldown_active(self, mock_should, mock_load, mock_write):
        from portfolio.message_throttle import queue_analysis
        mock_should.return_value = False
        mock_load.return_value = {}

        result = queue_analysis("test message", {})
        assert result == "queued"
        mock_write.assert_called_once()
        # Verify pending text was saved
        saved = mock_write.call_args[0][1]
        assert saved["pending_text"] == "test message"


class TestFlushAndSend:
    """Tests for flush_and_send()."""

    @patch("portfolio.message_throttle._send_now")
    @patch("portfolio.message_throttle.load_json")
    @patch("portfolio.message_throttle.should_send_analysis")
    def test_sends_pending_when_cooldown_elapsed(self, mock_should, mock_load, mock_send):
        from portfolio.message_throttle import flush_and_send
        mock_should.return_value = True
        mock_load.return_value = {"pending_text": "queued message", "pending_ts": time.time()}
        mock_send.return_value = "sent"

        result = flush_and_send({})
        assert result is True
        mock_send.assert_called_once_with("queued message", {})

    @patch("portfolio.message_throttle.load_json")
    @patch("portfolio.message_throttle.should_send_analysis")
    def test_no_pending(self, mock_should, mock_load):
        from portfolio.message_throttle import flush_and_send
        mock_should.return_value = True
        mock_load.return_value = {}

        result = flush_and_send({})
        assert result is False

    @patch("portfolio.message_throttle.load_json")
    @patch("portfolio.message_throttle.should_send_analysis")
    def test_cooldown_active(self, mock_should, mock_load):
        from portfolio.message_throttle import flush_and_send
        mock_should.return_value = False

        result = flush_and_send({})
        assert result is False


class TestSendNow:
    """Tests for _send_now()."""

    @patch("portfolio.message_throttle.atomic_write_json")
    @patch("portfolio.message_throttle.load_json")
    @patch("portfolio.message_store.send_or_store")
    def test_sends_and_updates_state(self, mock_send, mock_load, mock_write):
        from portfolio.message_throttle import _send_now
        mock_load.return_value = {"pending_text": "old", "pending_ts": 123}

        result = _send_now("new message", {"telegram": {"token": "x", "chat_id": "y"}})
        assert result == "sent"
        mock_send.assert_called_once()
        # Verify pending was cleared
        saved = mock_write.call_args[0][1]
        assert "pending_text" not in saved
        assert "pending_ts" not in saved
        assert "last_analysis_sent" in saved
