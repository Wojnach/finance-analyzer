"""Tests for portfolio.daily_digest — morning daily digest."""

import json
import time
from unittest.mock import patch, MagicMock

import pytest


class TestShouldSendDailyDigest:
    """Tests for should_send_daily_digest()."""

    @patch("portfolio.daily_digest.datetime")
    @patch("portfolio.daily_digest._get_last_daily_digest_time")
    def test_correct_hour_no_prior(self, mock_last, mock_dt):
        from portfolio.daily_digest import should_send_daily_digest
        mock_last.return_value = 0
        # Simulate 06:00 UTC
        mock_now = MagicMock()
        mock_now.hour = 6
        mock_dt.now.return_value = mock_now

        assert should_send_daily_digest({}) is True

    @patch("portfolio.daily_digest.datetime")
    @patch("portfolio.daily_digest._get_last_daily_digest_time")
    def test_wrong_hour(self, mock_last, mock_dt):
        from portfolio.daily_digest import should_send_daily_digest
        mock_last.return_value = 0
        mock_now = MagicMock()
        mock_now.hour = 14  # wrong hour
        mock_dt.now.return_value = mock_now

        assert should_send_daily_digest({}) is False

    @patch("portfolio.daily_digest.datetime")
    @patch("portfolio.daily_digest._get_last_daily_digest_time")
    def test_already_sent_today(self, mock_last, mock_dt):
        from portfolio.daily_digest import should_send_daily_digest
        mock_last.return_value = time.time() - 3600  # 1 hour ago
        mock_now = MagicMock()
        mock_now.hour = 6
        mock_dt.now.return_value = mock_now

        assert should_send_daily_digest({}) is False

    @patch("portfolio.daily_digest.datetime")
    @patch("portfolio.daily_digest._get_last_daily_digest_time")
    def test_custom_hour(self, mock_last, mock_dt):
        from portfolio.daily_digest import should_send_daily_digest
        mock_last.return_value = 0
        mock_now = MagicMock()
        mock_now.hour = 8
        mock_dt.now.return_value = mock_now

        config = {"notification": {"daily_digest_hour_utc": 8}}
        assert should_send_daily_digest(config) is True


class TestBuildDailyDigest:
    """Tests for build_daily_digest()."""

    @patch("portfolio.daily_digest.BOLD_STATE_FILE")
    @patch("portfolio.daily_digest.load_state")
    @patch("portfolio.daily_digest.load_json")
    def test_basic_message_structure(self, mock_load_json, mock_load_state, mock_bold_file):
        from portfolio.daily_digest import build_daily_digest

        mock_load_json.return_value = {
            "fx_rate": 10.5,
            "signals": {
                "XAG-USD": {"price_usd": 89.5},
                "BTC-USD": {"price_usd": 67000},
            },
            "cumulative_gains": {
                "ticker_changes": {
                    "XAG-USD": {"change_1d": 2.0, "change_3d": 5.0, "change_7d": 12.0},
                    "BTC-USD": {"change_1d": -0.5, "change_3d": -2.0, "change_7d": -3.0},
                },
                "movers": [],
            },
            "focus_probabilities": {
                "XAG-USD": {
                    "1d": {"direction": "up", "probability": 0.72},
                    "accuracy_1d": 0.71,
                    "accuracy_samples": 89,
                },
            },
        }
        mock_load_state.return_value = {
            "cash_sek": 425000,
            "initial_value_sek": 500000,
            "holdings": {},
        }
        mock_bold_file.exists.return_value = False

        config = {"notification": {"focus_tickers": ["XAG-USD", "BTC-USD"]}}
        msg = build_daily_digest(config)

        assert msg is not None
        assert "*DAILY*" in msg
        assert "XAG" in msg
        assert "BTC" in msg
        assert "Patient" in msg

    @patch("portfolio.daily_digest.load_json")
    def test_no_summary_returns_none(self, mock_load_json):
        from portfolio.daily_digest import build_daily_digest
        mock_load_json.return_value = {}

        msg = build_daily_digest({})
        assert msg is None


class TestFormatPrice:
    """Tests for _format_price()."""

    def test_large_price(self):
        from portfolio.daily_digest import _format_price
        assert _format_price(67000) == "$67K"

    def test_medium_price(self):
        from portfolio.daily_digest import _format_price
        assert _format_price(426) == "$426"

    def test_small_price(self):
        from portfolio.daily_digest import _format_price
        assert _format_price(89.5) == "$89.5"

    def test_very_large_price(self):
        from portfolio.daily_digest import _format_price
        assert _format_price(100000) == "$100K"


class TestMaybeSendDailyDigest:
    """Tests for maybe_send_daily_digest()."""

    @patch("portfolio.daily_digest.should_send_daily_digest")
    @patch("portfolio.daily_digest.build_daily_digest")
    @patch("portfolio.daily_digest.send_or_store")
    @patch("portfolio.daily_digest._set_last_daily_digest_time")
    def test_sends_when_should(self, mock_set, mock_send, mock_build, mock_should):
        from portfolio.daily_digest import maybe_send_daily_digest
        mock_should.return_value = True
        mock_build.return_value = "*DAILY* test message"

        maybe_send_daily_digest({"telegram": {"token": "x", "chat_id": "y"}})

        mock_send.assert_called_once()
        mock_set.assert_called_once()

    @patch("portfolio.daily_digest.should_send_daily_digest")
    @patch("portfolio.daily_digest.build_daily_digest")
    @patch("portfolio.daily_digest.send_or_store")
    def test_skips_when_not_time(self, mock_send, mock_build, mock_should):
        from portfolio.daily_digest import maybe_send_daily_digest
        mock_should.return_value = False

        maybe_send_daily_digest({})

        mock_build.assert_not_called()
        mock_send.assert_not_called()
