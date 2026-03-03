"""Tests for helper functions in portfolio/main.py.

Covers:
- _extract_triggered_tickers: regex parsing of trigger reason strings
- _run_post_cycle: post-cycle housekeeping (digest, flush, AV refresh)
"""

import pytest
from unittest import mock


# ---------------------------------------------------------------------------
# _extract_triggered_tickers tests (TEST-9 / BUG-48)
# ---------------------------------------------------------------------------

from portfolio.main import _extract_triggered_tickers


class TestExtractTriggeredTickers:
    """Parse ticker names from trigger reason strings."""

    def test_consensus_reason(self):
        reasons = ["MU consensus BUY (79%)"]
        assert _extract_triggered_tickers(reasons) == {"MU"}

    def test_consensus_with_dash_ticker(self):
        reasons = ["BTC-USD consensus SELL (65%)"]
        assert _extract_triggered_tickers(reasons) == {"BTC-USD"}

    def test_moved_reason(self):
        reasons = ["ETH-USD moved 3.1% up"]
        assert _extract_triggered_tickers(reasons) == {"ETH-USD"}

    def test_flipped_reason(self):
        reasons = ["NVDA flipped SELL->BUY (sustained)"]
        assert _extract_triggered_tickers(reasons) == {"NVDA"}

    def test_multiple_reasons(self):
        reasons = [
            "BTC-USD consensus BUY (80%)",
            "ETH-USD moved 2.5% down",
            "PLTR flipped HOLD->SELL (sustained)",
        ]
        result = _extract_triggered_tickers(reasons)
        assert result == {"BTC-USD", "ETH-USD", "PLTR"}

    def test_non_ticker_reason_ignored(self):
        reasons = ["post-trade reassessment"]
        assert _extract_triggered_tickers(reasons) == set()

    def test_fg_reason_ignored(self):
        reasons = ["F&G crossed 80 (45->82)"]
        assert _extract_triggered_tickers(reasons) == set()

    def test_sentiment_reason_ignored(self):
        reasons = ["BTC-USD sentiment bullish->bearish (sustained)"]
        # "sentiment" is not in the regex pattern (consensus|moved|flipped)
        assert _extract_triggered_tickers(reasons) == set()

    def test_empty_list(self):
        assert _extract_triggered_tickers([]) == set()

    def test_mixed_valid_and_invalid(self):
        reasons = [
            "MU consensus BUY (60%)",
            "post-trade reassessment",
            "F&G crossed 20 (25->18)",
            "XAG-USD moved 2.3% up",
        ]
        result = _extract_triggered_tickers(reasons)
        assert result == {"MU", "XAG-USD"}

    def test_metals_ticker(self):
        reasons = ["XAU-USD consensus BUY (70%)"]
        assert _extract_triggered_tickers(reasons) == {"XAU-USD"}

    def test_duplicate_ticker_deduplicated(self):
        reasons = [
            "BTC-USD consensus BUY (80%)",
            "BTC-USD moved 3.0% up",
        ]
        result = _extract_triggered_tickers(reasons)
        assert result == {"BTC-USD"}
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _run_post_cycle tests (ARCH-10 / REF-13)
# ---------------------------------------------------------------------------

from portfolio.main import _run_post_cycle


class TestRunPostCycle:
    """_run_post_cycle calls digest, daily digest, message flush, and AV refresh."""

    @mock.patch("portfolio.main._maybe_send_digest")
    def test_calls_digest(self, mock_digest):
        config = {"notification": {}}
        _run_post_cycle(config)
        mock_digest.assert_called_once_with(config)

    @mock.patch("portfolio.main._maybe_send_digest")
    @mock.patch("portfolio.daily_digest.maybe_send_daily_digest")
    def test_calls_daily_digest(self, mock_daily, mock_digest):
        config = {"notification": {}}
        _run_post_cycle(config)
        mock_daily.assert_called_once_with(config)

    @mock.patch("portfolio.main._maybe_send_digest")
    @mock.patch("portfolio.message_throttle.flush_and_send")
    def test_calls_message_flush(self, mock_flush, mock_digest):
        config = {"notification": {}}
        _run_post_cycle(config)
        mock_flush.assert_called_once_with(config)

    @mock.patch("portfolio.main._maybe_send_digest")
    @mock.patch("portfolio.daily_digest.maybe_send_daily_digest", side_effect=Exception("dd fail"))
    def test_daily_digest_failure_does_not_crash(self, mock_daily, mock_digest):
        config = {"notification": {}}
        # Should not raise
        _run_post_cycle(config)

    @mock.patch("portfolio.main._maybe_send_digest")
    @mock.patch("portfolio.message_throttle.flush_and_send", side_effect=Exception("flush fail"))
    def test_message_flush_failure_does_not_crash(self, mock_flush, mock_digest):
        config = {"notification": {}}
        # Should not raise
        _run_post_cycle(config)
