"""Tests for portfolio.earnings_calendar — earnings proximity gate."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from portfolio.earnings_calendar import (
    GATE_DAYS,
    clear_cache,
    get_all_earnings_proximity,
    get_earnings_proximity,
    should_gate_earnings,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear earnings cache before each test."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# should_gate_earnings
# ---------------------------------------------------------------------------

class TestShouldGateEarnings:
    @patch("portfolio.earnings_calendar._fetch_earnings_date")
    def test_gate_active_within_window(self, mock_fetch):
        """Gate is active when earnings are within GATE_DAYS."""
        tomorrow = (datetime.now(UTC).date() + timedelta(days=1)).isoformat()
        mock_fetch.return_value = {
            "earnings_date": tomorrow,
            "days_until": 1,
            "gate_active": True,
            "timing": "after_close",
        }

        assert should_gate_earnings("NVDA") is True

    @patch("portfolio.earnings_calendar._fetch_earnings_date")
    def test_gate_inactive_far_out(self, mock_fetch):
        """Gate is inactive when earnings are far away."""
        future = (datetime.now(UTC).date() + timedelta(days=30)).isoformat()
        mock_fetch.return_value = {
            "earnings_date": future,
            "days_until": 30,
            "gate_active": False,
            "timing": "unknown",
        }

        assert should_gate_earnings("NVDA") is False

    @patch("portfolio.earnings_calendar._fetch_earnings_date")
    def test_gate_inactive_no_data(self, mock_fetch):
        """Gate is inactive when no earnings data available."""
        mock_fetch.return_value = None
        assert should_gate_earnings("NVDA") is False

    def test_non_stock_always_false(self):
        """Crypto and metals are never gated."""
        assert should_gate_earnings("BTC-USD") is False
        assert should_gate_earnings("XAU-USD") is False
        assert should_gate_earnings("ETH-USD") is False
        assert should_gate_earnings("XAG-USD") is False

    def test_unknown_ticker_false(self):
        """Unknown tickers return False."""
        assert should_gate_earnings("UNKNOWN") is False


# ---------------------------------------------------------------------------
# get_earnings_proximity
# ---------------------------------------------------------------------------

class TestGetEarningsProximity:
    @patch("portfolio.earnings_calendar._fetch_earnings_date")
    def test_returns_proximity_for_stock(self, mock_fetch):
        """Returns proximity dict for valid stock ticker."""
        mock_fetch.return_value = {
            "earnings_date": "2026-04-15",
            "days_until": 15,
            "gate_active": False,
            "timing": "unknown",
        }

        result = get_earnings_proximity("PLTR")
        assert result is not None
        assert result["earnings_date"] == "2026-04-15"
        assert result["days_until"] == 15
        assert result["gate_active"] is False

    def test_returns_none_for_crypto(self):
        """Crypto tickers return None (no earnings)."""
        assert get_earnings_proximity("BTC-USD") is None

    def test_returns_none_for_metals(self):
        """Metal tickers return None (no earnings)."""
        assert get_earnings_proximity("XAG-USD") is None

    @patch("portfolio.earnings_calendar._fetch_earnings_date")
    def test_cache_hit(self, mock_fetch):
        """Second call uses cache, doesn't re-fetch."""
        mock_fetch.return_value = {
            "earnings_date": "2026-04-15",
            "days_until": 15,
            "gate_active": False,
            "timing": "unknown",
        }

        get_earnings_proximity("MU")
        get_earnings_proximity("MU")

        assert mock_fetch.call_count == 1

    @patch("portfolio.earnings_calendar._fetch_earnings_date")
    def test_cache_miss_different_tickers(self, mock_fetch):
        """Different tickers are cached independently."""
        mock_fetch.return_value = {
            "earnings_date": "2026-04-15",
            "days_until": 15,
            "gate_active": False,
            "timing": "unknown",
        }

        get_earnings_proximity("MU")
        get_earnings_proximity("NVDA")

        assert mock_fetch.call_count == 2


# ---------------------------------------------------------------------------
# get_all_earnings_proximity
# ---------------------------------------------------------------------------

class TestGetAllEarningsProximity:
    @patch("portfolio.earnings_calendar._fetch_earnings_date")
    def test_returns_dict_for_all_stocks(self, mock_fetch):
        """Returns proximity for all stock tickers that have data."""
        mock_fetch.return_value = {
            "earnings_date": "2026-04-15",
            "days_until": 15,
            "gate_active": False,
            "timing": "unknown",
        }

        result = get_all_earnings_proximity()
        assert isinstance(result, dict)
        # Should have entries for stock tickers (those that returned data)
        assert len(result) > 0
        for ticker in result:
            assert ticker in {"PLTR", "NVDA", "MU", "SMCI", "TSM", "TTWO", "VRT", "MSTR"}

    @patch("portfolio.earnings_calendar._fetch_earnings_date")
    def test_excludes_none_results(self, mock_fetch):
        """Tickers with no earnings data are excluded."""
        mock_fetch.return_value = None

        result = get_all_earnings_proximity()
        assert result == {}


# ---------------------------------------------------------------------------
# Gate timing edge cases
# ---------------------------------------------------------------------------

class TestGateEdgeCases:
    @patch("portfolio.earnings_calendar._fetch_earnings_date")
    def test_earnings_today(self, mock_fetch):
        """Earnings today = gate active."""
        today = datetime.now(UTC).date().isoformat()
        mock_fetch.return_value = {
            "earnings_date": today,
            "days_until": 0,
            "gate_active": True,
            "timing": "after_close",
        }
        assert should_gate_earnings("NVDA") is True

    @patch("portfolio.earnings_calendar._fetch_earnings_date")
    def test_earnings_exactly_at_gate_boundary(self, mock_fetch):
        """Earnings exactly at GATE_DAYS = gate active."""
        boundary = (datetime.now(UTC).date() + timedelta(days=GATE_DAYS)).isoformat()
        mock_fetch.return_value = {
            "earnings_date": boundary,
            "days_until": GATE_DAYS,
            "gate_active": True,
            "timing": "unknown",
        }
        assert should_gate_earnings("NVDA") is True

    @patch("portfolio.earnings_calendar._fetch_earnings_date")
    def test_earnings_one_past_gate(self, mock_fetch):
        """Earnings one day past GATE_DAYS = gate inactive."""
        past = (datetime.now(UTC).date() + timedelta(days=GATE_DAYS + 1)).isoformat()
        mock_fetch.return_value = {
            "earnings_date": past,
            "days_until": GATE_DAYS + 1,
            "gate_active": False,
            "timing": "unknown",
        }
        assert should_gate_earnings("NVDA") is False
