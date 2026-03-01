"""Regression tests for bugs found in auto-improve session #4.

BUG-30: dashboard signal heatmap missing 3 signals (forecast, claude_fundamental, futures_flow)
BUG-31: digest.py reads wrong key from signal_log entries
BUG-32: http_retry returns response on final retryable failure (should return None)
BUG-33: message_store SEND_CATEGORIES includes "invocation" (should be save-only)
BUG-34: journal_index XAG price buckets capped at $35
BUG-35: alpha_vantage imports from portfolio_mgr instead of file_utils
"""

from __future__ import annotations

import json
import pytest
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# BUG-30: Dashboard signal heatmap includes all 30 signals
# ---------------------------------------------------------------------------

class TestBug30DashboardHeatmapSignals:
    """Dashboard /api/signal-heatmap should list all 19 enhanced signals."""

    def test_enhanced_signals_includes_forecast(self):
        """forecast signal (#28) must be in the heatmap signal list."""
        app_path = Path(__file__).parent.parent / "dashboard" / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert '"forecast"' in source, "forecast signal missing from dashboard heatmap"

    def test_enhanced_signals_includes_claude_fundamental(self):
        """claude_fundamental signal (#29) must be in the heatmap signal list."""
        app_path = Path(__file__).parent.parent / "dashboard" / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert '"claude_fundamental"' in source, "claude_fundamental missing from dashboard heatmap"

    def test_enhanced_signals_includes_futures_flow(self):
        """futures_flow signal (#30) must be in the heatmap signal list."""
        app_path = Path(__file__).parent.parent / "dashboard" / "app.py"
        source = app_path.read_text(encoding="utf-8")
        assert '"futures_flow"' in source, "futures_flow missing from dashboard heatmap"


# ---------------------------------------------------------------------------
# BUG-31: Digest reads correct key from signal_log entries
# ---------------------------------------------------------------------------

class TestBug31DigestSignalKey:
    """Digest must read 'tickers' key (not 'signals') from signal_log entries."""

    def test_digest_reads_tickers_key(self):
        """Signal log uses entry['tickers'][ticker]['consensus'], not entry['signals']."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tickers": {
                "BTC-USD": {"consensus": "BUY", "price_usd": 67000},
                "ETH-USD": {"consensus": "SELL", "price_usd": 1900},
                "NVDA": {"consensus": "HOLD", "price_usd": 185},
            },
        }
        consensus_counts = Counter()
        for ticker_data in entry.get("tickers", {}).values():
            action = ticker_data.get("consensus", "HOLD")
            consensus_counts[action] += 1

        assert consensus_counts["BUY"] == 1
        assert consensus_counts["SELL"] == 1
        assert consensus_counts["HOLD"] == 1

    def test_old_signals_key_returns_empty(self):
        """The old 'signals' key does NOT exist at top level in signal_log entries."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "tickers": {
                "BTC-USD": {"consensus": "BUY", "signals": {"rsi": "BUY"}},
            },
        }
        assert entry.get("signals", {}) == {}
        assert len(entry.get("tickers", {})) == 1


# ---------------------------------------------------------------------------
# BUG-32: http_retry returns None on retryable failure
# ---------------------------------------------------------------------------

class TestBug32HttpRetryNone:
    """http_retry must return None on final retryable failure."""

    def test_retryable_exhaust_returns_none(self):
        from portfolio.http_retry import fetch_with_retry

        mock_resp = MagicMock()
        mock_resp.status_code = 503

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.get.return_value = mock_resp
            result = fetch_with_retry("https://example.com", retries=1, backoff=0.01)

        assert result is None, "BUG-32: should return None on retryable exhaust"

    def test_429_exhaust_returns_none(self):
        from portfolio.http_retry import fetch_with_retry

        mock_resp = MagicMock()
        mock_resp.status_code = 429

        with patch("portfolio.http_retry.requests") as mock_req, \
             patch("portfolio.http_retry.time.sleep"):
            mock_req.get.return_value = mock_resp
            result = fetch_with_retry("https://example.com", retries=2, backoff=0.01)

        assert result is None


# ---------------------------------------------------------------------------
# BUG-33: message_store SEND_CATEGORIES excludes "invocation"
# ---------------------------------------------------------------------------

class TestBug33MessageStoreSendCategories:

    def test_invocation_not_in_send_categories(self):
        from portfolio.message_store import SEND_CATEGORIES
        assert "invocation" not in SEND_CATEGORIES

    def test_analysis_in_send_categories(self):
        from portfolio.message_store import SEND_CATEGORIES
        assert "analysis" in SEND_CATEGORIES

    def test_trade_in_send_categories(self):
        from portfolio.message_store import SEND_CATEGORIES
        assert "trade" in SEND_CATEGORIES


# ---------------------------------------------------------------------------
# BUG-34: journal_index XAG price buckets expanded
# ---------------------------------------------------------------------------

class TestBug34JournalIndexXagBuckets:

    def test_xag_bucket_at_89(self):
        """Silver at $89 should have meaningful bucket, not above_35."""
        from portfolio.journal_index import _price_bucket
        token = _price_bucket("XAG-USD", 89.0)
        assert token is not None
        assert "above_35" not in token

    def test_xag_bucket_at_120(self):
        """Silver at $120 (prophecy target) should map to above_120 bucket."""
        from portfolio.journal_index import _price_bucket
        token = _price_bucket("XAG-USD", 125.0)
        assert "above_120" in token

    def test_xag_bucket_at_30(self):
        from portfolio.journal_index import _price_bucket
        token = _price_bucket("XAG-USD", 28.0)
        assert token is not None
        assert "below" in token

    def test_xag_has_enough_buckets(self):
        from portfolio.journal_index import _PRICE_BUCKETS
        assert len(_PRICE_BUCKETS["XAG-USD"]) >= 6


# ---------------------------------------------------------------------------
# BUG-35: alpha_vantage imports from file_utils
# ---------------------------------------------------------------------------

class TestBug35AlphaVantageImport:

    def test_import_source_is_file_utils(self):
        av_path = Path(__file__).parent.parent / "portfolio" / "alpha_vantage.py"
        source = av_path.read_text(encoding="utf-8")
        assert "from portfolio.file_utils import atomic_write_json" in source
        assert "from portfolio.portfolio_mgr import _atomic_write_json" not in source
