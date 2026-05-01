"""Tests for portfolio/funding_rate.py.

Adversarial review 04-29 DE-P1-1: `_fetch_funding_rate` indexes into the
Binance FAPI premiumIndex response without `.get()` guards. If Binance ever
returns a response missing `lastFundingRate` or `markPrice` (partial
deployment, schema change, weird symbol state), the worker thread raises
`KeyError` and the funding signal disappears for the cycle. The fix is
defensive `.get()` access with a None return on missing fields.
"""

from unittest.mock import patch

import portfolio.funding_rate as fr
from portfolio.funding_rate import _fetch_funding_rate, get_funding_rate


def _clear_cache():
    """Drop cached funding rate entries so each test starts fresh."""
    from portfolio.shared_state import _tool_cache
    keys = [k for k in list(_tool_cache.keys()) if isinstance(k, str) and k.startswith("funding_rate_")]
    for k in keys:
        _tool_cache.pop(k, None)


# ---------------------------------------------------------------------------
# Happy path (regression — make sure the .get() refactor didn't break valid responses)
# ---------------------------------------------------------------------------

class TestHappyPath:

    def test_btc_funding_rate_normal_response(self):
        _clear_cache()
        fake_response = {
            "lastFundingRate": "0.0001",
            "markPrice": "65000.0",
        }
        with patch("portfolio.funding_rate.fetch_json", return_value=fake_response):
            result = _fetch_funding_rate("BTC-USD")
        assert result is not None
        assert result["rate"] == 0.0001
        assert result["mark_price"] == 65000.0
        assert result["action"] == "HOLD"  # 0.01% is normal

    def test_high_funding_returns_sell(self):
        _clear_cache()
        fake_response = {
            "lastFundingRate": "0.0005",  # 0.05% — overleveraged longs
            "markPrice": "65000.0",
        }
        with patch("portfolio.funding_rate.fetch_json", return_value=fake_response):
            result = _fetch_funding_rate("BTC-USD")
        assert result["action"] == "SELL"

    def test_negative_funding_returns_buy(self):
        _clear_cache()
        fake_response = {
            "lastFundingRate": "-0.0002",  # overleveraged shorts
            "markPrice": "65000.0",
        }
        with patch("portfolio.funding_rate.fetch_json", return_value=fake_response):
            result = _fetch_funding_rate("BTC-USD")
        assert result["action"] == "BUY"


# ---------------------------------------------------------------------------
# Defensive: missing keys must return None (not raise KeyError)
# ---------------------------------------------------------------------------

class TestMissingFields:

    def test_missing_last_funding_rate_returns_none(self):
        """Binance partial response with no `lastFundingRate` must not crash."""
        _clear_cache()
        # markPrice present, lastFundingRate absent
        fake_response = {"markPrice": "65000.0", "symbol": "BTCUSDT"}
        with patch("portfolio.funding_rate.fetch_json", return_value=fake_response):
            result = _fetch_funding_rate("BTC-USD")
        assert result is None, "Missing lastFundingRate must return None, not KeyError"

    def test_missing_mark_price_returns_none(self):
        """Missing markPrice must not crash either."""
        _clear_cache()
        fake_response = {"lastFundingRate": "0.0001", "symbol": "BTCUSDT"}
        with patch("portfolio.funding_rate.fetch_json", return_value=fake_response):
            result = _fetch_funding_rate("BTC-USD")
        assert result is None, "Missing markPrice must return None, not KeyError"

    def test_empty_response_returns_none(self):
        """Empty dict must not crash."""
        _clear_cache()
        with patch("portfolio.funding_rate.fetch_json", return_value={}):
            result = _fetch_funding_rate("BTC-USD")
        assert result is None

    def test_none_response_returns_none(self):
        """fetch_json returning None must propagate to None."""
        _clear_cache()
        with patch("portfolio.funding_rate.fetch_json", return_value=None):
            result = _fetch_funding_rate("BTC-USD")
        assert result is None

    def test_invalid_funding_rate_value_returns_none(self):
        """Non-numeric lastFundingRate must not propagate ValueError."""
        _clear_cache()
        fake_response = {"lastFundingRate": "n/a", "markPrice": "65000.0"}
        with patch("portfolio.funding_rate.fetch_json", return_value=fake_response):
            result = _fetch_funding_rate("BTC-USD")
        assert result is None

    def test_invalid_mark_price_value_returns_none(self):
        """Non-numeric markPrice must not propagate ValueError."""
        _clear_cache()
        fake_response = {"lastFundingRate": "0.0001", "markPrice": "unavailable"}
        with patch("portfolio.funding_rate.fetch_json", return_value=fake_response):
            result = _fetch_funding_rate("BTC-USD")
        assert result is None


# ---------------------------------------------------------------------------
# Public API smoke
# ---------------------------------------------------------------------------

class TestPublicAPI:

    def test_unknown_ticker_returns_none(self):
        """Tickers not in SYMBOL_MAP return None without an API call."""
        with patch("portfolio.funding_rate.fetch_json") as mock_fetch:
            result = get_funding_rate("SOL-USD")
        assert result is None
        mock_fetch.assert_not_called()

    def test_get_funding_rate_uses_cache(self):
        """get_funding_rate goes through _cached helper (TTL caching)."""
        _clear_cache()
        # Patch the internal fetcher; outer get_funding_rate routes via _cached
        fake_response = {"lastFundingRate": "0.0001", "markPrice": "65000.0"}
        with patch("portfolio.funding_rate.fetch_json", return_value=fake_response) as mock_fetch:
            r1 = get_funding_rate("BTC-USD")
            r2 = get_funding_rate("BTC-USD")
        # Should only have called the API once thanks to caching
        assert r1 == r2
        assert mock_fetch.call_count == 1
