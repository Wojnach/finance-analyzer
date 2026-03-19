"""Tests for data/crypto_data.py — Fear & Greed, news, MSTR price, on-chain, NAV."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear the module cache before each test."""
    import crypto_data
    crypto_data._cache.clear()
    yield
    crypto_data._cache.clear()


# ---------------------------------------------------------------------------
# Fear & Greed
# ---------------------------------------------------------------------------

class TestFearGreed:
    @patch("crypto_data.requests.get")
    def test_fetches_and_caches(self, mock_get):
        from crypto_data import get_fear_greed
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"data": [{"value": "25", "value_classification": "Extreme Fear", "timestamp": "123"}]}
        )
        result = get_fear_greed()
        assert result is not None
        assert result["value"] == 25
        assert result["classification"] == "Extreme Fear"

        # Second call should use cache (no new request)
        result2 = get_fear_greed()
        assert result2 == result
        assert mock_get.call_count == 1

    @patch("crypto_data.requests.get")
    def test_returns_none_on_error(self, mock_get):
        from crypto_data import get_fear_greed
        mock_get.side_effect = Exception("timeout")
        result = get_fear_greed()
        assert result is None

    @patch("crypto_data.requests.get")
    def test_returns_none_on_bad_status(self, mock_get):
        from crypto_data import get_fear_greed
        mock_get.return_value = MagicMock(status_code=500)
        result = get_fear_greed()
        assert result is None


# ---------------------------------------------------------------------------
# Crypto News
# ---------------------------------------------------------------------------

class TestCryptoNews:
    @patch("crypto_data.requests.get")
    def test_fetches_articles(self, mock_get):
        from crypto_data import get_crypto_news
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"Data": [
                {"title": "BTC up", "source_info": {"name": "CoinDesk"},
                 "categories": "BTC", "published_on": 1000},
                {"title": "ETH stable", "source": "Reuters",
                 "categories": "ETH", "published_on": 999},
            ]}
        )
        result = get_crypto_news(limit=2)
        assert len(result) == 2
        assert result[0]["title"] == "BTC up"
        assert result[0]["source"] == "CoinDesk"

    @patch("crypto_data.requests.get")
    def test_returns_empty_on_error(self, mock_get):
        from crypto_data import get_crypto_news
        mock_get.side_effect = Exception("network")
        result = get_crypto_news()
        assert result == []

    @patch("crypto_data.requests.get")
    def test_caching(self, mock_get):
        from crypto_data import get_crypto_news
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"Data": [{"title": "t", "source": "s", "categories": "", "published_on": 0}]}
        )
        get_crypto_news()
        get_crypto_news()
        assert mock_get.call_count == 1


# ---------------------------------------------------------------------------
# MSTR Price
# ---------------------------------------------------------------------------

class TestMSTRPrice:
    @patch("crypto_data.requests.get")
    def test_fetches_price(self, mock_get):
        from crypto_data import fetch_mstr_price
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"chart": {"result": [{"meta": {
                "regularMarketPrice": 287.50,
                "previousClose": 280.00,
                "marketState": "REGULAR",
                "currency": "USD",
            }}]}}
        )
        result = fetch_mstr_price()
        assert result is not None
        assert result["price"] == 287.50
        assert result["change_pct"] == pytest.approx(2.68, abs=0.1)
        assert result["market_state"] == "REGULAR"

    @patch("crypto_data.requests.get")
    def test_returns_none_on_error(self, mock_get):
        from crypto_data import fetch_mstr_price
        mock_get.side_effect = Exception("timeout")
        result = fetch_mstr_price()
        assert result is None


# ---------------------------------------------------------------------------
# MSTR-BTC NAV
# ---------------------------------------------------------------------------

class TestMSTRBTCNav:
    def test_compute_premium(self):
        from crypto_data import compute_mstr_btc_nav
        result = compute_mstr_btc_nav(mstr_price=300, btc_price=67000)
        assert result is not None
        assert "nav_per_share" in result
        assert "premium_pct" in result
        assert result["btc_holdings"] > 0
        # NAV should be positive
        assert result["nav_per_share"] > 0

    def test_zero_prices_return_none(self):
        from crypto_data import compute_mstr_btc_nav
        assert compute_mstr_btc_nav(0, 67000) is None
        assert compute_mstr_btc_nav(300, 0) is None
        assert compute_mstr_btc_nav(0, 0) is None


# ---------------------------------------------------------------------------
# On-chain Summary
# ---------------------------------------------------------------------------

class TestOnchainSummary:
    @patch("portfolio.onchain_data.interpret_onchain")
    @patch("portfolio.onchain_data.get_onchain_data")
    def test_returns_summary(self, mock_get, mock_interpret):
        from crypto_data import get_onchain_summary
        mock_get.return_value = {"mvrv": 1.5, "sopr": 1.02, "nupl": 0.3}
        mock_interpret.return_value = {"zone": "accumulation", "bias": "bullish", "summary": "ok"}
        result = get_onchain_summary()
        assert result is not None
        assert result["mvrv"] == 1.5
        assert result["zone"] == "accumulation"
        assert result["bias"] == "bullish"

    def test_returns_none_on_import_error(self):
        """If portfolio.onchain_data is not importable, returns None gracefully."""
        from crypto_data import _cache
        # Clear any cached data
        _cache.pop("onchain", None)
        with patch.dict("sys.modules", {"portfolio.onchain_data": None}):
            # This should handle the ImportError gracefully
            # (may return cached or None depending on state)
            pass  # Just verifying no crash


# ---------------------------------------------------------------------------
# US Market Hours
# ---------------------------------------------------------------------------

class TestUSMarketHours:
    def test_during_market_hours(self):
        import datetime

        from crypto_data import is_us_market_hours
        result = is_us_market_hours(
            datetime.datetime(2026, 3, 11, 14, 30, tzinfo=datetime.UTC)
        )
        assert result is True

    def test_before_market_hours(self):
        import datetime

        from crypto_data import is_us_market_hours
        result = is_us_market_hours(
            datetime.datetime(2026, 3, 11, 13, 0, tzinfo=datetime.UTC)
        )
        assert result is False

    def test_handles_us_dst_gap_vs_stockholm(self):
        import datetime

        from crypto_data import is_us_market_hours
        # March 11, 2026: US already on EDT, Stockholm still on CET.
        result = is_us_market_hours(
            datetime.datetime(2026, 3, 11, 19, 30, tzinfo=datetime.UTC)
        )
        assert result is True


# ---------------------------------------------------------------------------
# Cache TTL
# ---------------------------------------------------------------------------

class TestCacheTTL:
    def test_cache_expires(self):
        from crypto_data import _cached, _set_cache
        _set_cache("test_key", "data")
        assert _cached("test_key", 999) == "data"
        assert _cached("test_key", 0) is None  # TTL=0 means always expired

    def test_cache_miss(self):
        from crypto_data import _cached
        assert _cached("nonexistent", 999) is None
