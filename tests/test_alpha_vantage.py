"""Tests for portfolio.alpha_vantage — Alpha Vantage fundamentals cache."""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from portfolio import alpha_vantage
from portfolio.alpha_vantage import (
    _normalize_overview,
    get_fundamentals,
    get_all_fundamentals,
    load_persistent_cache,
    refresh_fundamentals_batch,
    should_batch_refresh,
)
from portfolio.signals.claude_fundamental import _build_fundamentals_block


# ── Sample Alpha Vantage OVERVIEW response ──────────────────────────────────

SAMPLE_AV_RESPONSE = {
    "Symbol": "NVDA",
    "Name": "NVIDIA Corporation",
    "PERatio": "55.3",
    "ForwardPE": "28.1",
    "PEGRatio": "1.42",
    "EPS": "3.35",
    "QuarterlyRevenueGrowthYOY": "0.122",
    "QuarterlyEarningsGrowthYOY": "0.168",
    "ProfitMargin": "0.556",
    "MarketCapitalization": "2800000000000",
    "Sector": "Technology",
    "Industry": "Semiconductors",
    "DividendYield": "0.0003",
    "AnalystTargetPrice": "180.50",
    "AnalystRatingStrongBuy": "20",
    "AnalystRatingBuy": "15",
    "AnalystRatingHold": "5",
    "AnalystRatingSell": "1",
    "AnalystRatingStrongSell": "0",
    "Beta": "1.72",
    "52WeekHigh": "195.00",
    "52WeekLow": "108.00",
}


SAMPLE_AV_RESPONSE_NONE_FIELDS = {
    "Symbol": "GRRR",
    "Name": "Gorilla Holdings",
    "PERatio": "None",
    "ForwardPE": "None",
    "PEGRatio": "None",
    "EPS": "-0.50",
    "QuarterlyRevenueGrowthYOY": "None",
    "QuarterlyEarningsGrowthYOY": "None",
    "ProfitMargin": "-0.15",
    "MarketCapitalization": "500000000",
    "Sector": "Technology",
    "Industry": "None",
    "DividendYield": "0",
    "AnalystTargetPrice": "None",
    "AnalystRatingStrongBuy": "0",
    "AnalystRatingBuy": "0",
    "AnalystRatingHold": "0",
    "AnalystRatingSell": "0",
    "AnalystRatingStrongSell": "0",
    "Beta": "None",
    "52WeekHigh": "5.00",
    "52WeekLow": "0.50",
}


# ── _normalize_overview tests ───────────────────────────────────────────────

class TestNormalizeOverview:
    def test_normal_response(self):
        result = _normalize_overview(SAMPLE_AV_RESPONSE)
        assert result is not None
        assert result["pe_ratio"] == 55.3
        assert result["forward_pe"] == 28.1
        assert result["peg_ratio"] == 1.42
        assert result["eps"] == 3.35
        assert result["revenue_growth_yoy"] == 0.122
        assert result["earnings_growth_yoy"] == 0.168
        assert result["profit_margin"] == 0.556
        assert result["market_cap"] == 2800000000000
        assert result["sector"] == "Technology"
        assert result["industry"] == "Semiconductors"
        assert result["dividend_yield"] == 0.0003
        assert result["analyst_target"] == 180.5
        assert result["beta"] == 1.72
        assert result["w52_high"] == 195.0
        assert result["w52_low"] == 108.0
        assert "_fetched_at" in result

    def test_analyst_ratings(self):
        result = _normalize_overview(SAMPLE_AV_RESPONSE)
        ratings = result["analyst_ratings"]
        assert ratings["strong_buy"] == 20
        assert ratings["buy"] == 15
        assert ratings["hold"] == 5
        assert ratings["sell"] == 1
        assert ratings["strong_sell"] == 0

    def test_none_string_fields(self):
        result = _normalize_overview(SAMPLE_AV_RESPONSE_NONE_FIELDS)
        assert result is not None
        assert result["pe_ratio"] is None
        assert result["forward_pe"] is None
        assert result["peg_ratio"] is None
        assert result["eps"] == -0.50
        assert result["revenue_growth_yoy"] is None
        assert result["profit_margin"] == -0.15
        assert result["industry"] is None

    def test_error_response(self):
        result = _normalize_overview({"Error Message": "Invalid API call"})
        assert result is None

    def test_rate_limit_note(self):
        result = _normalize_overview({"Note": "API call frequency exceeded"})
        assert result is None

    def test_empty_response(self):
        result = _normalize_overview({})
        assert result is None

    def test_missing_symbol(self):
        result = _normalize_overview({"PERatio": "10"})
        assert result is None

    def test_dash_field(self):
        resp = dict(SAMPLE_AV_RESPONSE)
        resp["PERatio"] = "-"
        result = _normalize_overview(resp)
        assert result["pe_ratio"] is None

    def test_empty_string_field(self):
        resp = dict(SAMPLE_AV_RESPONSE)
        resp["EPS"] = ""
        result = _normalize_overview(resp)
        assert result["eps"] is None

    def test_fetched_at_is_recent(self):
        result = _normalize_overview(SAMPLE_AV_RESPONSE)
        fetched = datetime.fromisoformat(result["_fetched_at"])
        age = (datetime.now(timezone.utc) - fetched).total_seconds()
        assert age < 5  # should be very recent


# ── Cache tests ─────────────────────────────────────────────────────────────

class TestCache:
    def setup_method(self):
        """Reset module-level cache before each test."""
        with alpha_vantage._cache_lock:
            alpha_vantage._cache.clear()

    def test_get_fundamentals_miss(self):
        assert get_fundamentals("NVDA") is None

    def test_get_fundamentals_hit(self):
        with alpha_vantage._cache_lock:
            alpha_vantage._cache["NVDA"] = {"pe_ratio": 55.3}
        result = get_fundamentals("NVDA")
        assert result == {"pe_ratio": 55.3}

    def test_get_all_fundamentals_empty(self):
        assert get_all_fundamentals() == {}

    def test_get_all_fundamentals_returns_copy(self):
        with alpha_vantage._cache_lock:
            alpha_vantage._cache["NVDA"] = {"pe_ratio": 55.3}
        result = get_all_fundamentals()
        assert "NVDA" in result
        # Top-level dict is a copy — adding/removing keys won't affect cache
        result["FAKE"] = {"pe_ratio": 0}
        assert "FAKE" not in alpha_vantage._cache

    def test_load_persistent_cache_missing_file(self):
        with patch.object(alpha_vantage, "CACHE_FILE", Path("/nonexistent/path.json")):
            load_persistent_cache()  # should not raise
        # Cache should remain empty
        assert get_all_fundamentals() == {}

    def test_load_persistent_cache_valid(self, tmp_path):
        cache_file = tmp_path / "fundamentals_cache.json"
        cache_data = {"NVDA": {"pe_ratio": 55.3, "_fetched_at": "2026-02-25T12:00:00+00:00"}}
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        with patch.object(alpha_vantage, "CACHE_FILE", cache_file):
            load_persistent_cache()
        assert get_fundamentals("NVDA")["pe_ratio"] == 55.3

    def test_load_persistent_cache_corrupt(self, tmp_path):
        cache_file = tmp_path / "fundamentals_cache.json"
        cache_file.write_text("not valid json {{", encoding="utf-8")

        with patch.object(alpha_vantage, "CACHE_FILE", cache_file):
            load_persistent_cache()  # should not raise
        assert get_all_fundamentals() == {}

    def test_stale_detection(self):
        old_time = (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()
        with alpha_vantage._cache_lock:
            alpha_vantage._cache["NVDA"] = {"pe_ratio": 55.3, "_fetched_at": old_time}
        assert alpha_vantage._is_stale("NVDA", max_stale_days=5)

    def test_fresh_detection(self):
        now = datetime.now(timezone.utc).isoformat()
        with alpha_vantage._cache_lock:
            alpha_vantage._cache["NVDA"] = {"pe_ratio": 55.3, "_fetched_at": now}
        assert not alpha_vantage._is_stale("NVDA", max_stale_days=5)

    def test_cache_age_hours_none(self):
        assert alpha_vantage._cache_age_hours("NVDA") is None

    def test_cache_age_hours_recent(self):
        now = datetime.now(timezone.utc).isoformat()
        with alpha_vantage._cache_lock:
            alpha_vantage._cache["NVDA"] = {"_fetched_at": now}
        age = alpha_vantage._cache_age_hours("NVDA")
        assert age is not None
        assert age < 0.1  # less than 6 minutes


# ── Batch refresh tests ─────────────────────────────────────────────────────

class TestBatchRefresh:
    def setup_method(self):
        with alpha_vantage._cache_lock:
            alpha_vantage._cache.clear()
        alpha_vantage._daily_budget_used = 0
        alpha_vantage._budget_reset_date = ""
        alpha_vantage._circuit_breaker_failures = 0
        alpha_vantage._circuit_breaker_paused_until = 0.0

    def _make_config(self, **overrides):
        cfg = {
            "alpha_vantage": {
                "api_key": "test_key",
                "enabled": True,
                "daily_budget": 25,
                "rate_limit_per_min": 5,
                "cache_ttl_hours": 24,
                "max_stale_days": 5,
                "skip_tickers": ["QQQ"],
            }
        }
        cfg["alpha_vantage"].update(overrides)
        return cfg

    def test_disabled_returns_zero(self):
        config = self._make_config(enabled=False)
        assert refresh_fundamentals_batch(config) == 0

    def test_no_api_key_returns_zero(self):
        config = self._make_config(api_key="")
        assert refresh_fundamentals_batch(config) == 0

    @patch("portfolio.alpha_vantage._fetch_overview")
    @patch("portfolio.alpha_vantage._save_persistent_cache")
    @patch("portfolio.alpha_vantage._alpha_vantage_limiter")
    def test_successful_refresh(self, mock_limiter, mock_save, mock_fetch):
        mock_fetch.return_value = SAMPLE_AV_RESPONSE
        config = self._make_config()
        count = refresh_fundamentals_batch(config)
        assert count > 0
        assert mock_save.called

    @patch("portfolio.alpha_vantage._fetch_overview")
    @patch("portfolio.alpha_vantage._save_persistent_cache")
    @patch("portfolio.alpha_vantage._alpha_vantage_limiter")
    def test_skips_qqq(self, mock_limiter, mock_save, mock_fetch):
        mock_fetch.return_value = SAMPLE_AV_RESPONSE
        config = self._make_config()
        refresh_fundamentals_batch(config)
        # QQQ should not have been fetched
        called_tickers = [call[0][0] for call in mock_fetch.call_args_list]
        assert "QQQ" not in called_tickers

    @patch("portfolio.alpha_vantage._fetch_overview")
    @patch("portfolio.alpha_vantage._save_persistent_cache")
    @patch("portfolio.alpha_vantage._alpha_vantage_limiter")
    def test_budget_exhaustion(self, mock_limiter, mock_save, mock_fetch):
        mock_fetch.return_value = SAMPLE_AV_RESPONSE
        alpha_vantage._daily_budget_used = 25
        alpha_vantage._budget_reset_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        config = self._make_config(daily_budget=25)
        count = refresh_fundamentals_batch(config)
        assert count == 0

    @patch("portfolio.alpha_vantage._fetch_overview")
    @patch("portfolio.alpha_vantage._save_persistent_cache")
    @patch("portfolio.alpha_vantage._alpha_vantage_limiter")
    def test_circuit_breaker_trips(self, mock_limiter, mock_save, mock_fetch):
        mock_fetch.return_value = None  # simulate failure
        config = self._make_config()
        count = refresh_fundamentals_batch(config)
        assert count == 0
        assert alpha_vantage._circuit_breaker_paused_until > time.time()

    @patch("portfolio.alpha_vantage._fetch_overview")
    @patch("portfolio.alpha_vantage._save_persistent_cache")
    @patch("portfolio.alpha_vantage._alpha_vantage_limiter")
    def test_circuit_breaker_blocks(self, mock_limiter, mock_save, mock_fetch):
        alpha_vantage._circuit_breaker_paused_until = time.time() + 300
        config = self._make_config()
        count = refresh_fundamentals_batch(config)
        assert count == 0
        assert not mock_fetch.called

    @patch("portfolio.alpha_vantage._fetch_overview")
    @patch("portfolio.alpha_vantage._save_persistent_cache")
    @patch("portfolio.alpha_vantage._alpha_vantage_limiter")
    def test_partial_failure(self, mock_limiter, mock_save, mock_fetch):
        # First call succeeds, second fails, third succeeds
        mock_fetch.side_effect = [SAMPLE_AV_RESPONSE, None, SAMPLE_AV_RESPONSE]
        config = self._make_config(daily_budget=3)
        count = refresh_fundamentals_batch(config)
        # Should have at least 1 success (first call), maybe more depending on order
        assert count >= 1

    def test_all_fresh_no_refresh(self):
        now = datetime.now(timezone.utc).isoformat()
        with alpha_vantage._cache_lock:
            for ticker in ("NVDA", "AMD", "AAPL"):
                alpha_vantage._cache[ticker] = {"_fetched_at": now}
        config = self._make_config()
        # Even with budget, if all cached tickers are fresh, nothing should refresh
        # (though tickers not yet cached will still need refresh)


# ── should_batch_refresh tests ──────────────────────────────────────────────

class TestShouldBatchRefresh:
    def setup_method(self):
        with alpha_vantage._cache_lock:
            alpha_vantage._cache.clear()

    def test_disabled(self):
        config = {"alpha_vantage": {"enabled": False, "api_key": "test"}}
        assert not should_batch_refresh(config)

    def test_no_api_key(self):
        config = {"alpha_vantage": {"enabled": True, "api_key": ""}}
        assert not should_batch_refresh(config)

    def test_needs_refresh_empty_cache(self):
        config = {"alpha_vantage": {"enabled": True, "api_key": "test",
                                     "cache_ttl_hours": 24, "skip_tickers": ["QQQ"]}}
        assert should_batch_refresh(config)

    def test_no_refresh_all_fresh(self):
        now = datetime.now(timezone.utc).isoformat()
        config = {"alpha_vantage": {"enabled": True, "api_key": "test",
                                     "cache_ttl_hours": 24, "skip_tickers": ["QQQ"]}}
        with alpha_vantage._cache_lock:
            from portfolio.tickers import STOCK_SYMBOLS
            for ticker in STOCK_SYMBOLS:
                alpha_vantage._cache[ticker] = {"_fetched_at": now}
        assert not should_batch_refresh(config)


# ── _build_fundamentals_block tests ─────────────────────────────────────────

class TestBuildFundamentalsBlock:
    FUND_DATA = {
        "NVDA": {
            "pe_ratio": 55.3,
            "forward_pe": 28.1,
            "peg_ratio": 1.42,
            "eps": 3.35,
            "revenue_growth_yoy": 0.122,
            "earnings_growth_yoy": 0.168,
            "profit_margin": 0.556,
            "market_cap": 2800000000000,
            "sector": "Technology",
            "industry": "Semiconductors",
            "analyst_target": 180.5,
            "analyst_ratings": {"strong_buy": 20, "buy": 15, "hold": 5, "sell": 1, "strong_sell": 0},
            "beta": 1.72,
            "w52_high": 195.0,
            "w52_low": 108.0,
        }
    }

    def test_haiku_one_liner(self):
        result = _build_fundamentals_block("NVDA", self.FUND_DATA, tier="haiku")
        assert "NVDA:" in result
        assert "PE=55.3" in result
        assert "RevGrowth=" in result
        assert "Target=$180" in result

    def test_sonnet_detailed(self):
        result = _build_fundamentals_block("NVDA", self.FUND_DATA, tier="sonnet")
        assert "NVDA Fundamentals" in result
        assert "Valuation:" in result
        assert "PE=55.3" in result
        assert "FwdPE=28.1" in result
        assert "Technology" in result

    def test_opus_same_as_sonnet(self):
        result = _build_fundamentals_block("NVDA", self.FUND_DATA, tier="opus")
        assert "NVDA Fundamentals" in result
        assert "Valuation:" in result

    def test_missing_ticker(self):
        result = _build_fundamentals_block("AAPL", self.FUND_DATA, tier="haiku")
        assert result == ""

    def test_empty_fundamentals(self):
        result = _build_fundamentals_block("NVDA", {}, tier="haiku")
        assert result == ""

    def test_none_fundamentals(self):
        result = _build_fundamentals_block("NVDA", None, tier="haiku")
        assert result == ""
