"""Tests for portfolio.news_keywords — keyword dictionaries, scoring, sector maps."""

import pytest

from portfolio.news_keywords import (
    score_headline,
    keyword_severity,
    is_credible_source,
    get_sector_impact,
    SECTOR_MAP,
    TICKER_SECTORS,
    CRITICAL_KEYWORDS,
    HIGH_KEYWORDS,
    MODERATE_KEYWORDS,
    ALL_KEYWORDS,
    CREDIBLE_SOURCES,
)


class TestScoreHeadline:
    def test_empty_title(self):
        weight, matched = score_headline("")
        assert weight == 1.0
        assert matched == []

    def test_none_title(self):
        weight, matched = score_headline(None)
        assert weight == 1.0
        assert matched == []

    def test_no_keywords(self):
        weight, matched = score_headline("Stock market ends flat on quiet trading day")
        assert weight == 1.0
        assert matched == []

    def test_critical_keyword_tariff(self):
        weight, matched = score_headline("Trump threatens new tariff on Chinese imports")
        assert weight == 3.0
        assert "tariff" in matched

    def test_critical_keyword_war(self):
        weight, matched = score_headline("War breaks out in region, markets tumble")
        assert weight == 3.0
        assert "war" in matched

    def test_critical_keyword_crash(self):
        weight, matched = score_headline("Market crash feared as volatility spikes")
        assert weight == 3.0
        assert "crash" in matched

    def test_high_keyword_recession(self):
        weight, matched = score_headline("Economists warn of recession risk")
        assert weight == 2.0
        assert "recession" in matched

    def test_high_keyword_cpi(self):
        weight, matched = score_headline("CPI data comes in hot, above expectations")
        assert weight == 2.0
        assert "cpi" in matched

    def test_moderate_keyword_merger(self):
        weight, matched = score_headline("Big tech merger announced, deal worth billions")
        assert weight == 1.5
        assert "merger" in matched

    def test_multiple_keywords(self):
        weight, matched = score_headline("Trade war tariffs cause crash fears in semiconductor sector")
        assert weight == 3.0
        assert len(matched) >= 2  # tariff(s) + crash + trade war

    def test_case_insensitive(self):
        weight, matched = score_headline("TARIFF announcement shocks MARKETS")
        assert weight == 3.0

    def test_word_boundary(self):
        # "warfare" should not match "war" at boundary
        weight, matched = score_headline("Software development continues")
        assert weight == 1.0
        assert "war" not in matched


class TestKeywordSeverity:
    def test_critical(self):
        assert keyword_severity("New tariffs announced") == "critical"

    def test_high(self):
        assert keyword_severity("Recession fears grow") == "high"

    def test_moderate(self):
        assert keyword_severity("ETF approval expected") == "moderate"

    def test_normal(self):
        assert keyword_severity("Markets trade sideways today") == "normal"


class TestIsCredibleSource:
    def test_reuters(self):
        assert is_credible_source("Reuters") is True

    def test_bloomberg(self):
        assert is_credible_source("Bloomberg") is True

    def test_cnbc(self):
        assert is_credible_source("CNBC Finance") is True

    def test_unknown_blog(self):
        assert is_credible_source("CryptoMoonBlog") is False

    def test_empty_source(self):
        assert is_credible_source("") is False

    def test_none_source(self):
        assert is_credible_source(None) is False

    def test_case_insensitive(self):
        assert is_credible_source("REUTERS") is True


class TestGetSectorImpact:
    def test_tariff_semiconductor(self):
        assert get_sector_impact("tariff", "NVDA") == "SELL"

    def test_tariff_china(self):
        assert get_sector_impact("tariff", "BABA") == "SELL"

    def test_tariff_metals(self):
        assert get_sector_impact("tariff", "XAU-USD") == "BUY"

    def test_rate_cut_crypto(self):
        assert get_sector_impact("rate cut", "BTC-USD") == "BUY"

    def test_no_impact(self):
        assert get_sector_impact("tariff", "LMT") is None

    def test_unknown_keyword(self):
        assert get_sector_impact("unicorn", "NVDA") is None

    def test_hack_crypto(self):
        assert get_sector_impact("hack", "BTC-USD") == "SELL"

    def test_war_defense(self):
        assert get_sector_impact("war", "LMT") == "BUY"


class TestSectorMaps:
    def test_sector_map_not_empty(self):
        assert len(SECTOR_MAP) > 0

    def test_semiconductor_has_nvda(self):
        assert "NVDA" in SECTOR_MAP["semiconductor"]

    def test_crypto_has_btc(self):
        assert "BTC-USD" in SECTOR_MAP["crypto"]

    def test_ticker_sectors_reverse_map(self):
        assert "semiconductor" in TICKER_SECTORS["NVDA"]

    def test_all_tickers_in_some_sector(self):
        # Not all tickers must be in sectors, but key ones should
        assert "NVDA" in TICKER_SECTORS
        assert "BTC-USD" in TICKER_SECTORS
        assert "LMT" in TICKER_SECTORS

    def test_all_keywords_combines_tiers(self):
        for kw in CRITICAL_KEYWORDS:
            assert kw in ALL_KEYWORDS
        for kw in HIGH_KEYWORDS:
            assert kw in ALL_KEYWORDS
        for kw in MODERATE_KEYWORDS:
            assert kw in ALL_KEYWORDS
