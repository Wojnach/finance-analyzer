"""Tests for the headline relevance filter introduced 2026-04-28.

Background: the sentiment pipeline used to score every headline returned by
CryptoCompare/NewsAPI/Yahoo, including price-ticker noise like "Bitcoin:
$67,123" that just averages to neutral and drowns out real catalysts.

These tests pin the filter contract:
- keyword-matched headlines (tariff, war, hack, etc.) are kept
- ticker / synonym mentions are kept
- credible-source long headlines are kept (Reuters/Bloomberg/etc.)
- bare price tickers are dropped
- if filter would drop everything, fall back to most-recent N
"""
from __future__ import annotations

from portfolio.news_keywords import is_relevant_headline
from portfolio.sentiment import _filter_relevant_headlines


def _art(title: str, source: str = "Yahoo Finance", published: str = "2026-04-28T10:00:00+00:00") -> dict:
    return {"title": title, "source": source, "published": published}


# ---------------------------------------------------------------------------
# is_relevant_headline — single-headline classifier
# ---------------------------------------------------------------------------

class TestIsRelevantHeadline:
    def test_keyword_match_kept(self):
        # tariff is a CRITICAL keyword (weight 3.0)
        assert is_relevant_headline("Trump announces 25% tariff on Chinese semis", "NVDA")

    def test_ticker_mention_kept(self):
        assert is_relevant_headline("Nvidia beats earnings on AI demand", "NVDA")

    def test_btc_synonym_kept_for_btc(self):
        assert is_relevant_headline("Bitcoin treasury company adds 500 BTC", "BTC")

    def test_eth_synonym_kept_for_eth(self):
        assert is_relevant_headline("Ethereum staking yield jumps to 5%", "ETH")

    def test_gold_synonym_kept_for_xau(self):
        assert is_relevant_headline("Gold futures hit new all-time high", "XAU")

    def test_silver_synonym_kept_for_xag(self):
        assert is_relevant_headline("Silver shortage worsens as industrial demand surges", "XAG")

    def test_credible_source_long_title_passes_via_filter(self):
        # The single-headline classifier doesn't see source — that lives in the
        # filter wrapper. But a credible-source long title that doesn't
        # otherwise match should still be considered irrelevant by THIS check.
        assert not is_relevant_headline("Stocks mixed in afternoon trade", "NVDA")

    def test_bare_price_ticker_dropped(self):
        # Common Yahoo Finance noise: short factual updates with no real content
        assert not is_relevant_headline("Bitcoin: $67,123", "BTC")
        assert not is_relevant_headline("BTC/USD 1H", "BTC")

    def test_unrelated_ticker_news_dropped(self):
        # Generic market commentary that doesn't mention our ticker
        assert not is_relevant_headline("Markets close mixed on Wednesday", "MSTR")

    def test_empty_title_returns_false(self):
        assert not is_relevant_headline("", "BTC")
        assert not is_relevant_headline("   ", "BTC")

    def test_case_insensitive_synonym(self):
        assert is_relevant_headline("BITCOIN ETF approved by SEC", "BTC")
        assert is_relevant_headline("ethereum upgrade ships next quarter", "ETH")

    def test_partial_word_does_not_match(self):
        # "Bit" should not match "BTC" — must be a real word boundary
        assert not is_relevant_headline("Bitter fight over standards body", "BTC")


# ---------------------------------------------------------------------------
# _filter_relevant_headlines — the wrapper that handles fallback
# ---------------------------------------------------------------------------

class TestFilterRelevantHeadlines:
    def test_keeps_relevant_drops_irrelevant(self):
        articles = [
            _art("Bitcoin treasury firm adds 500 BTC"),         # keep (synonym)
            _art("Markets mixed in afternoon trade"),           # drop
            _art("Tariffs on Chinese chips imposed"),           # keep (keyword)
            _art("Generic price action update"),                # drop
        ]
        kept = _filter_relevant_headlines(articles, "BTC")
        titles = [a["title"] for a in kept]
        assert "Bitcoin treasury firm adds 500 BTC" in titles
        assert "Tariffs on Chinese chips imposed" in titles
        assert "Markets mixed in afternoon trade" not in titles
        assert "Generic price action update" not in titles

    def test_credible_source_long_title_kept(self):
        # Reuters + long title → kept even without keyword/synonym
        articles = [
            _art("Federal Reserve weighs unprecedented rate path amid market stress",
                 source="Reuters"),
            _art("Stocks mixed", source="Yahoo Finance"),
        ]
        kept = _filter_relevant_headlines(articles, "BTC")
        sources = [a["source"] for a in kept]
        assert "Reuters" in sources
        assert "Yahoo Finance" not in sources

    def test_credible_source_short_title_dropped(self):
        # Reuters but a short price-ticker title → dropped
        articles = [
            _art("Bitcoin: $67,123", source="Reuters"),
        ]
        kept = _filter_relevant_headlines(articles, "BTC")
        # Should fall back since 100% would be dropped
        assert len(kept) >= 1

    def test_all_irrelevant_falls_back_to_most_recent(self):
        articles = [
            _art("Markets mixed", published="2026-04-28T08:00:00+00:00"),
            _art("Stocks update",  published="2026-04-28T09:00:00+00:00"),
            _art("Generic news",   published="2026-04-28T10:00:00+00:00"),
            _art("Daily wrap",     published="2026-04-28T11:00:00+00:00"),
        ]
        kept = _filter_relevant_headlines(articles, "BTC", fallback_n=2)
        # Filter would drop all; fallback keeps most-recent 2
        assert len(kept) == 2
        kept_titles = [a["title"] for a in kept]
        # Most recent two by published time
        assert "Daily wrap" in kept_titles
        assert "Generic news" in kept_titles

    def test_empty_input_returns_empty(self):
        assert _filter_relevant_headlines([], "BTC") == []

    def test_keeps_all_when_all_relevant(self):
        articles = [
            _art("Bitcoin halving live blog"),
            _art("BTC ETF inflows hit record"),
        ]
        kept = _filter_relevant_headlines(articles, "BTC")
        assert len(kept) == 2

    def test_metals_ticker_synonyms_work(self):
        articles = [
            _art("Gold futures rally on Fed bets"),       # keep (synonym for XAU)
            _art("Silver pulled back from highs"),        # drop for XAU (silver != gold)
            _art("Markets quiet in Asia"),                # drop
        ]
        kept = _filter_relevant_headlines(articles, "XAU")
        titles = [a["title"] for a in kept]
        assert "Gold futures rally on Fed bets" in titles
        # Silver headline is irrelevant to XAU specifically
        assert "Silver pulled back from highs" not in titles
