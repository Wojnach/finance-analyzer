"""Tests for portfolio.signals.news_event — news/event detection signal."""

from unittest import mock

import numpy as np
import pandas as pd

from portfolio.signals.news_event import (
    _MAX_CONFIDENCE,
    _headline_velocity,
    _keyword_severity_vote,
    _sector_impact_vote,
    _sentiment_shift,
    _source_weight_vote,
    compute_news_event_signal,
)


def _make_df(n=100):
    """Create minimal OHLCV dataframe."""
    return pd.DataFrame({
        "open": np.random.uniform(100, 110, n),
        "high": np.random.uniform(110, 120, n),
        "low": np.random.uniform(90, 100, n),
        "close": np.random.uniform(100, 110, n),
        "volume": np.random.uniform(1000, 5000, n),
    })


def _make_headlines(titles, source="Unknown"):
    """Create headline dicts from title strings."""
    return [{"title": t, "source": source, "published": "2026-02-23T12:00:00Z"} for t in titles]


class TestComputeNewsEventSignal:
    def test_no_context_returns_hold(self):
        df = _make_df()
        result = compute_news_event_signal(df, context=None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_empty_ticker_returns_hold(self):
        df = _make_df()
        result = compute_news_event_signal(df, context={"ticker": "", "config": {}})
        assert result["action"] == "HOLD"

    @mock.patch("portfolio.signals.news_event._fetch_headlines", return_value=[])
    def test_no_headlines_returns_hold(self, _mock):
        df = _make_df()
        result = compute_news_event_signal(df, context={"ticker": "NVDA", "config": {}})
        assert result["action"] == "HOLD"

    @mock.patch("portfolio.signals.news_event._fetch_headlines")
    def test_critical_headlines_produce_sell(self, mock_fetch):
        mock_fetch.return_value = _make_headlines([
            "Trump announces massive tariff on chip imports",
            "Semiconductor sanctions deepen as trade war escalates",
            "Crash fears as tariffs hit NVDA suppliers",
            "New tariff round threatens semiconductor supply chain",
            "Markets crash on tariff news",
            "War fears drive tech selloff",
            "Ban on chip exports announced",
            "Industry crash after tariff shock",
            "Sanctions expanded to tech sector",
            "Tariff retaliation expected",
        ], source="Reuters")
        df = _make_df()
        result = compute_news_event_signal(df, context={"ticker": "NVDA", "config": {}})
        assert result["action"] == "SELL"
        assert result["confidence"] > 0
        assert result["confidence"] <= _MAX_CONFIDENCE

    @mock.patch("portfolio.signals.news_event._fetch_headlines")
    def test_confidence_capped(self, mock_fetch):
        mock_fetch.return_value = _make_headlines([
            "Crash crash crash tariff war sanctions",
        ] * 15, source="Bloomberg")
        df = _make_df()
        result = compute_news_event_signal(df, context={"ticker": "NVDA", "config": {}})
        assert result["confidence"] <= _MAX_CONFIDENCE

    @mock.patch("portfolio.signals.news_event._fetch_headlines")
    def test_neutral_headlines_hold(self, mock_fetch):
        mock_fetch.return_value = _make_headlines([
            "Markets trade sideways",
            "No major news today",
            "Volume remains low",
        ])
        df = _make_df()
        result = compute_news_event_signal(df, context={"ticker": "NVDA", "config": {}})
        assert result["action"] == "HOLD"

    def test_result_structure(self):
        df = _make_df()
        result = compute_news_event_signal(df, context=None)
        assert "action" in result
        assert "confidence" in result
        assert "sub_signals" in result
        assert "indicators" in result
        assert "headline_velocity" in result["sub_signals"]
        assert "keyword_severity" in result["sub_signals"]
        assert "sentiment_shift" in result["sub_signals"]
        assert "source_weight" in result["sub_signals"]
        assert "sector_impact" in result["sub_signals"]


class TestHeadlineVelocity:
    def test_low_count_hold(self):
        headlines = _make_headlines(["Normal day"] * 3)
        action, ind = _headline_velocity(headlines)
        assert action == "HOLD"

    def test_high_count_neg_keywords_sell(self):
        headlines = _make_headlines([
            "Tariff shock hits markets",
            "Crash fears intensify",
            "Sanctions widen",
            "War risk elevated",
            "Ban on imports expected",
            "Market crash deepens",
            "Tariff impact analysis",
            "Trade war escalation",
            "New sanctions announced",
            "Markets tumble on crash fears",
        ])
        action, ind = _headline_velocity(headlines)
        assert action == "SELL"

    def test_high_count_no_keywords_hold(self):
        headlines = _make_headlines(["Nothing interesting"] * 12)
        action, ind = _headline_velocity(headlines)
        assert action == "HOLD"


class TestKeywordSeverityVote:
    def test_critical_keyword_sells(self):
        headlines = _make_headlines(["Massive crash in chip stocks"])
        action, ind = _keyword_severity_vote(headlines)
        assert action == "SELL"

    def test_no_keywords_hold(self):
        headlines = _make_headlines(["Markets are calm today"])
        action, ind = _keyword_severity_vote(headlines)
        assert action == "HOLD"

    def test_indicators_contain_severity(self):
        headlines = _make_headlines(["Tariff announced"])
        _, ind = _keyword_severity_vote(headlines)
        assert "max_severity" in ind
        assert ind["max_severity"] == "critical"


class TestSentimentShift:
    def test_strong_negative_skew_sells(self):
        headlines = _make_headlines([
            "Tariff hits hard",
            "Crash imminent",
            "Sanctions widen",
            "War fears grow",
            "Markets calm otherwise",
        ])
        action, ind = _sentiment_shift(headlines)
        assert action == "SELL"

    def test_few_keyword_headlines_hold(self):
        headlines = _make_headlines([
            "Normal day on markets",
            "Stocks end flat",
        ])
        action, ind = _sentiment_shift(headlines)
        assert action == "HOLD"

    # SM-P1-1 (2026-05-02 adversarial follow-ups): the "cut" keyword
    # previously defaulted to POSITIVE except for "guidance cut". This treated
    # bearish phrases ("earnings cut", "dividend cut", "production cut",
    # "forecast cut", "outlook cut", "salary cut") as bullish.

    def test_rate_cut_remains_positive(self):
        """Whitelist phrase: 'rate cut' is bullish (Fed/central bank easing)."""
        headlines = _make_headlines([
            "Fed signals rate cut next month",
            "Powell hints rate cut coming",
            "Central bank rate cut likely",
        ])
        action, _ = _sentiment_shift(headlines)
        assert action == "BUY"

    def test_tax_cut_remains_positive(self):
        """Whitelist phrase: 'tax cut' is bullish (fiscal stimulus)."""
        # Use 'tax cut' but force severity by adding rate words that score.
        headlines = _make_headlines([
            "Tax cut bill passes amid recession fears",
            "Tax cut announced on tariff easing",
            "Tax cut clears senate as inflation drops",
        ])
        action, _ = _sentiment_shift(headlines)
        # All have rate-related severity ("recession", "tariff", "inflation"
        # are critical/high). "tax cut" should not flip them to BUY but it
        # should at minimum not be SOLD as a bearish "cut" — direction depends
        # on the dominant keyword, but the "cut" branch must classify
        # "tax cut" as positive.
        assert action in ("BUY", "HOLD"), (
            f"'tax cut' incorrectly classified as bearish (action={action})"
        )

    def test_guidance_cut_is_negative(self):
        """Bearish phrase: 'guidance cut' = lowered forward expectations."""
        headlines = _make_headlines([
            "Acme issues guidance cut citing recession",
            "Beta announces guidance cut on inflation pressure",
            "Gamma guidance cut hits stock as tariff bites",
        ])
        action, _ = _sentiment_shift(headlines)
        assert action == "SELL"

    def test_dividend_cut_is_negative(self):
        """Bearish phrase: 'dividend cut' should not be classified as bullish.
        With the bug, 'cut' alone defaulted to positive."""
        headlines = _make_headlines([
            "Acme announces dividend cut amid recession fears",
            "Beta dividend cut shocks investors as inflation rises",
            "Gamma dividend cut as tariff uncertainty grows",
        ])
        action, _ = _sentiment_shift(headlines)
        assert action == "SELL", (
            "'dividend cut' should be classified as bearish, not bullish. "
            "The bare 'cut' keyword must not default to pos."
        )

    def test_earnings_cut_is_negative(self):
        """Bearish phrase: 'earnings cut' (lowered forecasts)."""
        headlines = _make_headlines([
            "Acme earnings cut spooks investors amid recession",
            "Beta cuts earnings outlook as inflation persists",
            "Gamma slashes earnings forecast on tariff risks",
        ])
        action, _ = _sentiment_shift(headlines)
        assert action == "SELL"

    def test_production_cut_is_negative_or_hold(self):
        """'Production cut' — generally bearish for the producer."""
        headlines = _make_headlines([
            "Acme production cut shrinks output amid recession",
            "Beta announces production cut citing inflation",
            "Gamma production cut as tariff pressure mounts",
        ])
        action, _ = _sentiment_shift(headlines)
        # Should NOT be BUY (the bare "cut" bug previously classified it positive).
        assert action != "BUY", (
            f"'production cut' classified as BUY (action={action}); "
            "the bare 'cut' default must not flip these to bullish."
        )


class TestSourceWeightVote:
    def test_credible_negative_sells(self):
        headlines = _make_headlines([
            "Tariff shock hits markets",
            "Crash fears deepen",
            "Sanctions impact supply chains",
        ], source="Reuters")
        action, ind = _source_weight_vote(headlines)
        assert action == "SELL"
        assert ind["credible_sources"] == 3

    def test_non_credible_sources_hold(self):
        headlines = _make_headlines([
            "Tariff shock",
            "Crash incoming",
        ], source="RandomBlog")
        action, ind = _source_weight_vote(headlines)
        assert action == "HOLD"


class TestSectorImpactVote:
    def test_tariff_sells_semiconductor(self):
        headlines = _make_headlines(["New tariff on chips announced"])
        action, ind = _sector_impact_vote(headlines, "NVDA")
        assert action == "SELL"

    def test_tariff_buys_metals(self):
        headlines = _make_headlines(["Tariff fears boost gold"])
        action, ind = _sector_impact_vote(headlines, "XAU-USD")
        assert action == "BUY"

    def test_no_sector_impact_hold(self):
        headlines = _make_headlines(["Regular market news"])
        action, ind = _sector_impact_vote(headlines, "NVDA")
        assert action == "HOLD"
