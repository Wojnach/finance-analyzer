"""Tests for stock consensus threshold and sentiment hysteresis.

Covers:
- MIN_VOTERS=3 for stocks requires 3 active voters for consensus
- MIN_VOTERS=3 for crypto requires 3 active voters for consensus
- Sentiment hysteresis prevents rapid direction flips
- Stock signal generation produces correct vote counts
"""

import unittest.mock as mock

from conftest import make_indicators as _make_indicators

from portfolio.main import (
    CRYPTO_SYMBOLS,
    MIN_VOTERS_CRYPTO,
    MIN_VOTERS_STOCK,
    STOCK_SYMBOLS,
    generate_signal,
)


def _null_cached(key, ttl, func, *args):
    """Mock _cached that blocks external calls.

    Returns {} for accuracy/activation keys so H3 fail-closed doesn't fire
    (which would gate all signals to 0% accuracy and break vote count assertions).
    """
    if key and ("accuracy" in key or "activation_rates" in key):
        return {}
    return None


# Disable confidence penalty cascade so base consensus tests are isolated
_NO_PENALTIES = {"confidence_penalties": {"enabled": False}}


# Stock-price defaults for consensus tests (conftest defaults to crypto prices)
_STOCK_DEFAULTS = dict(
    close=130.0, ema9=130.0, ema21=130.0,
    bb_upper=135.0, bb_lower=125.0, bb_mid=130.0,
)


def make_indicators(**overrides):
    merged = {**_STOCK_DEFAULTS, **overrides}
    return _make_indicators(**merged)


class TestMinVotersConstants:
    def test_stock_threshold_is_3(self):
        assert MIN_VOTERS_STOCK == 3

    def test_crypto_threshold_is_3(self):
        assert MIN_VOTERS_CRYPTO == 3

    def test_stock_symbols_defined(self):
        # After Apr 09 ticker reduction, MSTR is the only stock retained
        # (kept as BTC NAV-premium reference for metals_loop). See
        # portfolio/tickers.py for the full rationale.
        assert "MSTR" in STOCK_SYMBOLS

    def test_crypto_symbols_defined(self):
        assert "BTC-USD" in CRYPTO_SYMBOLS
        assert "ETH-USD" in CRYPTO_SYMBOLS


class TestStockConsensus:
    """Stocks with MIN_VOTERS=3 require 3 active voters for consensus."""

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_stock_hold_with_2_voters(self, _mock):
        """RSI oversold + MACD crossover = 2 BUY voters → HOLD (need 3 for stocks)."""
        ind = make_indicators(
            rsi=25,  # oversold → BUY vote
            macd_hist=1.0,
            macd_hist_prev=-1.0,  # crossover → BUY vote
            ema9=130.0,
            ema21=130.0,  # no gap → abstains
            price_vs_bb="inside",  # inside → abstains
        )
        action, conf, extra = generate_signal(ind, ticker="MSTR")
        assert extra["_buy_count"] == 2
        assert extra["_sell_count"] == 0
        assert extra["_voters"] == 2
        assert action == "HOLD"  # 2 < MIN_VOTERS_STOCK(3)

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_stock_buy_with_3_voters(self, _mock):
        """RSI + MACD + BB = 3 BUY voters → BUY for stocks.

        Note: EMA is regime-gated in "ranging" (RSI=25 < 45 → can't reach trending-up
        with an EMA gap). Use BB below_lower as the reliable 3rd voter instead.
        """
        ind = make_indicators(
            rsi=25,  # oversold → BUY vote
            macd_hist=1.0,
            macd_hist_prev=-1.0,  # crossover → BUY vote
            ema9=130.0,
            ema21=130.0,  # no gap → no EMA vote
            price_vs_bb="below_lower",  # below lower band → BUY vote
        )
        action, conf, extra = generate_signal(
            ind, ticker="MSTR", config=_NO_PENALTIES,
        )
        assert extra["_buy_count"] >= 3
        assert action == "BUY"

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_stock_sell_with_3_voters(self, _mock):
        """RSI overbought + MACD down + BB above = 3 SELL voters → SELL for stocks.

        Note: EMA is regime-gated in ranging. Use BB above_upper as reliable 3rd voter.
        """
        ind = make_indicators(
            rsi=75,  # overbought → SELL vote
            macd_hist=-1.0,
            macd_hist_prev=1.0,  # cross down → SELL vote
            ema9=130.0,
            ema21=130.0,  # no gap
            price_vs_bb="above_upper",  # above upper band → SELL vote
        )
        action, conf, extra = generate_signal(
            ind, ticker="MSTR", config=_NO_PENALTIES,
        )
        assert extra["_sell_count"] >= 3
        assert action == "SELL"

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_stock_hold_with_1_voter(self, _mock):
        """Only 1 active voter → HOLD for stocks (need 3)."""
        ind = make_indicators(
            rsi=25,  # oversold → BUY vote (1 voter)
            macd_hist=1.0,
            macd_hist_prev=2.0,  # no crossover → abstains
            ema9=130.0,
            ema21=130.0,  # no gap → abstains
            price_vs_bb="inside",  # inside → abstains
        )
        action, conf, extra = generate_signal(ind, ticker="MSTR")
        assert extra["_voters"] == 1
        assert action == "HOLD"

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_all_stock_tickers_use_3_voter_threshold(self, _mock):
        """All stock tickers need 3+ voters for consensus."""
        for ticker in STOCK_SYMBOLS:
            ind = make_indicators(
                rsi=25,
                macd_hist=1.0,
                macd_hist_prev=-1.0,
                ema9=130.0,
                ema21=130.0,  # no gap; EMA gated in ranging anyway
                price_vs_bb="below_lower",  # 3 voters: RSI + MACD + BB
            )
            action, conf, extra = generate_signal(
                ind, ticker=ticker, config=_NO_PENALTIES,
            )
            assert action == "BUY", f"{ticker} should reach BUY consensus with 3 voters"


class TestCryptoConsensus:
    """Crypto with MIN_VOTERS=3 still requires 3 active voters."""

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_crypto_hold_with_2_voters(self, _mock):
        """2 BUY voters → HOLD for crypto (needs 3)."""
        ind = make_indicators(
            rsi=25,  # oversold → BUY vote
            macd_hist=1.0,
            macd_hist_prev=-1.0,  # crossover → BUY vote
            ema9=69000.0,
            ema21=69000.0,  # no gap → abstains
            price_vs_bb="inside",
            close=69000.0,
        )
        action, conf, extra = generate_signal(ind, ticker="BTC-USD")
        # Without external signals, only 2 technical voters
        assert extra["_voters"] == 2
        assert action == "HOLD"

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_crypto_buy_with_3_voters(self, _mock):
        """3 BUY voters → BUY for crypto (RSI + MACD + BB)."""
        ind = make_indicators(
            rsi=25,  # oversold → BUY vote
            macd_hist=1.0,
            macd_hist_prev=-1.0,  # crossover → BUY vote
            ema9=69000.0,
            ema21=69000.0,  # no gap; EMA gated in ranging
            price_vs_bb="below_lower",  # below lower band → BUY vote
            close=69000.0,
        )
        action, conf, extra = generate_signal(
            ind, ticker="BTC-USD", config=_NO_PENALTIES,
        )
        assert extra["_buy_count"] >= 3
        assert action == "BUY"


class TestSentimentHysteresis:
    """Sentiment hysteresis prevents rapid flip spam."""

    def setup_method(self):
        """Reset sentiment state between tests."""
        import portfolio.signal_engine as pse

        pse._prev_sentiment.clear()
        pse._prev_sentiment_loaded = True  # prevent file reads in tests

    def test_first_sentiment_uses_low_threshold(self):
        """First reading with no previous direction uses 0.40 threshold."""

        ind = make_indicators()
        # Mock sentiment returning positive with 0.45 confidence
        mock_sent = {
            "overall_sentiment": "positive",
            "confidence": 0.45,
            "num_articles": 5,
            "model": "test",
        }
        with mock.patch(
            "portfolio.signal_engine._cached",
            side_effect=lambda k, t, f, *a: mock_sent if "sentiment" in k else None,
        ):
            action, conf, extra = generate_signal(ind, ticker="MSTR", config={})

        # 0.45 > 0.40 threshold → should vote
        assert extra.get("sentiment") == "positive"

    def test_same_direction_uses_low_threshold(self):
        """Same direction as previous uses 0.40 threshold."""
        import portfolio.signal_engine as pse

        pse._prev_sentiment["MSTR"] = "positive"

        ind = make_indicators()
        mock_sent = {
            "overall_sentiment": "positive",
            "confidence": 0.45,
            "num_articles": 5,
            "model": "test",
        }
        with mock.patch(
            "portfolio.signal_engine._cached",
            side_effect=lambda k, t, f, *a: mock_sent if "sentiment" in k else None,
        ):
            action, conf, extra = generate_signal(ind, ticker="MSTR", config={})

        # Same direction, 0.45 > 0.40 → should vote
        assert extra.get("sentiment") == "positive"

    def test_flip_direction_requires_higher_threshold(self):
        """Flipping direction requires confidence > 0.55."""
        import portfolio.signal_engine as pse

        pse._prev_sentiment["MSTR"] = "positive"

        ind = make_indicators()
        # Sentiment flips to negative with 0.50 confidence (< 0.55 threshold)
        mock_sent = {
            "overall_sentiment": "negative",
            "confidence": 0.50,
            "num_articles": 5,
            "model": "test",
        }

        original_cached = None

        def mock_cached_fn(k, t, f, *a):
            if "sentiment" in k:
                return mock_sent
            return None

        with mock.patch("portfolio.signal_engine._cached", side_effect=mock_cached_fn):
            action, conf, extra = generate_signal(ind, ticker="MSTR", config={})

        # 0.50 < 0.55 flip threshold → should NOT add a vote
        assert extra.get("sentiment") == "negative"  # still reported
        assert extra.get("sentiment_conf") == 0.50  # confidence reported
        # The key test: the sentiment should NOT have counted as a sell vote
        # because 0.50 < 0.55 flip threshold
        assert extra["_sell_count"] == 0

    def test_flip_direction_above_threshold_votes(self):
        """Flipping direction with confidence > 0.55 should vote."""
        import portfolio.signal_engine as pse

        pse._prev_sentiment["MSTR"] = "positive"

        ind = make_indicators()
        mock_sent = {
            "overall_sentiment": "negative",
            "confidence": 0.60,
            "num_articles": 5,
            "model": "test",
        }

        def mock_cached_fn(k, t, f, *a):
            if "sentiment" in k:
                return mock_sent
            return None

        with mock.patch("portfolio.signal_engine._cached", side_effect=mock_cached_fn):
            action, conf, extra = generate_signal(ind, ticker="MSTR", config={})

        # 0.60 > 0.55 flip threshold → should count as a sell vote
        assert extra["_sell_count"] >= 1


class TestStockSignalVoteCounts:
    """Stock signal generation produces correct total_applicable counts."""

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_stock_total_applicable(self, _mock):
        ind = make_indicators()
        _, _, extra = generate_signal(ind, ticker="MSTR")
        assert extra["_total_applicable"] == 26  # stocks: 36 registered minus gpu+disabled+crypto-only exclusions

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_crypto_total_applicable(self, _mock):
        ind = make_indicators(close=69000.0)
        _, _, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra["_total_applicable"] == 31  # crypto: 36 signals minus gpu+disabled+metals-only exclusions

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_stock_max_technical_voters(self, _mock):
        """RSI + MACD + BB = 3 core technical BUY votes in ranging regime.

        EMA is regime-gated in ranging, so max core technical voters is 3 not 4.
        """
        ind = make_indicators(
            rsi=25,  # oversold → BUY
            macd_hist=1.0,
            macd_hist_prev=-1.0,  # crossover → BUY
            ema9=130.0,
            ema21=130.0,  # no gap; EMA gated in ranging
            price_vs_bb="below_lower",  # → BUY
        )
        _, _, extra = generate_signal(ind, ticker="MSTR")
        # 3 core technical signals vote BUY (EMA gated in ranging regime)
        assert extra["_buy_count"] >= 3
        assert extra["_voters"] >= 3
