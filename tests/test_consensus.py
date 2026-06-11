"""Tests for stock consensus threshold and sentiment hysteresis.

Covers:
- MIN_VOTERS=3 for stocks requires 3 active voters for consensus
- MIN_VOTERS=3 for crypto requires 3 active voters for consensus
- Sentiment hysteresis prevents rapid direction flips
- Stock signal generation produces correct vote counts
"""

import unittest.mock as mock

import pytest

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


# 2026-06-11 (suite-cleanup): the 3-voter consensus tests assert the
# consensus MECHANIC (3 same-direction voters → that direction). They were
# silently depending on the live data/accuracy_cache.json: generate_signal
# pulls directional accuracy via accuracy_stats.get_or_compute_*, and the
# directional gate (_DIRECTIONAL_GATE_THRESHOLD=0.44) force-HOLDs a signal
# whose buy/sell accuracy on that direction is below 44%. As production
# accuracy drifted, rsi/macd/bb BUY accuracy fell below 0.44 (observed
# 41.3%/43.8%/39.7% on MSTR), so all three raw BUY votes were
# direction-gated and consensus returned HOLD — exactly the
# accuracy_cache.json fragility documented in docs/TESTING.md. This was
# NOT a campaign change (test file + gate threshold unchanged since
# pre-campaign eed1f70a). Make the test hermetic by pinning the directional
# accuracy above the gate so it exercises the voting logic, not live data.
_HERMETIC_ACC = {
    sig: {
        "accuracy": 0.60, "total": 500,
        "buy_accuracy": 0.60, "total_buy": 250,
        "sell_accuracy": 0.60, "total_sell": 250,
    }
    for sig in ("rsi", "macd", "bb", "ema", "mean_reversion", "momentum")
}


def _pin_accuracy():
    """Patch accuracy_stats helpers so directional gating is deterministic.

    Returns a list of started patchers; caller is responsible for stopping
    them (use as ``with contextlib.ExitStack()`` or via the fixture below).
    """
    import contextlib
    stack = contextlib.ExitStack()
    for name in (
        "get_or_compute_accuracy",
        "get_or_compute_recent_accuracy",
    ):
        stack.enter_context(
            mock.patch(f"portfolio.accuracy_stats.{name}",
                       return_value=dict(_HERMETIC_ACC))
        )
    stack.enter_context(
        mock.patch("portfolio.accuracy_stats.get_or_compute_per_ticker_accuracy",
                   return_value={})
    )
    # The PER-TICKER directional cache overrides the global blend inside
    # generate_signal (signal_engine.py ~4521) — including buy_accuracy used
    # by the directional gate. It's the real source of the live drift (rsi
    # BUY 41%, bb 39% on MSTR). Return {} so the pinned globals stand.
    stack.enter_context(
        mock.patch("portfolio.accuracy_stats.accuracy_by_ticker_signal_cached",
                   return_value={})
    )
    return stack


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
        # 2026-06-11 (suite-cleanup): pin directional accuracy above the
        # 0.44 gate so live cache drift can't direction-gate the BUY votes.
        with _pin_accuracy():
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
        # 2026-06-11 (suite-cleanup): pin directional accuracy above the
        # 0.44 gate so live cache drift can't direction-gate the BUY votes.
        for ticker in STOCK_SYMBOLS:
            ind = make_indicators(
                rsi=25,
                macd_hist=1.0,
                macd_hist_prev=-1.0,
                ema9=130.0,
                ema21=130.0,  # no gap; EMA gated in ranging anyway
                price_vs_bb="below_lower",  # 3 voters: RSI + MACD + BB
            )
            with _pin_accuracy():
                action, conf, extra = generate_signal(
                    ind, ticker=ticker, config=_NO_PENALTIES,
                )
            assert action == "BUY", f"{ticker} should reach BUY consensus with 3 voters"


class TestCryptoConsensus:
    """Crypto with MIN_VOTERS=3 still requires 3 active voters."""

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_crypto_hold_with_2_voters(self, _mock):
        """2 BUY voters → HOLD for crypto (needs 3).

        2026-05-11 (Stage 2 Batch 1): the BB voter was previously
        force-HOLD when `price_vs_bb == "inside"`. After the dead-zone
        retrofit, BB emits a soft directional vote based on normalized
        band position. Earlier this test relied on `_STOCK_DEFAULTS`
        leaking through (bb_upper=135 vs close=69000 -> clamped
        position=+1.0 -> SELL), which masked a third voter. Pinning
        bb_mid at the BTC band centerline keeps BB neutral so this
        test continues to exercise the 2-voter HOLD path.
        """
        ind = make_indicators(
            rsi=25,  # oversold → BUY vote
            macd_hist=1.0,
            macd_hist_prev=-1.0,  # crossover → BUY vote
            ema9=69000.0,
            ema21=69000.0,  # no gap → abstains
            price_vs_bb="inside",
            close=69000.0,
            # Realign BB band around BTC's close so band_position == 0
            # (otherwise BB would emit a soft SELL via the new dead-zone path)
            bb_mid=69000.0,
            bb_upper=70000.0,
            bb_lower=68000.0,
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

    @pytest.mark.skip(
        reason="2026-05-19: sentiment signal disabled across all tickers "
        "(33.8% 3h_recent, 94.9% BUY-only). Hysteresis flip-direction "
        "behavior is moot while the signal is force-HOLD'd. Restore the "
        "assertion if sentiment is re-enabled."
    )
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

    @mock.patch("portfolio.market_timing.should_skip_gpu", return_value=False)
    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_stock_total_applicable(self, _mock, _gpu_mock):
        ind = make_indicators()
        _, _, extra = generate_signal(ind, ticker="MSTR")
        # 2026-05-28: 9 → 14 after enabling 5 proven regime signals.
        # 2026-06-11 (B6 audit): 14 → 12. The "ministral only counts on
        # crypto" special-case was removed (_compute_applicable_count now
        # counts ministral on ALL tickers), and the June disable wave
        # trimmed the MSTR set. Pinned should_skip_gpu=False so the GPU
        # signals (incl. ministral) count deterministically regardless of
        # MSTR market hours — was flaky on the unmocked path.
        assert extra["_total_applicable"] == 12

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_crypto_total_applicable(self, _mock):
        ind = make_indicators(close=69000.0)
        _, _, extra = generate_signal(ind, ticker="BTC-USD")
        # 2026-05-10: 33 → 26 after the disable wave above. Same
        # tripwire pattern as the stock counterpart.
        # 2026-05-28: 14 → 19 after enabling 5 proven regime signals.
        # 2026-06-11 (B6 audit + June disable wave): 19 → 15. ministral now
        # counts on all tickers (already counted for crypto, so no +1 here),
        # and the June signal disables trimmed the crypto applicable set.
        assert extra["_total_applicable"] == 15

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
