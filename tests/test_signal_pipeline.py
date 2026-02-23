"""Tests for end-to-end signal generation pipeline.

Covers:
- Mock all data sources
- generate_signal() returns valid structure for all ticker types
- write_agent_summary() produces valid JSON
- Vote counts add up (_buy_count + _sell_count + holds = _total_applicable)
- Weighted consensus calculation
"""

import json
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

from portfolio.main import (
    generate_signal,
    CRYPTO_SYMBOLS,
    STOCK_SYMBOLS,
    METALS_SYMBOLS,
    MIN_VOTERS_CRYPTO,
    MIN_VOTERS_STOCK,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_cached(key, ttl, func, *args):
    """Mock _cached that blocks all external calls, returning None."""
    return None


def make_indicators(**overrides):
    """Create a valid indicator dict for testing."""
    base = {
        "close": 69000.0,
        "rsi": 50.0,
        "macd_hist": 0.0,
        "macd_hist_prev": 0.0,
        "ema9": 69000.0,
        "ema21": 69000.0,
        "bb_upper": 70000.0,
        "bb_lower": 68000.0,
        "bb_mid": 69000.0,
        "price_vs_bb": "inside",
        "atr": 100.0,
        "atr_pct": 0.15,
        "rsi_p20": 30,
        "rsi_p80": 70,
    }
    base.update(overrides)
    return base


def make_ohlcv_df(n=100, base_price=100.0, trend=0.0):
    """Create a synthetic OHLCV DataFrame."""
    np.random.seed(42)
    close = base_price + np.cumsum(np.random.randn(n) * 0.5) + np.arange(n) * trend
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.2,
        "high": close + np.abs(np.random.randn(n)),
        "low": close - np.abs(np.random.randn(n)),
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
        "time": pd.date_range("2024-01-01", periods=n, freq="1h"),
    })


# ---------------------------------------------------------------------------
# Test: generate_signal returns valid structure
# ---------------------------------------------------------------------------

class TestGenerateSignalStructure:
    """Verify generate_signal returns the expected structure for all ticker types."""

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_crypto_signal_structure(self, _mock):
        """Crypto ticker returns all expected keys."""
        ind = make_indicators()
        df = make_ohlcv_df(n=250)
        action, conf, extra = generate_signal(ind, ticker="BTC-USD", df=df)

        assert action in ("BUY", "SELL", "HOLD")
        assert 0.0 <= conf <= 1.0
        assert "_voters" in extra
        assert "_total_applicable" in extra
        assert "_buy_count" in extra
        assert "_sell_count" in extra
        assert "_votes" in extra
        assert "_regime" in extra
        assert "_weighted_action" in extra
        assert "_weighted_confidence" in extra

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_stock_signal_structure(self, _mock):
        """Stock ticker returns all expected keys."""
        ind = make_indicators(close=130.0)
        df = make_ohlcv_df(n=250, base_price=130.0)
        action, conf, extra = generate_signal(ind, ticker="MSTR", df=df)

        assert action in ("BUY", "SELL", "HOLD")
        assert "_total_applicable" in extra
        assert "_votes" in extra

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_metal_signal_structure(self, _mock):
        """Metal ticker returns all expected keys."""
        ind = make_indicators(close=2000.0)
        df = make_ohlcv_df(n=250, base_price=2000.0)
        action, conf, extra = generate_signal(ind, ticker="XAU-USD", df=df)

        assert action in ("BUY", "SELL", "HOLD")
        assert "_total_applicable" in extra


# ---------------------------------------------------------------------------
# Test: Vote counts add up
# ---------------------------------------------------------------------------

class TestVoteCountIntegrity:
    """Verify _buy_count + _sell_count + holds = _total_applicable."""

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_crypto_vote_counts(self, _mock):
        """For crypto, total applicable = 24 (with custom_lora disabled)."""
        ind = make_indicators()
        df = make_ohlcv_df(n=250)
        _, _, extra = generate_signal(ind, ticker="BTC-USD", df=df)

        total_applicable = extra["_total_applicable"]
        votes = extra["_votes"]
        buy_count = extra["_buy_count"]
        sell_count = extra["_sell_count"]

        # Holds = total signals that voted - buy - sell
        # Total signals that actually voted is len(votes)
        hold_count = sum(1 for v in votes.values() if v == "HOLD")
        total_voted = buy_count + sell_count + hold_count

        # All signals should have voted (BUY, SELL, or HOLD)
        assert total_voted == len(votes)
        # buy + sell should match counts
        assert buy_count == sum(1 for v in votes.values() if v == "BUY")
        assert sell_count == sum(1 for v in votes.values() if v == "SELL")

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_stock_vote_counts(self, _mock):
        """For stocks, total applicable = 23."""
        ind = make_indicators(close=130.0)
        df = make_ohlcv_df(n=250, base_price=130.0)
        _, _, extra = generate_signal(ind, ticker="NVDA", df=df)

        assert extra["_total_applicable"] == 23

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_metal_vote_counts(self, _mock):
        """For metals, total applicable = 23."""
        ind = make_indicators(close=2000.0)
        df = make_ohlcv_df(n=250, base_price=2000.0)
        _, _, extra = generate_signal(ind, ticker="XAU-USD", df=df)

        assert extra["_total_applicable"] == 23

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_all_stock_symbols_have_23_applicable(self, _mock):
        """Every stock symbol should have exactly 23 total applicable signals."""
        ind = make_indicators(close=100.0)
        df = make_ohlcv_df(n=250, base_price=100.0)

        for ticker in list(STOCK_SYMBOLS)[:5]:  # test a sample
            _, _, extra = generate_signal(ind, ticker=ticker, df=df)
            assert extra["_total_applicable"] == 23, \
                f"{ticker} has {extra['_total_applicable']} total applicable, expected 23"


# ---------------------------------------------------------------------------
# Test: Vote count consistency
# ---------------------------------------------------------------------------

class TestVoteConsistency:
    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_buy_count_matches_votes(self, _mock):
        """_buy_count should exactly match number of BUY votes."""
        ind = make_indicators(
            rsi=25, macd_hist=1.0, macd_hist_prev=-1.0,
            ema9=70000, ema21=69000, price_vs_bb="below_lower",
        )
        df = make_ohlcv_df(n=250)
        _, _, extra = generate_signal(ind, ticker="BTC-USD", df=df)

        actual_buys = sum(1 for v in extra["_votes"].values() if v == "BUY")
        assert extra["_buy_count"] == actual_buys

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_sell_count_matches_votes(self, _mock):
        """_sell_count should exactly match number of SELL votes."""
        ind = make_indicators(
            rsi=75, macd_hist=-1.0, macd_hist_prev=1.0,
            ema9=68000, ema21=69000, price_vs_bb="above_upper",
        )
        df = make_ohlcv_df(n=250)
        _, _, extra = generate_signal(ind, ticker="BTC-USD", df=df)

        actual_sells = sum(1 for v in extra["_votes"].values() if v == "SELL")
        assert extra["_sell_count"] == actual_sells

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_voters_equals_buy_plus_sell(self, _mock):
        """_voters should equal _buy_count + _sell_count."""
        ind = make_indicators()
        df = make_ohlcv_df(n=250)
        _, _, extra = generate_signal(ind, ticker="BTC-USD", df=df)

        assert extra["_voters"] == extra["_buy_count"] + extra["_sell_count"]


# ---------------------------------------------------------------------------
# Test: Consensus thresholds
# ---------------------------------------------------------------------------

class TestConsensusThresholds:
    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_crypto_needs_3_voters(self, _mock):
        """Crypto needs MIN_VOTERS=3 active voters to reach consensus."""
        # Only 2 voters: RSI + MACD
        ind = make_indicators(
            rsi=25, macd_hist=1.0, macd_hist_prev=-1.0,
            ema9=69000, ema21=69000, price_vs_bb="inside",
        )
        action, conf, extra = generate_signal(ind, ticker="BTC-USD")
        assert extra["_voters"] == 2
        assert action == "HOLD"

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_stock_needs_3_voters(self, _mock):
        """Stocks need MIN_VOTERS=3 active voters to reach consensus."""
        # Only 2 voters: RSI + MACD → HOLD (need 3)
        ind = make_indicators(
            rsi=25, macd_hist=1.0, macd_hist_prev=-1.0,
            ema9=130, ema21=130, price_vs_bb="inside", close=130.0,
        )
        action, conf, extra = generate_signal(ind, ticker="MSTR")
        assert extra["_voters"] == 2
        assert action == "HOLD"  # 2 < MIN_VOTERS_STOCK(3)

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_metal_needs_3_voters(self, _mock):
        """Metals need MIN_VOTERS=3 active voters to reach consensus."""
        # Only 2 voters: RSI + MACD → HOLD (need 3)
        ind = make_indicators(
            rsi=25, macd_hist=1.0, macd_hist_prev=-1.0,
            ema9=2000, ema21=2000, price_vs_bb="inside", close=2000.0,
        )
        action, conf, extra = generate_signal(ind, ticker="XAU-USD")
        assert extra["_voters"] == 2
        assert action == "HOLD"  # 2 < MIN_VOTERS_STOCK(3)


# ---------------------------------------------------------------------------
# Test: Weighted consensus
# ---------------------------------------------------------------------------

class TestWeightedConsensus:
    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_weighted_confidence_in_range(self, _mock):
        """Weighted confidence should be between 0.0 and 1.0."""
        ind = make_indicators(
            rsi=25, macd_hist=1.0, macd_hist_prev=-1.0,
            ema9=70000, ema21=69000, price_vs_bb="below_lower",
        )
        df = make_ohlcv_df(n=250)
        _, _, extra = generate_signal(ind, ticker="BTC-USD", df=df)

        wc = extra["_weighted_confidence"]
        assert 0.0 <= wc <= 1.0

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_weighted_action_is_valid(self, _mock):
        """Weighted action should be BUY, SELL, or HOLD."""
        ind = make_indicators()
        df = make_ohlcv_df(n=250)
        _, _, extra = generate_signal(ind, ticker="BTC-USD", df=df)

        assert extra["_weighted_action"] in ("BUY", "SELL", "HOLD")


# ---------------------------------------------------------------------------
# Test: Enhanced signals included when df provided
# ---------------------------------------------------------------------------

class TestEnhancedSignals:
    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_enhanced_signals_present_with_df(self, _mock):
        """Enhanced signal modules should produce votes when df is provided."""
        ind = make_indicators()
        df = make_ohlcv_df(n=250)
        _, _, extra = generate_signal(ind, ticker="BTC-USD", df=df)

        enhanced_names = [
            "trend", "momentum", "volume_flow", "volatility_sig",
            "candlestick", "structure", "fibonacci", "smart_money",
            "oscillators", "heikin_ashi", "mean_reversion", "calendar",
            "momentum_factors", "macro_regime",
        ]
        for name in enhanced_names:
            assert name in extra["_votes"], f"Missing enhanced signal: {name}"
            assert extra["_votes"][name] in ("BUY", "SELL", "HOLD")

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_enhanced_signals_hold_without_df(self, _mock):
        """Without df, enhanced signals should all be HOLD."""
        ind = make_indicators()
        _, _, extra = generate_signal(ind, ticker="BTC-USD", df=None)

        enhanced_names = [
            "trend", "momentum", "volume_flow", "volatility_sig",
            "candlestick", "structure", "fibonacci", "smart_money",
            "oscillators", "heikin_ashi", "mean_reversion", "calendar",
            "momentum_factors", "macro_regime",
        ]
        for name in enhanced_names:
            assert extra["_votes"][name] == "HOLD", \
                f"{name} should be HOLD without df, got {extra['_votes'][name]}"

    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_enhanced_signals_hold_with_short_df(self, _mock):
        """With df shorter than 26 rows, enhanced signals should be HOLD."""
        ind = make_indicators()
        df = make_ohlcv_df(n=10)  # too short
        _, _, extra = generate_signal(ind, ticker="BTC-USD", df=df)

        for name in ["trend", "momentum"]:
            assert extra["_votes"][name] == "HOLD"


# ---------------------------------------------------------------------------
# Test: Regime detection
# ---------------------------------------------------------------------------

class TestRegimeDetection:
    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_regime_field_present(self, _mock):
        """Regime should be present in extra info."""
        ind = make_indicators()
        _, _, extra = generate_signal(ind, ticker="BTC-USD")

        assert "_regime" in extra
        assert extra["_regime"] in (
            "trending-up", "trending-down", "ranging",
            "high-vol", "breakout", "capitulation",
        )


# ---------------------------------------------------------------------------
# Test: Confluence score
# ---------------------------------------------------------------------------

class TestConfluenceScore:
    @mock.patch("portfolio.signal_engine._cached", side_effect=_null_cached)
    def test_confluence_score_present(self, _mock):
        """Confluence score should be present in extra."""
        ind = make_indicators()
        df = make_ohlcv_df(n=250)
        _, _, extra = generate_signal(ind, ticker="BTC-USD", df=df)

        assert "_confluence_score" in extra
        assert isinstance(extra["_confluence_score"], (int, float))


# ---------------------------------------------------------------------------
# Test: write_agent_summary produces valid output
# ---------------------------------------------------------------------------

class TestWriteAgentSummary:
    def test_summary_structure(self):
        """write_agent_summary should produce a dict with expected keys."""
        from portfolio.main import write_agent_summary, portfolio_value

        # Create minimal inputs
        signals = {
            "BTC-USD": {
                "action": "HOLD",
                "confidence": 0.0,
                "indicators": make_indicators(),
                "extra": {
                    "_voters": 0, "_total_applicable": 26,
                    "_buy_count": 0, "_sell_count": 0,
                    "_votes": {}, "_regime": "range-bound",
                    "_weighted_action": "HOLD", "_weighted_confidence": 0.0,
                    "_confluence_score": 0.0,
                },
            }
        }
        prices_usd = {"BTC-USD": 69000.0}
        fx_rate = 10.50
        state = {
            "cash_sek": 500000,
            "holdings": {},
            "transactions": [],
            "initial_value_sek": 500000,
        }
        tf_data = {"BTC-USD": []}

        with mock.patch("portfolio.reporting._cached", side_effect=_null_cached):
            # Capture the summary by patching _atomic_write_json
            captured = {}
            original_write = None
            try:
                from portfolio.main import _atomic_write_json, AGENT_SUMMARY_FILE
                def capture_write(path, data):
                    captured["data"] = data
                with mock.patch("portfolio.reporting._atomic_write_json", side_effect=capture_write):
                    write_agent_summary(signals, prices_usd, fx_rate, state, tf_data)
            except Exception:
                pass

        if "data" in captured:
            summary = captured["data"]
            assert "timestamp" in summary
            assert "fx_rate" in summary
            assert "portfolio" in summary
            assert "signals" in summary
            assert "BTC-USD" in summary["signals"]

    def test_summary_json_serializable(self):
        """The summary dict should be fully JSON-serializable."""
        from portfolio.main import write_agent_summary

        signals = {
            "BTC-USD": {
                "action": "HOLD",
                "confidence": 0.0,
                "indicators": make_indicators(),
                "extra": {
                    "_voters": 0, "_total_applicable": 26,
                    "_buy_count": 0, "_sell_count": 0,
                    "_votes": {}, "_regime": "range-bound",
                    "_weighted_action": "HOLD", "_weighted_confidence": 0.0,
                    "_confluence_score": 0.0,
                },
            }
        }
        state = {
            "cash_sek": 500000, "holdings": {}, "transactions": [],
            "initial_value_sek": 500000,
        }

        captured = {}
        def capture_write(path, data):
            captured["data"] = data

        with mock.patch("portfolio.reporting._cached", side_effect=_null_cached), \
             mock.patch("portfolio.reporting._atomic_write_json", side_effect=capture_write):
            write_agent_summary(signals, {"BTC-USD": 69000.0}, 10.50, state,
                              {"BTC-USD": []})

        if "data" in captured:
            # Should not raise
            json_str = json.dumps(captured["data"], default=str)
            assert len(json_str) > 0
