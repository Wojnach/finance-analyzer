"""Tests for connors_rsi2 signal module."""

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.connors_rsi2 import (
    MIN_ROWS,
    MAX_CONFIDENCE,
    RSI2_BUY,
    RSI2_SELL,
    _rsi2_level,
    _close_streak,
    _price_vs_sma5,
    compute_connors_rsi2_signal,
)


def _make_df(n=50, seed=42):
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_oversold_df(n=50):
    """Sharp selloff at end → RSI(2) < 10."""
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n - 5) * 0.2)
    tail = [close[-1] - 3.0 * (i + 1) for i in range(5)]
    close = np.concatenate([close, tail])
    return pd.DataFrame({
        "open": close + 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 5000.0),
    })


def _make_overbought_df(n=50):
    """Sharp rally at end → RSI(2) > 90."""
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n - 5) * 0.2)
    tail = [close[-1] + 3.0 * (i + 1) for i in range(5)]
    close = np.concatenate([close, tail])
    return pd.DataFrame({
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": np.full(n, 5000.0),
    })


class TestSignalInterface:
    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_connors_rsi2_signal(df, context={"ticker": "BTC-USD"})
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_connors_rsi2_signal(df, context={"ticker": "BTC-USD"})
        if "sub_signals" in result:
            subs = result["sub_signals"]
            assert "rsi2_level" in subs
            assert "close_streak" in subs
            assert "price_vs_sma5" in subs

    def test_confidence_capped(self):
        df = _make_df()
        result = compute_connors_rsi2_signal(df, context={"ticker": "BTC-USD"})
        assert result["confidence"] <= MAX_CONFIDENCE

    def test_insufficient_data(self):
        df = _make_df(n=3)
        result = compute_connors_rsi2_signal(df, context={"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_none_dataframe(self):
        result = compute_connors_rsi2_signal(None, context={"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"


class TestTickerFiltering:
    def test_non_crypto_returns_hold(self):
        df = _make_df()
        result = compute_connors_rsi2_signal(df, context={"ticker": "XAU-USD"})
        assert result["action"] == "HOLD"
        assert result.get("feature_unavailable") is True

    def test_btc_allowed(self):
        df = _make_df()
        result = compute_connors_rsi2_signal(df, context={"ticker": "BTC-USD"})
        assert "feature_unavailable" not in result or not result["feature_unavailable"]

    def test_eth_allowed(self):
        df = _make_df()
        result = compute_connors_rsi2_signal(df, context={"ticker": "ETH-USD"})
        assert "feature_unavailable" not in result or not result["feature_unavailable"]

    def test_mstr_excluded(self):
        df = _make_df()
        result = compute_connors_rsi2_signal(df, context={"ticker": "MSTR"})
        assert result.get("feature_unavailable") is True

    def test_xag_excluded(self):
        df = _make_df()
        result = compute_connors_rsi2_signal(df, context={"ticker": "XAG-USD"})
        assert result.get("feature_unavailable") is True


class TestSubIndicators:
    def test_rsi2_level_oversold(self):
        df = _make_oversold_df()
        val, sig = _rsi2_level(df["close"])
        assert val < RSI2_BUY
        assert sig == "BUY"

    def test_rsi2_level_overbought(self):
        df = _make_overbought_df()
        val, sig = _rsi2_level(df["close"])
        assert val > RSI2_SELL
        assert sig == "SELL"

    def test_rsi2_level_insufficient_data(self):
        val, sig = _rsi2_level(pd.Series([100.0, 101.0]))
        assert np.isnan(val)
        assert sig == "HOLD"

    def test_close_streak_down(self):
        close = pd.Series([100, 99, 98, 97, 96, 95], dtype=float)
        val, sig = _close_streak(close)
        assert val < 0
        assert sig == "BUY"

    def test_close_streak_up(self):
        close = pd.Series([100, 101, 102, 103, 104, 105], dtype=float)
        val, sig = _close_streak(close)
        assert val > 0
        assert sig == "SELL"

    def test_close_streak_short(self):
        close = pd.Series([100, 99, 100], dtype=float)
        val, sig = _close_streak(close)
        assert sig == "HOLD"

    def test_price_vs_sma5_below(self):
        close = pd.Series([100, 100, 100, 100, 100, 100, 100, 95], dtype=float)
        val, sig = _price_vs_sma5(close)
        assert val < 0
        assert sig == "BUY"

    def test_price_vs_sma5_above(self):
        close = pd.Series([100, 100, 100, 100, 100, 100, 100, 105], dtype=float)
        val, sig = _price_vs_sma5(close)
        assert val > 0
        assert sig == "SELL"


class TestDirectionalSignals:
    def test_oversold_produces_buy(self):
        df = _make_oversold_df()
        result = compute_connors_rsi2_signal(df, context={"ticker": "BTC-USD"})
        assert result["action"] == "BUY"
        assert result["confidence"] > 0

    def test_overbought_produces_sell(self):
        df = _make_overbought_df()
        result = compute_connors_rsi2_signal(df, context={"ticker": "ETH-USD"})
        assert result["action"] == "SELL"
        assert result["confidence"] > 0
