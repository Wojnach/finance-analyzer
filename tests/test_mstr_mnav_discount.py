"""Tests for MSTR mNAV discount signal module."""
import unittest.mock as mock

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.mstr_mnav_discount import (
    MAX_CONFIDENCE,
    MNAV_BUY,
    MNAV_SELL,
    MNAV_STRONG_BUY,
    _BTC_HOLDINGS,
    _SHARES_OUTSTANDING,
    _discount_depth_zscore,
    _mnav_level,
    _mnav_ratio,
    _mnav_velocity,
    compute_mstr_mnav_discount_signal,
)


class TestMnavRatio:
    def test_basic(self):
        ratio = _mnav_ratio(155.0, 75000.0)
        expected = (155.0 * _SHARES_OUTSTANDING) / (_BTC_HOLDINGS * 75000.0)
        assert abs(ratio - expected) < 1e-6

    def test_zero_btc_price(self):
        assert np.isnan(_mnav_ratio(155.0, 0))

    def test_zero_mstr_price(self):
        assert np.isnan(_mnav_ratio(0, 75000.0))

    def test_negative_prices(self):
        assert np.isnan(_mnav_ratio(-10, 75000.0))
        assert np.isnan(_mnav_ratio(155.0, -1))


class TestMnavLevel:
    def test_deep_discount_buy(self):
        _, vote = _mnav_level(0.80)
        assert vote == "BUY"

    def test_moderate_discount_buy(self):
        _, vote = _mnav_level(0.93)
        assert vote == "BUY"

    def test_premium_sell(self):
        _, vote = _mnav_level(1.60)
        assert vote == "SELL"

    def test_strong_premium_sell(self):
        _, vote = _mnav_level(2.5)
        assert vote == "SELL"

    def test_fair_value_hold(self):
        _, vote = _mnav_level(1.1)
        assert vote == "HOLD"

    def test_nan_hold(self):
        _, vote = _mnav_level(float("nan"))
        assert vote == "HOLD"


class TestMnavVelocity:
    def test_falling_ratio_buy(self):
        ratios = [1.2, 1.15, 1.10, 1.05, 1.0, 0.95, 0.90]
        _, vote = _mnav_velocity(ratios, window=5)
        assert vote == "BUY"

    def test_rising_ratio_sell(self):
        ratios = [0.9, 0.95, 1.0, 1.1, 1.2, 1.3, 1.5]
        _, vote = _mnav_velocity(ratios, window=5)
        assert vote == "SELL"

    def test_flat_hold(self):
        ratios = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        _, vote = _mnav_velocity(ratios, window=5)
        assert vote == "HOLD"

    def test_insufficient_data_hold(self):
        ratios = [1.0, 0.9]
        _, vote = _mnav_velocity(ratios, window=5)
        assert vote == "HOLD"


class TestDiscountDepthZscore:
    def test_deep_discount_buy(self):
        ratios = [1.1] * 60 + [0.7]
        _, vote = _discount_depth_zscore(ratios, lookback=60)
        assert vote == "BUY"

    def test_high_premium_sell(self):
        np.random.seed(0)
        ratios = list(1.0 + np.random.randn(60) * 0.05) + [2.0]
        _, vote = _discount_depth_zscore(ratios, lookback=60)
        assert vote == "SELL"

    def test_normal_hold(self):
        ratios = [1.0] * 60 + [1.0]
        _, vote = _discount_depth_zscore(ratios, lookback=60)
        assert vote == "HOLD"

    def test_insufficient_data(self):
        ratios = [1.0] * 10
        _, vote = _discount_depth_zscore(ratios, lookback=60)
        assert vote == "HOLD"


def _make_mstr_df(n=100, price=155.0):
    """Create a simple MSTR OHLCV DataFrame."""
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    np.random.seed(42)
    noise = np.random.randn(n) * 2
    closes = price + np.cumsum(noise)
    closes[-1] = price
    return pd.DataFrame({
        "open": closes - 0.5,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "volume": np.random.randint(100000, 1000000, n),
    }, index=dates)


class TestComputeSignal:
    @mock.patch("portfolio.signals.mstr_mnav_discount._fetch_btc_price", return_value=75000.0)
    def test_basic_signal(self, _mock_btc):
        df = _make_mstr_df(100, price=155.0)
        result = compute_mstr_mnav_discount_signal(df, {"ticker": "MSTR"})
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= MAX_CONFIDENCE
        assert "mnav_ratio" in result["indicators"]
        assert "btc_price" in result["indicators"]
        assert result["indicators"]["btc_holdings"] == _BTC_HOLDINGS

    @mock.patch("portfolio.signals.mstr_mnav_discount._fetch_btc_price", return_value=75000.0)
    def test_non_mstr_ticker_returns_hold(self, _mock_btc):
        df = _make_mstr_df(100, price=155.0)
        result = compute_mstr_mnav_discount_signal(df, {"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    @mock.patch("portfolio.signals.mstr_mnav_discount._fetch_btc_price", return_value=None)
    def test_btc_price_unavailable_returns_hold(self, _mock_btc):
        df = _make_mstr_df(100, price=155.0)
        result = compute_mstr_mnav_discount_signal(df, {"ticker": "MSTR"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_none_df_returns_hold(self):
        result = compute_mstr_mnav_discount_signal(None, {"ticker": "MSTR"})
        assert result["action"] == "HOLD"

    def test_short_df_returns_hold(self):
        df = _make_mstr_df(5, price=155.0)
        result = compute_mstr_mnav_discount_signal(df, {"ticker": "MSTR"})
        assert result["action"] == "HOLD"

    @mock.patch("portfolio.signals.mstr_mnav_discount._fetch_btc_price", return_value=75000.0)
    def test_deep_discount_produces_buy(self, _mock_btc):
        """At 0.85x mNAV, signal should produce BUY."""
        target_ratio = 0.85
        btc_nav = _BTC_HOLDINGS * 75000.0
        mstr_price = target_ratio * btc_nav / _SHARES_OUTSTANDING
        df = _make_mstr_df(100, price=mstr_price)
        df["close"] = mstr_price
        result = compute_mstr_mnav_discount_signal(df, {"ticker": "MSTR"})
        assert result["indicators"]["mnav_ratio"] < MNAV_BUY
        assert result["sub_signals"]["mnav_level"] == "BUY"

    @mock.patch("portfolio.signals.mstr_mnav_discount._fetch_btc_price", return_value=75000.0)
    def test_high_premium_produces_sell(self, _mock_btc):
        """At 2.0x mNAV, signal should produce SELL."""
        target_ratio = 2.0
        btc_nav = _BTC_HOLDINGS * 75000.0
        mstr_price = target_ratio * btc_nav / _SHARES_OUTSTANDING
        df = _make_mstr_df(100, price=mstr_price)
        df["close"] = mstr_price
        result = compute_mstr_mnav_discount_signal(df, {"ticker": "MSTR"})
        assert result["indicators"]["mnav_ratio"] >= MNAV_SELL
        assert result["sub_signals"]["mnav_level"] == "SELL"

    @mock.patch("portfolio.signals.mstr_mnav_discount._fetch_btc_price", return_value=75000.0)
    def test_confidence_capped(self, _mock_btc):
        df = _make_mstr_df(100, price=155.0)
        result = compute_mstr_mnav_discount_signal(df, {"ticker": "MSTR"})
        assert result["confidence"] <= MAX_CONFIDENCE

    @mock.patch("portfolio.signals.mstr_mnav_discount._fetch_btc_price", return_value=75000.0)
    def test_sub_signals_present(self, _mock_btc):
        df = _make_mstr_df(100, price=155.0)
        result = compute_mstr_mnav_discount_signal(df, {"ticker": "MSTR"})
        assert "mnav_level" in result["sub_signals"]
        assert "mnav_velocity" in result["sub_signals"]
        assert "discount_depth_zscore" in result["sub_signals"]

    @mock.patch("portfolio.signals.mstr_mnav_discount._fetch_btc_price", return_value=75000.0)
    def test_indicators_complete(self, _mock_btc):
        df = _make_mstr_df(100, price=155.0)
        result = compute_mstr_mnav_discount_signal(df, {"ticker": "MSTR"})
        ind = result["indicators"]
        assert "mnav_ratio" in ind
        assert "mnav_velocity_5d" in ind
        assert "discount_depth_z" in ind
        assert "btc_price" in ind
        assert "mstr_price" in ind
        assert ind["shares_outstanding"] == _SHARES_OUTSTANDING
        assert ind["holdings_updated"] == "2026-05-26"

    @mock.patch("portfolio.signals.mstr_mnav_discount._fetch_btc_price", return_value=75000.0)
    def test_no_context_ticker_runs(self, _mock_btc):
        """Empty context (no ticker) should still run — applies to all."""
        df = _make_mstr_df(100, price=155.0)
        result = compute_mstr_mnav_discount_signal(df, {})
        assert result["action"] in ("BUY", "SELL", "HOLD")
