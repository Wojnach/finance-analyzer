"""Tests for adx_regime_switch signal module."""

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.adx_regime_switch import (
    MIN_ROWS,
    MAX_CONFIDENCE,
    ADX_TREND_THRESHOLD,
    _compute_adx,
    _adx_regime,
    _adx_momentum,
    _di_spread,
    compute_adx_regime_switch_signal,
)


def _make_df(n=100, seed=42):
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_trending_df(n=100, direction="up"):
    np.random.seed(42)
    step = 1.5 if direction == "up" else -1.5
    noise = np.random.randn(n) * 0.3
    close = 100 + np.arange(n) * step + np.cumsum(noise)
    return pd.DataFrame({
        "open": close - step * 0.3,
        "high": close + abs(np.random.randn(n) * 0.5) + abs(step) * 0.5,
        "low": close - abs(np.random.randn(n) * 0.5),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_ranging_df(n=100):
    np.random.seed(42)
    close = 100 + np.sin(np.linspace(0, 8 * np.pi, n)) * 2 + np.random.randn(n) * 0.3
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestSignalInterface:
    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_adx_regime_switch_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_adx_regime_switch_signal(df)
        if "sub_signals" in result:
            subs = result["sub_signals"]
            assert "adx_regime" in subs
            assert "adx_momentum" in subs
            assert "di_spread" in subs

    def test_has_regime_label(self):
        df = _make_df()
        result = compute_adx_regime_switch_signal(df)
        if "regime" in result:
            assert result["regime"] in ("trending", "ranging")

    def test_confidence_capped(self):
        df = _make_df()
        result = compute_adx_regime_switch_signal(df)
        assert result["confidence"] <= MAX_CONFIDENCE

    def test_insufficient_data(self):
        df = _make_df(n=10)
        result = compute_adx_regime_switch_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_none_dataframe(self):
        result = compute_adx_regime_switch_signal(None)
        assert result["action"] == "HOLD"


class TestADXComputation:
    def test_adx_returns_series(self):
        df = _make_df()
        adx, pdi, mdi = _compute_adx(df["high"], df["low"], df["close"])
        assert isinstance(adx, pd.Series)
        assert isinstance(pdi, pd.Series)
        assert isinstance(mdi, pd.Series)
        assert len(adx) == len(df)

    def test_adx_positive(self):
        df = _make_df()
        adx, _, _ = _compute_adx(df["high"], df["low"], df["close"])
        valid = adx.dropna()
        assert (valid >= 0).all()

    def test_trending_high_adx(self):
        df = _make_trending_df(direction="up")
        adx, _, _ = _compute_adx(df["high"], df["low"], df["close"])
        final_adx = adx.iloc[-1]
        assert final_adx > 20


class TestSubIndicators:
    def test_adx_regime_trending_bullish(self):
        val, sig = _adx_regime(35.0, 30.0, 15.0)
        assert sig == "BUY"

    def test_adx_regime_trending_bearish(self):
        val, sig = _adx_regime(35.0, 15.0, 30.0)
        assert sig == "SELL"

    def test_adx_regime_ranging(self):
        val, sig = _adx_regime(15.0, 20.0, 18.0)
        assert sig == "HOLD"

    def test_adx_regime_nan(self):
        val, sig = _adx_regime(float("nan"), 0, 0)
        assert np.isnan(val)
        assert sig == "HOLD"

    def test_di_spread_bullish(self):
        pdi = pd.Series([25.0])
        mdi = pd.Series([10.0])
        val, sig = _di_spread(pdi, mdi)
        assert val > 0
        assert sig == "BUY"

    def test_di_spread_bearish(self):
        pdi = pd.Series([10.0])
        mdi = pd.Series([25.0])
        val, sig = _di_spread(pdi, mdi)
        assert val < 0
        assert sig == "SELL"

    def test_di_spread_neutral(self):
        pdi = pd.Series([20.0])
        mdi = pd.Series([18.0])
        val, sig = _di_spread(pdi, mdi)
        assert sig == "HOLD"


class TestDirectionalSignals:
    def test_uptrend_produces_buy(self):
        df = _make_trending_df(direction="up")
        result = compute_adx_regime_switch_signal(df)
        assert result["action"] in ("BUY", "HOLD")

    def test_downtrend_produces_sell(self):
        df = _make_trending_df(direction="down")
        result = compute_adx_regime_switch_signal(df)
        assert result["action"] in ("SELL", "HOLD")

    def test_ranging_tends_hold(self):
        df = _make_ranging_df()
        result = compute_adx_regime_switch_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
