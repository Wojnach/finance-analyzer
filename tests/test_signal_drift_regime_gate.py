"""Tests for drift_regime_gate signal module."""

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.drift_regime_gate import (
    MIN_ROWS,
    _drift_fraction,
    _drift_velocity,
    _price_vs_sma,
    compute_drift_regime_gate_signal,
)


def _make_df(n=100, seed=42):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_trending_up_df(n=100):
    """Create a DataFrame with a strong uptrend (most bars close higher)."""
    np.random.seed(42)
    # 80% of bars close higher → drift fraction > 0.60
    close = [100.0]
    for _ in range(n - 1):
        if np.random.random() < 0.80:
            close.append(close[-1] + abs(np.random.randn() * 0.3))
        else:
            close.append(close[-1] - abs(np.random.randn() * 0.1))
    close = np.array(close)
    return pd.DataFrame({
        "open": close - np.random.randn(n) * 0.05,
        "high": close + abs(np.random.randn(n) * 0.2),
        "low": close - abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_trending_down_df(n=100):
    """Create a DataFrame with a strong downtrend (most bars close lower)."""
    np.random.seed(42)
    close = [100.0]
    for _ in range(n - 1):
        if np.random.random() < 0.80:
            close.append(close[-1] - abs(np.random.randn() * 0.3))
        else:
            close.append(close[-1] + abs(np.random.randn() * 0.1))
    close = np.array(close)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.05,
        "high": close + abs(np.random.randn(n) * 0.2),
        "low": close - abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_drift_regime_gate_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_drift_regime_gate_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        result = compute_drift_regime_gate_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_drift_regime_gate_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_drift_regime_gate_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_drift_regime_gate_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_drift_regime_gate_signal(df, context=ctx)
        assert isinstance(result, dict)

    def test_none_dataframe_returns_hold(self):
        result = compute_drift_regime_gate_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_confidence_capped_at_0_7(self):
        df = _make_df(n=200)
        result = compute_drift_regime_gate_signal(df)
        assert result["confidence"] <= 0.7


class TestDriftFraction:
    """Test the drift fraction sub-indicator."""

    def test_uptrend_high_fraction(self):
        df = _make_trending_up_df(n=100)
        frac, vote = _drift_fraction(df["close"])
        assert frac > 0.60
        assert vote == "SELL"

    def test_downtrend_low_fraction(self):
        df = _make_trending_down_df(n=100)
        frac, vote = _drift_fraction(df["close"])
        assert frac < 0.40
        assert vote == "BUY"

    def test_random_walk_hold(self):
        df = _make_df(n=200, seed=123)
        frac, vote = _drift_fraction(df["close"])
        # Random walk should be ~0.50, which is HOLD
        assert 0.35 <= frac <= 0.65

    def test_insufficient_data(self):
        close = pd.Series([100.0, 101.0, 99.0])
        frac, vote = _drift_fraction(close)
        assert np.isnan(frac)
        assert vote == "HOLD"


class TestDriftVelocity:
    """Test the drift velocity sub-indicator."""

    def test_returns_float_and_vote(self):
        df = _make_df(n=200)
        vel, vote = _drift_velocity(df["close"])
        assert isinstance(vel, float)
        assert vote in ("BUY", "SELL", "HOLD")

    def test_insufficient_data(self):
        close = pd.Series([100.0] * 10)
        vel, vote = _drift_velocity(close)
        assert np.isnan(vel)
        assert vote == "HOLD"


class TestPriceVsSMA:
    """Test the price vs SMA sub-indicator."""

    def test_uptrend_positive_distance(self):
        df = _make_trending_up_df(n=100)
        dist, vote = _price_vs_sma(df["close"], df["high"], df["low"])
        assert dist > 0  # price above SMA in uptrend

    def test_downtrend_negative_distance(self):
        df = _make_trending_down_df(n=100)
        dist, vote = _price_vs_sma(df["close"], df["high"], df["low"])
        assert dist < 0  # price below SMA in downtrend


class TestComposite:
    """Test composite signal logic."""

    def test_uptrend_emits_sell(self):
        df = _make_trending_up_df(n=200)
        result = compute_drift_regime_gate_signal(df)
        # Strong uptrend should trigger SELL (mean reversion)
        assert result["action"] == "SELL"

    def test_downtrend_emits_buy(self):
        df = _make_trending_down_df(n=200)
        result = compute_drift_regime_gate_signal(df)
        # Strong downtrend should trigger BUY (contrarian)
        assert result["action"] == "BUY"

    def test_indicators_contain_all_values(self):
        df = _make_df(n=200)
        result = compute_drift_regime_gate_signal(df)
        assert "drift_fraction" in result["indicators"]
        assert "drift_velocity" in result["indicators"]
        assert "price_vs_sma_atr" in result["indicators"]

    def test_sub_signals_contain_all_votes(self):
        df = _make_df(n=200)
        result = compute_drift_regime_gate_signal(df)
        assert "drift_fraction" in result["sub_signals"]
        assert "drift_velocity" in result["sub_signals"]
        assert "price_vs_sma" in result["sub_signals"]

    def test_min_rows_constant(self):
        assert MIN_ROWS == 65
