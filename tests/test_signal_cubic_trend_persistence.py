"""Tests for cubic_trend_persistence signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.cubic_trend_persistence import (
    compute_cubic_trend_persistence_signal,
    _compute_weights,
    _compute_phi,
    _expected_return,
)


def _make_df(n=200, seed=42):
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_trending_df(n=200, direction=1):
    np.random.seed(42)
    trend = np.linspace(0, direction * 20, n)
    noise = np.random.randn(n) * 0.3
    close = 100 + trend + noise
    close = np.maximum(close, 1.0)
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
        result = compute_cubic_trend_persistence_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_cubic_trend_persistence_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "trend_direction" in result["sub_signals"]
        assert "cubic_expected" in result["sub_signals"]
        assert "trend_exhaustion" in result["sub_signals"]

    def test_has_indicators(self):
        df = _make_df()
        result = compute_cubic_trend_persistence_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "phi" in result["indicators"]
        assert "expected_return" in result["indicators"]
        assert "sigma" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_cubic_trend_persistence_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_cubic_trend_persistence_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_cubic_trend_persistence_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_cubic_trend_persistence_signal(df, context=ctx)
        assert isinstance(result, dict)

    def test_confidence_capped_at_0_7(self):
        df = _make_df()
        result = compute_cubic_trend_persistence_signal(df)
        assert result["confidence"] <= 0.7


class TestInternalFunctions:

    def test_weights_normalization(self):
        w = _compute_weights(60, 60)
        assert abs(np.sum(w**2) - 1.0) < 1e-8

    def test_weights_shape(self):
        w = _compute_weights(60, 60)
        assert len(w) == 60
        assert w[0] > 0

    def test_weights_peak_at_half_T(self):
        w = _compute_weights(60, 60)
        peak_idx = np.argmax(w)
        assert 25 <= peak_idx <= 35

    def test_phi_zero_for_flat_returns(self):
        returns = np.zeros(60)
        w = _compute_weights(60, 60)
        phi = _compute_phi(returns, w)
        assert phi == 0.0

    def test_phi_capped(self):
        returns = np.ones(60) * 100.0
        w = _compute_weights(60, 60)
        phi = _compute_phi(returns, w)
        assert phi <= 2.5

    def test_expected_return_positive_for_weak_trend(self):
        e_r = _expected_return(0.5, 0.0129, -0.0062)
        assert e_r > 0

    def test_expected_return_negative_for_strong_trend(self):
        e_r = _expected_return(2.0, 0.0129, -0.0062)
        assert e_r < 0

    def test_expected_return_zero_crossing(self):
        b, c = 0.0129, -0.0062
        phi_zero = (-b / c) ** 0.5
        assert 1.0 < phi_zero < 2.0
        assert _expected_return(phi_zero - 0.1, b, c) > 0
        assert _expected_return(phi_zero + 0.1, b, c) < 0


class TestDirectionalBehavior:

    def test_uptrend_positive_phi(self):
        df = _make_trending_df(n=200, direction=1)
        result = compute_cubic_trend_persistence_signal(df)
        assert result["indicators"]["phi"] > 0

    def test_strong_uptrend_predicts_reversion(self):
        df = _make_trending_df(n=200, direction=1)
        result = compute_cubic_trend_persistence_signal(df)
        phi = result["indicators"]["phi"]
        if phi > 1.5:
            assert result["sub_signals"]["trend_exhaustion"] in ("SELL", "HOLD")

    def test_downtrend_negative_phi(self):
        df = _make_trending_df(n=200, direction=-1)
        result = compute_cubic_trend_persistence_signal(df)
        assert result["indicators"]["phi"] < 0

    def test_strong_downtrend_predicts_reversion(self):
        df = _make_trending_df(n=200, direction=-1)
        result = compute_cubic_trend_persistence_signal(df)
        phi = result["indicators"]["phi"]
        if phi < -1.5:
            assert result["sub_signals"]["trend_exhaustion"] in ("BUY", "HOLD")

    def test_all_sub_signals_valid(self):
        df = _make_df()
        result = compute_cubic_trend_persistence_signal(df)
        for name, vote in result["sub_signals"].items():
            assert vote in ("BUY", "SELL", "HOLD"), f"{name} has invalid vote: {vote}"
