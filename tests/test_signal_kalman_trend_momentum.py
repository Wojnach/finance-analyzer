"""Tests for kalman_trend_momentum signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.kalman_trend_momentum import (
    compute_kalman_trend_momentum_signal,
    _run_kalman,
    _sub_velocity_direction,
)


def _make_df(n=200, trend=0.0, noise=1.0, seed=42):
    """Create test DataFrame with optional trend and noise."""
    np.random.seed(seed)
    close = 100 + trend * np.arange(n) + np.cumsum(np.random.randn(n) * noise)
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
        result = compute_kalman_trend_momentum_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_kalman_trend_momentum_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "velocity_direction" in result["sub_signals"]
        assert "trend_regime" in result["sub_signals"]
        assert "price_divergence" in result["sub_signals"]
        assert "velocity_momentum" in result["sub_signals"]

    def test_has_indicators(self):
        df = _make_df()
        result = compute_kalman_trend_momentum_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "kalman_filtered_price" in result["indicators"]
        assert "kalman_velocity" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_kalman_trend_momentum_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_kalman_trend_momentum_signal(df)
        assert result["action"] == "HOLD"

    def test_none_dataframe_returns_hold(self):
        result = compute_kalman_trend_momentum_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[80:85, df.columns.get_loc("close")] = np.nan
        result = compute_kalman_trend_momentum_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "asset_class": "metals", "regime": "trending-up"}
        result = compute_kalman_trend_momentum_signal(df, context=ctx)
        assert isinstance(result, dict)

    def test_sub_signals_are_valid_actions(self):
        df = _make_df()
        result = compute_kalman_trend_momentum_signal(df)
        for sub_name, sub_val in result["sub_signals"].items():
            assert sub_val in ("BUY", "SELL", "HOLD"), f"{sub_name} has invalid value {sub_val}"


class TestKalmanFilter:
    def test_kalman_output_shape(self):
        close = np.array([100.0 + i * 0.1 for i in range(100)])
        fp, vel = _run_kalman(close, 0.01, 0.001, 1.0)
        assert len(fp) == 100
        assert len(vel) == 100

    def test_kalman_trending_up_positive_velocity(self):
        close = np.array([100.0 + i * 0.5 for i in range(200)])
        fp, vel = _run_kalman(close, 0.01, 0.001, 1.0)
        assert vel[-1] > 0, "Uptrend should produce positive velocity"

    def test_kalman_trending_down_negative_velocity(self):
        close = np.array([200.0 - i * 0.5 for i in range(200)])
        fp, vel = _run_kalman(close, 0.01, 0.001, 1.0)
        assert vel[-1] < 0, "Downtrend should produce negative velocity"

    def test_kalman_flat_near_zero_velocity(self):
        np.random.seed(42)
        close = 100 + np.random.randn(200) * 0.1
        fp, vel = _run_kalman(close, 0.01, 0.001, 1.0)
        assert abs(vel[-1]) < 0.5, "Flat series should have near-zero velocity"

    def test_kalman_filtered_smoother_than_raw(self):
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(200) * 2)
        fp, vel = _run_kalman(close, 0.01, 0.001, 1.0)
        raw_std = np.std(np.diff(close))
        filt_std = np.std(np.diff(fp))
        assert filt_std < raw_std, "Filtered price should be smoother"


class TestDirectionalBehavior:
    def test_strong_uptrend_signals_buy(self):
        df = _make_df(n=200, trend=0.5, noise=0.1)
        result = compute_kalman_trend_momentum_signal(df)
        assert result["action"] == "BUY", f"Strong uptrend should BUY, got {result['action']}"

    def test_strong_downtrend_signals_sell(self):
        df = _make_df(n=200, trend=-0.5, noise=0.1)
        result = compute_kalman_trend_momentum_signal(df)
        assert result["action"] == "SELL", f"Strong downtrend should SELL, got {result['action']}"

    def test_ranging_market_low_conviction(self):
        results = []
        for seed in range(10):
            df = _make_df(n=200, trend=0.0, noise=0.5, seed=seed)
            r = compute_kalman_trend_momentum_signal(df)
            results.append(r)
        hold_or_low = sum(1 for r in results if r["action"] == "HOLD" or r["confidence"] < 0.6)
        assert hold_or_low >= 3, f"Ranging markets should have some HOLD/low-conf, got {hold_or_low}/10"

    def test_confidence_higher_for_stronger_trend(self):
        df_weak = _make_df(n=200, trend=0.1, noise=0.5)
        df_strong = _make_df(n=200, trend=1.0, noise=0.1)
        r_weak = compute_kalman_trend_momentum_signal(df_weak)
        r_strong = compute_kalman_trend_momentum_signal(df_strong)
        if r_strong["action"] != "HOLD" and r_weak["action"] != "HOLD":
            assert r_strong["confidence"] >= r_weak["confidence"]


class TestEdgeCases:
    def test_constant_price(self):
        df = pd.DataFrame({
            "open": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "close": [100.0] * 100,
            "volume": [5000.0] * 100,
        })
        result = compute_kalman_trend_momentum_signal(df)
        assert result["action"] == "HOLD"

    def test_single_large_jump(self):
        close = np.array([100.0] * 100 + [120.0] * 100)
        df = pd.DataFrame({
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(200, 5000.0),
        })
        result = compute_kalman_trend_momentum_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_minimum_rows_exactly(self):
        df = _make_df(n=60)
        result = compute_kalman_trend_momentum_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_very_small_prices(self):
        df = _make_df(n=200, noise=0.001)
        df["close"] = df["close"] * 0.001
        df["open"] = df["open"] * 0.001
        df["high"] = df["high"] * 0.001
        df["low"] = df["low"] * 0.001
        result = compute_kalman_trend_momentum_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_very_large_prices(self):
        df = _make_df(n=200)
        df["close"] = df["close"] * 100000
        df["open"] = df["open"] * 100000
        df["high"] = df["high"] * 100000
        df["low"] = df["low"] * 100000
        result = compute_kalman_trend_momentum_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")
