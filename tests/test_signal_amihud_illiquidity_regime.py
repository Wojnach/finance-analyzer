"""Tests for amihud_illiquidity_regime signal module."""

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.amihud_illiquidity_regime import (
    compute_amihud_illiquidity_regime_signal,
)


def _make_df(n=100, seed=42):
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.1,
            "high": close + abs(np.random.randn(n) * 0.3),
            "low": close - abs(np.random.randn(n) * 0.3),
            "close": close,
            "volume": np.random.randint(1000, 10000, n).astype(float),
        }
    )


class TestSignalInterface:
    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_amihud_illiquidity_regime_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_amihud_illiquidity_regime_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        result = compute_amihud_illiquidity_regime_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "illiq_z_score" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_amihud_illiquidity_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        result = compute_amihud_illiquidity_regime_signal(df)
        assert result["action"] == "HOLD"

    def test_none_input_returns_hold(self):
        result = compute_amihud_illiquidity_regime_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_amihud_illiquidity_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_zero_volume_handling(self):
        df = _make_df()
        df.iloc[80:90, df.columns.get_loc("volume")] = 0.0
        result = compute_amihud_illiquidity_regime_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_amihud_illiquidity_regime_signal(df, context=ctx)
        assert isinstance(result, dict)

    def test_confidence_capped_at_07(self):
        df = _make_df(n=200)
        result = compute_amihud_illiquidity_regime_signal(df)
        assert result["confidence"] <= 0.7

    def test_sub_signal_keys(self):
        df = _make_df()
        result = compute_amihud_illiquidity_regime_signal(df)
        if result["action"] != "HOLD" or result["sub_signals"]:
            expected_keys = {"illiq_z", "illiq_trend", "volume_confirm"}
            assert set(result["sub_signals"].keys()) == expected_keys


class TestLiquidityRegimeDetection:
    def test_thin_market_detection(self):
        """Illiquidity spike (very low volume + big moves) should produce SELL or z > 0."""
        df = _make_df(n=150)
        df.iloc[-10:, df.columns.get_loc("volume")] = 1.0
        base = df["close"].iloc[-11]
        df.iloc[-5:, df.columns.get_loc("close")] = base * np.array([1.10, 0.90, 1.12, 0.88, 1.15])
        result = compute_amihud_illiquidity_regime_signal(df)
        assert result["indicators"]["illiq_z_score"] > 0

    def test_thick_market_detection(self):
        """High volume + small moves should produce low illiquidity."""
        df = _make_df(n=150)
        df.iloc[-20:, df.columns.get_loc("volume")] = 100000.0
        stable_price = df["close"].iloc[-21]
        df.iloc[-20:, df.columns.get_loc("close")] = stable_price + np.random.randn(20) * 0.01
        result = compute_amihud_illiquidity_regime_signal(df)
        assert result["indicators"]["illiq_z_score"] < 1.0

    def test_deterministic_output(self):
        """Same input should produce same output."""
        df = _make_df(n=100, seed=123)
        r1 = compute_amihud_illiquidity_regime_signal(df.copy())
        r2 = compute_amihud_illiquidity_regime_signal(df.copy())
        assert r1["action"] == r2["action"]
        assert r1["confidence"] == r2["confidence"]
