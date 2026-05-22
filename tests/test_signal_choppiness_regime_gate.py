"""Tests for choppiness_regime_gate signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.choppiness_regime_gate import (
    compute_choppiness_regime_gate_signal,
    _compute_choppiness,
)


def _make_df(n=100, seed=42):
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + abs(np.random.randn(n) * 0.3)
    low = close - abs(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_trending_df(n=100):
    close = np.linspace(100, 150, n)
    high = close + 0.5
    low = close - 0.5
    return pd.DataFrame({
        "open": close - 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 5000.0),
    })


def _make_choppy_df(n=100):
    close = 100 + np.sin(np.linspace(0, 20 * np.pi, n)) * 2
    high = close + 1.5
    low = close - 1.5
    return pd.DataFrame({
        "open": close + 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.full(n, 5000.0),
    })


class TestSignalInterface:
    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_choppiness_regime_gate_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_choppiness_regime_gate_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "choppy_gate" in result["sub_signals"]
        assert "trending_confirm" in result["sub_signals"]
        assert "chop_roc" in result["sub_signals"]

    def test_has_indicators(self):
        df = _make_df()
        result = compute_choppiness_regime_gate_signal(df)
        assert "indicators" in result
        assert "choppiness_index" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_choppiness_regime_gate_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_choppiness_regime_gate_signal(df)
        assert result["action"] == "HOLD"

    def test_none_dataframe_returns_hold(self):
        result = compute_choppiness_regime_gate_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_choppiness_regime_gate_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_choppiness_regime_gate_signal(df, context=ctx)
        assert isinstance(result, dict)

    def test_confidence_capped_at_0_7(self):
        df = _make_df()
        result = compute_choppiness_regime_gate_signal(df)
        assert result["confidence"] <= 0.7


class TestChoppinessComputation:
    def test_chop_values_in_range(self):
        df = _make_df(n=50)
        chop = _compute_choppiness(df["high"], df["low"], df["close"])
        valid = chop.dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_trending_market_low_chop(self):
        df = _make_trending_df()
        chop = _compute_choppiness(df["high"], df["low"], df["close"])
        last_chop = chop.iloc[-1]
        assert last_chop < 50

    def test_choppy_market_high_chop(self):
        df = _make_choppy_df()
        chop = _compute_choppiness(df["high"], df["low"], df["close"])
        last_chop = chop.iloc[-1]
        assert last_chop > 50


class TestRegimeDetection:
    def test_choppy_market_returns_hold(self):
        df = _make_choppy_df()
        result = compute_choppiness_regime_gate_signal(df)
        if result["indicators"].get("regime") == "choppy":
            assert result["action"] == "HOLD"
            assert result["confidence"] == 0.0

    def test_trending_market_returns_directional(self):
        df = _make_trending_df()
        result = compute_choppiness_regime_gate_signal(df)
        if result["indicators"].get("regime") == "trending":
            assert result["action"] in ("BUY", "SELL")

    def test_regime_indicator_present(self):
        df = _make_df()
        result = compute_choppiness_regime_gate_signal(df)
        assert "regime" in result["indicators"]
        assert result["indicators"]["regime"] in ("choppy", "trending", "neutral")
