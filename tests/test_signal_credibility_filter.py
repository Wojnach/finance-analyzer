"""Tests for signal_credibility_filter signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.signal_credibility_filter import (
    compute_signal_credibility_filter_signal,
)


def _make_df(n=100):
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_trending_df(n=100, direction=1):
    np.random.seed(42)
    trend = np.linspace(0, direction * 10, n)
    noise = np.random.randn(n) * 0.2
    close = 100 + trend + noise
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_spike_df(n=100, spike_idx=80):
    np.random.seed(42)
    close = np.full(n, 100.0) + np.random.randn(n) * 0.1
    close[spike_idx] += 5.0
    vol = np.full(n, 5000.0)
    vol[spike_idx] = 50000.0
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.05,
        "high": close + abs(np.random.randn(n) * 0.2),
        "low": close - abs(np.random.randn(n) * 0.2),
        "close": close,
        "volume": vol,
    })


class TestSignalInterface:
    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_signal_credibility_filter_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_signal_credibility_filter_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        result = compute_signal_credibility_filter_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_signal_credibility_filter_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_signal_credibility_filter_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_signal_credibility_filter_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_signal_credibility_filter_signal(df, context=ctx)
        assert isinstance(result, dict)

    def test_confidence_capped_at_0_7(self):
        df = _make_trending_df(n=100, direction=1)
        result = compute_signal_credibility_filter_signal(df)
        assert result["confidence"] <= 0.7


class TestSignalLogic:
    def test_trending_up_generates_buy_or_hold(self):
        df = _make_trending_df(n=100, direction=1)
        result = compute_signal_credibility_filter_signal(df)
        assert result["action"] in ("BUY", "HOLD")

    def test_trending_down_generates_sell_or_hold(self):
        df = _make_trending_df(n=100, direction=-1)
        result = compute_signal_credibility_filter_signal(df)
        assert result["action"] in ("SELL", "HOLD")

    def test_spike_with_concentrated_volume(self):
        df = _make_spike_df(n=100, spike_idx=90)
        result = compute_signal_credibility_filter_signal(df)
        if result["action"] != "HOLD":
            assert "volume_hhi" in result["indicators"]

    def test_sub_signals_present_when_active(self):
        df = _make_trending_df(n=100, direction=1)
        result = compute_signal_credibility_filter_signal(df)
        if result["action"] != "HOLD":
            subs = result["sub_signals"]
            assert "persistence" in subs
            assert "volume_distribution" in subs
            assert "follow_through" in subs

    def test_none_df_returns_hold(self):
        result = compute_signal_credibility_filter_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_zero_volume_handled(self):
        df = _make_df()
        df["volume"] = 0.0
        result = compute_signal_credibility_filter_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_flat_price_returns_hold(self):
        n = 100
        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.1),
            "low": np.full(n, 99.9),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 5000.0),
        })
        result = compute_signal_credibility_filter_signal(df)
        assert result["action"] == "HOLD"

    def test_persistence_indicator_range(self):
        df = _make_trending_df(n=100, direction=1)
        result = compute_signal_credibility_filter_signal(df)
        if "persistence_ratio" in result["indicators"]:
            pr = result["indicators"]["persistence_ratio"]
            assert -1.0 <= pr <= 3.0

    def test_hhi_indicator_range(self):
        df = _make_df()
        result = compute_signal_credibility_filter_signal(df)
        if "volume_hhi" in result["indicators"]:
            hhi = result["indicators"]["volume_hhi"]
            assert 0.0 <= hhi <= 1.0
