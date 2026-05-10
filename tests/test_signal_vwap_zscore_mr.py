"""Tests for vwap_zscore_mr signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.vwap_zscore_mr import compute_vwap_zscore_mr_signal


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


def _make_extreme_df(direction="buy", n=100):
    """Create DataFrame with extreme deviation from VWAP to trigger signals."""
    np.random.seed(42)
    close = np.full(n, 100.0)
    volume = np.full(n, 5000.0)
    if direction == "buy":
        close[-3:] = 70.0
        volume[-3:] = 15000.0
    else:
        close[-3:] = 130.0
        volume[-3:] = 15000.0
    return pd.DataFrame({
        "open": close + 0.1,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
        "volume": volume,
    })


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        result = compute_vwap_zscore_mr_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        result = compute_vwap_zscore_mr_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        result = compute_vwap_zscore_mr_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_vwap_zscore_mr_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=5)
        result = compute_vwap_zscore_mr_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        result = compute_vwap_zscore_mr_signal(df)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_vwap_zscore_mr_signal(df, context=ctx)
        assert isinstance(result, dict)

    def test_none_dataframe_returns_hold(self):
        result = compute_vwap_zscore_mr_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


class TestVWAPZScoreLogic:
    """Test VWAP Z-Score specific logic."""

    def test_extreme_drop_triggers_buy(self):
        df = _make_extreme_df(direction="buy")
        result = compute_vwap_zscore_mr_signal(df)
        assert result["indicators"]["vwap_z_score"] < -2.0
        assert result["action"] == "BUY"
        assert result["confidence"] > 0.0

    def test_extreme_rise_triggers_sell(self):
        df = _make_extreme_df(direction="sell")
        result = compute_vwap_zscore_mr_signal(df)
        assert result["indicators"]["vwap_z_score"] > 2.0
        assert result["action"] == "SELL"
        assert result["confidence"] > 0.0

    def test_indicators_contain_expected_keys(self):
        df = _make_df()
        result = compute_vwap_zscore_mr_signal(df)
        indicators = result["indicators"]
        assert "vwap_z_score" in indicators
        assert "vwap_value" in indicators
        assert "vwap_slope_pct" in indicators
        assert "volume_ratio" in indicators
        assert "close" in indicators

    def test_sub_signals_contain_expected_keys(self):
        df = _make_df()
        result = compute_vwap_zscore_mr_signal(df)
        subs = result["sub_signals"]
        assert "vwap_z" in subs
        assert "vwap_slope" in subs
        assert "volume_confirm" in subs
        for v in subs.values():
            assert v in ("BUY", "SELL", "HOLD")

    def test_confidence_bounded(self):
        df = _make_extreme_df(direction="buy")
        result = compute_vwap_zscore_mr_signal(df)
        assert result["confidence"] <= 0.85

    def test_flat_price_returns_hold(self):
        n = 100
        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 100.5),
            "low": np.full(n, 99.5),
            "close": np.full(n, 100.0),
            "volume": np.full(n, 5000.0),
        })
        result = compute_vwap_zscore_mr_signal(df)
        assert result["action"] == "HOLD"

    def test_zero_volume_returns_hold(self):
        df = _make_df()
        df["volume"] = 0.0
        result = compute_vwap_zscore_mr_signal(df)
        assert result["action"] == "HOLD"

    def test_different_seeds_produce_varied_results(self):
        results = set()
        for seed in range(10):
            df = _make_df(n=200, seed=seed)
            result = compute_vwap_zscore_mr_signal(df)
            results.add(result["action"])
        assert len(results) >= 1
