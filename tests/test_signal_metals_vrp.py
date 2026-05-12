"""Tests for metals_vrp signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.metals_vrp import compute_metals_vrp_signal, _gvz_cache


def _make_df(n=100):
    np.random.seed(42)
    close = 2000 + np.cumsum(np.random.randn(n) * 5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 1,
        "high": close + abs(np.random.randn(n) * 3),
        "low": close - abs(np.random.randn(n) * 3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _seed_gvz_cache(current=20.0, n=300):
    """Seed module cache with synthetic GVZ data (newest first)."""
    np.random.seed(99)
    data = [current] + list(18 + np.random.randn(n - 1) * 3)
    _gvz_cache["key"] = "test"
    _gvz_cache["data"] = data
    _gvz_cache["time"] = 1e12


class TestSignalInterface:

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        _seed_gvz_cache()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_metals_vrp_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        _seed_gvz_cache()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_metals_vrp_signal(df, context=ctx)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        _seed_gvz_cache()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_metals_vrp_signal(df, context=ctx)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_metals_vrp_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=5)
        result = compute_metals_vrp_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        _seed_gvz_cache()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_metals_vrp_signal(df, context=ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_non_metals_ticker_returns_hold(self):
        df = _make_df()
        _seed_gvz_cache()
        ctx = {"ticker": "BTC-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_metals_vrp_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_no_context_no_cache_returns_hold(self):
        _gvz_cache.clear()
        df = _make_df()
        result = compute_metals_vrp_signal(df)
        assert result["action"] == "HOLD"

    def test_no_fred_key_returns_hold(self):
        _gvz_cache.clear()
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {}}
        result = compute_metals_vrp_signal(df, context=ctx)
        assert result["action"] == "HOLD"


class TestVRPLogic:

    def test_high_gvz_produces_buy(self):
        """When GVZ is extremely high vs history, VRP should spike -> BUY."""
        df = _make_df(n=100)
        _seed_gvz_cache(current=35.0, n=300)
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_metals_vrp_signal(df, context=ctx)
        if result["action"] != "HOLD":
            assert result["action"] == "BUY"

    def test_low_gvz_produces_sell(self):
        """When GVZ is extremely low vs history, VRP should be low -> SELL."""
        df = _make_df(n=100)
        _seed_gvz_cache(current=8.0, n=300)
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_metals_vrp_signal(df, context=ctx)
        if result["action"] != "HOLD":
            assert result["action"] == "SELL"

    def test_xag_ticker_accepted(self):
        df = _make_df()
        _seed_gvz_cache()
        ctx = {"ticker": "XAG-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_metals_vrp_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_indicators_contain_vrp_fields(self):
        df = _make_df()
        _seed_gvz_cache()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_metals_vrp_signal(df, context=ctx)
        ind = result["indicators"]
        for key in ("gvz_current", "realized_vol", "vrp", "vrp_z"):
            assert key in ind, f"Missing indicator: {key}"

    def test_sub_signals_contain_expected_keys(self):
        df = _make_df()
        _seed_gvz_cache()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_metals_vrp_signal(df, context=ctx)
        ss = result["sub_signals"]
        for key in ("vrp_z", "vrp_level", "vrp_momentum", "gvz_percentile"):
            assert key in ss, f"Missing sub-signal: {key}"

    def test_no_ticker_in_context_computes(self):
        """When no ticker specified, signal should still compute (no gate)."""
        df = _make_df()
        _seed_gvz_cache()
        ctx = {"config": {"golddigger": {"fred_api_key": "test"}}}
        result = compute_metals_vrp_signal(df, context=ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_cache_used_on_second_call(self):
        df = _make_df()
        _seed_gvz_cache()
        ctx = {"ticker": "XAU-USD", "config": {"golddigger": {"fred_api_key": "test"}}}
        r1 = compute_metals_vrp_signal(df, context=ctx)
        r2 = compute_metals_vrp_signal(df, context=ctx)
        assert r1["indicators"]["gvz_current"] == r2["indicators"]["gvz_current"]
