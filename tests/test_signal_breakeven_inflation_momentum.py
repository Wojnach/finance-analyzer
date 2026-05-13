"""Tests for breakeven_inflation_momentum signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.breakeven_inflation_momentum import (
    compute_breakeven_inflation_momentum_signal,
    _bei_cache,
    _compute_change_zscore,
    _compute_acceleration,
)


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


def _seed_bei_cache(current=2.3, n=300, trend=0.0):
    """Seed module cache with synthetic BEI data (newest first)."""
    np.random.seed(99)
    base = [current - trend * i + np.random.randn() * 0.02 for i in range(n)]
    _bei_cache["key"] = "test"
    _bei_cache["data"] = base
    _bei_cache["time"] = 1e12


def _ctx(ticker="XAU-USD"):
    return {"ticker": ticker, "config": {"golddigger": {"fred_api_key": "test"}}}


class TestSignalInterface:

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        _seed_bei_cache()
        result = compute_breakeven_inflation_momentum_signal(df, context=_ctx())
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        _seed_bei_cache()
        result = compute_breakeven_inflation_momentum_signal(df, context=_ctx())
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        _seed_bei_cache()
        result = compute_breakeven_inflation_momentum_signal(df, context=_ctx())
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "bei_current" in result["indicators"]
        assert "bei_z" in result["indicators"]
        assert "bei_20d_change" in result["indicators"]
        assert "bei_accel" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_breakeven_inflation_momentum_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_none_dataframe_returns_hold(self):
        result = compute_breakeven_inflation_momentum_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=5)
        result = compute_breakeven_inflation_momentum_signal(df)
        assert result["action"] == "HOLD"

    def test_confidence_capped_at_0_7(self):
        df = _make_df()
        _seed_bei_cache(current=3.0, trend=0.005)
        result = compute_breakeven_inflation_momentum_signal(df, context=_ctx())
        assert result["confidence"] <= 0.7


class TestTickerFiltering:

    def test_non_applicable_ticker_returns_hold(self):
        df = _make_df()
        _seed_bei_cache()
        result = compute_breakeven_inflation_momentum_signal(
            df, context=_ctx(ticker="MSTR"),
        )
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_eth_not_applicable(self):
        df = _make_df()
        _seed_bei_cache()
        result = compute_breakeven_inflation_momentum_signal(
            df, context=_ctx(ticker="ETH-USD"),
        )
        assert result["action"] == "HOLD"

    @pytest.mark.parametrize("ticker", [
        "XAU-USD", "XAG-USD", "BTC-USD",
        "XAUUSD", "XAGUSD", "BTCUSD",
    ])
    def test_applicable_tickers_produce_signal(self, ticker):
        df = _make_df()
        _seed_bei_cache()
        result = compute_breakeven_inflation_momentum_signal(
            df, context=_ctx(ticker=ticker),
        )
        assert "sub_signals" in result
        if result["action"] != "HOLD":
            assert result["confidence"] > 0.0

    def test_empty_ticker_still_computes(self):
        df = _make_df()
        _seed_bei_cache()
        result = compute_breakeven_inflation_momentum_signal(
            df, context={"config": {"golddigger": {"fred_api_key": "test"}}},
        )
        assert "sub_signals" in result


class TestSubSignals:

    def test_three_sub_signals(self):
        df = _make_df()
        _seed_bei_cache()
        result = compute_breakeven_inflation_momentum_signal(df, context=_ctx())
        subs = result["sub_signals"]
        assert "bei_momentum" in subs
        assert "bei_level" in subs
        assert "bei_acceleration" in subs
        for v in subs.values():
            assert v in ("BUY", "SELL", "HOLD")

    def test_high_bei_level_triggers_buy(self):
        df = _make_df()
        _seed_bei_cache(current=3.0)
        result = compute_breakeven_inflation_momentum_signal(df, context=_ctx())
        assert result["sub_signals"]["bei_level"] == "BUY"

    def test_low_bei_level_triggers_sell(self):
        df = _make_df()
        _seed_bei_cache(current=1.2)
        result = compute_breakeven_inflation_momentum_signal(df, context=_ctx())
        assert result["sub_signals"]["bei_level"] == "SELL"

    def test_mid_bei_level_is_hold(self):
        df = _make_df()
        _seed_bei_cache(current=2.0)
        result = compute_breakeven_inflation_momentum_signal(df, context=_ctx())
        assert result["sub_signals"]["bei_level"] == "HOLD"


class TestMathHelpers:

    def test_zscore_insufficient_data_returns_zero(self):
        values = [2.0] * 10
        assert _compute_change_zscore(values, 20, 60) == 0.0

    def test_zscore_flat_series_returns_zero(self):
        values = [2.0] * 200
        z = _compute_change_zscore(values, 20, 60)
        assert z == 0.0

    def test_zscore_trending_series_nonzero(self):
        np.random.seed(77)
        values = [2.0 + 0.01 * i + np.random.randn() * 0.05 for i in range(200)]
        z = _compute_change_zscore(values, 20, 60)
        assert z != 0.0

    def test_acceleration_insufficient_data_returns_zero(self):
        values = [2.0] * 5
        assert _compute_acceleration(values, 10) == 0.0

    def test_acceleration_flat_is_zero(self):
        values = [2.0] * 100
        assert _compute_acceleration(values, 10) == 0.0

    def test_acceleration_accelerating_series(self):
        values = [2.0 + 0.001 * i ** 1.5 for i in range(100)]
        accel = _compute_acceleration(values, 10)
        assert accel != 0.0


class TestCacheAndNoKey:

    def test_no_fred_key_returns_hold_when_cache_empty(self):
        _bei_cache.clear()
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {}}
        result = compute_breakeven_inflation_momentum_signal(df, context=ctx)
        assert result["action"] == "HOLD"

    def test_stale_cache_still_used_without_key(self):
        _seed_bei_cache()
        _bei_cache["time"] = 0
        df = _make_df()
        ctx = {"ticker": "XAU-USD", "config": {}}
        result = compute_breakeven_inflation_momentum_signal(df, context=ctx)
        assert "sub_signals" in result

    def test_insufficient_bei_values_returns_hold(self):
        _bei_cache["key"] = "test"
        _bei_cache["data"] = [2.0] * 10
        _bei_cache["time"] = 1e12
        df = _make_df()
        result = compute_breakeven_inflation_momentum_signal(df, context=_ctx())
        assert result["action"] == "HOLD"
