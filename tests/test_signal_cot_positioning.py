"""Tests for the COT positioning signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.cot_positioning import (
    _compute_cot_index,
    _sub_cot_index,
    _sub_commercial_change,
    _sub_managed_money,
    _sub_real_yield,
    compute_cot_positioning_signal,
)


def _make_df(n=100):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 0.1,
        "high": close + abs(np.random.randn(n) * 0.3),
        "low": close - abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_cot_data(nc_net=150000, comm_net=-180000, mm_net=80000, change=5000):
    """Create mock COT data matching metals_precompute output."""
    return {
        "report_date": "2026-04-01",
        "open_interest": 400000,
        "noncomm_long": nc_net + 10000 if nc_net > 0 else 10000,
        "noncomm_short": 10000 if nc_net > 0 else abs(nc_net) + 10000,
        "noncomm_net": nc_net,
        "comm_long": 200000,
        "comm_short": 200000 - comm_net,
        "comm_net": comm_net,
        "noncomm_net_change": change,
        "managed_money_long": mm_net + 5000 if mm_net > 0 else 5000,
        "managed_money_short": 5000 if mm_net > 0 else abs(mm_net) + 5000,
        "managed_money_net": mm_net,
    }


def _make_historical(n=50, base_nc_net=100000, spread=50000):
    """Create mock historical COT data for z-score computation."""
    np.random.seed(42)
    return [
        {"nc_net": int(base_nc_net + np.random.randn() * spread), "mm_net": int(50000 + np.random.randn() * 30000)}
        for _ in range(n)
    ]


class TestCotIndex:
    """Test COT Index percentile computation."""

    def test_basic_percentile(self):
        history = [100, 50, 150, 25, 175, 0, 200, 80, 120, 60]
        index = _compute_cot_index(history)
        # Current = 100, min = 0, max = 200, range = 200
        # Index = (100 - 0) / 200 * 100 = 50.0
        assert index == 50.0

    def test_extreme_high(self):
        history = [190, 50, 100, 25, 0, 75, 150, 200, 30, 60]
        index = _compute_cot_index(history)
        # Current = 190, min = 0, max = 200, range = 200
        # Index = 190/200 * 100 = 95.0
        assert index == 95.0

    def test_extreme_low(self):
        history = [10, 50, 100, 150, 200, 75, 125, 175, 80, 90]
        index = _compute_cot_index(history)
        # Current = 10, min = 10, max = 200, range = 190
        # Index = (10-10)/190 * 100 = 0.0
        assert index == 0.0

    def test_insufficient_data(self):
        history = [100, 200, 50]
        index = _compute_cot_index(history)
        assert index is None

    def test_no_variation(self):
        history = [100] * 20
        index = _compute_cot_index(history)
        assert index == 50.0


class TestSubCotIndex:
    """Test COT Index sub-indicator."""

    def test_extreme_bullish_returns_sell(self):
        cot_data = _make_cot_data(nc_net=190000)
        # Historical: range 0-200000, current at 190000 = 95th percentile
        historical = [{"nc_net": i * 10000} for i in range(21)]  # 0 to 200000
        vote, conf, ind = _sub_cot_index(cot_data, historical)
        assert vote == "SELL"
        assert conf > 0.4
        assert ind["cot_index"] > 80

    def test_extreme_bearish_returns_buy(self):
        cot_data = _make_cot_data(nc_net=10000)
        historical = [{"nc_net": i * 10000} for i in range(21)]
        vote, conf, ind = _sub_cot_index(cot_data, historical)
        assert vote == "BUY"
        assert conf > 0.4
        assert ind["cot_index"] < 20

    def test_neutral_returns_hold(self):
        cot_data = _make_cot_data(nc_net=100000)
        historical = [{"nc_net": i * 10000} for i in range(21)]
        vote, conf, ind = _sub_cot_index(cot_data, historical)
        assert vote == "HOLD"
        assert ind["cot_index"] == 50.0

    def test_missing_nc_net(self):
        cot_data = {"report_date": "2026-04-01"}
        vote, conf, ind = _sub_cot_index(cot_data, [])
        assert vote == "HOLD"
        assert conf == 0.0


class TestSubCommercialChange:
    """Test commercial hedger change sub-indicator."""

    def test_large_spec_increase_returns_sell(self):
        cot_data = _make_cot_data(change=10000)
        vote, ind = _sub_commercial_change(cot_data)
        assert vote == "SELL"

    def test_large_spec_decrease_returns_buy(self):
        cot_data = _make_cot_data(change=-10000)
        vote, ind = _sub_commercial_change(cot_data)
        assert vote == "BUY"

    def test_small_change_returns_hold(self):
        cot_data = _make_cot_data(change=1000)
        vote, ind = _sub_commercial_change(cot_data)
        assert vote == "HOLD"

    def test_missing_data(self):
        cot_data = {"report_date": "2026-04-01"}
        vote, ind = _sub_commercial_change(cot_data)
        assert vote == "HOLD"


class TestSubManagedMoney:
    """Test managed money intensity sub-indicator."""

    def test_extreme_long_returns_sell(self):
        # mm_net way above historical mean
        cot_data = _make_cot_data(mm_net=200000)
        historical = _make_historical(n=50, base_nc_net=100000)
        vote, ind = _sub_managed_money(cot_data, historical)
        assert vote == "SELL"
        assert ind["mm_zscore"] > 1.5

    def test_extreme_short_returns_buy(self):
        cot_data = _make_cot_data(mm_net=-100000)
        historical = _make_historical(n=50, base_nc_net=100000)
        vote, ind = _sub_managed_money(cot_data, historical)
        assert vote == "BUY"
        assert ind["mm_zscore"] < -1.5

    def test_neutral_returns_hold(self):
        cot_data = _make_cot_data(mm_net=50000)
        historical = _make_historical(n=50, base_nc_net=100000)
        vote, ind = _sub_managed_money(cot_data, historical)
        assert vote == "HOLD"

    def test_missing_data(self):
        cot_data = {"report_date": "2026-04-01"}
        vote, ind = _sub_managed_money(cot_data, [])
        assert vote == "HOLD"


class TestSubRealYield:
    """Test real yield direction sub-indicator."""

    def test_falling_yields_returns_buy(self):
        deep_ctx = {
            "refresh_data": {
                "fred": {"real_yield": 1.5, "real_yield_direction": "falling"}
            }
        }
        vote, ind = _sub_real_yield(deep_ctx, "XAU-USD")
        assert vote == "BUY"
        assert ind["real_yield_direction"] == "falling"

    def test_rising_yields_returns_sell(self):
        deep_ctx = {
            "refresh_data": {
                "fred": {"real_yield": 2.0, "real_yield_direction": "rising"}
            }
        }
        vote, ind = _sub_real_yield(deep_ctx, "XAU-USD")
        assert vote == "SELL"

    def test_stable_yields_returns_hold(self):
        deep_ctx = {
            "refresh_data": {
                "fred": {"real_yield": 1.8, "real_yield_direction": "stable"}
            }
        }
        vote, ind = _sub_real_yield(deep_ctx, "XAU-USD")
        assert vote == "HOLD"

    def test_missing_context(self):
        vote, ind = _sub_real_yield(None, "XAU-USD")
        assert vote == "HOLD"

    def test_no_fred_data(self):
        deep_ctx = {"refresh_data": {}}
        vote, ind = _sub_real_yield(deep_ctx, "XAU-USD")
        assert vote == "HOLD"


class TestSignalInterface:
    """Test that the signal follows the standard interface."""

    def test_returns_dict_with_required_keys(self, monkeypatch):
        # Mock deep context loading
        cot_data = _make_cot_data(nc_net=100000)
        deep_ctx = {
            "refresh_data": {
                "cot_gold": cot_data,
                "fred": {"real_yield": 1.5, "real_yield_direction": "stable"},
            }
        }
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_deep_context",
            lambda t: deep_ctx,
        )
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_cot_history",
            lambda m: _make_historical(30),
        )
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._fetch_cot_historical",
            lambda c: [],
        )

        df = _make_df()
        result = compute_cot_positioning_signal(df, context={"ticker": "XAU-USD"})
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self, monkeypatch):
        cot_data = _make_cot_data()
        deep_ctx = {"refresh_data": {"cot_gold": cot_data, "fred": {}}}
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_deep_context",
            lambda t: deep_ctx,
        )
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_cot_history",
            lambda m: _make_historical(30),
        )
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._fetch_cot_historical",
            lambda c: [],
        )

        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAU-USD"})
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "cot_index" in result["sub_signals"]
        assert "commercial_change" in result["sub_signals"]
        assert "managed_money" in result["sub_signals"]
        assert "real_yield" in result["sub_signals"]

    def test_has_indicators(self, monkeypatch):
        cot_data = _make_cot_data()
        deep_ctx = {"refresh_data": {"cot_gold": cot_data, "fred": {}}}
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_deep_context",
            lambda t: deep_ctx,
        )
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_cot_history",
            lambda m: _make_historical(30),
        )
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._fetch_cot_historical",
            lambda c: [],
        )

        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAU-USD"})
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_non_metals_returns_hold(self):
        df = _make_df()
        result = compute_cot_positioning_signal(df, context={"ticker": "BTC-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_no_context_returns_hold(self):
        df = _make_df()
        result = compute_cot_positioning_signal(df)
        assert result["action"] == "HOLD"

    def test_confidence_capped_at_0_7(self, monkeypatch):
        # Set up extreme positioning that would normally give high confidence
        cot_data = _make_cot_data(nc_net=200000, change=20000, mm_net=300000)
        deep_ctx = {
            "refresh_data": {
                "cot_gold": cot_data,
                "fred": {"real_yield": 2.5, "real_yield_direction": "rising"},
            }
        }
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_deep_context",
            lambda t: deep_ctx,
        )
        # Historical with current being extreme
        hist = [{"nc_net": i * 10000, "mm_net": i * 5000} for i in range(21)]
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_cot_history",
            lambda m: hist,
        )
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._fetch_cot_historical",
            lambda c: [],
        )

        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAU-USD"})
        assert result["confidence"] <= 0.7

    def test_silver_ticker(self, monkeypatch):
        cot_data = _make_cot_data(nc_net=20000)
        deep_ctx = {"refresh_data": {"cot_silver": cot_data, "fred": {}}}
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_deep_context",
            lambda t: deep_ctx,
        )
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_cot_history",
            lambda m: _make_historical(30),
        )
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._fetch_cot_historical",
            lambda c: [],
        )

        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAG-USD"})
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_missing_deep_context_returns_hold(self, monkeypatch):
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_deep_context",
            lambda t: None,
        )
        result = compute_cot_positioning_signal(_make_df(), context={"ticker": "XAU-USD"})
        assert result["action"] == "HOLD"

    def test_empty_dataframe(self, monkeypatch):
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_deep_context",
            lambda t: None,
        )
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_cot_positioning_signal(df, context={"ticker": "XAU-USD"})
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_with_kwargs_ticker(self, monkeypatch):
        monkeypatch.setattr(
            "portfolio.signals.cot_positioning._load_deep_context",
            lambda t: None,
        )
        result = compute_cot_positioning_signal(_make_df(), ticker="BTC-USD")
        assert result["action"] == "HOLD"
