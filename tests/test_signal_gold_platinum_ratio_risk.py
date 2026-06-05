"""Tests for gold_platinum_ratio_risk signal module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from portfolio.signals.gold_platinum_ratio_risk import (
    compute_gold_platinum_ratio_risk_signal,
    _gp_zscore,
    _gp_trend,
    _gp_momentum,
    _gold_plat_spread,
    _CACHE,
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


def _make_gp_combined(n=300, trend="flat"):
    np.random.seed(42)
    gold = 2000 + np.cumsum(np.random.randn(n) * 5)
    if trend == "rising_gp":
        plat = 900 - np.cumsum(np.random.randn(n) * 2 + 0.5)
    elif trend == "falling_gp":
        plat = 900 + np.cumsum(np.random.randn(n) * 2 + 1.0)
    else:
        plat = 900 + np.cumsum(np.random.randn(n) * 3)
    combined = pd.DataFrame({
        "gold": gold,
        "platinum": plat,
    })
    combined["gp_ratio"] = np.log(combined["gold"]) - np.log(combined["platinum"])
    return combined


@pytest.fixture(autouse=True)
def _clear_cache():
    _CACHE.clear()
    yield
    _CACHE.clear()


class TestSignalInterface:
    def _patch_fetch(self, combined):
        return patch(
            "portfolio.signals.gold_platinum_ratio_risk._fetch_gp_data",
            return_value=combined,
        )

    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        combined = _make_gp_combined()
        with self._patch_fetch(combined):
            result = compute_gold_platinum_ratio_risk_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        combined = _make_gp_combined()
        with self._patch_fetch(combined):
            result = compute_gold_platinum_ratio_risk_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        expected_keys = {"gp_zscore", "gp_trend", "gp_momentum", "gold_plat_spread"}
        assert set(result["sub_signals"].keys()) == expected_keys

    def test_has_indicators(self):
        df = _make_df()
        combined = _make_gp_combined()
        with self._patch_fetch(combined):
            result = compute_gold_platinum_ratio_risk_signal(df)
        assert "indicators" in result
        assert "gp_ratio" in result["indicators"]
        assert "gold_price" in result["indicators"]
        assert "platinum_price" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_gold_platinum_ratio_risk_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_gold_platinum_ratio_risk_signal(df)
        assert result["action"] == "HOLD"

    def test_none_dataframe_returns_hold(self):
        result = compute_gold_platinum_ratio_risk_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_fetch_failure_returns_hold(self):
        df = _make_df()
        with patch(
            "portfolio.signals.gold_platinum_ratio_risk._fetch_gp_data",
            return_value=None,
        ):
            result = compute_gold_platinum_ratio_risk_signal(df)
        assert result["action"] == "HOLD"

    def test_confidence_capped_at_0_7(self):
        df = _make_df()
        combined = _make_gp_combined(n=300, trend="rising_gp")
        with self._patch_fetch(combined):
            result = compute_gold_platinum_ratio_risk_signal(df)
        assert result["confidence"] <= 0.7

    def test_metals_inversion(self):
        df = _make_df()
        combined = _make_gp_combined(n=300, trend="rising_gp")
        ctx_risk = {"ticker": "BTC-USD", "asset_class": "crypto"}
        ctx_metals = {"ticker": "XAU-USD", "asset_class": "metals"}
        with self._patch_fetch(combined):
            risk_result = compute_gold_platinum_ratio_risk_signal(df, context=ctx_risk)
        with self._patch_fetch(combined):
            metals_result = compute_gold_platinum_ratio_risk_signal(df, context=ctx_metals)
        if risk_result["action"] != "HOLD" and metals_result["action"] != "HOLD":
            assert risk_result["action"] != metals_result["action"]

    def test_with_context(self):
        df = _make_df()
        combined = _make_gp_combined()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        with self._patch_fetch(combined):
            result = compute_gold_platinum_ratio_risk_signal(df, context=ctx)
        assert isinstance(result, dict)


class TestSubIndicators:
    def test_gp_zscore_normal_range(self):
        combined = _make_gp_combined(n=300)
        z = _gp_zscore(combined["gp_ratio"], window=252)
        assert -5.0 < z < 5.0

    def test_gp_zscore_short_series(self):
        combined = _make_gp_combined(n=30)
        z = _gp_zscore(combined["gp_ratio"], window=252)
        assert isinstance(z, float)

    def test_gp_trend_values(self):
        combined = _make_gp_combined(n=300)
        t = _gp_trend(combined["gp_ratio"])
        assert t in (-1, 0, 1)

    def test_gp_trend_short_series(self):
        combined = _make_gp_combined(n=30)
        t = _gp_trend(combined["gp_ratio"])
        assert t in (-1, 0, 1)

    def test_gp_momentum(self):
        combined = _make_gp_combined(n=100)
        m = _gp_momentum(combined["gp_ratio"], periods=21)
        assert isinstance(m, float)

    def test_gp_momentum_short(self):
        combined = _make_gp_combined(n=5)
        m = _gp_momentum(combined["gp_ratio"], periods=21)
        assert m == 0.0

    def test_gold_plat_spread(self):
        combined = _make_gp_combined(n=100)
        s = _gold_plat_spread(combined, periods=20)
        assert isinstance(s, float)

    def test_gold_plat_spread_short(self):
        combined = _make_gp_combined(n=5)
        s = _gold_plat_spread(combined, periods=20)
        assert s == 0.0
