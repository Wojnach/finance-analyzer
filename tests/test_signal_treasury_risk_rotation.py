"""Tests for treasury_risk_rotation signal module."""
import numpy as np
import pandas as pd
import pytest

from portfolio.signals.treasury_risk_rotation import (
    compute_treasury_risk_rotation_signal,
    _compute_spread_series,
    _sub_slope_direction,
    _sub_slope_zscore,
    _sub_regime_persistence,
    _invert,
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


class TestSignalInterface:

    def test_returns_dict_with_required_keys(self, monkeypatch):
        ief = pd.Series(np.linspace(90, 95, 300), name="IEF")
        tlt = pd.Series(np.linspace(100, 110, 300), name="TLT")
        monkeypatch.setattr(
            "portfolio.signals.treasury_risk_rotation._fetch_treasury_data",
            lambda: {"ief": ief, "tlt": tlt},
        )
        df = _make_df()
        result = compute_treasury_risk_rotation_signal(df)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self, monkeypatch):
        ief = pd.Series(np.linspace(90, 95, 300), name="IEF")
        tlt = pd.Series(np.linspace(100, 110, 300), name="TLT")
        monkeypatch.setattr(
            "portfolio.signals.treasury_risk_rotation._fetch_treasury_data",
            lambda: {"ief": ief, "tlt": tlt},
        )
        df = _make_df()
        result = compute_treasury_risk_rotation_signal(df)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "slope_direction" in result["sub_signals"]
        assert "slope_momentum" in result["sub_signals"]
        assert "slope_zscore" in result["sub_signals"]
        assert "regime_persistence" in result["sub_signals"]

    def test_has_indicators(self, monkeypatch):
        ief = pd.Series(np.linspace(90, 95, 300), name="IEF")
        tlt = pd.Series(np.linspace(100, 110, 300), name="TLT")
        monkeypatch.setattr(
            "portfolio.signals.treasury_risk_rotation._fetch_treasury_data",
            lambda: {"ief": ief, "tlt": tlt},
        )
        df = _make_df()
        result = compute_treasury_risk_rotation_signal(df)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "spread_65d" in result["indicators"]
        assert "zscore" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_treasury_risk_rotation_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_treasury_risk_rotation_signal(df)
        assert result["action"] == "HOLD"

    def test_none_dataframe_returns_hold(self):
        result = compute_treasury_risk_rotation_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_fetch_failure_returns_hold(self, monkeypatch):
        monkeypatch.setattr(
            "portfolio.signals.treasury_risk_rotation._fetch_treasury_data",
            lambda: None,
        )
        df = _make_df()
        result = compute_treasury_risk_rotation_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_confidence_capped_at_0_7(self, monkeypatch):
        ief = pd.Series(np.linspace(100, 90, 300), name="IEF")
        tlt = pd.Series(np.linspace(100, 120, 300), name="TLT")
        monkeypatch.setattr(
            "portfolio.signals.treasury_risk_rotation._fetch_treasury_data",
            lambda: {"ief": ief, "tlt": tlt},
        )
        df = _make_df()
        result = compute_treasury_risk_rotation_signal(df)
        assert result["confidence"] <= 0.7


class TestSafeHavenInversion:

    def test_safe_haven_inverts_action(self, monkeypatch):
        ief = pd.Series(np.linspace(100, 90, 300), name="IEF")
        tlt = pd.Series(np.linspace(100, 120, 300), name="TLT")
        monkeypatch.setattr(
            "portfolio.signals.treasury_risk_rotation._fetch_treasury_data",
            lambda: {"ief": ief, "tlt": tlt},
        )
        df = _make_df()

        risk_on = compute_treasury_risk_rotation_signal(df, {"ticker": "BTC-USD"})
        safe_haven = compute_treasury_risk_rotation_signal(df, {"ticker": "XAU-USD"})

        if risk_on["action"] in ("BUY", "SELL"):
            expected = "SELL" if risk_on["action"] == "BUY" else "BUY"
            assert safe_haven["action"] == expected

    def test_xag_is_safe_haven(self, monkeypatch):
        ief = pd.Series(np.linspace(100, 90, 300), name="IEF")
        tlt = pd.Series(np.linspace(100, 120, 300), name="TLT")
        monkeypatch.setattr(
            "portfolio.signals.treasury_risk_rotation._fetch_treasury_data",
            lambda: {"ief": ief, "tlt": tlt},
        )
        df = _make_df()
        result = compute_treasury_risk_rotation_signal(df, {"ticker": "XAG-USD"})
        assert result["indicators"]["is_safe_haven"] is True

    def test_btc_is_not_safe_haven(self, monkeypatch):
        ief = pd.Series(np.linspace(90, 95, 300), name="IEF")
        tlt = pd.Series(np.linspace(100, 110, 300), name="TLT")
        monkeypatch.setattr(
            "portfolio.signals.treasury_risk_rotation._fetch_treasury_data",
            lambda: {"ief": ief, "tlt": tlt},
        )
        df = _make_df()
        result = compute_treasury_risk_rotation_signal(df, {"ticker": "BTC-USD"})
        assert result["indicators"]["is_safe_haven"] is False


class TestSubIndicators:

    def test_slope_direction_buy(self):
        assert _sub_slope_direction(0.05) == "BUY"

    def test_slope_direction_sell(self):
        assert _sub_slope_direction(-0.05) == "SELL"

    def test_slope_direction_hold(self):
        assert _sub_slope_direction(0.005) == "HOLD"
        assert _sub_slope_direction(-0.005) == "HOLD"

    def test_zscore_buy(self):
        spread = pd.Series(np.concatenate([np.zeros(200), [2.0]]))
        z, vote = _sub_slope_zscore(spread)
        assert vote == "BUY"
        assert z > 1.0

    def test_zscore_sell(self):
        spread = pd.Series(np.concatenate([np.zeros(200), [-2.0]]))
        z, vote = _sub_slope_zscore(spread)
        assert vote == "SELL"
        assert z < -1.0

    def test_zscore_hold_neutral(self):
        spread = pd.Series(np.zeros(200))
        z, vote = _sub_slope_zscore(spread)
        assert vote == "HOLD"

    def test_zscore_insufficient_data(self):
        spread = pd.Series([0.01, 0.02])
        z, vote = _sub_slope_zscore(spread)
        assert vote == "HOLD"

    def test_persistence_buy(self):
        spread = pd.Series([0.01] * 10)
        assert _sub_regime_persistence(spread) == "BUY"

    def test_persistence_sell(self):
        spread = pd.Series([-0.01] * 10)
        assert _sub_regime_persistence(spread) == "SELL"

    def test_persistence_hold_recent_flip(self):
        spread = pd.Series([-0.01, -0.01, 0.01, 0.01])
        assert _sub_regime_persistence(spread) == "HOLD"


class TestHelpers:

    def test_invert_buy(self):
        assert _invert("BUY") == "SELL"

    def test_invert_sell(self):
        assert _invert("SELL") == "BUY"

    def test_invert_hold(self):
        assert _invert("HOLD") == "HOLD"

    def test_compute_spread_series(self):
        ief = pd.Series(np.linspace(100, 105, 100))
        tlt = pd.Series(np.linspace(100, 110, 100))
        spread = _compute_spread_series(ief, tlt)
        spread_clean = spread.dropna()
        assert len(spread_clean) > 0
        assert float(spread_clean.iloc[-1]) > 0


class TestWithContext:

    def test_with_full_context(self, monkeypatch):
        ief = pd.Series(np.linspace(90, 95, 300), name="IEF")
        tlt = pd.Series(np.linspace(100, 110, 300), name="TLT")
        monkeypatch.setattr(
            "portfolio.signals.treasury_risk_rotation._fetch_treasury_data",
            lambda: {"ief": ief, "tlt": tlt},
        )
        df = _make_df()
        ctx = {"ticker": "BTC-USD", "asset_class": "crypto", "regime": "trending-up"}
        result = compute_treasury_risk_rotation_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_without_context(self, monkeypatch):
        ief = pd.Series(np.linspace(90, 95, 300), name="IEF")
        tlt = pd.Series(np.linspace(100, 110, 300), name="TLT")
        monkeypatch.setattr(
            "portfolio.signals.treasury_risk_rotation._fetch_treasury_data",
            lambda: {"ief": ief, "tlt": tlt},
        )
        df = _make_df()
        result = compute_treasury_risk_rotation_signal(df)
        assert isinstance(result, dict)
        assert result["indicators"]["is_safe_haven"] is False
