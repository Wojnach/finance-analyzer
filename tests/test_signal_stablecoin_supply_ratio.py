"""Tests for stablecoin_supply_ratio signal module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from portfolio.signals.stablecoin_supply_ratio import (
    compute_stablecoin_supply_ratio_signal,
    _ssr_level,
    _supply_momentum,
    _supply_price_divergence,
)


def _make_df(n=100):
    np.random.seed(42)
    close = 2500 + np.cumsum(np.random.randn(n) * 10)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 2,
        "high": close + abs(np.random.randn(n) * 5),
        "low": close - abs(np.random.randn(n) * 5),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_supply_series(n=200, base=80e9, trend=0.001):
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    values = base * (1 + np.cumsum(np.random.randn(n) * 0.002 + trend))
    return pd.Series(values, index=dates, name="stablecoin_supply")


class TestSignalInterface:
    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        supply = _make_supply_series()
        with patch(
            "portfolio.signals.stablecoin_supply_ratio._fetch_stablecoin_supply",
            return_value=supply,
        ):
            result = compute_stablecoin_supply_ratio_signal(
                df, context={"ticker": "ETH-USD"}
            )
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        supply = _make_supply_series()
        with patch(
            "portfolio.signals.stablecoin_supply_ratio._fetch_stablecoin_supply",
            return_value=supply,
        ):
            result = compute_stablecoin_supply_ratio_signal(
                df, context={"ticker": "ETH-USD"}
            )
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "ssr_level" in result["sub_signals"]
        assert "supply_momentum" in result["sub_signals"]
        assert "price_supply_divergence" in result["sub_signals"]

    def test_has_indicators(self):
        df = _make_df()
        supply = _make_supply_series()
        with patch(
            "portfolio.signals.stablecoin_supply_ratio._fetch_stablecoin_supply",
            return_value=supply,
        ):
            result = compute_stablecoin_supply_ratio_signal(
                df, context={"ticker": "ETH-USD"}
            )
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)
        assert "ssr_zscore" in result["indicators"]
        assert "supply_7d_roc" in result["indicators"]
        assert "divergence" in result["indicators"]

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_stablecoin_supply_ratio_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        result = compute_stablecoin_supply_ratio_signal(df)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        supply = _make_supply_series()
        with patch(
            "portfolio.signals.stablecoin_supply_ratio._fetch_stablecoin_supply",
            return_value=supply,
        ):
            result = compute_stablecoin_supply_ratio_signal(
                df, context={"ticker": "ETH-USD"}
            )
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_with_context(self):
        df = _make_df()
        supply = _make_supply_series()
        with patch(
            "portfolio.signals.stablecoin_supply_ratio._fetch_stablecoin_supply",
            return_value=supply,
        ):
            result = compute_stablecoin_supply_ratio_signal(
                df,
                context={
                    "ticker": "ETH-USD",
                    "asset_class": "crypto",
                    "regime": "trending-up",
                },
            )
        assert isinstance(result, dict)

    def test_non_applicable_ticker_returns_hold(self):
        df = _make_df()
        result = compute_stablecoin_supply_ratio_signal(
            df, context={"ticker": "XAU-USD"}
        )
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_btc_ticker_accepted(self):
        df = _make_df()
        supply = _make_supply_series()
        with patch(
            "portfolio.signals.stablecoin_supply_ratio._fetch_stablecoin_supply",
            return_value=supply,
        ):
            result = compute_stablecoin_supply_ratio_signal(
                df, context={"ticker": "BTC-USD"}
            )
        assert isinstance(result, dict)
        assert result["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_capped(self):
        df = _make_df()
        supply = _make_supply_series()
        with patch(
            "portfolio.signals.stablecoin_supply_ratio._fetch_stablecoin_supply",
            return_value=supply,
        ):
            result = compute_stablecoin_supply_ratio_signal(
                df, context={"ticker": "ETH-USD"}
            )
        assert result["confidence"] <= 0.7

    def test_supply_fetch_failure_returns_hold(self):
        df = _make_df()
        with patch(
            "portfolio.signals.stablecoin_supply_ratio._fetch_stablecoin_supply",
            return_value=None,
        ):
            result = compute_stablecoin_supply_ratio_signal(
                df, context={"ticker": "ETH-USD"}
            )
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0


class TestSubIndicators:
    def test_ssr_level_buy_on_low_z(self):
        np.random.seed(42)
        vals = list(np.random.randn(100) * 0.1 + 5.0)
        vals.append(2.0)
        series = pd.Series(vals)
        z, vote = _ssr_level(series, lookback=90)
        assert vote == "BUY"
        assert z < Z_BUY

    def test_ssr_level_sell_on_high_z(self):
        np.random.seed(42)
        vals = list(np.random.randn(100) * 0.1 + 5.0)
        vals.append(8.0)
        series = pd.Series(vals)
        z, vote = _ssr_level(series, lookback=90)
        assert vote == "SELL"
        assert z > 1.0

    def test_ssr_level_hold_on_normal(self):
        np.random.seed(42)
        vals = list(np.random.randn(100) * 0.1 + 5.0)
        vals.append(5.05)
        series = pd.Series(vals)
        z, vote = _ssr_level(series, lookback=90)
        assert vote == "HOLD"

    def test_supply_momentum_buy_on_growth(self):
        vals = [80e9 + i * 1e9 for i in range(10)]
        series = pd.Series(vals)
        roc, vote = _supply_momentum(series, window=7)
        assert vote == "BUY"
        assert roc > 0

    def test_supply_momentum_sell_on_decline(self):
        vals = [80e9 - i * 1e9 for i in range(10)]
        series = pd.Series(vals)
        roc, vote = _supply_momentum(series, window=7)
        assert vote == "SELL"
        assert roc < 0

    def test_divergence_buy(self):
        supply = pd.Series([80e9 + i * 0.5e9 for i in range(10)])
        price = pd.Series([2500 - i * 50 for i in range(10)])
        div, vote = _supply_price_divergence(supply, price, window=7)
        assert vote == "BUY"

    def test_divergence_sell(self):
        supply = pd.Series([80e9 - i * 0.5e9 for i in range(10)])
        price = pd.Series([2500 + i * 50 for i in range(10)])
        div, vote = _supply_price_divergence(supply, price, window=7)
        assert vote == "SELL"


Z_BUY = -1.5
