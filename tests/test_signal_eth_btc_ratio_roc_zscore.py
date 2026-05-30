"""Tests for eth_btc_ratio_roc_zscore signal module."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.eth_btc_ratio_roc_zscore import (
    MIN_ROWS,
    compute_eth_btc_ratio_roc_zscore_signal,
)


def _make_df(n=200, base=3000.0, seed=42):
    np.random.seed(seed)
    close = base + np.cumsum(np.random.randn(n) * 10)
    close = np.maximum(close, 100)
    return pd.DataFrame({
        "open": close + np.random.randn(n) * 1,
        "high": close + abs(np.random.randn(n) * 5),
        "low": close - abs(np.random.randn(n) * 5),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    })


def _make_context(ticker="ETH-USD", include_btc=True, n=200):
    ctx = {"ticker": ticker}
    if include_btc and ticker == "ETH-USD":
        ctx["all_prices"] = {"BTC-USD": _make_df(n=n, base=60000, seed=99)}
    elif include_btc and ticker == "BTC-USD":
        ctx["all_prices"] = {"ETH-USD": _make_df(n=n, base=3000, seed=99)}
    return ctx


class TestSignalInterface:
    def test_returns_dict_with_required_keys(self):
        df = _make_df()
        ctx = _make_context()
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_has_sub_signals(self):
        df = _make_df()
        ctx = _make_context()
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)

    def test_has_indicators(self):
        df = _make_df()
        ctx = _make_context()
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        ctx = _make_context()
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=10)
        ctx = _make_context(n=10)
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert result["action"] == "HOLD"

    def test_nan_handling(self):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        ctx = _make_context()
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")


class TestCryptoOnly:
    def test_non_crypto_ticker_returns_hold(self):
        df = _make_df()
        ctx = {"ticker": "XAG-USD"}
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_mstr_returns_hold(self):
        df = _make_df()
        ctx = {"ticker": "MSTR"}
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert result["action"] == "HOLD"


class TestNoContext:
    def test_none_context_returns_hold(self):
        df = _make_df()
        result = compute_eth_btc_ratio_roc_zscore_signal(df, None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_empty_context_returns_hold(self):
        df = _make_df()
        result = compute_eth_btc_ratio_roc_zscore_signal(df, {})
        assert result["action"] == "HOLD"


class TestMissingCounterpartyData:
    @patch("portfolio.signals.eth_btc_ratio_roc_zscore._fetch_counterparty_close", return_value=None)
    def test_eth_without_btc_prices_returns_hold(self, mock_fetch):
        df = _make_df()
        ctx = {"ticker": "ETH-USD", "all_prices": {}}
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.eth_btc_ratio_roc_zscore._fetch_counterparty_close", return_value=None)
    def test_btc_without_eth_prices_returns_hold(self, mock_fetch):
        df = _make_df(base=60000)
        ctx = {"ticker": "BTC-USD", "all_prices": {}}
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.eth_btc_ratio_roc_zscore._fetch_counterparty_close", return_value=None)
    def test_no_all_prices_key(self, mock_fetch):
        df = _make_df()
        ctx = {"ticker": "ETH-USD"}
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert result["action"] == "HOLD"


class TestBothTickers:
    def test_eth_returns_valid_signal(self):
        df = _make_df(n=200, base=3000)
        ctx = _make_context("ETH-USD", n=200)
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 0.7

    def test_btc_returns_valid_signal(self):
        df = _make_df(n=200, base=60000, seed=99)
        ctx = _make_context("BTC-USD", n=200)
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 0.7


class TestConfidenceCap:
    def test_confidence_never_exceeds_cap(self):
        for seed in range(10):
            df = _make_df(n=200, seed=seed)
            ctx = _make_context("ETH-USD", n=200)
            result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
            assert result["confidence"] <= 0.7


class TestSubSignals:
    def test_sub_signal_names(self):
        df = _make_df(n=200)
        ctx = _make_context("ETH-USD", n=200)
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        if result["sub_signals"]:
            expected = {"ratio_roc_zscore", "ratio_vs_sma", "roc_acceleration"}
            assert set(result["sub_signals"].keys()) == expected

    def test_sub_signals_are_valid_votes(self):
        df = _make_df(n=200)
        ctx = _make_context("ETH-USD", n=200)
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        for name, vote in result["sub_signals"].items():
            assert vote in ("BUY", "SELL", "HOLD"), f"{name} has invalid vote {vote}"


class TestIndicatorValues:
    def test_indicators_are_finite_or_nan(self):
        df = _make_df(n=200)
        ctx = _make_context("ETH-USD", n=200)
        result = compute_eth_btc_ratio_roc_zscore_signal(df, ctx)
        for k, v in result["indicators"].items():
            assert isinstance(v, float), f"{k} is not float: {type(v)}"


class TestEdgeCases:
    def test_flat_ratio_returns_hold(self):
        n = 200
        flat = pd.DataFrame({
            "open": [3000.0] * n,
            "high": [3001.0] * n,
            "low": [2999.0] * n,
            "close": [3000.0] * n,
            "volume": [5000.0] * n,
        })
        btc_flat = pd.DataFrame({
            "open": [60000.0] * n,
            "high": [60001.0] * n,
            "low": [59999.0] * n,
            "close": [60000.0] * n,
            "volume": [5000.0] * n,
        })
        ctx = {"ticker": "ETH-USD", "all_prices": {"BTC-USD": btc_flat}}
        result = compute_eth_btc_ratio_roc_zscore_signal(flat, ctx)
        assert result["action"] == "HOLD"
