"""Tests for crypto_derivatives_composite signal module."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from portfolio.signals.crypto_derivatives_composite import (
    compute_crypto_derivatives_composite_signal,
    _zscore,
    _oi_momentum_vote,
    _funding_zscore_vote,
    _ls_contrarian_vote,
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


def _make_oi_history(n=30, base_oi=50000.0, trend=0.0):
    return [{"oi": base_oi + i * trend + np.random.randn() * 100, "timestamp": i * 300000} for i in range(n)]


def _make_funding_history(n=50, base_rate=0.0001):
    return [{"fundingRate": base_rate + np.random.randn() * 0.00005, "fundingTime": i * 28800000, "symbol": "BTCUSDT"} for i in range(n)]


def _make_ls_history(long_pct=0.5):
    return [{"longShortRatio": long_pct / (1 - long_pct), "longAccount": long_pct, "shortAccount": 1 - long_pct, "timestamp": i * 300000} for i in range(10)]


class TestZScore:
    def test_insufficient_data(self):
        assert _zscore([1.0, 2.0]) == 0.0

    def test_constant_values(self):
        assert _zscore([5.0] * 20) == 0.0

    def test_positive_zscore(self):
        values = [1.0] * 19 + [3.0]
        z = _zscore(values)
        assert z > 0

    def test_negative_zscore(self):
        values = [5.0] * 19 + [1.0]
        z = _zscore(values)
        assert z < 0


class TestOIMomentumVote:
    def test_insufficient_history(self):
        vote, ind = _oi_momentum_vote(None, _make_df())
        assert vote == "HOLD"

    def test_flat_oi(self):
        oi = _make_oi_history(n=30, base_oi=50000, trend=0)
        vote, ind = _oi_momentum_vote(oi, _make_df())
        assert vote in ("BUY", "SELL", "HOLD")

    def test_returns_indicators(self):
        oi = _make_oi_history(n=30, base_oi=50000, trend=100)
        _, ind = _oi_momentum_vote(oi, _make_df())
        assert "oi_z" in ind


class TestFundingZScoreVote:
    def test_insufficient_history(self):
        vote, ind = _funding_zscore_vote(None)
        assert vote == "HOLD"

    def test_normal_funding(self):
        funding = _make_funding_history(n=50, base_rate=0.0001)
        vote, ind = _funding_zscore_vote(funding)
        assert vote in ("BUY", "SELL", "HOLD")
        assert "funding_z" in ind
        assert "funding_apr" in ind

    def test_extreme_positive_funding(self):
        funding = _make_funding_history(n=50, base_rate=0.0001)
        funding[-1]["fundingRate"] = 0.005
        vote, ind = _funding_zscore_vote(funding)
        assert vote == "SELL"

    def test_extreme_negative_funding(self):
        funding = _make_funding_history(n=50, base_rate=0.0001)
        funding[-1]["fundingRate"] = -0.005
        vote, ind = _funding_zscore_vote(funding)
        assert vote == "BUY"


class TestLSContrarianVote:
    def test_insufficient_data(self):
        vote, ind = _ls_contrarian_vote(None)
        assert vote == "HOLD"

    def test_balanced(self):
        ls = _make_ls_history(long_pct=0.50)
        vote, ind = _ls_contrarian_vote(ls)
        assert vote == "HOLD"

    def test_extreme_long(self):
        ls = _make_ls_history(long_pct=0.70)
        vote, ind = _ls_contrarian_vote(ls)
        assert vote == "SELL"

    def test_extreme_short(self):
        ls = _make_ls_history(long_pct=0.30)
        vote, ind = _ls_contrarian_vote(ls)
        assert vote == "BUY"


class TestSignalInterface:
    @patch("portfolio.signals.crypto_derivatives_composite.get_open_interest_history", return_value=None)
    @patch("portfolio.signals.crypto_derivatives_composite.get_funding_rate_history", return_value=None)
    @patch("portfolio.signals.crypto_derivatives_composite.get_long_short_ratio", return_value=None)
    def test_returns_dict_with_required_keys(self, mock_ls, mock_fr, mock_oi):
        df = _make_df()
        ctx = {"ticker": "BTC-USD"}
        result = compute_crypto_derivatives_composite_signal(df, context=ctx)
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 1.0

    @patch("portfolio.signals.crypto_derivatives_composite.get_open_interest_history", return_value=None)
    @patch("portfolio.signals.crypto_derivatives_composite.get_funding_rate_history", return_value=None)
    @patch("portfolio.signals.crypto_derivatives_composite.get_long_short_ratio", return_value=None)
    def test_has_sub_signals(self, mock_ls, mock_fr, mock_oi):
        df = _make_df()
        ctx = {"ticker": "BTC-USD"}
        result = compute_crypto_derivatives_composite_signal(df, context=ctx)
        assert "sub_signals" in result
        assert isinstance(result["sub_signals"], dict)
        assert "oi_momentum" in result["sub_signals"]
        assert "funding_zscore" in result["sub_signals"]
        assert "ls_contrarian" in result["sub_signals"]

    @patch("portfolio.signals.crypto_derivatives_composite.get_open_interest_history", return_value=None)
    @patch("portfolio.signals.crypto_derivatives_composite.get_funding_rate_history", return_value=None)
    @patch("portfolio.signals.crypto_derivatives_composite.get_long_short_ratio", return_value=None)
    def test_has_indicators(self, mock_ls, mock_fr, mock_oi):
        df = _make_df()
        ctx = {"ticker": "ETH-USD"}
        result = compute_crypto_derivatives_composite_signal(df, context=ctx)
        assert "indicators" in result
        assert isinstance(result["indicators"], dict)

    def test_non_crypto_returns_hold(self):
        df = _make_df()
        ctx = {"ticker": "XAG-USD"}
        result = compute_crypto_derivatives_composite_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_no_context_returns_hold(self):
        df = _make_df()
        result = compute_crypto_derivatives_composite_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_empty_dataframe_returns_hold(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        ctx = {"ticker": "BTC-USD"}
        result = compute_crypto_derivatives_composite_signal(df, context=ctx)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_insufficient_rows_returns_hold(self):
        df = _make_df(n=3)
        ctx = {"ticker": "BTC-USD"}
        result = compute_crypto_derivatives_composite_signal(df, context=ctx)
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.crypto_derivatives_composite.get_open_interest_history")
    @patch("portfolio.signals.crypto_derivatives_composite.get_funding_rate_history")
    @patch("portfolio.signals.crypto_derivatives_composite.get_long_short_ratio")
    def test_confidence_capped(self, mock_ls, mock_fr, mock_oi):
        mock_oi.return_value = _make_oi_history(n=30, base_oi=50000, trend=500)
        extreme_funding = _make_funding_history(n=50, base_rate=0.0001)
        extreme_funding[-1]["fundingRate"] = 0.01
        mock_fr.return_value = extreme_funding
        mock_ls.return_value = _make_ls_history(long_pct=0.75)

        df = _make_df()
        ctx = {"ticker": "BTC-USD"}
        result = compute_crypto_derivatives_composite_signal(df, context=ctx)
        assert result["confidence"] <= 0.7

    @patch("portfolio.signals.crypto_derivatives_composite.get_open_interest_history", return_value=None)
    @patch("portfolio.signals.crypto_derivatives_composite.get_funding_rate_history", return_value=None)
    @patch("portfolio.signals.crypto_derivatives_composite.get_long_short_ratio", return_value=None)
    def test_nan_handling(self, mock_ls, mock_fr, mock_oi):
        df = _make_df()
        df.iloc[50:55, df.columns.get_loc("close")] = np.nan
        ctx = {"ticker": "BTC-USD"}
        result = compute_crypto_derivatives_composite_signal(df, context=ctx)
        assert result["action"] in ("BUY", "SELL", "HOLD")
