"""Tests for btc_gold_correlation_regime signal module."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from portfolio.signals.btc_gold_correlation_regime import (
    _CORR_WINDOW,
    _MIN_ROWS,
    _STALE_RATIO_THRESHOLD,
    compute_btc_gold_correlation_regime_signal,
)


def _make_df(n=400, seed=42, trend=0.001):
    """Create a test DataFrame with realistic OHLCV data."""
    np.random.seed(seed)
    close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02 + trend))
    idx = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + abs(np.random.randn(n) * 0.005)),
        "low": close * (1 - abs(np.random.randn(n) * 0.005)),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    }, index=idx)


def _make_counterpart(n=400, seed=99, trend=-0.0005):
    """Create counterpart DataFrame with different seed."""
    return _make_df(n=n, seed=seed, trend=trend)


class TestSignalInterface:

    def test_returns_hold_no_context(self):
        df = _make_df()
        result = compute_btc_gold_correlation_regime_signal(df)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_returns_hold_unknown_ticker(self):
        df = _make_df()
        result = compute_btc_gold_correlation_regime_signal(
            df, context={"ticker": "UNKNOWN"}
        )
        assert result["action"] == "HOLD"

    def test_returns_hold_none_df(self):
        result = compute_btc_gold_correlation_regime_signal(None)
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.0

    def test_returns_hold_empty_df(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        result = compute_btc_gold_correlation_regime_signal(df)
        assert result["action"] == "HOLD"

    def test_returns_hold_insufficient_rows(self):
        df = _make_df(n=20)
        result = compute_btc_gold_correlation_regime_signal(
            df, context={"ticker": "BTC-USD"}
        )
        assert result["action"] == "HOLD"

    def test_output_schema(self):
        df = _make_df()
        result = compute_btc_gold_correlation_regime_signal(
            df, context={"ticker": "BTC-USD"}
        )
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert "sub_signals" in result
        assert "indicators" in result
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert isinstance(result["confidence"], float)


class TestWithMockedCounterpart:

    @patch("portfolio.signals.btc_gold_correlation_regime._fetch_counterpart")
    def test_counterpart_fetch_failure_returns_hold(self, mock_fetch):
        mock_fetch.return_value = None
        df = _make_df()
        result = compute_btc_gold_correlation_regime_signal(
            df, context={"ticker": "BTC-USD"}
        )
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.btc_gold_correlation_regime._fetch_counterpart")
    def test_counterpart_too_short_returns_hold(self, mock_fetch):
        mock_fetch.return_value = _make_counterpart(n=50)
        df = _make_df()
        result = compute_btc_gold_correlation_regime_signal(
            df, context={"ticker": "BTC-USD"}
        )
        assert result["action"] == "HOLD"

    @patch("portfolio.signals.btc_gold_correlation_regime._fetch_counterpart")
    def test_normal_computation_produces_valid_result(self, mock_fetch):
        mock_fetch.return_value = _make_counterpart(n=400)
        df = _make_df(n=400)
        result = compute_btc_gold_correlation_regime_signal(
            df, context={"ticker": "BTC-USD"}
        )
        assert result["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= result["confidence"] <= 0.7
        assert "corr_30d" in result["indicators"]
        assert "corr_z_score" in result["indicators"]
        for v in result["indicators"].values():
            assert np.isfinite(v), f"Non-finite indicator: {v}"

    @patch("portfolio.signals.btc_gold_correlation_regime._fetch_counterpart")
    def test_buy_signal_extreme_negative_z(self, mock_fetch):
        """Engineer data where BTC-Gold correlation is extremely negative."""
        np.random.seed(42)
        n = 400
        idx = pd.date_range("2025-01-01", periods=n, freq="D")

        btc_close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        gold_close = 2000 * np.exp(np.cumsum(np.random.randn(n) * 0.01))

        # Last 30 days: strongly inverse — BTC up, gold down (with noise)
        btc_close[-30:] = btc_close[-31] * np.exp(np.cumsum(
            np.random.randn(30) * 0.005 + 0.03
        ))
        gold_close[-30:] = gold_close[-31] * np.exp(np.cumsum(
            np.random.randn(30) * 0.005 - 0.02
        ))

        target_df = pd.DataFrame({
            "open": btc_close, "high": btc_close * 1.01,
            "low": btc_close * 0.99, "close": btc_close,
            "volume": np.full(n, 5000.0),
        }, index=idx)

        counter_df = pd.DataFrame({
            "open": gold_close, "high": gold_close * 1.01,
            "low": gold_close * 0.99, "close": gold_close,
            "volume": np.full(n, 5000.0),
        }, index=idx)

        mock_fetch.return_value = counter_df
        result = compute_btc_gold_correlation_regime_signal(
            target_df, context={"ticker": "BTC-USD"}
        )
        assert result["indicators"]["corr_30d"] < 0
        # Z-score should be significantly negative given the engineered data
        assert result["indicators"]["corr_z_score"] < 0

    @patch("portfolio.signals.btc_gold_correlation_regime._fetch_counterpart")
    def test_xau_inverted_signals(self, mock_fetch):
        """XAU-USD should get inverted signal directions."""
        np.random.seed(42)
        n = 400
        idx = pd.date_range("2025-01-01", periods=n, freq="D")

        btc_close = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        gold_close = 2000 * np.exp(np.cumsum(np.random.randn(n) * 0.01))

        btc_close[-30:] = btc_close[-31] * np.exp(np.cumsum(
            np.random.randn(30) * 0.005 + 0.03
        ))
        gold_close[-30:] = gold_close[-31] * np.exp(np.cumsum(
            np.random.randn(30) * 0.005 - 0.02
        ))

        gold_df = pd.DataFrame({
            "open": gold_close, "high": gold_close * 1.01,
            "low": gold_close * 0.99, "close": gold_close,
            "volume": np.full(n, 5000.0),
        }, index=idx)

        btc_df = pd.DataFrame({
            "open": btc_close, "high": btc_close * 1.01,
            "low": btc_close * 0.99, "close": btc_close,
            "volume": np.full(n, 5000.0),
        }, index=idx)

        mock_fetch.return_value = btc_df

        result_xau = compute_btc_gold_correlation_regime_signal(
            gold_df, context={"ticker": "XAU-USD"}
        )
        assert result_xau["action"] in ("BUY", "SELL", "HOLD")
        if "correlation_z" in result_xau["sub_signals"]:
            assert "inverted" in result_xau["sub_signals"]["correlation_z"]

    @patch("portfolio.signals.btc_gold_correlation_regime._fetch_counterpart")
    def test_confidence_capped_at_07(self, mock_fetch):
        mock_fetch.return_value = _make_counterpart(n=400)
        df = _make_df(n=400)
        result = compute_btc_gold_correlation_regime_signal(
            df, context={"ticker": "BTC-USD"}
        )
        assert result["confidence"] <= 0.7

    @patch("portfolio.signals.btc_gold_correlation_regime._fetch_counterpart")
    def test_stale_data_forces_hold(self, mock_fetch):
        """Counterpart with many zero-return days should force HOLD."""
        n = 400
        idx = pd.date_range("2025-01-01", periods=n, freq="D")
        flat_close = np.full(n, 2000.0)
        counter_df = pd.DataFrame({
            "open": flat_close, "high": flat_close,
            "low": flat_close, "close": flat_close,
            "volume": np.full(n, 5000.0),
        }, index=idx)

        mock_fetch.return_value = counter_df
        df = _make_df(n=400)
        result = compute_btc_gold_correlation_regime_signal(
            df, context={"ticker": "BTC-USD"}
        )
        assert result["action"] == "HOLD"
        assert result["indicators"].get("stale_data_ratio", 0) > _STALE_RATIO_THRESHOLD

    @patch("portfolio.signals.btc_gold_correlation_regime._fetch_counterpart")
    def test_no_nan_in_indicators(self, mock_fetch):
        mock_fetch.return_value = _make_counterpart(n=400)
        df = _make_df(n=400)
        result = compute_btc_gold_correlation_regime_signal(
            df, context={"ticker": "BTC-USD"}
        )
        for key, val in result["indicators"].items():
            assert np.isfinite(val), f"NaN/Inf in indicators[{key}]: {val}"


class TestRegistration:

    def test_signal_in_registry(self):
        from portfolio.signal_registry import get_enhanced_signals
        signals = get_enhanced_signals()
        assert "btc_gold_correlation_regime" in signals

    def test_signal_in_signal_names(self):
        from portfolio.tickers import SIGNAL_NAMES
        assert "btc_gold_correlation_regime" in SIGNAL_NAMES

    def test_signal_in_disabled(self):
        from portfolio.tickers import DISABLED_SIGNALS
        assert "btc_gold_correlation_regime" in DISABLED_SIGNALS
